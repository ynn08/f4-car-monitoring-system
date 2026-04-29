import cv2
import numpy as np

class LocalMapBuilder:
    def __init__(self, 
                 pixels_per_meter=33,
                 initial_size=4000,
                 blend_decay=0.05,
                 min_weight=0.1,
                 use_distance_weighting=True,
                 scale_factor=0.25):
        """
        pixels_per_meter: масштаб карты (пикселей на метр)
        initial_size: начальный размер холста карты
        blend_decay: коэффициент затухания старых кадров (0.0-1.0)
        min_weight: минимальный вес для обновления пикселя
        use_distance_weighting: использовать взвешивание по расстоянию от машины
        scale_factor: масштаб для сжатия карты без потери точности одометрии
        """
        self.ppm = pixels_per_meter
        self.initial_size = initial_size
        self.blend_decay = blend_decay
        self.min_weight = min_weight
        self.use_distance_weighting = use_distance_weighting
        self.scale_factor = scale_factor
        
        # Инициализация карты
        self.map = np.zeros((initial_size, initial_size, 3), dtype=np.uint8)
        self.weight_map = np.zeros((initial_size, initial_size), dtype=np.float32)
        
        # Центр карты (начальная позиция машины)
        self.center_x = initial_size // 2
        self.center_y = initial_size // 2
        
        # История поз (x, y, theta)
        self.poses = []
        self.frame_indices = []
        self.refinement_count = 0
        self.current_pose = np.array([0.0, 0.0, 0.0])
        
        # Границы карты для расширения
        self.map_bounds = {
            'min_x': initial_size // 2,
            'max_x': initial_size // 2,
            'min_y': initial_size // 2,
            'max_y': initial_size // 2
        }
        
        # Кэш весовых масок
        self._distance_weight_cache = {}

    def _create_distance_weight_mask(self, shape, car_position_in_bev):
        """
        Создает маску весов: ближе к машине = больше вес, дальше = меньше вес.
        Это решает проблему с искажениями IPM на дальних дистанциях.
        """
        h, w = shape[:2]
        cache_key = (h, w, int(car_position_in_bev[0]), int(car_position_in_bev[1]))
        
        if cache_key in self._distance_weight_cache:
            return self._distance_weight_cache[cache_key]
        
        y, x = np.ogrid[:h, :w]
        center_x, center_y = car_position_in_bev
        
        # Евклидово расстояние от позиции машины
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Гауссово затухание весов
        max_dist = max(h, w) * 0.8
        weights = np.exp(-(distances**2) / (2 * (max_dist * 0.3)**2))
        
        # Сильно уменьшаем вес для самых дальних пикселей (там IPM неточен)
        far_mask = distances > max_dist * 0.7
        weights[far_mask] *= 0.3
        
        self._distance_weight_cache[cache_key] = weights
        return weights

    def _expand_map_if_needed(self, new_bounds):
        """Расширяет карту, если новые данные выходят за границы"""
        margin = 500
        
        # Boundaries of the ALLOCATED map memory relative to coordinate 0
        new_min_x = min(0, new_bounds['min_x']) - margin if new_bounds['min_x'] < 0 else 0
        new_max_x = max(self.map.shape[1], new_bounds['max_x']) + margin if new_bounds['max_x'] > self.map.shape[1] else self.map.shape[1]
        new_min_y = min(0, new_bounds['min_y']) - margin if new_bounds['min_y'] < 0 else 0
        new_max_y = max(self.map.shape[0], new_bounds['max_y']) + margin if new_bounds['max_y'] > self.map.shape[0] else self.map.shape[0]
        
        current_h, current_w = self.map.shape[:2]
        new_h = new_max_y - new_min_y
        new_w = new_max_x - new_min_x
        
        if new_h > current_h or new_w > current_w:
            print(f"🔄 Expanding map: {current_w}x{current_h} -> {new_w}x{new_h}")
            
            new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            new_weight_map = np.zeros((new_h, new_w), dtype=np.float32)
            
            # Смещение для копирования старых данных (0 в старой карте = offset_x в новой)
            offset_x = 0 - new_min_x
            offset_y = 0 - new_min_y
            
            new_map[offset_y:offset_y+current_h, offset_x:offset_x+current_w] = self.map
            new_weight_map[offset_y:offset_y+current_h, offset_x:offset_x+current_w] = self.weight_map
            
            self.map = new_map
            self.weight_map = new_weight_map
            
            # Смещаем глобальный центр (позу)
            self.center_x += offset_x
            self.center_y += offset_y

    def _pose_to_map_coords(self, pose):
        """Конвертирует позу из пикселей BEV в координаты карты"""
        map_x = self.center_x + pose[0] * self.scale_factor
        map_y = self.center_y + pose[1] * self.scale_factor
        theta = pose[2]
        return map_x, map_y, theta

    def refine_pose_against_map(self, bev_image, predicted_pose, frame_idx=None, search_range=10, angle_steps=0):
        """
        Уточняет позу, сопоставляя текущий BEV-кадр с уже построенной картой.
        bev_image: исходный кадр (полного размера)
        predicted_pose: [x, y, theta] от VO
        search_range: радиус поиска в пикселях карты
        """
        if len(self.poses) < 10: # Не уточняем первые кадры
            return predicted_pose, 0.0
            
        # 1. Подготавливаем текущий кадр (масштабируем и поворачиваем)
        if self.scale_factor != 1.0:
            bev_small = cv2.resize(bev_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
        else:
            bev_small = bev_image.copy()
            
        h_s, w_s = bev_small.shape[:2]
        map_x, map_y, theta = self._pose_to_map_coords(predicted_pose)
        
        # Поворачиваем малый кадр
        car_cy_s = int(h_s * 0.85)
        rot_mat = cv2.getRotationMatrix2D((w_s//2, car_cy_s), np.degrees(theta), 1.0)
        bev_rot = cv2.warpAffine(bev_small, rot_mat, (w_s, h_s), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Оставляем только центральную часть для матчинга (чтобы не ловить края)
        roi_size = int(min(h_s, w_s) * 0.6)
        y1_r, y2_r = car_cy_s - roi_size//2, car_cy_s + roi_size//2
        x1_r, x2_r = w_s//2 - roi_size//2, w_s//2 + roi_size//2
        
        # Проверка границ ROI
        y1_r, y2_r = max(0, y1_r), min(h_s, y2_r)
        x1_r, x2_r = max(0, x1_r), min(w_s, x2_r)
        
        template = cv2.cvtColor(bev_rot[y1_r:y2_r, x1_r:x2_r], cv2.COLOR_BGR2GRAY)
        
        # 2. Вырезаем кусок карты для поиска
        # Позиция машины в BEV относительно оффсета в add_frame:
        # offset_x = int(map_x - w // 2)
        # offset_y = int(map_y - int(h * 0.85))
        
        # Нам нужен кусок карты вокруг map_x, map_y
        pad = search_range + roi_size // 2 + 5
        search_y1 = int(map_y - pad)
        search_y2 = int(map_y + pad)
        search_x1 = int(map_x - pad)
        search_x2 = int(map_x + pad)
        
        if search_y1 < 0 or search_x1 < 0 or search_y2 >= self.map.shape[0] or search_x2 >= self.map.shape[1]:
            return predicted_pose, 0.0
            
        map_patch = cv2.cvtColor(self.map[search_y1:search_y2, search_x1:search_x2], cv2.COLOR_BGR2GRAY)
        
        # 3. Матчинг
        # Проверяем, есть ли на карте текстура (не черная ли она)
        if np.mean(map_patch) < 5:
            return predicted_pose, 0.0
            
        # 3. Матчинг с перебором углов (-2, 0, 2 градуса)
        best_val = 0.0
        best_loc = (0, 0)
        best_sub = (0, 0)
        best_d_theta = 0.0
        
        for d_theta in [-2.0, 0.0, 2.0]:
            if d_theta != 0.0:
                rot_mat_fine = cv2.getRotationMatrix2D((w_s//2, car_cy_s), np.degrees(theta) + d_theta, 1.0)
                bev_rot_fine = cv2.warpAffine(bev_small, rot_mat_fine, (w_s, h_s), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                template_fine = cv2.cvtColor(bev_rot_fine[y1_r:y2_r, x1_r:x2_r], cv2.COLOR_BGR2GRAY)
            else:
                template_fine = template

            res = cv2.matchTemplate(map_patch, template_fine, cv2.TM_CCOEFF_NORMED)
            _, current_max, _, current_loc = cv2.minMaxLoc(res)
            
            if current_max > best_val:
                best_val = current_max
                best_loc = current_loc
                best_d_theta = d_theta
                
                # Subpixel refinement
                mx, my = current_loc[0], current_loc[1]
                sub_x, sub_y = 0.0, 0.0
                if 0 < mx < res.shape[1] - 1 and 0 < my < res.shape[0] - 1:
                    left, center, right = res[my, mx-1], res[my, mx], res[my, mx+1]
                    denom_x = 2.0 * (left - 2.0 * center + right)
                    if denom_x != 0: sub_x = (left - right) / denom_x
                    up, down = res[my-1, mx], res[my+1, mx]
                    denom_y = 2.0 * (up - 2.0 * center + down)
                    if denom_y != 0: sub_y = (up - down) / denom_y
                best_sub = (sub_x, sub_y)

        if best_val > 0.85: # Повышаем порог уверенности
            # ПРОВЕРКА ПО КАРТЕ ВЕСОВ
            # Получаем значение веса в точке матчинга
            try:
                # Масштабируем dx_map, dy_map обратно в координаты патча
                match_y_in_patch = int(best_loc[1])
                match_x_in_patch = int(best_loc[0])
                
                # Извлекаем тот же патч из карты весов
                weight_patch = self.weight_map[int(py_s - pad):int(py_s + pad), int(px_s - pad):int(px_s + pad)]
                # Но вес надо смотреть у шаблона ( ROI )
                roi_weight = weight_patch[match_y_in_patch:match_y_in_patch+roi_size, match_x_in_patch:match_x_in_patch+roi_size]
                avg_weight = np.mean(roi_weight)
                
                # Если вес слишком мал (меньше 0.3), значит этот участок карты еще не надежен
                if avg_weight < 0.3:
                    return predicted_pose, 0.0
            except:
                pass

            # Смещение центра шаблона относительно начала патча
            dx_map = (best_loc[0] + best_sub[0]) - (pad - roi_size//2)
            dy_map = (best_loc[1] + best_sub[1]) - (pad - roi_size//2)
            
            # Корректируем позу
            refined_pose = predicted_pose.copy()
            refined_pose[0] += dx_map / self.scale_factor
            refined_pose[1] += dy_map / self.scale_factor
            refined_pose[2] += np.radians(best_d_theta)
            
            self.refinement_count += 1
            return refined_pose, best_val
            
        return predicted_pose, best_val

    def add_frame(self, bev_image, pose, frame_idx=None):
        """
        Добавляет кадр на карту с учетом веса и позиции
        """
        self.poses.append(pose.copy())
        if frame_idx is not None:
            self.frame_indices.append(frame_idx)
        else:
            self.frame_indices.append(len(self.poses))
        
        # Сжимаем изображение для глобальной карты
        if self.scale_factor != 1.0:
            bev_image = cv2.resize(bev_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
            
        h, w = bev_image.shape[:2]
        map_x, map_y, theta = self._pose_to_map_coords(pose)
        
        # 1. Поворачиваем BEV согласно ориентации машины вокруг задней оси!
        car_cy = int(h * 0.85)
        rotation_matrix = cv2.getRotationMatrix2D((w//2, car_cy), np.degrees(theta), 1.0)
        bev_rotated = cv2.warpAffine(bev_image, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
        
        # 2. Создаем маску весов
        car_pos_in_bev = (w // 2, int(h * 0.85))
        
        if self.use_distance_weighting:
            distance_weights = self._create_distance_weight_mask(bev_image.shape, car_pos_in_bev)
        else:
            distance_weights = np.ones((h, w), dtype=np.float32)
        
        # 3. Маска валидных пикселей (убираем черные края IPM и саму машину)
        validity_mask = (bev_rotated[:,:,0] > 10) | (bev_rotated[:,:,1] > 10) | (bev_rotated[:,:,2] > 10)
        validity_mask = validity_mask.astype(np.float32)
        
        # Полностью вырезаем машину (учитывая масштаб!)
        mask_radius = int(120 * self.scale_factor)
        cv2.circle(validity_mask, (w // 2, int(h * 0.85)), mask_radius, 0, -1)
        
        # 4. Комбинируем веса
        frame_weights = distance_weights * validity_mask
        
        # 5. Вычисляем область на карте для вставки
        # Машина в BEV находится внизу (0.85 высоты), это должно совпадать с map_x, map_y
        offset_x = int(map_x - w // 2)
        offset_y = int(map_y - int(h * 0.85))
        
        # 6. Проверяем и расширяем карту
        new_bounds = {
            'min_x': offset_x,
            'max_x': offset_x + w,
            'min_y': offset_y,
            'max_y': offset_y + h
        }
        self._expand_map_if_needed(new_bounds)
        
        # 7. Корректируем оффсеты после расширения
        offset_x = int(map_x - w // 2)
        offset_y = int(map_y - int(h * 0.85))
        
        # 8. Определяем область пересечения (ИСПРАВЛЕНО)
        map_h, map_w = self.map.shape[:2]
        
        # Координаты в источнике (кадр BEV)
        src_x_start = max(0, -offset_x)
        src_y_start = max(0, -offset_y)
        src_x_end = min(w, map_w - offset_x)
        src_y_end = min(h, map_h - offset_y)
        
        # Координаты в назначении (Карта) - 🔥 ДОБАВЛЕНО ОПРЕДЕЛЕНИЕ
        dst_x_start = max(0, offset_x)
        dst_y_start = max(0, offset_y)
        dst_x_end = min(map_w, offset_x + w)
        dst_y_end = min(map_h, offset_y + h)
        
        if src_x_end <= src_x_start or src_y_end <= src_y_start:
            return  # Кадр вне границ карты
        
        # 9. Извлекаем области
        frame_region = bev_rotated[src_y_start:src_y_end, src_x_start:src_x_end]
        weight_region = frame_weights[src_y_start:src_y_end, src_x_start:src_x_end]
        map_region = self.map[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        weight_map_region = self.weight_map[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        
        # 10. Затухание старых весов (ТОЛЬКО В ЗОНЕ ПЕРЕКРЫТИЯ!)
        weight_map_region *= (1.0 - self.blend_decay)
        
        # 11. Смешивание с весами
        new_weights = weight_map_region + weight_region
        new_weights = np.maximum(new_weights, self.min_weight)
        
        old_weight_ratio = weight_map_region / new_weights
        new_weight_ratio = weight_region / new_weights
        
        # 12. Обновляем карту (Alpha Blending)
        for c in range(3):
            map_region[:,:,c] = (
                map_region[:,:,c].astype(np.float32) * old_weight_ratio +
                frame_region[:,:,c].astype(np.float32) * new_weight_ratio
            ).astype(np.uint8)
        
        self.map[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = map_region
        self.weight_map[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = new_weights
        
        self.current_pose = pose

    def get_map(self, crop_to_content=True, draw_trajectory=False):
        """Возвращает итоговую карту"""
        res_map = self.map.copy()
        
        if draw_trajectory:
            trajectory = self.get_trajectory()
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    cv2.line(res_map, pt1, pt2, (255, 0, 0), 2)

        if crop_to_content:
            valid_mask = self.weight_map > self.min_weight
            if np.any(valid_mask):
                coords = np.argwhere(valid_mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                margin = 50
                y_min = max(0, y_min - margin)
                x_min = max(0, x_min - margin)
                y_max = min(res_map.shape[0], y_max + margin)
                x_max = min(res_map.shape[1], x_max + margin)
                
                return res_map[y_min:y_max, x_min:x_max].copy()
        
        return res_map

    def save_map(self, path, draw_trajectory=True):
        """Сохраняет карту в файл"""
        map_to_save = self.get_map(crop_to_content=True, draw_trajectory=draw_trajectory)
        cv2.imwrite(path, map_to_save)
        print(f"💾 Map saved: {path} ({map_to_save.shape[1]}x{map_to_save.shape[0]})")

    def get_trajectory(self):
        """Возвращает траекторию в координатах карты"""
        trajectory = []
        for pose in self.poses:
            map_x, map_y, _ = self._pose_to_map_coords(pose)
            trajectory.append((map_x, map_y))
        return np.array(trajectory)

    def clear_cache(self):
        """Очищает кэш весовых масок"""
        self._distance_weight_cache.clear()