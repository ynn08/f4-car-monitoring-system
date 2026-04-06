import cv2
import numpy as np

class LocalMapBuilder:
    def __init__(self, 
                 pixels_per_meter=33,
                 initial_size=4000,
                 blend_decay=0.05,
                 min_weight=0.1,
                 use_distance_weighting=True,
                 scale_factor=1.0):
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
        
        # История поз
        self.poses = []
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

    def add_frame(self, bev_image, pose):
        """
        Добавляет кадр на карту с учетом веса и позиции
        """
        self.poses.append(pose.copy())
        
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

    def get_map(self, crop_to_content=True):
        """Возвращает итоговую карту"""
        if crop_to_content:
            valid_mask = self.weight_map > self.min_weight
            if np.any(valid_mask):
                coords = np.argwhere(valid_mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                margin = 50
                y_min = max(0, y_min - margin)
                x_min = max(0, x_min - margin)
                y_max = min(self.map.shape[0], y_max + margin)
                x_max = min(self.map.shape[1], x_max + margin)
                
                return self.map[y_min:y_max, x_min:x_max].copy()
        
        return self.map.copy()

    def save_map(self, path):
        """Сохраняет карту в файл"""
        map_to_save = self.get_map(crop_to_content=True)
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