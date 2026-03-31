import cv2
import numpy as np


class LocalMapBuilder:
    def __init__(self, pixels_per_meter=50, initial_size=4000):
        self.ppm = pixels_per_meter
        self.offset_x = initial_size // 2
        self.offset_y = initial_size // 2
        self.map_canvas = np.zeros((initial_size, initial_size, 3), dtype=np.uint8) + 128
        self.mask_canvas = np.zeros((initial_size, initial_size), dtype=np.uint8)
        self.count_canvas = np.zeros((initial_size, initial_size), dtype=np.float32)  # Для усреднения
        self.sum_canvas = np.zeros((initial_size, initial_size, 3), dtype=np.float32)  # Для усреднения
        self.poses = []
        
    def _expand_canvas(self, min_size_needed):
        """Расширяет холст если нужно"""
        current_size = self.map_canvas.shape[0]
        
        if min_size_needed <= current_size:
            return
        
        new_size = max(current_size * 2, min_size_needed)
        print(f"Expanding canvas from {current_size} to {new_size} pixels")
        
        new_map = np.zeros((new_size, new_size, 3), dtype=np.uint8) + 128
        new_mask = np.zeros((new_size, new_size), dtype=np.uint8)
        new_sum = np.zeros((new_size, new_size, 3), dtype=np.float32)
        new_count = np.zeros((new_size, new_size), dtype=np.float32)
        
        start_x = (new_size - current_size) // 2
        start_y = (new_size - current_size) // 2
        
        new_map[start_y:start_y+current_size, start_x:start_x+current_size] = self.map_canvas
        new_mask[start_y:start_y+current_size, start_x:start_x+current_size] = self.mask_canvas
        new_sum[start_y:start_y+current_size, start_x:start_x+current_size] = self.sum_canvas
        new_count[start_y:start_y+current_size, start_x:start_x+current_size] = self.count_canvas
        
        self.offset_x += start_x
        self.offset_y += start_y
        
        self.map_canvas = new_map
        self.mask_canvas = new_mask
        self.sum_canvas = new_sum
        self.count_canvas = new_count
    
    def _create_validity_mask(self, image):
        """
        Создаёт маску валидных пикселей (исключает черные области)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Черные области = 0, валидные = > 10 (с запасом)
        valid_mask = gray > 10
        
        # Морфологическая операция для удаления шума
        kernel = np.ones((3, 3), np.uint8)
        valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return valid_mask.astype(np.uint8) * 255
    
    def add_frame(self, bev_image, vehicle_pose, use_averaging=True):
        """
        Добавляет кадр на карту
        
        Args:
            bev_image: IPM изображение
            vehicle_pose: [x, y, theta] в метрах и радианах
            use_averaging: Если True, использует усреднение вместо blending
        """
        x, y, theta = vehicle_pose
        h, w = bev_image.shape[:2]
        
        # Проверяем и расширяем холст при необходимости
        margin = max(w, h)
        center_x_est = int(x * self.ppm + self.offset_x)
        center_y_est = int(-y * self.ppm + self.offset_y)
        
        min_x = center_x_est - w//2 - margin
        max_x = center_x_est + w//2 + margin
        min_y = center_y_est - h//2 - margin
        max_y = center_y_est + h//2 + margin
        
        required_size = max(max_x, max_y, self.map_canvas.shape[0])
        self._expand_canvas(required_size)
        
        # Пересчитываем центр после расширения
        center_x = int(x * self.ppm + self.offset_x)
        center_y = int(-y * self.ppm + self.offset_y)
        
        # Поворот BEV кадра согласно ориентации автомобиля
        rotation_matrix = cv2.getRotationMatrix2D((w//2, h//2), np.degrees(theta), 1.0)
        rotated_bev = cv2.warpAffine(bev_image, rotation_matrix, (w, h))
        
        # Создаём маску валидных пикселей (исключаем черные зоны)
        validity_mask = self._create_validity_mask(rotated_bev)
        rotated_mask = cv2.warpAffine(
            np.ones((h, w), dtype=np.uint8) * 255, 
            rotation_matrix, (w, h)
        )
        
        # Объединяем маски
        combined_mask = cv2.bitwise_and(validity_mask, rotated_mask)
        
        # Координаты ROI
        x1, x2 = center_x - w//2, center_x + w//2
        y1, y2 = center_y - h//2, center_y + h//2
        
        # Проверка границ
        if (x1 < 0 or x2 > self.map_canvas.shape[1] or
            y1 < 0 or y2 > self.map_canvas.shape[0]):
            print(f"Warning: Frame exceeds boundaries. x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            return
        
        # Нормализуем маски
        blend_mask = combined_mask > 0
        
        if not np.any(blend_mask):
            return  # Нет валидных пикселей
        
        if use_averaging:
            # === УСРЕДНЕНИЕ (лучше для карты) ===
            roi_sum = self.sum_canvas[y1:y2, x1:x2]
            roi_count = self.count_canvas[y1:y2, x1:x2]
            
            # Добавляем новые данные по каналам
            for c in range(3):
                roi_sum[:, :, c][blend_mask] += rotated_bev[:, :, c][blend_mask].astype(np.float32)
            
            roi_count[blend_mask] += 1
            
            # Обновляем карту усреднением
            valid_count = roi_count > 0
            for c in range(3):
                self.map_canvas[y1:y2, x1:x2][:, :, c][valid_count] = (
                    roi_sum[:, :, c][valid_count] / roi_count[valid_count]
                ).astype(np.uint8)
            
            self.sum_canvas[y1:y2, x1:x2] = roi_sum
            self.count_canvas[y1:y2, x1:x2] = roi_count
        else:
            # === BLENDING (старый метод) ===
            roi_map = self.map_canvas[y1:y2, x1:x2]
            alpha = 0.5  # Более консервативное смешивание
            
            blended = cv2.addWeighted(roi_map, 1-alpha, rotated_bev, alpha, 0)
            
            # Копируем только валидные пиксели
            for c in range(3):
                roi_map[:, :, c][blend_mask] = blended[:, :, c][blend_mask]
            
            self.map_canvas[y1:y2, x1:x2] = roi_map
        
        # Обновляем маску заполненности
        self.mask_canvas[y1:y2, x1:x2] = cv2.bitwise_or(
            self.mask_canvas[y1:y2, x1:x2], 
            combined_mask
        )
        
        self.poses.append(vehicle_pose)
        
    def get_map(self, crop_to_content=True):
        if crop_to_content:
            coords = cv2.findNonZero(self.mask_canvas)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                return self.map_canvas[y:y+h, x:x+w]
        return self.map_canvas

    def save_map(self, path):
        cv2.imwrite(path, self.get_map())
        
    def draw_trajectory(self, color=(0, 0, 255), thickness=2):
        map_img = self.get_map().copy()
        if len(self.poses) < 2:
            return map_img
            
        points = []
        for x, y, _ in self.poses:
            px = int(x * self.ppm + self.offset_x)
            py = int(-y * self.ppm + self.offset_y)
            coords = cv2.findNonZero(self.mask_canvas)
            if coords is not None:
                ox, oy, _, _ = cv2.boundingRect(coords)
                px -= ox
                py -= oy
            points.append([px, py])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(map_img, [points], False, color, thickness)
        return map_img
    
    def reset(self):
        """Сбросить карту"""
        self.map_canvas = np.zeros_like(self.map_canvas) + 128
        self.mask_canvas = np.zeros_like(self.mask_canvas)
        self.sum_canvas = np.zeros_like(self.sum_canvas)
        self.count_canvas = np.zeros_like(self.count_canvas)
        self.poses = []