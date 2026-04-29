import cv2
import numpy as np

class VisualOdometryIPM:
    def __init__(self, 
                 use_telemetry=False,
                 debug_mode=False,
                 config_mode="custom",
                 bev_width=600,
                 bev_height=400):
        """
        config_mode: 
            - "balanced": одинаковые высоты для всех зон
            - "sides_short": боковые вытянутые, центральная длинная
            - "sides_short_v2": боковые очень вытянутые, центральная очень длинная
            - "sides_short_v3": разные варианты комбинаций
            - "synthetic": для разрешения 800x600
        """
        
        self.prev_gray = None
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.center_x = bev_width // 2
        
        # Глобальная поза (X, Y, Theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.use_telemetry = use_telemetry
        self.debug_mode = debug_mode
        self.frame_count = 0
        self.config_mode = config_mode
        
        self.stats = {
            'total_frames': 0,
            'successful_matches': 0,
        }
        
        # История dtheta для медианной фильтрации
        self.dtheta_history = []
        self.max_history = 1  # Убрана фильтрация для максимальной резкости поворотов
        
        # Состояние для модели постоянной скорости
        self.last_pure_dx = 0.0
        self.last_pure_dy = 0.0
        self.last_dtheta = 0.0
        
        # Конфигурации масок для разных режимов
        self._setup_masks(config_mode)

        self.confidence_threshold = 0.5

    def _setup_masks(self, mode):
        """Настройка масок в зависимости от режима"""
        if mode == "balanced":
            # БАЗОВАЯ: все зоны одинаковой высоты
            # Ближний
            self.t1_side_regions = [(100, 350), (250, 500)]  # Левая, правая
            self.t1_side_y1, self.t1_side_y2 = 150, 270  # Высота 120
            self.t1_center_regions = [(200, 400)]  # Центральная
            self.t1_center_y1, self.t1_center_y2 = 150, 270  # Высота 120
            # Дальний
            self.t2_side_regions = [(100, 350), (250, 500)]
            self.t2_side_y1, self.t2_side_y2 = 10, 110  # Высота 100
            self.t2_center_regions = [(200, 400)]
            self.t2_center_y1, self.t2_center_y2 = 10, 110  # Высота 100
            
        elif mode == "custom":
            # Настройки шаблона от пользователя (эксперимент с включением центральной зоны)
            # Ближний
            self.t1_side_y1, self.t1_side_y2 = 150, 270 
            self.t1_side_regions = [(100, 350), (250, 500)] # Большая левая зона, Большая правая зона
            self.t1_center_y1, self.t1_center_y2 = 0, 0
            self.t1_center_regions = [(0, 0)]  # Отключаем центральную зону полностью
            
            # Дальний
            self.t2_side_y1, self.t2_side_y2 = 30, 130  # Приближены к t1 для большей чувствительности dtheta
            self.t2_side_regions = [(100, 350), (250, 500)]
            self.t2_center_y1, self.t2_center_y2 = 0, 0
            self.t2_center_regions = [(0, 0)]  # Отключаем центральную зону полностью

            
        elif mode == "sides_short_v2":
            # ЭКСПЕРИМЕНТ 2: боковые очень вытянутые (80), центральная очень длинная (200)
            # Ближний
            self.t1_side_regions = [(100, 350), (250, 500)]
            self.t1_side_y1, self.t1_side_y2 = 170, 250  # Высота 80 (очень вытянутые)
            self.t1_center_regions = [(200, 400)]
            self.t1_center_y1, self.t1_center_y2 = 110, 310  # Высота 200 (очень длинные)
            # Дальний
            self.t2_side_regions = [(100, 350), (250, 500)]
            self.t2_side_y1, self.t2_side_y2 = 25, 95  # Высота 70 (очень вытянутые)
            self.t2_center_regions = [(200, 400)]
            self.t2_center_y1, self.t2_center_y2 = 0, 150  # Высота 150 (длинные)
            
        elif mode == "sides_short_v3":
            # ЭКСПЕРИМЕНТ 3: боковые среднее, центральная очень длинная (220)
            # Ближний
            self.t1_side_regions = [(100, 350), (250, 500)]
            self.t1_side_y1, self.t1_side_y2 = 155, 265  # Высота 110 (среднее)
            self.t1_center_regions = [(200, 400)]
            self.t1_center_y1, self.t1_center_y2 = 100, 320  # Высота 220 (очень длинные)
            # Дальний
            self.t2_side_regions = [(100, 350), (250, 500)]
            self.t2_side_y1, self.t2_side_y2 = 15, 105  # Высота 90 (среднее)
            self.t2_center_regions = [(200, 400)]
            self.t2_center_y1, self.t2_center_y2 = 0, 160  # Высота 160 (длинные)
            
        elif mode == "synthetic":
            # ЭКСПЕРИМЕНТ 4: для разрешения 800x600
            # Ближний
            self.t1_side_regions = [(150, 450), (350, 650)]
            self.t1_side_y1, self.t1_side_y2 = 250, 400  # Высота 150
            self.t1_center_regions = [(300, 500)]
            self.t1_center_y1, self.t1_center_y2 = 200, 450  # Высота 250
            # Дальний
            self.t2_side_regions = [(150, 450), (350, 650)]
            self.t2_side_y1, self.t2_side_y2 = 50, 180  # Высота 130
            self.t2_center_regions = [(300, 500)]
            self.t2_center_y1, self.t2_center_y2 = 20, 220  # Высота 200
            
        else:
            raise ValueError(f"Unknown config_mode: {mode}")

    def find_best_match(self, current_gray, regions, y1, y2, threshold=None):
        if threshold is None: threshold = self.confidence_threshold
        best_val = -1
        best_shift = None
        best_x1, best_x2 = 0, 0
        best_kp = None
        
        if not regions or y2 <= y1:
            return False, 0, 0, 0, 0, None
            
        for (x1, x2) in regions:
            template = self.prev_gray[y1:y2, x1:x2]
            if template.shape[0] == 0 or template.shape[1] == 0:
                continue
                
            search_x1 = max(0, x1 - 50)
            search_x2 = min(current_gray.shape[1], x2 + 50)
            search_y1 = max(0, y1 - 30)
            search_y2 = min(current_gray.shape[0], y2 + 70) 
            
            if search_y2 <= search_y1 or search_x2 <= search_x1: continue
            search_area = current_gray[search_y1:search_y2, search_x1:search_x2]
            
            if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
                continue
                
            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # Проверяем порог сразу
            if max_val > threshold and max_val > best_val:
                best_val = max_val
                match_x = search_x1 + max_loc[0]
                match_y = search_y1 + max_loc[1]
                best_shift = (match_x - x1, match_y - y1)
                best_x1, best_x2 = x1, x2
                best_kp = current_gray[y1:y2, x1:x2]
                
        if best_val >= threshold:
            return True, best_shift[0], best_shift[1], best_x1, best_x2, best_kp
        return False, 0, 0, 0, 0, None

    def update(self, bev_image, telemetry=None):
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        current_gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        current_gray = clahe.apply(current_gray)

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return np.array([self.x, self.y, self.theta]), None

        # Поиск матчей для боковых и центральной зон отдельно (с разными высотами)
        # Для центра ставим порог выше (0.75), чтобы он не брал "лысый" асфальт
        succ_side1, dx_side1, dy_side1, b1_x1_s, b1_x2_s, kp1_side = self.find_best_match(
            current_gray, self.t1_side_regions, self.t1_side_y1, self.t1_side_y2, threshold=0.5)
        succ_center1, dx_center1, dy_center1, b1_x1_c, b1_x2_c, kp1_center = self.find_best_match(
            current_gray, self.t1_center_regions, self.t1_center_y1, self.t1_center_y2, threshold=0.6)  # Снижен порог для лучшего захвата
        
        succ_side2, dx_side2, dy_side2, b2_x1_s, b2_x2_s, kp2_side = self.find_best_match(
            current_gray, self.t2_side_regions, self.t2_side_y1, self.t2_side_y2, threshold=0.5)
        succ_center2, dx_center2, dy_center2, b2_x1_c, b2_x2_c, kp2_center = self.find_best_match(
            current_gray, self.t2_center_regions, self.t2_center_y1, self.t2_center_y2, threshold=0.6)  # Снижен порог

        # Комбинируем результаты для t1: если есть оба - усредняем, иначе берем имеющийся
        succ1 = succ_side1 or succ_center1
        if succ_side1 and succ_center1:
            # Проверяем согласованность: если сдвиги сильно отличаются, игнорируем центральный
            if abs(dx_side1 - dx_center1) > 15 or abs(dy_side1 - dy_center1) > 15:
                # Используем только боковой
                dx1, dy1 = dx_side1, dy_side1
                b1_x1, b1_x2 = b1_x1_s, b1_x2_s
                kp1_img = kp1_side
                cY1 = (self.t1_side_y1 + self.t1_side_y2) / 2.0
            else:
                # Обе успешны и согласованы - усредняем
                dx1 = (dx_side1 + dx_center1) / 2.0
                dy1 = (dy_side1 + dy_center1) / 2.0
                b1_x1, b1_x2 = (b1_x1_s + b1_x1_c) / 2.0, (b1_x2_s + b1_x2_c) / 2.0
                kp1_img = kp1_side  # для визуализации
                cY1 = (self.t1_side_y1 + self.t1_side_y2) / 2.0  # используем боковую для расчета вращения
        elif succ_side1:
            dx1, dy1 = dx_side1, dy_side1
            b1_x1, b1_x2 = b1_x1_s, b1_x2_s
            kp1_img = kp1_side
            cY1 = (self.t1_side_y1 + self.t1_side_y2) / 2.0
        elif succ_center1:
            dx1, dy1 = dx_center1, dy_center1
            b1_x1, b1_x2 = b1_x1_c, b1_x2_c
            kp1_img = kp1_center
            cY1 = (self.t1_center_y1 + self.t1_center_y2) / 2.0
        else:
            dx1, dy1, b1_x1, b1_x2, kp1_img, cY1 = 0, 0, 0, 0, None, 0
        
        # Аналогично для t2
        succ2 = succ_side2 or succ_center2
        if succ_side2 and succ_center2:
            # Проверяем согласованность
            if abs(dx_side2 - dx_center2) > 15 or abs(dy_side2 - dy_center2) > 15:
                # Используем только боковой
                dx2, dy2 = dx_side2, dy_side2
                b2_x1, b2_x2 = b2_x1_s, b2_x2_s
                kp2_img = kp2_side
                cY2 = (self.t2_side_y1 + self.t2_side_y2) / 2.0
            else:
                dx2 = (dx_side2 + dx_center2) / 2.0
                dy2 = (dy_side2 + dy_center2) / 2.0
                b2_x1, b2_x2 = (b2_x1_s + b2_x2_s) / 2.0, (b2_x2_s + b2_x2_c) / 2.0
                kp2_img = kp2_side
                cY2 = (self.t2_side_y1 + self.t2_side_y2) / 2.0
        elif succ_side2:
            dx2, dy2 = dx_side2, dy_side2
            b2_x1, b2_x2 = b2_x1_s, b2_x2_s
            kp2_img = kp2_side
            cY2 = (self.t2_side_y1 + self.t2_side_y2) / 2.0
        elif succ_center2:
            dx2, dy2 = dx_center2, dy_center2
            b2_x1, b2_x2 = b2_x1_c, b2_x2_c
            kp2_img = kp2_center
            cY2 = (self.t2_center_y1 + self.t2_center_y2) / 2.0
        else:
            dx2, dy2, b2_x1, b2_x2, kp2_img, cY2 = 0, 0, 0, 0, None, 0

        if succ1: self.stats['successful_matches'] += 1
        if succ2: self.stats['successful_matches'] += 1

        if telemetry and self.use_telemetry:
            ppm = getattr(self, 'ppm', 62.6)  # 92px_car_width / 1.47m = 62.6 px/m
            dt = telemetry.get('dt', 1/30)
            pred_dy = telemetry.get('speed', 0) * dt * ppm
            pred_dx = 0.0
            pred_dtheta = telemetry.get('yaw_rate', 0) * dt
        else:
            pred_dx, pred_dy, pred_dtheta = self.last_pure_dx, self.last_pure_dy, self.last_dtheta

        raw_pure_dx, raw_pure_dy, raw_dtheta = 0.0, pred_dy, pred_dtheta

        if succ1 and succ2:
            # Вычисляем вращение из разности поперечных сдвигов ТОЛЬКО боковых зон для резкости
            if succ_side1 and succ_side2:
                denom = ((self.t1_side_y1 + self.t1_side_y2) / 2.0 - (self.t2_side_y1 + self.t2_side_y2) / 2.0)
                if denom != 0:
                    raw_dtheta = (dx_side1 - dx_side2) / denom * 1.2  # Увеличен коэффициент для резкости
                else:
                    raw_dtheta = pred_dtheta
            else:
                # Если нет боковых, используем комбинированные
                denom = (cY1 - cY2) if (cY1 - cY2) != 0 else 1.0
                raw_dtheta = (dx1 - dx2) / denom
            
            # Математически компенсируем тангенциальное движение (x=center_x)
            c1_x = (b1_x1 + b1_x2) / 2.0
            c2_x = (b2_x1 + b2_x2) / 2.0
            
            true_dy1 = dy1 - (c1_x - self.center_x) * raw_dtheta
            true_dy2 = dy2 - (c2_x - self.center_x) * raw_dtheta
            raw_pure_dy = (true_dy1 + true_dy2) / 2.0
            
            if self.frame_count > 5:
                if abs(raw_dtheta - self.last_dtheta) > 0.1: raw_dtheta = self.last_dtheta  # Увеличен порог для резких поворотов
                if abs(raw_pure_dy - self.last_pure_dy) > 20: raw_pure_dy = self.last_pure_dy

            alpha_pos, alpha_ang = (0.6, 0.3) if self.frame_count > 5 else (1.0, 1.0)  # Уменьшен alpha_pos для лучшей точности
            pure_dx = self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        elif succ1:
            c1_x = (b1_x1 + b1_x2) / 2.0
            raw_dtheta = pred_dtheta * 0.95
            raw_pure_dy = dy1 - (c1_x - self.center_x) * raw_dtheta
            
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        elif succ2:
            c2_x = (b2_x1 + b2_x2) / 2.0
            raw_dtheta = pred_dtheta * 0.95
            raw_pure_dy = dy2 - (c2_x - self.center_x) * raw_dtheta
            
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        else:
            pure_dx, pure_dy, dtheta = pred_dx * 0.9, pred_dy, pred_dtheta * 0.9 

        if telemetry and self.use_telemetry:
            pure_dy = pred_dy

        self.last_pure_dx, self.last_pure_dy = pure_dx, pure_dy
        self.dtheta_history.append(dtheta)
        if len(self.dtheta_history) > self.max_history: self.dtheta_history.pop(0)
        filtered_dtheta = np.median(self.dtheta_history)
        self.last_dtheta = filtered_dtheta
        
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        global_dx = -pure_dx * cos_t + pure_dy * sin_t
        global_dy = -pure_dx * sin_t - pure_dy * cos_t
        
        self.x += global_dx
        self.y += global_dy
        self.theta += filtered_dtheta
        self.prev_gray = current_gray
        
        debug_data = None
        if self.debug_mode:
            disp = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)
            # 1. Зоны поиска Ближнего (T1) - бледно-зеленый
            for (rx1, rx2) in self.t1_side_regions:
                cv2.rectangle(disp, (rx1, self.t1_side_y1), (rx2, self.t1_side_y2), (200, 255, 200), 1)
            for (rx1, rx2) in self.t1_center_regions:
                cv2.rectangle(disp, (rx1, self.t1_center_y1), (rx2, self.t1_center_y2), (200, 255, 200), 1)
            
            # 2. Зоны поиска Дальнего (T2) - нейтральный серый
            for (rx1, rx2) in self.t2_side_regions:
                cv2.rectangle(disp, (rx1, self.t2_side_y1), (rx2, self.t2_side_y2), (150, 150, 150), 1)
            for (rx1, rx2) in self.t2_center_regions:
                cv2.rectangle(disp, (rx1, self.t2_center_y1), (rx2, self.t2_center_y2), (150, 150, 150), 1)
                
            # 3. Подсвечиваем результаты Ближнего (T1) - ЯРКИЙ ЖИРНЫЙ ЗЕЛЕНЫЙ
            if succ_side1:
                cv2.rectangle(disp, (int(b1_x1_s), self.t1_side_y1), (int(b1_x2_s), self.t1_side_y2), (0, 255, 0), 2)
            if succ_center1:
                cv2.rectangle(disp, (int(b1_x1_c), self.t1_center_y1), (int(b1_x2_c), self.t1_center_y2), (0, 255, 0), 2)
                
            # 4. Результаты Дальнего (T2) - серый
            if succ_side2:
                cv2.rectangle(disp, (int(b2_x1_s), self.t2_side_y1), (int(b2_x2_s), self.t2_side_y2), (200, 200, 200), 1)
            if succ_center2:
                cv2.rectangle(disp, (int(b2_x1_c), self.t2_center_y1), (int(b2_x2_c), self.t2_center_y2), (200, 200, 200), 1)
            
            if not succ1: kp1_img = np.zeros((self.t1_side_y2 - self.t1_side_y1, 100), dtype=np.uint8)
            if not succ2: kp2_img = np.zeros((self.t2_side_y2 - self.t2_side_y1, 100), dtype=np.uint8)
            
            debug_data = (disp, kp1_img, kp2_img)

        return np.array([self.x, self.y, self.theta]), debug_data

    def get_stats(self):
        self.stats['avg_matches'] = self.stats['successful_matches'] / max(1, self.stats['total_frames'])
        return self.stats

    def reset(self):
        self.prev_gray = None
        self.x = self.y = self.theta = 0.0
        self.frame_count = 0
        self.last_pure_dx = self.last_pure_dy = self.last_dtheta = 0.0
        self.stats = {'total_frames': 0, 'successful_matches': 0}