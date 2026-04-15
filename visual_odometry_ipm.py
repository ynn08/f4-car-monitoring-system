import cv2
import numpy as np

class VisualOdometryIPM:
    def __init__(self, 
                 use_telemetry=False,
                 debug_mode=False):
        
        self.prev_gray = None
        
        # Глобальная поза (X, Y, Theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.use_telemetry = use_telemetry
        self.debug_mode = debug_mode
        self.frame_count = 0
        
        self.stats = {
            'total_frames': 0,
            'successful_matches': 0,
        }
        
        # История dtheta для медианной фильтрации
        self.dtheta_history = []
        self.max_history = 5
        
        # Состояние для модели постоянной скорости
        self.last_pure_dx = 0.0
        self.last_pure_dy = 0.0
        self.last_dtheta = 0.0
        
        # Настройки шаблона (ближний)
        self.t1_y1, self.t1_y2 = 150, 270 
        self.t1_regions = [(100, 350), (250, 500)] # Большая левая зона, Большая правая зона
        self.cY1 = (self.t1_y1 + self.t1_y2) / 2.0
        
        # Настройки шаблона (дальний)
        self.t2_y1, self.t2_y2 = 10, 110  
        self.t2_regions = [(100, 350), (250, 500)]
        self.cY2 = (self.t2_y1 + self.t2_y2) / 2.0

        self.confidence_threshold = 0.5

    def find_best_match(self, current_gray, regions, y1, y2):
        best_val = -1
        best_shift = None
        best_x1 = 0
        best_x2 = 0
        best_kp = None
        for (x1, x2) in regions:
            template = self.prev_gray[y1:y2, x1:x2]
            # Ищем в окрестности этого региона на новом кадре
            search_x1 = max(0, x1 - 50)
            search_x2 = min(current_gray.shape[1], x2 + 50)
            search_y1 = max(0, y1 - 30)
            search_y2 = min(current_gray.shape[0], y2 + 50 + 20) # ожидаем сдвиг вниз (машина едет вперед)
            if search_y2 <= search_y1 or search_x2 <= search_x1: continue
            
            search_area = current_gray[search_y1:search_y2, search_x1:search_x2]
            if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
                continue
                
            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                # Считаем сдвиг
                match_x = search_x1 + max_loc[0]
                match_y = search_y1 + max_loc[1]
                dx = match_x - x1
                dy = match_y - y1
                best_shift = (dx, dy)
                best_x1, best_x2 = x1, x2
                best_kp = current_gray[y1:y2, x1:x2]
                
        if best_val >= self.confidence_threshold:
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

        succ1, dx1, dy1, b1_x1, b1_x2, kp1_img = self.find_best_match(current_gray, self.t1_regions, self.t1_y1, self.t1_y2)
        succ2, dx2, dy2, b2_x1, b2_x2, kp2_img = self.find_best_match(current_gray, self.t2_regions, self.t2_y1, self.t2_y2)

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
            # Вычисляем вращение из разности поперечных сдвигов (не зависит от горизонтальной позиции если шаблон жесткий)
            raw_dtheta = (dx1 - dx2) / (self.cY1 - self.cY2)
            
            # Математически компенсируем тангенциальное движение, вызванное физическим отдалением шаблона от центра вращения машины (x=300)
            c1_x = (b1_x1 + b1_x2) / 2.0
            c2_x = (b2_x1 + b2_x2) / 2.0
            
            true_dy1 = dy1 - (c1_x - 300) * raw_dtheta
            true_dy2 = dy2 - (c2_x - 300) * raw_dtheta
            raw_pure_dy = (true_dy1 + true_dy2) / 2.0
            
            if self.frame_count > 5:
                if abs(raw_dtheta - self.last_dtheta) > 0.05: raw_dtheta = self.last_dtheta
                if abs(raw_pure_dy - self.last_pure_dy) > 30: raw_pure_dy = self.last_pure_dy

            alpha_pos, alpha_ang = (0.8, 0.3) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = self.last_pure_dx # поперечное смещение чисто от одометрии отключено для стабильности
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        elif succ1:
            c1_x = (b1_x1 + b1_x2) / 2.0
            raw_dtheta = pred_dtheta * 0.95
            raw_pure_dy = dy1 - (c1_x - 300) * raw_dtheta
            
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        elif succ2:
            c2_x = (b2_x1 + b2_x2) / 2.0
            raw_dtheta = pred_dtheta * 0.95
            raw_pure_dy = dy2 - (c2_x - 300) * raw_dtheta
            
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
        else:
            pure_dx, pure_dy, dtheta = pred_dx * 0.9, pred_dy, pred_dtheta * 0.9 

        if telemetry and self.use_telemetry:
            # Полностью доверяем скорости из телеметрии (решает проблему сжатия прямых трасс)
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
            for (rx1, rx2) in self.t1_regions: cv2.rectangle(disp, (rx1, self.t1_y1), (rx2, self.t1_y2), (0, 100, 0), 1)
            for (rx1, rx2) in self.t2_regions: cv2.rectangle(disp, (rx1, self.t2_y1), (rx2, self.t2_y2), (0, 0, 100), 1)
                
            if succ1:
                cv2.rectangle(disp, (b1_x1, self.t1_y1), (b1_x2, self.t1_y2), (0, 255, 0), 2)
                match_x = b1_x1 + int(dx1)
                match_y = self.t1_y1 + int(dy1)
                cv2.circle(disp, (match_x + (b1_x2-b1_x1)//2, match_y + (self.t1_y2-self.t1_y1)//2), 5, (0, 255, 255), -1)
            
            if not succ1: kp1_img = np.zeros((self.t1_y2 - self.t1_y1, 100), dtype=np.uint8)
            if not succ2: kp2_img = np.zeros((self.t2_y2 - self.t2_y1, 100), dtype=np.uint8)
            
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