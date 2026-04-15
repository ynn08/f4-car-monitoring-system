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
        self.t1_x1, self.t1_x2 = 150, 450
        self.cY1 = (self.t1_y1 + self.t1_y2) / 2.0
        
        # Настройки шаблона (дальний)
        self.t2_y1, self.t2_y2 = 10, 110  
        self.t2_x1, self.t2_x2 = 150, 450
        self.cY2 = (self.t2_y1 + self.t2_y2) / 2.0

        self.confidence_threshold = 0.5

    def _match_template(self, current_gray, template, t_x1, t_y1, pred_dx, pred_dy, w, h, t_angle=0):
        # Поворачиваем шаблон, если есть предсказанное вращение
        if t_angle != 0:
            h_t, w_t = template.shape[:2]
            rot_m = cv2.getRotationMatrix2D((w_t//2, h_t//2), np.degrees(t_angle), 1.0)
            template = cv2.warpAffine(template, rot_m, (w_t, h_t), borderMode=cv2.BORDER_REFLECT_101)

        search_x1 = max(0, int(t_x1 + pred_dx - 120))
        search_x2 = min(w, int(t_x1 + template.shape[1] + pred_dx + 120))
        
        if self.stats['successful_matches'] < 5:
            search_y1 = max(0, int(t_y1 - 40))
            search_y2 = min(h, int(t_y1 + template.shape[0] + 160))
        else:
            search_y1 = max(0, int(t_y1 + pred_dy - 100))
            search_y2 = min(h, int(t_y1 + template.shape[0] + pred_dy + 120))

        if search_x2 - search_x1 >= template.shape[1] and search_y2 - search_y1 >= template.shape[0]:
            search_roi = current_gray[search_y1:search_y2, search_x1:search_x2]
            res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= self.confidence_threshold:
                mx, my = max_loc[0], max_loc[1]
                sub_x, sub_y = 0.0, 0.0
                if 0 < mx < res.shape[1] - 1 and 0 < my < res.shape[0] - 1:
                    l, c, r = res[my, mx-1], res[my, mx], res[my, mx+1]
                    denom_x = 2.0 * (l - 2.0 * c + r)
                    if denom_x != 0: sub_x = (l - r) / denom_x
                    u, c, d = res[my-1, mx], res[my, mx], res[my+1, mx]
                    denom_y = 2.0 * (u - 2.0 * c + d)
                    if denom_y != 0: sub_y = (u - d) / denom_y

                match_x = mx + sub_x + search_x1
                match_y = my + sub_y + search_y1
                dx = match_x - t_x1
                dy = match_y - t_y1
                
                if pred_dy > 30 and dy < 20:
                    return False, 0.0, 0.0, max_val
                return True, dx, dy, max_val
        return False, 0.0, 0.0, 0.0

    def update(self, bev_image, telemetry=None):
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        current_gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        current_gray = clahe.apply(current_gray)

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return np.array([self.x, self.y, self.theta]), None

        template1 = self.prev_gray[self.t1_y1:self.t1_y2, self.t1_x1:self.t1_x2]
        template2 = self.prev_gray[self.t2_y1:self.t2_y2, self.t2_x1:self.t2_x2]
        h_img, w_img = current_gray.shape

        if telemetry and self.use_telemetry:
            ppm = getattr(self, 'ppm', 80.0) 
            dt = telemetry.get('dt', 1/30)
            pred_dy = telemetry.get('speed', 0) * dt * ppm
            pred_dx = 0.0
            pred_dtheta = telemetry.get('yaw_rate', 0) * dt
        else:
            pred_dx, pred_dy, pred_dtheta = self.last_pure_dx, self.last_pure_dy, self.last_dtheta

        Y_ROT = 340
        t1_pred_dx = pred_dx + (self.cY1 - Y_ROT) * pred_dtheta
        t2_pred_dx = pred_dx + (self.cY2 - Y_ROT) * pred_dtheta

        succ1, dx1, dy1, val1 = self._match_template(current_gray, template1, self.t1_x1, self.t1_y1, t1_pred_dx, pred_dy, w_img, h_img, t_angle=pred_dtheta)
        succ2, dx2, dy2, val2 = self._match_template(current_gray, template2, self.t2_x1, self.t2_y1, t2_pred_dx, pred_dy, w_img, h_img, t_angle=pred_dtheta)

        if succ1 and succ2:
            raw_dtheta = (dx1 - dx2) / (self.cY1 - self.cY2)
            raw_pure_dx = dx1 - (self.cY1 - 340) * raw_dtheta
            raw_pure_dy = (dy1 + dy2) / 2.0
            
            if self.frame_count > 5:
                if abs(raw_dtheta - self.last_dtheta) > 0.05: raw_dtheta = self.last_dtheta
                if abs(raw_pure_dy - self.last_pure_dy) > 30: raw_pure_dy = self.last_pure_dy
                if abs(raw_pure_dx - self.last_pure_dx) > 15: raw_pure_dx = self.last_pure_dx

            alpha_pos, alpha_ang = (0.8, 0.3) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            self.stats['successful_matches'] += 1
            
        elif succ1:
            raw_pure_dy, raw_dtheta = dy1, pred_dtheta * 0.95
            raw_pure_dx = dx1 - (self.cY1 - 340) * raw_dtheta
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            self.stats['successful_matches'] += 1
            
        elif succ2:
            raw_pure_dy, raw_dtheta = dy2, pred_dtheta * 0.95
            raw_pure_dx = dx2 - (self.cY2 - 340) * raw_dtheta
            alpha_pos, alpha_ang = (0.4, 0.2) if self.frame_count > 5 else (1.0, 1.0)
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            self.stats['successful_matches'] += 1
        else:
            pure_dx, pure_dy, dtheta = pred_dx * 0.9, pred_dy, pred_dtheta * 0.9 

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
        
        return np.array([self.x, self.y, self.theta]), None

    def get_stats(self):
        self.stats['avg_matches'] = self.stats['successful_matches'] / max(1, self.stats['total_frames'])
        return self.stats

    def reset(self):
        self.prev_gray = None
        self.x = self.y = self.theta = 0.0
        self.frame_count = 0
        self.last_pure_dx = self.last_pure_dy = self.last_dtheta = 0.0
        self.stats = {'total_frames': 0, 'successful_matches': 0}