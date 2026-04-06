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
        
        # Состояние для модели постоянной скорости (храним pure_dx, pure_dy, dtheta)
        self.last_pure_dx = 0.0
        self.last_pure_dy = 0.0
        self.last_dtheta = 0.0
        
        # Настройки шаблона для поиска текстуры асфальта (ближний шаблон)
        self.t1_y1, self.t1_y2 = 150, 270 
        self.t1_x1, self.t1_x2 = 150, 450 # 300px ширина
        self.cY1 = (self.t1_y1 + self.t1_y2) / 2.0
        
        # Настройки шаблона (дальний шаблон - для вычисления вращения)
        self.t2_y1, self.t2_y2 = 10, 110  
        self.t2_x1, self.t2_x2 = 150, 450 # 300px ширина
        self.cY2 = (self.t2_y1 + self.t2_y2) / 2.0

        self.confidence_threshold = 0.5 # Чуть снизим порог, так как шаблоны стали больше

    def _match_template(self, current_gray, template, t_x1, t_y1, pred_dx, pred_dy, w, h):
        # Расширяем поиск по горизонтали (позволяет отслеживать резкие повороты)
        search_x1 = max(0, int(t_x1 + pred_dx - 120))
        search_x2 = min(w, int(t_x1 + template.shape[1] + pred_dx + 120))

        # Расширяем по вертикали с запасом на торможение/ускорение
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
                
                # Subpixel refinement (parabola fitting)
                sub_x, sub_y = 0.0, 0.0
                if 0 < mx < res.shape[1] - 1 and 0 < my < res.shape[0] - 1:
                    left = res[my, mx-1]
                    center = res[my, mx]
                    right = res[my, mx+1]
                    denom_x = 2.0 * (left - 2.0 * center + right)
                    if denom_x != 0:
                        sub_x = (left - right) / denom_x
                        
                    up = res[my-1, mx]
                    down = res[my+1, mx]
                    denom_y = 2.0 * (up - 2.0 * center + down)
                    if denom_y != 0:
                        sub_y = (up - down) / denom_y

                match_x = mx + sub_x + search_x1
                match_y = my + sub_y + search_y1
                dx = match_x - t_x1
                dy = match_y - t_y1
                
                # Защита от статических бликов
                if pred_dy > 30 and dy < 20:
                    return False, 0.0, 0.0, max_val
                return True, dx, dy, max_val
        return False, 0.0, 0.0, 0.0

    def update(self, bev_image, telemetry=None):
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        current_gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE для вытягивания текстуры дороги
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        current_gray = clahe.apply(current_gray)

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return np.array([self.x, self.y, self.theta]), None

        # Извлекаем шаблоны из предыдущего кадра
        template1 = self.prev_gray[self.t1_y1:self.t1_y2, self.t1_x1:self.t1_x2]
        template2 = self.prev_gray[self.t2_y1:self.t2_y2, self.t2_x1:self.t2_x2]
                                  
        h, w = current_gray.shape

        if telemetry and self.use_telemetry:
            ppm = getattr(self, 'ppm', 80.0) 
            dt = telemetry.get('dt', 1/30)
            dist_m = telemetry.get('speed', 0) * dt
            pred_dy = dist_m * ppm
            pred_dx = 0.0
            pred_dtheta = telemetry.get('yaw_rate', 0) * dt
        else:
            pred_dx = self.last_pure_dx
            pred_dy = self.last_pure_dy
            pred_dtheta = self.last_dtheta

        Y_ROT = 340 # Задняя ось машины
        # Для шаблонов pred_dx включает ротацию
        t1_pred_dx = pred_dx + (self.cY1 - Y_ROT) * pred_dtheta
        t2_pred_dx = pred_dx + (self.cY2 - Y_ROT) * pred_dtheta

        succ1, dx1, dy1, val1 = self._match_template(current_gray, template1, self.t1_x1, self.t1_y1, t1_pred_dx, pred_dy, w, h)
        succ2, dx2, dy2, val2 = self._match_template(current_gray, template2, self.t2_x1, self.t2_y1, t2_pred_dx, pred_dy, w, h)

        if succ1 and succ2:
            # Вычисляем сырые (raw) значения
            raw_dtheta = (dx1 - dx2) / (self.cY1 - self.cY2)
            raw_pure_dx = dx1 - (self.cY1 - 340) * raw_dtheta
            raw_pure_dy = (dy1 + dy2) / 2.0
            
            # 1. Отбраковка жестких физических выбросов (Outlier Rejection)
            # Машина не может мгновенно изменить угол поворота руля на огромную величину
            if self.frame_count > 5:
                if abs(raw_dtheta - self.last_dtheta) > 0.05: 
                    raw_dtheta = self.last_dtheta # Игнорируем скачок
                if abs(raw_pure_dy - self.last_pure_dy) > 30:
                    raw_pure_dy = self.last_pure_dy
                if abs(raw_pure_dx - self.last_pure_dx) > 15:
                    raw_pure_dx = self.last_pure_dx

            # 2. Кинематическое сглаживание (Exponential Moving Average)
            # Отпускаем фильтр: болид должен быстро реагировать на руль (alpha_ang = 0.7)
            alpha_pos = 0.8 if self.frame_count > 5 else 1.0
            alpha_ang = 0.7 if self.frame_count > 5 else 1.0 
            
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
            self.last_pure_dx = pure_dx
            self.last_pure_dy = pure_dy
            self.last_dtheta = dtheta
            self.stats['successful_matches'] += 1
            
        elif succ1:
            raw_pure_dy = dy1
            raw_dtheta = pred_dtheta * 0.95 # Плавное возвращение руля в 0
            raw_pure_dx = dx1 - (self.cY1 - 340) * raw_dtheta
            
            alpha_pos = 0.4 if self.frame_count > 5 else 1.0
            alpha_ang = 0.2 if self.frame_count > 5 else 1.0
            
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
            self.last_pure_dx = pure_dx
            self.last_pure_dy = pure_dy
            self.last_dtheta = dtheta
            self.stats['successful_matches'] += 1
            
        elif succ2:
            raw_pure_dy = dy2
            raw_dtheta = pred_dtheta * 0.95
            raw_pure_dx = dx2 - (self.cY2 - 340) * raw_dtheta
            
            alpha_pos = 0.4 if self.frame_count > 5 else 1.0
            alpha_ang = 0.2 if self.frame_count > 5 else 1.0
            
            pure_dx = alpha_pos * raw_pure_dx + (1.0 - alpha_pos) * self.last_pure_dx
            pure_dy = alpha_pos * raw_pure_dy + (1.0 - alpha_pos) * self.last_pure_dy
            dtheta = alpha_ang * raw_dtheta + (1.0 - alpha_ang) * self.last_dtheta
            
            self.last_pure_dx = pure_dx
            self.last_pure_dy = pure_dy
            self.last_dtheta = dtheta
            self.stats['successful_matches'] += 1
            
        else:
            # Fallback к инерции (едем прямо, плавно отпуская руль)
            pure_dx = pred_dx * 0.9
            pure_dy = pred_dy
            dtheta = pred_dtheta * 0.9 
            
            self.last_pure_dx = pure_dx
            self.last_pure_dy = pure_dy
            self.last_dtheta = dtheta

        # Обновляем глобальную позу
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        # pure_dx это сдвиг картинки вправо, то есть машина едет ВЛЕВО.
        # global_dx = (-pure_dx * cos_t + pure_dy * sin_t) - это старый код
        global_dx = (-pure_dx * cos_t + pure_dy * sin_t)
        global_dy = (-pure_dx * sin_t - pure_dy * cos_t)
        
        self.x += global_dx
        self.y += global_dy
        self.theta += dtheta
        
        self.prev_gray = current_gray
        
        return np.array([self.x, self.y, self.theta]), None

    def get_stats(self):
        self.stats['avg_matches'] = self.stats['successful_matches'] / max(1, self.stats['total_frames'])
        return self.stats

    def reset(self):
        self.prev_gray = None
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.frame_count = 0
        self.last_pure_dx = 0.0
        self.last_pure_dy = 0.0
        self.last_dtheta = 0.0
        self.stats = {
            'total_frames': 0,
            'successful_matches': 0,
        }