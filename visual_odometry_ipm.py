import cv2
import numpy as np

class VisualOdometryIPM:
    def __init__(self, 
                 nfeatures=2000, 
                 edge_threshold=31, 
                 patch_size=31, 
                 fast_threshold=20,
                 use_telemetry=False,
                 debug_mode=False):
        
        self.detector = cv2.ORB_create(
            nfeatures=nfeatures, 
            edgeThreshold=edge_threshold, 
            patchSize=patch_size, 
            fastThreshold=fast_threshold
        )
        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.prev_kp = None
        self.prev_desc = None
        self.prev_frame = None
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.use_telemetry = use_telemetry
        self.debug_mode = debug_mode
        self.frame_count = 0
        
        self.ransac_reproj_threshold = 3.0
        self.min_matches = 10
        
        # Для статистики
        self.stats = {
            'total_frames': 0,
            'successful_matches': 0,
            'avg_matches': 0,
            'total_matches': 0
        }

    def _detect_features(self, image, mask=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if mask is None:
            mask = gray > 10
        
        kp, desc = self.detector.detectAndCompute(gray, mask=mask.astype(np.uint8))
        return kp, desc, gray

    def _get_car_mask(self, shape):
        """Создает маску, исключающую машину из кадра BEV"""
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        
        # На основе вашего изображения - машина внизу по центру
        cx, cy = w // 2, int(h * 0.90)
        car_radius = int(min(w, h) * 0.08)
        
        # Круглая маска для машины
        cv2.circle(mask, (cx, cy), car_radius, 0, -1)
        
        # Дополнительно убираем нижнюю часть кадра (там часто артефакты IPM)
        mask[int(h * 0.85):, :] = 0
        return mask

    def _estimate_motion(self, kp1, desc1, kp2, desc2):
        if desc1 is None or desc2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None, None

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            return None, good_matches

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, 
                                              ransacReprojThreshold=self.ransac_reproj_threshold)
        
        if M is None:
            return None, good_matches

        return M, good_matches

    def _matrix_to_delta(self, M):
        dx = M[0, 2]
        dy = M[1, 2]
        angle = np.arctan2(M[1, 0], M[0, 0])
        return dx, dy, angle

    def create_debug_image(self, frame1, frame2, kp1, kp2, matches, M=None):
        """Создает отладочное изображение с ключевыми точками и матчами"""
        
        # Конвертируем в BGR если нужно
        if len(frame1.shape) == 2:
            img1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        else:
            img1 = frame1.copy()
            img2 = frame2.copy()
        
        # Рисуем ключевые точки
        kp1_img = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
        kp2_img = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
        
        # Рисуем матчи
        if matches:
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:50],  # Показываем первые 50 матчей
                None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=2
            )
        else:
            match_img = np.hstack([img1, img2])
            cv2.putText(match_img, "NO MATCHES", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Добавляем статистику
        n_matches = len(matches) if matches else 0
        info_text = [
            f"Frame: {self.frame_count}",
            f"KP1: {len(kp1)}",
            f"KP2: {len(kp2)}",
            f"Matches: {n_matches}",
            f"Success: {self.stats['successful_matches']}/{self.stats['total_frames']}",
        ]
        
        if M is not None:
            dx, dy, angle = self._matrix_to_delta(M)
            info_text.extend([
                f"dx: {dx:.1f} px",
                f"dy: {dy:.1f} px",
                f"dtheta: {np.degrees(angle):.2f} deg",
            ])
        
        y_offset = 30
        for text in info_text:
            cv2.putText(match_img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 25
        
        return match_img, kp1_img, kp2_img

    def update(self, bev_image, telemetry=None):
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        car_mask = self._get_car_mask(bev_image.shape)
        current_kp, current_desc, current_gray = self._detect_features(bev_image, mask=car_mask)
        
        dx, dy, dtheta = 0.0, 0.0, 0.0
        confidence = 0.0
        M = None
        matches = None
        
        debug_data = None

        if self.prev_desc is not None:
            M, matches = self._estimate_motion(self.prev_kp, self.prev_desc, current_kp, current_desc)
            
            if M is not None:
                dx, dy, dtheta = self._matrix_to_delta(M)
                print(dx, dy, dtheta)
                n_matches = len(matches)
                confidence = min(1.0, n_matches / 50.0)
                self.stats['successful_matches'] += 1
                self.stats['total_matches'] += n_matches
                self.stats['avg_matches'] = self.stats['total_matches'] / self.stats['successful_matches']
            else:
                confidence = 0.0
        
        if telemetry and self.use_telemetry:
            ppm = 50/0
            dist_m = telemetry.get('speed', 0) * telemetry.get('dt', 1/30)
            dist_px = dist_m * ppm
            
            tel_dy = dist_px
            tel_dx = 0.0
            tel_dtheta = telemetry.get('yaw_rate', 0) * telemetry.get('dt', 1/30)
            
            if confidence > 0.3:
                dtheta = 0.7 * dtheta + 0.3 * tel_dtheta
                dy = 0.4 * dy + 0.6 * tel_dy
            else:
                dx, dy, dtheta = tel_dx, tel_dy, tel_dtheta

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        
        global_dx = (-dx * cos_t + dy * sin_t)
        global_dy = (-dx * sin_t - dy * cos_t)
        
        self.x += global_dx
        self.y += global_dy
        self.theta += dtheta
        
        # Сохраняем для отладки
        if self.debug_mode and self.prev_frame is not None:
            debug_data = self.create_debug_image(
                self.prev_frame, current_gray, 
                self.prev_kp, current_kp, 
                matches, M
            )
        
        self.prev_kp = current_kp
        self.prev_desc = current_desc
        self.prev_frame = current_gray
        
        return np.array([self.x, self.y, self.theta]), debug_data

    def get_stats(self):
        return self.stats

    def reset(self):
        self.prev_kp = None
        self.prev_desc = None
        self.prev_frame = None
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.frame_count = 0
        self.stats = {
            'total_frames': 0,
            'successful_matches': 0,
            'avg_matches': 0,
            'total_matches': 0
        }