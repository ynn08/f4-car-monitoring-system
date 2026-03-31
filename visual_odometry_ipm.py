import cv2
import numpy as np


class VisualOdometryIPM:
    def __init__(self, max_features=3000, min_matches=50, ransac_thresh=5.0):
        self.detector = cv2.ORB_create(nfeatures=max_features, edgeThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_desc = None
        self.prev_kp = None
        self.prev_pos = np.array([0.0, 0.0, 0.0])
        self.motion_buffer = []
        self.smoothing_window = 3  # Уменьшено для лучшей реакции
        
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self.frame_skip = 0  # Пропуск кадров для стабильности
        
    def _detect_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Убираем черные края из рассмотрения
        mask = gray > 10
        # 600 400
        # 338 329
        # 389 386
        mask[329:386, 338:390] = False
        mask[352:370, 385:410] = False

        kp, desc = self.detector.detectAndCompute(gray, mask=mask.astype(np.uint8))

        result = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))
        # cv2.imshow('Result' + str(random.randint(0, 1000)), result)
        img_with_kp = cv2.drawKeypoints(gray, kp, None, color=(0, 255, 0))
        # cv2.imshow('img_with_kp' + str(random.randint(0, 1000)), img_with_kp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        
        return kp, desc

    def update(self, bev_image, dt=0.1):
        self.frame_skip += 1
        
        kp, desc = self._detect_features(bev_image)
        if self.prev_desc is None:
            self.prev_kp, self.prev_desc = kp, desc
            return self.prev_pos.copy()
        
        # Пропускаем каждый 2-й кадр для стабильности
        if self.frame_skip < 2:
            return self.prev_pos.copy()
        self.frame_skip = 0
            
        if desc is None or len(kp) < self.min_matches:
            return self._predict_motion(dt)
        
        matches = self.bf.match(self.prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Берём только лучшие совпадения
        matches = matches[:min(200, len(matches))]
        
        if len(matches) < self.min_matches:
            return self._predict_motion(dt)
            
        prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, self.ransac_thresh)
        
        if H is None:
            return self._predict_motion(dt)
        
        # Считаем количество инлайеров
        inlier_ratio = np.sum(mask) / len(mask)
        if inlier_ratio < 0.3:  # Мало инлайеров - ненадёжно
            return self._predict_motion(dt)
            
        dx, dy, dtheta = self._decompose_homography(H, prev_pts, curr_pts, mask)
        
        # Ограничиваем максимальное движение между кадрами
        max_movement = 2.0  # метра
        max_rotation = 0.3  # радиан (~17 градусов)
        
        dx = np.clip(dx, -max_movement, max_movement)
        dy = np.clip(dy, -max_movement, max_movement)
        dtheta = np.clip(dtheta, -max_rotation, max_rotation)
        
        smoothed_dtheta = self._smooth_motion(dtheta, dt)
        
        new_x = self.prev_pos[0] + dx * np.cos(self.prev_pos[2]) - dy * np.sin(self.prev_pos[2])
        new_y = self.prev_pos[1] + dx * np.sin(self.prev_pos[2]) + dy * np.cos(self.prev_pos[2])
        new_theta = self.prev_pos[2] + smoothed_dtheta
        
        self.prev_pos = np.array([new_x, new_y, new_theta])
        self.prev_kp, self.prev_desc = kp, desc
        
        return self.prev_pos.copy()
    
    def _decompose_homography(self, H, prev_pts, curr_pts, mask):
        H = H / H[2, 2]
        theta = np.arctan2(H[1, 0], H[0, 0])
        
        if mask is not None:
            inlier_prev = prev_pts[mask.ravel() == 1]
            inlier_curr = curr_pts[mask.ravel() == 1]
            if len(inlier_prev) > 0:
                center_prev = np.mean(inlier_prev, axis=0)
                center_curr = np.mean(inlier_curr, axis=0)
                dx = center_curr[0, 0] - center_prev[0, 0]
                dy = center_curr[0, 1] - center_prev[0, 1]
            else:
                dx, dy = H[0, 2], H[1, 2]
        else:
            dx, dy = H[0, 2], H[1, 2]
            
        return dx, dy, theta
    
    def _smooth_motion(self, dtheta, dt):
        angular_vel = dtheta / dt if dt > 0 else 0
        self.motion_buffer.append(angular_vel)
        if len(self.motion_buffer) > self.smoothing_window:
            self.motion_buffer.pop(0)
        avg_vel = np.median(self.motion_buffer)
        return avg_vel * dt
    
    def _predict_motion(self, dt):
        if len(self.motion_buffer) > 0:
            last_vel = self.motion_buffer[-1]
            dtheta = last_vel * dt
            dx = 0.5 * dt
            new_x = self.prev_pos[0] + dx * np.cos(self.prev_pos[2])
            new_y = self.prev_pos[1] + dx * np.sin(self.prev_pos[2])
            new_theta = self.prev_pos[2] + dtheta
            self.prev_pos = np.array([new_x, new_y, new_theta])
        return self.prev_pos.copy()

    def reset(self):
        self.prev_desc = None
        self.prev_kp = None
        self.prev_pos = np.array([0.0, 0.0, 0.0])
        self.motion_buffer = []
        self.frame_skip = 0