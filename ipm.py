import cv2
import numpy as np


class IPM_4Points:
    def __init__(self, src_pts, dst_pts, output_size):
        """
        src_pts: 4 точки на исходном изображении [x, y] (трапеция дороги)
                 Порядок: [лево-низ, право-низ, право-верх, лево-верх]
        dst_pts: 4 точки на целевом BEV изображении [x, y] (прямоугольник)
                 Порядок: [лево-низ, право-низ, право-верх, лево-верх]
        output_size: (width, height) выходного BEV изображения в пикселях
        """
        self.src_pts = np.float32(src_pts)
        self.dst_pts = np.float32(dst_pts)
        self.output_size = output_size
        
        # Вычисляем гомографию
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
            
    def transform(self, image):
        """Преобразует изображение в BEV вид"""
        bev = cv2.warpPerspective(image, self.H, self.output_size)
        return bev
    
    def draw_source_points(self, image, color=(0, 255, 0), radius=5):
        """Рисует выбранные точки на изображении для визуализации"""
        img_copy = image.copy()
        for i, pt in enumerate(self.src_pts):
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), radius, color, -1)
            cv2.putText(img_copy, str(i), (int(pt[0])+10, int(pt[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img_copy