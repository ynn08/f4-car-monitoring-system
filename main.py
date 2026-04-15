import os
import cv2
import numpy as np
import json
from tqdm import tqdm

from ipm import IPM_4Points
from map_builder import LocalMapBuilder
from visual_odometry_ipm import VisualOdometryIPM


import csv

# Синхронизация: телеметрия 0:085 (0.085с) = видео кадр 139 (139/30 = 4.633с)
# => video_time = telemetry_time + TELEMETRY_VIDEO_OFFSET
TELEMETRY_VIDEO_OFFSET = 139.0 / 30.0 - 0.085  # ≈ 4.548 секунд

def load_telemetry(telemetry_path):
    """Загрузка телеметрии из CSV с точной синхронизацией по видео"""
    if not telemetry_path or not os.path.exists(telemetry_path):
        return None
    try:
        times = []
        speeds = []
        with open(telemetry_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                t_str = row.get('Time', '0:000').strip()
                parts = t_str.split(':')
                # Формат M:SSS (минуты:миллисекунды)
                t_sec = float(parts[0]) * 60.0 + float(parts[1]) / 1000.0 if len(parts) == 2 else 0.0
                
                speed_str = row.get('CarSpeed', '0').replace(',', '.')
                speed_ms = float(speed_str) / 3.6  # km/h -> m/s
                
                times.append(t_sec)
                speeds.append(speed_ms)
        
        print(f"📡 Telemetry loaded: {len(times)} records, {times[-1]:.1f}s duration")
        print(f"   Speed range: {min(speeds)*3.6:.1f} - {max(speeds)*3.6:.1f} km/h")
        print(f"   Sync offset: {TELEMETRY_VIDEO_OFFSET:.3f}s (tel 0:085 = video frame 139)")
        
        return {
            'time': np.array(times),
            'speed': np.array(speeds)
        }
    except Exception as e:
        print(f"❌ Error loading telemetry: {e}")
        return None


def build_map_from_video_4points(video_path, 
                                  telemetry_path=None,
                                  output_map_path="local_map.png",
                                  start_frame=0,
                                  end_frame=None,
                                  save_bev_video=False,
                                  debug_mode=True,          # 🔥 Включить отладку
                                  show_debug_windows=True,  # 🔥 Показывать окна
                                  debug_skip_frames=5       # 🔥 Показывать каждый N-й кадр
                                  ):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps
    cap.release()

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        print(f"Error: start_frame ({start_frame}) >= end_frame ({end_frame})")
        return
    
    process_frames = end_frame - start_frame
    print(f"\n{'='*60}")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Frames: {start_frame} - {end_frame} ({process_frames} frames)")
    print(f"Debug Mode: {debug_mode}")
    print(f"{'='*60}\n")
    
    # Идеально отцентрированная трапеция, которая сохраняет ОРИГИНАЛЬНЫЙ МАСШТАБ (ширина 446px снизу, 260px сверху),
    # но симметрична относительно Vanishing Point (x=823). Убирает Shear, сохраняет 99% успешных матчей.
    src_pts = np.float32([
        [823 - 223, 535], # Bottom Left 
        [823 + 223, 535], # Bottom Right
        [823 - 130, 435], # Top Left    
        [823 + 130, 435]  # Top Right   
    ])
    bev_width = 600
    bev_height = 400
    
    car_size = (1.47, 2.80)
    ppm = 92 # Возвращаем оригинальный масштаб, дающий лучшую точность
    rect_size = (ppm, ppm * car_size[1] / car_size[0])
    
    # Смещаем дорогу к нижнему краю (0.90) и центрируем по горизонтали
    rect_left_up_pos = (bev_width // 2 - rect_size[0] // 2, bev_height * 0.90 - rect_size[1])

    dst_pts = np.float32([
        [rect_left_up_pos[0], rect_left_up_pos[1] + rect_size[1]],                   
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1] + rect_size[1]],   
        [rect_left_up_pos[0], rect_left_up_pos[1]],                                  
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1]],                   
    ])

    ipm = IPM_4Points(src_pts, dst_pts, output_size=(bev_width, bev_height))
    builder = LocalMapBuilder(
        pixels_per_meter=int(ppm / 1.47),      # Синхронизировано с PPM
        initial_size=5000,        # Начальный холст
        blend_decay=0.02,         # Уменьшаем затухание для более плотной карты
        use_distance_weighting=True,
        scale_factor=0.02        # Увеличиваем разрешение карты в 2 раза
    )
    
    use_telemetry = telemetry_path is not None
    vo = VisualOdometryIPM(
        use_telemetry=use_telemetry,
        debug_mode=debug_mode
    )

    telemetry_data = load_telemetry(telemetry_path) if telemetry_path else None

    bevs = []
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    debug_frame_count = 0
    pbar = tqdm(total=process_frames, desc="Building map")

    # Создаем окна для отладки
    if debug_mode and show_debug_windows:
        cv2.namedWindow("BEV + Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Keypoints Frame 1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Keypoints Frame 2", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", 1200, 400)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
        frame_count += 1
        
        # 1. IPM трансформация
        bev = ipm.transform(frame)
        # cv2.imwrite(f"BEV{frame_count}.png", bev)
        bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if len(bev.shape) == 3 else bev.copy()
        
        # 2. Маска машины (для визуализации и отсечения)
        # Вырезаем прямоугольником ровно нижнюю-центральную часть, где торчит нос машины
        car_mask = np.ones(bev.shape[:2], dtype=np.uint8) * 255
        cv2.rectangle(car_mask, 
                      (bev.shape[1] // 2 - 120, int(bev.shape[0] * 0.75)), 
                      (bev.shape[1] // 2 + 120, bev.shape[0]), 
                      0, -1)
        bev_masked = cv2.bitwise_and(bev, bev, mask=car_mask)
        if debug_mode:
            weight_viz = (car_mask).astype(np.uint8)
            weight_viz = cv2.applyColorMap(weight_viz, cv2.COLORMAP_JET)
            cv2.imshow("Distance Weights", weight_viz)
        
        # 3. Телеметрия — интерполируем скорость по времени видеокадра
        current_telemetry = None
        if telemetry_data is not None:
            video_time = frame_count / fps  # абсолютное время видео
            telemetry_time = video_time - TELEMETRY_VIDEO_OFFSET
            if telemetry_time >= telemetry_data['time'][0]:
                speed_interp = float(np.interp(
                    telemetry_time,
                    telemetry_data['time'],
                    telemetry_data['speed']
                ))
                current_telemetry = {
                    'dt': dt,
                    'speed': speed_interp
                }

        # 4. Визуальная одометрия
        pose, debug_data = vo.update(bev, telemetry=current_telemetry)
        
        # 5. Уточнение по карте
        refined_pose, ref_val = builder.refine_pose_against_map(bev, pose, search_range=10)
        if ref_val > 0.85: # Еще немного повысим порог
            # Проверка на физическую правдоподобность (не прыгаем больше чем на 1 метр за кадр)
            dist = np.linalg.norm(refined_pose[:2] - pose[:2])
            if dist < 80: # ~0.8-0.9 метра при ppm=92
                alpha = 0.15
                pose = (1.0 - alpha) * pose + alpha * refined_pose
            
        # 6. Отладочная визуализация
        if debug_mode and show_debug_windows and debug_data is not None:
            debug_frame_count += 1
            if debug_frame_count % debug_skip_frames == 0:
                match_img, kp1_img, kp2_img = debug_data
                
                # Показываем BEV с маской
                bev_display = bev.copy()
                bev_display[car_mask == 0] = [0, 0, 255]  # Красная зона маски
                cv2.putText(bev_display, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("BEV + Mask", bev_display)
                cv2.imshow("Matches", match_img)
                cv2.imshow("Keypoints Frame 1", kp1_img)
                cv2.imshow("Keypoints Frame 2", kp2_img)
                cv2.waitKey(1)
                
                # Ждем 1мс, если нажали 'q' - выходим
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ Interrupted by user")
                    break
                elif key == ord(' '):
                    # Пробел - пауза
                    cv2.waitKey(0)
        
        # 7. Добавляем на карту
        builder.add_frame(bev, pose, frame_idx=frame_count)
        
        if save_bev_video:
            bevs.append(bev)
            
        pbar.update(1)

    cap.release()
    pbar.close()
    
    if debug_mode and show_debug_windows:
        cv2.destroyAllWindows()

    # Статистика одометрии
    stats = vo.get_stats()
    print(f"\n{'='*60}")
    print(f"📊 ODOMETRY STATISTICS:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Successful matches: {stats['successful_matches']}")
    print(f"  Success rate: {stats['successful_matches']/max(1, stats['total_frames'])*100:.1f}%")
    print(f"  Avg matches per frame: {stats['avg_matches']:.1f}")
    print(f"  Map-to-Frame Refinements: {builder.refinement_count}")
    print(f"{'='*60}\n")

    if save_bev_video:
        ipm_video = cv2.VideoWriter(
            f"{video_path[:video_path.rfind('.')]}_ipm_formatted.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (bev_width, bev_height)
        )
        for bev in bevs:
            ipm_video.write(bev)
        ipm_video.release()

    builder.save_map(output_map_path, draw_trajectory=True)
    print(f"✓ Map saved with trajectory to: {output_map_path}")

    final_map = builder.get_map(draw_trajectory=True)

    cv2.imshow("Local Map", final_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n📈 FINAL STATISTICS:")
    print(f"  Frames processed: {frame_count - start_frame}")
    print(f"  Map size: {final_map.shape[1]}x{final_map.shape[0]} pixels")
    print(f"  Trajectory points: {len(builder.poses)}")
    
    # Сохраняем траекторию
    trajectory_file = "trajectory.json"
    trajectory_data = []
    for i in range(len(builder.poses)):
        p = builder.poses[i]
        trajectory_data.append({
            'frame': int(builder.frame_indices[i]),
            'x': float(p[0]),
            'y': float(p[1]),
            'theta': float(p[2])
        })
    
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=4)
    print(f"✓ Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    VIDEO_PATH = "data/Q2 354.MP4"
    TELEMETRY_PATH = "data/telemetry.csv"
    
    # START_FRAME = 1038
    START_FRAME = 0

    END_FRAME = START_FRAME + 3200
    
    build_map_from_video_4points(
        VIDEO_PATH, 
        telemetry_path=TELEMETRY_PATH,
        start_frame=START_FRAME, 
        end_frame=END_FRAME,
        save_bev_video=False,
        debug_mode=True,       # 🔥 Включить отладку
        show_debug_windows=True,# 🔥 Показывать окна
        debug_skip_frames=1,       # 🔥 Каждый 3-й кадр
    )