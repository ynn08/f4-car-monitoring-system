import os
import cv2
import numpy as np
import json
from collections import deque
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
                                  save_window_map_video=False,
                                  window_map_video_path=None,
                                  save_window_map_image=False,
                                  window_map_image_path=None,
                                  build_global_map=False,
                                  debug_mode=True,          # 🔥 Включить отладку
                                  show_debug_windows=True,  # 🔥 Показывать окна
                                  debug_skip_frames=5,      # 🔥 Показывать каждый N-й кадр
                                  config_mode="balanced"    # 🔥 Конфигурация масок
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
    print(f"Mask Config: {config_mode}")
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
    builder = None
    if build_global_map:
        builder = LocalMapBuilder(
            pixels_per_meter=int(ppm / 1.47),      # Синхронизировано с PPM
            initial_size=5000,        # Начальный холст
            blend_decay=0.02,         # Уменьшаем затухание для более плотной карты
            use_distance_weighting=True,
            scale_factor=1        # Увеличиваем разрешение карты в 2 раза
        )

    use_telemetry = telemetry_path is not None
    vo = VisualOdometryIPM(
        use_telemetry=use_telemetry,
        debug_mode=debug_mode,
        config_mode=config_mode
    )

    sliding_frames = deque(maxlen=6)
    sliding_bevs = deque(maxlen=6)

    def build_sliding_strip(frames, cell_size=(256, 144)):
        visible = len(frames)
        max_cells = 5
        out_h = cell_size[1]
        out_w = cell_size[0] * max_cells
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        for idx in range(max_cells):
            x1 = idx * cell_size[0]
            x2 = x1 + cell_size[0]
            if idx < visible:
                frame_img, frame_idx = frames[idx]
                resized = cv2.resize(frame_img, cell_size, interpolation=cv2.INTER_AREA)
                cv2.rectangle(resized, (0, 0), (cell_size[0], 24), (0, 0, 0), -1)
                label = f"Frame {frame_idx}"
                if idx == visible - 1:
                    label = f"Current {frame_idx}"
                cv2.putText(resized, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                if idx == visible - 1:
                    cv2.rectangle(resized, (0, 0), (cell_size[0] - 1, cell_size[1] - 1), (0, 255, 0), 2)
                out[:, x1:x2] = resized
            else:
                out[:, x1:x2] = (35, 35, 35)

        return out

    def build_window_map(frames, max_display_size=(960, 800)):
        valid_frames = [(bev_img, pose, frame_idx) for bev_img, pose, frame_idx in frames if pose is not None]
        if not valid_frames:
            return np.zeros((max_display_size[1], max_display_size[0], 3), dtype=np.uint8)

        anchor_pose = valid_frames[-1][1]
        cos_a = np.cos(-anchor_pose[2])
        sin_a = np.sin(-anchor_pose[2])

        preview_builder = LocalMapBuilder(
            pixels_per_meter=int(ppm / 1.47),
            initial_size=2500,
            blend_decay=0.02,
            use_distance_weighting=True,
            scale_factor=1
        )

        current_frame_idx = valid_frames[-1][2]
        for bev_img, pose, frame_idx in valid_frames:
            dx = pose[0] - anchor_pose[0]
            dy = pose[1] - anchor_pose[1]
            local_x = dx * cos_a - dy * sin_a
            local_y = dx * sin_a + dy * cos_a
            local_theta = pose[2] - anchor_pose[2]
            local_pose = np.array([local_x, local_y, local_theta], dtype=np.float32)
            preview_builder.add_frame(
                bev_img,
                local_pose,
                frame_idx=frame_idx,
                mask_car=(frame_idx != current_frame_idx)
            )

        full_map = preview_builder.map
        out_w, out_h = max_display_size
        
        car_out_x = out_w // 2
        car_out_y = int(out_h * 0.50)
        
        car_map_x = preview_builder.center_x
        car_map_y = preview_builder.center_y
        
        start_x = car_map_x - car_out_x
        start_y = car_map_y - car_out_y
        
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        map_h, map_w = full_map.shape[:2]
        
        src_x1 = max(0, start_x)
        src_y1 = max(0, start_y)
        src_x2 = min(map_w, start_x + out_w)
        src_y2 = min(map_h, start_y + out_h)
        
        dst_x1 = max(0, -start_x)
        dst_y1 = max(0, -start_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = full_map[src_y1:src_y2, src_x1:src_x2]

        # Отрисовка текущей машины абсолютно четко и непрозрачно
        current_bev = valid_frames[-1][0]
        bev_h, bev_w = current_bev.shape[:2]
        car_center_x_bev = bev_w // 2
        car_center_y_bev = int(bev_h * 0.85)
        
        radius = 120
        # Вырезаем область машины из current_bev, учитывая границы
        y1_bev = max(0, car_center_y_bev - radius)
        y2_bev = min(bev_h, car_center_y_bev + radius)
        x1_bev = max(0, car_center_x_bev - radius)
        x2_bev = min(bev_w, car_center_x_bev + radius)
        
        car_patch = current_bev[y1_bev:y2_bev, x1_bev:x2_bev]
        
        # Смещение относительно идеального центра, если вышли за границы BEV
        dy_off = y1_bev - (car_center_y_bev - radius)
        dx_off = x1_bev - (car_center_x_bev - radius)
        
        # Вставляем на canvas (где центр машины - car_out_x, car_out_y)
        y1_canv = car_out_y - radius + dy_off
        y2_canv = y1_canv + (y2_bev - y1_bev)
        x1_canv = car_out_x - radius + dx_off
        x2_canv = x1_canv + (x2_bev - x1_bev)
        
        # Создаем маску круга и обрезаем ее под размер патча
        mask_full = np.zeros((radius*2, radius*2), dtype=np.uint8)
        cv2.circle(mask_full, (radius, radius), radius, 255, -1)
        mask = mask_full[dy_off : dy_off + (y2_bev - y1_bev), dx_off : dx_off + (x2_bev - x1_bev)]
        
        # Безопасная вставка на canvas
        if y1_canv >= 0 and y2_canv <= out_h and x1_canv >= 0 and x2_canv <= out_w:
            for c in range(3):
                canvas_roi = canvas[y1_canv:y2_canv, x1_canv:x2_canv, c]
                patch_roi = (car_patch[:, :, c].astype(np.float32) * 1.2).clip(0, 255).astype(np.uint8)
                canvas[y1_canv:y2_canv, x1_canv:x2_canv, c] = np.where(mask == 255, patch_roi, canvas_roi)

        return canvas

    telemetry_data = load_telemetry(telemetry_path) if telemetry_path else None

    bevs = []
    window_map_writer = None
    if save_window_map_video:
        if window_map_video_path is None:
            window_map_video_path = f"{os.path.splitext(video_path)[0]}_window_map.mp4"
        window_map_writer = cv2.VideoWriter(
            window_map_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (960, 800)
        )

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
        cv2.namedWindow("Sliding Window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Window Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", 1200, 400)
        cv2.resizeWindow("Sliding Window", 1280, 180)
        cv2.resizeWindow("Window Map", 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
        frame_count += 1

        # 1. IPM трансформация
        bev = ipm.transform(frame)
        bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if len(bev.shape) == 3 else bev.copy()

        # Сохраняем текущий кадр для скользящего окна
        sliding_frames.append((frame.copy(), frame_count))

        # Заготовка для локальной карты окна
        sliding_bevs.append((bev.copy(), None, frame_count))

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

        # 5. Уточнение по карте (только при наличии глобального билдерa)
        if builder is not None:
            refined_pose, ref_val = builder.refine_pose_against_map(bev, pose, search_range=10)
            if ref_val > 0.85:  # Еще немного повысим порог
                # Проверка на физическую правдоподобность (не прыгаем больше чем на 1 метр за кадр)
                dist = np.linalg.norm(refined_pose[:2] - pose[:2])
                if dist < 80:  # ~0.8-0.9 метра при ppm=92
                    alpha = 0.15
                    pose = (1.0 - alpha) * pose + alpha * refined_pose

        sliding_bevs[-1] = (sliding_bevs[-1][0], pose.copy(), frame_count)
        if builder is not None:
            builder.add_frame(bev, pose, frame_idx=frame_count)

        # 6. Отладочная визуализация
        if debug_mode and show_debug_windows and debug_data is not None:
            debug_frame_count += 1
            if debug_frame_count % debug_skip_frames == 0:
                match_img, kp1_img, kp2_img = debug_data

                bev_display = bev.copy()
                bev_display[car_mask == 0] = [0, 0, 255]
                cv2.putText(bev_display, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("BEV + Mask", bev_display)
                cv2.imshow("Matches", match_img)
                cv2.imshow("Keypoints Frame 1", kp1_img)
                cv2.imshow("Keypoints Frame 2", kp2_img)
                cv2.imshow("Sliding Window", build_sliding_strip(list(sliding_frames)))
                cv2.imshow("Window Map", build_window_map(list(sliding_bevs)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ Interrupted by user")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

        if window_map_writer is not None:
            window_frame = build_window_map(list(sliding_bevs))
            window_map_writer.write(window_frame)

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
    print(f"  Map-to-Frame Refinements: {builder.refinement_count if builder is not None else 0}")
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

    if window_map_writer is not None:
        window_map_writer.release()
        print(f"✓ Window map video saved to: {window_map_video_path}")

    if save_window_map_image:
        window_map_image_path = window_map_image_path or f"{os.path.splitext(video_path)[0]}_window_map.png"
        final_window_map = build_window_map(list(sliding_bevs))
        cv2.imwrite(window_map_image_path, final_window_map)
        print(f"✓ Window map saved to: {window_map_image_path}")

    if builder is not None:
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
    else:
        print(f"\n📈 FINAL STATISTICS:")
        print(f"  Frames processed: {frame_count - start_frame}")
        print(f"  Global map building disabled.")


if __name__ == "__main__":
    VIDEO_PATH = "data/race_1.mp4"
    TELEMETRY_PATH = "data/telemetry.csv"
    
    # START_FRAME = 1038
    START_FRAME = 0

    END_FRAME = None  # None для обработки всего видео, или укажите конкретный кадр для остановки
    
    # CONFIG: balanced, sides_short, sides_short_v2, sides_short_v3
    CONFIG = "custom"
    
    build_map_from_video_4points(
        VIDEO_PATH, 
        telemetry_path=TELEMETRY_PATH,
        start_frame=START_FRAME, 
        end_frame=END_FRAME,
        save_bev_video=False,
        debug_mode=True,       # 🔥 Включить отладку
        show_debug_windows=True,# 🔥 Показывать окна
        debug_skip_frames=1,    # 🔥 Каждый N-й кадр
        config_mode=CONFIG,     # 🔥 Конфигурация масок
        save_window_map_video=True,
        window_map_video_path="window_map.mp4",
    )