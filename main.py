import os
import cv2
import numpy as np
from tqdm import tqdm

from ipm import IPM_4Points
from map_builder import LocalMapBuilder
from visual_odometry_ipm import VisualOdometryIPM


def load_telemetry(video_path):
    """Заглушка для телеметрии"""
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
    
    # === IPM НАСТРОЙКИ ===
    src_pts = np.float32([[143, 511], [520, 507], [411, 428], [647, 423]])
    bev_width = 600
    bev_height = 400
    
    car_size = (1.47, 2.80)
    scale_koef = 0.0583333
    rect_size = (bev_width * scale_koef, bev_height * scale_koef * car_size[1] / car_size[0])
    rect_left_up_koef = (0.5, 0.7)
    rect_left_up_pos = (bev_width * rect_left_up_koef[0], bev_height * rect_left_up_koef[1])

    dst_pts = np.float32([
        [rect_left_up_pos[0], rect_left_up_pos[1] + rect_size[1]],                   
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1] + rect_size[1]],   
        [rect_left_up_pos[0], rect_left_up_pos[1]],                                  
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1]],                   
    ])

    ipm = IPM_4Points(src_pts, dst_pts, output_size=(bev_width, bev_height))
    builder = LocalMapBuilder(
        pixels_per_meter=33,      # Попробуйте 30-40 для калибровки
        initial_size=5000,        # Увеличил запас
        blend_decay=0.05,
        use_distance_weighting=True
    )
    
    use_telemetry = telemetry_path is not None
    vo = VisualOdometryIPM(
        nfeatures=2000,
        use_telemetry=use_telemetry,
        debug_mode=debug_mode
    )

    telemetry_data = load_telemetry(video_path) if telemetry_path else None

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
        
        # 2. Маска машины (для визуализации)
        car_mask = vo._get_car_mask(bev.shape)
        bev_masked = cv2.bitwise_and(bev, bev, mask=car_mask)
        if debug_mode:
            distance_weights = vo._get_car_mask(bev.shape)  # Для примера
            weight_viz = (distance_weights * 255).astype(np.uint8)
            weight_viz = cv2.applyColorMap(weight_viz, cv2.COLORMAP_JET)
            cv2.imshow("Distance Weights", weight_viz)
        
        # 3. Телеметрия
        current_telemetry = None
        if telemetry_data and frame_count - start_frame < len(telemetry_data):
            t_idx = frame_count - start_frame
            current_telemetry = telemetry_data[t_idx]
            current_telemetry['dt'] = dt

        # 4. Визуальная одометрия
        pose, debug_data = vo.update(bev, telemetry=current_telemetry)
        
        # 5. Отладочная визуализация
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
                cv2.imshow("Keypoints Frame 1", kp1_img)
                cv2.imshow("Keypoints Frame 2", kp2_img)
                cv2.imshow("Matches", match_img)
                cv2.waitKey(0)
                
                # Ждем 1мс, если нажали 'q' - выходим
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠ Interrupted by user")
                    break
                elif key == ord(' '):
                    # Пробел - пауза
                    cv2.waitKey(0)
        
        # 6. Добавляем на карту
        builder.add_frame(bev, pose)
        
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

    builder.save_map(output_map_path)
    print(f"✓ Map saved to: {output_map_path}")

    final_map = builder.get_map()
    
    if len(builder.poses) > 1:
        poses = np.array(builder.poses)
        for i in range(1, len(poses)):
            pt1 = (int(poses[i-1][0]), int(poses[i-1][1]))
            pt2 = (int(poses[i][0]), int(poses[i][1]))
            cv2.line(final_map, pt1, pt2, (255, 0, 0), 2)

    cv2.imshow("Local Map", final_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n📈 FINAL STATISTICS:")
    print(f"  Frames processed: {frame_count - start_frame}")
    print(f"  Map size: {final_map.shape[1]}x{final_map.shape[0]} pixels")
    print(f"  Trajectory points: {len(builder.poses)}")


if __name__ == "__main__":
    VIDEO_PATH = "data/Q2 354.MP4"
    TELEMETRY_PATH = None
    
    START_FRAME = 1038
    END_FRAME = START_FRAME + 20
    
    build_map_from_video_4points(
        VIDEO_PATH, 
        telemetry_path=TELEMETRY_PATH,
        start_frame=START_FRAME, 
        end_frame=END_FRAME,
        save_bev_video=False,
        debug_mode=True,           # 🔥 Включить отладку
        show_debug_windows=True,   # 🔥 Показывать окна
        debug_skip_frames=1,       # 🔥 Каждый 3-й кадр
    )