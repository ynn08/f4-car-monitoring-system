import os
import cv2
import numpy as np
from tqdm import tqdm

from ipm import IPM_4Points
from map_builder import LocalMapBuilder
from visual_odometry_ipm import VisualOdometryIPM


def build_map_from_video_4points(video_path, 
                                  output_map_path="local_map.png",
                                  start_frame=0,
                                  end_frame=None,
                                  save_bev_video=False):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video file!")
        return
    cap.release()
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        print(f"Error: start_frame ({start_frame}) >= end_frame ({end_frame})")
        return
    process_frames = end_frame - start_frame
    print(f"\nVideo: {video_path}")
    print(f"FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"Processing frames: {start_frame} - {end_frame} ({process_frames} frames)")
    
    # === ПРЕДОПРЕДЕЛЁННЫЕ ТОЧКИ ===
    x_delta, y_delta = 100, 150
    src_pts = np.float32([[85, 345], [350, 340], [275, 285], [430, 280]]) + [x_delta, y_delta]
    src_pts = np.float32([[143, 511], [520, 507], [411, 428], [647, 423]])
    # bev_width = 1080
    # bev_height = 720
    bev_width = 600
    bev_height = 400

    car_size = (1.47, 2.80)
    scale_koef = 0.0583333
    rect_size = (bev_width * scale_koef, bev_height * scale_koef * car_size[1] / car_size[0])
    rect_left_up_koef = (0.5, 0.7)
    rect_left_up_pos = (bev_width * rect_left_up_koef[0], bev_height * rect_left_up_koef[1])

    dst_pts = np.float32([
        [rect_left_up_pos[0], rect_left_up_pos[1] + rect_size[1]],                   # Левый верхний (src дальний левый)
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1] + rect_size[1]] ,   # Правый верхний (src дальний правый)
        [rect_left_up_pos[0], rect_left_up_pos[1]],                                  # Левый нижний (src ближний левый)
        [rect_left_up_pos[0] + rect_size[0], rect_left_up_pos[1]],                   # Правый нижний (src ближний правый)
    ])

    ipm = IPM_4Points(src_pts, dst_pts, output_size=(bev_width, bev_height))
    builder = LocalMapBuilder(pixels_per_meter=50, initial_size=4000)
    vo = VisualOdometryIPM(use_sift=False)

    bevs = []
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    processed_count = 0
    pbar = tqdm(total=process_frames, desc="Building map")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= end_frame:
            break
        frame_count += 1
        
        # 1. IPM трансформация
        bev = ipm.transform(frame)
        bevs.append(bev)
    
        # 2. Визуальная одометрия
        pose = vo.update(bev)
        
        # 3. Добавляем на карту
        builder.add_frame(bev, pose)
        processed_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    if save_bev_video:
        ipm_video = cv2.VideoWriter(
            f"{video_path[:video_path.rfind('.')]}_ipm_formatted.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (bev_width, bev_height)
        )
        for bev in bevs:
            ipm_video.write(bev)
        ipm_video.release()


    # ========================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================================
    builder.save_map(output_map_path)
    print(f"\n✓ Map saved to: {output_map_path}")

    final_map = builder.get_map()
    cv2.imshow("Local Map", final_map)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"\nStatistics:")
    print(f"  Frame range: {start_frame} - {end_frame}")
    print(f"  Frames processed: {processed_count}")
    print(f"  Map size: {final_map.shape[1]}x{final_map.shape[0]} pixels")
    print(f"  Trajectory points: {len(builder.poses)}")


if __name__ == "__main__":
    VIDEO_PATH = "data/Q2 354.MP4"
    START_FRAME = 3038
    END_FRAME = 4039
    END_FRAME = 3060
    # START_FRAME = 0
    # END_FRAME = None
    
    build_map_from_video_4points(
        VIDEO_PATH, 
        start_frame=START_FRAME, 
        end_frame=END_FRAME,
        save_bev_video=False,
    )