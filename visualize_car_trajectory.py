import json
import csv
import cv2
import numpy as np
import os

def visualize_trajectory(json_path, csv_path, output_path):
    # Load track points
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return
    with open(json_path, 'r') as f:
        track_data = json.load(f)
        
    all_track_pts = []
    for cat in track_data:
        points = track_data[cat]
        # Mirror Y axis for each point in track_data
        mirrored_points = [[p[0], -p[1]] for p in points]
        track_data[cat] = mirrored_points
        all_track_pts.extend(mirrored_points)
    all_track_pts = np.array(all_track_pts)

    # Load car trajectory
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    car_pts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(filter(lambda l: not l.startswith('#'), f))
        for row in reader:
            # Mirror Z axis (mapped to Y in 2D)
            car_pts.append([float(row['position.x']), -float(row['position.z'])])
    car_pts = np.array(car_pts)

    if len(all_track_pts) == 0 or len(car_pts) == 0:
        print("No points found in data files.")
        return

    # Combined bounding box for normalization
    all_pts = np.vstack([all_track_pts, car_pts])
    min_coords = np.min(all_pts, axis=0)
    max_coords = np.max(all_pts, axis=0)
    
    # High resolution
    width = 4000
    height = 4000
    margin = 100
    
    range_x = max_coords[0] - min_coords[0]
    range_z = max_coords[1] - min_coords[1]
    scale = (width - 2 * margin) / max(range_x, range_z)
    
    # Base image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Colors (BGR)
    colors = {
        "track": (200, 200, 200),
        "kerbs": (0, 0, 255),
        "runoff": (0, 255, 0),
        "car": (255, 255, 0) # Cyan-ish
    }
    
    # Draw track points
    for cat, points in track_data.items():
        color = colors.get(cat, (255, 255, 255))
        for p in points:
            x = int((p[0] - min_coords[0]) * scale) + margin
            y = int((max_coords[1] - p[1]) * scale) + margin
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), 2, color, -1)
                
    # Draw car trajectory as a line
    pixel_pts = []
    for p in car_pts:
        x = int((p[0] - min_coords[0]) * scale) + margin
        y = int((max_coords[1] - p[1]) * scale) + margin
        pixel_pts.append([x, y])
    
    pixel_pts = np.array(pixel_pts, dtype=np.int32)
    cv2.polylines(img, [pixel_pts], False, colors["car"], 3)
    
    # Legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (20, 10), (600, 280), (30, 30, 30), -1)
    cv2.putText(img, "Track (Gray)", (50, 60), font, 1.5, colors["track"], 4)
    cv2.putText(img, "Kerbs (Red)", (50, 130), font, 1.5, colors["kerbs"], 4)
    cv2.putText(img, "Runoff (Green)", (50, 200), font, 1.5, colors["runoff"], 4)
    cv2.putText(img, "Car Path (Cyan)", (50, 270), font, 1.5, colors["car"], 4)
        
    cv2.imwrite(output_path, img)
    print(f"Trajectory overlay saved to {output_path}")

if __name__ == "__main__":
    visualize_trajectory("/home/yn/Documents/Projects/f4-car-monitoring-system/data/track_points.json",
                         "/home/yn/Documents/Projects/f4-car-monitoring-system/data/output.csv",
                         "/home/yn/Documents/Projects/f4-car-monitoring-system/data/trajectory_overlay.png")
