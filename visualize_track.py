import json
import cv2
import numpy as np

def visualize_point_cloud(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    all_points = []
    for cat_pts in data.values():
        all_points.extend(cat_pts)
    
    if not all_points:
        print("No points to visualize!")
        return
        
    all_points = np.array(all_points)
    
    # Normalize points to image coordinates
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    
    # Very high resolution for pixel-level smoothness
    width = 4000
    height = 4000
    
    margin = 100
    range_x = max_coords[0] - min_coords[0]
    range_z = max_coords[1] - min_coords[1]
    scale = (width - 2 * margin) / max(range_x, range_z)
    
    # Background - dark for better contrast
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define colors (BGR) - Vibrant colors for better contrast
    colors = {
        "track": (200, 200, 200),   # Bright Gray/Silver
        "kerbs": (0, 0, 255),       # Pure Red
        "runoff": (0, 255, 0)       # Pure Green
    }
    
    for cat, points in data.items():
        color = colors.get(cat, (255, 255, 255))
        radius = 2 # Increased radius for better visibility
        for p in points:
            x = int((p[0] - min_coords[0]) * scale) + margin
            y = int((max_coords[1] - p[1]) * scale) + margin
            
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), radius, color, -1)
            
    # Add legend with larger font and background for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (20, 10), (500, 220), (30, 30, 30), -1)
    cv2.putText(img, "Track (Roads)", (50, 60), font, 1.5, (200, 200, 200), 4)
    cv2.putText(img, "Kerbs (Red)", (50, 130), font, 1.5, (0, 0, 255), 4)
    cv2.putText(img, "Runoff (Green)", (50, 200), font, 1.5, (0, 255, 0), 4)
        
    cv2.imwrite(output_path, img)
    print(f"High-resolution visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_point_cloud("/home/yn/Documents/Projects/f4-car-monitoring-system/track_points.json", 
                          "/home/yn/Documents/Projects/f4-car-monitoring-system/extracted_track_view.png")
