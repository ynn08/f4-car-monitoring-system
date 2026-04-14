import sys
import json
import numpy as np

def extract_track_points(obj_path):
    print(f"Reading {obj_path}...")
    vertices = []
    
    # Categories and their materials
    categories = {
        "track": ["roads"],
        "kerbs": ["cerb_paint", "cerb_new"],
        "runoff": ["carpet1"]
    }
    
    # Store vertex indices for each category
    category_vertex_indices = {cat: set() for cat in categories}
    
    current_material = None
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            cmd = parts[0]
            
            if cmd == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == 'usemtl':
                current_material = parts[1]
            elif cmd == 'f':
                # Check which category this material belongs to
                cat_found = None
                for cat, mats in categories.items():
                    if current_material in mats:
                        cat_found = cat
                        break
                
                if cat_found:
                    for p in parts[1:]:
                        idx = int(p.split('/')[0])
                        category_vertex_indices[cat_found].add(idx - 1)
                    
    print(f"Total vertices in OBJ: {len(vertices)}")
    
    result = {}
    for cat, indices in category_vertex_indices.items():
        points = []
        for idx in indices:
            v = vertices[idx]
            points.append([v[0], v[2]]) # X, Z
        result[cat] = points
        print(f"Category '{cat}': {len(points)} unique points")
        
    return result

def main():
    obj_path = "/home/yn/Documents/Projects/f4-car-monitoring-system/track.obj"
    output_path = "/home/yn/Documents/Projects/f4-car-monitoring-system/track_points.json"
    
    extracted_data = extract_track_points(obj_path)
    
    with open(output_path, 'w') as f:
        json.dump(extracted_data, f)
        
    print(f"Successfully extracted all vertices to {output_path}")

if __name__ == "__main__":
    main()
