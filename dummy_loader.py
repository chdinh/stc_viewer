import numpy as np
import trimesh

# Fallback loader for offline/test environments
def load_brain_data():
    print("Using dummy brain data (SSL failed)...")
    
    # Create a sphere as dummy brain
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    normals = mesh.vertex_normals.astype(np.float32)
    
    # Random colors
    colors = np.random.uniform(0.2, 0.9, (len(vertices), 3)).astype(np.float32)
    
    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "colors": colors
    }
