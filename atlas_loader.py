import numpy as np
from nilearn import datasets
import trimesh
import os
import subprocess
import nibabel as nib

def ensure_destrieux_downloaded():
    """Manual download using curl to bypass Python SSL issues."""
    data_dir = os.path.expanduser("~/nilearn_data/destrieux_surface")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    files = {
        "lh.aparc.a2009s.annot": "https://www.nitrc.org/frs/download.php/9343/lh.aparc.a2009s.annot",
        "rh.aparc.a2009s.annot": "https://www.nitrc.org/frs/download.php/9342/rh.aparc.a2009s.annot"
    }
    
    for filename, url in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size < 50 * 1024 or size > 500 * 1024:
                os.remove(filepath)

        if not os.path.exists(filepath):
            try:
                subprocess.run(["curl", "-k", "-L", "-o", filepath, url], check=True)
            except subprocess.CalledProcessError:
                pass # Already printed in earlier attempts

def _load_hemisphere(mesh_path, annot_path, label_colors):
    print(f"Loading mesh from {mesh_path}...")
    if mesh_path.endswith('.gii') or mesh_path.endswith('.gii.gz'):
        gii = nib.load(mesh_path)
        coords = gii.darrays[0].data
        faces = gii.darrays[1].data
    else:
        coords, faces = nib.freesurfer.read_geometry(mesh_path)
        
    faces = faces.astype(np.uint32)
    coords = coords.astype(np.float32)
    
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    
    # -------------------------------------------------------------
    # SIMULATION: Temporal Source Dynamics
    # -------------------------------------------------------------
    n_frames = 200
    time_series = np.zeros((len(labels), n_frames), dtype=np.float32)
    
    # Base Noise
    noise = np.random.normal(0.0, 0.05, (len(labels), n_frames))
    time_series += noise
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    active_clusters = np.random.choice(unique_labels, size=5, replace=False)
    
    print(f"Simulating MNE Time Traces for Clusters: {active_clusters}")
    
    t = np.linspace(0, 4*np.pi, n_frames)
    raw_traces = [] # For Butterfly Plot
    
    for i, cluster_id in enumerate(active_clusters):
        mask = (labels == cluster_id)
        n_verts = np.sum(mask)
        
        # Pattern: Sine wave
        phase = i * (np.pi / 2.0)
        freq = 1.0 + (i * 0.5)
        
        signal = np.abs(np.sin(freq * t + phase))
        raw_traces.append(signal) # Store clean signal
        
        cluster_signal = np.tile(signal, (n_verts, 1))
        cluster_signal += np.random.normal(0, 0.1, cluster_signal.shape)
        
        time_series[mask] = cluster_signal * 0.8 + 0.2

    time_series = np.clip(time_series, 0.0, 1.0)
    
    # Pre-calculate Colors for Animation
    base_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    # Plasma Blue / Electric Light for Activations
    peak_color = np.array([0.1, 0.5, 1.0], dtype=np.float32) # Electric Blue
    diff = peak_color - base_color
    
    vertex_color_frames = base_color + (time_series[..., np.newaxis] * diff)
    vertex_color_frames = vertex_color_frames.astype(np.float32)
    
    # -------------------------------------------------------------
    # ATLAS COLORS
    # -------------------------------------------------------------
    max_id = np.max(labels)
    if max_id >= len(label_colors):
        needed = max_id + 1
        current = len(label_colors)
        extra = np.random.uniform(0.2, 0.9, (needed - current, 3))
        local_label_colors = np.vstack([label_colors, extra])
    else:
        local_label_colors = label_colors
        
    atlas_colors = local_label_colors[labels].astype(np.float32)

    # Initial color
    vertex_colors = vertex_color_frames[:, 0, :]
    
    mesh = trimesh.Trimesh(vertices=coords, faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    trimesh.repair.fix_normals(mesh)
    
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    normals = mesh.vertex_normals.astype(np.float32)
    
    curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 1.0)
    c_min, c_max = np.percentile(curvature, [2, 98])
    curvature = np.clip(curvature, c_min, c_max)
    curvature = (curvature - c_min) / (c_max - c_min)
    curvature = curvature.astype(np.float32)
    
    return vertices, faces, normals, vertex_colors, curvature, vertex_color_frames, atlas_colors, raw_traces

def load_brain_data():
    print("Fetching fsaverage5 surface...")
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    ensure_destrieux_downloaded()
    
    data_dir = os.path.expanduser("~/nilearn_data/destrieux_surface")
    left_annot_path = f"{data_dir}/lh.aparc.a2009s.annot"
    right_annot_path = f"{data_dir}/rh.aparc.a2009s.annot"
    
    np.random.seed(42)
    label_colors = np.random.uniform(0.2, 0.9, (200, 3))
    label_colors[0] = [0.5, 0.5, 0.5]
    
    # Load hemispheres
    v_l, f_l, n_l, c_l, curv_l, frames_l, atlas_l, traces_l = _load_hemisphere(
        fsaverage.pial_left, left_annot_path, label_colors
    )
    
    if hasattr(fsaverage, 'pial_right'):
        mesh_path_r = fsaverage.pial_right
    else:
        mesh_path_r = fsaverage.pial_left.replace("left", "right").replace("lh", "rh")
        
    v_r, f_r, n_r, c_r, curv_r, frames_r, atlas_r, traces_r = _load_hemisphere(
        mesh_path_r, right_annot_path, label_colors
    )
    
    print("Merging hemispheres...")
    vertices = np.vstack([v_l, v_r])
    normals = np.vstack([n_l, n_r])
    colors = np.vstack([c_l, c_r])
    curvature = np.concatenate([curv_l, curv_r])
    color_frames = np.concatenate([frames_l, frames_r], axis=0) # (N, T, 3)
    atlas_colors = np.vstack([atlas_l, atlas_r]) # (N, 3)
    
    # Combine traces (List of arrays)
    all_traces = traces_l + traces_r
    
    f_r_offset = f_r + len(v_l)
    faces = np.vstack([f_l, f_r_offset])
    
    print("Centering combined mesh...")
    centroid = np.mean(vertices, axis=0)
    extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    scale = 1.0 / np.max(extents)
    vertices -= centroid
    vertices *= scale
    
    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "colors": colors,
        "curvature": curvature,
        "color_frames": color_frames,
        "atlas_colors": atlas_colors,
        "traces": all_traces
    }

if __name__ == "__main__":
    load_brain_data()
