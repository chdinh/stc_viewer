
import numpy as np
import trimesh
import pyrr 

class BrainInteraction:
    """
    Handles user interaction with the brain mesh, specifically calculating
    the brain region label under the mouse cursor using raycasting.
    """
    
    def __init__(self, vertices, faces, labels, region_names):
        """
        Initialize the interaction handler.
        
        Args:
            vertices (np.ndarray): (N, 3) float32 array of vertex positions.
            faces (np.ndarray): (M, 3) uint32 array of face indices.
            labels (np.ndarray): (N,) int array of atlas labels per vertex.
            region_names (list): List of region names corresponding to label indices.
        """
        # Create a trimesh object for raycasting
        # process=False avoids re-ordering or merging vertices, keeping 1:1 mapping with our buffers
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # If pyembree is installed, trimesh uses it automatically.
        # We can construct a ray_mesh_intersector for speed if needed, 
        # but the default mesh.ray.intersects_id is fine for mouse picking.
        
        self.labels = labels
        self.region_names = region_names

    def get_region_at_mouse(self, mouse_norm_x, mouse_norm_y, view_matrix, projection_matrix):
        """
        Find the brain region under the given normalized mouse coordinates.
        
        Args:
            mouse_norm_x (float): Mouse X in NDC [-1, 1].
            mouse_norm_y (float): Mouse Y in NDC [-1, 1].
            view_matrix (np.ndarray): 4x4 View Matrix.
            projection_matrix (np.ndarray): 4x4 Projection Matrix.
            
        Returns:
            str: Name of the region, or None if no intersection.
        """
        # 1. Unproject mouse to World Ray
        # NDC -> Clip -> View -> World
        
        # Inverse matrices
        try:
            inv_proj = np.linalg.inv(projection_matrix)
            inv_view = np.linalg.inv(view_matrix)
        except np.linalg.LinAlgError:
            return None
            
        # Ray Start (Near Plane) and End (Far Plane) in NDC
        ray_start_ndc = np.array([mouse_norm_x, mouse_norm_y, -1.0, 1.0], dtype=np.float32)
        ray_end_ndc = np.array([mouse_norm_x, mouse_norm_y, 1.0, 1.0], dtype=np.float32)
        
        # To View Space
        ray_start_view = np.dot(inv_proj, ray_start_ndc)
        ray_start_view /= ray_start_view[3]
        
        ray_end_view = np.dot(inv_proj, ray_end_ndc)
        ray_end_view /= ray_end_view[3]
        
        # To World Space
        ray_start_world = np.dot(inv_view, ray_start_view)
        ray_end_world = np.dot(inv_view, ray_end_view)
        
        origin = ray_start_world[:3]
        direction = ray_end_world[:3] - origin
        direction = direction / np.linalg.norm(direction)
        
        # 2. Raycast against Mesh
        # intersects_location returns locations, index_ray, index_tri
        # We only have 1 ray.
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(index_tri) == 0:
            return None
            
        # 3. Find closest hit
        # locations is (N_hits, 3). We want the closest one to origin.
        # But intersects_location might not be sorted.
        # Calculate distances
        diffs = locations - origin
        dists = np.linalg.norm(diffs, axis=1)
        closest_idx = np.argmin(dists)
        
        tri_idx = index_tri[closest_idx]
        
        # 4. Get Label from Triangle
        # A triangle has 3 vertices. We can take the majority label or just the first one.
        # Faces array: (M, 3)
        face_verts = self.mesh.faces[tri_idx] # [v0, v1, v2] indices
        
        # Look up labels
        # labels is (N_vertices,)
        l0 = self.labels[face_verts[0]]
        
        # Just pick the first one for simplicity
        label_id = l0
        
        # 5. Get Name
        if 0 <= label_id < len(self.region_names):
            return self.region_names[label_id]
        
        return "Unknown"
