import wgpu
import wgpu.backends.wgpu_native
from rendercanvas.offscreen import RenderCanvas
from brain_renderer import BrainRenderer
from camera import Camera
from atlas_loader import load_brain_data
import numpy as np
import time
from PIL import Image

def run_test():
    print("Initializing Offscreen Renderer...")
    # Use Offscreen Canvas
    canvas = RenderCanvas(size=(800, 600))
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    
    # Load Data
    print("Loading Brain Data...")
    brain_data = load_brain_data()
    
    # Setup Components
    renderer = BrainRenderer(device, brain_data, canvas)
    # Trace renderer not needed for this visual check
    
    # Setup Camera
    camera = Camera(canvas)
    
    # Determine Format
    # For offscreen, we need a texture to draw into
    render_format = wgpu.TextureFormat.bgra8unorm
    texture = device.create_texture(
        size=(800, 600, 1),
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        format=render_format,
    )
    
    def render_and_save(filename, camera_pos_override=None):
        current_view = texture.create_view()
        
        # Calculate View Matrix manually if overriding pos
        if camera_pos_override is not None:
            # Look at origin
            eye = np.array(camera_pos_override, dtype=np.float32)
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Create View Matrix (LookAt)
            z = eye - target
            z = z / np.linalg.norm(z)
            x = np.cross(up, z)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            
            view_mat = np.eye(4, dtype=np.float32)
            view_mat[:3, 0] = x
            view_mat[:3, 1] = y
            view_mat[:3, 2] = z
            view_mat[3, :3] = -np.dot(np.array([x, y, z]), eye)
            
            # Transpose for Column-Major (if needed by pyrr/wgpu expectations? pyrr usually handles it)
            # Actually brain_renderer uses pyrr.matrix44. 
            # Let's just use existing Camera class if possible, or manual.
            # BrainRenderer expects `view_matrix`.
            # Let's try to just set the view matrix directly.
        else:
             view_mat = camera.get_view_matrix()
             eye = None # Should fallback to internal default or passed pos

        aspect = 800/600
        
        # Draw
        # We need to pass camera_pos specifically because brain_renderer uses it for calculating View Dir
        renderer.draw(current_view, aspect, view_mat, camera_pos=camera_pos_override)
        
        # Readback
        data = device.queue.read_texture(
             {"texture": texture, "origin": (0, 0, 0)},
             {"offset": 0, "bytes_per_row": 800 * 4, "rows_per_image": 600},
             texture.size,
        )
        img = Image.frombytes("RGBA", (800, 600), bytes(data))
        img.save(filename)
        print(f"Saved {filename}")

    # 1. Top View (Standard)
    # Camera at (0, 0, 5) roughly?
    # Actually standard camera init is (0, 0, 4) or similar?
    # Let's enforce a Top View position: Z-axis is usually Top/Bottom in neurological space?
    # Wait, in graphics, usually Y is Up, Z is Forward/Back.
    # In `main.py`, camera is initialized.
    # Let's try standard Front/Top view.
    # User said "From Top". Usually this means looking down Y or Z.
    
    # Let's try (0, 5, 0.1) -> Looking down Y?
    # Or (0, 0, 5) -> Looking down Z?
    # Let's assume standard view is "Top".
    render_and_save("view_top.png", camera_pos_override=[0.0, 0.0, 5.0])
    
    # 2. Bottom View
    # Rotate 180 degrees around X or Y?
    # Determine "Bottom" by flipping position.
    render_and_save("view_bottom.png", camera_pos_override=[0.0, 0.0, -5.0])
    
    # 3. Side View (for completeness)
    render_and_save("view_side.png", camera_pos_override=[5.0, 0.0, 0.0])

    # 4. Reference Match View (Quarter View)
    # The reference image is viewed from roughly Top-Left-Front.
    # Vector: [-3, -4, 3] normalized * 5 ?
    render_and_save("view_reference_match.png", camera_pos_override=[-3.0, -3.0, 3.0])

if __name__ == "__main__":
    run_test()
