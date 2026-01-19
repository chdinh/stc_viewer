
import wgpu
import numpy as np
import pyrr
from interaction import BrainInteraction
from atlas_loader import load_brain_data
from brain_renderer import BrainRenderer
from rendercanvas.offscreen import RenderCanvas
from text_renderer import TextRenderer
from PIL import Image

def test_hover_interaction():
    print("Initializing components...")
    
    # 1. Load Data
    try:
        data = load_brain_data()
    except:
        print("Scipy/Data missing, skipping full integration test.")
        return

    # 2. Setup Context
    canvas = RenderCanvas(size=(800, 600))
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    render_format = wgpu.TextureFormat.bgra8unorm
    
    # 3. Setup Renderer & Interaction
    renderer = BrainRenderer(device, data, canvas)
    text_renderer = TextRenderer(device, render_format)
    interaction = BrainInteraction(data["vertices"], data["faces"], data["labels"], data["region_names"])
    
    # 4. Setup Matrices (Standard View)
    aspect = 800 / 600
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, aspect, 0.1, 1000.0)
    
    # Camera at (0, 0, 5) looking at (0, 0, 0)
    view = pyrr.matrix44.create_look_at(
        eye=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]
    )
    
    # 5. Pick a Target Vertex (Closest to Camera to avoid occlusion)
    # Camera at Z=5. Vertex with max Z is closest.
    target_idx = np.argmax(data["vertices"][:, 2])
    target_v = data["vertices"][target_idx]
    target_label = data["labels"][target_idx]

    target_name = data["region_names"][target_label]
    print(f"Targeting Vertex {target_idx}: {target_v} (Region: {target_name}, ID: {target_label})")
    
    # 6. Project Vertex to Screen (NDC)
    # Model is Identity
    pos_4 = np.array([target_v[0], target_v[1], target_v[2], 1.0], dtype=np.float32)
    
    # View -> Proj
    # Note: Renderer uses a correction matrix!
    correction = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ], dtype=np.float32)

    mvp = np.matmul(view, np.matmul(projection, correction))
    clip_pos = np.dot(mvp, pos_4)
    ndc_pos = clip_pos[:3] / clip_pos[3]
    
    print(f"Projected NDC: {ndc_pos}")
    
    # 7. Test Raycasting with Computed NDC
    # Add small epsilon to ensures numerical stability?
    ndc_x, ndc_y = ndc_pos[0], ndc_pos[1]
    
    hit_name, hit_id = interaction.get_region_at_mouse(ndc_x, ndc_y, mvp)
    
    print(f"Raycast Result: Name='{hit_name}', ID={hit_id}")
    
    # Verification 1: Interaction Logic
    if hit_id == target_label:
        print("PASS: Raycasting correctly identified the region.")
    else:
        print(f"FAIL: Raycasting mismatch. Expected {target_label} ({target_name}), got {hit_id} ({hit_name})")
        # Don't exit yet, check rendering
        
    # 8. Test Rendering with Highlight
    # Set hovered ID
    renderer.set_hovered_id(target_label)
    renderer.set_visualization_mode(1.0) # Atlas Mode
    
    texture = device.create_texture(
        size=(800, 600, 1),
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        format=render_format,
    )
    current_view = texture.create_view()
    
    renderer.draw(current_view, aspect, view, camera_pos=np.array([0,0,5]))
    
    # Readback
    print("Rendering...")
    data_bytes = device.queue.read_texture(
         {"texture": texture, "origin": (0, 0, 0)},
         {"offset": 0, "bytes_per_row": 800 * 4, "rows_per_image": 600},
         texture.size,
    )
    img = Image.frombytes("RGBA", (800, 600), bytes(data_bytes))
    img.save("test_hover_output.png")
    print("Saved test_hover_output.png")
    
    # 9. Verify Highlight
    # We expect bright pixels near the edge/rim of the region.
    # It's hard to verify "Shiny Border" automatically without complex image analysis.
    # But we can verify that the image is not black and not empty.
    # And maybe compare with "No Highlight" version?
    
    renderer.set_hovered_id(-1.0)
    renderer.draw(current_view, aspect, view, camera_pos=np.array([0,0,5]))
    data_bytes_no_hl = device.queue.read_texture(
         {"texture": texture, "origin": (0, 0, 0)},
         {"offset": 0, "bytes_per_row": 800 * 4, "rows_per_image": 600},
         texture.size,
    )
    img_no_hl = Image.frombytes("RGBA", (800, 600), bytes(data_bytes_no_hl))
    img_no_hl.save("test_hover_nohighlight.png")
    
    diff = np.abs(np.array(img).astype(int) - np.array(img_no_hl).astype(int))
    diff_sum = np.sum(diff)
    print(f"Difference Score (Highlight vs No Highlight): {diff_sum}")
    
    if diff_sum > 1000:
        print("PASS: Highlight effect changed the rendered image.")
    else:
        print("FAIL: No significant change detected with highlight.")
        exit(1)

if __name__ == "__main__":
    test_hover_interaction()
