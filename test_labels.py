import wgpu
import wgpu.backends.wgpu_native
from rendercanvas.offscreen import RenderCanvas
from text_renderer import TextRenderer
import numpy as np
from PIL import Image

def test_label_rendering():
    print("Testing Label Rendering...")
    
    # 1. Setup Offscreen
    canvas = RenderCanvas(size=(800, 600))
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    
    # 2. Setup Renderer
    render_format = wgpu.TextureFormat.bgra8unorm
    renderer = TextRenderer(device, render_format)
    
    # 3. Create Target
    texture = device.create_texture(
        size=(800, 600, 1),
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        format=render_format,
    )
    view = texture.create_view()
    
    # 4. Set Text
    test_text = "Test Region Label"
    renderer.set_text(test_text)
    
    # 5. Render
    print("Rendering Frame...")
    renderer.draw(view)
    
    # 6. Readback
    data = device.queue.read_texture(
         {"texture": texture, "origin": (0, 0, 0)},
         {"offset": 0, "bytes_per_row": 800 * 4, "rows_per_image": 600},
         texture.size,
    )
    img = Image.frombytes("RGBA", (800, 600), bytes(data))
    img.save("test_label_output.png")
    print("Saved test_label_output.png")
    
    # 7. Analyze
    # Check if there are white pixels in the top-left area
    # The quad is mapped to [-1, 1] x [0.85, 1.0] in NDC
    # Screen is 800x600.
    # X: 0 to 400 (if quad covers half width? No, quad is -1 to 0 -> Left half)
    # Y: 0 to 45 (top 15% of 600 is 90px? No 0.85 to 1.0 is 15% range -> 0.15 * 600 / 2 = 45px, top half is +Y)
    
    # Let's crop the top left corner 400x100
    crop = img.crop((0, 0, 400, 100))
    arr = np.array(crop)
    
    # Search for white pixels (255, 255, 255)
    # Allow some tolerance for anti-aliasing
    white_pixels = np.any((arr[..., 0] > 200) & (arr[..., 1] > 200) & (arr[..., 2] > 200))
    
    if white_pixels:
        print("PASS: White text pixels detected.")
    else:
        print("FAIL: No white text pixels found.")
        exit(1)

if __name__ == "__main__":
    test_label_rendering()
