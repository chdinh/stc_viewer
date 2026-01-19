import wgpu
import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from brain_renderer import BrainRenderer
from trace_renderer import TraceRenderer
import time

def test_renderers():
    print("Setting up Headless/Automated Test...")
    
    # 1. Setup Context
    # We use a small canvas, but won't loop forever.
    canvas = RenderCanvas(title="Test Window", size=(400, 300))
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    
    present_context = canvas.get_context("wgpu")
    render_format = present_context.get_preferred_format(adapter)
    present_context.configure(device=device, format=render_format)
    
    # 2. Mock Data
    print("Generating Mock Data...")
    N = 900
    vertices = np.random.rand(N, 3).astype(np.float32)
    normals = np.random.rand(N, 3).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colors = np.ones((N, 3), dtype=np.float32)
    curvature = np.zeros(N, dtype=np.float32)
    faces = np.arange(N, dtype=np.uint32).reshape(-1, 3) # Disconnected triangles
    faces = faces[:N//3]
    
    brain_data = {
        "vertices": vertices * 100.0,
        "faces": faces,
        "normals": normals,
        "colors": colors,
        "curvature": curvature
    }
    
    traces = [np.random.rand(100).astype(np.float32) for _ in range(5)]
    
    # 3. Init Renderers
    print("Initializing BrainRenderer...")
    renderer = BrainRenderer(device, brain_data, canvas)
    
    print("Initializing TraceRenderer...")
    trace_renderer = TraceRenderer(device, render_format)
    trace_renderer.set_data(traces)
    
    # 4. Draw Call
    print("Attempting Draw Call...")
    try:
        current_texture = present_context.get_current_texture()
        current_view = current_texture.create_view()
        
        # Brain Draw
        import pyrr
        view_matrix = pyrr.matrix44.create_look_at(
            eye=[0, 0, 300], target=[0, 0, 0], up=[0, 1, 0]
        ).astype(np.float32)
        renderer.draw(current_view, 1.33, view_matrix)
        
        # Trace Draw
        trace_renderer.draw(current_view, 0)
        
        print("Draw Successful!")
        print("TEST PASSED.")
        
    except Exception as e:
        print(f"TEST FAILED with Error: {e}")
        raise e
        
    print("Closing...")
    # canvas.close() # rendercanvas might not have close() method directly exposed easily?
    # actually we just exit the script.

if __name__ == "__main__":
    test_renderers()
