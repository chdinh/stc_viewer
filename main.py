import time
import wgpu
from rendercanvas.auto import RenderCanvas, loop
from brain_renderer import BrainRenderer
from camera import Camera
from atlas_loader import load_brain_data

# Setup window
canvas = RenderCanvas(title="Python WebGPU Renderer", size=(800, 600))
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# Load Data
print("Initializing Data...")
try:
    brain_data = load_brain_data()
except Exception as e:
    print(f"Data load failed ({e}). using fallback dummy data.")
    from dummy_loader import load_brain_data
    brain_data = load_brain_data()

# Setup Renderer and Camera
from trace_renderer import TraceRenderer

# Setup Renderer and Camera
renderer = BrainRenderer(device, brain_data, canvas)
camera = Camera(canvas)

# Configure context (initial)
present_context = canvas.get_context("wgpu")
render_format = present_context.get_preferred_format(adapter)
present_context.configure(device=device, format=render_format)

# Setup Trace Overlay
trace_renderer = TraceRenderer(device, render_format)
trace_renderer.set_data(brain_data.get("traces", []))

# Animation State
color_frames = brain_data.get("color_frames")
atlas_colors = brain_data.get("atlas_colors")
start_time = time.time()
render_mode = "dynamic" 
show_traces = True # Default On

def draw():
    try:
        current_texture = present_context.get_current_texture()
        current_view = current_texture.create_view()
        
        # Calculate aspect ratio
        size = current_texture.size
        # Correctly handle minimized window
        if size[1] == 0:
             return

        aspect = size[0] / size[1]
        
        elapsed = time.time() - start_time
        frame_idx = 0
        
        # --- MNE Time Trace Animation ---
        if render_mode == "dynamic" and color_frames is not None:
             # Play at 30 Hz (MNE real-time speed)
            frame_idx = int(elapsed * 30) % color_frames.shape[1]
            current_colors = color_frames[:, frame_idx, :]
            renderer.update_colors(current_colors)
            
            # Keep loop running for animation
            canvas.request_draw()
            
        elif render_mode == "atlas" and atlas_colors is not None:
             pass 

        view_matrix = camera.get_view_matrix()
        renderer.draw(current_view, aspect, view_matrix, camera_pos=camera.position)
        
        # --- Trace Overlay ---
        if show_traces:
            # We also pass frame_idx if we want to draw a vertical time bar (future)
            trace_renderer.draw(current_view, frame_idx)
        
    except Exception as e:
        print(f"Draw Loop Error: {e}")
        import traceback
        traceback.print_exc()

# Handle events
def handle_event(event):
    global render_mode, show_traces
    # Camera events
    camera.handle_event(event)
    
    # Keyboard toggle
    if event["event_type"] == "key_down":
        if event["key"] == "t":
            if render_mode == "dynamic":
                render_mode = "atlas"
                print("Switched to Surface Atlas Mode.")
                if atlas_colors is not None:
                    renderer.update_colors(atlas_colors)
            else:
                render_mode = "dynamic"
                print("Switched to Dynamic Source Mode.")
            canvas.request_draw()
            
        elif event["key"] == "p":
            show_traces = not show_traces
            print(f"Butterfly Plot Overlay: {show_traces}")
            canvas.request_draw()

canvas.add_event_handler(handle_event, "pointer_down", "pointer_up", "pointer_move", "wheel", "key_down")
canvas.request_draw(draw)
print("Renderer started. Check the window.")
print("Controls: Left Click to Orbit, Right Click to Pan, Scroll to Zoom.")
print("Key 't': Toggle between Source Animation and Atlas Colors.")
print("Key 'p': Toggle Butterfly Plot Overlay.")
loop.run()
