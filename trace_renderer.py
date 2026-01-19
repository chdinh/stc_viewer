import wgpu
import numpy as np

class TraceRenderer:
    """
    Renders 2D time-series traces (butterfly plots) using WebGPU.

    Visualizes the signal activity of selected brain clusters. The traces are rendered
    as line strips against a basic coordinate system.

    Attributes:
        device (wgpu.GPUDevice): The WebGPU device used for rendering.
        render_format (wgpu.TextureFormat): The color attachment format (e.g., bgra8unorm).
        traces (list): List of 1D numpy arrays containing signal data [0..1].
        pipeline (wgpu.RenderPipeline): The configured rendering pipeline for lines.
        vbo (wgpu.GPUBuffer): Vertex Buffer Object storing line segments and axes.
        vertex_count (int): Number of vertices to draw.
        shader (wgpu.GPUShaderModule): The shader module for 2D line rendering.
    """

    def __init__(self, device, render_format):
        """
        Initialize the TraceRenderer.

        Args:
            device (wgpu.GPUDevice): WebGPU device context.
            render_format (wgpu.TextureFormat): Output texture format.
        """
        self.device = device
        self.render_format = render_format
        self.traces = []
        self.pipeline = None
        self.vbo = None
        self.vertex_count = 0
        
        # Shader for 2D lines
        shader_source = """
        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) color: vec3<f32>,
        };
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        };
        
        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;
            // Input position is already in NDC or ready to be mapped
            // We map x [0..1] to [0.5, 0.95] (Right side)
            // We map y [0..1] to [-0.9, -0.6] (Bottom side)
            
            let x = in.position.x;
            let y = in.position.y;
            
            let screen_x = 0.5 + (x * 0.45);
            let screen_y = -0.9 + (y * 0.3);
            
            out.position = vec4<f32>(screen_x, screen_y, 0.0, 1.0);
            out.color = in.color;
            return out;
        }
        
        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(in.color, 1.0);
        }
        """
        self.shader = device.create_shader_module(code=shader_source)
        self._create_pipeline()
        
    def _create_pipeline(self):
        """Configure the WebGPU render pipeline for 2D line drawing."""
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": self.shader,
                "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 5 * 4, # x,y,r,g,b
                    "step_mode": "vertex",
                    "attributes": [
                        {"format": "float32x2", "offset": 0, "shader_location": 0},
                        {"format": "float32x3", "offset": 8, "shader_location": 1},
                    ]
                }]
            },
            fragment={
                "module": self.shader,
                "entry_point": "fs_main",
                "targets": [{
                    "format": self.render_format,
                    "blend": {
                         "color": {
                            "src_factor": wgpu.BlendFactor.src_alpha,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        },
                        "alpha": {
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            "operation": wgpu.BlendOperation.add,
                        }
                    }
                }]
            },
            primitive={
                "topology": "line-list",
            },
            depth_stencil=None
        )
        
    def set_data(self, traces):
        """
        Update the trace data and rebuild the vertex buffer.

        Args:
            traces (list): A list of 1D numpy arrays (float32) representing the signal over time.
        """
        # Traces: List of np.arrays (frames,) containing signal [0..1]
        self.traces = traces
        if not traces:
            return
            
        points = []
        self.n_frames = len(traces[0])
        n_frames = self.n_frames
        
        # Colors for the 5 clusters
        colors = [
            [1.0, 0.0, 0.0], # Red
            [0.0, 1.0, 0.0], # Green
            [0.0, 0.0, 1.0], # Blue
            [1.0, 1.0, 0.0], # Yellow
            [1.0, 0.0, 1.0]  # Magenta
        ]
        
        # --- Traces ---
        # Build line segments (line-list)
        for i, trace in enumerate(traces):
            xs = np.linspace(0, 1, n_frames)
            ys = trace 
            c = colors[i % len(colors)]
            
            for j in range(n_frames - 1):
                points.extend([xs[j], ys[j], c[0], c[1], c[2]])
                points.extend([xs[j+1], ys[j+1], c[0], c[1], c[2]])
        
        # --- Axes (White) ---
        ac = [1.0, 1.0, 1.0] # Axis Color
        # Y-Axis (at x=0)
        points.extend([0.0, 0.0, ac[0], ac[1], ac[2]])
        points.extend([0.0, 1.0, ac[0], ac[1], ac[2]])
        # X-Axis (at y=0)
        points.extend([0.0, 0.0, ac[0], ac[1], ac[2]])
        points.extend([1.0, 0.0, ac[0], ac[1], ac[2]])
        
        vertex_data = np.array(points, dtype=np.float32)
        self.vertex_count = len(points) // 5
        
        if self.vertex_count > 0:
            self.vbo = self.device.create_buffer_with_data(data=vertex_data, usage=wgpu.BufferUsage.VERTEX)
        else:
            self.vbo = None
            
        # Create Cursor Buffer (2 vertices, dynamic)
        # Usage: VERTEX | COPY_DST so we can write to it
        self.cursor_vbo = self.device.create_buffer(size=2 * 5 * 4, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)


    def draw(self, target_view, frame_idx):
        """
        Execute the draw call for the trace overlay.

        Args:
            target_view (wgpu.GPUTextureView): The view to render into (usually the canvas current texture).
            frame_idx (int): The current animation frame index for cursor positioning.
        """
        if not self.vbo or self.vertex_count == 0:
            return

        # Dynamic Format Adaptation
        if target_view.texture.format != self.render_format:
            print(f"TraceRenderer: Format changed from {self.render_format} to {target_view.texture.format}. Recreating pipeline.")
            self.render_format = target_view.texture.format
            self._create_pipeline()
            
        # Update Cursor Logic
        if self.n_frames > 1:
            progress = frame_idx / (self.n_frames - 1)
        else:
            progress = 0.0
            
        # Cursor Line (Vertical White Line at 'progress')
        cc = [1.0, 1.0, 1.0]
        cursor_data = np.array([
            progress, 0.0, cc[0], cc[1], cc[2],
            progress, 1.0, cc[0], cc[1], cc[2]
        ], dtype=np.float32)
        
        self.device.queue.write_buffer(self.cursor_vbo, 0, cursor_data)
        
        # Draw
        command_encoder = self.device.create_command_encoder()
        
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target_view,
                "resolve_target": None,
                "load_op": "load", 
                "store_op": "store",
            }],
        )
        
        render_pass.set_pipeline(self.pipeline)
        
        # 1. Draw Traces & Axes
        render_pass.set_vertex_buffer(0, self.vbo, 0, self.vbo.size)
        render_pass.draw(self.vertex_count, 1, 0, 0)
        
        # 2. Draw Cursor
        render_pass.set_vertex_buffer(0, self.cursor_vbo, 0, self.cursor_vbo.size)
        render_pass.draw(2, 1, 0, 0)
        
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
