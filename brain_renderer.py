import os
import requests
import numpy as np
import wgpu
import pyrr
import trimesh

MESH_URL = "https://raw.githubusercontent.com/icemiliang/spherical_harmonic_maps/master/data/brain.obj"
MESH_FILENAME = "brain.obj"

class BrainRenderer:
    """
    Core renderer for the 3D Brain visualization using WebGPU.

    Implements a multi-pass rendering strategy to achieve high-quality transparency (Order Independent Transparency approximation).
    It manages the geometry buffers (Vertex/Index), Unified shader bindings, and the draw calls for the brain mesh.

    Attributes:
        device (wgpu.GPUDevice): The WebGPU device.
        canvas_context (wgpu.GPUCanvasContext): Context for the rendering surface.
        shader_module (wgpu.GPUShaderModule): Compiled WGSL shader.
        uniform_buffer (wgpu.GPUBuffer): Buffer backing the global uniforms.
        vbo (wgpu.GPUBuffer): Vertex Buffer Object.
        ibo (wgpu.GPUBuffer): Index Buffer Object.
        pipeline_layout (wgpu.GPUPipelineLayout): Layout defining bind groups.
        pipe_back_depth, pipe_back_color, pipe_front_depth, pipe_front_color (wgpu.RenderPipeline): Pipelines for the 4-pass render.
    """

    def __init__(self, device, geometry_data, canvas_context=None):
        """
        Initialize the BrainRenderer.

        Args:
            device (wgpu.GPUDevice): The WebGPU device context.
            geometry_data (dict): Dictionary containing keys 'vertices', 'normals', 'faces', 'colors'.
            canvas_context (wgpu.GPUCanvasContext, optional): Canvas context. Defaults to None.
        """
        self.device = device
        self.canvas_context = canvas_context
        
        # Load Shader
        with open("shader.wgsl", "r") as f:
            shader_source = f.read()
        self.shader_module = device.create_shader_module(code=shader_source)
        
        # Prepare geometry from injected data
        if geometry_data:
            self._init_geometry(geometry_data)
        
        # Uniforms
        # Uniforms
        self.uniform_data = np.zeros((44,), dtype=np.float32) # size=176 bytes (MVP+Model+CamPos+LightDir+Params)
        self.uniform_buffer = self.device.create_buffer(size=self.uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        
        self.visualization_mode = 0.0 # 0.0 = Electric, 1.0 = Atlas
        self.hovered_id = -1.0
        
        # Pipeline setup
        self._init_pipeline()
        
        self.start_time = 0.0

    def set_visualization_mode(self, mode):
        """
        Set the visualization mode.
        
        Args:
            mode (float): 0.0 for Electric/Holographic, 1.0 for Solid Atlas.
        """
        self.visualization_mode = mode

    def set_hovered_id(self, region_id):
        """Set the ID of the region to highlight."""
        self.hovered_id = float(region_id)

    def set_data(self, data):
        """Update geometry data completely."""
        self._init_geometry(data)

    def _init_geometry(self, data):
        """
        Initialize Vertex and Index buffers from data dictionary.

        Memory Layout per vertex (interleaved):
        - Position: 3 floats (offset 0)
        - Normal: 3 floats (offset 12)
        - Color: 3 floats (offset 24)
        - Curvature: 1 float (offset 36)
        Total Stride: 40 bytes.

        Args:
            data (dict): Geometry data with keys 'vertices', 'normals', 'colors', 'faces', 'curvature'.
        """
        # Data extraction from dictionary
        vertices = data["vertices"].astype(np.float32)
        vertex_normals = data["normals"].astype(np.float32)
        colors = data["colors"].astype(np.float32) # (N, 3)
        faces = data["faces"].astype(np.uint32)
        
        # Curvature (N,) -> (N, 1)
        if "curvature" in data:
            curvature = data["curvature"].astype(np.float32).reshape(-1, 1)
        else:
            curvature = np.ones((len(vertices), 1), dtype=np.float32) * 0.5
            
        # Labels (N,) -> (N, 1)
        if "labels" in data:
            labels = data["labels"].astype(np.float32).reshape(-1, 1)
        else:
            labels = np.ones((len(vertices), 1), dtype=np.float32) * -1.0

        self.n_indices = len(faces) * 3
        
        # Interleave: Pos(3) + Norm(3) + Color(3) + Curve(1) + Label(1) = 11 floats per vertex
        vertex_data = np.hstack([vertices, vertex_normals, colors, curvature, labels]).flatten().astype(np.float32)
        index_data = faces.flatten().astype(np.uint32)

        self.vbo = self.device.create_buffer_with_data(data=vertex_data, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self.ibo = self.device.create_buffer_with_data(data=index_data, usage=wgpu.BufferUsage.INDEX)

        # Store for dynamic updates
        self.vertices_stored = vertices
        self.normals_stored = vertex_normals
        self.curvature_stored = curvature
        self.labels_stored = labels

    def update_colors(self, new_colors):
        """
        Dynamically update vertex colors (e.g., for time-series animation).

        Args:
            new_colors (np.ndarray): New RGB colors for all vertices. Shape (N, 3).
        """
        # new_colors: (N, 3)
        # Re-interleave data
        # Pos(3) + Norm(3) + Color(3) + Curve(1) + Label(1)
        vertex_data = np.hstack([self.vertices_stored, self.normals_stored, new_colors, self.curvature_stored, self.labels_stored]).flatten().astype(np.float32)
        self.device.queue.write_buffer(self.vbo, 0, vertex_data)

    def _init_pipeline(self):
        """Initialize Bind Groups and Pipeline Layout."""
        self.bind_group_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}}
        ])
        
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout, 
            entries=[
                {"binding": 0, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_data.nbytes}}
            ]
        )
        
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        
        # Default format for initialization
        self.current_format = wgpu.TextureFormat.bgra8unorm
        # Trigger pipeline creation
        self._create_pipelines(self.current_format)

    def _create_pipelines(self, target_format):
        """
        Create the 4 distinct rendering pipelines required for the hollow shell technique.
        
        Strategy: Draw Farthest Back Surface + Closest Front Surface to hide internal mesh structures.
        
        Args:
            target_format (wgpu.TextureFormat): The output swap chain format (usually bgra8unorm).
        """
        # We don't have texture bind group anymore (only uniforms)
        
        common_vertex = {
            "module": self.shader_module,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 11 * 4, # Pos(3)+Norm(3)+Color(3)+Curve(1)+Label(1)
                    "step_mode": "vertex",
                    "attributes": [
                        {"format": "float32x3", "offset": 0, "shader_location": 0},   # pos
                        {"format": "float32x3", "offset": 3 * 4, "shader_location": 1}, # norm
                        {"format": "float32x3", "offset": 6 * 4, "shader_location": 2}, # color
                        {"format": "float32",   "offset": 9 * 4, "shader_location": 3},  # curvature
                        {"format": "float32",   "offset": 10 * 4, "shader_location": 4}  # label
                    ]
                }
            ]
        }
        
        common_fragment = {
            "module": self.shader_module,
            "entry_point": "fs_main",
            "targets": [{
                "format": target_format,
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
        }
        
        # ---------------------------------------------------------
        # 4-Pass Hollow Shell Rendering Pipelines
        # Strategy: Draw Farthest Back Surface + Closest Front Surface.
        # This hides all internal mesh noise while maintaining transparency/volume.
        
        # 1. Back Depth (Find Farthest Surface)
        self.pipe_back_depth = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=common_vertex,
            fragment=None,
            primitive={"topology": "triangle-list", "cull_mode": "front"}, # Draw Back Faces
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True, 
                "depth_compare": wgpu.CompareFunction.greater, # Keep Farthest
            },
        )
        
        # 2. Back Color (Draw Farthest Surface)
        self.pipe_back_color = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=common_vertex,
            fragment=common_fragment,
            primitive={"topology": "triangle-list", "cull_mode": "front"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.equal,
            },
        )

        # 3. Front Depth (Find Closest Surface)
        self.pipe_front_depth = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=common_vertex,
            fragment=None,
            primitive={"topology": "triangle-list", "cull_mode": "back"}, # Draw Front Faces
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True, 
                "depth_compare": wgpu.CompareFunction.less, # Keep Closest
            },
        )
        
        # 4. Front Color (Draw Closest Surface)
        self.pipe_front_color = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=common_vertex,
            fragment=common_fragment,
            primitive={"topology": "triangle-list", "cull_mode": "back"},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.equal, 
            },
        )
        
        return self.pipe_front_color

    def draw(self, target_texture_view, aspect_ratio, view_matrix, camera_pos=None):
        """
        Execute the 4-pass render cycle to draw the brain.

        Passes:
        1. Back Depth: Write depth of farthest back faces.
        2. Back Color: Render farthest back faces where depth matches.
        3. Front Depth: Write depth of closest front faces.
        4. Front Color: Render closest front faces where depth matches.

        Args:
            target_texture_view (wgpu.GPUTextureView): Output color attachment.
            aspect_ratio (float): Screen aspect ratio for projection matrix.
            view_matrix (np.ndarray): 4x4 Camera View Matrix.
            camera_pos (np.ndarray, optional): Camera world position for lighting calc.
        """


        # Update Uniforms
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, aspect_ratio, 0.1, 1000.0)
        view = view_matrix
        model_matrix = pyrr.matrix44.create_identity()
        correction = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ], dtype=np.float32)

        mvp = np.matmul(model_matrix, np.matmul(view, np.matmul(projection, correction)))
        
        mvp_flat = np.ascontiguousarray(mvp, dtype=np.float32).flatten()
        model_flat = np.ascontiguousarray(model_matrix, dtype=np.float32).flatten()
        
        if camera_pos is None:
            cp = np.array([0.0, 0.0, 5.0, 1.0], dtype=np.float32)
        else:
            cp = np.array([camera_pos[0], camera_pos[1], camera_pos[2], 1.0], dtype=np.float32)
            
        # --- Light Direction (Fixed to Camera) ---
        # We want light coming from "Top" relative to the Camera (View Space +Y).
        # View Matrix transforms World -> View.
        # Inverse View Matrix transforms View -> World.
        # Top Vector in View Space: (0, 1, 0, 0)
        view_inv = np.linalg.inv(view)
        light_dir_view = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        light_dir_world = np.dot(view_inv, light_dir_view) # Result is vec4
        ld = light_dir_world.astype(np.float32)
        # Normalize in case of scaling (though rotation shouldn't scale)
        ld_norm = ld / np.linalg.norm(ld)

        # Params (Viz Mode)
        params = np.array([self.visualization_mode, self.hovered_id, 0.0, 0.0], dtype=np.float32)

        combined_uniforms = np.concatenate([mvp_flat, model_flat, cp, ld_norm, params])
        self.device.queue.write_buffer(self.uniform_buffer, 0, combined_uniforms)

        # Depth Texture
        depth_texture = self.device.create_texture(
            size=target_texture_view.texture.size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            format=wgpu.TextureFormat.depth24plus,
        )
        depth_view = depth_texture.create_view()

        command_encoder = self.device.create_command_encoder()
        
        # --- PHASE A: BACK SHELL (Farthest) ---
        
        # Pass 1: Back Depth (Clear to 0.0 for 'greater')
        pass_bd = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_clear_value": 0.0, # Initialize far away (reversed Z view?) No, 0 is Near, 1 is Far?
                # Wait: WGPU default: 0 is Near (Screen), 1 is Far.
                # Standard 'less': Clear 1.0 (Far). Keep if Closer (Less).
                # 'greater': Clear 0.0 (Near). Keep if Farther (Greater).
                "depth_load_op": "clear",
                "depth_store_op": "store",
            }
        )
        pass_bd.set_bind_group(0, self.bind_group, [], 0, 99)
        pass_bd.set_vertex_buffer(0, self.vbo, 0, self.vbo.size)
        pass_bd.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32, 0, self.ibo.size)
        pass_bd.set_pipeline(self.pipe_back_depth)
        pass_bd.draw_indexed(self.n_indices, 1, 0, 0, 0)
        pass_bd.end()
        
        # Pass 2: Back Color (Render blended background)
        pass_bc = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target_texture_view,
                "resolve_target": None,
                "load_op": "clear", # Clear screen
                "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0)
            }],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_clear_value": 0.0, # Ignored
                "depth_load_op": "load",
                "depth_store_op": "discard",
            }
        )
        pass_bc.set_bind_group(0, self.bind_group, [], 0, 99)
        pass_bc.set_vertex_buffer(0, self.vbo, 0, self.vbo.size)
        pass_bc.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32, 0, self.ibo.size)
        pass_bc.set_pipeline(self.pipe_back_color)
        pass_bc.draw_indexed(self.n_indices, 1, 0, 0, 0)
        pass_bc.end()
        
        # --- PHASE B: FRONT SHELL (Closest) ---
        
        # Pass 3: Front Depth (Clear to 1.0 for 'less')
        pass_fd = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_clear_value": 1.0, # Reset depth to Far
                "depth_load_op": "clear", # Must clear for second phase
                "depth_store_op": "store",
            }
        )
        pass_fd.set_bind_group(0, self.bind_group, [], 0, 99)
        pass_fd.set_vertex_buffer(0, self.vbo, 0, self.vbo.size)
        pass_fd.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32, 0, self.ibo.size)
        pass_fd.set_pipeline(self.pipe_front_depth)
        pass_fd.draw_indexed(self.n_indices, 1, 0, 0, 0)
        pass_fd.end()
        
        # Pass 4: Front Color (Render top shell)
        pass_fc = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target_texture_view,
                "resolve_target": None,
                "load_op": "load", # Keep Background
                "store_op": "store",
                "clear_value": (0.1, 0.1, 0.1, 1.0)
            }],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_clear_value": 1.0, 
                "depth_load_op": "load",
                "depth_store_op": "discard",
            }
        )
        pass_fc.set_bind_group(0, self.bind_group, [], 0, 99)
        pass_fc.set_vertex_buffer(0, self.vbo, 0, self.vbo.size)
        pass_fc.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32, 0, self.ibo.size)
        pass_fc.set_pipeline(self.pipe_front_color)
        pass_fc.draw_indexed(self.n_indices, 1, 0, 0, 0)
        pass_fc.end()
        
        self.device.queue.submit([command_encoder.finish()])



if __name__ == "__main__":
    print("-" * 50)
    print("This is the shared library file 'brain_renderer.py'.")
    print("It is NOT meant to be run directly for the visible application.")
    print("\nPlease run:")
    print("  python3 main.py           -> to start the visual output")
    print("  python3 test_renderer.py  -> to run the verification test")
    print("-" * 50)
