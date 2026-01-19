import wgpu
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextRenderer:
    """
    Renders 2D text overlays using WebGPU key-value texture mapping.
    Uses PIL (Pillow) to generate the text bitmap and uploads it to a GPU Texture.
    """

    def __init__(self, device, render_format):
        self.device = device
        self.render_format = render_format
        self.pipeline = None
        self.texture = None
        self.sampler = None
        self.bind_group = None
        self.current_text = ""
        
        # Shader: Renders a full-screen or positioned quad with texture
        # We will render a quad in the top-left corner
        shader_source = """
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };
        
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
            var out: VertexOutput;
            // Generates a quad (0,0) to (1,1) in UV space
            // Maps to Top-Left of screen in NDC
            // We want fixed pixels usually, but for simplicity let's use fixed NDC area
            // Top-Left (-1, 1) to (-0.5, 0.9) ?
            
            // Standard full-screen triangle strip logic, cropped
            // Let's rely on a uniform or just hardcode a "Title Bar" area
            // Area: Top Left, 400px wide, 50px high?
            // Let's assume Screen is 800x600.
            
            var pos = array<vec2<f32>, 4>(
                vec2<f32>(-1.0, 1.0),  // TL
                vec2<f32>(-1.0, 0.85), // BL
                vec2<f32>( 0.0, 1.0),  // TR
                vec2<f32>( 0.0, 0.85)  // BR
            );
            
            var uv = array<vec2<f32>, 4>(
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(1.0, 0.0),
                vec2<f32>(1.0, 1.0)
            );
            
            out.position = vec4<f32>(pos[index], 0.0, 1.0);
            out.uv = uv[index];
            return out;
        }
        
        @group(0) @binding(0) var t_diffuse: texture_2d<f32>;
        @group(0) @binding(1) var s_diffuse: sampler;
        
        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            return textureSample(t_diffuse, s_diffuse, in.uv);
        }
        """
        self.shader = device.create_shader_module(code=shader_source)
        
        self.sampler = device.create_sampler(
            mag_filter="linear",
            min_filter="linear",
        )
        
        self._create_pipeline()
        
        # Initialize with empty text
        self.set_text("Initializing...")

    def _create_pipeline(self):
        self.bind_group_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
        ])
        
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.shader,
                "entry_point": "vs_main",
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
            primitive={"topology": "triangle-strip"},
        )

    def set_text(self, text):
        if text == self.current_text:
            return
        self.current_text = text
        
        # Create Image
        W, H = 512, 64 # Texture size
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw Text
        # Default font
        try:
            # Try to load a nicer font if possible, or default
            # Does not support size in default font, but let's try
            font = ImageFont.truetype("Arial", 40)
        except:
             font = ImageFont.load_default()
             
        # White Text with Shadow for visibility
        draw.text((3, 3), text, font=font, fill=(0, 0, 0, 255)) # Shadow
        draw.text((2, 2), text, font=font, fill=(255, 255, 255, 255)) # White
        
        # Upload
        data = np.array(img).flatten() # Ensure correct layout? PIL is usually row-major
        # wgpu expects bytes
        # Image is RGBA
        
        if self.texture is None:
            self.texture = self.device.create_texture(
                size=(W, H, 1),
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                format=wgpu.TextureFormat.rgba8unorm,
            )
            
        self.device.queue.write_texture(
            {"texture": self.texture},
            data.tobytes(),
            {"bytes_per_row": W * 4, "rows_per_image": H},
            (W, H, 1)
        )
        
        # Recreate Bind Group
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.texture.create_view()},
                {"binding": 1, "resource": self.sampler},
            ]
        )

    def draw(self, target_view):
        if not self.bind_group or not self.texture:
            return
            
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(
            color_attachments=[{
                "view": target_view,
                "resolve_target": None,
                "load_op": "load",
                "store_op": "store",
            }]
        )
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self.bind_group, [], 0, 99)
        pass_enc.draw(4, 1, 0, 0)
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])
