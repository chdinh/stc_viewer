struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) curvature: f32,
};

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) view_dir: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) curvature: f32,
};

struct Uniforms {
    model_view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>, // .xyz used
    light_dir: vec4<f32>,  // .xyz used (World Space)
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Calculate World Position
    let world_pos = (uniforms.model * vec4<f32>(in.position, 1.0)).xyz;
    
    out.position = uniforms.model_view_projection * vec4<f32>(in.position, 1.0);
    out.normal_world = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    
    // View Direction (Camera - WorldPos)
    out.view_dir = normalize(uniforms.camera_pos.xyz - world_pos);
    
    out.color = in.color;
    out.curvature = in.curvature;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Double-Sided Lighting: Flip normal if we are looking at the back face
    var N = normalize(in.normal_world);
    if (!is_front) {
        N = -N;
    }

    let V = normalize(in.view_dir);
    
    // --- COLOR PALETTE ---
    // Target Deeper Blue (less Turquoise)
    let deep_blue = vec3<f32>(0.2, 0.5, 1.0);
    // Mix original data color with deep blue
    let base_color = mix(in.color, deep_blue, 0.2); 

    // --- 1. AMBIENT & CURVATURE ---
    // Curvature: 0.0 (Concave/Sulci) -> 1.0 (Convex/Gyri)
    let ao_strength = mix(0.1, 1.0, pow(in.curvature, 0.5)); 
    
    // --- DATA DETECTION (Saturation) ---
    // Gray (Background) has ~0 saturation. Activations/Atlas are colorful.
    // We use this to separate "Shell" logic from "Data" logic.
    let gray_val = dot(in.color, vec3<f32>(0.299, 0.587, 0.114));
    let saturation = length(in.color - vec3<f32>(gray_val));
    let is_data = smoothstep(0.05, 0.2, saturation); // 0.0 = Shell, 1.0 = Data

    // --- 2. DIFFUSE ---
    // User requested lighting fixed to camera (Top-Down relative to Camera)
    let L = normalize(uniforms.light_dir.xyz); 
    let diff = max(dot(N, L), 0.0);
    
    // Shell: Low diffuse (Transparent). Data: Near-zero (Gaseous).
    // Not "Solid Plastic" (0.8), but "Gas" (0.05).
    let diff_strength = mix(0.15, 0.05, is_data);
    let diffuse_color = base_color * diff * ao_strength * diff_strength;
    
    // --- 3. SPECULAR ---
    let H = normalize(L + V); 
    let spec_intensity = 0.5;
    let shininess = 64.0;
    let spec = pow(max(dot(N, H), 0.0), shininess);
    let specular_color = vec3<f32>(1.0) * spec * spec_intensity;
    
    // --- 4. RIM LIGHT ---
    let fresnel_power = 3.5; 
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), fresnel_power);
    
    // Shell: Blue Rim. Data: "Electric Light" / "Plasma" Rim using its own color.
    // Boost intensity for data to make it glow.
    let data_rim = in.color * fresnel * 3.0; // Strong glow
    let shell_rim = deep_blue * fresnel * 2.0;
    let rim_color = mix(shell_rim, data_rim, is_data); 
    
    // --- 5. EMISSIVE ---
    // Shell: Edge Glow. Data: "Plasma" feel (White Hot Core + Blue Halo).
    
    // Calculate "Hot Core": Areas facing the camera (N.V ~ 1.0) are "deep" in the volume or center of the glow.
    // Or we can use Specular logic for the "Hotspot".
    // Let's add a "Core Glow" that is White.
    let core_intensity = pow(max(dot(N, V), 0.0), 2.0); // Facing camera
    
    // Data Emission:
    // Base: Atomic Blue (in.color)
    // Edge: Electric Fresnel
    // Core: White Hot mix
    let data_base_glow = in.color * 0.5;
    let data_edge_glow = in.color * fresnel * 2.0;
    
    // Mix white into the core
    let data_hot_core = vec3<f32>(1.0) * core_intensity * 0.8;
    
    let data_emit = data_base_glow + data_edge_glow + data_hot_core;
    
    let shell_emit = deep_blue * fresnel * 0.4;
    let emissive = mix(shell_emit, data_emit, is_data);

    // --- COMBINE ---
    let final_color = diffuse_color + specular_color + rim_color + emissive;
    
    // --- ALPHA / TRANSPARENCY ---
    // Shell: Transparent (0.1). Data: "Transparent Plasma" (0.4).
    // Not Opaque (0.9) as before.
    let alpha_center = mix(0.1, 0.4, is_data); 
    let alpha_edge = 0.95;
    let alpha = clamp(alpha_center + (alpha_edge - alpha_center) * fresnel, 0.0, 1.0);
    
    return vec4<f32>(final_color, alpha);
}
