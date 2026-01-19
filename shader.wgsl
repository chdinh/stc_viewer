/*
    Core WGSL Shader
    
    This shader implements the rendering logic for the brain surface visualization.
    It supports two distinct rendering modes based on vertex saturation:
    1. "Shell": The base brain (low saturation), rendered as transparent blue glass.
    2. "Data": Activated regions (high saturation), rendered as emissive, electric holographic plasma.
    
    Lighting Model:
    - Ambient Occlusion (based on curvature)
    - Diffuse (Lambert)
    - Specular (Phong)
    - Rim Lighting (Fresnel)
    - Emissive (Core intensity)
*/

struct VertexInput {
    @location(0) position: vec3<f32>,  // Model Space Position
    @location(1) normal: vec3<f32>,    // Model Space Normal
    @location(2) color: vec3<f32>,     // Per-vertex Color
    @location(3) curvature: f32,       // Normalized Curvature [0..1]
    @location(4) region_id: f32,       // Region ID
};

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>, // Clip Space Position
    @location(0) normal_world: vec3<f32>,              // Interpolated World Normal
    @location(1) view_dir: vec3<f32>,                  // View Direction vector
    @location(2) color: vec3<f32>,                     // Pass-through interpolated color
    @location(3) curvature: f32,                       // Pass-through curvature
    @location(4) region_id: f32,                       // Pass-through Region ID
};

struct Uniforms {
    model_view_projection: mat4x4<f32>, // MVP Matrix
    model: mat4x4<f32>,                 // Model Matrix
    camera_pos: vec4<f32>,              // Camera World Position (.xyz)
    light_dir: vec4<f32>,               // Light Direction Normalized (.xyz)
    params: vec4<f32>,                  // .x = visualization_mode (0=Electric, 1=Atlas)
                                        // .y = hovered_id (-1 = None)
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
    out.region_id = in.region_id;
    
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
    let L = normalize(uniforms.light_dir.xyz); 

    // --- CHECK VISUALIZATION MODE ---
    let mode = uniforms.params.x;
    
    // --- MODE 1: ATLAS (Standard Rendering) ---
    if (mode > 0.5) {
        // Simple Phong shading to show true atlas colors
        
        // Ambient
        let ambient = 0.3 * in.color;
        
        // Diffuse
        let diff = max(dot(N, L), 0.0);
        let diffuse = diff * in.color * 0.7;
        
        // Specular
        let H = normalize(L + V);
        let spec = pow(max(dot(N, H), 0.0), 32.0);
        let specular = vec3<f32>(0.2) * spec; // Low specular for matte look
        
        var final_color = ambient + diffuse + specular;
        
        // --- HOVER HIGHLIGHT (Atlas Mode) ---
        let hovered_id = uniforms.params.y;
        if (hovered_id >= -0.5 && abs(in.region_id - hovered_id) < 0.2) {
             let highlight = vec3<f32>(1.0, 1.0, 0.8) * 0.6; // Constant glow
             final_color = final_color + highlight;
        }
        
        // Opaque
        return vec4<f32>(final_color, 1.0);
    }
    
    // --- MODE 0: ELECTRIC / HOLOGRAPHIC (Original) ---

    // --- DATA DETECTION (Saturation) ---
    // Gray (Background) has ~0 saturation. Activations/Atlas are colorful.
    // We use this to separate "Shell" logic from "Data" logic.
    let gray_val = dot(in.color, vec3<f32>(0.299, 0.587, 0.114));
    let saturation = length(in.color - vec3<f32>(gray_val));
    let is_data = smoothstep(0.05, 0.2, saturation); 

    // --- COLOR PALETTE ---
    // Shell: Turquoise Blue (Original)
    let deep_blue = vec3<f32>(0.2, 0.5, 1.0);
    // Data: Futuristic Electric Blue
    let electric_blue = vec3<f32>(0.05, 0.15, 1.0);

    // Mix Colors: Shell uses Turquoise, Data uses Electric Blue
    let shell_base = mix(in.color, deep_blue, 0.2);
    let data_base = mix(in.color, electric_blue, 0.5);
    let base_color = mix(shell_base, data_base, is_data);

    // --- 1. AMBIENT & CURVATURE ---
    let ao_strength = mix(0.1, 1.0, pow(in.curvature, 0.5)); 
    
    // --- 2. DIFFUSE ---
    let diff = max(dot(N, L), 0.0);
    let diff_strength = mix(0.15, 0.05, is_data);
    let diffuse_color = base_color * diff * ao_strength * diff_strength;
    
    // --- 3. SPECULAR ---
    let H = normalize(L + V); 
    let spec_intensity = 0.5;
    let shininess = 64.0;
    let spec = pow(max(dot(N, H), 0.0), shininess);
    let specular_color = vec3<f32>(1.0) * spec * spec_intensity;
    
    // --- 4. RIM LIGHT ---
    let fresnel_power = 3.0; 
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), fresnel_power);
    
    // Shell Rim: Turquoise, softer
    let shell_rim = deep_blue * fresnel * 2.0;

    // Data Rim: Electric Blue, sharper, brighter
    // Use electric_blue for the halo instead of raw data color to unify the look
    let data_rim_color = mix(in.color, electric_blue, 0.5);
    // Super sharp and bright rim for "forcefield" effect
    // Power 3.0 -> 5.0 (Thinner edge)
    // Mult 5.0 -> 8.0 (Brighter edge)
    let data_rim_fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    let data_rim = data_rim_color * data_rim_fresnel * 8.0; 

    let rim_color = mix(shell_rim, data_rim, is_data); 
    
    // --- 5. EMISSIVE & CORE ---
    let N_dot_V = max(dot(N, V), 0.0);
    
    // Shell Emission: Edge Glow only
    let shell_emit = deep_blue * fresnel * 0.4;

    // Data Emission: White Hot Core + Electric Body
    let core_metric = pow(N_dot_V, 3.0); 
    let data_core_color = mix(data_base, vec3<f32>(1.0), core_metric); // White center
    let data_emit = data_core_color * (0.8 + core_metric * 4.0); // High intensity

    let emissive = mix(shell_emit, data_emit, is_data);

    // --- COMBINE ---
    var final_color = diffuse_color + specular_color + rim_color + emissive;
    
    // --- HOVER HIGHLIGHT (Mode 1 Only usually, or both?) ---
    // Let's allow it in both modes.
    let hovered_id = uniforms.params.y;
    // Check if this vertex belongs to the hovered region
    // in.region_id is interpolated result. Since region IDs are integers, 
    // we must check with a small epsilon.
    // However, interpolation across triangle boundaries between different regions 
    // might produce intermediate values.
    // For atlas visualization, typically vertices at boundaries are duplicated or share ID?
    // In our loader, we just pushed vertex labels.
    // Vertices are shared -> Smooth shading -> Interpolated IDs?
    // If vertices are shared between regions, one vertex has one ID.
    // So the triangle will interpret from ID A to ID B.
    // This might cause artifacts at borders.
    // But let's try strict check logic:
    
    // If |ID - Hover| < 0.5
    if (abs(in.region_id - hovered_id) < 0.2) {
        // Apply Shiny Highlight
        // Add strong rim light (White/Gold)
        let highlight_rim = vec3<f32>(1.0, 1.0, 0.8) * fresnel * 3.0;
        final_color = final_color + highlight_rim + vec3<f32>(0.2); // Add ambient boost
    }

    // --- ALPHA / TRANSPARENCY ---
    // Shell: Original logic (Transparent center, opaque rim)
    let alpha_shell = clamp(0.1 + 0.85 * fresnel, 0.0, 1.0);
    
    // Data: Holographic (Transparent body, Opaque Core)
    // Reduce edge alpha (for hologram feel), Increase core alpha (for white light)
    // Max alpha 0.6 to keep it "ghostly" / "holographic" even at center
    let alpha_data = clamp(0.1 + 0.5 * core_metric, 0.0, 0.6); 
    
    let final_alpha = mix(alpha_shell, alpha_data, is_data);
    
    return vec4<f32>(final_color, final_alpha);
}
