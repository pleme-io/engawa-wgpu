// Catalog fragment — colorblind simulation.
//
// Machado et al. 2009 color-vision-deficiency simulation
// matrices at severity 1.0, ported VERBATIM from mado
// src/render.rs COLORBLIND_SHADER (2026-06-12). The literals
// below are byte-exact; catalog/colorblind.rs mirrors them as
// Rust consts and the unit tests pin both copies.
//
// VsOut + vs_main come from engawa-wgpu's shared fullscreen
// vertex (combined_shader_source).

struct ColorblindParams {
    mode: u32,  // 0=none, 1=protanopia, 2=deuteranopia, 3=tritanopia
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: ColorblindParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, input_samp, in.uv);

    if params.mode == 0u { return color; }

    let r = color.r; let g = color.g; let b = color.b;
    var out_r: f32; var out_g: f32; var out_b: f32;

    // Machado et al. 2009 simulation matrices (severity = 1.0)
    if params.mode == 1u {
        // Protanopia (red-blind)
        out_r = 0.152286 * r + 1.052583 * g - 0.204868 * b;
        out_g = 0.114503 * r + 0.786281 * g + 0.099216 * b;
        out_b = -0.003882 * r - 0.048116 * g + 1.051998 * b;
    } else if params.mode == 2u {
        // Deuteranopia (green-blind)
        out_r = 0.367322 * r + 0.860646 * g - 0.227968 * b;
        out_g = 0.280085 * r + 0.672501 * g + 0.047413 * b;
        out_b = -0.011820 * r + 0.042940 * g + 0.968881 * b;
    } else if params.mode == 3u {
        // Tritanopia (blue-blind)
        out_r = 1.255528 * r - 0.076749 * g - 0.178779 * b;
        out_g = -0.078411 * r + 0.930809 * g + 0.147602 * b;
        out_b = 0.004733 * r + 0.691367 * g + 0.303900 * b;
    } else {
        // Out-of-contract mode words (a Pod-cast bypass of
        // ColorblindParams::new can mint any u32) degrade to
        // pass-through — never to a silent Tritanopia, which is
        // what the former catch-all `else` rendered (M3 review
        // 2026-06-12).
        return color;
    }

    return vec4(clamp(out_r, 0.0, 1.0), clamp(out_g, 0.0, 1.0), clamp(out_b, 0.0, 1.0), color.a);
}
