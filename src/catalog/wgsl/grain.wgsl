// Catalog fragment — paper-grain "tooth".
//
// A luma-only film grain: a faint fabric texture laid on TOP of
// everything (priority 750, above colorblind). For each pixel we
// hash the physical-pixel coordinate (scaled by `scale`) plus a
// slowly-quantized time and add a luminance-only jitter of
// ±`opacity`. Chroma is never perturbed — the jitter is a grey
// delta, so accents stay clean.
//
// At `opacity` 0.0 this is an exact pass-through (the grey delta
// is multiplied by opacity, which is also clamped to 0).
//
// VsOut + vs_main come from engawa-wgpu's shared fullscreen
// pipeline prelude — the catalog prepends them, so this fragment
// only declares fs_main.

struct GrainParams {
    resolution: vec2<f32>,
    opacity: f32,
    scale: f32,
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: GrainParams;

// Cheap 2D hash → [0, 1). Single fract/sin pair — one ALU burst,
// no texture fetch beyond the scene sample.
fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, input_samp, in.uv);

    // Physical-pixel coordinate scaled by `scale` — finer scale =
    // larger grain cells.
    let px = in.uv * params.resolution * max(params.scale, 0.0001);
    // Quantize time to ~5 updates/sec so the grain shimmers slowly
    // rather than re-rolling every frame.
    let t_q = floor(params.time * 5.0);
    // Hash → [0,1), recentre to [-1, 1] so the jitter is signed.
    let n = hash21(px + vec2<f32>(t_q, t_q * 1.7)) * 2.0 - 1.0;

    // Luminance-only grey delta — same value added to r, g, b, so
    // chroma is untouched. Multiplied by opacity (clamped) so 0 =
    // exact pass-through.
    let delta = n * clamp(params.opacity, 0.0, 1.0);
    let rgb = clamp(color.rgb + vec3<f32>(delta), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, color.a);
}
