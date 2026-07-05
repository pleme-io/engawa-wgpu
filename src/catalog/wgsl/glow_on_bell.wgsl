// Catalog fragment — radial glow on bell (BEL).
//
// `bell_intensity` is the consumer's clock: set it to 1.0 when
// BEL arrives and decay it per frame on the host (typical:
// intensity * 0.92^(dt * 60)). The shader is stateless — the
// glow fades because the uniform fades.

struct GlowOnBellParams {
    resolution: vec2<f32>,
    center_px: vec2<f32>,
    bell_intensity: f32,
    radius_px: f32,
    _pad0: f32,
    _pad1: f32,
    // Additive glow colour (rgb; a unused). vec4 for 16-byte alignment
    // at offset 32. Defaults to the cool near-white BELL_TINT host-side.
    tint: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: GlowOnBellParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(input_tex, input_samp, in.uv);
    let p = in.uv * params.resolution;
    let d = distance(p, params.center_px);
    let sigma = max(params.radius_px, 1.0);
    let falloff = exp(-(d * d) / (2.0 * sigma * sigma));
    let glow = clamp(params.bell_intensity, 0.0, 1.0) * falloff;
    let rgb = min(scene.rgb + params.tint.rgb * glow, vec3<f32>(1.0));
    return vec4<f32>(rgb, scene.a);
}
