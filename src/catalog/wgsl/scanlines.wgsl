// Catalog fragment — CRT-style horizontal scanlines.
//
// Darkens rows on a cosine profile with period `period_px`
// physical pixels; intensity 0.0 is an exact pass-through.

struct ScanlinesParams {
    resolution: vec2<f32>,
    period_px: f32,
    intensity: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: ScanlinesParams;

const TAU: f32 = 6.283185307179586;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, input_samp, in.uv);
    let y_px = in.uv.y * params.resolution.y;
    let phase = y_px / max(params.period_px, 1.0);
    // 0.0 at line centres, 1.0 between lines.
    let line = 0.5 + 0.5 * cos(phase * TAU);
    let darken = 1.0 - clamp(params.intensity, 0.0, 1.0) * line;
    return vec4<f32>(color.rgb * darken, color.a);
}
