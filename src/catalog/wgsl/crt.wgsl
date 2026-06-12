// Catalog fragment — CRT curvature + chromatic aberration +
// vignette.
//
// All textureSample calls stay in uniform control flow (no
// early return before sampling) — naga's uniformity analysis
// rejects implicit-derivative sampling after divergent control
// flow, so the out-of-bounds border is applied as an `inside`
// multiplier instead of a branch.

struct CrtParams {
    resolution: vec2<f32>,
    curvature: f32,
    vignette: f32,
    aberration: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: CrtParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let centered = in.uv * 2.0 - vec2<f32>(1.0);
    let r2 = dot(centered, centered);
    let warped = centered * (1.0 + params.curvature * r2);
    let uv = warped * 0.5 + vec2<f32>(0.5);
    let cuv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let px = vec2<f32>(1.0) / max(params.resolution, vec2<f32>(1.0));
    let shift = centered * params.aberration;
    let red = textureSample(input_tex, input_samp, cuv + shift * px).r;
    let center_samp = textureSample(input_tex, input_samp, cuv);
    let blue = textureSample(input_tex, input_samp, cuv - shift * px).b;

    let inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
    let v = 1.0 - clamp(params.vignette, 0.0, 1.0) * smoothstep(0.5, 1.4, length(centered));
    let rgb = vec3<f32>(red, center_samp.g, blue) * v * inside;
    return vec4<f32>(rgb, center_samp.a);
}
