// Bloom stage 3/4 — separable gaussian blur, VERTICAL pass.
//
// Pairs with bloom_blur_h.wgsl — identical 9-tap gaussian, the
// two files differ ONLY in the AXIS vector (see that file for
// why the axis is baked instead of uniform-driven).

struct BloomParams {
    resolution: vec2<f32>,
    threshold: f32,
    intensity: f32,
    radius_px: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: BloomParams;

const AXIS: vec2<f32> = vec2<f32>(0.0, 1.0);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<f32>(1.0) / max(params.resolution, vec2<f32>(1.0));
    let step_uv = AXIS * px * max(params.radius_px, 0.0) * 0.5;
    var weights = array<f32, 4>(0.1945946, 0.1216216, 0.054054, 0.016216);
    var acc = textureSample(input_tex, input_samp, in.uv).rgb * 0.227027;
    for (var i = 0; i < 4; i = i + 1) {
        let offset = step_uv * f32(i + 1);
        acc = acc + textureSample(input_tex, input_samp, in.uv + offset).rgb * weights[i];
        acc = acc + textureSample(input_tex, input_samp, in.uv - offset).rgb * weights[i];
    }
    return vec4<f32>(acc, 1.0);
}
