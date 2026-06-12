// Bloom stage 4/4 — additive composite of the blurred bright
// buffer over the untouched scene.

struct BloomParams {
    resolution: vec2<f32>,
    threshold: f32,
    intensity: f32,
    radius_px: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var input_samp: sampler;
@group(0) @binding(3) var<uniform> params: BloomParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, input_samp, in.uv);
    let bloom = textureSample(bloom_tex, input_samp, in.uv);
    let rgb = scene.rgb + bloom.rgb * max(params.intensity, 0.0);
    return vec4<f32>(min(rgb, vec3<f32>(1.0)), scene.a);
}
