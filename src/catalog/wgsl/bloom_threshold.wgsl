// Bloom stage 1/4 — luminance threshold.
//
// Pixels below `threshold` go black; pixels above pass into the
// bright buffer with a soft 0.1-wide knee so the cutoff doesn't
// shimmer on antialiased glyph edges.

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

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, input_samp, in.uv);
    let luma = dot(color.rgb, LUMA);
    let keep = smoothstep(params.threshold, params.threshold + 0.1, luma);
    return vec4<f32>(color.rgb * keep, 1.0);
}
