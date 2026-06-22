// Catalog fragment — window-depth (inner-edge vignette).
//
// A subtle darkening that hugs the four window borders and fades
// inward, giving the whole surface a recessed, "depth around the
// edges" feel. Unlike a classic radial vignette this is an INNER
// shadow keyed off the distance to the NEAREST edge (in physical
// pixels, so it's symmetric regardless of aspect), so the frame —
// not the centre — carries the depth. It composes the SAME depth
// language the popup-elevation chrome uses, so the window and the
// session-switcher card read as one consistent design.
//
// At `intensity` 0.0 this is an exact pass-through. Colour is the
// tint the edges darken TOWARD — fed from the resolved theme
// (no hardcoded effect colour), default near-black.
//
// VsOut + vs_main come from engawa-wgpu's shared fullscreen
// pipeline prelude — the catalog prepends them, so this fragment
// only declares fs_main.

struct WindowDepthParams {
    resolution: vec2<f32>,
    depth: f32,
    intensity: f32,
    color: vec3<f32>,
    softness: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> params: WindowDepthParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(input_tex, input_samp, in.uv);

    // Physical-pixel position + distance to the nearest of the
    // four edges (aspect-correct because it's all in px).
    let px = in.uv * params.resolution;
    let edge_dist = min(
        min(px.x, params.resolution.x - px.x),
        min(px.y, params.resolution.y - px.y),
    );

    // The vignette reaches inward by `depth` as a fraction of the
    // shorter dimension (e.g. 0.08 = 8%). Guard against a zero band.
    let band = max(params.depth, 0.0001) * min(params.resolution.x, params.resolution.y);

    // 0 at the edge → 1 once we're `band` px inward.
    let t = clamp(edge_dist / band, 0.0, 1.0);
    // Shadow weight: 1 at the edge, fading to 0 inward. `softness`
    // shapes the falloff (higher = tighter to the edge).
    let shadow = pow(1.0 - t, max(params.softness, 0.0001));

    // Darken toward the tint by intensity*shadow (clamped → 0 is an
    // exact pass-through).
    let factor = clamp(params.intensity, 0.0, 1.0) * shadow;
    let rgb = mix(color.rgb, params.color, factor);
    return vec4<f32>(rgb, color.a);
}
