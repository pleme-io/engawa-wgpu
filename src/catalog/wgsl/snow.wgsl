// Catalog fragment — snow overlay.
//
// ABSORBED from engawa-snow assets/snow.wgsl (2026-06-12); the
// engawa-snow repo stays the standalone demo, the catalog copy
// is the dispatcher-native one. Two deliberate deltas from the
// upstream asset, everything else verbatim:
//
//   1. Bindings re-banded to the catalog convention: scene
//      texture @binding(0), shared sampler @binding(1), params
//      uniform @binding(2) (upstream had only the uniform at 0).
//   2. The final composite happens IN-SHADER (premultiplied
//      over): the standalone overlay relied on hardware
//      SrcAlpha blending in mado's bespoke pipeline, but
//      WgpuDispatcher pipelines are blend-free ping-pong
//      passes, so the shader samples the scene and mixes.
//
// — original engawa-snow header follows —
//
// engawa-snow — fractal flakes, pure-gravity fall, particle pile.
//
// Performance budget:
//   * 1 atan2 + 1 cos + 1 pow per visible pixel inside a flake
//   * Single-octave noise for pile contour (not fbm)
//   * Empty-cell early-out hits ~55% of pixels before any SDF work
//   * No turbulence / sway / wind / typing-pulse compositing
//
// Visual model:
//   * Tiny 6-arm fractal-looking star — cheap `cos(3θ)` lobes
//     give the dendrite silhouette at 3-6 px without aliasing
//   * Pure +y gravity, slow terminal velocity
//   * The pile at the bottom IS made of particles — a denser
//     cell grid below the accumulation line renders the same
//     flake primitive packed together. No painted body fill;
//     the white floor emerges from particle density.
//   * Cold = pile grows. Warm = pile melts. Host integrates
//     the level; shader is stateless per-frame.

struct SnowParams {
    frame: vec4<f32>,        // (time, intensity, wind, typing_pulse)
    // params.x = accumulation [0..1] (host-integrated pile height)
    // params.y = layer_count [1..3]
    // params.z = temperature [0..1] — cold = grow, warm = melt
    // params.w = _reserved
    params: vec4<f32>,
    resolution: vec4<f32>,   // (rx, ry, _, _)
    cursor: vec4<f32>,       // (cx, cy, _, _)
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> P: SnowParams;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const DENSITY: f32 = 9.0;
const PILE_DENSITY: f32 = 32.0;
const LAYERS_MAX: f32 = 3.0;
const MAX_ALPHA: f32 = 0.35;

// ── hashing & noise ────────────────────────────────────────────

fn hash12(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(0.1031, 0.1030));
    q = q + dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash12(p),
        hash12(p + vec2<f32>(17.13, 31.17)),
    );
}

// Single-octave value noise (cheap pile-surface contour).
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ── tiny fractal dendrite ──────────────────────────────────────
//
// Six-fold-symmetric "fractal-looking" silhouette using a
// `cos(3θ)` lobe modulation. Each arm peaks at the same radial
// distance and decays toward the gaps; pow() sharpens the
// lobes into distinct dendrite arms.
fn fractal_dendrite(p_in: vec2<f32>, angle: f32) -> f32 {
    let p = vec2<f32>(
        p_in.x * cos(-angle) - p_in.y * sin(-angle),
        p_in.x * sin(-angle) + p_in.y * cos(-angle),
    );
    let r = length(p);
    if (r > 1.0) { return 0.0; }
    let theta = atan2(p.y, p.x);
    // 6 lobes (cos(3θ) has period 2π/3, abs gives 6 peaks/360°)
    let arm = abs(cos(theta * 3.0));
    let arm_density = pow(arm, 5.0);
    // Falls off radially; brighter core for the central nucleus.
    let core = smoothstep(0.30, 0.0, r) * 0.4;
    let alpha = (1.0 - r) * arm_density + core;
    return clamp(alpha, 0.0, 1.0);
}

// ── one falling flake within a tile cell ───────────────────────

fn render_flake(cell_uv: vec2<f32>, jitter: vec2<f32>, depth: f32) -> vec4<f32> {
    // 55% of cells empty — sparse natural distribution.
    if (jitter.x < 0.45) {
        return vec4<f32>(0.0);
    }

    let size_jitter = mix(0.6, 1.0, jitter.y);
    let base_size = mix(0.08, 0.025, depth);
    let size = base_size * size_jitter;

    let center = vec2<f32>(0.5) + (hash22(jitter * 7.0) - vec2<f32>(0.5)) * 0.4;
    let p = (cell_uv - center) / size;
    // Subtle per-flake rotation (hashed phase). Each flake has
    // a fixed orientation; no time-spin to keep things calm.
    let angle = jitter.x * TAU;
    let shape_a = fractal_dendrite(p, angle);

    let near_tint = vec3<f32>(1.00, 1.00, 0.98);
    let far_tint  = vec3<f32>(0.80, 0.88, 1.00);
    let tint = mix(near_tint, far_tint, depth);

    let bright = mix(1.0, 0.35, depth);
    let alpha = shape_a * bright;
    return vec4<f32>(tint * alpha, alpha);
}

// ── one parallax layer — pure +y gravity ───────────────────────

fn snow_layer(uv: vec2<f32>, layer_idx: f32, layers_total: f32, t: f32) -> vec4<f32> {
    let depth = layer_idx / max(layers_total, 1.0);
    let scale = DENSITY * mix(0.9, 2.2, depth);

    // Near falls faster than far in screen-space (parallax: same
    // real velocity, closer object covers more visual field).
    let speed = mix(0.10, 0.05, depth);

    // Subtract t * speed so the tile grid scrolls UP relative to
    // the viewport → each flake appears to FALL DOWN through it.
    var tiled = uv * scale;
    tiled.x = tiled.x + layer_idx * 3.7;
    tiled.y = tiled.y - t * speed - layer_idx * 7.3;

    let cell_id = floor(tiled);
    let cell_uv = fract(tiled);
    let jitter = hash22(cell_id);
    return render_flake(cell_uv, jitter, depth);
}

// ── pile particles ─────────────────────────────────────────────
//
// Below the pile-surface contour, render a dense static grid
// of the same flake primitive packed together. The pile's
// "white floor" emerges from particle density rather than from
// a painted body fill.
fn pile_particles(uv: vec2<f32>, t: f32) -> vec4<f32> {
    let acc = P.params.x;
    if (acc <= 0.0) { return vec4<f32>(0.0); }

    let pile_height = acc * 0.25;
    let line_y = 1.0 - pile_height;
    // Subtle one-octave contour so the pile top isn't a perfect
    // line. Very slow x-drift (t * 0.005) keeps the surface
    // alive.
    let contour = vnoise(vec2<f32>(uv.x * 3.0, t * 0.005)) * 0.018;
    let surface = line_y - contour;
    if (uv.y < surface) { return vec4<f32>(0.0); }

    let temperature = clamp(P.params.z, 0.0, 1.0);
    let melt = max(temperature - 0.5, 0.0) * 2.0;  // 0..1

    // Dense static particle grid (PILE_DENSITY ≈ 32 vs DENSITY
    // ≈ 9 for falling). Aspect-corrected so cells are square.
    let aspect = P.resolution.x / max(P.resolution.y, 1.0);
    var tiled = vec2<f32>(uv.x * aspect, uv.y) * PILE_DENSITY;
    let cell_id = floor(tiled);
    let cell_uv = fract(tiled);
    let jitter = hash22(cell_id);

    // 20% of cells empty inside the pile for natural texture.
    if (jitter.x < 0.20) { return vec4<f32>(0.0); }

    let size = mix(0.30, 0.55, jitter.y);
    let center = vec2<f32>(0.5) + (hash22(jitter * 11.0) - vec2<f32>(0.5)) * 0.25;
    let p = (cell_uv - center) / size;
    let angle = jitter.x * TAU;
    let shape_a = fractal_dendrite(p, angle);

    // Cold = bright white. Warm = cool-blue translucent (melt
    // appearance). Pile particles don't change density with
    // temperature — only color — so melting feels gradual.
    let cold_tint = vec3<f32>(0.96, 0.98, 1.02);
    let warm_tint = vec3<f32>(0.65, 0.78, 0.95);
    let tint = mix(cold_tint, warm_tint, melt);

    // Pile particles are more opaque than falling flakes —
    // they're "settled" so they don't twinkle as much.
    let alpha = shape_a * 0.75 * (1.0 - melt * 0.3);
    return vec4<f32>(tint * alpha, alpha);
}

// ── final grade ────────────────────────────────────────────────

fn grade(rgb: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let d = distance(uv, vec2<f32>(0.5));
    let v = 1.0 - smoothstep(0.78, 1.12, d) * 0.10;
    return rgb * v;
}

// ── main ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Scene sampled FIRST, in trivially uniform control flow —
    // the layer loop below has uniform bounds but sampling early
    // keeps naga's uniformity analysis unambiguous.
    let scene = textureSample(input_tex, input_samp, in.uv);

    let uv = in.uv;
    let aspect = P.resolution.x / max(P.resolution.y, 1.0);
    let auv = vec2<f32>(uv.x * aspect, uv.y);
    let t = P.frame.x;

    var accum = vec4<f32>(0.0);

    let layer_count = clamp(P.params.y, 1.0, LAYERS_MAX);
    var li: f32 = 0.0;
    loop {
        if (li >= layer_count) { break; }
        let layer_idx = layer_count - 1.0 - li;
        let f = snow_layer(auv, layer_idx, layer_count, t);
        let a = f.a;
        let safe = max(f.a, 0.0001);
        let contrib = f.rgb * (a / safe);
        accum = vec4<f32>(
            accum.rgb * (1.0 - a) + contrib,
            1.0 - (1.0 - accum.a) * (1.0 - a),
        );
        li = li + 1.0;
    }

    // Pile = particles (not painted band).
    let pile = pile_particles(uv, t);
    accum = vec4<f32>(
        accum.rgb * (1.0 - pile.a) + pile.rgb,
        1.0 - (1.0 - accum.a) * (1.0 - pile.a),
    );

    let intensity = P.frame.y;
    let scaled_alpha = min(accum.a * intensity, MAX_ALPHA);
    accum = vec4<f32>(accum.rgb * intensity, scaled_alpha);

    let graded = grade(accum.rgb, uv);
    // In-shader premultiplied-over composite (delta 2 in the
    // header): graded is already alpha-weighted, so the scene
    // contributes the remaining (1 - alpha).
    let rgb = scene.rgb * (1.0 - scaled_alpha) + graded;
    return vec4<f32>(rgb, scene.a);
}
