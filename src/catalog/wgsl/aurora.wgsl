// Catalog fragment — aurora (the Borealis signature curtain).
//
// Slow-drifting vertical light bands with noise-driven shimmer,
// composited over the scene at low opacity and concentrated
// toward the TOP of the frame (sky above a horizon line — the
// prompt area below the horizon is never touched).
//
// Craft contract (each bullet is anchored by a unit test in
// src/catalog/aurora.rs):
//
//   * QUALITY TIERS — P.tier.x selects the cost/beauty point:
//       0 Off    — early-out: the scene passes through
//                  byte-exact. The pass still costs one
//                  fullscreen textureSample + copy (~a blit);
//                  consumers wanting TRUE zero cost omit the
//                  node from the graph instead.
//       1 Low    — single-octave value-noise curtain, cheap mix.
//       2 Medium — 3-octave fbm curtain + one-octave shimmer.
//       3 High   — 4-octave fbm + vertical ray-march shimmer
//                  (MARCH_STEPS fold-displaced samples).
//     Out-of-contract quality words (>= 4) degrade to
//     pass-through — the same posture as colorblind's default
//     arm: a Pod-bytes-minted word must never invent a tier.
//
//   * BANDING-FREE — a spatial (frame-stable) hash dither at
//     ±0.5/255 is added where the aurora contributes, breaking
//     8-bit banding in the slow alpha gradients. The dither has
//     NO time term: it is identical every frame, so it cannot
//     introduce temporal noise. It is gated by alpha so pixels
//     the aurora does not touch stay byte-exact.
//
//   * TEMPORAL STABILITY — drift is continuous time through
//     C1-continuous value noise (smoothstep-interpolated) with
//     no fract()/wrap discontinuities and no per-frame random
//     phase, so slow drift speeds cannot strobe or pop.
//
//   * PREMULTIPLIED OVER — the composite happens in-shader
//     (scene.rgb * (1 - a) + curtain_premul), because
//     WgpuDispatcher pipelines are blend-free ping-pong passes
//     (same delta as the snow absorption).

struct AuroraParams {
    frame: vec4<f32>,        // (time_seconds, intensity, drift, shimmer)
    geometry: vec4<f32>,     // (horizon, band_scale, res_x, res_y)
    color_green: vec4<f32>,  // curtain base — oxygen 557.7 nm (linear rgb, w unused)
    color_cyan: vec4<f32>,   // curtain mid  — high-altitude cyan
    color_violet: vec4<f32>, // curtain edge — nitrogen violet
    tier: vec4<u32>,         // (quality, _, _, _)
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
@group(0) @binding(2) var<uniform> P: AuroraParams;

const QUALITY_OFF: u32 = 0u;
const QUALITY_LOW: u32 = 1u;
const QUALITY_MEDIUM: u32 = 2u;
const QUALITY_HIGH: u32 = 3u;

// Hard ceiling on aurora coverage — the scene always reads
// through the curtain (it is sky dressing, not content).
const MAX_ALPHA: f32 = 0.5;
// High-tier vertical ray-march sample count.
const MARCH_STEPS: i32 = 12;
// Curtain lower border: sharp ramp-in width + upward decay rate
// (real curtains have a crisp bottom edge and a diffuse top).
const BORDER_SOFT: f32 = 0.06;
const DECAY_RATE: f32 = 2.6;

// ── hashing & noise ────────────────────────────────────────────

fn hash12(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(0.1031, 0.1030));
    q = q + dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x);
}

// C1-continuous value noise — the smoothstep fade keeps the
// derivative continuous so slow drift cannot shear or pop.
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

// fbm with a runtime octave count. The count derives from the
// quality uniform, so control flow stays uniform.
fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i: i32 = 0; i < octaves; i = i + 1) {
        v = v + amp * vnoise(q);
        // Non-integer lacunarity + offset de-correlates octaves
        // (no lattice-alignment artifacts).
        q = q * 2.03 + vec2<f32>(19.7, 7.3);
        amp = amp * 0.5;
    }
    return v;
}

// ── High tier: vertical ray-march shimmer ─────────────────────
//
// Walks up the curtain accumulating fold-displaced noise: the
// fold field bends the sampling column (curtain pleats), the
// inner noise reads ray structure at ~9x band frequency. Lower
// samples dominate (1 - fi * 0.7 weight) so the bottom border
// stays the brightest — matching real auroral curtains.
fn ray_factor(xb: f32, drift_t: f32) -> f32 {
    var acc = 0.0;
    var wsum = 0.0;
    for (var i: i32 = 0; i < MARCH_STEPS; i = i + 1) {
        let fi = f32(i) / f32(MARCH_STEPS);
        let fold = (fbm(vec2<f32>(xb * 0.35 + fi * 0.8, drift_t * 0.4), 2) - 0.5) * 1.4;
        let s = vnoise(vec2<f32>(xb * 9.0 + fold * 3.0, fi * 2.0 - drift_t * 1.7));
        let wt = 1.0 - fi * 0.7;
        acc = acc + s * wt;
        wsum = wsum + wt;
    }
    let r = acc / max(wsum, 0.0001);
    // Contrasty streak mask centred near 1.0 so shimmer
    // modulates the curtain instead of dimming it.
    return 0.55 + 0.9 * smoothstep(0.35, 0.75, r);
}

// ── main ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Scene sampled FIRST, in trivially uniform control flow,
    // before any branching (naga uniformity analysis).
    let scene = textureSample(input_tex, input_samp, in.uv);

    let quality = P.tier.x;
    // Off-equivalent early-out + out-of-contract degrade: both
    // return the scene byte-exact (no dither, no grade).
    if (quality == QUALITY_OFF || quality > QUALITY_HIGH) {
        return scene;
    }

    let horizon = clamp(P.geometry.x, 0.05, 1.0);
    // Sky altitude: 1 at the top of the frame, 0 at the horizon
    // line, negative below it (no aurora over the prompt area).
    let alt = 1.0 - in.uv.y / horizon;
    if (alt <= 0.0) {
        return scene;
    }

    let aspect = P.geometry.z / max(P.geometry.w, 1.0);
    let xb = in.uv.x * aspect * P.geometry.y;
    let drift_t = P.frame.x * 0.02 * P.frame.z;
    let shimmer = clamp(P.frame.w, 0.0, 1.0);

    // The lower border wanders slowly per-column — one cheap
    // octave at every tier; the wander is what sells the curtain.
    let border = 0.08 + 0.22 * vnoise(vec2<f32>(xb * 0.31 + drift_t * 0.5, drift_t * 0.23));
    let rel = alt - border;
    if (rel <= 0.0) {
        return scene;
    }
    // Sharp lower edge, diffuse upward decay.
    let profile = smoothstep(0.0, BORDER_SOFT, rel) * exp(-rel * DECAY_RATE);

    // Curtain density by tier (normalised to ~[0, 1]).
    let curtain_p = vec2<f32>(xb + drift_t, alt * 1.2 + drift_t * 0.31);
    var density: f32;
    if (quality == QUALITY_LOW) {
        density = vnoise(curtain_p);
    } else if (quality == QUALITY_MEDIUM) {
        density = fbm(curtain_p, 3) / 0.875;
    } else {
        density = fbm(curtain_p, 4) / 0.9375;
    }
    var curtain = smoothstep(0.30, 0.78, density);
    curtain = curtain * curtain;

    // Shimmer: High pays the ray march; Medium one extra octave
    // of column noise; Low none — that is the budget.
    if (quality == QUALITY_HIGH) {
        curtain = curtain * mix(1.0, ray_factor(xb, drift_t), shimmer);
    } else if (quality == QUALITY_MEDIUM) {
        let cheap = 0.7 + 0.6 * vnoise(vec2<f32>(xb * 7.0, drift_t * 1.3));
        curtain = curtain * mix(1.0, cheap, shimmer * 0.6);
    }

    let intensity = clamp(P.frame.y, 0.0, 1.0);
    let alpha = min(profile * curtain * intensity, MAX_ALPHA);

    // Altitude-keyed spectrum: oxygen green at the lower border,
    // cyan through the middle, nitrogen violet at the high edge.
    let th = clamp(rel * 1.6, 0.0, 1.0);
    var col = mix(P.color_green.rgb, P.color_cyan.rgb, smoothstep(0.0, 0.55, th));
    col = mix(col, P.color_violet.rgb, smoothstep(0.5, 1.0, th));

    // Spatial (frame-stable) dither, gated to where the aurora
    // contributes so untouched pixels stay byte-exact.
    let dith = (hash12(in.pos.xy) - 0.5) / 255.0;
    let dith_gate = smoothstep(0.0, 0.015, alpha);

    // Premultiplied over.
    let rgb = scene.rgb * (1.0 - alpha) + col * alpha + vec3<f32>(dith * dith_gate);
    return vec4<f32>(rgb, scene.a);
}
