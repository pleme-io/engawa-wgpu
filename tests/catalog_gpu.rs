//! L2 GPU test — catalog end-to-end on a real adapter: a
//! pool-leased scene texture → a catalog graph → `HeadlessTarget`,
//! with `FrameUniforms` carrying the params, and the rendered
//! BYTES read back and asserted (pixel proofs, not WGSL reading).
//!
//! Colorblind: mode `None` must pass pure green through;
//! `Protanopia` must transform pure green per the Machado matrix
//! (r' = 1.052583 clamps to 1.0 → the red channel saturates),
//! pinning the verbatim port at the pixel level.
//!
//! Aurora: `Medium` must visibly draw the curtain above the
//! horizon while the prompt area below it stays scene byte-exact;
//! `Off` and out-of-contract tier words must return the scene —
//! the module's byte-exact pass-through claims, proven on pixels.

#![cfg(feature = "gpu_tests")]

use engawa_wgpu::catalog::{
    aurora::{self, AuroraQuality},
    colorblind, CatalogEffect, CATALOG_SAMPLER, OUT, SCENE,
};
use engawa_wgpu::{
    BoundResource, BoundResources, FrameUniforms, TextureKey, TexturePool, WgpuDispatcher,
};
use garasu::headless::HeadlessTarget;
use garasu::GpuContext;

const W: u32 = 64;
const H: u32 = 64;

/// Dispatch one single-node catalog effect over a uniformly
/// cleared scene texture and read the target pixels back — the
/// shared pixel-proof harness (colorblind + aurora both run
/// through it): pool lease → scene clear → graph compile →
/// dispatch → readback.
fn run_catalog_effect<P: bytemuck::Pod>(
    effect: CatalogEffect,
    scene_clear: wgpu::Color,
    params: &P,
) -> Vec<u8> {
    assert!(
        effect.aux_resources().is_empty(),
        "pixel harness covers single-node effects only; {} declares aux resources",
        effect.name()
    );
    assert_eq!(
        size_of::<P>(),
        effect.params_size(),
        "params type must match {}'s declared params size",
        effect.name()
    );

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let target = HeadlessTarget::new(&gpu, W, H, format);

    let mut pool = TexturePool::new();
    let scene_lease = pool.lease(&gpu.device, TextureKey::offscreen(W, H, format));

    // Paint the scene with the caller's flat color.
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("catalog-gpu scene clear"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_lease.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(scene_clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(effect.params_resource()),
        size: effect.params_size() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("catalog sampler"),
        ..wgpu::SamplerDescriptor::default()
    });

    let graph = effect.graph().compile().expect("catalog graph compiles");

    let bindings = engawa::ResourceBindings::new()
        .with(SCENE, engawa::ResourceHandle::Texture(SCENE.into()))
        .with(OUT, engawa::ResourceHandle::Texture(OUT.into()));
    let bound = BoundResources::new()
        .with(SCENE, scene_lease.bound_resource())
        .with(OUT, BoundResource::Texture { view: target.view().clone(), format })
        .with(CATALOG_SAMPLER, BoundResource::Sampler(sampler))
        .with(effect.params_resource(), BoundResource::Uniform(params_buf));
    let frame = FrameUniforms::new().with(effect.params_resource(), params);

    let mut dispatcher = WgpuDispatcher::new(&gpu.device, &gpu.queue, format);
    let cmd = dispatcher
        .dispatch_with(&graph, &bindings, bound, &frame)
        .expect("dispatch");
    gpu.queue.submit(std::iter::once(cmd));
    let _ = gpu.device.poll(wgpu::PollType::Wait);

    pool.release(scene_lease);
    assert_eq!(pool.free_count(), 1, "released lease must land in the free list");

    target.read_pixels_rgba8(&gpu)
}

fn run_colorblind(params: colorblind::ColorblindParams) -> Vec<u8> {
    // Pure green scene — the Machado expectations derive from it.
    run_catalog_effect(
        CatalogEffect::Colorblind,
        wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },
        &params,
    )
}

fn center_pixel(pixels: &[u8]) -> [u8; 4] {
    let i = ((H / 2 * W + W / 2) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// sRGB encode (linear → stored byte) — what the `Rgba8UnormSrgb`
/// attachment does to the shader's linear output. The expected
/// pixel values below are DERIVED from the Machado constants
/// through this function, never hand-literals: a blend/gamma-space
/// regression in the graded route (e.g. a non-sRGB SCENE view
/// losing the decode) shifts the stored bytes far past the ±2
/// rounding tolerance and fails here (M3 review 2026-06-12 — the
/// prior r>200/g>150 thresholds passed a sRGB-space-applied
/// matrix).
// The clamp to [0,1] bounds s*255 within u8 range; the cast is the
// storage conversion itself.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn srgb_encode_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn assert_pixel_close(got: [u8; 4], expected_rgb: [u8; 3], label: &str) {
    let mut failures: Vec<(&str, u8, u8)> = Vec::new();
    for (channel, (g, e)) in ["R", "G", "B"].iter().zip(got.iter().zip(expected_rgb.iter())) {
        if (i32::from(*g) - i32::from(*e)).abs() > 2 {
            failures.push((channel, *g, *e));
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: channels off by more than ±2 (channel, got, expected): {failures:?}"
    );
}

#[test]
fn colorblind_mode_none_passes_green_through() {
    let pixels = run_colorblind(colorblind::ColorblindParams::new(
        colorblind::ColorblindMode::None,
    ));
    // Pass-through is byte-exact modulo rounding: pure green in,
    // pure green stored.
    assert_pixel_close(center_pixel(&pixels), [0, 255, 0], "mode none");
}

#[test]
fn colorblind_protanopia_transforms_pure_green_per_machado() {
    let pixels = run_colorblind(colorblind::ColorblindParams::new(
        colorblind::ColorblindMode::Protanopia,
    ));
    // Pure green selects the matrix's middle column; expected
    // stored bytes derive from the SAME constants the shader pins
    // (r' = 1.052583 clamps to 1.0, g' = 0.786281 → sRGB-encoded,
    // b' = -0.048116 clamps to 0.0).
    let expected = [
        srgb_encode_u8(colorblind::MACHADO_PROTANOPIA[0][1]),
        srgb_encode_u8(colorblind::MACHADO_PROTANOPIA[1][1]),
        srgb_encode_u8(colorblind::MACHADO_PROTANOPIA[2][1]),
    ];
    assert_pixel_close(center_pixel(&pixels), expected, "protanopia");
}

#[test]
fn out_of_contract_mode_word_degrades_to_pass_through() {
    // The Pod bytes ingress mints a mode word the constructor never
    // produces; the WGSL's explicit default arm must render it as
    // pass-through — the former catch-all silently simulated
    // Tritanopia for every word >= 3.
    let params: colorblind::ColorblindParams = bytemuck::cast([7_u32, 0, 0, 0]);
    let pixels = run_colorblind(params);
    assert_pixel_close(center_pixel(&pixels), [0, 255, 0], "mode 7 (out of contract)");
}

// ── aurora pixel proofs ───────────────────────────────────────
//
// The WGSL-reading asserts in src/catalog/aurora.rs pin the
// contract's TEXT; these pin its PIXELS — a regression that
// zeroes alpha (broken horizon/border math sending every pixel
// down the `return scene` paths) passes the dispatch matrix and
// the perf smoke (the ALU still runs) but fails here.

/// Dark-navy "terminal scene" the aurora proofs composite over —
/// dark enough that the curtain's green/cyan reads as a large
/// byte delta, and a genuine mid-tone so the byte-exact
/// pass-through claims exercise a real sRGB round trip (not a
/// trivial 0→0 / 255→255).
const AURORA_SCENE: wgpu::Color = wgpu::Color { r: 0.02, g: 0.03, b: 0.08, a: 1.0 };

/// Shipped defaults (Medium tier, horizon 0.62) at the test
/// resolution — the exact shape mado wires up.
fn aurora_test_params() -> aurora::AuroraParams {
    #[allow(clippy::cast_precision_loss)]
    aurora::AuroraParams::default().with_resolution([W as f32, H as f32])
}

fn run_aurora(params: aurora::AuroraParams) -> Vec<u8> {
    run_catalog_effect(CatalogEffect::Aurora, AURORA_SCENE, &params)
}

/// First buffer row whose pixel-centre uv.y sits at or below the
/// horizon — the shader's `alt <= 0` early-out region (the prompt
/// area). Rows `first_below_horizon_row()..H` must be scene
/// byte-exact under every tier.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn first_below_horizon_row(horizon: f32) -> u32 {
    // uv.y = (y + 0.5) / H  >=  horizon  ⇔  y >= horizon*H - 0.5.
    // At 64px / 0.62 this is row 40 (uv.y = 0.6328) — a ~0.013 uv
    // margin over the boundary, far beyond interpolation error.
    (horizon * H as f32 - 0.5).ceil() as u32
}

/// `Off` is the rebuild-free kill-switch: the pass must render
/// the scene, proven against ground truth derived from the clear
/// color through the same sRGB encode the attachment performs.
/// (This also dispatches the `Off` tier on a real adapter — the
/// perf smoke covers the three live tiers.)
#[allow(clippy::cast_possible_truncation)]
#[test]
fn aurora_off_tier_renders_the_scene() {
    let pixels = run_aurora(aurora_test_params().with_quality(AuroraQuality::Off));
    let expected = [
        srgb_encode_u8(AURORA_SCENE.r as f32),
        srgb_encode_u8(AURORA_SCENE.g as f32),
        srgb_encode_u8(AURORA_SCENE.b as f32),
    ];
    let mut off_pixels = 0usize;
    let mut first: Option<(u32, u32, [u8; 4])> = None;
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            let got = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
            let close = got[..3]
                .iter()
                .zip(expected.iter())
                .all(|(g, e)| (i32::from(*g) - i32::from(*e)).abs() <= 2)
                && got[3] == 255;
            if !close {
                off_pixels += 1;
                first.get_or_insert((x, y, got));
            }
        }
    }
    assert_eq!(
        off_pixels, 0,
        "Off tier must render the scene everywhere: {off_pixels} pixels off \
         (first at {first:?}, expected ~{expected:?})"
    );
}

/// A Pod-bytes-minted tier word the typed surface never produces
/// must take the same `return scene` path as `Off` — byte-exact
/// equality of the two full buffers (same shader path ⇒ identical
/// bytes; any divergence means word >= 4 invented a tier).
#[test]
fn aurora_out_of_contract_tier_word_matches_off_byte_exact() {
    let off = run_aurora(aurora_test_params().with_quality(AuroraQuality::Off));
    let rogue = aurora::AuroraParams { tier: [7, 0, 0, 0], ..aurora_test_params() };
    let pixels = run_aurora(rogue);
    let diff_bytes = pixels.iter().zip(off.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff_bytes, 0,
        "tier word 7 must be byte-identical to Off pass-through; {diff_bytes} bytes differ"
    );
}

/// The shipped default (Medium) must actually draw: above the
/// horizon the curtain visibly changes pixels; below it (the
/// prompt area) every pixel stays scene byte-exact — compared
/// against the Off pass-through of the identical scene.
#[allow(clippy::cast_possible_truncation)]
#[test]
fn aurora_medium_draws_above_the_horizon_and_never_below() {
    let params = aurora_test_params();
    assert_eq!(params.quality(), Some(AuroraQuality::Medium), "shipped default tier");
    let off = run_aurora(aurora_test_params().with_quality(AuroraQuality::Off));
    let med = run_aurora(params);

    let below_start = (first_below_horizon_row(params.geometry[0]) * W * 4) as usize;

    // (a) Prompt area: byte-exact scene (identical to the Off
    // pass-through bytes — same `return scene` path).
    let below_diff = med[below_start..]
        .iter()
        .zip(off[below_start..].iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        below_diff, 0,
        "below-horizon pixels must be scene byte-exact; {below_diff} bytes differ"
    );

    // (b) Sky: the curtain draws. Count differing pixels and the
    // max channel delta over the above-horizon region; the delta
    // floor (16) is far above dither magnitude (±0.5/255), so a
    // pass requires actual curtain color, not dither residue.
    let mut diff_pixels = 0usize;
    let mut max_delta = 0i32;
    for (m, o) in med[..below_start].chunks_exact(4).zip(off[..below_start].chunks_exact(4)) {
        let delta = m
            .iter()
            .zip(o.iter())
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap_or(0);
        if delta > 0 {
            diff_pixels += 1;
        }
        max_delta = max_delta.max(delta);
    }
    assert!(
        diff_pixels > 0,
        "Medium must change at least one above-horizon pixel — the curtain never drew"
    );
    assert!(
        max_delta >= 16,
        "curtain contribution too weak to be the curtain (max channel delta {max_delta}, \
         {diff_pixels} pixels differ) — alpha is collapsing toward zero"
    );
    eprintln!(
        "aurora medium pixel proof: {diff_pixels} above-horizon pixels differ, \
         max channel delta {max_delta}"
    );
}

#[test]
fn every_catalog_effect_dispatches_on_a_real_adapter() {
    // The shader-level matrix: every effect's WGSL goes through
    // naga + a real pipeline build + one dispatch. A WGSL typo in
    // ANY catalog shader fails here, not at a consumer's first
    // frame. Failures aggregate per effect before asserting.
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let target = HeadlessTarget::new(&gpu, W, H, format);
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("catalog sampler"),
        ..wgpu::SamplerDescriptor::default()
    });
    let mut pool = TexturePool::new();
    let scene_lease = pool.lease(&gpu.device, TextureKey::offscreen(W, H, format));
    let mut dispatcher = WgpuDispatcher::new(&gpu.device, &gpu.queue, format);

    let mut failures: Vec<(&'static str, String)> = Vec::new();
    for e in CatalogEffect::ALL {
        let graph = match e.graph().compile() {
            Ok(g) => g,
            Err(err) => {
                failures.push((e.name(), err.to_string()));
                continue;
            }
        };

        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(e.params_resource()),
            size: e.params_size() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&params_buf, 0, &e.default_params_bytes());

        let mut bindings = engawa::ResourceBindings::new()
            .with(SCENE, engawa::ResourceHandle::Texture(SCENE.into()))
            .with(OUT, engawa::ResourceHandle::Texture(OUT.into()));
        let mut bound = BoundResources::new()
            .with(SCENE, scene_lease.bound_resource())
            .with(OUT, BoundResource::Texture { view: target.view().clone(), format })
            .with(CATALOG_SAMPLER, BoundResource::Sampler(sampler.clone()))
            .with(e.params_resource(), BoundResource::Uniform(params_buf));
        let mut aux_leases = Vec::new();
        for (id, _kind) in e.aux_resources() {
            bindings.insert(id, engawa::ResourceHandle::Texture(id.into()));
            let lease = pool.lease(&gpu.device, TextureKey::offscreen(W, H, format));
            bound.insert(id, lease.bound_resource());
            aux_leases.push(lease);
        }

        match dispatcher.dispatch_with(&graph, &bindings, bound, &FrameUniforms::new()) {
            Ok(cmd) => {
                gpu.queue.submit(std::iter::once(cmd));
                let _ = gpu.device.poll(wgpu::PollType::Wait);
            }
            Err(err) => failures.push((e.name(), err.to_string())),
        }
        for lease in aux_leases {
            pool.release(lease);
        }
    }

    assert!(
        failures.is_empty(),
        "{} catalog effects failed to dispatch:\n{:#?}",
        failures.len(),
        failures
    );
    // 6 single-material effects + bloom's 4 stages = 10 distinct
    // pipelines, each compiled exactly once.
    assert_eq!(dispatcher.cached_pipeline_count(), 10);
}

#[test]
fn texture_pool_reuses_released_textures_by_key() {
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let key = TextureKey::offscreen(32, 32, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut pool = TexturePool::new();

    let a = pool.lease(&gpu.device, key);
    assert_eq!(pool.free_count(), 0);
    pool.release(a);
    assert_eq!(pool.free_count(), 1);

    // Same key → the freed texture is handed back out.
    let b = pool.lease(&gpu.device, key);
    assert_eq!(pool.free_count(), 0, "matching lease must reuse, not allocate");

    // Different key → fresh allocation, free list untouched.
    pool.release(b);
    let other = pool.lease(
        &gpu.device,
        TextureKey::offscreen(64, 32, wgpu::TextureFormat::Rgba8UnormSrgb),
    );
    assert_eq!(pool.free_count(), 1, "mismatched key must not consume the free list");
    pool.release(other);
    assert_eq!(pool.free_count(), 2);
}

#[test]
fn retain_evicts_stale_size_buckets() {
    // The live-resize discipline: after the surface size changes,
    // the consumer retains only current-size buckets — stale-size
    // textures must not survive (the unbounded-growth leak class).
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut pool = TexturePool::new();

    let old = pool.lease(&gpu.device, TextureKey::offscreen(32, 32, format));
    pool.release(old);
    let new = pool.lease(&gpu.device, TextureKey::offscreen(64, 64, format));
    pool.release(new);
    assert_eq!(pool.free_count(), 2);

    pool.retain(|k| k.width == 64 && k.height == 64);
    assert_eq!(pool.free_count(), 1, "stale 32x32 bucket must be evicted");

    // The surviving bucket still serves leases.
    let reused = pool.lease(&gpu.device, TextureKey::offscreen(64, 64, format));
    assert_eq!(pool.free_count(), 0, "retained texture must be reused, not reallocated");
    pool.release(reused);
}

/// Coarse mechanical perf smoke for the aurora quality tiers:
/// dispatch a batch of frames at Off / Low / Medium / High on the
/// real adapter, print the aggregate timings (informational), and
/// assert the monotone cost ordering off <= low <= med <= high
/// (`Off` costs ~a blit — the rebuild-free kill-switch claim).
///
/// This is an ORDERING proof, not an absolute-ms gate — wall
/// clocks on shared CI adapters are too noisy for absolute
/// budgets, so a slack factor absorbs timer jitter while still
/// failing if a "cheaper" tier ever costs categorically more
/// than a "richer" one (the tier contract inverting).
// The u32→f32 / usize→f32 casts feed shader uniforms + timing
// math where the precision loss is irrelevant.
#[allow(clippy::cast_precision_loss)]
#[test]
fn aurora_tier_cost_is_monotone_off_low_med_high() {
    use std::time::Instant;

    // Big enough that per-pixel ALU dominates the per-frame
    // submit/poll overhead; small enough to stay CI-friendly.
    const PW: u32 = 1024;
    const PH: u32 = 1024;
    const WARMUP: usize = 8;
    const FRAMES: usize = 64;
    /// Timer-noise slack for the monotonicity assertion.
    const SLACK: f64 = 1.25;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let target = HeadlessTarget::new(&gpu, PW, PH, format);
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("catalog sampler"),
        ..wgpu::SamplerDescriptor::default()
    });
    let mut pool = TexturePool::new();
    let scene_lease = pool.lease(&gpu.device, TextureKey::offscreen(PW, PH, format));
    let mut dispatcher = WgpuDispatcher::new(&gpu.device, &gpu.queue, format);
    let graph = CatalogEffect::Aurora
        .graph()
        .compile()
        .expect("aurora graph compiles");
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(aurora::PARAMS_RESOURCE),
        size: CatalogEffect::Aurora.params_size() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut time_tier = |q: AuroraQuality| -> f64 {
        let mut params = aurora::AuroraParams::default()
            .with_resolution([PW as f32, PH as f32])
            .with_quality(q)
            .with_intensity(0.5);
        let mut start = Instant::now();
        for i in 0..(WARMUP + FRAMES) {
            if i == WARMUP {
                // Warmup absorbs pipeline compile + first-frame
                // allocation; the timed window is steady-state.
                start = Instant::now();
            }
            params.set_time(i as f32 / 60.0);
            let bindings = engawa::ResourceBindings::new()
                .with(SCENE, engawa::ResourceHandle::Texture(SCENE.into()))
                .with(OUT, engawa::ResourceHandle::Texture(OUT.into()));
            let bound = BoundResources::new()
                .with(SCENE, scene_lease.bound_resource())
                .with(OUT, BoundResource::Texture { view: target.view().clone(), format })
                .with(CATALOG_SAMPLER, BoundResource::Sampler(sampler.clone()))
                .with(aurora::PARAMS_RESOURCE, BoundResource::Uniform(params_buf.clone()));
            let frame = FrameUniforms::new().with(aurora::PARAMS_RESOURCE, &params);
            let cmd = dispatcher
                .dispatch_with(&graph, &bindings, bound, &frame)
                .expect("aurora dispatch");
            gpu.queue.submit(std::iter::once(cmd));
            // Per-frame wait serialises GPU work so the wall
            // clock actually measures shader cost, not queueing.
            let _ = gpu.device.poll(wgpu::PollType::Wait);
        }
        start.elapsed().as_secs_f64() * 1000.0
    };

    let off_ms = time_tier(AuroraQuality::Off);
    let low_ms = time_tier(AuroraQuality::Low);
    let med_ms = time_tier(AuroraQuality::Medium);
    let high_ms = time_tier(AuroraQuality::High);

    eprintln!(
        "aurora perf smoke ({FRAMES} frames @ {PW}x{PH}): \
         off={off_ms:.2}ms ({:.3}ms/frame)  \
         low={low_ms:.2}ms ({:.3}ms/frame)  \
         med={med_ms:.2}ms ({:.3}ms/frame)  \
         high={high_ms:.2}ms ({:.3}ms/frame)",
        off_ms / FRAMES as f64,
        low_ms / FRAMES as f64,
        med_ms / FRAMES as f64,
        high_ms / FRAMES as f64,
    );

    assert!(
        off_ms <= low_ms * SLACK,
        "tier cost inversion: Off ({off_ms:.2}ms) costs more than Low ({low_ms:.2}ms) × slack"
    );
    assert!(
        low_ms <= med_ms * SLACK,
        "tier cost inversion: Low ({low_ms:.2}ms) costs more than Medium ({med_ms:.2}ms) × slack"
    );
    assert!(
        med_ms <= high_ms * SLACK,
        "tier cost inversion: Medium ({med_ms:.2}ms) costs more than High ({high_ms:.2}ms) × slack"
    );
}
