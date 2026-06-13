//! L2 GPU test — catalog end-to-end on a real adapter: a
//! pool-leased scene texture → the colorblind catalog graph →
//! `HeadlessTarget`, with `FrameUniforms` carrying the mode.
//!
//! Mode `None` must pass pure green through; `Protanopia` must
//! transform pure green per the Machado matrix (r' = 1.052583
//! clamps to 1.0 → the red channel saturates), pinning the
//! verbatim port at the pixel level.

#![cfg(feature = "gpu_tests")]

use engawa_wgpu::catalog::{colorblind, CatalogEffect, CATALOG_SAMPLER, OUT, SCENE};
use engawa_wgpu::{
    BoundResource, BoundResources, FrameUniforms, TextureKey, TexturePool, WgpuDispatcher,
};
use garasu::headless::HeadlessTarget;
use garasu::GpuContext;

const W: u32 = 64;
const H: u32 = 64;

fn run_colorblind(params: colorblind::ColorblindParams) -> Vec<u8> {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let target = HeadlessTarget::new(&gpu, W, H, format);

    let mut pool = TexturePool::new();
    let scene_lease = pool.lease(&gpu.device, TextureKey::offscreen(W, H, format));

    // Paint the scene pure green.
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
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }),
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
        label: Some("colorblind params"),
        size: CatalogEffect::Colorblind.params_size() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("catalog sampler"),
        ..wgpu::SamplerDescriptor::default()
    });

    let graph = CatalogEffect::Colorblind
        .graph()
        .compile()
        .expect("catalog graph compiles");

    let bindings = engawa::ResourceBindings::new()
        .with(SCENE, engawa::ResourceHandle::Texture(SCENE.into()))
        .with(OUT, engawa::ResourceHandle::Texture(OUT.into()));
    let bound = BoundResources::new()
        .with(SCENE, scene_lease.bound_resource())
        .with(OUT, BoundResource::Texture { view: target.view().clone(), format })
        .with(CATALOG_SAMPLER, BoundResource::Sampler(sampler))
        .with(colorblind::PARAMS_RESOURCE, BoundResource::Uniform(params_buf));
    let frame = FrameUniforms::new().with(colorblind::PARAMS_RESOURCE, &params);

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
/// dispatch a batch of frames at Low / Medium / High on the real
/// adapter, print the aggregate timings (informational), and
/// assert the monotone cost ordering low <= med <= high.
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
fn aurora_tier_cost_is_monotone_low_med_high() {
    use std::time::Instant;

    use engawa_wgpu::catalog::aurora::{self, AuroraQuality};

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

    let low_ms = time_tier(AuroraQuality::Low);
    let med_ms = time_tier(AuroraQuality::Medium);
    let high_ms = time_tier(AuroraQuality::High);

    eprintln!(
        "aurora perf smoke ({FRAMES} frames @ {PW}x{PH}): \
         low={low_ms:.2}ms ({:.3}ms/frame)  \
         med={med_ms:.2}ms ({:.3}ms/frame)  \
         high={high_ms:.2}ms ({:.3}ms/frame)",
        low_ms / FRAMES as f64,
        med_ms / FRAMES as f64,
        high_ms / FRAMES as f64,
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
