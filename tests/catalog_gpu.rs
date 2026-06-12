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

fn run_colorblind(mode: colorblind::ColorblindMode) -> Vec<u8> {
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
    let frame = FrameUniforms::new()
        .with(colorblind::PARAMS_RESOURCE, &colorblind::ColorblindParams::new(mode));

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

#[test]
fn colorblind_mode_none_passes_green_through() {
    let pixels = run_colorblind(colorblind::ColorblindMode::None);
    let [r, g, b, _] = center_pixel(&pixels);
    assert!(r < 30, "mode none must keep red dark, got R={r}");
    assert!(g > 200, "mode none must keep green bright, got G={g}");
    assert!(b < 30, "mode none must keep blue dark, got B={b}");
}

#[test]
fn colorblind_protanopia_transforms_pure_green_per_machado() {
    let pixels = run_colorblind(colorblind::ColorblindMode::Protanopia);
    let [r, g, b, _] = center_pixel(&pixels);
    // Pure green through the protanopia matrix: r' = 1.052583
    // (clamps to 1.0), g' = 0.786281, b' = -0.048116 (clamps to
    // 0.0) — the red channel saturating is the fingerprint.
    assert!(r > 200, "protanopia must saturate red for pure green, got R={r}");
    assert!(g > 150, "protanopia keeps most green, got G={g}");
    assert!(b < 30, "protanopia clamps blue to zero, got B={b}");
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
    // 5 single-material effects + bloom's 4 stages = 9 distinct
    // pipelines, each compiled exactly once.
    assert_eq!(dispatcher.cached_pipeline_count(), 9);
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
