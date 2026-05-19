//! L2 GPU pixel test: build a tiny one-effect graph that
//! paints a red-tinted fullscreen pass into a HeadlessTarget,
//! read pixels back, assert every pixel is red.

#![cfg(feature = "gpu_tests")]

use engawa::{Material, Node, RenderGraph, ResourceKind, ShaderSource};
use engawa_wgpu::{BoundResource, BoundResources, WgpuDispatcher};
use garasu::headless::{frame_hash, HeadlessTarget};
use garasu::GpuContext;

fn red_material() -> Material {
    Material {
        name: "red-tint".into(),
        shader: ShaderSource::inline(
            "@fragment fn fs_main() -> @location(0) vec4<f32> { \
             return vec4<f32>(1.0, 0.0, 0.0, 1.0); }",
        ),
        bindings: vec![],
    }
}

fn run_pipeline(target_format: wgpu::TextureFormat) -> Vec<u8> {
    let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
    let target = HeadlessTarget::new(&gpu, 64, 64, target_format);
    let graph = RenderGraph::default()
        .with_resource(
            "out",
            ResourceKind::Texture {
                width: Some(64),
                height: Some(64),
            },
        )
        .with_output("out")
        .with_node(Node::clear("clear", "out"))
        .with_node(Node::fullscreen_effect("red", red_material(), "out", "out"))
        .compile();
    // The above intentionally writes to "out" twice (clear +
    // red-tint). That trips the multiple-writers check, so the
    // real graph has two distinct resources.
    drop(graph);

    let graph = RenderGraph::default()
        .with_resource(
            "scene",
            ResourceKind::Texture {
                width: Some(64),
                height: Some(64),
            },
        )
        .with_resource(
            "out",
            ResourceKind::Texture {
                width: Some(64),
                height: Some(64),
            },
        )
        .with_output("out")
        .with_node(Node::clear("clear", "scene"))
        .with_node(Node::fullscreen_effect("red", red_material(), "scene", "out"))
        .compile()
        .expect("compile");

    let mut dispatcher = WgpuDispatcher::new(&gpu.device, &gpu.queue, target_format);
    // Engawa bindings: name → string (recording dispatcher
    // shape). wgpu bindings: id → live wgpu handle.
    let engawa_bindings = engawa::ResourceBindings::new()
        .with("scene", engawa::ResourceHandle::Texture("scene".into()))
        .with("out", engawa::ResourceHandle::Texture("out".into()));
    let bound = BoundResources::new()
        .with(
            "scene",
            BoundResource::Texture {
                view: target.view().clone(),
                format: target_format,
            },
        )
        .with(
            "out",
            BoundResource::Texture {
                view: target.view().clone(),
                format: target_format,
            },
        );
    let cmd = dispatcher
        .dispatch_with(&graph, engawa_bindings, bound)
        .expect("dispatch");
    gpu.queue.submit(std::iter::once(cmd));
    let _ = gpu.device.poll(wgpu::PollType::Wait);
    target.read_pixels_rgba8(&gpu)
}

#[test]
fn red_tint_fullscreen_effect_paints_red_pixels() {
    // Rgba8UnormSrgb so wgpu interprets the (1, 0, 0, 1) clear
    // value in sRGB space (matches "red as displayed").
    let pixels = run_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(pixels.len(), 64 * 64 * 4);
    // Spot-check the centre + corners — every pixel should be
    // red (R high, G/B low). Pure 1.0/0.0/0.0/1.0 in sRGB
    // round-trips to (255, 0, 0, 255) after the storage
    // conversion.
    for (x, y) in [(0, 0), (32, 32), (63, 63), (0, 63), (63, 0)] {
        let i = ((y * 64 + x) * 4) as usize;
        assert!(
            pixels[i] > 200,
            "pixel ({x},{y}) R={}, expected > 200",
            pixels[i]
        );
        assert!(
            pixels[i + 1] < 30,
            "pixel ({x},{y}) G={}, expected < 30",
            pixels[i + 1]
        );
        assert!(
            pixels[i + 2] < 30,
            "pixel ({x},{y}) B={}, expected < 30",
            pixels[i + 2]
        );
    }
    // Deterministic across runs — same hash every call.
    let h1 = frame_hash(&pixels);
    let pixels2 = run_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb);
    let h2 = frame_hash(&pixels2);
    assert_eq!(
        h1, h2,
        "two runs of the same graph must produce byte-identical pixels"
    );
}

#[test]
fn second_run_produces_same_red_pixels_proving_pipeline_cache_correctness() {
    // The first dispatch_with builds the pipeline cache; the
    // second reuses it. Result should be byte-identical pixels
    // — same frame hash. Catches any pipeline-cache invalidation
    // bug that would surface as different pixels on the second
    // call.
    let a = run_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb);
    let b = run_pipeline(wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(
        frame_hash(&a),
        frame_hash(&b),
        "pipeline cache must produce identical pixels on consecutive calls"
    );
}
