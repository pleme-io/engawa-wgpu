# engawa-wgpu

wgpu-backed `Dispatcher` impl for [engawa](https://github.com/pleme-io/engawa)
render graphs. Compiles `Material` → `wgpu::RenderPipeline`;
walks the compiled graph; dispatches fullscreen-effect passes
against any `wgpu::TextureView` (most commonly a
`garasu::HeadlessTarget` for tests + a winit surface for live).

## What it does

- **`WgpuDispatcher::new(device, queue, target_format)`** —
  construct once per consumer.
- **`dispatcher.dispatch_with(graph, engawa_bindings, bound_resources)`**
  — pre-compiles every Material referenced in the graph into a
  cached `wgpu::RenderPipeline`, walks the execution order, builds
  per-node bind groups, returns a `wgpu::CommandBuffer` ready to
  submit.
- **Built-in fullscreen vertex shader** (`FULLSCREEN_VERTEX_WGSL`)
  — operators' WGSL only needs `fs_main`; the vertex stage is
  shared across every effect.
- **Pipeline cache** keyed by `Material.name` — recompile happens
  only when a new Material is referenced.

## Status (v0.1)

- Render passes (fullscreen effects + clear-only nodes). **This release.**
- Compute + blit passes: pending.
- Multi-target render attachments (MRT): pending.

## What it does NOT do

- Allocate textures / buffers / samplers — the consumer owns
  those. Pass live handles via `BoundResources`.
- Manage the swapchain — the consumer owns winit / surface.
- Hot-reload shaders — that's shikumi + the consumer's notify
  watcher; this crate just compiles whatever WGSL it's handed.

## Example

```rust
use engawa::{Material, Node, RenderGraph, ResourceKind, ShaderSource};
use engawa_wgpu::{BoundResource, BoundResources, WgpuDispatcher};

let material = Material {
    name: "red-tint".into(),
    shader: ShaderSource::inline(
        "@fragment fn fs_main() -> @location(0) vec4<f32> { \
         return vec4<f32>(1.0, 0.0, 0.0, 1.0); }",
    ),
    bindings: vec![],
};

let graph = RenderGraph::default()
    .with_resource("scene", ResourceKind::Texture { width: Some(800), height: Some(600) })
    .with_resource("out",   ResourceKind::Texture { width: Some(800), height: Some(600) })
    .with_output("out")
    .with_node(Node::clear("clear", "scene"))
    .with_node(Node::fullscreen_effect("red", material, "scene", "out"))
    .compile()?;

let mut dispatcher = WgpuDispatcher::new(&gpu.device, &gpu.queue, target_format);
let cmd = dispatcher.dispatch_with(&graph, engawa_bindings, bound_resources)?;
gpu.queue.submit(std::iter::once(cmd));
```

## Tests

```bash
cargo test                          # pure-data tests, no GPU
cargo test --features gpu_tests     # adds GPU pixel tests via garasu HeadlessTarget
```

## Layering

| Crate | Concern |
|---|---|
| [engawa](https://github.com/pleme-io/engawa) | Typed render-graph IR (no GPU) |
| **engawa-wgpu** (this crate) | `Dispatcher` impl backed by wgpu |
| [garasu](https://github.com/pleme-io/garasu) | GPU context + headless harness |
| [madori](https://github.com/pleme-io/madori) | winit window + event loop |

License: MIT
