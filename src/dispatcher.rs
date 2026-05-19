//! `WgpuDispatcher` — engawa's `Dispatcher` trait realised
//! against wgpu.
//!
//! Pipeline cache keyed by Material name; pipelines compile
//! once per Material per WgpuDispatcher lifetime. Each
//! dispatch_node call begins one render pass + draws one
//! fullscreen triangle.
//!
//! Bind-group construction lives at the boundary: callers
//! pass a `BoundResources` map (engawa `ResourceId` →
//! `BoundResource` containing the live wgpu handle), and this
//! crate constructs the bind group on demand from the Material's
//! declared bindings. The consumer owns the wgpu textures /
//! buffers / samplers; this crate orchestrates the dispatch.

use std::collections::BTreeMap;

use engawa::{
    BindingKind, CompiledGraph, DispatchError, Dispatcher, Material, Node, NodeId,
    PassKind, ResourceBindings, ResourceId,
};
use thiserror::Error;

use crate::pipeline::combined_shader_source;

#[derive(Debug, Error)]
pub enum WgpuDispatcherError {
    #[error("engawa dispatch error: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("unsupported pass kind for v0.1: {0:?}; only Render is implemented today")]
    UnsupportedPass(PassKind),
    #[error("node {node:?} has no material but pass kind requires one")]
    MissingMaterial { node: NodeId },
    #[error(
        "node {node:?} binding {binding} expects {expected:?} but bound resource for {resource:?} is {actual:?}"
    )]
    BindingKindMismatch {
        node: NodeId,
        binding: u32,
        resource: ResourceId,
        expected: BindingKind,
        actual: &'static str,
    },
    #[error(
        "node {node:?} output {resource:?} has no bound wgpu::TextureView (output bindings must be textures)"
    )]
    OutputNotBound {
        node: NodeId,
        resource: ResourceId,
    },
    #[error("node {node:?} binding {binding} resource {resource:?} not present in BoundResources")]
    BoundResourceMissing {
        node: NodeId,
        binding: u32,
        resource: ResourceId,
    },
}

/// Live wgpu handle wrapped in a tagged enum so the dispatcher
/// can match the bind type the Material declared. Operators
/// build this from their own wgpu resources at dispatch time.
#[derive(Clone)]
pub enum BoundResource {
    Texture {
        view: wgpu::TextureView,
        format: wgpu::TextureFormat,
    },
    Uniform(wgpu::Buffer),
    Storage(wgpu::Buffer),
    Sampler(wgpu::Sampler),
}

/// Per-frame map of engawa `ResourceId` → live wgpu handle.
/// The consumer (mado, future ayatsuri) populates this before
/// calling `dispatch_graph`. Engawa already validated at
/// compile time that every node references a resource that's
/// either an input or another node's output; the dispatcher
/// validates that every referenced resource has a `BoundResource`
/// entry at dispatch time.
#[derive(Default, Clone)]
pub struct BoundResources {
    inner: BTreeMap<ResourceId, BoundResource>,
}

impl BoundResources {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with(
        mut self,
        id: impl Into<ResourceId>,
        resource: BoundResource,
    ) -> Self {
        self.inner.insert(id.into(), resource);
        self
    }

    pub fn insert(&mut self, id: impl Into<ResourceId>, resource: BoundResource) {
        self.inner.insert(id.into(), resource);
    }

    #[must_use]
    pub fn get(&self, id: &ResourceId) -> Option<&BoundResource> {
        self.inner.get(id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Per-Material wgpu pipeline cache entry.
struct CachedPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Dispatcher that compiles engawa render graphs to wgpu
/// commands. Construct once; call `dispatch_graph` per frame.
pub struct WgpuDispatcher<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    target_format: wgpu::TextureFormat,
    pipelines: BTreeMap<String, CachedPipeline>,
    /// Encoder used for the current `dispatch_graph` call. The
    /// caller passes their own encoder via `set_encoder`; the
    /// dispatcher uses it for every per-node render pass, then
    /// the caller submits.
    encoder: Option<wgpu::CommandEncoder>,
    /// Per-frame bound resources. Set by `dispatch_with` before
    /// the graph walk.
    bound: Option<BoundResources>,
}

impl<'a> WgpuDispatcher<'a> {
    #[must_use]
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            target_format,
            pipelines: BTreeMap::new(),
            encoder: None,
            bound: None,
        }
    }

    /// One-shot helper: compile (if needed), build bindings,
    /// walk the graph, return the recorded `CommandBuffer`
    /// ready to submit. Wraps the trait's `dispatch_graph` +
    /// encoder lifecycle so the call site stays one line.
    pub fn dispatch_with(
        &mut self,
        graph: &CompiledGraph,
        bindings: ResourceBindings,
        bound: BoundResources,
    ) -> Result<wgpu::CommandBuffer, WgpuDispatcherError> {
        // Pre-compile every Material referenced in the graph.
        for node in graph.iter_nodes() {
            if let Some(material) = &node.material {
                if !self.pipelines.contains_key(&material.name) {
                    let cached = self.build_pipeline(material)?;
                    self.pipelines.insert(material.name.clone(), cached);
                }
            }
        }

        // Encoder live for the entire graph walk; one submit at
        // the end.
        self.encoder = Some(
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engawa-wgpu graph"),
                }),
        );
        self.bound = Some(bound);

        // Walk via the engawa trait's default impl — it validates
        // ResourceBindings + delegates each node to dispatch_node.
        self.dispatch_graph(graph, &bindings)?;

        let encoder = self.encoder.take().expect("encoder set");
        self.bound = None;
        Ok(encoder.finish())
    }

    fn build_pipeline(
        &self,
        material: &Material,
    ) -> Result<CachedPipeline, WgpuDispatcherError> {
        let fragment_wgsl = match &material.shader {
            engawa::ShaderSource::Inline { wgsl } => wgsl.clone(),
            engawa::ShaderSource::Path { path } => {
                std::fs::read_to_string(path).unwrap_or_else(|e| {
                    // Surface error via tracing; pipeline will
                    // fail to compile and the wgpu error scope
                    // will catch it.
                    eprintln!(
                        "engawa-wgpu: failed to read shader at {path}: {e}; \
                         falling back to red-tint placeholder"
                    );
                    "@fragment fn fs_main() -> @location(0) vec4<f32> { \
                     return vec4<f32>(1.0, 0.0, 0.0, 1.0); }"
                        .to_string()
                })
            }
        };
        let combined = combined_shader_source(&fragment_wgsl);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&material.name),
            source: wgpu::ShaderSource::Wgsl(combined.into()),
        });

        // Bind-group layout from the Material's declared bindings.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = material
            .bindings
            .iter()
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b.binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: binding_kind_to_wgpu(b.kind),
                count: None,
            })
            .collect();
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&material.name),
                entries: &entries,
            });
        let pipeline_layout =
            self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&material.name),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&material.name),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(CachedPipeline {
            pipeline,
            bind_group_layout,
        })
    }
}

fn binding_kind_to_wgpu(kind: BindingKind) -> wgpu::BindingType {
    match kind {
        BindingKind::Uniform => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingKind::StorageRead => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingKind::StorageReadWrite => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingKind::Texture => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        BindingKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    }
}

impl<'a> Dispatcher for WgpuDispatcher<'a> {
    fn dispatch_node(
        &mut self,
        node: &Node,
        _bindings: &ResourceBindings,
    ) -> Result<(), DispatchError> {
        if node.pass != PassKind::Render {
            // v0.1 scope. Compute / Blit land next iteration.
            return Err(DispatchError::Backend(format!(
                "engawa-wgpu v0.1 only supports Render; node {:?} requested {:?}",
                node.id, node.pass
            )));
        }

        // Clear-only nodes (no material): paint a black load+clear
        // into the first output. Mado typically uses this as the
        // first node in the graph.
        let Some(material) = node.material.as_ref() else {
            let output_id = node.outputs.first().ok_or_else(|| {
                DispatchError::Backend(format!(
                    "clear node {:?} has no outputs",
                    node.id
                ))
            })?;
            let bound = self.bound.as_ref().ok_or_else(|| {
                DispatchError::Backend("dispatch called without bound resources".into())
            })?;
            let view = match bound.get(output_id) {
                Some(BoundResource::Texture { view, .. }) => view,
                _ => {
                    return Err(DispatchError::Backend(format!(
                        "clear node {:?} output {:?} is not a Texture binding",
                        node.id, output_id
                    )));
                }
            };
            let encoder = self
                .encoder
                .as_mut()
                .ok_or_else(|| DispatchError::Backend("no encoder live".into()))?;
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(node.id.as_str()),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            return Ok(());
        };

        // Fullscreen-effect node: bind group + draw 3 vertices.
        let cached = self.pipelines.get(&material.name).ok_or_else(|| {
            DispatchError::Backend(format!(
                "pipeline not built for material {} — call dispatch_with",
                material.name
            ))
        })?;

        let bound = self.bound.as_ref().ok_or_else(|| {
            DispatchError::Backend("dispatch called without bound resources".into())
        })?;

        // Build bind group from declared bindings.
        let entries: Vec<wgpu::BindGroupEntry> = material
            .bindings
            .iter()
            .map(|b| {
                let resource = bound.get(&b.resource).ok_or_else(|| {
                    DispatchError::Backend(format!(
                        "node {:?} binding {} references resource {:?} not in BoundResources",
                        node.id, b.binding, b.resource
                    ))
                })?;
                let binding_resource = match (b.kind, resource) {
                    (BindingKind::Uniform, BoundResource::Uniform(buf))
                    | (BindingKind::StorageRead, BoundResource::Storage(buf))
                    | (BindingKind::StorageReadWrite, BoundResource::Storage(buf)) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: buf,
                            offset: 0,
                            size: None,
                        })
                    }
                    (BindingKind::Texture, BoundResource::Texture { view, .. }) => {
                        wgpu::BindingResource::TextureView(view)
                    }
                    (BindingKind::Sampler, BoundResource::Sampler(s)) => {
                        wgpu::BindingResource::Sampler(s)
                    }
                    _ => {
                        return Err(DispatchError::Backend(format!(
                            "node {:?} binding {} kind mismatch (expected {:?})",
                            node.id, b.binding, b.kind
                        )));
                    }
                };
                Ok(wgpu::BindGroupEntry {
                    binding: b.binding,
                    resource: binding_resource,
                })
            })
            .collect::<Result<Vec<_>, DispatchError>>()?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(node.id.as_str()),
            layout: &cached.bind_group_layout,
            entries: &entries,
        });

        // Target view = first output (we don't support MRT in v0.1).
        let output_id = node.outputs.first().ok_or_else(|| {
            DispatchError::Backend(format!(
                "fullscreen-effect node {:?} has no outputs",
                node.id
            ))
        })?;
        let view = match bound.get(output_id) {
            Some(BoundResource::Texture { view, .. }) => view,
            _ => {
                return Err(DispatchError::Backend(format!(
                    "node {:?} output {:?} is not a Texture binding",
                    node.id, output_id
                )));
            }
        };

        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| DispatchError::Backend("no encoder live".into()))?;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(node.id.as_str()),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);

        // queue is captured for future per-frame uniform writes;
        // silence the unused-field lint for now.
        let _ = self.queue;

        Ok(())
    }
}
