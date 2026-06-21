//! `WgpuDispatcher` — engawa's `Dispatcher` trait realised
//! against wgpu.
//!
//! **Per-call dispatch is the canonical path.** Construct once
//! with `new` (the device/queue handles are cloned in — wgpu
//! handles are internally reference-counted, so this shares the
//! underlying device, it does not duplicate it), then call
//! [`WgpuDispatcher::dispatch_with`] per frame with the graph,
//! the bindings, the live wgpu handles, and the per-frame
//! [`FrameUniforms`].
//!
//! LAW (2026-06-12): the former `WgpuDispatcher<'a>` struct-level
//! `&'a Device + &'a Queue` borrow is deleted. It forced mado to
//! bypass the dispatcher entirely for post-process + snow
//! rendering (mado's `TerminalRenderer` does not own the device,
//! so it could not hold a dispatcher that borrowed one). Owned
//! Arc-backed handles + per-call `dispatch_with` is the
//! successor; do not reintroduce a lifetime here.
//!
//! Pipeline cache keyed by Material name; pipelines compile
//! once per Material per `WgpuDispatcher` lifetime (see
//! [`WgpuDispatcher::invalidate_material`] for the hot-reload
//! seam). Each `dispatch_node` call begins one render pass +
//! draws one fullscreen triangle.
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

/// Typed dispatch-failure surface. Every variant here is
/// CONSTRUCTED by the dispatch path — the M3 review (2026-06-12)
/// found five declared-but-never-built variants advertising a typed
/// API the error paths didn't deliver; `MissingMaterial` (materialless
/// nodes are clear nodes by design, so the state was unreachable) is
/// deleted and the rest now flow through [`Dispatcher::dispatch_node`]
/// via Display at the trait seam (the engawa trait returns
/// `DispatchError`, so typed variants stringify exactly once there).
#[derive(Debug, Error)]
pub enum WgpuDispatcherError {
    #[error("engawa dispatch error: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("unsupported pass kind for v0.1: {0:?}; only Render is implemented today")]
    UnsupportedPass(PassKind),
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
    #[error(
        "material {material} shader at {path} is unreadable: {source} — refusing to dispatch a placeholder"
    )]
    ShaderUnreadable {
        material: String,
        path: String,
        source: std::io::Error,
    },
    #[error(
        "material {material} uses raw Metal Shading Language (ShaderSource::Msl), which the wgpu backend cannot accept — it compiles WGSL only; dispatch MSL materials on a Metal backend"
    )]
    UnsupportedShaderSource { material: String },
    #[error(
        "frame uniform for {resource:?} has no BoundResources entry — bind the uniform buffer before dispatch"
    )]
    FrameUniformUnbound { resource: ResourceId },
    #[error(
        "frame uniform for {resource:?} expects a Uniform buffer but the bound resource is {actual}"
    )]
    FrameUniformKindMismatch {
        resource: ResourceId,
        actual: &'static str,
    },
    #[error(
        "frame uniform for {resource:?} is {actual} bytes but wgpu writes must be multiples of {} bytes — pad the Pod struct to the next 4-byte boundary",
        wgpu::COPY_BUFFER_ALIGNMENT
    )]
    FrameUniformMisaligned { resource: ResourceId, actual: usize },
    #[error(
        "frame uniform for {resource:?} is {actual} bytes but the bound buffer holds exactly {capacity} — partial writes leave stale tail bytes and are never intended"
    )]
    FrameUniformSizeMismatch {
        resource: ResourceId,
        actual: usize,
        capacity: u64,
    },
}

/// The one stringify seam: the engawa `Dispatcher` trait returns
/// `DispatchError`, so typed `WgpuDispatcherError` values constructed
/// inside the walk convert here — typed at the source, Display'd once
/// at the boundary, never `format!`-composed inline.
fn backend(err: &WgpuDispatcherError) -> DispatchError {
    DispatchError::Backend(err.to_string())
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

impl BoundResource {
    /// Variant name for typed error reporting.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            BoundResource::Texture { .. } => "Texture",
            BoundResource::Uniform(_) => "Uniform",
            BoundResource::Storage(_) => "Storage",
            BoundResource::Sampler(_) => "Sampler",
        }
    }
}

/// Per-frame map of engawa `ResourceId` → live wgpu handle.
/// The consumer (mado, future ayatsuri) populates this before
/// calling `dispatch_with`. Engawa already validated at
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

/// Per-frame uniform payloads — a typed map of engawa
/// `ResourceId` → `bytemuck`-encoded bytes that
/// [`WgpuDispatcher::dispatch_with`] writes into the
/// corresponding [`BoundResource::Uniform`] buffers *before*
/// any pass of that dispatch is encoded, so every node in the
/// graph walk sees the same frame data.
///
/// Entries are inserted through the typed [`FrameUniforms::set`]
/// / [`FrameUniforms::with`] surface (`bytemuck::Pod` values
/// only) — there is no raw-bytes ingress, so a non-Pod or
/// padding-carrying struct cannot enter the map (compile error
/// at the bound, not a runtime check).
#[derive(Default, Clone)]
pub struct FrameUniforms {
    inner: BTreeMap<ResourceId, Vec<u8>>,
}

impl FrameUniforms {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style insert of one Pod params value.
    #[must_use]
    pub fn with<P: bytemuck::Pod>(
        mut self,
        id: impl Into<ResourceId>,
        params: &P,
    ) -> Self {
        self.set(id, params);
        self
    }

    /// Insert (or replace) one Pod params value.
    pub fn set<P: bytemuck::Pod>(&mut self, id: impl Into<ResourceId>, params: &P) {
        self.inner
            .insert(id.into(), bytemuck::bytes_of(params).to_vec());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate entries in deterministic (`BTreeMap`) order.
    pub fn iter(&self) -> impl Iterator<Item = (&ResourceId, &[u8])> {
        self.inner.iter().map(|(id, bytes)| (id, bytes.as_slice()))
    }
}

/// Per-Material wgpu pipeline cache entry.
struct CachedPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Dispatcher that compiles engawa render graphs to wgpu
/// commands. Construct once; call `dispatch_with` per frame.
pub struct WgpuDispatcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_format: wgpu::TextureFormat,
    pipelines: BTreeMap<String, CachedPipeline>,
    /// Encoder used for the current `dispatch_with` call; the
    /// dispatcher uses it for every per-node render pass, then
    /// finishes it into the returned `CommandBuffer`.
    encoder: Option<wgpu::CommandEncoder>,
    /// Per-frame bound resources. Set by `dispatch_with` before
    /// the graph walk.
    bound: Option<BoundResources>,
}

impl WgpuDispatcher {
    /// Construct a dispatcher. The device/queue handles are
    /// cloned (wgpu handles are internally reference-counted) —
    /// the dispatcher holds no lifetime borrow, so a consumer
    /// that does not own its device (mado's `TerminalRenderer`)
    /// can still own a dispatcher.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            target_format,
            pipelines: BTreeMap::new(),
            encoder: None,
            bound: None,
        }
    }

    /// Number of Materials with a compiled pipeline in the
    /// cache. Pipelines compile once per Material name; a
    /// second `dispatch_with` over the same graph must not grow
    /// this count.
    #[must_use]
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Drop the cached pipeline for one Material name. The
    /// cache is keyed by name only — a hot-reload that swaps a
    /// Material's shader under the same name MUST call this or
    /// the stale pipeline keeps dispatching.
    pub fn invalidate_material(&mut self, name: &str) {
        self.pipelines.remove(name);
    }

    /// Canonical per-call dispatch: write the per-frame
    /// uniforms, compile any uncached Materials, walk the
    /// graph, return the recorded `CommandBuffer` ready to
    /// submit.
    pub fn dispatch_with(
        &mut self,
        graph: &CompiledGraph,
        bindings: &ResourceBindings,
        bound: BoundResources,
        frame: &FrameUniforms,
    ) -> Result<wgpu::CommandBuffer, WgpuDispatcherError> {
        // Pre-compile every Material referenced in the graph. A
        // Path-sourced shader that cannot be read is a typed error
        // HERE — never a silently-compiling placeholder pipeline.
        for node in graph.iter_nodes() {
            if let Some(material) = &node.material
                && !self.pipelines.contains_key(&material.name)
            {
                let cached = self.build_pipeline(material)?;
                self.pipelines.insert(material.name.clone(), cached);
            }
        }

        // Per-frame uniform writes happen before any pass is
        // encoded so every node in this dispatch sees the same
        // frame data.
        for (id, bytes) in frame.iter() {
            let Some(resource) = bound.get(id) else {
                return Err(WgpuDispatcherError::FrameUniformUnbound {
                    resource: id.clone(),
                });
            };
            let BoundResource::Uniform(buf) = resource else {
                return Err(WgpuDispatcherError::FrameUniformKindMismatch {
                    resource: id.clone(),
                    actual: resource.kind_name(),
                });
            };
            // wgpu's write_buffer PANICS on a data size that is not
            // a COPY_BUFFER_ALIGNMENT multiple — reject it as a typed
            // error before the panic path is reachable (M3 review
            // 2026-06-12). Size must match EXACTLY: an under-sized
            // write passes wgpu validation but leaves the buffer tail
            // holding the previous frame's bytes — a silent wrong
            // answer, not an error anyone sees.
            if !(bytes.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                return Err(WgpuDispatcherError::FrameUniformMisaligned {
                    resource: id.clone(),
                    actual: bytes.len(),
                });
            }
            if bytes.len() as u64 != buf.size() {
                return Err(WgpuDispatcherError::FrameUniformSizeMismatch {
                    resource: id.clone(),
                    actual: bytes.len(),
                    capacity: buf.size(),
                });
            }
            self.queue.write_buffer(buf, 0, bytes);
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
        let walked = self.dispatch_graph(graph, bindings);
        self.bound = None;
        let encoder = self.encoder.take().expect("encoder set");
        walked?;
        Ok(encoder.finish())
    }

    fn build_pipeline(
        &self,
        material: &Material,
    ) -> Result<CachedPipeline, WgpuDispatcherError> {
        // LAW (2026-06-12): an unreadable Path shader is a typed
        // error, never a fallback. The deleted red-tint placeholder
        // was valid WGSL — the pipeline compiled and dispatched wrong
        // pixels with only a stderr line as signal (the silent-wrong-
        // answer anti-pattern the typed-spec rules forbid).
        let fragment_wgsl = match &material.shader {
            engawa::ShaderSource::Inline { wgsl } => wgsl.clone(),
            engawa::ShaderSource::Path { path } => std::fs::read_to_string(path)
                .map_err(|source| WgpuDispatcherError::ShaderUnreadable {
                    material: material.name.clone(),
                    path: path.clone(),
                    source,
                })?,
            // engawa's typed Metal-only escape hatch (ShaderSource::Msl).
            // The wgpu backend compiles WGSL (via naga) only, so per
            // engawa's documented contract we reject raw MSL at dispatch
            // with a typed error rather than silently mis-rendering — the
            // same silent-wrong-answer anti-pattern guarded against above.
            engawa::ShaderSource::Msl { .. } => {
                return Err(WgpuDispatcherError::UnsupportedShaderSource {
                    material: material.name.clone(),
                });
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

/// First-output texture view lookup shared by the clear + effect
/// paths. Associated fn (not a method) so callers can keep a
/// disjoint `&mut self.encoder` borrow alive alongside it.
fn first_output_view<'b>(
    node: &Node,
    bound: &'b BoundResources,
) -> Result<&'b wgpu::TextureView, DispatchError> {
    let output_id = node.outputs.first().ok_or_else(|| {
        DispatchError::Backend(format!("node {:?} has no outputs", node.id))
    })?;
    let Some(BoundResource::Texture { view, .. }) = bound.get(output_id) else {
        return Err(backend(&WgpuDispatcherError::OutputNotBound {
            node: node.id.clone(),
            resource: output_id.clone(),
        }));
    };
    Ok(view)
}

/// Build the wgpu bind-group entries a Material's declared
/// bindings resolve to against the per-frame `BoundResources`.
fn bind_group_entries<'b>(
    node: &Node,
    material: &Material,
    bound: &'b BoundResources,
) -> Result<Vec<wgpu::BindGroupEntry<'b>>, DispatchError> {
    material
        .bindings
        .iter()
        .map(|b| {
            let resource = bound.get(&b.resource).ok_or_else(|| {
                backend(&WgpuDispatcherError::BoundResourceMissing {
                    node: node.id.clone(),
                    binding: b.binding,
                    resource: b.resource.clone(),
                })
            })?;
            let binding_resource = match (b.kind, resource) {
                (BindingKind::Uniform, BoundResource::Uniform(buf))
                | (
                    BindingKind::StorageRead | BindingKind::StorageReadWrite,
                    BoundResource::Storage(buf),
                ) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
                (BindingKind::Texture, BoundResource::Texture { view, .. }) => {
                    wgpu::BindingResource::TextureView(view)
                }
                (BindingKind::Sampler, BoundResource::Sampler(s)) => {
                    wgpu::BindingResource::Sampler(s)
                }
                _ => {
                    return Err(backend(&WgpuDispatcherError::BindingKindMismatch {
                        node: node.id.clone(),
                        binding: b.binding,
                        resource: b.resource.clone(),
                        expected: b.kind,
                        actual: resource.kind_name(),
                    }));
                }
            };
            Ok(wgpu::BindGroupEntry {
                binding: b.binding,
                resource: binding_resource,
            })
        })
        .collect::<Result<Vec<_>, DispatchError>>()
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

impl Dispatcher for WgpuDispatcher {
    fn dispatch_node(
        &mut self,
        node: &Node,
        _bindings: &ResourceBindings,
    ) -> Result<(), DispatchError> {
        if node.pass != PassKind::Render {
            // v0.1 scope. Compute / Blit land next iteration.
            return Err(backend(&WgpuDispatcherError::UnsupportedPass(node.pass)));
        }

        let bound = self.bound.as_ref().ok_or_else(|| {
            DispatchError::Backend("dispatch called without bound resources".into())
        })?;
        let view = first_output_view(node, bound)?;

        // Clear-only nodes (no material): paint a black load+clear
        // into the first output. Mado typically uses this as the
        // first node in the graph.
        let Some(material) = node.material.as_ref() else {
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

        let entries = bind_group_entries(node, material, bound)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(node.id.as_str()),
            layout: &cached.bind_group_layout,
            entries: &entries,
        });

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
                    // Clear, not Load: every effect node is a
                    // fullscreen triangle with blending disabled, so
                    // the previous attachment contents are always
                    // fully overwritten. On tile-based GPUs (Apple
                    // Silicon) Load forces a full attachment restore
                    // into tile memory per pass — pure bandwidth
                    // waste at scene-sized textures (M3 review
                    // 2026-06-12). Clear resolves in tile memory.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

        Ok(())
    }
}
