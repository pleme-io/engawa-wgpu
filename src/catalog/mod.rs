//! Typed effect catalog — the built-in post-process effects
//! every engawa consumer composes from (engawa roadmap v0.4,
//! realised here in the wgpu backend crate so the WGSL lives
//! next to the dispatcher that runs it).
//!
//! Each effect ships four artifacts:
//!
//! 1. embedded WGSL (`include_str!` from `src/catalog/wgsl/`),
//! 2. a `#[repr(C)]` `bytemuck` Pod+Zeroable `…Params` struct
//!    (the typed payload for [`crate::FrameUniforms`]),
//! 3. a constructor returning an [`engawa::Effect`] plus the
//!    canonical `lower(input, output) -> Vec<Node>` lowering
//!    (priorities in the post range 200..=799),
//! 4. a `(defeffect …)` tatara-lisp form at `effects/<name>.tlisp`
//!    declaring the params byte size.
//!
//! ## Mechanical registry
//!
//! [`CatalogEffect::ALL`] is **derived from the enum variants**
//! (`pleme-allvariants-derive`) — never hand-listed. The matrix
//! forcing test (`tests/catalog_matrix.rs`) keeps one row per
//! effect and asserts `MATRIX.len() == ALL.len()`, so a new
//! variant cannot land without a matrix row: the derive grows
//! `ALL`, the len-equality fails, and every exhaustive `match`
//! below refuses to compile until the new effect is wired.
//!
//! ## Resource conventions
//!
//! Single-input effects bind: input texture `@binding(0)`, the
//! shared [`CATALOG_SAMPLER`] `@binding(1)`, the effect's params
//! uniform `@binding(2)`. The canonical graph shape is
//! [`SCENE`] → effect → [`OUT`]; consumers ping-pong via
//! [`crate::TexturePool`] leases.

pub mod aurora;
pub mod bloom;
pub mod colorblind;
pub mod crt;
pub mod glow_on_bell;
pub mod scanlines;
pub mod snow;

use engawa::{
    BindingKind, Effect, Material, Node, RenderGraph, ResourceId, ResourceKind,
    ShaderSource, UniformBinding,
};
use pleme_allvariants_derive::AllVariants;

/// Canonical scene-input resource id — the texture the catalog
/// effect reads (the consumer's rendered frame so far).
pub const SCENE: &str = "scene";

/// Canonical output resource id — the texture the catalog
/// effect writes (next ping-pong target or the surface).
pub const OUT: &str = "out";

/// One shared filtering sampler every catalog material binds —
/// consumers create a single `wgpu::Sampler` and bind it here.
pub const CATALOG_SAMPLER: &str = "catalog:sampler";

/// Standard single-input post-process material: input texture
/// at binding 0, shared sampler at 1, params uniform at 2.
pub(crate) fn post_material(
    name: &str,
    wgsl: &str,
    input: &ResourceId,
    params_resource: &str,
) -> Material {
    Material {
        name: name.to_string(),
        shader: ShaderSource::inline(wgsl),
        bindings: vec![
            UniformBinding {
                binding: 0,
                kind: BindingKind::Texture,
                resource: input.clone(),
            },
            UniformBinding {
                binding: 1,
                kind: BindingKind::Sampler,
                resource: CATALOG_SAMPLER.into(),
            },
            UniformBinding {
                binding: 2,
                kind: BindingKind::Uniform,
                resource: params_resource.into(),
            },
        ],
    }
}

/// The catalog registry. `ALL` is emitted by the derive — adding
/// a variant mechanically grows the registry and breaks every
/// exhaustive match below until the effect is fully wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, AllVariants)]
pub enum CatalogEffect {
    Colorblind,
    Crt,
    Scanlines,
    Bloom,
    GlowOnBell,
    Snow,
    Aurora,
}

impl CatalogEffect {
    /// Operator-facing effect name (matches the `(defeffect …)`
    /// form name and the YAML toggle key).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Colorblind => colorblind::EFFECT_NAME,
            Self::Crt => crt::EFFECT_NAME,
            Self::Scanlines => scanlines::EFFECT_NAME,
            Self::Bloom => bloom::EFFECT_NAME,
            Self::GlowOnBell => glow_on_bell::EFFECT_NAME,
            Self::Snow => snow::EFFECT_NAME,
            Self::Aurora => aurora::EFFECT_NAME,
        }
    }

    /// Render-order priority — all catalog effects live in
    /// engawa's post range (200..=799).
    #[must_use]
    pub const fn priority(self) -> u16 {
        match self {
            Self::Colorblind => colorblind::PRIORITY,
            Self::Crt => crt::PRIORITY,
            Self::Scanlines => scanlines::PRIORITY,
            Self::Bloom => bloom::PRIORITY,
            Self::GlowOnBell => glow_on_bell::PRIORITY,
            Self::Snow => snow::PRIORITY,
            Self::Aurora => aurora::PRIORITY,
        }
    }

    /// Resource id of the effect's params uniform buffer.
    #[must_use]
    pub const fn params_resource(self) -> &'static str {
        match self {
            Self::Colorblind => colorblind::PARAMS_RESOURCE,
            Self::Crt => crt::PARAMS_RESOURCE,
            Self::Scanlines => scanlines::PARAMS_RESOURCE,
            Self::Bloom => bloom::PARAMS_RESOURCE,
            Self::GlowOnBell => glow_on_bell::PARAMS_RESOURCE,
            Self::Snow => snow::PARAMS_RESOURCE,
            Self::Aurora => aurora::PARAMS_RESOURCE,
        }
    }

    /// `size_of` the effect's Params struct — must equal the
    /// `(params-size N)` declared in the effect's `.tlisp` form
    /// (enforced by the matrix test).
    #[must_use]
    pub const fn params_size(self) -> usize {
        match self {
            Self::Colorblind => size_of::<colorblind::ColorblindParams>(),
            Self::Crt => size_of::<crt::CrtParams>(),
            Self::Scanlines => size_of::<scanlines::ScanlinesParams>(),
            Self::Bloom => size_of::<bloom::BloomParams>(),
            Self::GlowOnBell => size_of::<glow_on_bell::GlowOnBellParams>(),
            Self::Snow => size_of::<snow::SnowParams>(),
            Self::Aurora => size_of::<aurora::AuroraParams>(),
        }
    }

    /// Repo-relative path of the effect's `(defeffect …)` form.
    #[must_use]
    pub const fn tlisp_path(self) -> &'static str {
        match self {
            Self::Colorblind => "effects/colorblind.tlisp",
            Self::Crt => "effects/crt.tlisp",
            Self::Scanlines => "effects/scanlines.tlisp",
            Self::Bloom => "effects/bloom.tlisp",
            Self::GlowOnBell => "effects/glow_on_bell.tlisp",
            Self::Snow => "effects/snow.tlisp",
            Self::Aurora => "effects/aurora.tlisp",
        }
    }

    /// Default params, bytemuck-encoded — ready to seed a
    /// uniform buffer or a [`crate::FrameUniforms`] entry. Going
    /// through `bytemuck::bytes_of` is also the matrix test's
    /// Pod proof: a non-Pod Params would not compile here.
    #[must_use]
    pub fn default_params_bytes(self) -> Vec<u8> {
        match self {
            Self::Colorblind => {
                bytemuck::bytes_of(&colorblind::ColorblindParams::default()).to_vec()
            }
            Self::Crt => bytemuck::bytes_of(&crt::CrtParams::default()).to_vec(),
            Self::Scanlines => {
                bytemuck::bytes_of(&scanlines::ScanlinesParams::default()).to_vec()
            }
            Self::Bloom => bytemuck::bytes_of(&bloom::BloomParams::default()).to_vec(),
            Self::GlowOnBell => {
                bytemuck::bytes_of(&glow_on_bell::GlowOnBellParams::default()).to_vec()
            }
            Self::Snow => bytemuck::bytes_of(&snow::SnowParams::default()).to_vec(),
            Self::Aurora => bytemuck::bytes_of(&aurora::AuroraParams::default()).to_vec(),
        }
    }

    /// The operator-facing toggle unit (engawa `Effect`). For
    /// multi-node effects (bloom) this carries the material
    /// that lands on the output; [`CatalogEffect::lower`] is the
    /// canonical node surface either way.
    #[must_use]
    pub fn effect(self) -> Effect {
        match self {
            Self::Colorblind => colorblind::effect(),
            Self::Crt => crt::effect(),
            Self::Scanlines => scanlines::effect(),
            Self::Bloom => bloom::effect(),
            Self::GlowOnBell => glow_on_bell::effect(),
            Self::Snow => snow::effect(),
            Self::Aurora => aurora::effect(),
        }
    }

    /// Canonical Effect → Node lowering: read `input`, write
    /// `output`, plus the effect's internal ping-pong nodes
    /// (bloom emits 4 nodes; everything else 1).
    #[must_use]
    pub fn lower(self, input: &ResourceId, output: &ResourceId) -> Vec<Node> {
        match self {
            Self::Colorblind => colorblind::lower(input, output),
            Self::Crt => crt::lower(input, output),
            Self::Scanlines => scanlines::lower(input, output),
            Self::Bloom => bloom::lower(input, output),
            Self::GlowOnBell => glow_on_bell::lower(input, output),
            Self::Snow => snow::lower(input, output),
            Self::Aurora => aurora::lower(input, output),
        }
    }

    /// Intermediate (node-produced) resources the lowering
    /// introduces beyond `input`/`output` — the consumer leases
    /// these from a [`crate::TexturePool`] and the graph
    /// declares them.
    #[must_use]
    pub fn aux_resources(self) -> Vec<(&'static str, ResourceKind)> {
        match self {
            Self::Bloom => vec![
                (
                    bloom::BRIGHT_RESOURCE,
                    ResourceKind::Texture { width: None, height: None },
                ),
                (
                    bloom::BLUR_H_RESOURCE,
                    ResourceKind::Texture { width: None, height: None },
                ),
                (
                    bloom::BLUR_V_RESOURCE,
                    ResourceKind::Texture { width: None, height: None },
                ),
            ],
            Self::Colorblind
            | Self::Crt
            | Self::Scanlines
            | Self::GlowOnBell
            | Self::Snow
            | Self::Aurora => Vec::new(),
        }
    }

    /// The canonical single-effect graph: [`SCENE`] (input) →
    /// effect nodes → [`OUT`], with the sampler + params uniform
    /// declared as graph inputs. `graph().compile()` succeeding
    /// is the matrix test's wiring proof.
    #[must_use]
    pub fn graph(self) -> RenderGraph {
        let scene: ResourceId = SCENE.into();
        let out: ResourceId = OUT.into();
        let params_size =
            u32::try_from(self.params_size()).expect("catalog params structs are tiny");
        let mut g = RenderGraph::default()
            .with_resource(SCENE, ResourceKind::Texture { width: None, height: None })
            .with_resource(OUT, ResourceKind::Texture { width: None, height: None })
            .with_resource(CATALOG_SAMPLER, ResourceKind::Sampler)
            .with_resource(
                self.params_resource(),
                ResourceKind::Uniform { size_bytes: params_size },
            )
            .with_input(SCENE)
            .with_input(CATALOG_SAMPLER)
            .with_input(self.params_resource())
            .with_output(OUT);
        for (id, kind) in self.aux_resources() {
            g = g.with_resource(id, kind);
        }
        for node in self.lower(&scene, &out) {
            g = g.with_node(node);
        }
        g
    }
}
