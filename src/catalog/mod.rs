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
//! The [`CatalogEffect`] enum AND every per-variant dispatch method
//! are emitted by the [`catalog_effects!`] table macro — ONE row per
//! effect (its module, params type, and aux resources), never the 9×
//! parallel hand-authored `match` blocks this used to be (the ★★
//! EMITTER SUBSTRATE rule: generation over composition). `ALL` is in
//! turn derived from the enum variants (`pleme-allvariants-derive`),
//! so the matrix forcing test (`tests/catalog_matrix.rs`) asserting
//! `MATRIX.len() == ALL.len()` still refuses a new effect that lacks
//! a matrix row.
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
pub mod grain;
pub mod scanlines;
pub mod snow;
pub mod window_depth;

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
        // engawa added fixed-function render state under 0.1.x; all-default
        // = opaque / no-cull / ccw / triangle-list, i.e. unchanged behaviour
        // for these full-screen effects (per engawa's RenderState contract).
        state: Default::default(),
        name: name.to_string(),
        shader: ShaderSource::inline(wgsl),
        bindings: vec![
            UniformBinding {
                group: 0,
                stages: Default::default(),
                binding: 0,
                kind: BindingKind::Texture,
                resource: input.clone(),
            },
            UniformBinding {
                group: 0,
                stages: Default::default(),
                binding: 1,
                kind: BindingKind::Sampler,
                resource: CATALOG_SAMPLER.into(),
            },
            UniformBinding {
                group: 0,
                stages: Default::default(),
                binding: 2,
                kind: BindingKind::Uniform,
                resource: params_resource.into(),
            },
        ],
    }
}

/// Emit the [`CatalogEffect`] registry + every per-variant dispatch
/// method from a single table — ONE row per effect declares its
/// `Variant => module::ParamsType` (plus an optional `aux: [...]` list
/// of intermediate texture resources the lowering introduces). Adding
/// an effect is one row; the 9-arm parallel matches this used to be
/// are generated. `graph()` stays hand-written (it's generic over the
/// rows, composing the generated methods).
///
/// This is the ★★ EMITTER SUBSTRATE rule applied to the catalog
/// dispatch: a recurring impl shape becomes generated, not repeated.
macro_rules! catalog_effects {
    (
        $(
            $variant:ident => $module:ident :: $params:ident
                $(, aux: [ $($aux:expr),* $(,)? ])?
        );+ $(;)?
    ) => {
        /// The catalog registry. `ALL` is emitted by the derive — adding
        /// a row to [`catalog_effects!`] grows the registry and breaks the
        /// matrix forcing test until the effect ships a matrix row.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, AllVariants)]
        pub enum CatalogEffect {
            $( $variant ),+
        }

        impl CatalogEffect {
            /// Operator-facing effect name (matches the `(defeffect …)`
            /// form name and the YAML toggle key).
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self { $( Self::$variant => $module::EFFECT_NAME ),+ }
            }

            /// Render-order priority — all catalog effects live in
            /// engawa's post range (200..=799).
            #[must_use]
            pub const fn priority(self) -> u16 {
                match self { $( Self::$variant => $module::PRIORITY ),+ }
            }

            /// Resource id of the effect's params uniform buffer.
            #[must_use]
            pub const fn params_resource(self) -> &'static str {
                match self { $( Self::$variant => $module::PARAMS_RESOURCE ),+ }
            }

            /// `size_of` the effect's Params struct — must equal the
            /// `(params-size N)` declared in the effect's `.tlisp` form
            /// (enforced by the matrix test).
            #[must_use]
            pub const fn params_size(self) -> usize {
                match self { $( Self::$variant => ::core::mem::size_of::<$module::$params>() ),+ }
            }

            /// Repo-relative path of the effect's `(defeffect …)` form —
            /// `effects/<module>.tlisp` (module name == effect name).
            #[must_use]
            pub const fn tlisp_path(self) -> &'static str {
                match self { $( Self::$variant => concat!("effects/", stringify!($module), ".tlisp") ),+ }
            }

            /// Default params, bytemuck-encoded — ready to seed a uniform
            /// buffer or a [`crate::FrameUniforms`] entry. Going through
            /// `bytemuck::bytes_of` is also the matrix test's Pod proof: a
            /// non-Pod Params would not compile here.
            #[must_use]
            pub fn default_params_bytes(self) -> Vec<u8> {
                match self {
                    $( Self::$variant => bytemuck::bytes_of(&$module::$params::default()).to_vec() ),+
                }
            }

            /// The operator-facing toggle unit (engawa `Effect`). For
            /// multi-node effects (bloom) this carries the material that
            /// lands on the output; [`CatalogEffect::lower`] is the
            /// canonical node surface either way.
            #[must_use]
            pub fn effect(self) -> Effect {
                match self { $( Self::$variant => $module::effect() ),+ }
            }

            /// Canonical Effect → Node lowering: read `input`, write
            /// `output`, plus the effect's internal ping-pong nodes
            /// (bloom emits 4 nodes; everything else 1).
            #[must_use]
            pub fn lower(self, input: &ResourceId, output: &ResourceId) -> Vec<Node> {
                match self { $( Self::$variant => $module::lower(input, output) ),+ }
            }

            /// Intermediate (node-produced) resources the lowering
            /// introduces beyond `input`/`output` — the consumer leases
            /// these from a [`crate::TexturePool`] and the graph declares
            /// them. Empty for single-node effects.
            #[must_use]
            pub fn aux_resources(self) -> Vec<(&'static str, ResourceKind)> {
                match self {
                    $(
                        Self::$variant => {
                            #[allow(unused_mut)]
                            let mut v: Vec<(&'static str, ResourceKind)> = Vec::new();
                            $( $( v.push((
                                $aux,
                                ResourceKind::Texture { width: None, height: None, format: None, sample_count: None },
                            )); )* )?
                            v
                        }
                    ),+
                }
            }
        }
    };
}

catalog_effects! {
    Colorblind  => colorblind::ColorblindParams;
    Crt         => crt::CrtParams;
    Scanlines   => scanlines::ScanlinesParams;
    Bloom       => bloom::BloomParams, aux: [bloom::BRIGHT_RESOURCE, bloom::BLUR_H_RESOURCE, bloom::BLUR_V_RESOURCE];
    GlowOnBell  => glow_on_bell::GlowOnBellParams;
    Snow        => snow::SnowParams;
    Aurora      => aurora::AuroraParams;
    Grain       => grain::GrainParams;
    WindowDepth => window_depth::WindowDepthParams;
}

impl CatalogEffect {
    /// The canonical single-effect graph: [`SCENE`] (input) →
    /// effect nodes → [`OUT`], with the sampler + params uniform
    /// declared as graph inputs. `graph().compile()` succeeding
    /// is the matrix test's wiring proof. Generic over the table
    /// rows (composes the generated methods), so it stays
    /// hand-written rather than macro-emitted.
    #[must_use]
    pub fn graph(self) -> RenderGraph {
        let scene: ResourceId = SCENE.into();
        let out: ResourceId = OUT.into();
        let params_size =
            u32::try_from(self.params_size()).expect("catalog params structs are tiny");
        let mut g = RenderGraph::default()
            .with_resource(SCENE, ResourceKind::Texture { width: None, height: None, format: None, sample_count: None })
            .with_resource(OUT, ResourceKind::Texture { width: None, height: None, format: None, sample_count: None })
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
