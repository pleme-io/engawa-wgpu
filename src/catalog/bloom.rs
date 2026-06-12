//! Bloom — the catalog's multi-node effect: luminance threshold
//! → separable gaussian blur (horizontal then vertical) →
//! additive composite over the untouched scene.
//!
//! Four nodes, four materials, ONE shared `BloomParams` uniform.
//! The blur axis is baked per-material (two near-identical WGSL
//! files) so the uniform stays frame-invariant across both blur
//! passes — see `wgsl/bloom_blur_h.wgsl` for the rationale.

use bytemuck::{Pod, Zeroable};
use engawa::{
    BindingKind, Effect, Material, Node, PassKind, ResourceId, ShaderSource,
    UniformBinding,
};

use super::{post_material, CATALOG_SAMPLER, SCENE};

pub const EFFECT_NAME: &str = "bloom";
pub const PRIORITY: u16 = 300;
pub const PARAMS_RESOURCE: &str = "bloom:params";

pub const THRESHOLD_WGSL: &str = include_str!("wgsl/bloom_threshold.wgsl");
pub const BLUR_H_WGSL: &str = include_str!("wgsl/bloom_blur_h.wgsl");
pub const BLUR_V_WGSL: &str = include_str!("wgsl/bloom_blur_v.wgsl");
pub const COMPOSITE_WGSL: &str = include_str!("wgsl/bloom_composite.wgsl");

/// Material names — the dispatcher's pipeline cache keys.
pub const THRESHOLD_MATERIAL: &str = "bloom:threshold";
pub const BLUR_H_MATERIAL: &str = "bloom:blur-h";
pub const BLUR_V_MATERIAL: &str = "bloom:blur-v";
pub const COMPOSITE_MATERIAL: &str = "bloom:composite";

/// Intermediate ping-pong resources the lowering introduces —
/// the consumer leases matching textures from a `TexturePool`.
pub const BRIGHT_RESOURCE: &str = "bloom:bright";
pub const BLUR_H_RESOURCE: &str = "bloom:blur-h";
pub const BLUR_V_RESOURCE: &str = "bloom:blur-v";

/// Uniform payload — 32 bytes, shared by all four nodes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct BloomParams {
    /// Physical-pixel resolution of the target.
    pub resolution: [f32; 2],
    /// Luminance cutoff 0..=1 — below goes black.
    pub threshold: f32,
    /// Additive gain of the blurred bright buffer.
    pub intensity: f32,
    /// Blur tap spread in physical pixels.
    pub radius_px: f32,
    _pad: [f32; 3],
}

impl BloomParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self { resolution, ..Self::default() }
    }
}

impl Default for BloomParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            threshold: 0.75,
            intensity: 0.6,
            radius_px: 2.5,
            _pad: [0.0; 3],
        }
    }
}

/// Stage-4 composite material: scene texture at binding 0, the
/// blurred bright buffer at 1, shared sampler at 2, params at 3.
#[must_use]
pub fn composite_material(scene: &ResourceId, blurred: &ResourceId) -> Material {
    Material {
        name: COMPOSITE_MATERIAL.to_string(),
        shader: ShaderSource::inline(COMPOSITE_WGSL),
        bindings: vec![
            UniformBinding {
                binding: 0,
                kind: BindingKind::Texture,
                resource: scene.clone(),
            },
            UniformBinding {
                binding: 1,
                kind: BindingKind::Texture,
                resource: blurred.clone(),
            },
            UniformBinding {
                binding: 2,
                kind: BindingKind::Sampler,
                resource: CATALOG_SAMPLER.into(),
            },
            UniformBinding {
                binding: 3,
                kind: BindingKind::Uniform,
                resource: PARAMS_RESOURCE.into(),
            },
        ],
    }
}

/// Operator-facing toggle. Carries the composite material (the
/// one that lands on the output); [`lower`] is the canonical
/// multi-node surface.
#[must_use]
pub fn effect() -> Effect {
    Effect {
        name: EFFECT_NAME.to_string(),
        enabled: true,
        priority: PRIORITY,
        material: composite_material(&SCENE.into(), &BLUR_V_RESOURCE.into()),
    }
}

/// threshold → blur-h → blur-v → composite. Four nodes; the
/// engawa compiler's topo sort orders them by the resource
/// chain, and the composite reads BOTH `input` and the blurred
/// buffer.
#[must_use]
pub fn lower(input: &ResourceId, output: &ResourceId) -> Vec<Node> {
    let bright: ResourceId = BRIGHT_RESOURCE.into();
    let blur_h: ResourceId = BLUR_H_RESOURCE.into();
    let blur_v: ResourceId = BLUR_V_RESOURCE.into();
    vec![
        Node::fullscreen_effect(
            THRESHOLD_MATERIAL,
            post_material(THRESHOLD_MATERIAL, THRESHOLD_WGSL, input, PARAMS_RESOURCE),
            input.clone(),
            bright.clone(),
        ),
        Node::fullscreen_effect(
            BLUR_H_MATERIAL,
            post_material(BLUR_H_MATERIAL, BLUR_H_WGSL, &bright, PARAMS_RESOURCE),
            bright,
            blur_h.clone(),
        ),
        Node::fullscreen_effect(
            BLUR_V_MATERIAL,
            post_material(BLUR_V_MATERIAL, BLUR_V_WGSL, &blur_h, PARAMS_RESOURCE),
            blur_h,
            blur_v.clone(),
        ),
        Node {
            id: COMPOSITE_MATERIAL.into(),
            pass: PassKind::Render,
            inputs: vec![input.clone(), blur_v.clone()],
            outputs: vec![output.clone()],
            material: Some(composite_material(input, &blur_v)),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_passes_differ_only_in_axis() {
        assert!(BLUR_H_WGSL.contains("vec2<f32>(1.0, 0.0)"));
        assert!(BLUR_V_WGSL.contains("vec2<f32>(0.0, 1.0)"));
        // Same gaussian taps in both passes.
        for w in ["0.227027", "0.1945946", "0.1216216", "0.054054", "0.016216"] {
            assert!(BLUR_H_WGSL.contains(w), "blur-h lost gaussian weight {w}");
            assert!(BLUR_V_WGSL.contains(w), "blur-v lost gaussian weight {w}");
        }
    }

    #[test]
    fn lowering_chains_threshold_blur_blur_composite() {
        let nodes = lower(&SCENE.into(), &"out".into());
        let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![THRESHOLD_MATERIAL, BLUR_H_MATERIAL, BLUR_V_MATERIAL, COMPOSITE_MATERIAL]
        );
        // Composite reads the original scene AND the blurred buffer.
        let composite = &nodes[3];
        assert_eq!(composite.inputs.len(), 2);
    }
}
