//! Scanlines — cosine-profile CRT row darkening.

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "scanlines";
pub const PRIORITY: u16 = 600;
pub const PARAMS_RESOURCE: &str = "scanlines:params";
pub const WGSL: &str = include_str!("wgsl/scanlines.wgsl");

/// Uniform payload — 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ScanlinesParams {
    /// Physical-pixel resolution of the target.
    pub resolution: [f32; 2],
    /// Scanline period in physical pixels (shader floor: 1.0).
    pub period_px: f32,
    /// Darkening strength 0..=1 (0 = exact pass-through).
    pub intensity: f32,
}

impl ScanlinesParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self { resolution, ..Self::default() }
    }
}

impl Default for ScanlinesParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            period_px: 3.0,
            intensity: 0.25,
        }
    }
}

#[must_use]
pub fn material(input: &ResourceId) -> Material {
    post_material(EFFECT_NAME, WGSL, input, PARAMS_RESOURCE)
}

#[must_use]
pub fn effect() -> Effect {
    Effect {
        name: EFFECT_NAME.to_string(),
        enabled: true,
        priority: PRIORITY,
        material: material(&SCENE.into()),
    }
}

#[must_use]
pub fn lower(input: &ResourceId, output: &ResourceId) -> Vec<Node> {
    vec![Node::fullscreen_effect(
        EFFECT_NAME,
        material(input),
        input.clone(),
        output.clone(),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_in_operator_range() {
        let p = ScanlinesParams::default();
        assert!((0.0..=1.0).contains(&p.intensity));
        assert!(p.period_px >= 1.0);
    }
}
