//! CRT — barrel curvature + chromatic aberration + vignette.

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "crt";
pub const PRIORITY: u16 = 650;
pub const PARAMS_RESOURCE: &str = "crt:params";
pub const WGSL: &str = include_str!("wgsl/crt.wgsl");

/// Uniform payload — 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct CrtParams {
    /// Physical-pixel resolution of the target.
    pub resolution: [f32; 2],
    /// Barrel distortion strength (0 = flat; 0.05..0.15 typical).
    pub curvature: f32,
    /// Edge-darkening strength 0..=1.
    pub vignette: f32,
    /// Chromatic-aberration shift in pixels at the screen edge.
    pub aberration: f32,
    _pad: [f32; 3],
}

impl CrtParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self { resolution, ..Self::default() }
    }
}

impl Default for CrtParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            curvature: 0.08,
            vignette: 0.25,
            aberration: 0.6,
            _pad: [0.0; 3],
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
        let p = CrtParams::default();
        assert!((0.0..=1.0).contains(&p.vignette));
        assert!(p.curvature >= 0.0);
        assert!(p.aberration >= 0.0);
    }
}
