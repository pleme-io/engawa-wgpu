//! Glow-on-bell — radial gaussian glow driven by a
//! `bell_intensity` uniform.
//!
//! The shader is stateless; **the consumer supplies the clock**:
//! set [`GlowOnBellParams::bell_intensity`] to 1.0 when BEL
//! arrives ([`GlowOnBellParams::ring`]) and decay it per frame
//! on the host ([`GlowOnBellParams::decay`], typical factor
//! `0.92f32.powf(dt * 60.0)` — the same frame-rate-independent
//! half-life shape mado's snow typing-pulse uses).

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "glow_on_bell";
pub const PRIORITY: u16 = 400;
pub const PARAMS_RESOURCE: &str = "glow_on_bell:params";
pub const WGSL: &str = include_str!("wgsl/glow_on_bell.wgsl");

/// Uniform payload — 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GlowOnBellParams {
    /// Physical-pixel resolution of the target.
    pub resolution: [f32; 2],
    /// Glow centre in physical pixels (typically the cursor).
    pub center_px: [f32; 2],
    /// 0..=1 — the consumer-decayed bell clock.
    pub bell_intensity: f32,
    /// Gaussian sigma in physical pixels.
    pub radius_px: f32,
    _pad: [f32; 2],
}

impl GlowOnBellParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self {
            center_px: [resolution[0] * 0.5, resolution[1] * 0.5],
            resolution,
            ..Self::default()
        }
    }

    /// BEL arrived — saturate the clock.
    pub fn ring(&mut self) {
        self.bell_intensity = 1.0;
    }

    /// Per-frame host decay; `factor` 0..=1 (e.g.
    /// `0.92f32.powf(dt * 60.0)`).
    pub fn decay(&mut self, factor: f32) {
        self.bell_intensity = (self.bell_intensity * factor.clamp(0.0, 1.0)).max(0.0);
    }
}

impl Default for GlowOnBellParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            center_px: [400.0, 300.0],
            bell_intensity: 0.0,
            radius_px: 240.0,
            _pad: [0.0; 2],
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

    // ring()/default set exact constants — bit equality is the
    // intended assertion, no epsilon involved.
    #[allow(clippy::float_cmp)]
    #[test]
    fn idle_default_is_a_pass_through() {
        let p = GlowOnBellParams::default();
        assert_eq!(p.bell_intensity, 0.0, "no bell — no glow");
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn ring_then_decay_converges_to_zero() {
        let mut p = GlowOnBellParams::default();
        p.ring();
        assert_eq!(p.bell_intensity, 1.0);
        for _ in 0..600 {
            p.decay(0.92);
        }
        assert!(p.bell_intensity < 1e-6, "decayed clock must reach silence");
    }
}
