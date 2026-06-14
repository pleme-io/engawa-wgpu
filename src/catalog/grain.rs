//! Grain — luma-only paper-grain "tooth".
//!
//! A faint fabric texture laid on TOP of everything: priority
//! 750, above colorblind (700), so the tooth is the last thing
//! applied. The jitter is luminance-only (a grey delta added to
//! r/g/b alike) so chroma — and therefore any accent colour —
//! stays clean. At `opacity` 0.0 it is an exact pass-through.

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "grain";
/// Grain applies LAST — above colorblind (700) — so the tooth
/// goes on top of everything the rest of the catalog produced.
pub const PRIORITY: u16 = 750;
pub const PARAMS_RESOURCE: &str = "grain:params";
pub const WGSL: &str = include_str!("wgsl/grain.wgsl");

/// Uniform payload — 32 bytes (std140-friendly: two leading
/// vec-aligned tuples padded out to a multiple of 16).
///
/// Layout:
/// * `resolution` — physical-pixel resolution of the target.
/// * `opacity`    — luma-jitter amplitude 0..=1 (0 = exact
///   pass-through). Defaults to a barely-perceptible 1.5%.
/// * `scale`      — grain-cell scale multiplier; 1.0 = 1 cell per
///   physical pixel.
/// * `time`       — seconds; quantized to ~5 updates/sec in the
///   shader so the grain shimmers slowly, not every frame.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GrainParams {
    pub resolution: [f32; 2],
    pub opacity: f32,
    pub scale: f32,
    pub time: f32,
    _pad: [f32; 3],
}

impl GrainParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self { resolution, ..Self::default() }
    }

    /// Set `time` (seconds). Drives the slow shimmer.
    #[must_use]
    pub fn with_time(mut self, t: f32) -> Self {
        self.time = t;
        self
    }
    pub fn set_time(&mut self, t: f32) {
        self.time = t;
    }

    /// Luma-jitter amplitude. 0..=1, default 1.5%.
    #[must_use]
    pub fn with_opacity(mut self, o: f32) -> Self {
        self.opacity = o.clamp(0.0, 1.0);
        self
    }
    pub fn set_opacity(&mut self, o: f32) {
        self.opacity = o.clamp(0.0, 1.0);
    }

    /// Grain-cell scale multiplier. Defaults to 1.0.
    #[must_use]
    pub fn with_scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }
    pub fn set_scale(&mut self, s: f32) {
        self.scale = s;
    }

    /// Screen resolution in physical pixels.
    #[must_use]
    pub fn with_resolution(mut self, [w, h]: [f32; 2]) -> Self {
        self.resolution = [w, h];
        self
    }
    pub fn set_resolution(&mut self, [w, h]: [f32; 2]) {
        self.resolution = [w, h];
    }
}

impl Default for GrainParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            opacity: 0.015,
            scale: 1.0,
            time: 0.0,
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

    #[allow(clippy::float_cmp)]
    #[test]
    fn defaults_are_barely_perceptible_and_in_range() {
        let p = GrainParams::default();
        assert!((0.0..=1.0).contains(&p.opacity));
        assert_eq!(p.opacity, 0.015, "default tooth must be barely perceptible");
        assert_eq!(p.scale, 1.0);
        assert_eq!(p.time, 0.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn opacity_clamps_within_range() {
        assert_eq!(GrainParams::default().with_opacity(2.0).opacity, 1.0);
        assert_eq!(GrainParams::default().with_opacity(-0.5).opacity, 0.0);
    }

    #[test]
    fn wgsl_is_luma_only_and_time_quantized() {
        // Luma-only: one grey delta added to all channels.
        assert!(WGSL.contains("vec3<f32>(delta)"));
        // Slow shimmer: time quantized (not re-rolled per frame).
        assert!(WGSL.contains("floor(params.time * 5.0)"));
        // Cheap: a single hash, one texture sample.
        assert!(WGSL.contains("fn hash21"));
        assert_eq!(WGSL.matches("textureSample(input_tex").count(), 1);
    }
}
