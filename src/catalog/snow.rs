//! Snow — parallax fractal flakes + particle pile, absorbed
//! from `engawa-snow` (2026-06-12).
//!
//! `SnowParams` (layout, ranges, builder surface) is ported
//! verbatim from `engawa-snow/src/lib.rs`; the WGSL is the
//! upstream asset re-banded to the catalog binding convention
//! with an in-shader premultiplied-over composite (see the
//! header of `wgsl/snow.wgsl` for the two deliberate deltas).
//! The engawa-snow repo remains the standalone demo; this copy
//! is the dispatcher-native catalog entry.

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "snow";
pub const PRIORITY: u16 = 500;
pub const PARAMS_RESOURCE: &str = "snow:params";
pub const WGSL: &str = include_str!("wgsl/snow.wgsl");

/// Per-frame snow uniform. 64 bytes, std140-friendly (every
/// field is a vec4-aligned tuple of f32s).
///
/// Tuple layout (4 floats each):
/// * `frame      = (time_seconds, intensity, wind, typing_pulse)`
/// * `params     = (accumulation, layer_count, temperature, _)`
///   - `temperature`: 0 = freezing (pile grows from incoming
///     snowfall, no melt), 0.5 = neutral (no growth, no melt),
///     1 = warm (pile melts visibly + tint shifts cool-blue).
/// * `resolution = (width, height, _, _)`
/// * `cursor     = (x, y, _, _)`  in pixel coords; (<0, <0) = none
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct SnowParams {
    pub frame: [f32; 4],
    pub params: [f32; 4],
    pub resolution: [f32; 4],
    pub cursor: [f32; 4],
}

impl Default for SnowParams {
    fn default() -> Self {
        Self {
            frame: [0.0, 1.0, 0.0, 0.0],     // time, intensity, wind, typing_pulse
            params: [0.0, 3.0, 0.0, 0.0],    // accumulation, layer_count, temperature, _
            resolution: [800.0, 600.0, 0.0, 0.0],
            cursor: [-1.0, -1.0, 0.0, 0.0],  // no cursor
        }
    }
}

impl SnowParams {
    /// Set `time_seconds`. Drives all motion.
    #[must_use]
    pub fn with_time(mut self, t: f32) -> Self {
        self.frame[0] = t;
        self
    }
    pub fn set_time(&mut self, t: f32) {
        self.frame[0] = t;
    }

    /// Master gain. 0..1, default 1.0.
    #[must_use]
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.frame[1] = i.clamp(0.0, 1.0);
        self
    }
    pub fn set_intensity(&mut self, i: f32) {
        self.frame[1] = i.clamp(0.0, 1.0);
    }

    /// Horizontal wind. -1..1. Drives snowflake drift +
    /// accumulation pile shape.
    #[must_use]
    pub fn with_wind(mut self, w: f32) -> Self {
        self.frame[2] = w.clamp(-1.0, 1.0);
        self
    }
    pub fn set_wind(&mut self, w: f32) {
        self.frame[2] = w.clamp(-1.0, 1.0);
    }

    /// Typing pulse. 0..1. Decays toward 0 between keystrokes;
    /// caller is responsible for the decay (typical: pulse =
    /// (pulse * 0.92).max(0.0) per frame).
    #[must_use]
    pub fn with_typing_pulse(mut self, p: f32) -> Self {
        self.frame[3] = p.clamp(0.0, 1.0);
        self
    }
    pub fn set_typing_pulse(&mut self, p: f32) {
        self.frame[3] = p.clamp(0.0, 1.0);
    }
    /// Inject a fresh typing pulse, taking the max of the
    /// existing pulse so a slow decay doesn't swallow a rapid
    /// burst.
    pub fn pulse_typing(&mut self, p: f32) {
        self.frame[3] = self.frame[3].max(p.clamp(0.0, 1.0));
    }

    /// Ground accumulation. 0..1. 0 = no pile; 1 = thick pile
    /// covering ~25% of screen height.
    #[must_use]
    pub fn with_accumulation(mut self, a: f32) -> Self {
        self.params[0] = a.clamp(0.0, 1.0);
        self
    }
    pub fn set_accumulation(&mut self, a: f32) {
        self.params[0] = a.clamp(0.0, 1.0);
    }

    /// Number of parallax layers. 1..3.
    #[must_use]
    pub fn with_layer_count(mut self, n: f32) -> Self {
        self.params[1] = n.clamp(1.0, 3.0);
        self
    }
    pub fn set_layer_count(&mut self, n: f32) {
        self.params[1] = n.clamp(1.0, 3.0);
    }

    /// Temperature (0..1). 0 = freezing — pile accumulates from
    /// incoming snowfall, no melt. 0.5 = neutral. 1 = warm —
    /// pile melts visibly (shrinks over time, tint shifts to
    /// cool blue).
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.params[2] = t.clamp(0.0, 1.0);
        self
    }
    pub fn set_temperature(&mut self, t: f32) {
        self.params[2] = t.clamp(0.0, 1.0);
    }

    /// Screen resolution in pixels. Used for aspect-correct
    /// noise + cursor normalization.
    #[must_use]
    pub fn with_resolution(mut self, [w, h]: [f32; 2]) -> Self {
        self.resolution[0] = w;
        self.resolution[1] = h;
        self
    }
    pub fn set_resolution(&mut self, [w, h]: [f32; 2]) {
        self.resolution[0] = w;
        self.resolution[1] = h;
    }

    /// Cursor position in pixels. Pass (-1, -1) to disable
    /// cursor deflection.
    #[must_use]
    pub fn with_cursor(mut self, [x, y]: [f32; 2]) -> Self {
        self.cursor[0] = x;
        self.cursor[1] = y;
        self
    }
    pub fn set_cursor(&mut self, [x, y]: [f32; 2]) {
        self.cursor[0] = x;
        self.cursor[1] = y;
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
    fn builders_clamp_within_range() {
        let p = SnowParams::default()
            .with_intensity(2.0)
            .with_wind(-99.0)
            .with_typing_pulse(5.5)
            .with_accumulation(-0.5)
            .with_layer_count(99.0);
        assert_eq!(p.frame[1], 1.0);
        assert_eq!(p.frame[2], -1.0);
        assert_eq!(p.frame[3], 1.0);
        assert_eq!(p.params[0], 0.0);
        assert_eq!(p.params[1], 3.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn pulse_typing_takes_max_not_overwrite() {
        let mut p = SnowParams::default().with_typing_pulse(0.6);
        p.pulse_typing(0.3);
        assert_eq!(p.frame[3], 0.6, "existing pulse must survive a smaller injected one");
        p.pulse_typing(0.9);
        assert_eq!(p.frame[3], 0.9);
    }

    #[test]
    fn absorbed_wgsl_keeps_the_upstream_anchors_and_composites_in_shader() {
        // Structural anchors from the upstream asset — catches
        // accidental truncation of the embedded file.
        assert!(WGSL.len() > 1000, "snow.wgsl looks suspiciously small");
        assert!(WGSL.contains("@fragment"));
        assert!(WGSL.contains("fn snow_layer"));
        assert!(WGSL.contains("fn pile_particles"));
        assert!(WGSL.contains("fn fractal_dendrite"));
        assert!(WGSL.contains("fn grade"));
        // Delta 2: the catalog copy samples the scene (blend-free
        // dispatcher composite) — upstream did not.
        assert!(WGSL.contains("textureSample(input_tex"));
    }
}
