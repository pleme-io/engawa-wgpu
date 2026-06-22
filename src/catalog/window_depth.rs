//! Window-depth — inner-edge vignette that gives the whole surface
//! a recessed, "depth around the sides and edges" feel.
//!
//! A subtle darkening that hugs the four window borders and fades
//! inward (an INNER shadow keyed off the distance to the nearest
//! edge, not a radial centre vignette). Priority 760 — the LAST
//! post effect, above grain (750) — so the depth frames everything
//! the rest of the catalog produced. At `intensity` 0.0 it is an
//! exact pass-through. The edge tint (`color`) is fed from the
//! resolved theme palette (no hardcoded effect colour), so it
//! shares the popup-elevation chrome's depth language and the two
//! read as one consistent design.

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "window_depth";
/// Window-depth applies LAST — above grain (750) — so the recessed
/// frame sits on top of everything the rest of the catalog made.
pub const PRIORITY: u16 = 760;
pub const PARAMS_RESOURCE: &str = "window_depth:params";
pub const WGSL: &str = include_str!("wgsl/window_depth.wgsl");

/// Uniform payload — 32 bytes (std140-friendly: two 16-byte rows,
/// no implicit padding).
///
/// Layout:
/// * `resolution` — physical-pixel resolution of the target.
/// * `depth`      — vignette reach inward, as a fraction of the
///   shorter dimension (0.08 = 8%). Defaults to a gentle 8%.
/// * `intensity`  — max edge darkening 0..=1 (0 = exact
///   pass-through). Defaults to a subtle 22%.
/// * `color`      — the linear-RGB tint the edges darken toward
///   (fed from the resolved theme; default near-black).
/// * `softness`   — falloff exponent; higher hugs the edge tighter.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct WindowDepthParams {
    pub resolution: [f32; 2],
    pub depth: f32,
    pub intensity: f32,
    pub color: [f32; 3],
    pub softness: f32,
}

impl WindowDepthParams {
    #[must_use]
    pub fn new(resolution: [f32; 2]) -> Self {
        Self { resolution, ..Self::default() }
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

    /// Vignette reach inward as a fraction of the shorter dimension.
    #[must_use]
    pub fn with_depth(mut self, d: f32) -> Self {
        self.depth = d.max(0.0);
        self
    }
    pub fn set_depth(&mut self, d: f32) {
        self.depth = d.max(0.0);
    }

    /// Max edge darkening, 0..=1 (0 = exact pass-through).
    #[must_use]
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.intensity = i.clamp(0.0, 1.0);
        self
    }
    pub fn set_intensity(&mut self, i: f32) {
        self.intensity = i.clamp(0.0, 1.0);
    }

    /// The linear-RGB tint the edges darken toward (from the theme).
    #[must_use]
    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = color;
    }

    /// Falloff exponent; higher hugs the edge tighter.
    #[must_use]
    pub fn with_softness(mut self, s: f32) -> Self {
        self.softness = s.max(0.0001);
        self
    }
    pub fn set_softness(&mut self, s: f32) {
        self.softness = s.max(0.0001);
    }
}

impl Default for WindowDepthParams {
    fn default() -> Self {
        Self {
            resolution: [800.0, 600.0],
            depth: 0.08,
            intensity: 0.22,
            color: [0.0, 0.0, 0.0],
            softness: 1.6,
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
    fn defaults_are_subtle_and_in_range() {
        let p = WindowDepthParams::default();
        assert!((0.0..=1.0).contains(&p.intensity));
        assert_eq!(p.intensity, 0.22, "default depth must be subtle");
        assert_eq!(p.depth, 0.08);
        assert_eq!(p.color, [0.0, 0.0, 0.0]);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn intensity_clamps_within_range() {
        assert_eq!(WindowDepthParams::default().with_intensity(2.0).intensity, 1.0);
        assert_eq!(WindowDepthParams::default().with_intensity(-0.5).intensity, 0.0);
    }

    #[test]
    fn wgsl_is_an_inner_edge_vignette() {
        // Inner shadow: keyed off distance to the nearest edge.
        assert!(WGSL.contains("edge_dist"));
        // Pass-through at intensity 0 via clamp.
        assert!(WGSL.contains("clamp(params.intensity, 0.0, 1.0)"));
        // One scene sample, cheap.
        assert_eq!(WGSL.matches("textureSample(input_tex").count(), 1);
    }
}
