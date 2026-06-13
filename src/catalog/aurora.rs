//! Aurora — the Borealis signature curtain: slow-drifting
//! vertical light bands with noise-driven shimmer, composited
//! over the scene at low opacity and concentrated toward the
//! TOP of the frame (sky above a horizon line — the prompt area
//! below the horizon is never touched).
//!
//! ## Quality tiers — the measured cost story
//!
//! [`AuroraQuality`] (uniform word `tier.x`) selects the
//! cost/beauty point. Two GPU suites in `tests/catalog_gpu.rs`
//! (`gpu_tests` feature) pin the tier contract on a real
//! adapter: the perf smoke dispatches every tier and asserts the
//! cost ordering Off ≤ Low ≤ Medium ≤ High, and the pixel proofs
//! read the rendered bytes back — Medium visibly draws above the
//! horizon, the prompt area below the horizon stays scene
//! byte-exact, and `Off` / out-of-contract words (≥ 4) return
//! the scene:
//!
//! | tier | shader work per sky pixel | intent |
//! |---|---|---|
//! | `Off` (0) | scene pass-through (byte-exact) | runtime kill-switch |
//! | `Low` (1) | 2 × value noise, cheap mix | trivially < 0.3 ms @1440p-class |
//! | `Medium` (2) | 3-octave fbm + 1-octave shimmer | the shipped default |
//! | `High` (3) | 4-octave fbm + 12-step ray-march shimmer | the gorgeous one |
//!
//! ## `Off` vs omitting the node — the documented tradeoff
//!
//! `quality = Off` keeps the node in the graph; the shader
//! early-outs after one `textureSample`, so the pass costs
//! ~a fullscreen blit (bandwidth, zero ALU). That buys the
//! consumer a **rebuild-free toggle** — a perf governor can flip
//! the uniform word per frame without recompiling the graph.
//! TRUE zero cost requires omitting the node (zero nodes), which
//! is a graph rebuild. Rule of thumb: governors use `Off`;
//! persistent user settings omit the node.
//!
//! ## `reduce_motion` contract
//!
//! The CONSUMER omits the node entirely (zero nodes) when the
//! platform reduce-motion setting is on — the same contract as
//! snow and `glow_on_bell`. `Off` is the perf-governing path, not
//! the accessibility path.
//!
//! ## Clock
//!
//! The shader is stateless; the consumer supplies time via
//! [`AuroraParams::set_time`] each frame — the same `frame.x`
//! plumbing as snow.
//!
//! ## Colors
//!
//! The three stops are physically-inspired **uniform params**,
//! not shader constants: [`DEFAULT_GREEN`] approximates the
//! oxygen 557.7 nm emission line (the auroral green), [`DEFAULT_CYAN`]
//! the high-altitude cyan wash, [`DEFAULT_VIOLET`] the nitrogen
//! ~427.8 nm edge. The Borealis theme's exact ishou hexes plug in
//! at mado wiring time via [`AuroraParams::with_colors`].

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "aurora";
pub const PRIORITY: u16 = 450;
pub const PARAMS_RESOURCE: &str = "aurora:params";
pub const WGSL: &str = include_str!("wgsl/aurora.wgsl");

/// Curtain base stop — linear-rgb approximation of the oxygen
/// 557.7 nm emission line (the canonical auroral green).
pub const DEFAULT_GREEN: [f32; 3] = [0.10, 0.95, 0.35];
/// Curtain mid stop — high-altitude cyan wash.
pub const DEFAULT_CYAN: [f32; 3] = [0.05, 0.60, 0.70];
/// Curtain edge stop — nitrogen ~427.8 nm violet.
pub const DEFAULT_VIOLET: [f32; 3] = [0.45, 0.18, 0.85];

/// Typed quality tier — the only Rust-side way to mint the
/// `tier.x` uniform word, so an out-of-contract word is
/// unrepresentable on the typed surface. The Pod-bytes ingress
/// can still mint any `u32`; the WGSL degrades words `>= 4` to
/// pass-through (colorblind's default-arm posture).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuroraQuality {
    /// Shader early-outs to a byte-exact scene pass-through.
    /// Costs ~one fullscreen blit (the node still dispatches);
    /// see the module docs for the tradeoff vs omitting the node.
    Off = 0,
    /// Single-octave value-noise curtain — the frame-budget
    /// floor tier a perf governor steps down to.
    Low = 1,
    /// 3-octave fbm curtain + one-octave shimmer — the shipped
    /// default.
    Medium = 2,
    /// 4-octave fbm + 12-step vertical ray-march shimmer — the
    /// gorgeous one.
    High = 3,
}

impl AuroraQuality {
    /// The uniform word the WGSL switches on.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Typed read-back of a uniform word; `None` for
    /// out-of-contract words (which the shader renders as
    /// pass-through).
    #[must_use]
    pub const fn from_u32(word: u32) -> Option<Self> {
        match word {
            0 => Some(Self::Off),
            1 => Some(Self::Low),
            2 => Some(Self::Medium),
            3 => Some(Self::High),
            _ => None,
        }
    }
}

/// Per-frame aurora uniform. 96 bytes, std140-friendly (every
/// field is a vec4-aligned 16-byte tuple).
///
/// Tuple layout:
/// * `frame        = (time_seconds, intensity, drift, shimmer)`
/// * `geometry     = (horizon, band_scale, res_x, res_y)`
///   - `horizon`: screen-space y (0 = top, 1 = bottom) below
///     which the curtain is zero — keeps the aurora above the
///     prompt-line area.
/// * `color_green  = (r, g, b, _)` — curtain base stop
/// * `color_cyan   = (r, g, b, _)` — curtain mid stop
/// * `color_violet = (r, g, b, _)` — curtain edge stop
/// * `tier         = (quality, _, _, _)` as `vec4<u32>`
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct AuroraParams {
    pub frame: [f32; 4],
    pub geometry: [f32; 4],
    pub color_green: [f32; 4],
    pub color_cyan: [f32; 4],
    pub color_violet: [f32; 4],
    pub tier: [u32; 4],
}

impl Default for AuroraParams {
    fn default() -> Self {
        Self {
            frame: [0.0, 0.35, 1.0, 0.5], // time, intensity, drift, shimmer
            geometry: [0.62, 2.4, 800.0, 600.0], // horizon, band_scale, res
            color_green: [DEFAULT_GREEN[0], DEFAULT_GREEN[1], DEFAULT_GREEN[2], 0.0],
            color_cyan: [DEFAULT_CYAN[0], DEFAULT_CYAN[1], DEFAULT_CYAN[2], 0.0],
            color_violet: [DEFAULT_VIOLET[0], DEFAULT_VIOLET[1], DEFAULT_VIOLET[2], 0.0],
            tier: [AuroraQuality::Medium as u32, 0, 0, 0],
        }
    }
}

impl AuroraParams {
    /// Set `time_seconds`. Drives all drift + shimmer motion.
    #[must_use]
    pub fn with_time(mut self, t: f32) -> Self {
        self.frame[0] = t;
        self
    }
    pub fn set_time(&mut self, t: f32) {
        self.frame[0] = t;
    }

    /// Master opacity gain. 0..1, default 0.35 — the curtain is
    /// sky dressing; the scene always reads through (the shader
    /// additionally caps coverage at `MAX_ALPHA = 0.5`).
    #[must_use]
    pub fn with_intensity(mut self, i: f32) -> Self {
        self.frame[1] = i.clamp(0.0, 1.0);
        self
    }
    pub fn set_intensity(&mut self, i: f32) {
        self.frame[1] = i.clamp(0.0, 1.0);
    }

    /// Drift-speed multiplier over the slow base rate. 0..4,
    /// default 1.0 (a curtain crosses one noise cell in ~50 s).
    #[must_use]
    pub fn with_drift(mut self, d: f32) -> Self {
        self.frame[2] = d.clamp(0.0, 4.0);
        self
    }
    pub fn set_drift(&mut self, d: f32) {
        self.frame[2] = d.clamp(0.0, 4.0);
    }

    /// Shimmer amount. 0..1, default 0.5. Low tier ignores it
    /// (that is the budget); Medium pays one extra noise octave;
    /// High pays the full vertical ray march.
    #[must_use]
    pub fn with_shimmer(mut self, s: f32) -> Self {
        self.frame[3] = s.clamp(0.0, 1.0);
        self
    }
    pub fn set_shimmer(&mut self, s: f32) {
        self.frame[3] = s.clamp(0.0, 1.0);
    }

    /// Horizon line in screen-space y (0 = top, 1 = bottom);
    /// the curtain is zero below it. 0.05..1, default 0.62.
    #[must_use]
    pub fn with_horizon(mut self, h: f32) -> Self {
        self.geometry[0] = h.clamp(0.05, 1.0);
        self
    }
    pub fn set_horizon(&mut self, h: f32) {
        self.geometry[0] = h.clamp(0.05, 1.0);
    }

    /// Horizontal curtain frequency. 0.5..8, default 2.4.
    #[must_use]
    pub fn with_band_scale(mut self, b: f32) -> Self {
        self.geometry[1] = b.clamp(0.5, 8.0);
        self
    }
    pub fn set_band_scale(&mut self, b: f32) {
        self.geometry[1] = b.clamp(0.5, 8.0);
    }

    /// Target resolution in physical pixels (aspect correction).
    #[must_use]
    pub fn with_resolution(mut self, [w, h]: [f32; 2]) -> Self {
        self.geometry[2] = w;
        self.geometry[3] = h;
        self
    }
    pub fn set_resolution(&mut self, [w, h]: [f32; 2]) {
        self.geometry[2] = w;
        self.geometry[3] = h;
    }

    /// Select the quality tier (typed — out-of-contract words
    /// cannot be minted through this surface).
    #[must_use]
    pub fn with_quality(mut self, q: AuroraQuality) -> Self {
        self.tier[0] = q.as_u32();
        self
    }
    pub fn set_quality(&mut self, q: AuroraQuality) {
        self.tier[0] = q.as_u32();
    }

    /// Typed read-back of the tier word; `None` if the word was
    /// minted out-of-contract via the Pod-bytes ingress.
    #[must_use]
    pub const fn quality(&self) -> Option<AuroraQuality> {
        AuroraQuality::from_u32(self.tier[0])
    }

    /// Replace the three spectrum stops (linear rgb) — this is
    /// where the Borealis ishou hexes plug in at mado wiring
    /// time.
    #[must_use]
    pub fn with_colors(
        mut self,
        green: [f32; 3],
        cyan: [f32; 3],
        violet: [f32; 3],
    ) -> Self {
        self.color_green = [green[0], green[1], green[2], 0.0];
        self.color_cyan = [cyan[0], cyan[1], cyan[2], 0.0];
        self.color_violet = [violet[0], violet[1], violet[2], 0.0];
        self
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

    // Defaults and tier words are exact constants — bit equality
    // is the intended assertion, no epsilon involved.
    #[allow(clippy::float_cmp)]
    #[test]
    fn default_color_stops_are_pinned() {
        let p = AuroraParams::default();
        assert_eq!(&p.color_green[..3], &DEFAULT_GREEN);
        assert_eq!(&p.color_cyan[..3], &DEFAULT_CYAN);
        assert_eq!(&p.color_violet[..3], &DEFAULT_VIOLET);
        assert_eq!(p.color_green[3], 0.0);
        // The physically-inspired ordering: green is the
        // brightest/greenest stop, violet the bluest.
        assert!(p.color_green[1] > p.color_cyan[1]);
        assert!(p.color_violet[2] > p.color_green[2]);
    }

    #[test]
    fn tier_constants_match_the_wgsl_contract() {
        assert_eq!(AuroraQuality::Off.as_u32(), 0);
        assert_eq!(AuroraQuality::Low.as_u32(), 1);
        assert_eq!(AuroraQuality::Medium.as_u32(), 2);
        assert_eq!(AuroraQuality::High.as_u32(), 3);
        // The WGSL consts must pin the same words — the uniform
        // is the wire contract between the two.
        assert!(WGSL.contains("const QUALITY_OFF: u32 = 0u"));
        assert!(WGSL.contains("const QUALITY_LOW: u32 = 1u"));
        assert!(WGSL.contains("const QUALITY_MEDIUM: u32 = 2u"));
        assert!(WGSL.contains("const QUALITY_HIGH: u32 = 3u"));
    }

    #[test]
    fn quality_round_trips_and_rejects_out_of_contract_words() {
        for q in [
            AuroraQuality::Off,
            AuroraQuality::Low,
            AuroraQuality::Medium,
            AuroraQuality::High,
        ] {
            assert_eq!(AuroraQuality::from_u32(q.as_u32()), Some(q));
        }
        assert_eq!(AuroraQuality::from_u32(4), None);
        assert_eq!(AuroraQuality::from_u32(u32::MAX), None);

        // The pub-field / Pod-bytes ingress can mint any word; the
        // typed read-back surfaces it as None instead of inventing
        // a tier (the WGSL renders it as pass-through).
        let p = AuroraParams { tier: [7, 0, 0, 0], ..AuroraParams::default() };
        assert_eq!(p.quality(), None);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn defaults_pin_the_recommended_mado_shape() {
        let p = AuroraParams::default();
        assert_eq!(p.quality(), Some(AuroraQuality::Medium), "shipped default tier");
        assert_eq!(p.frame[1], 0.35, "default opacity — sky dressing, scene reads through");
        assert_eq!(p.frame[2], 1.0, "default drift multiplier");
        assert_eq!(p.frame[3], 0.5, "default shimmer");
        assert_eq!(p.geometry[0], 0.62, "default horizon — above the prompt area");
        assert_eq!(p.geometry[1], 2.4, "default band scale");
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn builders_clamp_within_range() {
        let p = AuroraParams::default()
            .with_intensity(2.0)
            .with_drift(-1.0)
            .with_shimmer(9.0)
            .with_horizon(0.0)
            .with_band_scale(100.0);
        assert_eq!(p.frame[1], 1.0);
        assert_eq!(p.frame[2], 0.0);
        assert_eq!(p.frame[3], 1.0);
        assert_eq!(p.geometry[0], 0.05);
        assert_eq!(p.geometry[1], 8.0);
    }

    #[test]
    fn params_layout_is_96_bytes_of_six_vec4s() {
        assert_eq!(size_of::<AuroraParams>(), 96);
        assert_eq!(align_of::<AuroraParams>(), 4);
    }

    #[test]
    fn wgsl_pins_the_craft_contract() {
        assert!(WGSL.len() > 1000, "aurora.wgsl looks suspiciously small");
        assert!(WGSL.contains("@fragment"));
        // Tiered noise ladder.
        assert!(WGSL.contains("fn vnoise"));
        assert!(WGSL.contains("fn fbm"));
        assert!(WGSL.contains("fn ray_factor"), "High tier's vertical ray march");
        // Scene sampled in uniform control flow before branching,
        // composited in-shader (blend-free dispatcher pipelines).
        assert!(WGSL.contains("textureSample(input_tex"));
        assert!(WGSL.contains("scene.rgb * (1.0 - alpha)"), "premultiplied over");
        // Banding-free: spatial dither keyed on the pixel position
        // with NO time term (frame-stable — no temporal noise).
        assert!(WGSL.contains("hash12(in.pos.xy)"), "spatial dither anchor");
        // Off + out-of-contract degrade to pass-through.
        assert!(WGSL.contains("quality == QUALITY_OFF || quality > QUALITY_HIGH"));
    }
}
