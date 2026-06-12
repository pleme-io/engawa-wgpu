//! Colorblind simulation — Machado et al. 2009 color-vision-
//! deficiency matrices at severity 1.0.
//!
//! Constants ported VERBATIM from mado `src/render.rs`
//! `COLORBLIND_SHADER` (2026-06-12) — the Rust consts below and
//! the WGSL literals are the same digits, and the unit tests pin
//! both copies so the port cannot silently drift.

// LAW: the Machado constants are a VERBATIM byte-exact port from
// mado src/render.rs — digit-group separators (0.152_286) would
// break the literal pinning the tests enforce, so the readability
// lint yields to the port contract for this module.
#![allow(clippy::unreadable_literal)]

use bytemuck::{Pod, Zeroable};
use engawa::{Effect, Material, Node, ResourceId};

use super::{post_material, SCENE};

pub const EFFECT_NAME: &str = "colorblind";
pub const PRIORITY: u16 = 700;
pub const PARAMS_RESOURCE: &str = "colorblind:params";
pub const WGSL: &str = include_str!("wgsl/colorblind.wgsl");

/// Machado et al. 2009 severity-1.0 simulation matrix —
/// protanopia (red-blind). Row-major rows `(r', g', b')` over
/// columns `(r, g, b)`.
pub const MACHADO_PROTANOPIA: [[f32; 3]; 3] = [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
];

/// Machado et al. 2009 severity-1.0 simulation matrix —
/// deuteranopia (green-blind).
pub const MACHADO_DEUTERANOPIA: [[f32; 3]; 3] = [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
];

/// Machado et al. 2009 severity-1.0 simulation matrix —
/// tritanopia (blue-blind).
pub const MACHADO_TRITANOPIA: [[f32; 3]; 3] = [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
];

/// Simulation mode — the wire values match the WGSL `mode`
/// branch arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum ColorblindMode {
    #[default]
    None = 0,
    Protanopia = 1,
    Deuteranopia = 2,
    Tritanopia = 3,
}

/// Uniform payload — 16 bytes (`mode` + std140 padding).
///
/// Tier-honest (M3 review 2026-06-12): the field is sealed and
/// [`ColorblindParams::new`] is the sole non-bytes constructor, but
/// the `Pod` derive the uniform path requires is a safe bytes
/// ingress — `bytemuck::cast::<[u32; 4], ColorblindParams>([7, 0, 0, 0])`
/// mints an out-of-contract mode word without touching `new`. So
/// out-of-contract words are **only-mitigated**, not unrepresentable.
/// The WGSL contract is therefore total over the whole u32 domain:
/// unknown mode words degrade to pass-through (mode-0 semantics),
/// pinned by `wgsl_is_total_over_the_mode_word` below and the
/// out-of-contract GPU pixel test.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct ColorblindParams {
    mode: u32,
    _pad: [u32; 3],
}

impl ColorblindParams {
    #[must_use]
    pub fn new(mode: ColorblindMode) -> Self {
        Self { mode: mode as u32, _pad: [0; 3] }
    }

    #[must_use]
    pub fn mode_word(self) -> u32 {
        self.mode
    }
}

impl Default for ColorblindParams {
    fn default() -> Self {
        Self::new(ColorblindMode::None)
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

    // Bit-exact equality IS the assertion — the verbatim port is
    // pinned, so float_cmp's epsilon advice does not apply.
    #[allow(clippy::float_cmp)]
    #[test]
    fn machado_matrices_pin_the_verbatim_mado_values() {
        assert_eq!(MACHADO_PROTANOPIA[0], [0.152286, 1.052583, -0.204868]);
        assert_eq!(MACHADO_PROTANOPIA[1], [0.114503, 0.786281, 0.099216]);
        assert_eq!(MACHADO_PROTANOPIA[2], [-0.003882, -0.048116, 1.051998]);
        assert_eq!(MACHADO_DEUTERANOPIA[0], [0.367322, 0.860646, -0.227968]);
        assert_eq!(MACHADO_DEUTERANOPIA[1], [0.280085, 0.672501, 0.047413]);
        assert_eq!(MACHADO_DEUTERANOPIA[2], [-0.011820, 0.042940, 0.968881]);
        assert_eq!(MACHADO_TRITANOPIA[0], [1.255528, -0.076749, -0.178779]);
        assert_eq!(MACHADO_TRITANOPIA[1], [-0.078411, 0.930809, 0.147602]);
        assert_eq!(MACHADO_TRITANOPIA[2], [0.004733, 0.691367, 0.303900]);
    }

    #[test]
    fn wgsl_embeds_the_same_machado_literals() {
        // Every coefficient, in its exact textual form (leading
        // negatives are attached, mid-expression minus is the
        // binary operator — same as mado's source). If anyone
        // "cleans up" the WGSL digits the port is no longer
        // verbatim and this fails.
        for lit in [
            "0.152286", "1.052583", "0.204868", "0.114503", "0.786281", "0.099216",
            "-0.003882", "0.048116", "1.051998", "0.367322", "0.860646", "0.227968",
            "0.280085", "0.672501", "0.047413", "-0.011820", "0.042940", "0.968881",
            "1.255528", "0.076749", "0.178779", "-0.078411", "0.930809", "0.147602",
            "0.004733", "0.691367", "0.303900",
        ] {
            assert!(WGSL.contains(lit), "colorblind WGSL lost Machado literal {lit}");
        }
    }

    #[test]
    fn mode_words_match_the_wgsl_branch_arms() {
        assert_eq!(ColorblindParams::new(ColorblindMode::None).mode_word(), 0);
        assert_eq!(ColorblindParams::new(ColorblindMode::Protanopia).mode_word(), 1);
        assert_eq!(ColorblindParams::new(ColorblindMode::Deuteranopia).mode_word(), 2);
        assert_eq!(ColorblindParams::new(ColorblindMode::Tritanopia).mode_word(), 3);
    }

    /// The WGSL must name every contract arm EXPLICITLY (1u/2u/3u)
    /// and end in a pass-through default — a catch-all matrix arm
    /// silently rendered every out-of-contract mode word (reachable
    /// via the Pod bytes ingress) as Tritanopia.
    #[test]
    fn wgsl_is_total_over_the_mode_word() {
        for arm in ["params.mode == 1u", "params.mode == 2u", "params.mode == 3u"] {
            assert!(WGSL.contains(arm), "colorblind WGSL lost explicit arm {arm}");
        }
        let tail = WGSL.split("params.mode == 3u").nth(1).expect("3u arm present");
        let default_arm = tail.split("} else {").nth(1).expect("default arm present");
        assert!(
            default_arm.contains("return color;"),
            "out-of-contract mode words must degrade to pass-through"
        );
    }

    /// The Pod bytes ingress is REAL (this is the only-mitigated
    /// tier stated honestly): casting raw words mints a mode the
    /// constructor would never produce. Pin that the ingress exists
    /// so the doc claim above stays falsifiable.
    #[test]
    fn pod_cast_can_mint_out_of_contract_mode_words() {
        let params: ColorblindParams = bytemuck::cast([7_u32, 0, 0, 0]);
        assert_eq!(params.mode_word(), 7);
    }
}
