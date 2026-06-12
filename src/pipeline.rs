//! Built-in fullscreen vertex shader + render-pipeline
//! construction helpers.

/// The canonical fullscreen triangle: three vertices in clip
/// space (no buffers needed). `vertex_index` 0/1/2 maps to
/// (-1,-1) / (3,-1) / (-1,3), which fully covers [-1, 1]² with
/// no overdraw beyond the viewport.
///
/// Operators' fullscreen-effect WGSL only needs to provide
/// `fs_main`; the vertex stage is shared across every effect
/// to keep pipeline cost down (one less shader-module compile
/// per material).
pub const FULLSCREEN_VERTEX_WGSL: &str = r"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // (-1,-1), (3,-1), (-1,3) — covers the entire viewport
    // with a single triangle (no buffers).
    let x = f32(i32(idx & 1u) * 4 - 1);
    let y = f32(i32(idx & 2u) * 2 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, y * -0.5 + 0.5);
    return out;
}
";

/// Concatenate the built-in fullscreen vertex shader with the
/// operator's WGSL (which provides `fs_main`). Returns the
/// combined source ready to hand to
/// `wgpu::Device::create_shader_module`.
#[must_use]
pub fn combined_shader_source(fragment_wgsl: &str) -> String {
    let mut out =
        String::with_capacity(FULLSCREEN_VERTEX_WGSL.len() + fragment_wgsl.len() + 2);
    out.push_str(FULLSCREEN_VERTEX_WGSL);
    out.push('\n');
    out.push_str(fragment_wgsl);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_shader_includes_vertex_then_fragment() {
        let fs = "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }";
        let combined = combined_shader_source(fs);
        assert!(combined.contains("fn vs_main"));
        assert!(combined.contains("fn fs_main"));
        // Vertex must precede fragment so wgpu picks the correct entry.
        let v_pos = combined.find("fn vs_main").unwrap();
        let f_pos = combined.find("fn fs_main").unwrap();
        assert!(v_pos < f_pos);
    }

    #[test]
    fn fullscreen_vertex_emits_three_vertices() {
        // Smoke test: the shader source includes the three-vertex
        // bit-manipulation pattern (idx & 1, idx & 2) that
        // generates the covering triangle.
        assert!(FULLSCREEN_VERTEX_WGSL.contains("idx & 1u"));
        assert!(FULLSCREEN_VERTEX_WGSL.contains("idx & 2u"));
    }
}
