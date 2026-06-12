//! `TexturePool` — offscreen texture alloc/reuse keyed by
//! `(size, format, usage)`.
//!
//! Subsumes the per-effect `ensure_offscreen` pattern (mado
//! `render.rs::PostProcessPipeline::ensure_offscreen`): instead
//! of each consumer hand-tracking `last_width`/`last_height` +
//! `Option<wgpu::Texture>` per effect, the consumer leases a
//! texture for the frame and releases it back. A resize is just
//! a lease under a different key; entries for stale sizes stay
//! in the free list until [`TexturePool::clear`].
//!
//! ## Lease discipline (tier-honest)
//!
//! [`TexturePool::lease`] returns a **move-only**
//! [`TextureLease`] — the only handout the pool makes. A pooled
//! texture is either in the free list OR inside exactly one
//! live lease value (moved out on `lease`, moved back on
//! [`TexturePool::release`]), so the pool can never hand the
//! same texture to two callers simultaneously, and "use a
//! texture you did not lease" has no API path.
//!
//! **Tier: only-mitigated at the wgpu-handle layer, API-shape
//! enforced at the pool layer** — wgpu handles are internally
//! reference-counted, so a caller CAN `.clone()` the inner
//! `TextureView` out of a lease (e.g. into [`crate::BoundResource`],
//! which is the intended dispatch path) and deliberately hold
//! that clone past `release`. Nothing in Rust's type system
//! revokes a cloned Arc-backed GPU handle, so use-after-release
//! is not truly unrepresentable; it requires an explicit clone
//! escape rather than being the default, which is the honest
//! ceiling for wgpu's handle model.

use std::collections::HashMap;

use crate::dispatcher::BoundResource;

/// Allocation key: textures are interchangeable iff size,
/// format, and usage all match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

impl TextureKey {
    /// The canonical post-process offscreen shape: render into
    /// it in one pass, sample it in the next
    /// (`RENDER_ATTACHMENT | TEXTURE_BINDING`). Zero dimensions
    /// are clamped to 1 — same guard mado's `ensure_offscreen`
    /// carried for minimized windows.
    #[must_use]
    pub fn offscreen(width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        }
    }
}

/// A leased pooled texture. Move-only — holding the lease IS
/// the right to use the texture (see the module doc for the
/// honest tier statement). Dropping a lease without
/// [`TexturePool::release`] simply lets wgpu free the texture;
/// safe, just forfeits reuse.
#[derive(Debug)]
pub struct TextureLease {
    key: TextureKey,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl TextureLease {
    #[must_use]
    pub fn key(&self) -> TextureKey {
        self.key
    }

    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The dispatch-path bridge: a [`BoundResource::Texture`]
    /// carrying a clone of the leased view, ready for
    /// [`crate::BoundResources`].
    #[must_use]
    pub fn bound_resource(&self) -> BoundResource {
        BoundResource::Texture {
            view: self.view.clone(),
            format: self.key.format,
        }
    }
}

/// Free-list pool of offscreen textures. One per consumer
/// render loop; lease at frame start, release at frame end.
#[derive(Debug, Default)]
pub struct TexturePool {
    free: HashMap<TextureKey, Vec<(wgpu::Texture, wgpu::TextureView)>>,
}

impl TexturePool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lease a texture matching `key` — reuses a free pooled
    /// texture when one exists, otherwise allocates.
    #[must_use]
    pub fn lease(&mut self, device: &wgpu::Device, key: TextureKey) -> TextureLease {
        if let Some((texture, view)) =
            self.free.get_mut(&key).and_then(Vec::pop)
        {
            return TextureLease { key, texture, view };
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engawa-wgpu pooled texture"),
            size: wgpu::Extent3d {
                width: key.width,
                height: key.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: key.format,
            usage: key.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureLease { key, texture, view }
    }

    /// Return a leased texture to the free list for reuse.
    pub fn release(&mut self, lease: TextureLease) {
        self.free
            .entry(lease.key)
            .or_default()
            .push((lease.texture, lease.view));
    }

    /// Total free (releasable) textures across all keys.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }

    /// Drop every pooled texture (e.g. after a resize storm
    /// left stale-size entries behind).
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offscreen_key_clamps_zero_dimensions_and_sets_postprocess_usage() {
        let key = TextureKey::offscreen(0, 0, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(key.width, 1);
        assert_eq!(key.height, 1);
        assert!(key.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(key.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn keys_differ_by_any_axis() {
        let base = TextureKey::offscreen(64, 64, wgpu::TextureFormat::Rgba8UnormSrgb);
        let wider = TextureKey::offscreen(128, 64, wgpu::TextureFormat::Rgba8UnormSrgb);
        let other_format =
            TextureKey::offscreen(64, 64, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_ne!(base, wider);
        assert_ne!(base, other_format);
        assert_eq!(
            base,
            TextureKey::offscreen(64, 64, wgpu::TextureFormat::Rgba8UnormSrgb)
        );
    }
}
