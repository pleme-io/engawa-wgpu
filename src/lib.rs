//! wgpu-backed `Dispatcher` impl for engawa render graphs.
//!
//! engawa owns the typed IR; this crate owns the wgpu wiring.
//! Operators construct a `WgpuDispatcher` once per graph (or
//! once per consumer lifetime), pass it the device + queue + a
//! target format, and call `dispatch_graph` from their render
//! loop.
//!
//! **Scope (v0.1):** fullscreen-effect graphs. Every node with
//! a `Material` is dispatched as a render-pass that draws a
//! single fullscreen triangle through a built-in vertex shader,
//! with the operator's WGSL providing the fragment. Compute and
//! blit passes are pending — same trait, follow-on work.
//!
//! **Per-call dispatch is canonical:** `WgpuDispatcher::new`
//! clones the (internally reference-counted) device/queue
//! handles — no lifetime borrow — and `dispatch_with` takes the
//! graph + bindings + live handles + per-frame [`FrameUniforms`]
//! every call. Offscreen ping-pong textures come from the
//! [`TexturePool`] lease API.
//!
//! **Scope (NOT v0.1):** swapchain management, bind-group
//! authoring. Engawa's `ResourceBindings` is the operator-facing
//! handoff for those — the consumer binds wgpu handles to engawa
//! `ResourceId`s. This crate dispatches; the consumer feeds it.

#![forbid(unsafe_code)]
#![doc(html_root_url = "https://docs.rs/engawa-wgpu/0.1.0")]

pub mod catalog;
mod dispatcher;
mod pipeline;
mod pool;

pub use dispatcher::{
    BoundResource, BoundResources, FrameUniforms, WgpuDispatcher, WgpuDispatcherError,
};
pub use pipeline::{combined_shader_source, FULLSCREEN_VERTEX_WGSL};
pub use pool::{TextureKey, TextureLease, TexturePool};
