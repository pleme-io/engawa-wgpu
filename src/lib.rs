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
//! **Scope (NOT v0.1):** texture allocation, swapchain
//! management, bind-group authoring. Engawa's `ResourceBindings`
//! is the operator-facing handoff for those — the consumer
//! allocates / reuses wgpu textures + binds them to engawa
//! `ResourceId`s. This crate dispatches; the consumer feeds it.

#![forbid(unsafe_code)]
#![doc(html_root_url = "https://docs.rs/engawa-wgpu/0.1.0")]

mod dispatcher;
mod pipeline;

pub use dispatcher::{
    BoundResource, BoundResources, WgpuDispatcher, WgpuDispatcherError,
};
pub use pipeline::FULLSCREEN_VERTEX_WGSL;
