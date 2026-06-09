{
  description = "engawa-wgpu — wgpu-backed Dispatcher impl for engawa render graphs";

  inputs = {
    nixpkgs.follows = "substrate/nixpkgs";
    crate2nix.url = "github:nix-community/crate2nix";
    flake-utils.url = "github:numtide/flake-utils";
    substrate = {
      url = "github:pleme-io/substrate";
    };
  };

  outputs = { self, nixpkgs, crate2nix, flake-utils, substrate, ... }:
    (import "${substrate}/lib/rust-library-flake.nix" {
      inherit nixpkgs crate2nix flake-utils;
    }) {
      libName = "engawa-wgpu";
      src = self;
      repo = "pleme-io/engawa-wgpu";
    };
}
