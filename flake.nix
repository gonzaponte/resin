{
  description = "Implementation of a NN for reconstruction from raw SiPM signals in NEXT";

  inputs = {
    nixpkgs         .url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-compat = { url = "github:edolstra/flake-compat"; flake = false; };
    nosys           .url = "github:divnix/nosys";
    rust-overlay = { url = "github:oxalica/rust-overlay"; inputs.nixpkgs.follows = "nixpkgs"; };
  };

  outputs = inputs @ {
    nosys,
    nixpkgs, # <---- This `nixpkgs` still has the `system` e.g. legacyPackages.${system}.zlib
    ...
  }: let outputs = import ./flake/outputs.nix;
         systems = [ "x86_64-linux" ];
    in nosys (inputs // { inherit systems; }) outputs;
}
