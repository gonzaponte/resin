{ self
, nixpkgs # <---- This `nixpkgs` has systems removed e.g. legacyPackages.zlib
, rust-overlay
, ...
}: let
  pkgs = import nixpkgs {
    inherit (nixpkgs.legacyPackages) system;
    overlays = [
      # ===== Specification of the rust toolchain to be used ====================
      rust-overlay.overlays.default (final: prev:
        let
          # If you have a rust-toolchain file for rustup, choose `rustup =
          # rust-tcfile` further down to get the customized toolchain
          # derivation.
          rust-tcfile  = final.rust-bin.fromRustupToolchainFile ./rust-toolchain;
          rust-latest  = final.rust-bin.stable .latest      ;
          rust-beta    = final.rust-bin.beta   .latest      ;
          rust-nightly = final.rust-bin.nightly."2024-01-16";
          rust-stable  = final.rust-bin.stable ."1.75.0"    ; # nix flake lock --update-input rust-overlay
          rust-analyzer-preview-on = date:
            final.rust-bin.nightly.${date}.default.override
              { extensions = [ "rust-analyzer-preview" ]; };
        in
          rec {
            # The version of the Rust system to be used in buildInputs. Choose between
            # tcfile/latest/beta/nightly/stable (see above) on the next line
            rustup = rust-stable;
            rustc = rustup.default;
            #cargo = rustup.default; # overriding cargo causes problems on 23.11, but we don't needed it?
            rust-analyzer-preview = rust-analyzer-preview-on "2024-01-16";
          })
#      (import ./rust-overlay.nix)
    ];

    config.allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) [
      "triton"
      "cuda_cudart"
      "cuda_nvtx"
      "torch"
    ];
  };

  python-packages = pypkgs: with pypkgs; [
    parquet
    tables
    pandas
    ipython
    pyarrow
    (if pkgs.stdenv.isx86_64 && pkgs.stdenv.isDarwin then torch else torch-bin)
    scikit-learn
    matplotlib
    jupyter
    polars
    click
  ];

  python-with-packages = (pkgs.python3.withPackages python-packages);

  rust-packages = with pkgs; [
    rust-analyzer-preview
    bacon
  ];

  other-packages = with pkgs; [ just qt5.wrapQtAppsHook ];

  in {
    # Used by `direnv` when entering this directory (also by `nix develop <URL to this flake>`)
    devShell   = pkgs.mkShell {
      name     = "resin devenv";
      packages = [ python-with-packages ] ++ rust-packages ++ other-packages;
      RUST_SRC_PATH = "${pkgs.rustup.rust-src}/lib/rustlib/src/rust/library";
      QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.libsForQt5.qt5.qtbase.bin}/lib/qt-${pkgs.libsForQt5.qt5.qtbase.version}/plugins";
    };
  }
