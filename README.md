This repo holds the code for the development of a NN for reconstruction from raw SiPM signals in NEXT. It is built using Nix. In order to set it up on your own (good luck), look at the dependencies in `flake/outputs.nix`.

# Setup

At the moment no apps are defined, so you need to clone the repo
```
git clone github:gonzaponte/resin.git # or git clone https://github/gonzaponte/resin.git
cd resin
```

## With direnv
Nothing, you are done :)

## Without direnv
You will need to prepend `nix develop .# -c --` to every command
```
nix develop .# -c -- <command>
```

or run `nix develop .#` that will put you in an ugly shell with the environment set up.

## Available commands
- `just rust/debug`: compiles the analytical psf data generator in debug mode
- `just rust/release`: compiles the analytical psf data generator in release mode (quietly)
- `just rust/run [args...]`: runs the analytical psf data generator
