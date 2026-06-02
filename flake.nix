{
  description = "Flake utils demo";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils = {
    inputs.nixpkgs.follows = "nixpkgs";
    url = "github:numtide/flake-utils";
  };
  inputs.slang-gen-ninja = {
    inputs.nixpkgs.follows = "nixpkgs";
    url = "github:expenses/slang-gen-ninja";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    slang-gen-ninja,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        slang-gen-ninja-bin = slang-gen-ninja.packages.${system}.default;
      in {
        devShells.default = with pkgs;
          mkShell {
            nativeBuildInputs = [
              slang-gen-ninja-bin
              shader-slang
              spirv-tools
              cargo-expand
              perf
              hotspot
              renderdoc
              gdb
              clang-tools
              cargo-udeps
              ninja
              eog
              tev
              libclang
              rustPlatform.bindgenHook
              directx-shader-compiler
              (writeShellScriptBin "format-shaders" ''
                ${clang-tools}/bin/clang-format -i shaders/**/*.slang shaders/*.slang
              '')
            ];
            buildInputs = [vulkan-loader];
            LD_LIBRARY_PATH = lib.makeLibraryPath [vulkan-loader wayland libxkbcommon libxcursor libx11 libxi];
          };
        devShells.slang-build = with pkgs;
          mkShell {
            nativeBuildInputs = [cmake ninja python3];
            buildInputs = [vulkan-headers];
          };
        devShells.vcc-testing = with pkgs;
          mkShell {
            nativeBuildInputs = [vcc spirv-tools];
          };
      }
    );
}
