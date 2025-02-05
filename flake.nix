{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = with pkgs;
          mkShell {
            nativeBuildInputs = [
              shader-slang
              spirv-tools
              cargo-expand
              linuxPackages_latest.perf
              hotspot
              renderdoc
              gdb
            ];
            buildInputs = [vulkan-loader];
            LD_LIBRARY_PATH = lib.makeLibraryPath [wayland libxkbcommon];
          };
      }
    );
}
