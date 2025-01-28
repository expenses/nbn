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
              (shader-slang.overrideAttrs (attrs: {
                src = fetchFromGitHub {
                  owner = "shader-slang";
                  repo = "slang";
                  rev = "a9ce7520e5f1b97b09e5de69455258bef55e10d2";
                  hash = "sha256-ffB0Fer9r6sqGZztywHfV0o68Zh4dRQmdItOGqBIOPI=";
                  fetchSubmodules = true;
                };
                cmakeFlags = attrs.cmakeFlags ++ ["-DSLANG_RHI_ENABLE_CUDA=0"];
              }))
              spirv-tools cargo-expand linuxPackages_latest.perf hotspot renderdoc
            ];
            buildInputs = [vulkan-loader];
          };
      }
    );
}
