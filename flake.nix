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

        render-pipeline-shaders = with pkgs;
          stdenv.mkDerivation {
            name = "rps";
            src =
              (fetchGit {
                url = "https://github.com/GPUOpen-LibrariesAndSDKs/RenderPipelineShaders";
                rev = "f3330f5306d15af8529a310f6255225c864b0961";
                submodules = true;
              })
              + "/tools/rps_hlslc/linux-x64";
            doBuild = false;
            installPhase = "cp -r $src $out";
            nativeBuildInputs = [autoPatchelfHook makeWrapper];
            buildInputs = [stdenv.cc.cc libtinfo];
            postFixup = "wrapProgram $out/bin/rps-hlslc --prefix LD_LIBRARY_PATH : $out/lib";
          };
      in {
        packages = {inherit render-pipeline-shaders;};
        devShells.default = with pkgs;
          mkShell {
            nativeBuildInputs = [
              directx-shader-compiler
              render-pipeline-shaders
              (shader-slang.overrideAttrs (arrs: {
                src = fetchFromGitHub {
                  owner = "shader-slang";

                  repo = "slang";
                  rev = "57b09a8986668626c37055e431fa0ac6449d7214";
                  fetchSubmodules = true;
                  hash = "sha256-q4qMmNbDJ/B3svaJsIU0Tl5Eak/Wmj92peb+X3B4dqM=";
                };
              }))
              spirv-tools
              cargo-expand
              linuxPackages_latest.perf
              hotspot
              renderdoc
              gdb
              clang-tools
            ];
            buildInputs = [vulkan-loader];
            LD_LIBRARY_PATH = lib.makeLibraryPath [wayland libxkbcommon];
          };
      }
    );
}
