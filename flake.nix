{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.slang-gen-ninja.url = "github:expenses/slang-gen-ninja";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    slang-gen-ninja
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        slang-gen-ninja-bin = slang-gen-ninja.packages.${system}.default;

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
                slang-gen-ninja-bin
              directx-shader-compiler
              ((shader-slang.override {
                spirv-headers = (spirv-headers.overrideAttrs {
                    src = fetchFromGitHub {
                        owner = "KhronosGroup";
                        repo = "SPIRV-Headers";
                        rev = "767e901c986e9755a17e7939b3046fc2911a4bbd";
                        hash = "sha256-mXj6HDIEEjvGLO3nJEIRxdJN28/xUA2W+r9SRnh71LU=";
                      };
                });
                
              }).overrideAttrs (arrs: {
                src = fetchFromGitHub {
                  owner = "shader-slang";

                  repo = "slang";
                  rev = "e043428c5ac263fdb61072bd769fef769e8b08e4";
                  fetchSubmodules = true;
                  hash = "sha256-qyUx0fwr6pLo9fAynBlo4m7JFcs3lnTcnebA6LGqTo0=";
                };
              }))
              spirv-tools
              cargo-expand
              linuxPackages_latest.perf
              hotspot
              renderdoc
              gdb
              clang-tools
              ninja
              (writeShellScriptBin "format-shaders" ''
                ${clang-tools}/bin/clang-format -i shaders/*.slang --style '{IndentWidth: 4}'
              '')
            ];
            buildInputs = [vulkan-loader];
            LD_LIBRARY_PATH = lib.makeLibraryPath [wayland libxkbcommon];
          };
      }
    );
}
