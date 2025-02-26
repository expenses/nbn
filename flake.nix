{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.slang-gen-ninja.url = "github:expenses/slang-gen-ninja";

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
        vcc = with pkgs;
          stdenv.mkDerivation {
            name = "vcc";
            src = fetchGit {
              url = "https://github.com/shady-gang/shady";
              rev = "9bf761f5a7df3d9a63ee99cdf686e2f559fd8638";
              submodules = true;
            };
            nativeBuildInputs = [cmake python3 llvmPackages_18.clang-unwrapped llvmPackages_18.clang-unwrapped.lib.dev];
            buildInputs = [json_c spirv-headers   clang llvmPackages_18.libllvm llvmPackages_18.clang-unwrapped.lib];
            cmakeFlags = ["-DSHADY_USE_FETCHCONTENT=0" "-DPython_EXECUTABLE=${python3}/bin/python3"];
            NIX_CFLAGS_COMPILE = "-isystem ${llvmPackages_18.clang-unwrapped.lib}/lib/clang/18/include";
          };
      in {
        packages = {inherit render-pipeline-shaders vcc;};
        devShells.default = with pkgs;
          mkShell {
            nativeBuildInputs = [
              slang-gen-ninja-bin
              directx-shader-compiler
              shader-slang
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
        devShells.slang-build = with pkgs;
          mkShell {
            nativeBuildInputs = [cmake ninja python3];
            buildInputs = [vulkan-headers];
          };
      }
    );
}
