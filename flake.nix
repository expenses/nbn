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
            patches = [
              ./nix/shady.patch
            ];
            nativeBuildInputs = [cmake makeWrapper];
            buildInputs = [json_c vulkan-headers vulkan-loader llvmPackages_17.libllvm];
            cmakeFlags = ["-DSHADY_USE_FETCHCONTENT=0" "-DPython_EXECUTABLE=${python3}/bin/python3" "-DBUILD_TESING=0"];
            postInstall = "wrapProgram $out/bin/vcc --prefix PATH : ${lib.makeBinPath [llvmPackages_17.clang-unwrapped]}";
          };
      in {
        packages = {inherit render-pipeline-shaders vcc;};
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
              (writeShellScriptBin "format-shaders" ''
                ${clang-tools}/bin/clang-format -i shaders/**/*.slang shaders/*.slang
              '')
            ];
            buildInputs = [vulkan-loader];
            LD_LIBRARY_PATH = lib.makeLibraryPath [wayland libxkbcommon xorg.libXcursor xorg.libX11 xorg.libXi];
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
