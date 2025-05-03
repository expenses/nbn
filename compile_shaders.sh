rm ./shaders/compiled/*
slang-gen-ninja ./shaders/* -I FidelityFX-SDK/sdk/include/FidelityFX/gpu  --build-dir ./shaders/compiled --output ./build.ninja -- -O3 -warnings-as-errors all
ninja
