rm ./shaders/compiled/*
slang-gen-ninja ./shaders/* --build-dir ./shaders/compiled --output ./build.ninja -- -O3 -warnings-as-errors all
ninja
