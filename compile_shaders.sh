rm ./shaders/compiled/*
slang-gen-ninja ./shaders/* ./shaders/*/*  --build-dir ./shaders/compiled --output ./build.ninja -x -- -O3 -warnings-as-errors all -I shaders
ninja
