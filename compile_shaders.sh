rm ./shaders/compiled/*
slang-gen-ninja ./shaders/* --build-dir ./shaders/compiled --output ./build.ninja -- -O3 -warnings-as-errors all
slangc ./sorting/VkRadixSort/singleradixsort/resources/shaders/single_radixsort.comp -o shaders/compiled/radixsort.spv
ninja
