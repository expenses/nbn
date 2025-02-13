rm -r shaders/compiled/*
slang_with_args="slangc -O3 -warnings-as-errors all"
$slang_with_args shaders/triangle.slang -o shaders/compiled/triangle.spv
$slang_with_args shaders/voxelize.slang -o shaders/compiled/voxelize.spv
#$slang_with_args VkRadixSort/singleradixsort/resources/shaders/single_radixsort.comp -o shaders/compiled/radixsort.spv
#$slang_with_args VkRadixSort/multiradixsort/resources/shaders/multi_radixsort_histograms.comp -o shaders/compiled/multi_radixsort_histograms.spv
