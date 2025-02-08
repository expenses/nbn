slang_with_args="slangc -O3 -warnings-as-errors all"
$slang_with_args shader.slang  -o shader.spv
$slang_with_args triangle.slang -o triangle.spv
$slang_with_args voxelize.slang -o voxelize.spv
slang_sort="slangc -O3 -Wno-30081 -I FidelityFX-SDK/sdk/include/FidelityFX/gpu"

#$slang_sort sort.slang -O3 -warnings-as-errors all -o sort.spv
#$slang_sort sort_d.slang -O3 -warnings-as-errors all -o sort_d.spv
