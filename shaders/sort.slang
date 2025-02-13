#define FFX_HLSL 1
#define FFX_HLSL_SM 67
#define FFX_HALF 1
#define FFX_GPU 1
#define FFX_NO_16_BIT_CAST 1
#define FFX_PARALLELSORT_OPTION_HAS_PAYLOAD 1
#define FFX_PARALLELSORT_BIND_UAV_COUNT_SCATTER_ARGS 1
#define FFX_PARALLELSORT_BIND_UAV_REDUCE_SCAN_ARGS 1
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/parallelsort/ffx_parallelsort_common.h"
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/parallelsort/ffx_parallelsort_resources.h"

#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif

#define FFX_DECLARE_UAV_REGISTER(regIndex) u##regIndex
#define FFX_DECLARE_CB_REGISTER(regIndex) b##regIndex
#define FFX_PARALLELSORT_DECLARE_UAV(regIndex)                                 \
    register(FFX_DECLARE_UAV_REGISTER(regIndex))
#define FFX_PARALLELSORT_DECLARE_CB(regIndex)                                  \
    register(FFX_DECLARE_CB_REGISTER(regIndex))

uniform struct {
    FfxUInt32 numKeys;
    FfxInt32 numBlocksPerThreadGroup;
    FfxUInt32 numThreadGroups;
    FfxUInt32 numThreadGroupsWithAdditionalBlocks;
    FfxUInt32 numReduceThreadgroupPerBin;
    FfxUInt32 numScanValues;
    FfxUInt32 shift;
    FfxUInt32 padding;
} *cbParallelSort;

#define FFX_PARALLELSORT_CONSTANT_BUFFER_1_SIZE                                \
    8 // Number of 32-bit float/uint values in the constant buffer.

FfxUInt32 NumKeys() { return cbParallelSort.numKeys; }

FfxInt32 NumBlocksPerThreadGroup() {
    return cbParallelSort.numBlocksPerThreadGroup;
}

FfxUInt32 NumThreadGroups() { return cbParallelSort.numThreadGroups; }

FfxUInt32 NumThreadGroupsWithAdditionalBlocks() {
    return cbParallelSort.numThreadGroupsWithAdditionalBlocks;
}

FfxUInt32 NumReduceThreadgroupPerBin() {
    return cbParallelSort.numReduceThreadgroupPerBin;
}

FfxUInt32 NumScanValues() { return cbParallelSort.numScanValues; }

FfxUInt32 ShiftBit() { return cbParallelSort.shift; }

uniform FfxUInt32 *rw_source_keys;

uniform FfxUInt32 *rw_dest_keys;

uniform FfxUInt32 *rw_source_payloads;

uniform FfxUInt32 *rw_dest_payloads;

uniform FfxUInt32 *rw_sum_table;

uniform FfxUInt32 *rw_reduce_table;

uniform FfxUInt32 *rw_scan_source;

uniform FfxUInt32 *rw_scan_dest;

uniform FfxUInt32 *rw_scan_scratch;

// UAV declarations
uniform FfxUInt32 *rw_count_scatter_args;

uniform FfxUInt32 *rw_reduce_scan_args; //              ;

FfxUInt32 LoadSourceKey(FfxUInt32 index) { return rw_source_keys[index]; }

void StoreDestKey(FfxUInt32 index, FfxUInt32 value) {
    rw_dest_keys[index] = value;
}

FfxUInt32 LoadSourcePayload(FfxUInt32 index) {
    return rw_source_payloads[index];
}

void StoreDestPayload(FfxUInt32 index, FfxUInt32 value) {
    rw_dest_payloads[index] = value;
}

FfxUInt32 LoadSumTable(FfxUInt32 index) { return rw_sum_table[index]; }

void StoreSumTable(FfxUInt32 index, FfxUInt32 value) {
    rw_sum_table[index] = value;
}

void StoreReduceTable(FfxUInt32 index, FfxUInt32 value) {
    rw_reduce_table[index] = value;
}

FfxUInt32 LoadScanSource(FfxUInt32 index) { return rw_scan_source[index]; }

void StoreScanDest(FfxUInt32 index, FfxUInt32 value) {
    rw_scan_dest[index] = value;
}

FfxUInt32 LoadScanScratch(FfxUInt32 index) { return rw_scan_scratch[index]; }

void StoreCountScatterArgs(FfxUInt32 x, FfxUInt32 y, FfxUInt32 z) {
    rw_count_scatter_args[0] = x;
    rw_count_scatter_args[1] = y;
    rw_count_scatter_args[2] = z;
}

void StoreReduceScanArgs(FfxUInt32 x, FfxUInt32 y, FfxUInt32 z) {
    rw_reduce_scan_args[0] = x;
    rw_reduce_scan_args[1] = y;
    rw_reduce_scan_args[2] = z;
}

#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/parallelsort/ffx_parallelsort_setup_indirect_args.h"

[numthreads(1, 1, 1)] void main(uint3 dispatch_thread_id : SV_GroupThreadID) {
    FfxParallelSortSetupIndirectArgs(dispatch_thread_id.x);
}

#undef FFX_PARALLELSORT_COPY_VALUE
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/parallelsort/ffx_parallelsort_scan.h"

    [numthreads(1, 1, 1)] void main(uint LocalID : SV_GroupThreadID,
                                    uint GroupID : SV_GroupID) {
    FfxParallelSortScan(LocalID, GroupID);
}
