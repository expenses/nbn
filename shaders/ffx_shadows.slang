#define FFX_HLSL 1
#define FFX_GPU 1
#define FFX_HLSL_SM 66
#define FFX_HALF 1
#define FFX_WAVE 1
#define FFX_DENOISER_OPTION_INVERTED_DEPTH 1
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/ffx_common_types.h"
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/ffx_core_hlsl.h"
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/ffx_core_gpu_common.h"
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/denoiser/ffx_denoiser_shadows_filter.h"
#include "../FidelityFX-SDK/sdk/include/FidelityFX/gpu/denoiser/ffx_denoiser_shadows_tileclassification.h"

import uniforms;
import shared.linalg;
import shared.util;

[vk::binding(0, 0)] __DynamicResource sampled_images[];
[vk::binding(1, 0)] __DynamicResource storage_images[];


[[vk::push_constant]] uniform UniformBuffer* uniforms;

int2 BufferDimensions()
{
    return uniforms.extent;
}

float2 InvBufferDimensions() {
    return 1.0 / float2(BufferDimensions());
}

float LoadDepth(int2 p) {
    let uv = thread_to_uv(p, uniforms.extent);
    Sampler2D<float> depthbuffer = sampled_images[uniforms.depthbuffer];
    return depthbuffer.SampleLevel(uv, 0);
}

float4x4 ProjectionInverse() {
    return inverse(uniforms.perspective);
}

float4x4 ViewProjectionInverse() {
    return uniforms.mat_inv;
}

float4x4 ReprojectionMatrix() {
    return float4x4(0);
}

FfxFloat32x3 Eye() {
    return uniforms.camera_position;
}


FfxFloat32 LoadHistory(FfxFloat32x2 p) {
    return 1.0;
}

FfxFloat32 LoadPreviousDepth(FfxInt32x2 p) {
    return 1.0;
}

float3 LoadNormals(int2 p) {
    return float3(0);
}

FfxFloat32x3 LoadPreviousMomentsBuffer(FfxInt32x2 p) {
    return float3(0);
}

float16_t2 LoadFilterInput(int2 p) {
    return float16_t2(0);
}

float DepthSimilaritySigma() {
    return 1.0;
}

bool IsFirstFrame() {
    return true;
}

FfxBoolean IsShadowReciever(FfxUInt32x2 p)
{
    FfxFloat32 depth = LoadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

FfxUInt32 LoadTileMetaData(FfxUInt32 p) {
    return 0;
}

FfxUInt32 LoadRaytracedShadowMask(FfxUInt32 p)
{
    return 0;
}

void StoreHistory(FfxUInt32x2 p, FfxFloat32x2 val) {}

void StoreFilterOutput(FfxUInt32x2 p, FfxFloat32 val){}

void StoreMoments(FfxUInt32x2 p, FfxFloat32x3 val){}

void StoreReprojectionResults(FfxUInt32x2 p, FfxFloat32x2 val){}

void StoreMetadata(FfxUInt32 p, FfxUInt32 val) {}

FfxFloat32x2 LoadVelocity(FfxInt32x2 p){
    return float2(0);
}

// Helpers

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
static const float k_pushOff = 4e-2f;
static const uint2 k_tileSize = uint2(TILE_SIZE_X, TILE_SIZE_Y);


uint LaneIdToBitShift(uint2 localID)
{
    return localID.y * k_tileSize.x + localID.x;
}

uint BoolToWaveMask(bool b, uint2 localID)
{
    const uint value = b << LaneIdToBitShift(localID);
    return WaveActiveBitOr(value);
}

bool WaveMaskToBool(uint mask, uint2 localID)
{
    return bool((1u << LaneIdToBitShift(localID)) & mask);
}
