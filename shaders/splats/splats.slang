import shared.colour;
import shared.util;
import shared.noise;
import fallback_bindings;
import splats.structs;


[shader("compute")]
[NumThreads(64, 1, 1)]
void splat(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.num_splats) {
        return;
    }

    let splat = pc.splats[thread];

    let p = mul(pc.camera, float4(splat.center * -100, 1.0));

    var pos = p.xy / p.w;

    if (any(abs(pos) > 0.9) || p.w < 0) {
        return;
    }

    let pos2 = uint2((pos + 1) / 2.0 * pc.extent);

    pc.bitmasks[pos2.x + pos2.y * pc.extent.x] = uint(splat.dc.r * 255) | (uint(splat.dc.g * 255) << 8) | (uint(splat.dc.b * 255) << 16);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void output(uint2 thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (any(thread >= pc.extent)) {
        return;
    }

    let mask = pc.bitmasks[thread.x + thread.y * pc.extent.x];

    let value = float3(mask & 255, (mask >> 8) & 255, (mask >> 16) & 255) / 255;

    WTexture2D<float4> output = storage_images[pc.image];
    output.Store(thread, float4(srgb_lin2encoded(value), 1));
}
