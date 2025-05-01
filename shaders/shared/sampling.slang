
// Take 13 samples in a grid around the center pixel:
// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
// These samples are interpreted as 4 overlapping boxes
// plus a center box.
float3 sample_13_tap_box_filter(Sampler2D<float3> texture, float2 uv, float2 texel_size) {
    float3 a = texture.SampleLevel(uv + texel_size * float2(-1, -1), 0);
    float3 b = texture.SampleLevel(uv + texel_size * float2(0, -1), 0);
    float3 c = texture.SampleLevel(uv + texel_size * float2(1, -1), 0);
    
    float3 d = texture.SampleLevel(uv + texel_size * float2(-0.5, -0.5), 0);
    float3 e = texture.SampleLevel(uv + texel_size * float2(0.5, -0.5), 0);
    
    float3 f = texture.SampleLevel(uv + texel_size * float2(-1, 0), 0);
    float3 g = texture.SampleLevel(uv, 0);
    float3 h = texture.SampleLevel(uv + texel_size * float2(1, 0), 0);

    float3 i = texture.SampleLevel(uv + texel_size * float2(-0.5, 0.5), 0);
    float3 j = texture.SampleLevel(uv + texel_size * float2(0.5, 0.5), 0);
    
    float3 k = texture.SampleLevel(uv + texel_size * float2(-1, 1), 0);
    float3 l = texture.SampleLevel(uv + texel_size * float2(0, 1), 0);
    float3 m = texture.SampleLevel(uv + texel_size * float2(1, 1), 0);

    float3 center_pixels = d + e + i + j;

    float3 top_left = a + b + f + g;
    float3 top_right = b + c + g + h;
    float3 bottom_left = f + g + k + l;
    float3 bottom_right = g + h + l + m;

    return center_pixels * 0.25 * 0.5 + (top_left + top_right + bottom_left + bottom_right) * 0.25 * 0.125;
}

// Sample in a 3x3 grid but with weights to produce a tent filter:
//
//        a*1 b*2 c*1
// 1/16 * d*2 e*4 f*2
//        g*1 h*2 i*1
float3 sample_3x3_tent_filter(
    Sampler2D<float3> texture, float2 uv, float2 texel_size
) {
    float3 a = texture.SampleLevel(uv + texel_size * float2(-1, -1), 0);
    float3 b = texture.SampleLevel(uv + texel_size * float2(0, -1), 0);
    float3 c = texture.SampleLevel(uv + texel_size * float2(1, -1), 0);

    float3 d = texture.SampleLevel(uv + texel_size * float2(-1, 0), 0);
    float3 e = texture.SampleLevel(uv, 0);
    float3 f = texture.SampleLevel(uv + texel_size * float2(1, 0), 0);

    float3 g = texture.SampleLevel(uv + texel_size * float2(-1, 1), 0);
    float3 h = texture.SampleLevel(uv + texel_size * float2(0, 1), 0);
    float3 i = texture.SampleLevel(uv + texel_size * float2(1, 1), 0);

    return ((a + c + g + i) + (b + d + f + h) * 2.0 + e * 4.0) / 16.0;
}
