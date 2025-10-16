
public

float3 colour_from_val(uint val) {
    return float3(float(val & 1), float(val & 3) / 4, float(val & 7) / 8);
}

public

float3 srgb_lin2encoded(float3 value) {
    return select(
        value <= 0.003130,
        value * 12.92,
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    );
}

public

float3 srgb_encoded2lin(float3 value) {
    return select(
        value <= 0.04045,
        value / 12.92,
        pow((value + 0.055) / 1.055, 2.4)
    );
}

public

float3 tony_mc_mapface(Sampler3D<float3> tony_mc_mapface_lut, float3 stimulus) {
    // Apply a non-linear transform that the LUT is encoded with.
    const float3 encoded = stimulus / (stimulus + 1.0);

    // Align the encoded range to texel centers.
    const float LUT_DIMS = 48.0;
    const float3 uv = encoded * ((LUT_DIMS - 1.0) / LUT_DIMS) + 0.5 / LUT_DIMS;

    // Note: for OpenGL, do `uv.y = 1.0 - uv.y`

    return tony_mc_mapface_lut.SampleLevel(uv, 0);
}

// https://panoskarabelas.com/blog/posts/hdr_in_under_10_minutes/
public float3 linear_to_hdr10(float3 color, float max_desired_brightness_in_nits)
{
    // Convert Rec.709 to Rec.2020 color space to broaden the palette
    static const float3x3 from709to2020 =
    {
        { 0.6274040f, 0.3292820f, 0.0433136f },
        { 0.0690970f, 0.9195400f, 0.0113612f },
        { 0.0163916f, 0.0880132f, 0.8955950f }
    };   
    color = mul(from709to2020, color);

    // Normalize HDR scene values ([0..>1] to [0..1]) for ST.2084 curve
    const float st2084_max = 10000.0f;
    color *= max_desired_brightness_in_nits / st2084_max;

    // Apply ST.2084 (PQ curve) for HDR10 standard
    static const float m1 = 2610.0 / 4096.0 / 4;
    static const float m2 = 2523.0 / 4096.0 * 128;
    static const float c1 = 3424.0 / 4096.0;
    static const float c2 = 2413.0 / 4096.0 * 32;
    static const float c3 = 2392.0 / 4096.0 * 32;
    float3 cp             = pow(abs(color), m1);
    color                 = pow((c1 + c2 * cp) / (1 + c3 * cp), m2);

    return color;
}