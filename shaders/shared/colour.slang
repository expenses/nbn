
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
