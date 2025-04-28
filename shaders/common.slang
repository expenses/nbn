static uint max_value = 4294967295;

float3 colour_from_val(uint val) {
    return float3(float(val & 1), float(val & 3) / 4, float(val & 7) / 8);
}

float3 srgb_lin2encoded(float3 value) {
    return select(
        value <= 0.003130,
        value * 12.92,
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    );
}

float3 srgb_encoded2lin(float3 value) {
    return select(
        value <= 0.04045,
        value / 12.92,
        pow((value + 0.055) / 1.055, 2.4)
    );
}

struct WithDerivatives<T> {
    T val;
    T dx;
    T dy;
};

func gen_derivatives<T, let N : int>(vector<T, N> val)
    -> WithDerivatives<vector<T, N>> where T: __BuiltinFloatingPointType {
    WithDerivatives<vector<T, N>> out;
    out.val = val;
    out.dx = ddx(val);
    out.dy = ddy(val);
    return out;
}

float length_squared(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Adapted from http://www.thetenthplanet.de/archives/1180
float3x3 compute_cotangent_frame(
    float3 normal,
    WithDerivatives<float3> position,
    WithDerivatives<float2> uv
) {
    float3 delta_pos_y_perp = cross(position.dy, normal);
    float3 delta_pos_x_perp = cross(normal, position.dx);

    float3 t = delta_pos_y_perp * uv.dx.x + delta_pos_x_perp * uv.dy.x;
    float3 b = delta_pos_y_perp * uv.dx.y + delta_pos_x_perp * uv.dy.y;

    float invmax = 1.0 / sqrt(max(length_squared(t), length_squared(b)));
    return float3x3(t * invmax, b * invmax, normal);
}

uint div_ceil(uint value, uint divisor) {
    return (value + divisor - 1) / divisor;
}
