import fallback_bindings;

struct Interp {
    float4 pos: SV_Position;
    float3 normal;
    float3 color;
};

float4x4 perspective_reversed_infinite_z_vk(
    float vertical_fov,
    float aspect_ratio,
    float z_near,
)  {
    let t = tan(vertical_fov / 2.0);

    let sy = 1.0 / t;

    let sx = sy / aspect_ratio;

    // Construct a matrix from a list of rows
    return float4x4(
        float4(sx, 0.0, 0.0, 0.0),
        float4(0.0, -sy, 0.0, 0.0),
        float4(0.0, 0.0, 0.0, z_near),
        float4(0.0, 0.0, -1.0, 0.0),
    );
}

float4x4 look_to_rh(float3 eye, float3 dir, float3 up) {
    let f = dir;
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    return float4x4(
        float4(s, -dot(eye, s)),
        float4(u, -dot(eye, u)),
        float4(-f, dot(eye, f)),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

float4x4 look_at_rh(float3 eye, float3 center, float3 up) {
    return look_to_rh(eye, normalize(center - eye), up);
}

[shader("vertex")]
Interp vertex(uint* address, uniform uint image ,uint vertex_id : SV_VertexID) {

    Sampler2D s_image = sampled_images[image];

    let num_indices = address[0];
    let num_vertices = address[1];
    uint* indices = address+2;
    float3* positions = (float3*) (indices + num_indices);
    float3* normals = positions + num_vertices;

    let index = indices[vertex_id];
    let position = positions[index];
    let normal = normals[index];
    
    let proj = perspective_reversed_infinite_z_vk(1.0297443, 1.0, 0.001);
    let view = look_at_rh(float3(-12.0, 4.0, -12.0), float3(0.0), float3(0.0, 0.0, -1.0));

    Interp interp;
    interp.pos = mul(mul(proj, view), float4(position, 1.0));
    interp.normal = normal;
    interp.color = s_image.SampleLevel(float2(0.5), 0).rgb;
    return interp;
}

[shader("pixel")]
float4 fragment(Interp in) {
    let normal = normalize(in.normal);
    let lighting = max(dot(normal, float3(0, 0, -1)), 0);
    return float4(lighting * in.color, 1.0);
}
