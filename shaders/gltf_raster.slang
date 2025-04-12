import common;
import gltf;

struct PushConstants {
    float4x4 mat;
    Gltf *gltf;
};

struct Interp {
    float4 pos : SV_Position;
    float3 world_pos;
    float2 uv;
    float3 normal;
    uint material_index;
};

struct WithDerivatives<T> {
    T val;
    T dx;
    T dy;
}

func gen_derivatives<T, let N: int>(vector<T, N> val) -> WithDerivatives<vector<T, N>>
where T: __BuiltinFloatingPointType
{
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


[shader("vertex")] Interp vertex(uniform PushConstants push_constants,
                                 uint vertex_index : SV_VertexID,
                                 uint primitive_index: SV_StartInstanceLocation) {
    let primitive = push_constants.gltf.primitives[primitive_index];

    let index = primitive.read_index(push_constants.gltf, vertex_index);

    let position = primitive.positions_ptr(push_constants.gltf)[index];

    Interp out;
    out.pos = mul(push_constants.mat, float4(position, 1.0));
    out.world_pos = position;
    out.uv = primitive.uvs_ptr(push_constants.gltf)[index];
    out.normal = primitive.normals_ptr(push_constants.gltf)[index];
    out.material_index = primitive.material;

    return out;
}

[shader("pixel")]
float4 fragment(uniform PushConstants push_constants, Interp in) {
    let material = push_constants.gltf.materials[in.material_index];

    var base_colour = float3(1);
    var roughness = 1.0;
    var is_metallic = 0.0;

    var normal = normalize(in.normal);


    if (material.normal_image != max_value) {
        let sampled_normal =
            DescriptorHandle<Sampler2D>(material.normal_image)
                .Sample(in.uv).xyz;

        if (all(sampled_normal != float3(0.0))) {
            let map_normal = sampled_normal * 255.0 / 127.0 - 128.0 / 127.0;
             normal = normalize(
                mul(compute_cotangent_frame(normal, gen_derivatives(in.world_pos), gen_derivatives(in.uv)), map_normal)
            );
        }
    }

    if (material.metallic_roughness_image != max_value) {
        let metal_roughness =
            DescriptorHandle<Sampler2D>(material.metallic_roughness_image)
                .Sample(in.uv, 0);
        roughness = metal_roughness.g;
        is_metallic = metal_roughness.b;
    }

    if (material.base_colour_image != max_value) {
        float4 sample = DescriptorHandle<Sampler2D>(material.base_colour_image).Sample(in.uv);

        if (material.flags > 0 && sample.a < 0.5) {
            discard;
        }

        base_colour = sample.rgb;
    }

    let sun = normalize(float3(1.0,2.0,0.75));

    var factor = dot(sun, normal);

    factor = max(factor, .1);

    return float4(factor * base_colour, 1.0);
}
