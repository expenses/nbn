[vk::binding(0,0)] Sampler2D sampled_images[];
[vk::binding(1,0)] __DynamicResource storage_images[];

import common;
import gltf;
import models;
import visbuffer;

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

struct BlitPC {
float4x4 mat;
    uint2 extent;
    Instance* instances;
    MeshletInstance* meshlet_instances;
    uint input_image;
    uint output_image;
};

float3 linear_to_srgb_transfer_function(float3 value) {
    return select(
        value <= 0.003130,
        value * 12.92,
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    );
}

float4 sample(uint index, WithDerivatives<float2> uv) {
    return sampled_images[NonUniformResourceIndex(index)].SampleGrad(uv.val, uv.dx, uv.dy);
}

[shader("compute")]
[NumThreads(8, 8, 1)]
void resolve_visbuffer(uint2 thread: SV_DispatchThreadID, uniform BlitPC constants) {
    Texture2D<uint> input_image = storage_images[constants.input_image];
    RWTexture2D<float3> output_image = storage_images[constants.output_image];

    if (any(thread > constants.extent)) {
        return;
    }

    let val = input_image[thread];

    if (val == max_value) {
        output_image.Store(thread, float3(1,1,0));
        return;
    }

    let meshlet_instance_index = val >> 7;
    let triangle_index = val & 0b1111111;
    let meshlet_instance = constants.meshlet_instances[meshlet_instance_index];
    let meshlet = meshlet_instance.meshlet;
    let instance = constants.instances[meshlet_instance.instance_index];
    let model = instance.model;
    let material = model.material;

    let indices = uint3(
        model.vertices[meshlet.vertex_offset + model.triangles[meshlet.triangle_offset + triangle_index * 3 + 0]],
        model.vertices[meshlet.vertex_offset + model.triangles[meshlet.triangle_offset + triangle_index * 3 + 1]],
        model.vertices[meshlet.vertex_offset + model.triangles[meshlet.triangle_offset + triangle_index * 3 + 2]],
    );

    let pos_a = model.positions[indices.x];
    let pos_b = model.positions[indices.y];
    let pos_c = model.positions[indices.z];

    let ndc =
        float2(thread) / float2(constants.extent) * 2.0
        - 1.0;

    let bary = CalcFullBary(
        mul(constants.mat, float4(pos_a, 1.0)),
        mul(constants.mat, float4(pos_b, 1.0)),
        mul(constants.mat, float4(pos_c, 1.0)),
        ndc,
        constants.extent
    );

    let pos = interpolate(bary, pos_a, pos_b, pos_c);

    let uv = interpolate(bary, model.uvs[indices.x], model.uvs[indices.y], model.uvs[indices.z]);

    var normal = normalize(interpolate(bary, model.normals[indices.x], model.normals[indices.y], model.normals[indices.z]).val);

    var base_colour = float3(1);
    var roughness = 1.0;
    var is_metallic = 0.0;

    if (material.normal_image != max_value) {
        let sampled_normal = sample(material.normal_image, uv).xyz;

        if (all(sampled_normal != float3(0.0))) {
            let map_normal = sampled_normal * 255.0 / 127.0 - 128.0 / 127.0;
             normal = normalize(
                mul(compute_cotangent_frame(normal, pos, uv), map_normal)
            );
        }
    }

    if (material.metallic_roughness_image != max_value) {
        let metal_roughness =sample(material.metallic_roughness_image, uv);
        roughness = metal_roughness.g;
        is_metallic = metal_roughness.b;
    }

    if (material.base_colour_image != max_value) {
        float4 sample = sample(material.base_colour_image, uv);
        base_colour = sample.rgb;
    }

    let sun = normalize(float3(1.0,2.0,0.75));

    var factor = dot(sun, normal);

    factor = max(factor, .1);

    let output = factor * base_colour;
    output_image.Store(thread, linear_to_srgb_transfer_function(output));
}
