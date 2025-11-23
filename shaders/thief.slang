import shared.colour;
import shared.util;
import fallback_bindings;
import thief_models;
import shared.barycentrics;

struct PushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    float4x4 proj_view;
    uint64_t as;
    Model* models;
    float* light_values;
    uint2 extent;
    uint image;
    uint swapchain_image;
    uint num_lights;
    uint accum_index;
    uint frame_index;
    
    Light get_light(uint index) {
        Light light;
        light.position[0] = light_values[index * 7 + 0];
        light.position[1] = light_values[index * 7 + 1];
        light.position[2] = light_values[index * 7 + 2];
        light.color[0] = light_values[index * 7 + 3];
        light.color[1] = light_values[index * 7 + 4];
        light.color[2] = light_values[index * 7 + 5];
        light.intensity = light_values[index * 7 + 6];
        return light;
    }
}

    [shader("compute")][NumThreads(8, 8, 1)] void
    write(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    if (any(thread >= constants.extent)) {
        return;
    }

    let as = RaytracingAccelerationStructure(constants.as);

    let d = thread_to_clip(thread, constants.extent);
    let origin = mul(constants.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(constants.proj_inv, float4(d.x, d.y, 1.0, 1.0));
    let direction =
        mul(constants.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

    RayDesc desc;
    desc.Direction = direction;
    desc.Origin = origin;
    desc.TMax = 100000.0;
    desc.TMin = 0.0;

    var query = RayQuery<RAY_FLAG_NONE>();
    query.TraceRayInline(
        as,
        RAY_FLAG_NONE,
        0xff,
        desc
    );

    query.Proceed();

    var value = float3(0.0);

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        value = handle_hit(query, constants);
    }

    RWTexture2D<float3> output = storage_images[constants.image];
    output[thread] = lerp(output[thread], value, 1.0/float(constants.accum_index+1));
}

float3 handle_hit(RayQuery query, PushConstants constants) {
    let barycentrics = query.CommittedTriangleBarycentrics();
    let model_index = query.CommittedRayInstanceId();
    let primitive_index = query.CommittedRayPrimitiveIndex();
    let model = constants.models[model_index];


    let indices = uint3(
        model.read_index(primitive_index * 3 + 0),
        model.read_index(primitive_index * 3 + 1),
        model.read_index(primitive_index * 3 + 2),
    );

    let uv = interpolate_from_buffer(model.uvs, indices, barycentrics);
    let normal = model.normals[indices.x];
    let position = interpolate_from_buffer(model.positions, indices, barycentrics);
    let image_index = model.image_indices[primitive_index];

    Sampler2D<float3> img = sampled_images[NonUniformResourceIndex(image_index)];

    let val = img.SampleLevel(uv, 0);

    let l = normalize(float3(1,5,2));
    let f = abs(dot(normal,l));

    let col = val;//colour_from_val(image_index) + (uv.xyy %1) * 0.1;
    return col * calculate_lighting(constants,position, normal);
}


float3 calculate_lighting_raw(Light light, float pdf, float3 position, float3 normal, RaytracingAccelerationStructure tlas) {
    let vector = light.position - position;
    let distance_sq = dot(vector,vector);
    let distance = sqrt(distance_sq);

    let direction = vector / distance;

    RayDesc desc;
    desc.Origin = position;
    desc.Direction = direction;
    desc.TMin = 1e-4f;
    desc.TMax = distance;
    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();
    let visibility = float(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);

    return visibility * light.color * light.intensity / distance_sq * abs(dot(normal, direction)) / pdf;
}

float3 calculate_lighting(PushConstants constants, float3 position, float3 normal) {
    let light_index = constants.frame_index % constants.num_lights;
    let pdf = 1.0 / float(constants.num_lights);

    let light = constants.get_light(light_index);
    return calculate_lighting_raw(light, pdf, position, normal, RaytracingAccelerationStructure(constants.as));
}

bool trace_shadow_ray(RayDesc desc, RaytracingAccelerationStructure tlas) {
    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

float3 calculate_lighting2(PushConstants constants, float3 position, float3 normal) {
    let pdf = 1.0;

    var sum = float3(0.0);

    for (uint i = 0; i < constants.num_lights; i++) {
        let light = constants.get_light(i);
        sum += calculate_lighting_raw(light, pdf, position, normal, RaytracingAccelerationStructure(constants.as));
    }

    return sum;
}

struct LightInterp {
    float4 pos: SV_Position;
    float3 col;
    float point_size: SV_PointSize;
};

[shader("vertex")] LightInterp lights_vert(uint vertex_id : SV_VertexID, uniform PushConstants constants) {
    let light = constants.get_light(vertex_id);

    LightInterp output;
    output.col = light.color * light.intensity;
    output.pos = mul(constants.proj_view, float4(light.position, 1.0));
    output.point_size = 2.0;
    return output;
}

[shader("pixel")] float3 lights_frag(LightInterp interp) {
    return interp.col;
}

[shader("compute")][NumThreads(8, 8, 1)] void resolve(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    if (any(thread >= constants.extent)) {
        return;
    }

    Texture2D<float3> input = storage_images[constants.image];

    WTexture2D<float4> output = storage_images[constants.swapchain_image];
    output.Store(thread, float4(srgb_lin2encoded(input[thread]), 1));
}
