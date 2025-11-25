import shared.colour;
import shared.util;
import fallback_bindings;
import thief_structs;
import shared.barycentrics;
import shared.restir;

[shader("compute")][NumThreads(8, 8, 1)]
void write(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    Uniforms uniforms = *constants.uniforms;

    if (any(thread >= uniforms.extent)) {
        return;
    }

    let as = RaytracingAccelerationStructure(uniforms.tlas);

    let d = thread_to_clip(thread, uniforms.extent);
    let origin = mul(uniforms.view_inv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = mul(uniforms.proj_inv, float4(d.x, d.y, 1.0, 1.0));
    let direction =
        mul(uniforms.view_inv, float4(normalize(temp.xyz), 0.0)).xyz;

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
        value = handle_hit(thread, query, uniforms);
    }

    RWTexture2D<float3> output = storage_images[uniforms.hdr_image];
    output[thread] = lerp(output[thread], value, 1.0/float(uniforms.accum_index+1));
}

float3 handle_hit(uint2 thread, RayQuery query, Uniforms uniforms) {
    let barycentrics = query.CommittedTriangleBarycentrics();
    let model_index = query.CommittedRayInstanceId();
    let primitive_index = query.CommittedRayPrimitiveIndex();
    let model = uniforms.models[model_index];


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
    return col * calculate_lighting_ris(thread, uniforms, position, normal);
}

float3 calculate_lighting_ris(uint2 thread, Uniforms uniforms, float3 position, float3 normal) {
    float pdf = rcp(uniforms.num_lights);
    var state = RestirState<int>(pdf, -1);

    uint num_samples = 16;
    for (uint i = 0; i < num_samples; i++) {
        uint light_index = uint(uniforms.blue_noise_sample(thread, 1+i) * uniforms.num_lights);
        Light light = uniforms.get_light(light_index);
        let vector = position - light.position;
        let distance_sq = dot(vector,vector);
        let distance = sqrt(distance_sq);

        let direction = vector / distance;

        let p_hat = length(light.color) * light.intensity * max(dot(normal, direction),0.0) / distance_sq;
        state.update(light_index, p_hat, uniforms.blue_noise_sample(thread, 21+i));
    }

    let light_index = state.get_data();

    if (light_index < 0) {
        return float3(0);
    }

    let light = uniforms.get_light(light_index);
    let vector = position - light.position;

    let distance_sq = dot(vector,vector);
    let distance = sqrt(distance_sq);

    let direction = vector / distance;

    RayDesc desc;
    desc.Origin = position;
    desc.Direction = -direction;
    desc.TMin = 1e-3f;
    desc.TMax = distance;
    var query = RayQuery<
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
    query.TraceRayInline(
        RaytracingAccelerationStructure(uniforms.tlas),
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        desc
    );
    query.Proceed();
    let visibility = float(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);

    var radiance = light.color * (visibility * light.intensity * max(dot(normal, direction),0.0) / distance_sq);

    state.update_from_frame_sample(length(radiance));
    return radiance * state.get_weight();
}

struct LightInterp {
    float4 pos: SV_Position;
    float3 col;
    float point_size: SV_PointSize;
};

[shader("vertex")] LightInterp lights_vert(uint vertex_id : SV_VertexID, uniform PushConstants constants) {
    let uniforms = constants.uniforms;
    let light = uniforms.get_light(vertex_id);

    LightInterp output;
    output.col = light.color * light.intensity;
    output.pos = mul(uniforms.proj_view, float4(light.position, 1.0));
    output.point_size = 2.0;
    return output;
}

[shader("pixel")] float3 lights_frag(LightInterp interp) {
    return interp.col;
}

[shader("compute")][NumThreads(8, 8, 1)] void resolve(uint2 thread : SV_DispatchThreadID, uniform PushConstants constants) {
    let uniforms = constants.uniforms;
    
    if (any(thread >= uniforms.extent)) {
        return;
    }

    Texture2D<float3> input = storage_images[uniforms.hdr_image];

    WTexture2D<float4> output = storage_images[uniforms.swapchain_image];
    output.Store(thread, float4(srgb_lin2encoded(input[thread]), 1));
}
