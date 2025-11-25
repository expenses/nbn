import shared.colour;
import shared.util;
import fallback_bindings;
import thief_structs;
import shared.barycentrics;
import shared.restir;
import shared.constants;

struct PrimInterp {
    float4 pos: SV_Position;
    uint prim;
};

[shader("vertex")]
PrimInterp mesh_vert(uint vertex_id : SV_VertexID, uniform PushConstants constants) {
    let uniforms = constants.uniforms;
    let model = uniforms.model;
    
    let index = model.indices[vertex_id];
    let position = model.positions[index]; 
    
    PrimInterp output;
    output.prim = vertex_id / 3;
    output.pos = mul(uniforms.proj_view, float4(position, 1.0));
    return output;
}

[shader("pixel")]
uint mesh_frag(PrimInterp interp) {
    return interp.prim;
}

float3 calculate_lighting_ris(uint2 thread, Uniforms uniforms, float3 position, float3 normal) {
    float pdf = rcp(uniforms.num_lights);
    var state = RestirState<int>(pdf, -1);

    uint num_samples = 8;
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
    Uniforms uniforms = *constants.uniforms;
    
    if (any(thread >= uniforms.extent)) {
        return;
    }

    Texture2D<uint> prim_image = storage_images[uniforms.prims_image];
    let prim = prim_image[thread];
    let value = resolve_inner(thread, uniforms, prim);

    RWTexture2D<float3> hdr = storage_images[uniforms.hdr_image];
    hdr[thread] = lerp(hdr[thread], value, 1.0/float(uniforms.accum_index+1));
    
    WTexture2D<float4> output = storage_images[uniforms.swapchain_image];
    output.Store(thread, float4(srgb_lin2encoded(hdr[thread]), 1));
}

float3 resolve_inner(uint2 thread, Uniforms uniforms, uint prim) {
    if (prim == MAX_UINT_VALUE) {
        return float3(0);   
    }
    
    let model = uniforms.model;

    let indices = uint3(
        model.read_index(prim * 3 + 0),
        model.read_index(prim * 3 + 1),
        model.read_index(prim * 3 + 2),
    );
    
    let bary_and_pos = calculate_barycentrics_from_world_positions(
        uniforms.proj_view,
        model.positions[indices.x],
        model.positions[indices.y],
        model.positions[indices.z],
        thread,
        uniforms.extent
    );
    let pos = bary_and_pos.position.val;
    let bary = bary_and_pos.barycentrics;
    
    let uv = interpolate(
        bary,
        model.uvs[indices.x],
        model.uvs[indices.y],
        model.uvs[indices.z]
    );

    let normal = model.normals[indices.x];        
    let image_index = model.image_indices[prim];

    Sampler2D<float3> img = sampled_images[NonUniformResourceIndex(image_index)];

    let col = img.SampleGrad(uv.val, uv.dx, uv.dy);

    return col * calculate_lighting_ris(thread, uniforms, pos, normal);
}