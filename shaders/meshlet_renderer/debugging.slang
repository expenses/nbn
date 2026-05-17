import uniforms;
import models;
import shared.constants;
import shared.util;
import shared.colour;

struct PC {
    Model* model;
    UniformBuffer* uniforms;
};

struct Interp {
    float4 pos: SV_Position;
    float2 uv;
};

[shader("vertex")] Interp
model_debug_vertex(uniform PC constants, uint vertex_index : SV_VertexID) {
    let index = constants.model.read_index(vertex_index);
    let position = constants.model.positions[index];
    let uv = constants.model.uvs[index];
    return {mul(constants.uniforms.mat, float4(position, 1.0)), uv};
}

    [shader("vertex")] Interp meshlet_debug_vertex(
        uniform PC constants,
        uint vertex_index : SV_VertexID,
        uint start_instance : SV_StartInstanceLocation
    ) {
    let model = constants.model;

    let meshlet = model.meshlets[start_instance];

    let index = model.vertices
                    [meshlet.vertex_offset
                     + model.triangles[meshlet.triangle_offset + vertex_index]];
    let position = model.positions[index];
    let uv = model.uvs[index];
    return {mul(constants.uniforms.mat, float4(position, 1.0)), uv};
}

[shader("pixel")] float4 model_debug_fragment(Interp in, uniform PC constants) {
    float3 base_colour = float3(in.uv, .5);

    let material = constants.model.material;

    if (material.base_colour_image != MAX_UINT_VALUE) {
        base_colour = DescriptorHandle<Sampler2D>(material.base_colour_image)
                          .Sample(in.uv)
                          .rgb;
    }

    return float4(srgb_lin2encoded(base_colour), 1.0);
}
