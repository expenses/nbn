import shared.colour;

struct Vertex {
    float2 pos;
    float2 uv;
    uint8_t4 colour;
};

struct Interp {
    float4 pos: SV_Position;
    float2 uv;
    float4 colour;
}

struct Constants {
    Vertex* vertices;
    uint* indices;
    uint2 screen_size;
    float scale_factor;
    uint id;
};

[shader("vertex")] Interp
vertex(uint vertex_id : SV_VertexID, uniform Constants constants) {
    let index = constants.indices[vertex_id];
    let vertex = constants.vertices[index];
    let pos = 2.0 * (vertex.pos * constants.scale_factor)
            / float2(constants.screen_size)
        - 1.0;
    let colour = float4(vertex.colour) / 255.0;
    return {
        float4(pos, 0.0, 1.0),
        vertex.uv,
        float4(srgb_encoded2lin(colour.rgb), colour.a)
    };
}

    [shader("fragment")] float4
    fragment_srgb(Interp in, uniform Constants constants) {
    let image = DescriptorHandle<Sampler2D>(constants.id);
    return in.colour * image.Sample(in.uv).x;
}

[shader("fragment")] float4
fragment_non_srgb(Interp in, uniform Constants constants) {
    let output = fragment_srgb(in, constants);
    return float4(srgb_lin2encoded(output.rgb), output.a);
}
