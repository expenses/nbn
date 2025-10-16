import shared.colour;

struct Interp {
    float4 pos: SV_Position;
    float3 col;
};

struct PushConstants {
    uint flags;
    float max_desired_brightness;
};

[shader("vertex")] Interp vertex(uniform PushConstants push_constants, uint vertex_id : SV_VertexID) {
    Interp interp;

    let white_triangle = push_constants.flags >> 1;

    interp.col = float3(white_triangle);
    interp.col[vertex_id] = 1.0;
    interp.pos = float4(0.0, 0.5, 0.0, 1.0);
    if (vertex_id == 0) {
        interp.pos.y = -0.5;
    } else if (vertex_id == 1) {
        interp.pos.x = 0.5;
    } else {
        interp.pos.x = -0.5;
    }

    return interp;
}

[shader("pixel")] float4 fragment(
    uniform PushConstants push_constants,Interp in
) {
    var output = in.col;

    let is_hdr = push_constants.flags & 1;

    if (is_hdr != 0) {
        output = linear_to_hdr10(output, push_constants.max_desired_brightness);
    }

    return float4(output, 1.0);
}
