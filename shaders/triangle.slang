struct Interp {
    float4 pos: SV_Position;
    float3 col;
};

[shader("vertex")] Interp vertex(uint vertex_id : SV_VertexID) {
    Interp interp;
    interp.col = float3(0);
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

    [shader("pixel")] float4 fragment(Interp in) {
    return float4(in.col, 1.0);
}
