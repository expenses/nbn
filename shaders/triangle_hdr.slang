import shared.colour;

[Flags]
enum Flags {
    IsHdr,
    WhiteTriangle,
    ClampBeforeTransferFunction,
    Tonemap
};


struct Interp {
    float4 pos: SV_Position;
    float3 col;
};

struct PushConstants {
    Flags flags;
    float max_desired_brightness;
    float exposure;
    uint tonemapping_lut;
};

[shader("vertex")] Interp vertex(uniform PushConstants push_constants, uint vertex_id : SV_VertexID) {
    Interp interp;

    interp.col = float3(push_constants.flags & Flags::WhiteTriangle);
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
    var output = in.col * pow(2, push_constants.exposure);

    if (push_constants.flags & Flags::Tonemap) {
        output = tony_mc_mapface(DescriptorHandle<Sampler3D<float3>>(push_constants.tonemapping_lut), output);
    }

    if (push_constants.flags & Flags::ClampBeforeTransferFunction) {
        output = min(output, float3(1));
    }

    if (push_constants.flags & Flags::IsHdr) {
        output = linear_to_hdr10(output, push_constants.max_desired_brightness);
    }

    return float4(output, 1.0);
}
