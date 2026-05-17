import shared.colour;
import shared.hdr10;

[Flags]
enum Flags {
    IsHdr,
    WhiteTriangle,
    ClampBeforeTransferFunction,
    Tonemap,
    BarelyVisibleTriangle,
};


struct Interp {
    float4 pos: SV_Position;
    float3 col;
};

struct PushConstants {
    float max_desired_brightness;
    float exposure;
    uint tonemapping_lut;
    uint background_image;
    uint8_t flags;
};

[shader("vertex")] Interp vertex(uniform PushConstants push_constants, uint vertex_id : SV_VertexID) {
    Interp interp;

    let flags = Flags(push_constants.flags);

    if (flags & Flags::BarelyVisibleTriangle) {
        interp.col = float3(1.0 / pow(2, 16));
    } else {
        interp.col = float3(bool(Flags(push_constants.flags) & Flags::WhiteTriangle));
        interp.col[vertex_id] = 1.0;
    }

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

float4 handle_output(float3 col, PushConstants push_constants) {
    var output = col * pow(2, push_constants.exposure);

    Flags flags = Flags(push_constants.flags);

    if (flags & Flags::Tonemap) {
        output = tony_mc_mapface(DescriptorHandle<Sampler3D<float3>>(push_constants.tonemapping_lut), output);
    }

    if (flags & Flags::ClampBeforeTransferFunction) {
        output = min(output, float3(1));
    }

    if (flags & Flags::IsHdr) {
        output = ConvertToHDR10(output, push_constants.max_desired_brightness);
    } else {
        output = srgb_lin2encoded(output);
    }

    return float4(output, 1.0);
}

[shader("pixel")] float4 fragment(
    uniform PushConstants push_constants,Interp in
) {
    return handle_output(in.col, push_constants);
}


struct Interp2 {
    float4 pos: SV_Position;
    float2 uv;
};

[shader("vertex")] Interp2 fullscreen_tri(uint vertex_id : SV_VertexID) {
    Interp2 interp;
    interp.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    interp.pos = float4(interp.uv * 2.0 + -1.0, 0.0, 1.0);
    return interp;
}

[shader("pixel")] float4 fragment_2(
    uniform PushConstants push_constants,Interp2 in
) {
    let image = DescriptorHandle<Sampler2D<float3>>(push_constants.background_image);
    return handle_output(image.Sample(in.uv), push_constants);
}