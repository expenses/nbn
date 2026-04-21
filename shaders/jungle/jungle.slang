import structs;
import shared.colour;
import fallback_bindings;

struct Interp {
    float4 position: SV_Position;
    float2 uv;
    uint image;
}

[shader("vertex")]
Interp vertex(uint vertex_id : SV_VertexID, uniform PushConstants pc) {
    let instance = pc.instances[pc.instance_index];
    let index = instance.read_index(vertex_id);
    let position = instance.positions[index] * instance.scale + instance.translation;


    Interp interp;
    interp.position = mul(pc.camera, float4(position, 1.0));
    interp.uv = instance.uvs[index];
    interp.image = instance.image;
    return interp;
}

[shader("pixel")]
float3 pixel(Interp interp) {
    Sampler2D img = sampled_images[interp.image];
    return srgb_lin2encoded(img.Sample(interp.uv).rgb);
}
