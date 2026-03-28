import particles_structs;
import shared.colour;
import shared.util;
import shared.constants;
import shared.noise;

uint add_particle(out uint count, out uint dispatch_count) {
    uint index;
    InterlockedAdd(count, 1, index);
    InterlockedMax(dispatch_count, div_ceil(index+1,64));
    return index;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void reset(uniform PushConstants pc) {
    pc.other_draw_command.instance_count = 1;
    pc.other_draw_command.vertex_count = 0;
    pc.other_dispatch_command[0] = 0;
    pc.other_dispatch_command[1] = 1;
    pc.other_dispatch_command[2] = 1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn(uint i : SV_DispatchThreadID, uniform PushConstants pc) {
    var rng = SobolRng(uint2(0), pc.frame * 128 + i);

    let index = add_particle(pc.draw_command.vertex_count, pc.dispatch_command[0]);
    Particle* particle = &pc.particles[index];

    if (i < 96) {
        particle.position = float2(-1,-0.5)+rng.rand2() * 0.1;
        particle.velocity = (rng.rand2() - 0.5) * 0.005 + 0.005 * rng.rand();
        particle.velocity += float2(0.01,-0.01);
        particle.color = rng.rand3();
        particle.final_frame = pc.frame+128;
    } else {
        particle.position = float2(1,-0.5) - rng.rand2() * 0.05;
        particle.velocity = ((rng.rand2() - 0.5)) * 0.005 + 0.005 * rng.rand();
        particle.velocity += float2(-0.01,-0.01);
        particle.color = float3(rng.rand2(), 1.0);
        particle.final_frame = pc.frame+80;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void copy(uint thread : SV_DispatchThreadID, uniform PushConstants pc) {
    if (thread >= pc.draw_command.vertex_count) {
        return;
    }

    Particle* particle = &pc.particles[thread];

    if (particle.final_frame < pc.frame) {
        return;
    }

    let index = add_particle(pc.other_draw_command.vertex_count, pc.other_dispatch_command[0]);

    Particle* other_particle = &pc.other_particles[index];

    *other_particle = *particle;
    other_particle.position += other_particle.velocity;
    other_particle.velocity += float2(0,0.0005);
}

struct VertexOut {
    float4 pos: SV_Position;
    float3 col;
    float point_size: SV_PointSize;
};

struct FragmentIn {
    float4 pos: SV_Position;
    float3 col;
};

[shader("vertex")] VertexOut vertex(uint vertex_id : SV_VertexID, uniform PushConstants pc) {
    VertexOut interp;
    let particle = pc.particles[vertex_id];
    interp.col = particle.color;
    interp.pos = float4(particle.position, 0.0, 1.0);
    interp.point_size = 2.0;

    return interp;
}

[shader("pixel")] float4 fragment(FragmentIn in) {
    return float4(srgb_lin2encoded(in.col), 1.0);
}
