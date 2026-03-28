import shared.util;

struct Particle {
    float2 position;
    float2 velocity;
    float3 color;
    uint final_frame;
}

struct PushConstants {
    Particle* particles;
    DrawIndirectCommand* draw_command;
    uint* dispatch_command;
    Particle* other_particles;
    DrawIndirectCommand* other_draw_command;
    uint* other_dispatch_command;
    uint frame;
}
