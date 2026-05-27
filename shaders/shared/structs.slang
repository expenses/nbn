struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
}

struct Meshlet {
    float3 center;
    float radius;
    int8_t3 _cone_axis;
    int8_t _cone_cutoff;
    uint vertex_offset;
    uint triangle_offset;
    uint8_t vertex_count;
    uint8_t triangle_count;
    uint8_t2 pad;

    property float3 cone_axis {
        get {
            return float3(_cone_axis) / 127.0;
        }
    }

    property float cone_cutoff {
        get {
            return float(_cone_cutoff) / 127.0;
        }
    }
}
