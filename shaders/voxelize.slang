import gltf;
import shared.constants;

func fetch_3<T>(T* values, uint[3] indices) -> T[3] {
    return {values[indices[0]], values[indices[1]], values[indices[2]]};
}

struct Interp {
    float4 pos: SV_Position;
    float2 uv;
    uint material_index;
    uint8_t swizzle;
};

struct PushConstants {
    uint4 tile_offset;
    Gltf* gltf;
    uint3* output;
    Atomic<uint>* num_outputs;
    uint dim_size;
    uint output_size;
    float scale;
    uint num_tiles_on_side;
};

float3 normal_for_triangle(float3[3] positions) {
    let u = positions[1] - positions[0];
    let v = positions[2] - positions[0];
    return cross(u, v);
}

[shader("vertex")] Interp
vertex(uniform PushConstants push_constants, uint vertex_id : SV_VertexID) {
    uint primitive_index;
    uint index_offset = 0;
    uint num_indices;
    Primitive primitive;

    for (primitive_index = 0;
         primitive_index < push_constants.gltf.num_primitives;
         primitive_index++) {
        primitive = push_constants.gltf.primitives[primitive_index];
        num_indices = primitive.num_indices(push_constants.gltf);
        let total_index_count = index_offset + num_indices;
        if (vertex_id < total_index_count) {
            break;
        }
        index_offset = total_index_count;
    }

    let vertex_index = vertex_id - index_offset;

    let vertex_in_triangle = vertex_id % 3;

    let start_vertex = vertex_index - vertex_in_triangle;

    uint[3] indices = {
        primitive.read_index(push_constants.gltf, start_vertex),
        primitive.read_index(push_constants.gltf, start_vertex + 1),
        primitive.read_index(push_constants.gltf, start_vertex + 2)
    };
    var positions =
        fetch_3(primitive.positions_ptr(push_constants.gltf), indices);
    var position = positions[vertex_in_triangle];
    let normal = abs(normal_for_triangle(positions));
    let max = max3(normal.x, normal.y, normal.z);

    // Scale down
    position /= push_constants.scale;
    // Scale up by num times to a size (so a scene normalized to -1,1 might
    // become -4,4)
    position *= push_constants.num_tiles_on_side;
    // Offset for the specific tile (*2 to turn 1,3 into -1,1)
    position -= push_constants.tile_offset.xzy * 2;
    // Offset to put the middle tile into the middle.
    position += (push_constants.num_tiles_on_side / 2) * 2;

    uint8_t swizzle;
    if (normal.z == max) {
        // do nothing
        swizzle = 0;
    } else if (normal.y == max) {
        swizzle = 1;
        position = position.xzy;
    } else {
        swizzle = 2;
        position = position.yzx;
    }

    let index = indices[vertex_in_triangle];

    Interp interp;
    interp.uv = primitive.uvs_ptr(push_constants.gltf)[index];
    interp.material_index = primitive.material;

    interp.pos = float4(position.xy, (position.z / 2) + 0.5, 1.0);
    interp.swizzle = swizzle;
    return interp;
}

uint64_t separate_bits_64(uint32_t n) {
    uint64_t n = n;
    n = (n ^ (n << 32))
        & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
    n = (n ^ (n << 16))
        & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
    n = (n ^ (n << 8))
        & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
    n = (n ^ (n << 4))
        & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
    n = (n ^ (n << 2))
        & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
    return n;
};

// works for integers up to 2^21
uint64_t morton_encode(uint3 xyz) {
    return separate_bits_64(xyz.x) | (separate_bits_64(xyz.y) << 1)
        | (separate_bits_64(xyz.z) << 2);
}

uint8_t posterize(float value, uint bits) {
    return uint8_t(round(value * (pow(2, bits) - 1)));
}

func linear_to_srgb<int N>(value : vector<float, N>) -> vector<uint8_t, N> {
    let srgb = select(
        value <= 0.0031308,
        value * 12.92,
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    );
    // Convert to u8 (0-255 range)
    return vector<uint8_t, N>(round(srgb * 255.0));
}

[shader("pixel")] void
fragment(Interp values, uniform PushConstants push_constants, ) {
    values.pos.z *= push_constants.dim_size;

    if (values.swizzle == 1) {
        values.pos.xzy = values.pos.xyz;
    } else if (values.swizzle == 2) {
        values.pos.xyz = values.pos.zxy;
    }

    values.pos.yz = values.pos.zy;

    let material = push_constants.gltf.materials[values.material_index];
    var base_colour = float3(1);

    if (material.base_colour_image != MAX_UINT_VALUE) {
        float4 sample = DescriptorHandle<Sampler2D>(material.base_colour_image)
                            .SampleLevel(values.uv, 0);

        if (sample.a < 0.5) {
            return;
        }

        base_colour = sample.rgb;
    }

    var emissive_sample = float3(0);
    if (material.emissive_image != MAX_UINT_VALUE) {
        emissive_sample = DescriptorHandle<Sampler2D>(material.emissive_image)
                              .SampleLevel(values.uv, 0)
                              .rgb;
    }

    bool is_emissive = any(emissive_sample > 0.1);

    if (is_emissive) {
        base_colour = emissive_sample;
    }

    var roughness = 1.0;
    var is_metallic = false;

    if (material.metallic_roughness_image != MAX_UINT_VALUE) {
        let metal_roughness =
            DescriptorHandle<Sampler2D>(material.metallic_roughness_image)
                .SampleLevel(values.uv, 0);
        roughness = metal_roughness.g;
        is_metallic = metal_roughness.b > 0.5;
    }

    // Needs to be -1 so that 4096 becomes 4095 and morton encoding for tiles
    // works.
    let position = clamp(uint3(values.pos.xyz), 0, push_constants.dim_size - 1)
        + push_constants.tile_offset.xyz * push_constants.dim_size;

    let emissive_factor = 50;

    var aux = is_emissive ? log10(emissive_factor) / 4 : roughness;

    let output_index = push_constants.num_outputs.add(1);
    if (output_index < push_constants.output_size) {
        let encoded = morton_encode(position);
        push_constants.output[output_index] = uint3(
            uint(encoded >> 32),
            uint(encoded),
            reinterpret<uint, uint8_t4>(uint8_t4(
                linear_to_srgb(base_colour),
                uint8_t(is_emissive) | (uint8_t(is_metallic) << 1)
                    | (posterize(aux, 6) << 2)
            ))
        );
    }
}
