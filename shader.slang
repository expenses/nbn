#include "gltf.slang"

interface IVisitor {
    void visit(int3 pos);
}

void VoxelizeTriangleSurface<V: IVisitor>(float3 v[3], V visitor) {
    float3 e[3] = { v[1] - v[0], v[2] - v[1], v[0] - v[2] };

    let norm = normalize(cross(e[0], e[1]));
    let c = max(sign(norm), 0.0f);  // critical point

    let d1 = dot(norm, c - v[0]);
    let d2 = dot(norm, (1.0f - c) - v[0]);

    float2 ne[3][3];
    float de[3][3];

    for (uint32_t i = 0; i < 3; i++) {
        ne[0][i] = float2(-e[i].y, e[i].x) * (norm.z < 0 ? -1.0f : +1.0f);
        de[0][i] = -dot(ne[0][i], float2(v[i].x, v[i].y)) + max(0.0f, ne[0][i].x) + max(0.0f, ne[0][i].y);

        ne[1][i] = float2(-e[i].x, e[i].z) * (norm.y < 0 ? -1.0f : +1.0f);
        de[1][i] = -dot(ne[1][i], float2(v[i].z, v[i].x)) + max(0.0f, ne[1][i].x) + max(0.0f, ne[1][i].y);

        ne[2][i] = float2(-e[i].z, e[i].y) * (norm.x < 0 ? -1.0f : +1.0f);
        de[2][i] = -dot(ne[2][i], float2(v[i].y, v[i].z)) + max(0.0f, ne[2][i].x) + max(0.0f, ne[2][i].y);
    }

    int3 boundMin = int3(min(min(v[0], v[1]), v[2]));
    int3 boundMax = int3(max(max(v[0], v[1]), v[2]));

    // TODO: Test specialization (iterate only over 2D plane for dominant axis)
    for (int y = boundMin.y; y <= boundMax.y; y++) {
        for (int z = boundMin.z; z <= boundMax.z; z++) {
            for (int x = boundMin.x; x <= boundMax.x; x++) {
                float3 p = { x, y, z };

                // Triangle plane overlap check
                float NdotP = dot(norm, p);
                if ((NdotP + d1) * (NdotP + d2) > 0.0f) continue;

                // 2D projection overlap checks
                bool overlaps = true;

                for (uint32_t i = 0; i < 3 && overlaps; i++) {
                    overlaps &= dot(ne[0][i], float2(p.x, p.y)) + de[0][i] >= 0.0f;
                    overlaps &= dot(ne[1][i], float2(p.z, p.x)) + de[1][i] >= 0.0f;
                    overlaps &= dot(ne[2][i], float2(p.y, p.z)) + de[2][i] >= 0.0f;
                }

                if (overlaps) visitor.visit({ x, y, z });
            }
        }
    }
}

float3 ProjectPointOnTriangle(float3 p, const float3 vtx[3]) {
    let u = vtx[1] - vtx[0];
    let v = vtx[2] - vtx[0];
    let n = cross(u, v);
    let w = p - vtx[0];
    // Barycentric coordinates of the projection P′of P onto T:
    // γ=[(u×w)⋅n]/n²
    let gamma = dot(cross(u, w), n) / dot(n, n);
    // β=[(w×v)⋅n]/n²
    float beta = dot(cross(w, v), n) / dot(n, n);
    float alpha = 1 - gamma - beta;

    return { alpha, beta, gamma };
}

func interpolate<T: IFloat>(T[3] values, float3 barycentrics) -> T {
    return values[0] * T(barycentrics.x) + values[1] * T(barycentrics.y) + values[2] * T(barycentrics.z);
}


func fetch_3<T, I: IInteger>(T* values, I* indices) -> T[3] {
    return {
        values[indices[0].toUInt()],
        values[indices[1].toUInt()],
        values[indices[2].toUInt()]
    };
}

uint8_t posterize(float value, uint bits) {
    return uint8_t(round(value * (pow(2, bits) - 1)));
}

func linear_to_srgb<int N>(value: vector<float, N>) -> vector<uint8_t, N> {
    let srgb = select(value <= 0.0031308, value * 12.92, 1.055 * pow(value, 1.0 / 2.4) - 0.055);
    // Convert to u8 (0-255 range)
    return vector<uint8_t, N>(round(srgb * 255.0));
}

struct Writer: IVisitor {
    uint8_t4* output;
    uint scene_size;
    float3[3] positions;
    float2[3] uvs;
    Optional<DescriptorHandle<Sampler2D>> base_colour_image;
    Optional<DescriptorHandle<Sampler2D>> emissive_image;
    Optional<DescriptorHandle<Sampler2D>> metal_roughness_image;
    bool is_alpha_clipped;

    void visit(int3 pos) {
        if (any(pos >= scene_size) || any(pos < 0)) {
            return;
        }

        let barycentrics = ProjectPointOnTriangle(pos, positions);
        let uv = interpolate(uvs, barycentrics);

        var base_colour = float3(1);

        if (let base_colour_image = base_colour_image) {
            float4 sample = base_colour_image.SampleLevel(uv, 0);

            if (is_alpha_clipped && sample.a < 0.5) {
                return;
            }

            base_colour = sample.rgb;
        }

        var emissive_sample = float3(0);
        if (let emissive_image = emissive_image) {
            emissive_sample = emissive_image.SampleLevel(uv, 0).rgb;
        }

        bool is_emissive = any(emissive_sample > 0.1);

        if (is_emissive) {
            base_colour = emissive_sample;
        }

        var roughness = 1.0;
        var is_metallic = false;

        if (let metal_roughness_image = metal_roughness_image) {
             let metal_roughness = metal_roughness_image.SampleLevel(uv, 0);
            roughness = metal_roughness.g;
            is_metallic = metal_roughness.b > 0.5;
        }

        output[pos.x + pos.y * scene_size + pos.z * scene_size * scene_size] = uint8_t4(
            linear_to_srgb(base_colour),
            uint8_t(is_emissive) | (uint8_t(is_metallic) << 1) | (posterize(roughness, 6) << 2)
        );
    }
}

bool outside(float3 vertex, uint scene_size) {
    return any(vertex < 0) || any(vertex >= scene_size);
}

[numthreads(64,1,1)]
void main(
    // this is sadly not allowed to be a uint8 type as that causes a validation
    // error without `storagePushConstant8` enabled even though it shouldn't require it.
    uint32_t* output,
    Gltf* gltf,
    uniform uint scene_size,
    uint3 dispatch_thread_id : SV_DispatchThreadID,
) {
    uint primitive_index;
    uint triangle_offset = 0;
    uint num_triangles;
    Primitive primitive;
    bool broke_out = false;

    let thread = dispatch_thread_id.x;

    for (primitive_index = 0; primitive_index < gltf.num_primitives; primitive_index++) {
        primitive = gltf.primitives[primitive_index];
        num_triangles = primitive.num_indices(gltf) / 3;
        let total_triangle_count = triangle_offset + num_triangles;
        if (thread < total_triangle_count) {
            broke_out = true;
            break;
        }
        triangle_offset = total_triangle_count;
    }

    if (!broke_out) {
        return;
    }

    let triangle_index = thread - triangle_offset;

    let indices = primitive.indices_ptr(gltf) + triangle_index * 3;
    let positions = primitive.positions_ptr(gltf);

    let material = gltf.materials[primitive.material];

    let size = 1.0/3.0;//float(scene_size) / 4;
    let offset = float(scene_size) / 2;

    var triangle = fetch_3(positions, indices);
    triangle[0] = triangle[0].xzy;
    triangle[1] = triangle[1].xzy;
    triangle[2] = triangle[2].xzy;

    triangle[0] = triangle[0] * size + offset;
    triangle[1] = triangle[1] * size + offset;
    triangle[2] = triangle[2] * size + offset;

    if (outside(triangle[0], scene_size) && outside(triangle[1], scene_size) && outside(triangle[2], scene_size)) {
        return;
    }

    let uvs = fetch_3(primitive.uvs_ptr(gltf), indices);

    Writer writer;
    writer.output = (uint8_t4*)output;
    writer.scene_size = scene_size;
    writer.positions = triangle;
    writer.uvs = uvs;
    writer.is_alpha_clipped = (material.flags & 1) == 1;

    uint max_value = 4294967295;

    if (material.base_colour_image != max_value) {
        writer.base_colour_image = DescriptorHandle<Sampler2D>(material.base_colour_image);
    }
    if (material.emissive_image != max_value) {
        writer.emissive_image = DescriptorHandle<Sampler2D>(material.emissive_image);
    }
    if (material.metallic_roughness_image != max_value) {
        writer.metal_roughness_image = DescriptorHandle<Sampler2D>(material.metallic_roughness_image);
    }
    VoxelizeTriangleSurface(triangle, writer);
}
