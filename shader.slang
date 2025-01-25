struct BufferView {
    uint byte_offset;
}

struct Accessor {
    uint buffer_view;
    uint count;

    uint8_t* ptr(Gltf* gltf) {
        BufferView buffer_view = gltf.buffer_views[buffer_view];
        return gltf.buffer + buffer_view.byte_offset;
    }
};

struct Primitive {
    uint indices;
    uint positions;
    uint uvs;

    uint num_indices(Gltf* gltf) {
        return gltf.accessors[indices].count;
    }

    uint16_t* indices_ptr(Gltf* gltf) {
        return (uint16_t*)gltf.accessors[indices].ptr(gltf);
    }

    float3* positions_ptr(Gltf* gltf) {
        return (float3*)gltf.accessors[positions].ptr(gltf);
    }

    float2* uvs_ptr(Gltf* gltf) {
        return (float2*)gltf.accessors[uvs].ptr(gltf);
    }
}

struct Gltf {
    BufferView* buffer_views;
    Accessor* accessors;
    Primitive* primitives;
    uint8_t* buffer;
};

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

struct Writer: IVisitor {
    uint8_t* output;
    uint scene_size;
    float3[3] positions;
    float2[3] uvs;
    DescriptorHandle<Sampler2D> image;

    void visit(int3 pos) {
        let barycentrics = ProjectPointOnTriangle(pos, positions);
        let uv = interpolate(uvs, barycentrics);
        let value = image.SampleLevel(uv, 0).x;
        output[pos.x + pos.y * scene_size + pos.z * scene_size * scene_size] = uint8_t(value * 128.0);
    }
}

func fetch_3<T, I: IInteger>(T* values, I* indices) -> T[3] {
    return {
        values[indices[0].toUInt()],
        values[indices[1].toUInt()],
        values[indices[2].toUInt()]
    };
}

[numthreads(64,1,1)]
void main(
    // really a uint8_t* but I get a validation warning if `storagePushConstant8` is not enabled
    uint32_t* output,
    Gltf* gltf,
    uniform uint scene_size,
    uniform uint image,
    uint3 dispatch_thread_id : SV_DispatchThreadID,
) {
    let primitive = (*gltf).primitives[0];
    let num_triangles = primitive.num_indices(gltf) / 3;
    let indices = primitive.indices_ptr(gltf);
    let positions = primitive.positions_ptr(gltf);

    let thread = dispatch_thread_id.x;

    if (thread >= num_triangles) {
        return;
    }

    let size = 200;//float(scene_size) / 4;
    let offset = float(scene_size) / 2;

    var triangle = fetch_3(positions, indices + thread * 3);
    triangle[0] = triangle[0] * size + offset;
    triangle[1] = triangle[1] * size + offset;
    triangle[2] = triangle[2] * size + offset;

    let uvs = fetch_3(primitive.uvs_ptr(gltf), indices + thread * 3);

    let writer = Writer((uint8_t*)output, scene_size, triangle, uvs, DescriptorHandle<Sampler2D>(image));

    VoxelizeTriangleSurface(triangle, writer);
}
