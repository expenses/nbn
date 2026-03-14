import envmap;

struct Alias {
    float threshold;
    float inv_pdf;
    uint fallback;
};

struct AliasAndIndex {
    Alias alias;
    uint index;
}

AliasAndIndex sample_alias_table(Alias* aliases, uint num_aliases, float2 rand) {
    var index = uint(rand.x * num_aliases);
    var alias = aliases[index];

    if (rand.y > alias.threshold) {
        index = alias.fallback;
        alias = aliases[index];
    }

    AliasAndIndex output;
    output.index = index;
    output.alias = alias;
    return output;
}

float mis_power_heuristic(float pdf_a, float pdf_b) {
    let pdf_a2 = pdf_a * pdf_a;
    let pdf_b2 = pdf_b * pdf_b;
    return pdf_a2 / (pdf_a2 + pdf_b2);
}

struct LightSample {
    float3 radiance;
    float3 direction;
    float pdf;
}

LightSample sample_envmap_by_rand(Alias* aliases, Sampler2D envmap, uint2 size, float2 rand) {
    let alias = sample_alias_table(aliases, size.x * size.y, rand);
    let tex_coord = uint2(alias.index % size.x, alias.index / size.x);
    let uv = (tex_coord+0.5) / size;
    
    LightSample sample;
    sample.radiance = envmap[tex_coord].rgb;
    sample.direction = envmap_uv_to_direction(uv);
    sample.pdf = 1.0/alias.alias.inv_pdf;
    return sample;
}

LightSample sample_envmap_by_direction(Alias* aliases, Sampler2D envmap, uint2 size, float3 direction) {
    let uv = envmap_direction_to_uv(direction);
    let location = uint2(uv * size);
    let index = location.y * size.x + location.x;

    LightSample sample;
    sample.radiance = envmap[location].rgb;
    sample.pdf = 1.0/aliases[index].inv_pdf;
    sample.direction = direction;
    return sample;
}