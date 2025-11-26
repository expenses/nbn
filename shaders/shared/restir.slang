import random;
import sampling;
import util;
import noise;
import constants;

struct Reservoir<T> {
    // Data of the selected sample (e.g. the index of most important light)
    T y;
    // Weight of the selected sample
    float W_y = 0.0;
    // Sum of all weights for all samples processed
    float W_sum = 0.0;
    // Number of samples processed for this reservoir
    uint M = 0;

    __init(T initial) {
        y = initial;
    }

    property T data {
        get {
            return y;
        }
    }

    property float weight {
        get {
            return W_y;
        }
    }

    [mutating]
    bool update(T value, float w, uint c, float rand) {
        W_sum += w;
        M += c;

        if (rand < w / W_sum) {
            y = value;
            return true;
        }

        return false;
    }

    [mutating]
    bool update<R: Rng>(T value, float w, uint c, R* rng) {
        return update(value, w, c, rng.rand());
    }

    [mutating]
    bool update_from_reservoir(Reservoir<T> reservoir, float p_hat, float rand) {
        return update(reservoir.y, p_hat * reservoir.W_y * reservoir.M, reservoir.M, rand);
    }

    [mutating]
    bool update_from_reservoir<R: Rng>(Reservoir<T> reservoir, float p_hat, R* rng) {
        return update_from_reservoir(reservoir, p_hat, rng.rand());
    }

    float average_weight() {
        return W_sum / M;
    }

    [mutating]
    void set_weight(float p_hat) {
        W_y = p_hat > 0.0 ? average_weight() / p_hat : 0.0;
    }

    [mutating]
    void clamp(uint max_samples) {
        M = min(M, max_samples);
    }
};

interface Contribution<T> {
    UnshadowedContribution unshadowed_contribution(T data);
    float3 shadowed_contribution(T data);
}

struct UnshadowedContribution {
    float3 contribution;
    // 1.0 / pdf
    float inv_pdf;
}

struct SamplePoint {
    float3 position;
    float3 normal;
    
    [mutating]
    void fix_normal(float3 camera_pos) {
        if (dot(camera_pos - position, normal) < 0) {
            normal = -normal;
        }
    }
}

struct Environment: Contribution<float2> {
    SamplePoint point;
    RaytracingAccelerationStructure acceleration_structure;
    float3 colour;
    
    __init(SamplePoint point_, RaytracingAccelerationStructure acceleration_structure_, float3 colour_) {
        point = point_;
        acceleration_structure = acceleration_structure_;
        colour = colour_;
    }

    UnshadowedContribution unshadowed_contribution(float2 _rand2) {
        UnshadowedContribution contribution;
        contribution.contribution = colour;
        contribution.inv_pdf = 1;
        return contribution;
    }

    float3 shadowed_contribution(float2 rand) {
        let hemisphere_dir = sample_cosine_weighted_hemisphere(rand);
        let rotated_dir = mul(build_tbn(point.normal), hemisphere_dir);

        RayDesc desc;
        desc.Origin = point.position;
        desc.Direction = rotated_dir;
        desc.TMin = 2e-4f;
        desc.TMax = 100000.0;
        var query = RayQuery<
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>();
        query.TraceRayInline(
            acceleration_structure,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            desc
        );
        query.Proceed();
        let visibility = float(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT);

        return visibility * colour;
    }
}


struct Combined<A: Contribution<float2>, B: Contribution<float2>>: Contribution<float2> {
    A a;
    B b;
    float factor;
    
    __init(A a_, B b_, float factor_) {
        a = a_;
        b = b_;
        factor = factor_;
    }
    
    UnshadowedContribution unshadowed_contribution(float2 rand2) {
        UnshadowedContribution contribution;
        if (rand2.x < factor) {
            rand2.x /= factor;
            contribution = a.unshadowed_contribution(rand2);
            contribution.inv_pdf /= factor;
        } else {
            rand2.x = inv_lerp(rand2.x, factor, 1.0);
            contribution = b.unshadowed_contribution(rand2);
            contribution.inv_pdf /= (1.0-factor);
        }
        return contribution;
    }

    float3 shadowed_contribution(float2 rand2) {
        if (rand2.x < factor) {
            rand2.x /= factor;
            return a.shadowed_contribution(rand2);
        } else {
            rand2.x = inv_lerp(rand2.x, factor, 1.0);
            return b.shadowed_contribution(rand2);
        }
    }
}

struct BlueNoise<T: Contribution<float2>>: Contribution<uint> {
    T inner;
    BlueNoiseData blue_noise_data;
    
    __init(T inner_, BlueNoiseData blue_noise_data_) {
        inner = inner_;
        blue_noise_data = blue_noise_data_;
    }
    
    UnshadowedContribution unshadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(blue_noise_data, blue_noise_state);
        return inner.unshadowed_contribution(blue_noise.rand2());
    }
    
    float3 shadowed_contribution(uint blue_noise_state) {
        var blue_noise = ThreadBlueNoise(blue_noise_data, blue_noise_state);
        return inner.shadowed_contribution(blue_noise.rand2());
    }
    
    float3 shadowed_contribution_from_reservoir(Reservoir<uint> reservoir) {
        if (reservoir.data != MAX_UINT_VALUE) {
            return shadowed_contribution(reservoir.data);  
        } else {
            return float3(0.0);
        }
    }
    
    float3 unshadowed_contribution_from_reservoir(Reservoir<uint> reservoir) {
        if (reservoir.data != MAX_UINT_VALUE) {
            return unshadowed_contribution(reservoir.data).contribution;  
        } else {
            return float3(0.0);
        }
    }
}

void stream_into_reservoir<T: Contribution<float2>, R: Rng>(BlueNoise<T> state, Reservoir<uint>* reservoir, R* rng, uint2 thread, uint frame_index, uint num_samples) {
    for (uint i = 0; i < num_samples; i++) {
        let rng_state = pack_blue_noise_state(thread, frame_index * num_samples + i);
        // Get the unshadowed contribution for a random light
        let contribution = state.unshadowed_contribution(rng_state);
        let p_hat = length(contribution.contribution);
        let weight = p_hat * contribution.inv_pdf;
        reservoir.update(rng_state, weight, 1, rng.rand());
    }
}

void temporal_accumulation<T: Contribution<float2>, R: Rng>(
    Reservoir<uint>* frame_reservoir,
    float p_hat,
    Reservoir<uint> previous_reservoir,
    BlueNoise<T> state,
    R* rng, 
    uint num_samples
) {
    var previous_reservoir = previous_reservoir;
    var temporal_reservoir = Reservoir<uint>(MAX_UINT_VALUE);

    // Update from current reservoir
    temporal_reservoir.update_from_reservoir(*frame_reservoir, p_hat, rng);

    // Clamp previous reservoir to the last 20 frames.
    previous_reservoir.clamp(20 * num_samples);

    // Update from prevous reservoir
    let p_hat_prev = length(state.unshadowed_contribution_from_reservoir(previous_reservoir));
    temporal_reservoir.update_from_reservoir(previous_reservoir, p_hat_prev, rng);

    *frame_reservoir = temporal_reservoir;
}