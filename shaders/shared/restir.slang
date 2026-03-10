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
    void set_weight_from_p(float p) {
        W_y = p > 0.0 ? average_weight() / p : 0.0;
    }

    [mutating]
    void clamp(uint max_samples) {
        M = min(M, max_samples);
    }
    
    [mutating]
    float3 set_from_shadowed_contribution<C: Contribution<T>>(C contributor, bool is_set) {
        if (is_set) {
            let value = contributor.shadowed_contribution(data);
            set_weight_from_p(value.p);
            return value.contribution * weight;
        } else {
            W_y = 0.0;
            return float3(0);
        }
    }
};

interface Contribution<T> {
    UnshadowedValue unshadowed_contribution(T data);
    ShadowedValue shadowed_contribution(T data);
}

struct UnshadowedValue {
    float p_hat = 0.0;
    // 1.0 / pdf
    float inv_pdf = 0.0;

    float weight() {
        return p_hat * inv_pdf;
    }
}

struct ShadowedValue {
    float3 contribution = float3(0.0);
    float p = 0.0;
}

struct ContributionInfo {
    UnshadowedValue unshadowed;
    float3 direction;
    float distance;
    float3 contribution;
}

struct FullShadowedValue {
    ShadowedValue shadowed;
    float inv_pdf;
    
    float3 adjusted() {
        return shadowed.contribution * inv_pdf;
    }
}
/*
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
*/
