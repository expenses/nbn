import random;

struct Reservoir<T> {
    // Data of the selected sample (e.g. the index of most important light)
    T Y;
    // Weight of the selected sample
    float W_y = 0.0;
    // Sum of all weights for all samples processed
    float W_sum = 0.0;
    // Number of samples processed for this reservoir
    uint M = 0;

    __init(T initial) {
        Y = initial;
    }

    property T data {
        get {
            return Y;
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
            Y = value;
            return true;
        }

        return false;
    }

    [mutating]
    bool update<R: Rng>(T value, float w, uint c, R* rng) {
        return update(value, w, c, rng.rand());
    }

    [mutating]
    bool update_from_reservoir<R: Rng>(Reservoir<T> reservoir, float p_hat, R* rng) {
        return update(reservoir.Y, p_hat * reservoir.W_y * reservoir.M, reservoir.M, rng);
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
    float3 get_unshadowed_contribution(T data);
    float3 get_shadowed_contribution(T data);
}
