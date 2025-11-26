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

    T get_data() {
        return Y;
    }

    [mutating]
    bool update<R: Rng>(T value, float w, uint c, R* rng) {
        W_sum += w;
        M += c;

        if (rng.rand() < w / W_sum) {
            Y = value;
            return true;
        }

        return false;
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
    
    float get_weight() {
        return W_y;
    }
    
    [mutating]
    void clamp(uint max_samples) {
        M = min(M, max_samples);
    }
};
