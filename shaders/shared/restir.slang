struct Reservoir<T> {
    // Data (e.g. the index of most important light) of the sample
    T Y;
    // sample weight
    float W_y = 0.0;
    // sum of all weights for all samples processed
    float W_sum = 0.0;
    // number of samples processed for this reservoir
    float M = 0.0;

    __init(T initial) {
        Y = initial;
    }
    
    T get_data() {
        return Y;
    }

    [mutating]
    bool update(T value, float w, float c, float rand) {
        W_sum += w;
        M += c;

        if ( rand < w / W_sum  )
        {
            Y = value;
            return true;
        }

        return false;
    }

    [mutating]
    bool update_from_reservoir(Reservoir<T> reservoir, float p_hat, float rand) {
        return update(reservoir.Y, p_hat * reservoir.W_y * reservoir.M, reservoir.M, rand);
    }

    float average_weight() {
        return W_sum / M;
    }

    [mutating]
    void set_weight(float p_hat) {
        W_y = p_hat > 0.0 ? average_weight() / p_hat : 0.0;
    }
};

struct RestirState<T> {
    Reservoir<T> res;
    float pdf;

    __init(float pdf_value, T initial_sample_data) {
        pdf = pdf_value;
        res = Reservoir<T>(initial_sample_data);
    }

    [mutating]
    void update(T value, float p_hat, float rand) {
        let weight = p_hat / pdf;
        res.update(value, weight, 1, rand);
    }

    T get_data() {
        return res.get_data();
    }

    [mutating]
    void update_from_frame_sample(float p_hat) {
        // calculate the weight of this sample
        res.set_weight(p_hat);
    }

    float get_weight() {
        return res.W_y;
    }
}
