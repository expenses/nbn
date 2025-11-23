struct Reservoir {
    uint Y = 0; // index of most important light
    //float W_y = 0.0; // light weight
    float W_sum = 0.0; // sum of all weights for all lights processed
    float M = 0.0; // number of lights processed for this reservoir

    //bool is_valid = false;

    [mutating]
    bool update(uint X, float w, float c, float rand) {
        W_sum += w;
        M += c;

        if ( rand < w / W_sum  )
        {
            Y = X;
            //is_valid = true;
            return true;
        }

        return false;
    }
};
