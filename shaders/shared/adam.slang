// Default Adam values.
static const float ADAM_BETA_1 = 0.9;
static const float ADAM_BETA_2 = 0.999;
static const float ADAM_EPSILON = 1e-8;

float3 adam_step(
    float3 grad,
    inout float3 mean,
    inout float3 variance,
    float learning_rate,
    float adam_m_factor,
    float adam_v_factor,
) {
    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates
    // the true values at the beginning. Adam divides by a compensation factor
    let mHat = mean * adam_m_factor;
    let vHat = variance * adam_v_factor;
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    let step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);
    // flip sign
    return -step;
}


float adam_step(
    float grad,
    inout float mean,
    inout float variance,
    float learning_rate,
    float adam_m_factor,
    float adam_v_factor,
) {
    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates
    // the true values at the beginning. Adam divides by a compensation factor
    let mHat = mean * adam_m_factor;
    let vHat = variance * adam_v_factor;
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    let step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);
    // flip sign
    return -step;
}
