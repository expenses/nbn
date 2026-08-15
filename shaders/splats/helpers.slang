import shared.constants;

#define RECIPROCAL_2e32 2.3283064365387e-10f

uint2 pcg2d_uint(uint2 seed) {
   	seed.x = seed.x * 1664525u + 1013904223u;
   	seed.y = seed.y * 1664525u + 1013904223u;

   	seed.x += seed.y * 1664525u;
   	seed.y += seed.x * 1664525u;

   	seed.x ^= seed.x >> 16u;
   	seed.y ^= seed.y >> 16u;

   	seed.x += seed.y * 1664525u;
   	seed.y += seed.x * 1664525u;

   	seed.x ^= seed.x >> 16u;
   	seed.y ^= seed.y >> 16u;
   	return seed;
}


float2 pcg2d(inout uint2 seed) {
   	seed = pcg2d_uint(seed);
   	return { seed.x * RECIPROCAL_2e32, seed.y * RECIPROCAL_2e32 };
}

uint32_t hash32(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 15;
	x *= 0x846ca68b;
	x ^= x >> 16;
	return x;
}

uint2 makeSeed(uint32_t idx, uint32_t frame) {

	const uint32_t C1 = 0x9E3779B9u;
	const uint32_t C2 = 0xBB67AE85u;
	const uint32_t C3 = 0x3C6EF372u;
	const uint32_t C4 = 0xA54FF53Au;

	uint32_t h1 = hash32(idx ^ (frame * C1));
	uint32_t h2 = hash32((idx * C2) ^ frame);

	h1 ^= (h2 * C3);
	h2 ^= (h1 * C4);

	return uint2(h1, h2);
}


// Dilogarithm Li2(x) = -integral_0^x ln(1-s)/s ds.
// Polynomial fit to Li2(x)-(1-x)ln(1-x), then (1-x)ln(1-x) added back.
float dilog(float x) {
    float y = -6.09201442e-01f;
	y = fma(y, x, 1.79126616e+00f);
	y = fma(y, x, -2.14953223e+00f);
	y = fma(y, x, 1.26304372e+00f);
	y = fma(y, x, -4.59069895e-01f);
	y = fma(y, x, -1.87417414e-01f);
	y = fma(y, x, 1.99603130e+00f);
	y = fma(y, x, 5.99669467e-05f);

	float s = 1.0f - x;
	if (s > 0.0f)
		y += s * log(max(s, 1e-37f));

	return y;
}

// Li2(1) = pi^2 / 6
static const float LI2_MAX = 1.6449340668482264;

// Inverse dilogarithm: minimax degree-10 polynomial in t = x / Li2(1).
float inv_dilog(float x) {
    float t = min(x / LI2_MAX, 1.0f);

	float y = -1.27463503e+01f;
	y = fma(y, t, 5.88993459e+01f);
	y = fma(y, t, -1.16025780e+02f);
	y = fma(y, t, 1.26945827e+02f);
	y = fma(y, t, -8.43108826e+01f);
	y = fma(y, t, 3.48799862e+01f);
	y = fma(y, t, -8.89606235e+00f);
	y = fma(y, t, 1.38640936e+00f);
	y = fma(y, t, -7.80640876e-01f);
	y = fma(y, t, 1.64841888e+00f);
	y = fma(y, t, -2.82836687e-05f);

	return y;
}

float2 boxMuller(float u1, float u2) {
	float R = sqrt(max(-2.0f * log(clamp(u1, 1.e-37f, 1.0f)), 0.0f));
	float theta = 2.0f * PI * u2;
	return float2(R * cos(theta), R * sin(theta));
}

// Opacity-corrected Box-Muller: returns a 2D standard normal whose radius follows
// the opacity-corrected Gaussian density. Multiply by the Cholesky factor of cov2D.
float2 correctedBoxMuller(float u1, float u2, float alpha) {
	float a = 1.0f / max(alpha, 1.e-37f) * inv_dilog((1.0f - u1) * dilog(alpha));
	a = clamp(a, 1.e-37f, 1.0f);
	float R = sqrt(max(-2.0f * log(a), 0.0f));

	float theta = 2.0f * PI * u2;
	return float2(R * cos(theta), R * sin(theta));
}

// gaussian-point-splatting/src/core/random/poisson.cuh
uint32_t poisson(inout uint2 state, float lambda) {
    float2 u = pcg2d(state);

    float w = boxMuller(u.x, u.y).x;
    float w2 = w * w;
    float w3 = w2 * w;
    float w4 = w2 * w2;

    float s = sqrt(lambda);   // sqrt(lambda)
    float inv_s = 1.0f / s;             // lambda^{-1/2}
    float inv_l = 1.0f / lambda;        // lambda^{-1}

    // QN1: lambda + sqrt(lambda)*w + (w^2 - 1) / 6
    float kf = lambda + s * w + (w2 - 1.0f) / 6.0f;

    // QN2 term: lambda^{-1/2} * ( -1/36 * w - 1/72 * w^3 )
    kf += inv_s * (-(1.0f / 36.0f) * w - (1.0f / 72.0f) * w3);

    // QN3 term: lambda^{-1} * ( -8/405 + 7/810 * w^2 + 1/270 * w^4 )
    kf += inv_l * (-(8.0f / 405.0f) + (7.0f / 810.0f) * w2 + (1.0f / 270.0f) * w4);

    int ki = int(round(kf));
    uint32_t k = (uint32_t)max(ki, 0);

    return k;
}
