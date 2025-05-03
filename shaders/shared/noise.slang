// x and y are in pixels.
public

float interleaved_gradient_noise(int x, int y) {
    return fract(
        52.9829189f * fract(0.06711056f * float(x) + 0.00583715f * float(y))
    );
}

public

struct BlueNoiseData {
    int* sobol;
    int* scrambling_tile;
    int* ranking_tile;

    float
    sample(int2 pixel, int sampleIndex, int sampleDimension) {
        // wrap arguments
        pixel.x = pixel.x & 127;
        pixel.y = pixel.y & 127;
        sampleIndex = sampleIndex & 255;
        sampleDimension = sampleDimension & 255;

        // xor index based on optimized ranking
        // jb: 1spp blue noise has all 0 in g_blueNoiseRankingTile so we can skip the load
        int rankedSampleIndex = sampleIndex
            ^ ranking_tile
                [sampleDimension + (pixel.x + pixel.y * 128) * 8];

        // fetch value in sequence
        int value = sobol[sampleDimension + rankedSampleIndex * 256];

        // If the dimension is optimized, xor sequence value based on optimized scrambling
        value = value
            ^ scrambling_tile
                [(sampleDimension % 8) + (pixel.x + pixel.y * 128) * 8];

        // convert to float and return
        float v = (0.5f + value) / 256.0f;
        return v;
    }

    public float2 sample(int2 pixel, int sampleIndex) {
        return float2(
            sample(pixel, sampleIndex, 0),
             sample(pixel, sampleIndex, 1),
        );
    }
};
