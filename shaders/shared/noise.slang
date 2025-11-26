import random;

// x and y are in pixels.
public float interleaved_gradient_noise(int x, int y) {
    return fract(
        52.9829189f * fract(0.06711056f * float(x) + 0.00583715f * float(y))
    );
}

public struct BlueNoiseData {
    public int* sobol;
    public int* scrambling_tile;
    public int* ranking_tile;

    public float sample(int2 pixel, int sampleIndex, int sampleDimension) {
        // wrap arguments
        pixel.x = pixel.x & 127;
        pixel.y = pixel.y & 127;
        sampleIndex = sampleIndex & 255;
        sampleDimension = sampleDimension & 255;

        // xor index based on optimized ranking
        // jb: 1spp blue noise has all 0 in g_blueNoiseRankingTile so we can skip the load
        int rankedSampleIndex = sampleIndex
            ^ ranking_tile[sampleDimension + (pixel.x + pixel.y * 128) * 8];

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

public struct ThreadBlueNoise: Rng {
    public BlueNoiseData data;
    uint2 thread;
    uint frame_index;
    uint dimension = 0;

    public __init(BlueNoiseData data_, uint frame_index_, uint2 thread_) {
        data = data_;
        frame_index = frame_index_;
        thread = thread_;
    }

    public __init(BlueNoiseData data_, uint state) {
        data = data_;
        thread = uint2(state & 127, (state >> 8) & 127);
        frame_index = (state >> 16) & 255;
        dimension = (state >> 24) & 255;
    }

    public uint state() {
        return (thread.x & 127) | ((thread.y & 127) << 8) | ((frame_index & 255) << 16) | ((dimension & 255) << 24);
    }

    [mutating]
    public void advance(uint times) {
        dimension += times;
    }

    [mutating]
    public float rand() {
        let value = data.sample(thread, frame_index, dimension);
        dimension++;
        return value;
    }
};
