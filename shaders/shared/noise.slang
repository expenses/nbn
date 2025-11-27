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

    public uint sample_uint(int2 pixel, int sampleIndex, int sampleDimension) {
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

        return value;
    }

    public float sample(int2 pixel, int sampleIndex, int sampleDimension) {
        let value = sample_uint(pixel, sampleIndex, sampleDimension);
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
    uint index = 0;

    public __init(BlueNoiseData data_, uint frame_index_, uint2 thread_) {
        data = data_;
        frame_index = frame_index_;
        thread = thread_;
    }

    public __init(BlueNoiseData data_, uint state) {
        data = data_;
        thread = uint2(state & 127, (state >> 8) & 127);
        frame_index = (state >> 16) & 255;
        index = (state >> 24) & 255;
    }

    public uint state() {
        return (thread.x & 127) | ((thread.y & 127) << 8) | ((frame_index & 255) << 16) | ((index & 255) << 24);
    }

    [mutating]
    public void advance(uint times) {
        index += times;
    }

    [mutating]
    public float rand() {
        let value = data.sample(thread + uint2(R2(index) * 128), frame_index, 0);
        index += R2;
        return value;
    }
};

// https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/Falcor/Utils/Sampling/Pseudorandom/LCG.slang#L47-L56
public struct LCG: Rng {
    public uint state;
    
    public __init(uint state_) {
        state = state_;
    }
    
    public __init(uint2 thread, uint frame_index) {
        state = blockCipherTEA(interleave_32bit(thread), frame_index).x;
    }
    
    [mutating]
    public float rand() {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return bits_to_float(state);
    }
}

// https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/Falcor/Utils/Sampling/SampleGeneratorInterface.slang#L41-L56
float bits_to_float(uint bits) {
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    uint bits = bits;
    return ((bits >> 8) * 0x1p-24);
}

// https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/Falcor/Utils/Math/HashUtils.slang#L50-L74
uint2 blockCipherTEA(uint v0, uint v1, uint iterations = 16)
{
    uint sum = 0;
    const uint delta = 0x9e3779b9;
    const uint k[4] = { 0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e }; // 128-bit key.
    for (uint i = 0; i < iterations; i++)
    {
        sum += delta;
        v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }
    return uint2(v0, v1);
}

// https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/Falcor/Utils/Math/BitTricks.slang#L47C1-L62C2
uint interleave_32bit(uint2 v)
{
    uint x = v.x & 0x0000ffff;       // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x | (x << 8)) & 0x00FF00FF; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x | (x << 4)) & 0x0F0F0F0F; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x | (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x | (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0


    uint y = v.y & 0x0000ffff;
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;


    return x | (y << 1);
}

// https://github.com/electronicarts/fastnoise/blob/main/FastNoiseDesign.md#references
// R2 is from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// R2 Low discrepancy sequence
float2 R2(int index)
{
	static const float g  = 1.32471795724474602596f;
	static const float a1 = 1 / g;
	static const float a2 = 1 / (g * g);
	return float2(frac(float(index) * a1), frac(float(index) * a2));
}