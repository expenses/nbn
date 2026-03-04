
#define SOBOL_BURLEY_DIMENSIONS 4
#define SOBOL_BURLEY_BITS 32

/*
 * A fast, high-quality 32-bit mixing function.
 *
 * From https://github.com/skeeto/hash-prospector
 */
uint hp_mix32(uint n) {
    // The actual mixing function.
    n ^= n >> 16;
    n *= 0x21f0aaad;
    n ^= n >> 15;
    n *= 0xd35a2d97;
    n ^= n >> 15;

    // Xor by a random number so input zero doesn't map to output zero.
    // The particular number used here isn't special.
    return n ^ 0xe6fe3beb;
}

/*
 * Performs base-2 Owen scrambling on a reversed-bit integer.
 *
 * This is essentially the Laine-Karras permutation, but much higher
 * quality.  See https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
 */
uint reversed_bit_owen(uint n, uint seed) {
  n ^= n * 0x3d20adea;
  n += seed;
  n *= (seed >> 16) | 1;
  n ^= n * 0x05526c56;
  n ^= n * 0x53a22864;

  return n;
}

uint count_leading_zeros(uint v) {
    return 31 - firstbithigh(v);
}

/*
 * The direction vectors for the first four dimensions of the Sobol
 * sequence, stored with reversed-order bits.
 *
 * We don't need more than four dimensions because we achieve higher
 * dimensions with padding.  They're stored with reversed bits
 * because we need them reversed for the fast hash-based Owen
 * scrambling anyway, and this avoids doing that at run time.
 */
const static uint sobol_burley_table[SOBOL_BURLEY_DIMENSIONS][SOBOL_BURLEY_BITS] = {
  {
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
  },
  {
    0x00000001, 0x00000003, 0x00000005, 0x0000000f,
    0x00000011, 0x00000033, 0x00000055, 0x000000ff,
    0x00000101, 0x00000303, 0x00000505, 0x00000f0f,
    0x00001111, 0x00003333, 0x00005555, 0x0000ffff,
    0x00010001, 0x00030003, 0x00050005, 0x000f000f,
    0x00110011, 0x00330033, 0x00550055, 0x00ff00ff,
    0x01010101, 0x03030303, 0x05050505, 0x0f0f0f0f,
    0x11111111, 0x33333333, 0x55555555, 0xffffffff,
  },
  {
    0x00000001, 0x00000003, 0x00000006, 0x00000009,
    0x00000017, 0x0000003a, 0x00000071, 0x000000a3,
    0x00000116, 0x00000339, 0x00000677, 0x000009aa,
    0x00001601, 0x00003903, 0x00007706, 0x0000aa09,
    0x00010117, 0x0003033a, 0x00060671, 0x000909a3,
    0x00171616, 0x003a3939, 0x00717777, 0x00a3aaaa,
    0x01170001, 0x033a0003, 0x06710006, 0x09a30009,
    0x16160017, 0x3939003a, 0x77770071, 0xaaaa00a3,
  },
  {
    0x00000001, 0x00000003, 0x00000004, 0x0000000a,
    0x0000001f, 0x0000002e, 0x00000045, 0x000000c9,
    0x0000011b, 0x000002a4, 0x0000079a, 0x00000b67,
    0x0000101e, 0x0000302d, 0x00004041, 0x0000a0c3,
    0x0001f104, 0x0002e28a, 0x000457df, 0x000c9bae,
    0x0011a105, 0x002a7289, 0x0079e7db, 0x00b6dba4,
    0x0100011a, 0x030002a7, 0x0400079e, 0x0a000b6d,
    0x1f001001, 0x2e003003, 0x45004004, 0xc900a00a,
  },
};

/*
 * Computes a single dimension of a sample from an Owen-scrambled
 * Sobol sequence.  This is used in the main sampling functions,
 * sobol_burley_sample_#D(), below.
 *
 * - rev_bit_index: the sample index, with reversed order bits.
 * - dimension:     the sample dimension.
 * - scramble_seed: the Owen scrambling seed.
 *
 * Note that the seed must be well randomized before being
 * passed to this function.
 */
float sobol_burley(uint rev_bit_index, uint dimension, uint scramble_seed)
{
  uint result = 0;

  if (dimension == 0) {
    // Fast-path for dimension 0, which is just Van der corput.
    // This makes a notable difference in performance since we reuse
    // dimensions for padding, and dimension 0 is reused the most.
    result = reversebits(rev_bit_index);
  } else {
    uint i = 0;
    while (rev_bit_index != 0) {
      uint j = count_leading_zeros(rev_bit_index);
      result ^= sobol_burley_table[dimension][i + j];
      i += j + 1;

      // We can't do "<<= j + 1" because that can overflow the shift
      // operator, which doesn't do what we need on at least x86.
      rev_bit_index <<= j;
      rev_bit_index <<= 1;
    }
  }

  // Apply Owen scrambling.
  result = reversebits(reversed_bit_owen(result, scramble_seed));

  return (float)result * (1.0f / (float)0xFFFFFFFF);
}


/*
 * Computes a 1D Owen-scrambled and shuffled Sobol sample.
 */
float sobol_burley_sample_1D(uint index, uint dimension, uint seed)
{
  // Include the dimension in the seed, so we get decorrelated
  // sequences for different dimensions via shuffling.
  seed ^= hp_mix32(dimension);

  // Shuffle.
  index = reversed_bit_owen(reversebits(index), seed ^ 0xbff95bfe);

  return sobol_burley(index, 0, seed ^ 0x635c77bd);
}

/*
 * Computes a 2D Owen-scrambled and shuffled Sobol sample.
 */
float2 sobol_burley_sample_2D(uint index,
                                       uint dimension_set,
                                       uint seed)
{
  // Include the dimension set in the seed, so we get decorrelated
  // sequences for different dimension sets via shuffling.
  seed ^= hp_mix32(dimension_set);

  // Shuffle.
  index = reversed_bit_owen(reversebits(index), seed ^ 0xf8ade99a);

  float2 out;
  out.x = sobol_burley(index, 0, seed ^ 0xe0aaaf76);
  out.y = sobol_burley(index, 1, seed ^ 0x94964d4e);
  return out;
}

/*
 * Computes a 3D Owen-scrambled and shuffled Sobol sample.
 */
float3 sobol_burley_sample_3D(uint index,
                                       uint dimension_set,
                                       uint seed)
{
  // Include the dimension set in the seed, so we get decorrelated
  // sequences for different dimension sets via shuffling.
  seed ^= hp_mix32(dimension_set);

  // Shuffle.
  index = reversed_bit_owen(reversebits(index), seed ^ 0xcaa726ac);

  float3 out;
  out.x = sobol_burley(index, 0, seed ^ 0x9e78e391);
  out.y = sobol_burley(index, 1, seed ^ 0x67c33241);
  out.z = sobol_burley(index, 2, seed ^ 0x78c395c5);
  return out;
}

/*
 * Computes a 4D Owen-scrambled and shuffled Sobol sample.
 */
float4 sobol_burley_sample_4D(uint index,
                                       uint dimension_set,
                                       uint seed)
{
  // Include the dimension set in the seed, so we get decorrelated
  // sequences for different dimension sets via shuffling.
  seed ^= hp_mix32(dimension_set);

  // Shuffle.
  index = reversed_bit_owen(reversebits(index), seed ^ 0xc2c1a055);

  float4 out;
  out.x = sobol_burley(index, 0, seed ^ 0x39468210);
  out.y = sobol_burley(index, 1, seed ^ 0xe9d8a845);
  out.z = sobol_burley(index, 2, seed ^ 0x5f32b482);
  out.w = sobol_burley(index, 3, seed ^ 0x1524cc56);
  return out;
}
