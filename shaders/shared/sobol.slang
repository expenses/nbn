// Adapted from https://gist.github.com/cessen/697b880ae07f32502db26edef1b1cec2

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

uint count_leading_zeros(uint v) {
    return 31 - firstbithigh(v);
}

/* [0, uint_max] -> [0.0, 1.0) */
float uint_to_float_excl(const uint n)
{
  /* NOTE: we divide by 4294967808 instead of 2^32 because the latter
   * leads to a [0.0, 1.0] mapping instead of [0.0, 1.0) due to floating
   * point rounding error. 4294967808 unfortunately leaves (precisely)
   * one unused ULP between the max number this outputs and 1.0, but
   * that's the best you can do with this construction. */
  return (float)n * (1.0f / 4294967808.0f);
}

/*
 * A shuffled, Owen-scrambled Sobol sampler, implemented with the
 * techniques from the paper "Practical Hash-based Owen Scrambling"
 * by Brent Burley, 2020, Journal of Computer Graphics Techniques.
 *
 * Note that unlike a standard high-dimensional Sobol sequence, this
 * Sobol sampler uses padding to achieve higher dimensions, as described
 * in Burley's paper.
 */

/*
 * A fast, high-quality 32-bit mixing function.
 *
 * From https://github.com/skeeto/hash-prospector
 */
uint hash_hp_uint(uint n) {
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

/*
 * The direction vectors for the first four dimensions of the Sobol
 * sequence, stored with reversed-order bits.
 *
 * This is used in the Sobol-Burley sampler implementation.  We don't
 * need more than four dimensions because we achieve higher dimensions
 * with padding.  They're stored with reversed bits because we need
 * them reversed for the fast hash-based Owen scrambling anyway, and
 * this avoids doing that at run time.
 */
const static uint sobol_burley_table[4][32] = {
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
float sobol_burley(uint rev_bit_index,
                                          const uint dimension,
                                          const uint scramble_seed)
{
  uint result = 0;

  if (dimension == 0) {
    /* Fast-path for dimension 0, which is just Van der corput.
     * This makes a notable difference in performance since we reuse
     * dimensions for padding, and dimension 0 is reused the most. */
    result = reversebits(rev_bit_index);
  }
  else {
    uint i = 0;
    while (rev_bit_index != 0) {
      const uint j = count_leading_zeros(rev_bit_index);
      result ^= sobol_burley_table[dimension][i + j];
      i += j + 1;

      /* We can't do `<<= j + 1` because that can overflow the shift
       * operator, which doesn't do what we need, at least on x86. */
      rev_bit_index <<= j;
      rev_bit_index <<= 1;
    }
  }

  /* Apply Owen scrambling. */
  result = reversebits(reversed_bit_owen(result, scramble_seed));

  return uint_to_float_excl(result);
}

/*
 * NOTE: the functions below intentionally produce samples that are
 * uncorrelated between functions.  For example, a 1D sample and 2D
 * sample produced with the same index, dimension, and seed are
 * uncorrelated with each other.  This allows more care-free usage
 * of the functions together, without having to worry about
 * e.g. 1D and 2D samples being accidentally correlated with each
 * other.
 */

/*
 * Computes a 1D Owen-scrambled and shuffled Sobol sample.
 *
 * `index` is the index of the sample in the sequence.
 *
 * `dimension` is which dimensions of the sample you want to fetch.  Note
 * that different 1D dimensions are uncorrelated.  For samples with > 1D
 * stratification, use the multi-dimensional sampling methods below.
 *
 * `seed`: different seeds produce statistically independent,
 * uncorrelated sequences.
 *
 * `shuffled_index_mask` limits the sample sequence length, improving
 * performance. It must be a string of binary 1 bits followed by a
 * string of binary 0 bits (e.g. 0xffff0000) for the sampler to operate
 * correctly. In general, `reverse_integer_bits(shuffled_index_mask)`
 * should be >= the maximum number of samples expected to be taken. A safe
 * default (but least performant) is 0xffffffff, for maximum sequence
 * length.
 */
float sobol_burley_sample_1D(uint index,
                                        const uint dimension,
                                        uint seed,
                                        const uint shuffled_index_mask)
{
  /* Include the dimension in the seed, so we get decorrelated
   * sequences for different dimensions via shuffling. */
  seed ^= hash_hp_uint(dimension);

  /* Shuffle and mask.  The masking is just for better
   * performance at low sample counts. */
  index = reversed_bit_owen(reversebits(index), seed ^ 0xbff95bfe);
  index &= shuffled_index_mask;

  return sobol_burley(index, 0, seed ^ 0x635c77bd);
}

/*
 * Computes a 2D Owen-scrambled and shuffled Sobol sample.
 *
 * `dimension_set` is which two dimensions of the sample you want to
 * fetch.  For example, 0 is the first two, 1 is the second two, etc.
 * The dimensions within a single set are stratified, but different sets
 * are uncorrelated.
 *
 * See sobol_burley_sample_1D for further usage details.
 */
float2 sobol_burley_sample_2D(uint index,
                                         const uint dimension_set,
                                         uint seed,
                                         const uint shuffled_index_mask)
{
  /* Include the dimension set in the seed, so we get decorrelated
   * sequences for different dimension sets via shuffling. */
  seed ^= hash_hp_uint(dimension_set);

  /* Shuffle and mask.  The masking is just for better
   * performance at low sample counts. */
  index = reversed_bit_owen(reversebits(index), seed ^ 0xf8ade99a);
  index &= shuffled_index_mask;

  return float2(sobol_burley(index, 0, seed ^ 0xe0aaaf76),
                     sobol_burley(index, 1, seed ^ 0x94964d4e));
}

/*
 * Computes a 3D Owen-scrambled and shuffled Sobol sample.
 *
 * `dimension_set` is which three dimensions of the sample you want to
 * fetch.  For example, 0 is the first three, 1 is the second three, etc.
 * The dimensions within a single set are stratified, but different sets
 * are uncorrelated.
 *
 * See sobol_burley_sample_1D for further usage details.
 */
float3 sobol_burley_sample_3D(uint index,
                                         const uint dimension_set,
                                         uint seed,
                                         const uint shuffled_index_mask)
{
  /* Include the dimension set in the seed, so we get decorrelated
   * sequences for different dimension sets via shuffling. */
  seed ^= hash_hp_uint(dimension_set);

  /* Shuffle and mask.  The masking is just for better
   * performance at low sample counts. */
  index = reversed_bit_owen(reversebits(index), seed ^ 0xcaa726ac);
  index &= shuffled_index_mask;

  return float3(sobol_burley(index, 0, seed ^ 0x9e78e391),
                     sobol_burley(index, 1, seed ^ 0x67c33241),
                     sobol_burley(index, 2, seed ^ 0x78c395c5));
}

/*
 * Computes a 4D Owen-scrambled and shuffled Sobol sample.
 *
 * `dimension_set` is which four dimensions of the sample you want to
 * fetch.  For example, 0 is the first four, 1 is the second four, etc.
 * The dimensions within a single set are stratified, but different sets
 * are uncorrelated.
 *
 * See sobol_burley_sample_1D for further usage details.
 */
float4 sobol_burley_sample_4D(uint index,
                                         const uint dimension_set,
                                         uint seed,
                                         const uint shuffled_index_mask)
{
  /* Include the dimension set in the seed, so we get decorrelated
   * sequences for different dimension sets via shuffling. */
  seed ^= hash_hp_uint(dimension_set);

  /* Shuffle and mask.  The masking is just for better
   * performance at low sample counts. */
  index = reversed_bit_owen(reversebits(index), seed ^ 0xc2c1a055);
  index &= shuffled_index_mask;

  return float4(sobol_burley(index, 0, seed ^ 0x39468210),
                     sobol_burley(index, 1, seed ^ 0xe9d8a845),
                     sobol_burley(index, 2, seed ^ 0x5f32b482),
                     sobol_burley(index, 3, seed ^ 0x1524cc56));
}
