uint64_t wave64_pack_bools(bool boolean) {
    let ballot = WaveActiveBallot(boolean);
    return uint64_t(ballot.x) | (uint64_t(ballot.y) << 32);
}

bool wave64_unpack_bools(uint64_t packed, uint2 local_thread_index) {
    let index = local_thread_index.x + local_thread_index.y * 8;
    return bool((packed >> index) & 1);
}
