
func fetch_3<T>(T *values, uint[3] indices) -> T[3] {
    return {values[indices[0]], values[indices[1]], values[indices[2]]};
}

static uint max_value = 4294967295;