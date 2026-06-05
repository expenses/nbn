pub fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut inv_b = 1.0 / base as f32;
    let mut i = index;
    while i > 0 {
        f += (i % base) as f32 * inv_b;
        i /= base;
        inv_b /= base as f32;
    }
    f - 1.0
}

pub fn jitter(frame_index: u32) -> [f32; 2] {
    [
        halton(frame_index + 1, 2) - 0.5,
        halton(frame_index + 1, 3) - 0.5,
    ]
}
