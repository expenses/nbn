use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};
use std::io::BufRead;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct PlySplat {
    xyz: [f32; 3],
    f_dc: [f32; 3],
    opacity: f32,
    scale: [f32; 3],
    rot: [f32; 4],
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct PlySplatNormals {
    xyz: [f32; 3],
    n: [f32; 3],
    f_dc: [f32; 3],
    opacity: f32,
    scale: [f32; 3],
    rot: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Splat {
    center: [f32; 3],
    color_opacity: u32,
    scale: u32,
    rot: u32,
}

impl From<PlySplat> for Splat {
    #[inline]
    fn from(s: PlySplat) -> Self {
        let opacity = 1.0 / (1.0 + (-s.opacity).exp());

        Self {
            // (COLMAP y-down -> y-up)
            center: [s.xyz[0], -s.xyz[1], s.xyz[2]],
            color_opacity: quantize_8b_sh(s.f_dc[0])
                | (quantize_8b_sh(s.f_dc[1]) << 8)
                | (quantize_8b_sh(s.f_dc[2]) << 16)
                | (quantize_8b(opacity) << 24),
            scale: quantize_10b_log(s.scale[0])
                | (quantize_10b_log(s.scale[1]) << 10)
                | (quantize_10b_log(s.scale[2]) << 20),
            // colmap flip again
            rot: quantize_quat([-s.rot[1], s.rot[2], -s.rot[3], s.rot[0]]),
        }
    }
}

impl From<PlySplatNormals> for Splat {
    #[inline]
    fn from(s: PlySplatNormals) -> Self {
        Self::from(PlySplat {
            xyz: s.xyz,
            f_dc: s.f_dc,
            opacity: s.opacity,
            scale: s.scale,
            rot: s.rot,
        })
    }
}

const SH_C0: f32 = 0.28209479177387814;

pub fn cast_slice<I: Copy, O: Copy>(slice: &[I]) -> &[O] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}

pub fn cast_slice_mut<I: Copy, O: Copy>(slice: &mut [I]) -> &mut [O] {
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr() as *mut O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}

const LOG_MIN: f32 = -8.0;
const LOG_MAX: f32 = 4.0;

fn quantize_8b(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u32
}

fn quantize_8b_sh(v: f32) -> u32 {
    quantize_8b(srgb_lin2encoded((v * SH_C0) + 0.5))
}

fn quantize_10b_log(log_s: f32) -> u32 {
    let t = (log_s - LOG_MIN) / (LOG_MAX - LOG_MIN);
    quantize_10b_u(t)
}

fn quantize_10b_u(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) * 1023.0).round() as u32
}

fn quantize_10b(v: f32) -> u32 {
    // for 10 bits we use
    // 2**(10 - 1) - 1 = 511
    // scale value between -511 and 511
    let q = (v.clamp(-1.0, 1.0) * 511.0).round() as i32;
    // move to between 0 and 1022
    (q + 511) as u32
}

// pack as 3x10b + 2b
fn quantize_quat(q: [f32; 4]) -> u32 {
    // Find index of the largest component.
    let mut qc = 0;
    for i in 1..4 {
        if q[i].abs() > q[qc].abs() {
            qc = i;
        }
    }

    // To ensure that this largest component is positive.
    let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };

    // As none of the other three components can exceed 1/sqrt(2),
    // we can freely scale them by sqrt(2) for increased precision
    let scaler = std::f32::consts::SQRT_2;

    // note: we encode a cyclical swizzle to be able to recover the order via rotation
    let a = quantize_10b(q[(qc + 1) % 4] * scaler * sign);
    let b = quantize_10b(q[(qc + 2) % 4] * scaler * sign);
    let c = quantize_10b(q[(qc + 3) % 4] * scaler * sign);

    a | (b << 10) | (c << 20) | ((qc as u32) << 30)
}

fn srgb_lin2encoded(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn read_fill<R: std::io::Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total = 0;

    while total < buf.len() {
        match reader.read(&mut buf[total..])? {
            0 => break,
            n => total += n,
        }
    }

    Ok(total)
}

fn load_splats<T: Default + Copy + Sync + Into<Splat>, R: std::io::Read>(
    reader: &mut R,
) -> Vec<Splat> {
    let mut splats = Vec::new();

    let mut arr = vec![T::default(); 1_000_000];

    loop {
        let bytes_read = read_fill(reader, cast_slice_mut(&mut arr)).unwrap();

        let splats_read = bytes_read / std::mem::size_of::<T>();

        splats.par_extend(arr[..splats_read].par_iter().map(|&splat| splat.into()));

        dbg!(splats.len());

        if splats_read != arr.len() {
            break;
        }
    }

    splats
}

fn expand_bits(mut x: u64) -> u64 {
    x = (x | x << 32) & 0b1111111111111111000000000000000000000000000000001111111111111111;
    x = (x | x << 16) & 0b0000000011111111000000000000000011111111000000000000000011111111;
    x = (x | x << 08) & 0b1111000000001111000000001111000000001111000000001111000000001111;
    x = (x | x << 04) & 0b0011000011000011000011000011000011000011000011000011000011000011;
    x = (x | x << 02) & 0b1001001001001001001001001001001001001001001001001001001001001001;

    x
}

fn main() {
    let filename = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();

    let mut splats = {
        let file = std::fs::File::open(&filename).unwrap();

        let mut reader = std::io::BufReader::new(&file);

        let mut header = String::new();
        loop {
            let start = header.len();
            reader.read_line(&mut header).unwrap();
            if header[start..].trim() == "end_header" {
                break;
            }
        }

        println!("{}", header);

        if header.contains("property float nx") {
            load_splats::<PlySplatNormals, _>(&mut reader)
        } else {
            load_splats::<PlySplat, _>(&mut reader)
        }
    };

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for splat in splats.iter() {
        for i in 0..3 {
            min[i] = min[i].min(splat.center[i]);
            max[i] = max[i].max(splat.center[i]);
        }
    }

    let max_v = (1 << 21) - 1;

    let scale: [_; 3] = std::array::from_fn(|i| (max[i] - min[i]).recip() * max_v as f32);

    radsort::sort_by_key(&mut splats, |splat| {
        (0..3).fold(0u64, |code, i| {
            let v = ((splat.center[i] - min[i]) * scale[i])
                .round()
                .clamp(0.0, max_v as f32) as u64;

            code | (expand_bits(v) << i)
        })
    });

    std::fs::write(&output, cast_slice(&splats)).unwrap();
}
