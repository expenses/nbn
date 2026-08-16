use either::Either;
use std::io::Write;

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

impl From<PlySplatNormals> for PlySplat {
    fn from(s: PlySplatNormals) -> Self {
        Self {
            xyz: s.xyz,
            f_dc: s.f_dc,
            opacity: s.opacity,
            scale: s.scale,
            rot: s.rot,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Splat {
    center: [f32; 3],
    color_opacity: u32,
    scale: u32,
    rot: u32,
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

fn main() {
    let filename = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();
    let file = std::fs::File::open(&filename).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };

    let end_header = b"end_header\n";

    let end_header_loc =
        memchr::memmem::find(&mmap[..4096], end_header).unwrap() + end_header.len();
    let header = std::str::from_utf8(&mmap[..end_header_loc]).unwrap();
    println!("{}", header);

    let slice = &mmap[end_header_loc..];

    dbg!(slice.len());

    let iter = if header.contains("property float nx") {
        Either::Left(
            slice
                .chunks(std::mem::size_of::<PlySplatNormals>())
                .map(|chunk| {
                    let mut splat = [PlySplatNormals::default()];
                    cast_slice_mut(&mut splat).copy_from_slice(chunk);
                    splat[0].into()
                }),
        )
    } else {
        Either::Right(slice.chunks(std::mem::size_of::<PlySplat>()).map(|chunk| {
            let mut splat = [PlySplat::default()];
            cast_slice_mut(&mut splat).copy_from_slice(chunk);
            splat[0]
        }))
    };

    let mut output = std::io::BufWriter::new(std::fs::File::create(&output).unwrap());

    for (i, s) in iter.enumerate() {
        if i == 0 {
            dbg!(s);
        }
        if i % 1_000_000 == 0 {
            dbg!(i);
        }
        let opacity = 1.0 / (1.0 + (-s.opacity).exp());

        output
            .write_all(cast_slice(&[Splat {
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
            }]))
            .unwrap();
    }
}
