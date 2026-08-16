use std::io::{Read, Write};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PlySplat {
    xyz: [f32; 3],
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

use ply_rs::ply;

const SH_C0: f32 = 0.28209479177387814;

pub fn cast_slice<I: Copy, O: Copy>(slice: &[I]) -> &[O] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const O,
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
    let mut buf_read = std::io::BufReader::new(std::fs::File::open(&filename).unwrap());
    let p = ply_rs::parser::Parser::<ply::DefaultElement>::new();
    let header = p.read_header(&mut buf_read).unwrap();
    dbg!(&header);
    let mut remaining = Vec::new();
    buf_read.read_to_end(&mut remaining).unwrap();
    let splats = cast_slice::<_, PlySplat>(&remaining);

    let mut output = std::fs::File::create(&output).unwrap();

    let splats: Vec<_> = splats
        .iter()
        .map(|s| {
            let opacity = 1.0 / (1.0 + (-s.opacity).exp());

            Splat {
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
        })
        .collect();

    output.write_all(cast_slice(&splats)).unwrap();
}
