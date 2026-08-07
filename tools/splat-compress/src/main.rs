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
    rot: [f32; 4],
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

    // Something to do with the fact that none of the other
    // three components can exceed 1/sqrt(2) in magnitude
    let scaler = std::f32::consts::SQRT_2;

    // note: we encode a cyclical swizzle to be able to recover the order via rotation
    let a = quantize_10b(q[(qc + 1) % 4] * scaler * sign);
    let b = quantize_10b(q[(qc + 2) % 4] * scaler * sign);
    let c = quantize_10b(q[(qc + 3) % 4] * scaler * sign);

    a | (b << 10) | (c << 20) | ((qc as u32) << 30)
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

            let quantize = |v: f32| (v.clamp(0.0, 1.0).round() * 255.0) as u32;
            let quantize_sh = |v: f32| quantize((v * SH_C0) + 0.5);

            let scale = s.scale.map(f32::exp);

            Splat {
                center: s.xyz,
                color_opacity: quantize_sh(s.f_dc[0])
                    | (quantize_sh(s.f_dc[1]) << 8)
                    | (quantize_sh(s.f_dc[2]) << 16)
                    | (quantize(opacity) << 24),
                scale: quantize_10b(scale[0])
                    | (quantize_10b(scale[1]) << 10)
                    | (quantize_10b(scale[2]) << 20),
                // PLY (w,x,y,z) -> (x,y,z,w)
                rot: ([s.rot[1], s.rot[2], s.rot[3], s.rot[0]]),
            }
        })
        .collect();

    output.write_all(cast_slice(&splats)).unwrap();
}
