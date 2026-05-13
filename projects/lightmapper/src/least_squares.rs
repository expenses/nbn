use ordered_float::OrderedFloat;
use std::collections::HashMap;

use glam::{Vec2, Vec3};

// A UV seam consisting of 2 edges. Both edges have the same world positions.
#[derive(Debug)]
pub struct Seam {
    a: [Vec2; 2],
    b: [Vec2; 2],
}

fn uv_to_screen(uv: Vec2, w: u32, h: u32) -> Vec2 {
    // Not really sure what the - 0.5 is doing here. It's important though!
    uv * Vec2::new(w as _, h as _) - 0.5
}

impl Seam {
    // The number of samples is based on the length of the lines in uv screen space.
    fn num_samples(&self, w: u32, h: u32) -> u32 {
        let len_a = (uv_to_screen(self.a[0], w, h) - uv_to_screen(self.a[1], w, h)).length();
        let len_b = (uv_to_screen(self.b[0], w, h) - uv_to_screen(self.b[1], w, h)).length();
        let max_len = len_a.max(len_b).max(2.0);
        (max_len * 3.0) as u32
    }
}

pub fn find_seams(indices: &[u32], positions: &[f32], normals: &[f32], uvs: &[f32]) -> Vec<Seam> {
    let mut edge_map: HashMap<
        ([OrderedFloat<f32>; 3], [OrderedFloat<f32>; 3]),
        (Vec3, Vec3, Vec2, Vec2),
        _,
    > = HashMap::new();

    let mut seams = Vec::new();

    let get_position = |index| {
        [
            OrderedFloat(positions[index * 3]),
            OrderedFloat(positions[index * 3 + 1]),
            OrderedFloat(positions[index * 3 + 2]),
        ]
    };

    let get_normal = |index| {
        Vec3::new(
            normals[index * 3],
            normals[index * 3 + 1],
            normals[index * 3 + 2],
        )
        .normalize()
    };

    let get_uv = |index| Vec2::new(uvs[index * 2], uvs[index * 2 + 1]);

    for tri in indices.chunks(3) {
        for (from, to) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let from_pos = get_position(from as usize);
            let to_pos = get_position(to as usize);
            let from_uv = get_uv(from as usize);
            let to_uv = get_uv(to as usize);
            let from_normal = get_normal(from as usize);
            let to_normal = get_normal(to as usize);

            // See if the same edge in world space has been inserted.
            // this has the positions flipped as we're assuming a consistent winding order.
            match edge_map.entry((to_pos, from_pos)) {
                std::collections::hash_map::Entry::Vacant(_) => {
                    // Insert if not
                    edge_map.insert((from_pos, to_pos), (from_normal, to_normal, from_uv, to_uv));
                }
                // This edge has already been added once, so we have enough information to see if it's a normal edge, or a "seam edge".
                std::collections::hash_map::Entry::Occupied(entry) => {
                    let (other_from_normal, other_to_normal, other_from_uv, other_to_uv) =
                        *entry.get();
                    if other_from_normal.dot(to_normal) > 0.9
                        && other_to_normal.dot(from_normal) > 0.9
                        && (other_from_uv != to_uv || other_to_uv != from_uv)
                    {
                        // UV don't match, so we have a seam
                        seams.push(Seam {
                            a: [from_uv, to_uv],
                            // It's important that there are flipped!
                            b: [other_to_uv, other_from_uv],
                        });
                    }
                    // No longer need this edge, remove it to keep storage low
                    entry.remove();
                }
            }
        }
    }

    seams
}

pub struct Coverage<'a> {
    pub coverage: &'a [f32],
    pub width: u32,
    pub height: u32,
}

impl<'a> Coverage<'a> {
    // We're using the data copied out of the visbuffer as a coverage mask.
    fn is_covered(&self, index: usize) -> bool {
        self.coverage[index * 4 + 3] > 0.5
    }
}

pub struct PixelInfo {
    pub x: u32,
    pub y: u32,
    pub is_covered: bool,
}

fn wrap_coordinate(mut x: i32, size: u32) -> u32 {
    while x < 0 {
        x += size as i32;
    }
    while x >= size as i32 {
        x -= size as i32;
    }
    x as u32
}

fn compute_pixel_info(seams: &[Seam], coverage: &Coverage) -> (Vec<PixelInfo>, Vec<i32>) {
    let w = coverage.width;
    let h = coverage.height;

    let mut pixel_to_pixel_info_map = vec![-1; w as usize * h as usize];
    let mut pixel_info = Vec::new();

    for s in seams {
        let num_samples = s.num_samples(w, h);
        for e in [s.a, s.b] {
            let e0 = uv_to_screen(e[0], w, h);
            let e1 = uv_to_screen(e[1], w, h);
            let dt = (e1 - e0) / (num_samples - 1) as f32;
            let mut sample_point = e0;

            for _ in 0..num_samples {
                let s_x = sample_point.x as u32;
                let s_y = sample_point.y as u32;
                // Go through the four bilinear sample taps
                let xs = [s_x, s_x + 1, s_x, s_x + 1];
                let ys = [s_y, s_y, s_y + 1, s_y + 1];

                for tap in 0..4 {
                    let x = wrap_coordinate(xs[tap] as i32, w);
                    let y = wrap_coordinate(ys[tap] as i32, h);

                    let index = y as usize * w as usize + x as usize;

                    if pixel_to_pixel_info_map[index] == -1 {
                        let is_covered = coverage.is_covered(index);

                        pixel_to_pixel_info_map[index] = pixel_info.len() as i32;
                        pixel_info.push(PixelInfo { x, y, is_covered });
                    }
                }

                sample_point += dt;
            }
        }
    }

    (pixel_info, pixel_to_pixel_info_map)
}

use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

fn dilate_pixel(centerx: u32, centery: u32, image: &[f32], coverage: &Coverage) -> Vec3 {
    let mut num_pixels = 0;
    let mut sum = Vec3::ZERO;
    for yix in centery as i32 - 1..=centery as i32 + 1 {
        for xix in centerx as i32 - 1..=centerx as i32 + 1 {
            let x = wrap_coordinate(xix, coverage.width);
            let y = wrap_coordinate(yix, coverage.height);
            let index = (y * coverage.width + x) as usize;
            if coverage.is_covered(index) {
                num_pixels += 1;
                let c = Vec3::new(image[index * 4], image[index * 4 + 1], image[index * 4 + 2]);
                sum += c;
            }
        }
    }

    if num_pixels > 0 {
        sum / num_pixels as f32
    } else {
        Vec3::ZERO
    }
}

struct LeastSquaresData {
    a_tb_r: DVector<f32>,
    a_tb_g: DVector<f32>,
    a_tb_b: DVector<f32>,
    initial_guess_r: DVector<f32>,
    initial_guess_g: DVector<f32>,
    initial_guess_b: DVector<f32>,
}

fn setup_least_squares(
    pixel_info: &[PixelInfo],
    image: &[f32],
    coverage: &Coverage,
) -> LeastSquaresData {
    let num_pixels_to_optimise = pixel_info.len();

    let mut a_tb_r = DVector::zeros(num_pixels_to_optimise);
    let mut a_tb_g = DVector::zeros(num_pixels_to_optimise);
    let mut a_tb_b = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_r = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_g = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_b = DVector::zeros(num_pixels_to_optimise);

    for (i, pi) in pixel_info.iter().enumerate() {
        // Set up equality cost, trying to keep the pixel at its original value
        // Note: for non-covered pixels the weight is much lower, since those are the pixels
        // we primarily want to modify (we'll want to keep it >0 though, to reduce the risk
        // of extreme values that can't fit in 8 bit color channels)
        let weight = if pi.is_covered {
            COVERED_PIXELS_WEIGHT
        } else {
            NONCOVERED_PIXELS_WEIGHT
        };

        let colour = if pi.is_covered {
            let index = (pi.y * coverage.width + pi.x) as usize;
            Vec3::new(image[index * 4], image[index * 4 + 1], image[index * 4 + 2])
        } else {
            dilate_pixel(pi.x, pi.y, image, coverage)
        };

        // Set up the three right hand sides (one for R, G, and B).
        // Note AtRHS represents the transpose of the system matrix A multiplied by the RHS
        a_tb_r[i] = colour.x * weight;
        a_tb_g[i] = colour.y * weight;
        a_tb_b[i] = colour.z * weight;

        // Set up the initial guess for the solution.
        initial_guess_r[i] = colour.x;
        initial_guess_g[i] = colour.y;
        initial_guess_b[i] = colour.z;
    }

    LeastSquaresData {
        a_tb_r,
        a_tb_g,
        a_tb_b,
        initial_guess_r,
        initial_guess_g,
        initial_guess_b,
    }
}

const COVERED_PIXELS_WEIGHT: f32 = 1.0;
const NONCOVERED_PIXELS_WEIGHT: f32 = 0.1;
const EDGE_CONSTRAINTS_WEIGHT: f32 = 5.0;

fn conjugate_gradient_optimize(
    a: &CsrMatrix<f32>,
    guess: DVector<f32>,
    b: &DVector<f32>,
    num_iterations: u32,
    tolerance: f32,
) -> DVector<f32> {
    let mut solution: DVector<f32> = guess;
    let mut r: DVector<f32> = b - a * &solution;
    let mut p: DVector<f32> = r.clone();
    let mut rsq: f32 = r.norm_squared();
    let tolerance_sq: f32 = tolerance * tolerance;
    for i in 0..num_iterations {
        let a_p: DVector<f32> = a * &p;
        let alpha: f32 = rsq / p.dot(&a_p);
        solution.axpy(alpha, &p, 1.0);
        r.axpy(-alpha, &a_p, 1.0);
        let rsqnew: f32 = r.norm_squared();
        if rsqnew < tolerance_sq {
            dbg!(i);
            break;
        }
        if i % 10 == 0 {
            dbg!(rsqnew.sqrt());
        }
        let beta: f32 = rsqnew / rsq;
        p.axpy(1.0, &r, beta);
        rsq = rsqnew;
    }

    solution
}

fn setup_ata_matrix(
    seams: &[Seam],
    pixel_info: &[PixelInfo],
    pixel_to_pixel_info_map: &[i32],
    w: u32,
    h: u32,
) -> CsrMatrix<f32> {
    let num_pixels_to_optimise = pixel_info.len();

    let mut matrix_map: HashMap<_, f32> = HashMap::with_capacity(num_pixels_to_optimise);

    for s in seams {
        // Step through the samples of this edge, and compute sample locations for each side of the seam
        let num_samples = s.num_samples(w, h);

        let first_half_edge_start = uv_to_screen(s.a[0], w, h);
        let first_half_edge_end = uv_to_screen(s.a[1], w, h);

        let second_half_edge_start = uv_to_screen(s.b[0], w, h);
        let second_half_edge_end = uv_to_screen(s.b[1], w, h);

        let first_half_edge_step =
            (first_half_edge_end - first_half_edge_start) / (num_samples - 1) as f32;
        let second_half_edge_step =
            (second_half_edge_end - second_half_edge_start) / (num_samples - 1) as f32;

        let mut first_half_edge_sample = first_half_edge_start;
        let mut second_half_edge_sample = second_half_edge_start;
        for _ in 0..num_samples {
            // Sample locations for the two corresponding sets of sample points
            let (first_half_edge, first_half_edge_weights) = calculate_samples_and_weights(
                pixel_to_pixel_info_map,
                first_half_edge_sample,
                w,
                h,
            );
            let (second_half_edge, second_half_edge_weights) = calculate_samples_and_weights(
                pixel_to_pixel_info_map,
                second_half_edge_sample,
                w,
                h,
            );

            /*
            Now, compute the covariance for the difference of these two vectors.
            If a is the first vector (first half edge) and b is the second, then we compute the covariance, without
            intermediate storage, like so:
            (a-b)*(a-b)^t = a*a^t + b*b^t - a*b^t-b*a^t
            */
            for i in 0..4 {
                for j in 0..4 {
                    // + a*a^t
                    *matrix_map
                        .entry((first_half_edge[i], first_half_edge[j]))
                        .or_default() += first_half_edge_weights[i] * first_half_edge_weights[j];
                    // + b*b^t
                    *matrix_map
                        .entry((second_half_edge[i], second_half_edge[j]))
                        .or_default() += second_half_edge_weights[i] * second_half_edge_weights[j];

                    // - a*b^t
                    *matrix_map
                        .entry((first_half_edge[i], second_half_edge[j]))
                        .or_default() -= first_half_edge_weights[i] * second_half_edge_weights[j];

                    // - b*a^t
                    *matrix_map
                        .entry((second_half_edge[i], first_half_edge[j]))
                        .or_default() -= second_half_edge_weights[i] * first_half_edge_weights[j];
                }
            }

            first_half_edge_sample += first_half_edge_step;
            second_half_edge_sample += second_half_edge_step;
        }
    }

    for (i, pi) in pixel_info.iter().enumerate() {
        // Set up equality cost, trying to keep the pixel at its original value
        // Note: for non-covered pixels the weight is much lower, since those are the pixels
        // we primarily want to modify (we'll want to keep it >0 though, to reduce the risk
        // of extreme values that can't fit in 8 bit color channels)
        let weight = if pi.is_covered {
            COVERED_PIXELS_WEIGHT
        } else {
            NONCOVERED_PIXELS_WEIGHT
        };

        *matrix_map.entry((i as i32, i as i32)).or_default() += weight;
    }

    let mut coo_matrix = CooMatrix::new(pixel_info.len(), pixel_info.len());

    for (&(x, y), &value) in matrix_map.iter() {
        coo_matrix.push(x as usize, y as usize, value);
    }

    CsrMatrix::from(&coo_matrix)
}

fn calculate_samples_and_weights(
    pixel_map: &[i32],
    sample: Vec2,
    width: u32,
    height: u32,
) -> ([i32; 4], [f32; 4]) {
    let truncu = sample.x as i32;
    let truncv = sample.y as i32;

    let xs = [truncu, truncu + 1, truncu + 1, truncu];
    let ys = [truncv, truncv, truncv + 1, truncv + 1];
    let mut out_ixs = [0; 4];
    for i in 0..4 {
        let x = wrap_coordinate(xs[i], width);
        let y = wrap_coordinate(ys[i], height);
        let index = y as usize * width as usize + x as usize;
        out_ixs[i] = pixel_map[index];
    }

    let frac_x = sample.x - truncu as f32;
    let frac_y = sample.y - truncv as f32;
    let out_weights = [
        EDGE_CONSTRAINTS_WEIGHT * (1.0 - frac_x) * (1.0 - frac_y),
        EDGE_CONSTRAINTS_WEIGHT * frac_x * (1.0 - frac_y),
        EDGE_CONSTRAINTS_WEIGHT * frac_x * frac_y,
        EDGE_CONSTRAINTS_WEIGHT * (1.0 - frac_x) * frac_y,
    ];

    (out_ixs, out_weights)
}

pub fn get_solutions(
    coverage: &Coverage,
    seams: &[Seam],
    output_slice: &[f32],
) -> (DVector<f32>, DVector<f32>, DVector<f32>, Vec<PixelInfo>) {
    let (pixel_info, img) = compute_pixel_info(seams, coverage);

    dbg!(pixel_info.len());

    let a_t_a = setup_ata_matrix(seams, &pixel_info, &img, coverage.width, coverage.height);

    dbg!("data");

    let data = setup_least_squares(&pixel_info, output_slice, coverage);

    dbg!("running");

    let solution_r =
        conjugate_gradient_optimize(&a_t_a, data.initial_guess_r, &data.a_tb_r, 10000, 1e-3);

    let solution_g =
        conjugate_gradient_optimize(&a_t_a, data.initial_guess_g, &data.a_tb_g, 10000, 1e-3);

    let solution_b =
        conjugate_gradient_optimize(&a_t_a, data.initial_guess_b, &data.a_tb_b, 10000, 1e-3);

    (solution_r, solution_g, solution_b, pixel_info)
}
