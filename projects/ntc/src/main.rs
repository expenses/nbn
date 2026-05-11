slang_struct::slang_include!("shaders/ntc/structs.slang");

use nbn::vk;
use ndarray_npy as npy;
use rand::{Rng, RngExt};

use ndarray::{Array0, Array1};

#[derive(clap::Subcommand)]
enum Mode {
    Eval {
        path: std::path::PathBuf,
    },
    Train {
        images: Vec<std::path::PathBuf>,
        #[arg(short, long)]
        iterations: u32,
        #[arg(short, long)]
        size: u32,
    },
}

use clap::Parser;

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    mode: Mode,
}

fn upload_array(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    reader: &mut npy::NpzReader<std::fs::File>,
    name: &str,
) -> nbn::Buffer {
    let array: Array1<f32> = reader.by_name(name).unwrap();
    let slice = array.as_slice().unwrap();

    staging_buffer.create_buffer_from_slice(device, name, slice)
}

fn network_data<R: Rng>(rng: &mut R) -> Vec<f32> {
    let mut data = Vec::new();
    for _ in 0..16 * 64 {
        data.push(rng.random_range(-0.5..0.5));
    }
    for _ in 0..64 {
        data.push(0.0);
    }
    for _ in 0..64 * 64 {
        data.push(rng.random_range(-0.5..0.5));
    }
    for _ in 0..64 {
        data.push(0.0);
    }
    for _ in 0..64 * 16 {
        data.push(rng.random_range(-0.5..0.5));
    }
    for _ in 0..16 {
        data.push(0.0);
    }
    data
}

struct TextureData {
    endpoints: Tensor,
    alpha: Tensor,
    size: u32,
    num_mip_levels: i32,
}

impl TextureData {
    fn train<R: Rng>(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        size: u32,
        rng: &mut R,
    ) -> Self {
        let mut size_in_blocks = size / 4;
        let mut num_blocks = size_in_blocks * size_in_blocks;

        let mut num_mip_levels = 1;
        while size_in_blocks > 1 {
            num_mip_levels += 1;
            size_in_blocks >>= 1;
            num_blocks += size_in_blocks * size_in_blocks
        }

        Self {
            endpoints: Tensor::from_halfs(
                device,
                staging_buffer,
                &(0..num_blocks * 3 * 2)
                    .map(|_| half::f16::from_f32(rng.random_range(0.0..1.0)))
                    .collect::<Vec<_>>(),
            ),
            alpha: Tensor::from_halfs(
                device,
                staging_buffer,
                &(0..num_blocks * 16)
                    .map(|_| half::f16::from_f32(rng.random_range(0.0..1.0)))
                    .collect::<Vec<_>>(),
            ),
            size,
            num_mip_levels,
        }
    }

    fn as_struct(&self) -> LatentTexture {
        LatentTexture {
            endpoints: *self.endpoints.data,
            alpha: *self.alpha.data,
            endpoints_grad: self
                .endpoints
                .training
                .as_ref()
                .map(|t| *t.grad)
                .unwrap_or(0),
            alpha_grad: self.alpha.training.as_ref().map(|t| *t.grad).unwrap_or(0),
            size: self.size as _,
            num_mip_levels: self.num_mip_levels,
        }
    }

    fn optimize(&self, device: &nbn::Device, command_buffer: &nbn::CommandBuffer, iter: u32) {
        optimize_half(device, command_buffer, &self.alpha, iter);
        optimize_half(device, command_buffer, &self.endpoints, iter);
    }
}

struct Training {
    grad: nbn::Buffer,
    m: nbn::Buffer,
    v: nbn::Buffer,
}

struct Tensor {
    data: nbn::Buffer,
    training: Option<Training>,
    size: usize,
}

impl Tensor {
    fn eval_only(data: nbn::Buffer) -> Self {
        Self {
            data,
            training: None,
            size: 0,
        }
    }

    fn from_halfs(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        data: &[half::f16],
    ) -> Self {
        let zeros = vec![0.0_f32; data.len()];

        Self {
            data: staging_buffer.create_buffer_from_slice(device, "data", data),
            training: Some(Training {
                grad: staging_buffer.create_buffer_from_slice(device, "grad", &zeros),
                m: staging_buffer.create_buffer_from_slice(device, "m", &zeros),
                v: staging_buffer.create_buffer_from_slice(device, "v", &zeros),
            }),
            size: data.len(),
        }
    }

    fn from_data(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        data: &[f32],
    ) -> Self {
        let zeros = vec![0.0_f32; data.len()];

        Self {
            data: staging_buffer.create_buffer_from_slice(device, "data", data),
            training: Some(Training {
                grad: staging_buffer.create_buffer_from_slice(device, "grad", &zeros),
                m: staging_buffer.create_buffer_from_slice(device, "m", &zeros),
                v: staging_buffer.create_buffer_from_slice(device, "v", &zeros),
            }),
            size: data.len(),
        }
    }
}

struct NetworkData {
    weights_and_biases: Tensor,
    latent_texture_1: TextureData,
    latent_texture_2: TextureData,
    latent_texture_3: TextureData,
    latent_texture_4: TextureData,
    size: u32,
}

impl NetworkData {
    fn from_npz(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        npz: &mut npy::NpzReader<std::fs::File>,
    ) -> Self {
        let size: Array0<i64> = npz.by_name("size").unwrap();
        let size = size.into_scalar() as u32;

        Self {
            weights_and_biases: Tensor::eval_only(upload_array(
                device,
                staging_buffer,
                npz,
                "weights_and_biases",
            )),
            latent_texture_1: TextureData {
                endpoints: Tensor::eval_only(upload_array(
                    device,
                    staging_buffer,
                    npz,
                    "lt1_endpoints",
                )),
                alpha: Tensor::eval_only(upload_array(device, staging_buffer, npz, "lt1_alpha")),
                size,
                num_mip_levels: 1,
            },
            latent_texture_2: TextureData {
                endpoints: Tensor::eval_only(upload_array(
                    device,
                    staging_buffer,
                    npz,
                    "lt2_endpoints",
                )),
                alpha: Tensor::eval_only(upload_array(device, staging_buffer, npz, "lt2_alpha")),
                size,
                num_mip_levels: 1,
            },
            latent_texture_3: TextureData {
                endpoints: Tensor::eval_only(upload_array(
                    device,
                    staging_buffer,
                    npz,
                    "lt3_endpoints",
                )),

                alpha: Tensor::eval_only(upload_array(device, staging_buffer, npz, "lt3_alpha")),
                size: size / 2,
                num_mip_levels: 1,
            },
            latent_texture_4: TextureData {
                endpoints: Tensor::eval_only(upload_array(
                    device,
                    staging_buffer,
                    npz,
                    "lt4_endpoints",
                )),
                alpha: Tensor::eval_only(upload_array(device, staging_buffer, npz, "lt4_alpha")),
                size: size / 2,
                num_mip_levels: 1,
            },
            size: size as _,
        }
    }

    fn train<R: Rng>(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        size: u32,
        rng: &mut R,
    ) -> Self {
        Self {
            weights_and_biases: Tensor::from_data(device, staging_buffer, &network_data(rng)),
            latent_texture_1: TextureData::train(device, staging_buffer, size, rng),
            latent_texture_2: TextureData::train(device, staging_buffer, size, rng),
            latent_texture_3: TextureData::train(device, staging_buffer, size / 2, rng),
            latent_texture_4: TextureData::train(device, staging_buffer, size / 2, rng),
            size,
        }
    }

    fn as_struct(&self) -> Network {
        Network {
            weights_and_biases: *self.weights_and_biases.data,
            weights_and_biases_grad: self
                .weights_and_biases
                .training
                .as_ref()
                .map(|t| *t.grad)
                .unwrap_or(0),
            latent_texture_1: self.latent_texture_1.as_struct(),
            latent_texture_2: self.latent_texture_2.as_struct(),
            latent_texture_3: self.latent_texture_3.as_struct(),
            latent_texture_4: self.latent_texture_4.as_struct(),
        }
    }
}

fn eval(device: &nbn::Device, shader: &nbn::ShaderModule, network: &nbn::Buffer, size: u32) {
    let render = device.create_compute_pipeline(&shader, c"render_compute");

    let output = device
        .create_buffer(nbn::BufferDescriptor {
            size: size as u64 * size as u64 * 3 * 4,
            name: "output floats",
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let pc = RenderComputePushConstants {
        output: *output,
        network: **network,
        resolution: size as _,
    };

    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();

        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, *render);
        device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);
        device.push_constants::<RenderComputePushConstants>(&command_buffer, pc);
        device.cmd_dispatch(*command_buffer, (size as u32).div_ceil(64), size as u32, 1);
        device.end_command_buffer(*command_buffer).unwrap();
        device.submit_and_wait_on_command_buffer(&command_buffer);
    }

    let slice = output.try_as_slice::<f32>().unwrap();

    image::ImageBuffer::<image::Rgb<f32>, Vec<f32>>::from_raw(size as _, size as _, slice.to_vec())
        .unwrap()
        .save("out.exr")
        .unwrap();
}

fn optimize(
    device: &nbn::Device,
    command_buffer: &nbn::CommandBuffer,
    tensor: &Tensor,
    iteration: u32,
) {
    unsafe {
        let training = tensor.training.as_ref().unwrap();
        device.push_constants::<OptimizerPushConstants>(
            command_buffer,
            OptimizerPushConstants {
                primal: *tensor.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate: 0.001,
                num_values: tensor.size as _,
                iteration: iteration as _,
            },
        );
        device.cmd_dispatch(**command_buffer, (tensor.size as u32).div_ceil(64), 1, 1);
    }
}

fn optimize_half(
    device: &nbn::Device,
    command_buffer: &nbn::CommandBuffer,
    tensor: &Tensor,
    iteration: u32,
) {
    unsafe {
        let training = tensor.training.as_ref().unwrap();
        device.push_constants::<OptimizerHalfPushConstants>(
            command_buffer,
            OptimizerHalfPushConstants {
                primal: *tensor.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate: 0.001,
                num_values: tensor.size as _,
                iteration: iteration as _,
            },
        );
        device.cmd_dispatch(**command_buffer, (tensor.size as u32).div_ceil(64), 1, 1);
    }
}

fn main() {
    let device = nbn::Device::new(None);

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Transfer);

    let shader = device.load_shader("shaders/compiled/ntc.spv");

    let args = Arguments::parse();

    match args.mode {
        Mode::Eval { path } => {
            let file = std::fs::File::open(path).unwrap();
            let mut npz = npy::NpzReader::new(file).unwrap();

            println!("Keys: {:?}", npz.names());

            let network = NetworkData::from_npz(&device, &mut staging_buffer, &mut npz);

            let network_buffer =
                staging_buffer.create_buffer_from_slice(&device, "network", &[network.as_struct()]);

            staging_buffer.finish(&device);

            eval(&device, &shader, &network_buffer, network.size);
        }
        Mode::Train {
            images,
            iterations,
            size,
        } => {
            let (images, indices) = images
                .into_iter()
                .map(|filepath| {
                    let image = image::open(&filepath).unwrap().to_rgba8();
                    let image = device.register_owned_image(
                        staging_buffer.create_sampled_image(
                            &device,
                            nbn::SampledImageDescriptor {
                                name: &filepath.display().to_string(),
                                format: vk::Format::R8G8B8A8_SRGB,
                                extent: nbn::ImageExtent::D2 {
                                    width: image.width(),
                                    height: image.height(),
                                },
                            },
                            &image,
                            nbn::QueueType::Compute,
                            &[0],
                        ),
                        false,
                    );

                    let index = *image;
                    (image, index)
                })
                .collect::<(Vec<_>, Vec<u32>)>();
            let image_indices =
                staging_buffer.create_buffer_from_slice(&device, "image_indices", &indices);

            let calculate_grads = device.create_compute_pipeline(&shader, c"calculate_grads");
            let optimizer_step = device.create_compute_pipeline(&shader, c"optimizer_step");
            let optimizer_step_half =
                device.create_compute_pipeline(&shader, c"optimizer_step_half");
            let sum_loss = device.create_compute_pipeline(&shader, c"sum_loss");

            let mut rng = rand::rng();
            let network = NetworkData::train(&device, &mut staging_buffer, size, &mut rng);

            let network_buffer =
                staging_buffer.create_buffer_from_slice(&device, "network", &[network.as_struct()]);

            staging_buffer.finish(&device);

            let start = std::time::Instant::now();

            for i in 0..iterations {
                let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

                if i % 100 == 0 {
                    println!("{}", i);
                }
                unsafe {
                    device
                        .begin_command_buffer(*command_buffer, &Default::default())
                        .unwrap();
                    device.bind_internal_descriptor_sets(
                        &command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                    );
                }

                let loss_total = device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "loss_total",
                        size: 4,
                        ty: nbn::MemoryLocation::GpuToCpu,
                    })
                    .unwrap();

                let grid_size = 64;

                unsafe {
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *calculate_grads,
                    );
                    device.push_constants::<CalculateGradsPushConstants>(
                        &command_buffer,
                        CalculateGradsPushConstants {
                            network: *network_buffer,
                            textures: *image_indices,
                            num_textures: images.len() as _,
                            iteration: i as _,
                            grid_size,
                        },
                    );
                    device.cmd_dispatch(
                        *command_buffer,
                        (grid_size * grid_size).div_ceil(64),
                        1,
                        1,
                    );
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *optimizer_step,
                    );
                    optimize(&device, &command_buffer, &network.weights_and_biases, i + 1);
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *optimizer_step_half,
                    );
                    network
                        .latent_texture_1
                        .optimize(&device, &command_buffer, i + 1);
                    network
                        .latent_texture_2
                        .optimize(&device, &command_buffer, i + 1);
                    network
                        .latent_texture_3
                        .optimize(&device, &command_buffer, i + 1);
                    network
                        .latent_texture_4
                        .optimize(&device, &command_buffer, i + 1);
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *sum_loss,
                    );

                    if i % 1000 == 0 {
                        device.cmd_fill_buffer(
                            *command_buffer,
                            *loss_total.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0,
                        );

                        device.push_constants::<SumLossPushConstants>(
                            &command_buffer,
                            SumLossPushConstants {
                                network: *network_buffer,
                                textures: *image_indices,
                                resolution: network.size as _,
                                num_textures: images.len() as _,
                                total: *loss_total,
                            },
                        );
                        device.cmd_dispatch(
                            *command_buffer,
                            (network.size * network.size).div_ceil(64),
                            1,
                            1,
                        );
                    }
                }

                unsafe {
                    device.end_command_buffer(*command_buffer).unwrap();
                    device.submit_and_wait_on_command_buffer(&command_buffer);
                }

                if i % 1000 == 0 {
                    let loss = loss_total.try_as_slice::<f32>().unwrap()[0];
                    let mae = loss / network.size as f32 / network.size as f32 / 16.0;
                    let psnr = 10.0 * (1.0 / mae).log10();
                    println!("Loss: {:.8} PSNR: {:.4} dB", mae, psnr);
                }
            }

            let duration = std::time::Instant::now() - start;
            dbg!(duration);

            eval(&device, &shader, &network_buffer, network.size);
        }
    };
}
