slang_struct::slang_include!("shaders/ntc/structs.slang");

use nbn::vk;
use rand::{Rng, RngExt};
use std::fmt::Write;
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(clap::Subcommand)]
enum Mode {
    Eval {
        path: std::path::PathBuf,
    },
    Train {
        #[arg(long, num_args = 1..)]
        srgb: Vec<std::path::PathBuf>,
        #[arg(long, num_args = 1..)]
        non_srgb: Vec<std::path::PathBuf>,
        #[arg(long, num_args = 1..)]
        scalar: Vec<std::path::PathBuf>,
        #[arg(short, long, default_value_t = 10_000)]
        iterations: u32,
        #[arg(short, long)]
        size: Option<u32>,
        #[arg(long, default_value_t = 1000)]
        loss_eval_freq: u32,
        #[arg(long, default_value = "0.01")]
        learning_rate: f32,
        #[arg(long, default_value = "0.001")]
        mlp_learning_rate: f32,
        #[arg(long, default_value = "64")]
        batch_size: u32,
    },
}

use clap::Parser;

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    mode: Mode,
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
    data: Tensor,
    size: u32,
    num_mip_levels: i32,
    num_blocks: u32,
    bitmasks: nbn::Buffer,
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

        let num_values = num_blocks as usize * (3 * 2 + 16);

        let bitmasks = staging_buffer.create_buffer_from_slice(
            device,
            "bitmasks",
            &vec![0_u32; num_values.div_ceil(32)],
        );

        Self {
            data: Tensor::from_halfs(
                device,
                staging_buffer,
                &(0..num_values)
                    .map(|_| half::f16::from_f32(rng.random_range(0.0..1.0)))
                    .collect::<Vec<_>>(),
            ),
            bitmasks,
            num_blocks,
            size,
            num_mip_levels,
        }
    }

    fn as_struct(&self) -> LatentTexture {
        LatentTexture {
            data: *self.data.data,
            grads: self.data.training.as_ref().map(|t| *t.grad).unwrap_or(0),
            num_blocks: self.num_blocks,
            size: self.size as _,
            num_mip_levels: self.num_mip_levels,
            bitmasks: *self.bitmasks,
        }
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

    for i in 0..3 {
        let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

        unsafe {
            device
                .begin_command_buffer(*command_buffer, &Default::default())
                .unwrap();

            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, *render);
            device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);
            device.push_constants::<RenderComputePushConstants>(
                &command_buffer,
                RenderComputePushConstants {
                    output: *output,
                    network: **network,
                    resolution: size as _,
                    channel_offset: i * 3,
                    mip_level: 1.0,
                },
            );
            device.cmd_dispatch(*command_buffer, (size as u32).div_ceil(64), size as u32, 1);
            device.end_command_buffer(*command_buffer).unwrap();
            device.submit_and_wait_on_command_buffer(&command_buffer);
        }

        let slice = output.try_as_slice::<f32>().unwrap();

        image::ImageBuffer::<image::Rgb<f32>, Vec<f32>>::from_raw(
            size as _,
            size as _,
            slice.to_vec(),
        )
        .unwrap()
        .save(&format!("{}.exr", i))
        .unwrap();
    }
}

fn optimize(
    device: &nbn::Device,
    command_buffer: &nbn::CommandBuffer,
    tensor: &Tensor,
    iteration: u32,
    learning_rate: f32,
) {
    let training = tensor.training.as_ref().unwrap();
    unsafe {
        device.push_constants::<OptimizerPushConstants>(
            command_buffer,
            OptimizerPushConstants {
                primal: *tensor.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate,
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
    texture: &TextureData,
    iteration: u32,
    learning_rate: f32,
) {
    let training = texture.data.training.as_ref().unwrap();
    unsafe {
        device.push_constants::<OptimizeLatentTexturesConstants>(
            command_buffer,
            OptimizeLatentTexturesConstants {
                primal: *texture.data.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate,
                num_values: texture.data.size as _,
                iteration: iteration as _,
                bitmask: *texture.bitmasks,
            },
        );
        device.cmd_dispatch(
            **command_buffer,
            (texture.data.size as u32).div_ceil(64),
            1,
            1,
        );
    }
}

fn main() {
    let device = nbn::Device::new(None);

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Graphics);

    let shader = device.load_shader("shaders/compiled/ntc.spv");

    let args = Arguments::parse();

    match args.mode {
        Mode::Eval { path } => {}
        Mode::Train {
            srgb,
            non_srgb,
            scalar,
            iterations,
            mut size,
            loss_eval_freq,
            learning_rate,
            batch_size,
            mlp_learning_rate,
        } => {
            let (images, indices, channel_counts) = srgb
                .iter()
                .map(|image| (image, vk::Format::R8G8B8A8_SRGB, 3))
                .chain(
                    non_srgb
                        .iter()
                        .map(|image| (image, vk::Format::R8G8B8A8_UNORM, 3)),
                )
                .chain(
                    scalar
                        .iter()
                        .map(|image| (image, vk::Format::R8G8B8A8_UNORM, 1)),
                )
                .map(|(filepath, format, channel_count)| {
                    let image = image::open(&filepath)
                        .expect(&format!("{}", filepath.display()))
                        .to_rgba8();
                    size = size.or(Some(image.width()));

                    let mut mip_size = image.width().max(image.height());
                    let mut num_mips = 0;
                    while mip_size > 0 {
                        num_mips += 1;
                        mip_size >>= 1;
                    }

                    dbg!(num_mips);

                    let image = device.register_owned_image(
                        staging_buffer.create_sampled_image(
                            &device,
                            nbn::SampledImageDescriptor {
                                name: &filepath.display().to_string(),
                                format,
                                extent: nbn::ImageExtent::D2 {
                                    width: image.width(),
                                    height: image.height(),
                                },
                            },
                            &image,
                            nbn::QueueType::Compute,
                            nbn::ImageLods::Generate(num_mips),
                        ),
                        false,
                    );

                    let index = *image;
                    (image, index, channel_count)
                })
                .collect::<(Vec<_>, Vec<u32>, Vec<u32>)>();
            let size = size.unwrap();
            dbg!(size);

            let image_indices =
                staging_buffer.create_buffer_from_slice(&device, "image_indices", &indices);

            let calculate_grads = device.create_compute_pipeline(&shader, c"calculate_grads");
            let optimizer_step = device.create_compute_pipeline(&shader, c"optimizer_step");
            let optimize_latent_textures =
                device.create_compute_pipeline(&shader, c"optimize_latent_textures");
            let sum_loss = device.create_compute_pipeline(&shader, c"sum_loss");

            let mut rng = rand::rng();
            let network = NetworkData::train(&device, &mut staging_buffer, size, &mut rng);

            let network_buffer =
                staging_buffer.create_buffer_from_slice(&device, "network", &[network.as_struct()]);

            let channel_counts =
                staging_buffer.create_buffer_from_slice(&device, "channel_counts", &channel_counts);

            staging_buffer.finish(&device);

            let run_name = {
                let mut name = String::from("lr");
                write!(name, "{}", learning_rate).unwrap();
                name.push_str("_mlr");
                write!(name, "{}", mlp_learning_rate).unwrap();
                name.push_str("_bs");
                write!(name, "{}", batch_size).unwrap();
                name
            };
            let log_dir = format!("logs/{}", run_name);
            let mut writer = SummaryWriter::new(&log_dir);

            let start = std::time::Instant::now();

            let loss_total = device
                .create_buffer(nbn::BufferDescriptor {
                    name: "loss_total",
                    size: 4,
                    ty: nbn::MemoryLocation::GpuToCpu,
                })
                .unwrap();

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
                            grid_size: batch_size,
                            channel_counts: *channel_counts,
                        },
                    );
                    device.cmd_dispatch(
                        *command_buffer,
                        (batch_size * batch_size).div_ceil(64),
                        1,
                        1,
                    );
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *optimizer_step,
                    );
                    optimize(
                        &device,
                        &command_buffer,
                        &network.weights_and_biases,
                        i + 1,
                        mlp_learning_rate,
                    );
                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *optimize_latent_textures,
                    );
                    optimize_half(
                        &device,
                        &command_buffer,
                        &network.latent_texture_1,
                        i + 1,
                        learning_rate,
                    );
                    optimize_half(
                        &device,
                        &command_buffer,
                        &network.latent_texture_2,
                        i + 1,
                        learning_rate,
                    );
                    optimize_half(
                        &device,
                        &command_buffer,
                        &network.latent_texture_3,
                        i + 1,
                        learning_rate,
                    );
                    optimize_half(
                        &device,
                        &command_buffer,
                        &network.latent_texture_4,
                        i + 1,
                        learning_rate,
                    );

                    if i % loss_eval_freq == 0 {
                        device.cmd_fill_buffer(
                            *command_buffer,
                            *loss_total.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0,
                        );
                        device.cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            *sum_loss,
                        );
                        device.push_constants::<SumLossPushConstants>(
                            &command_buffer,
                            SumLossPushConstants {
                                network: *network_buffer,
                                textures: *image_indices,
                                resolution: network.size as _,
                                num_textures: images.len() as _,
                                channel_counts: *channel_counts,
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

                    device.end_command_buffer(*command_buffer).unwrap();
                    device.submit_and_wait_on_command_buffer(&command_buffer);
                }

                if i % loss_eval_freq == 0 {
                    let loss = loss_total.try_as_slice::<f32>().unwrap()[0];
                    let loss = loss / network.size as f32 / network.size as f32 / 16.0;
                    println!(
                        "Loss: {:.8} PSNR: {:.4} dB, Batch Size: {}",
                        loss,
                        l2_psnr(loss),
                        batch_size
                    );
                    writer.add_scalar("loss", loss, i as usize);
                    writer.add_scalar("psnr", l2_psnr(loss), i as usize);
                    writer.flush();
                }
            }

            let duration = std::time::Instant::now() - start;
            dbg!(duration);

            eval(&device, &shader, &network_buffer, network.size);
        }
    };
}

fn l1_psnr(loss: f32) -> f32 {
    20.0 * (1.0 / loss).log10()
}

fn l2_psnr(loss: f32) -> f32 {
    10.0 * (1.0 / loss).log10()
}
