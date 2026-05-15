slang_struct::slang_include!("shaders/ntc/structs.slang");

mod viewer;
mod writer;

use viewer::App;
use writer::{AntcTexture, AntcWriter};

use nbn::{vk, winit};
use rand::{Rng, RngExt};
use std::fmt::Write;
use std::path::PathBuf;
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
enum Channels {
    Srgba,
    Urgba,
    Srgb,
    Urgb,
    Srg,
    Urg,
    Sgb,
    Ugb,
    Sr,
    Ur,
    Sg,
    Ug,
    Sb,
    Ub,
}

impl Channels {
    fn is_srgb(&self) -> bool {
        match self {
            Self::Srgba | Self::Srgb | Self::Srg | Self::Sgb | Self::Sr | Self::Sg | Self::Sb => {
                true
            }
            _ => false,
        }
    }

    fn as_bitmask(&self) -> u8 {
        match self {
            Self::Srgba | Self::Urgba => 0b1111,
            Self::Srgb | Self::Urgb => 0b0111,
            Self::Srg | Self::Urg => 0b0011,
            Self::Sgb | Self::Ugb => 0b0110,
            Self::Sr | Self::Ur => 0b0001,
            Self::Sg | Self::Ug => 0b0010,
            Self::Sb | Self::Ub => 0b0100,
        }
    }
}

#[derive(clap::Args)]
struct ImagePaths {
    #[arg(long, num_args = 1..)]
    paths: Vec<PathBuf>,
    #[arg(long, num_args = 1..)]
    channels: Vec<Channels>,
}

#[derive(clap::Subcommand)]
enum Mode {
    Eval {
        path: PathBuf,
    },
    Train {
        #[command(flatten)]
        paths: ImagePaths,
        #[arg(short, long, default_value_t = 100_000)]
        iterations: u32,
        #[arg(short, long)]
        size: Option<u32>,
        #[arg(long, default_value_t = 0.02)]
        learning_rate: f32,
        #[arg(long, default_value_t = 0.002)]
        mlp_learning_rate: f32,
        #[arg(long, default_value_t = 64)]
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

const NUM_PARAMS: usize = 6288;

#[test]
fn check_num_values() {
    let mut rng = rand::rng();
    assert_eq!(network_data(&mut rng).len(), NUM_PARAMS);
}

struct TextureData {
    data: Tensor,
    size: u32,
    num_mip_levels: i32,
    num_blocks: u32,
    block_offsets: Vec<u32>,
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

        let mut block_offsets = vec![0];

        let mut num_mip_levels = 1;
        while size_in_blocks > 1 {
            num_mip_levels += 1;
            size_in_blocks >>= 1;
            block_offsets.push(num_blocks * 8);
            num_blocks += size_in_blocks * size_in_blocks;
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
            block_offsets,
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
    textures: [TextureData; 4],
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
            textures: [
                TextureData::train(device, staging_buffer, size, rng),
                TextureData::train(device, staging_buffer, size, rng),
                TextureData::train(device, staging_buffer, size / 2, rng),
                TextureData::train(device, staging_buffer, size / 2, rng),
            ],
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
            textures: std::array::from_fn(|i| self.textures[i].as_struct()),
        }
    }
}

const ADAM_BETA_1: f32 = 0.9;
const ADAM_BETA_2: f32 = 0.999;

fn optimize(
    device: &nbn::Device,
    command_buffer: &nbn::CommandBuffer,
    tensor: &Tensor,
    iteration: u32,
    learning_rate: f32,
) {
    let training = tensor.training.as_ref().unwrap();
    unsafe {
        device.push_constants::<OptimizePushConstants>(
            command_buffer,
            OptimizePushConstants {
                primal: *tensor.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate,
                num_values: tensor.size as _,
                adam_m_factor: (1.0 - ADAM_BETA_1.powi(iteration as _)).recip(),
                adam_v_factor: (1.0 - ADAM_BETA_2.powi(iteration as _)).recip(),
                // unused
                bitmask: 0,
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
        device.push_constants::<OptimizePushConstants>(
            command_buffer,
            OptimizePushConstants {
                primal: *texture.data.data,
                grad: *training.grad,
                mean: *training.m,
                variance: *training.v,
                learning_rate,
                num_values: texture.data.size as _,
                adam_m_factor: (1.0 - ADAM_BETA_1.powi(iteration as _)).recip(),
                adam_v_factor: (1.0 - ADAM_BETA_2.powi(iteration as _)).recip(),
                bitmask: *texture.bitmasks,
            },
        );
        device.cmd_dispatch(
            **command_buffer,
            (texture.data.size as u32).div_ceil(64 * 64),
            1,
            1,
        );
    }
}

struct Pipelines {
    calculate_grads: nbn::Pipeline,
    optimizer_step: nbn::Pipeline,
    optimize_latent_textures: nbn::Pipeline,
    calculate_mlp_grads_only: nbn::Pipeline,
}

fn optimization_iter(
    device: &nbn::Device,
    i: u32,
    batch_size: u32,
    push_constants: CalculateGradsPushConstants,
    pipelines: &Pipelines,
    loss_total: &nbn::Buffer,
    network: &NetworkData,
    mlp_learning_rate: f32,
    learning_rate: f32,
    writer: &mut SummaryWriter,
) {
    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();
        device.cmd_fill_buffer(*command_buffer, *loss_total.buffer, 0, vk::WHOLE_SIZE, 0);
        device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *pipelines.calculate_grads,
        );
        device.push_constants::<CalculateGradsPushConstants>(&command_buffer, push_constants);
        device.cmd_dispatch(
            *command_buffer,
            (batch_size * batch_size).div_ceil(64),
            1,
            1,
        );
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *pipelines.optimizer_step,
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
            *pipelines.optimize_latent_textures,
        );
        for texture in &network.textures {
            optimize_half(&device, &command_buffer, texture, i + 1, learning_rate);
        }

        device.end_command_buffer(*command_buffer).unwrap();
        device.submit_and_wait_on_command_buffer(&command_buffer);
    }

    let loss = loss_total.try_as_slice::<f32>().unwrap()[0];
    let loss = loss / batch_size as f32 / batch_size as f32 / 16.0;
    writer.add_scalar("loss", loss, i as usize);
    writer.add_scalar("psnr", l1_psnr(loss), i as usize);
    writer.flush();
    println!("{}, Loss: {:.8} PSNR: {:.4} dB,", i, loss, l1_psnr(loss),);
}

fn mlp_optimization_iter(
    device: &nbn::Device,
    i: u32,
    batch_size: u32,
    push_constants: CalculateGradsPushConstants,
    pipelines: &Pipelines,
    loss_total: &nbn::Buffer,
    network: &NetworkData,
    mlp_learning_rate: f32,
    writer: &mut SummaryWriter,
) {
    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();
        device.cmd_fill_buffer(*command_buffer, *loss_total.buffer, 0, vk::WHOLE_SIZE, 0);
        device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *pipelines.calculate_mlp_grads_only,
        );
        device.push_constants::<CalculateGradsPushConstants>(&command_buffer, push_constants);
        device.cmd_dispatch(
            *command_buffer,
            (batch_size * batch_size).div_ceil(64),
            1,
            1,
        );
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *pipelines.optimizer_step,
        );
        optimize(
            &device,
            &command_buffer,
            &network.weights_and_biases,
            i + 1,
            mlp_learning_rate,
        );
        device.end_command_buffer(*command_buffer).unwrap();
        device.submit_and_wait_on_command_buffer(&command_buffer);
    }

    let loss = loss_total.try_as_slice::<f32>().unwrap()[0];
    let loss = loss / batch_size as f32 / batch_size as f32 / 16.0;
    writer.add_scalar("loss", loss, i as usize);
    writer.add_scalar("psnr", l1_psnr(loss), i as usize);
    writer.flush();

    if i % 100 == 0 {
        println!("{}, Loss: {:.8} PSNR: {:.4} dB,", i, loss, l1_psnr(loss),);
    }
}

fn main() {
    let args = Arguments::parse();

    match args.mode {
        Mode::Eval { path } => {
            let event_loop = winit::event_loop::EventLoop::new().unwrap();
            event_loop.run_app(&mut App::new(path)).unwrap();
        }
        Mode::Train {
            paths,
            iterations,
            size,
            learning_rate,
            batch_size,
            mlp_learning_rate,
        } => {
            let device = nbn::Device::new(None);
            let shader = device.load_shader("shaders/compiled/ntc.spv");

            let mut staging_buffer =
                nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Graphics);

            let (images, image_indices, channel_bitmasks, image_size) =
                load_images(&device, &mut staging_buffer, &paths);
            let size = size.unwrap_or(image_size);
            dbg!(size);

            let pipelines = Pipelines {
                calculate_grads: device.create_compute_pipeline(&shader, c"calculate_grads"),
                optimizer_step: device.create_compute_pipeline(&shader, c"optimizer_step"),
                optimize_latent_textures: device
                    .create_compute_pipeline(&shader, c"optimize_latent_textures"),
                calculate_mlp_grads_only: device
                    .create_compute_pipeline(&shader, c"calculate_mlp_grads_only"),
            };

            let compress_blocks = device.create_compute_pipeline(&shader, c"compress_blocks");
            let copy_network_params =
                device.create_compute_pipeline(&shader, c"copy_network_params");

            let mut rng = rand::rng();
            let network = NetworkData::train(&device, &mut staging_buffer, size, &mut rng);

            let network_buffer =
                staging_buffer.create_buffer_from_slice(&device, "network", &[network.as_struct()]);

            let latent_textures: [_; 4] = std::array::from_fn(|i| {
                device.register_owned_image(
                    device.create_image(nbn::ImageDescriptor {
                        name: "latent texture",
                        format: vk::Format::BC1_RGB_UNORM_BLOCK,
                        extent: nbn::ImageExtent::D2 {
                            width: network.textures[i].size,
                            height: network.textures[i].size,
                        },
                        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        mip_levels: network.textures[i].num_mip_levels as _,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    }),
                    false,
                )
            });

            let latent_texture_indices: [u32; 4] = std::array::from_fn(|i| *latent_textures[i]);
            let latent_texture_indices = staging_buffer.create_buffer_from_slice(
                &device,
                "latent_texture_indices",
                &latent_texture_indices,
            );

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
                optimization_iter(
                    &device,
                    i,
                    batch_size,
                    CalculateGradsPushConstants {
                        network: *network_buffer,
                        textures: *image_indices,
                        num_textures: images.len() as _,
                        iteration: i as _,
                        batch_size,
                        channel_bitmasks: *channel_bitmasks,
                        total: *loss_total,
                        latent_textures: *latent_texture_indices,
                    },
                    &pipelines,
                    &loss_total,
                    &network,
                    mlp_learning_rate,
                    learning_rate,
                    &mut writer,
                );
            }

            let duration = std::time::Instant::now() - start;
            dbg!(duration);

            let block_buffers: [_; 4] = std::array::from_fn(|i| {
                device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "block_buffer",
                        size: network.textures[i].num_blocks as u64 * 8,
                        ty: nbn::MemoryLocation::GpuToCpu,
                    })
                    .unwrap()
            });

            let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

            unsafe {
                device
                    .begin_command_buffer(*command_buffer, &Default::default())
                    .unwrap();
                device
                    .bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);

                device.cmd_bind_pipeline(
                    *command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *compress_blocks,
                );

                for i in 0..4 {
                    device.push_constants::<CompressBlocksPushConstants>(
                        &command_buffer,
                        CompressBlocksPushConstants {
                            network: *network_buffer,
                            blocks: *block_buffers[i],
                            texture_index: i as _,
                        },
                    );
                    device.cmd_dispatch(
                        *command_buffer,
                        network.textures[i].num_blocks.div_ceil(64),
                        1,
                        1,
                    );

                    device.insert_image_pipeline_barrier(
                        &command_buffer,
                        &latent_textures[i].image,
                        None,
                        nbn::BarrierOp::TransferWrite,
                    );

                    let copies: Vec<_> = network.textures[i]
                        .block_offsets
                        .iter()
                        .enumerate()
                        .map(|(mip, &offset)| {
                            vk::BufferImageCopy::default()
                                .buffer_offset(offset as u64)
                                .image_extent(vk::Extent3D {
                                    width: network.textures[i].size >> mip,
                                    height: network.textures[i].size >> mip,
                                    depth: 1,
                                })
                                .image_subresource(
                                    latent_textures[i]
                                        .image
                                        .subresource_layer()
                                        .mip_level(mip as _)
                                        .layer_count(1),
                                )
                        })
                        .collect();

                    device.cmd_copy_buffer_to_image(
                        *command_buffer,
                        *block_buffers[i].buffer,
                        **latent_textures[i].image,
                        vk::ImageLayout::GENERAL,
                        &copies,
                    );
                }

                device.end_command_buffer(*command_buffer).unwrap();
                device.submit_and_wait_on_command_buffer(&command_buffer);
            }

            /*
            for i in iterations..iterations+100_000 {
                mlp_optimization_iter(
                    &device,
                    i,
                    batch_size,
                    CalculateGradsPushConstants {
                        network: *network_buffer,
                        textures: *image_indices,
                        num_textures: images.len() as _,
                        iteration: i as _,
                        batch_size,
                        channel_bitmasks: *channel_bitmasks,
                        total: *loss_total,
                        latent_textures: *latent_texture_indices,
                    },
                    &pipelines,
                    &loss_total,
                    &network,
                    mlp_learning_rate,
                    &mut writer,
                );
            }
            */

            let float_network_params = device
                .create_buffer(nbn::BufferDescriptor {
                    name: "float_network_params",
                    size: NUM_PARAMS as u64 * 4,
                    ty: nbn::MemoryLocation::GpuToCpu,
                })
                .unwrap();

            let half_network_params = device
                .create_buffer(nbn::BufferDescriptor {
                    name: "half_network_params",
                    size: NUM_PARAMS as u64 * 2,
                    ty: nbn::MemoryLocation::GpuToCpu,
                })
                .unwrap();

            let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

            unsafe {
                device
                    .begin_command_buffer(*command_buffer, &Default::default())
                    .unwrap();
                device
                    .bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);

                device.cmd_bind_pipeline(
                    *command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *copy_network_params,
                );
                device.push_constants::<CopyNetworkParamsPushConstants>(
                    &command_buffer,
                    CopyNetworkParamsPushConstants {
                        network: *network_buffer,
                        output: *half_network_params,
                        write_halfs: 1,
                    },
                );
                device.cmd_dispatch(*command_buffer, (NUM_PARAMS as u32).div_ceil(64), 1, 1);
                device.push_constants::<CopyNetworkParamsPushConstants>(
                    &command_buffer,
                    CopyNetworkParamsPushConstants {
                        network: *network_buffer,
                        output: *float_network_params,
                        write_halfs: 0,
                    },
                );
                device.cmd_dispatch(*command_buffer, (NUM_PARAMS as u32).div_ceil(64), 1, 1);

                device.end_command_buffer(*command_buffer).unwrap();
                device.submit_and_wait_on_command_buffer(&command_buffer);
            }

            let textures = std::array::from_fn(|i| AntcTexture {
                size: network.textures[i].size,
                block_offsets: &network.textures[i].block_offsets,
                bytes: block_buffers[i].try_as_slice::<u8>().unwrap(),
            });

            let float_writer = AntcWriter {
                weights: float_network_params.try_as_slice::<u8>().unwrap(),
                textures: &textures,
            };
            let mut float_output = Vec::new();
            float_writer.write(&mut float_output).unwrap();

            let half_writer = AntcWriter {
                weights: half_network_params.try_as_slice::<u8>().unwrap(),
                textures: &textures,
            };
            half_writer
                .write(&mut std::fs::File::create("texture.antc").unwrap())
                .unwrap();
            let mut half_output = Vec::new();
            half_writer.write(&mut half_output).unwrap();

            eval_textures(
                &device,
                nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Compute),
                &image_indices,
                &channel_bitmasks,
                size,
                images.len() as _,
                &float_output,
                &shader,
                false,
            );

            eval_textures(
                &device,
                nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Compute),
                &image_indices,
                &channel_bitmasks,
                size,
                images.len() as _,
                &half_output,
                &shader,
                true,
            );
        }
    };
}

fn load_images(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    paths: &ImagePaths,
) -> (Vec<nbn::IndexedImage>, nbn::Buffer, nbn::Buffer, u32) {
    assert_eq!(paths.paths.len(), paths.channels.len());

    let mut size = None;
    let (images, indices) = paths
        .paths
        .iter()
        .zip(&paths.channels)
        .map(|(filepath, channels)| {
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

            let image = device.register_owned_image(
                staging_buffer.create_sampled_image(
                    &device,
                    nbn::SampledImageDescriptor {
                        name: &filepath.display().to_string(),
                        format: if channels.is_srgb() {
                            vk::Format::R8G8B8A8_SRGB
                        } else {
                            vk::Format::R8G8B8A8_UNORM
                        },
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
            (image, index)
        })
        .collect::<(Vec<_>, Vec<u32>)>();

    let channel_bitmasks: Vec<u8> = paths
        .channels
        .iter()
        .map(|channels| channels.as_bitmask())
        .collect();

    let size = size.unwrap();

    let image_indices = staging_buffer.create_buffer_from_slice(&device, "image_indices", &indices);

    let channel_bitmasks =
        staging_buffer.create_buffer_from_slice(&device, "channel_bitmasks", &channel_bitmasks);

    (images, image_indices, channel_bitmasks, size)
}

fn l1_psnr(loss: f32) -> f32 {
    20.0 * (1.0 / loss).log10()
}

fn l2_psnr(loss: f32) -> f32 {
    10.0 * (1.0 / loss).log10()
}

fn load_ntc_texture(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    bytes: &[u8],
    use_halfs: bool,
) -> ([nbn::IndexedImage; 4], nbn::Buffer, u32) {
    let values: &[u32] = nbn::cast_slice(&bytes[4..]);

    let version = values[0];
    assert_eq!(version, 0);

    let weight_offset = values[1] as usize;

    let params = staging_buffer.create_buffer_from_slice(
        device,
        "params",
        &bytes[weight_offset..weight_offset + NUM_PARAMS * if use_halfs { 2 } else { 4 }],
    );

    let texture_info: &[[u32; 2]] = nbn::cast_slice(&values[2..2 + 4 * 2]);

    let mut offset = (2 + 4 * 2) as usize;

    let latent_textures: [_; 4] = std::array::from_fn(|i| {
        let num_mips = texture_info[i][1] as usize;
        let first_offset = values[offset];

        let img = device.register_owned_image(
            staging_buffer.create_sampled_image(
                device,
                nbn::SampledImageDescriptor {
                    name: "tex",
                    format: vk::Format::BC1_RGB_UNORM_BLOCK,
                    extent: nbn::ImageExtent::D2 {
                        width: texture_info[i][0],
                        height: texture_info[i][0],
                    },
                },
                &bytes[first_offset as usize..],
                nbn::QueueType::Compute,
                nbn::ImageLods::Offsets(
                    &values[offset..offset + num_mips]
                        .iter()
                        .map(|&offset| offset as u64 - first_offset as u64)
                        .collect::<Vec<_>>(),
                ),
            ),
            false,
        );

        offset += num_mips;
        img
    });

    (latent_textures, params, texture_info[0][0])
}

fn eval_textures(
    device: &nbn::Device,
    mut staging_buffer: nbn::StagingBuffer,
    image_indices: &nbn::Buffer,
    channel_bitmasks: &nbn::Buffer,
    size: u32,
    num_textures: i32,
    bytes: &[u8],
    shader: &nbn::ShaderModule,
    use_halfs: bool,
) {
    let sum_textures_loss = device.create_compute_pipeline(shader, c"sum_textures_loss");
    let render = device.create_compute_pipeline(&shader, c"render_compute");

    let (latent_textures, params, _) =
        load_ntc_texture(device, &mut staging_buffer, bytes, use_halfs);
    let latent_texture_indices: [u32; 4] = std::array::from_fn(|i| *latent_textures[i]);
    let latent_texture_indices = staging_buffer.create_buffer_from_slice(
        &device,
        "latent_texture_indices",
        &latent_texture_indices,
    );

    staging_buffer.finish(&device);

    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    let loss_total = device
        .create_buffer(nbn::BufferDescriptor {
            name: "loss_total",
            size: 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();
        device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);

        device.cmd_fill_buffer(*command_buffer, *loss_total.buffer, 0, vk::WHOLE_SIZE, 0);
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *sum_textures_loss,
        );
        device.push_constants::<SumTexturesLossPushConstants>(
            &command_buffer,
            SumTexturesLossPushConstants {
                params: *params,
                textures: **image_indices,
                resolution: size as _,
                num_textures,
                channel_bitmasks: **channel_bitmasks,
                total: *loss_total,
                latent_textures: *latent_texture_indices,
                use_halfs: use_halfs as u32,
            },
        );
        device.cmd_dispatch(*command_buffer, (size * size).div_ceil(64), 1, 1);

        device.end_command_buffer(*command_buffer).unwrap();
        device.submit_and_wait_on_command_buffer(&command_buffer);
    }

    let loss = loss_total.try_as_slice::<f32>().unwrap()[0];
    let loss = loss / size as f32 / size as f32 / 16.0;
    println!("Loss: {:.8} PSNR: {:.4} dB", loss, l2_psnr(loss),);

    if !use_halfs {
        return;
    }

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
                    params: *params,
                    latent_textures: *latent_texture_indices,
                    resolution: size as _,
                    channel_offset: i * 3,
                },
            );
            device.cmd_dispatch(*command_buffer, (size * size).div_ceil(64), 1, 1);
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
