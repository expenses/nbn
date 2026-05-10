slang_struct::slang_include!("shaders/ntc/structs.slang");

use nbn::vk;
use ndarray_npy as npy;
use rand::{Rng, RngExt};

use ndarray::{Array0, Array1};

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
    size: i32,
    num_mip_levels: i32,
}

impl TextureData {
    fn train<R: Rng>(device: &nbn::Device, staging_buffer: &mut nbn::StagingBuffer, size: i32, rng: &mut R) -> Self {
        let mut size_in_blocks = size / 4;
        let mut num_blocks = size_in_blocks * size_in_blocks;

        let mut num_mip_levels = 1;
        while size_in_blocks > 1 {
            num_mip_levels += 1;
            size_in_blocks >>= 1;
            num_blocks += size_in_blocks * size_in_blocks
        }

        Self {
            endpoints: Tensor::from_data(
                device,
                staging_buffer,
                &(0..num_blocks * 3 * 2)
                    .map(|_| rng.random_range(0.0..1.0))
                    .collect::<Vec<_>>(),
            ),
            alpha: Tensor::from_data(
                device,
                staging_buffer,
                &(0..num_blocks * 16)
                    .map(|_| rng.random_range(0.0..1.0))
                    .collect::<Vec<_>>(),
            ),
            size,num_mip_levels
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
            size: self.size,
            num_mip_levels: self.num_mip_levels,
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
}

impl Tensor {
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
        }
    }
}

struct NetworkData {
    weights_and_biases: Tensor,
    latent_texture_1: TextureData,
    latent_texture_2: TextureData,
    latent_texture_3: TextureData,
    latent_texture_4: TextureData,
    size: i32,
}

impl NetworkData {
    fn from_npz(
        device: &nbn::Device,
        staging_buffer: &mut nbn::StagingBuffer,
        npz: &mut npy::NpzReader<std::fs::File>,
    ) -> Self {
        let size: Array0<i64> = npz.by_name("size").unwrap();
        let size = size.into_scalar() as i32;

        Self {
            weights_and_biases: Tensor {
                data: upload_array(device, staging_buffer, npz, "weights_and_biases"),
                training: None,
            },
            latent_texture_1: TextureData {
                endpoints: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt1_endpoints"),
                    training: None,
                },
                alpha: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt1_alpha"),
                    training: None,
                },
                size,
                num_mip_levels: 1,
            },
            latent_texture_2: TextureData {
                endpoints: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt2_endpoints"),
                    training: None,
                },
                alpha: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt2_alpha"),
                    training: None,
                },
                size,
                num_mip_levels: 1,
            },
            latent_texture_3: TextureData {
                endpoints: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt3_endpoints"),
                    training: None,
                },

                alpha: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt3_alpha"),
                    training: None,
                },
                size: size /2,
                num_mip_levels: 1,
            },
            latent_texture_4: TextureData {
                endpoints: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt4_endpoints"),
                    training: None,
                },
                alpha: Tensor {
                    data: upload_array(device, staging_buffer, npz, "lt4_alpha"),
                    training: None,
                },
                size: size /2,
                num_mip_levels: 1,
            },
            size: size as _,
        }
    }

    fn train<R: Rng>(device: &nbn::Device, staging_buffer: &mut nbn::StagingBuffer, size: i32, rng: &mut R) -> Self {
        Self {
            weights_and_biases: Tensor::from_data(device, staging_buffer, &network_data(rng)),
            latent_texture_1: TextureData::train(device, staging_buffer, size, rng),
            latent_texture_2: TextureData::train(device, staging_buffer, size, rng),
            latent_texture_3: TextureData::train(device, staging_buffer, size/2, rng),
            latent_texture_4: TextureData::train(device, staging_buffer, size/2, rng),
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

fn main() {
    let mode = std::env::args().nth(1).unwrap();

    let device = nbn::Device::new(None);

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Transfer);

    let network = if mode == "eval" {
        let file = std::fs::File::open(&std::env::args().nth(2).unwrap()).unwrap();
        let mut npz = npy::NpzReader::new(file).unwrap();

        println!("Keys: {:?}", npz.names());

        NetworkData::from_npz(&device, &mut staging_buffer, &mut npz)
    } else {
        let mut rng = rand::rng();
        NetworkData::train(&device, &mut staging_buffer, 256, &mut rng)
    };

    let size = network.size;

    let network_buffer =
        staging_buffer.create_buffer_from_slice(&device, "network", &[network.as_struct()]);

    staging_buffer.finish(&device);

    let output = device
        .create_buffer(nbn::BufferDescriptor {
            size: size as u64 * size as u64 * 3 * 4,
            name: "output floats",
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let pc = RenderComputePushConstants {
        output: *output,
        network: *network_buffer,
        resolution: size,
    };

    let shader = device.load_shader("shaders/compiled/ntc.spv");

    let pipeline = device.create_compute_pipeline(&shader, c"main");

    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();

        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, *pipeline);
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
