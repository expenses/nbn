slang_struct::slang_include!("shaders/ntc/structs.slang");

use nbn::vk;
use ndarray_npy as npy;

use ndarray::{Array0, Array1};

fn upload_array(device: &nbn::Device, staging_buffer: &mut nbn::StagingBuffer, reader: &mut npy::NpzReader<std::fs::File>, name: &str) -> nbn::Buffer {
    let array: Array1<f32> = reader.by_name(name).unwrap();
    let slice = array.as_slice().unwrap();

    staging_buffer.create_buffer_from_slice(device, name, slice)
}

fn main() {
    let file = std::fs::File::open(&std::env::args().nth(1).unwrap()).unwrap();
    let mut npz = npy::NpzReader::new(file).unwrap();

    let device = nbn::Device::new(None);
    let mut staging_buffer = nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Transfer);

    println!("Keys: {:?}", npz.names());

    let size: Array0<i64> = npz.by_name("size").unwrap();
    let size = size.into_scalar() as i32;
    dbg!(&size);

    let weights_and_biases = upload_array(&device, &mut staging_buffer, &mut npz, "weights_and_biases");
    let lt1_alpha = upload_array(&device, &mut staging_buffer, &mut npz, "lt1_alpha");
    let lt2_alpha = upload_array(&device, &mut staging_buffer, &mut npz, "lt2_alpha");
    let lt3_alpha = upload_array(&device, &mut staging_buffer, &mut npz, "lt3_alpha");
    let lt4_alpha = upload_array(&device, &mut staging_buffer, &mut npz, "lt4_alpha");
    let lt1_endpoints = upload_array(&device, &mut staging_buffer, &mut npz, "lt1_endpoints");
    let lt2_endpoints = upload_array(&device, &mut staging_buffer, &mut npz, "lt2_endpoints");
    let lt3_endpoints = upload_array(&device, &mut staging_buffer, &mut npz, "lt3_endpoints");
    let lt4_endpoints = upload_array(&device, &mut staging_buffer, &mut npz, "lt4_endpoints");

    let network = Network {
        weights_and_biases: *weights_and_biases,
        weights_and_biases_grad: 0,
        latent_texture_1: LatentTexture {
            endpoints: Float3Tensor {
                ptr: *lt1_endpoints,
                grads: 0,
            },
            alpha: FloatTensor {
                ptr: *lt1_alpha,
                grads: 0,
            },
            size,
            num_mip_levels: 1,
        },
        latent_texture_2: LatentTexture {
            endpoints: Float3Tensor {
                ptr: *lt2_endpoints,
                grads: 0,
            },
            alpha: FloatTensor {
                ptr: *lt2_alpha,
                grads: 0,
            },
            size,
            num_mip_levels: 1,
        },
        latent_texture_3: LatentTexture {
            endpoints: Float3Tensor {
                ptr: *lt3_endpoints,
                grads: 0,
            },
            alpha: FloatTensor {
                ptr: *lt3_alpha,
                grads: 0,
            },
            size: size/2,
            num_mip_levels: 1,
        },
        latent_texture_4: LatentTexture {
            endpoints: Float3Tensor {
                ptr: *lt4_endpoints,
                grads: 0,
            },
            alpha: FloatTensor {
                ptr: *lt4_alpha,
                grads: 0,
            },
            size:size/2,
            num_mip_levels: 1,
        },
    };

    let network_buffer = staging_buffer.create_buffer_from_slice(&device, "network", &[network]);

    staging_buffer.finish(&device);

    let output = device.create_buffer(nbn::BufferDescriptor {
        size: size as u64 * size as u64 * 3 * 4,
        name: "output floats",
       ty: nbn::MemoryLocation::GpuToCpu
    }).unwrap();

    let pc = RenderComputePushConstants {
        output: *output,
        network: *network_buffer,
        resolution: size
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
        device.cmd_dispatch(
            *command_buffer,
            (size as u32).div_ceil(8),
            (size as u32).div_ceil(8),
            1,
        );
        device.end_command_buffer(*command_buffer).unwrap();
        device.submit_and_wait_on_command_buffer(&command_buffer);
    }

    let slice = output.try_as_slice::<f32>().unwrap();

    image::DynamicImage::ImageRgb32F(image::ImageBuffer::<image::Rgb<f32>, Vec<f32>>::from_raw(size as _, size as _, slice.to_vec())
        .unwrap()).to_rgb8()
        .save("out.png")
        .unwrap();
}
