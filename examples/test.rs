use ash::vk;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

slang_struct::slang_include!("gltf.slang");

#[derive(Clone, Copy, Default, PartialEq, Debug, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
struct PackedMaterial {
    base_colour: [u8; 3],
    ty_and_aux_value: u8,
}

impl PackedMaterial {
    const INVALID: Self = Self {
        base_colour: [0; 3],
        ty_and_aux_value: 1,
    };
}

fn create_image(
    device: &nbn::Device,
    filename: &str,
    format: vk::Format,
    transition_to: nbn::QueueType,
) -> nbn::PendingImageUpload {
    let image_data = image::open(filename).unwrap().to_rgba8();

    device.create_image_with_data(
        nbn::ImageDescriptor {
            name: filename,
            extent: vk::Extent3D {
                width: image_data.width(),
                height: image_data.height(),
                depth: 1,
            },
            format,
        },
        &image_data,
        transition_to,
    )
}

fn main() {
    env_logger::init();

    let base = "Models/Corset/glTF/";

    let bytes = std::fs::read(&format!("{}/Corset.gltf", base)).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    assert_eq!(gltf.nodes.len(), 1);
    let node = &gltf.nodes[0];
    let mesh = &gltf.meshes[node.mesh.unwrap()];
    assert_eq!(mesh.primitives.len(), 1);
    let primitive = &mesh.primitives[0];
    //dbg!(primitive);
    //dbg!(&gltf.accessors);

    let mut image_formats = vec![vk::Format::R8G8B8A8_UNORM; gltf.images.len()];

    for material in &gltf.materials {
        if let Some(tex) = &material.emissive_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
        if let Some(tex) = &material.pbr_metallic_roughness.base_color_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
    }

    let instance = nbn::Instance::new(None);
    let device = instance.create_device();

    let images = gltf
        .images
        .par_iter()
        .zip(&image_formats)
        .map(|(image, format)| {
            let path = format!("{}/{}", base, image.uri.as_ref().unwrap());
            create_image(&device, &path, *format, nbn::QueueType::Compute)
        })
        .collect::<Vec<_>>();

    let buffer = std::fs::read(&format!(
        "{}/{}",
        base,
        gltf.buffers[0].uri.as_ref().unwrap()
    ))
    .unwrap();

    let primitives: Vec<Primitive> = mesh
        .primitives
        .iter()
        .map(|primitive| Primitive {
            indices: primitive.indices.unwrap() as u32,
            positions: primitive.attributes.position.unwrap() as u32,
            uvs: primitive.attributes.texcoord_0.unwrap() as u32,
            material: primitive.material.unwrap() as u32,
        })
        .collect();

    let materials: Vec<Material> = gltf
        .materials
        .iter()
        .map(|material| Material {
            base_colour_image: *images[gltf.textures[material
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .unwrap()
                .index]
                .source
                .unwrap()]
            .image,
            metallic_roughness_image: *images[gltf.textures[material
                .pbr_metallic_roughness
                .metallic_roughness_texture
                .as_ref()
                .unwrap()
                .index]
                .source
                .unwrap()]
            .image,
            emissive_image: material
                .emissive_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
        })
        .collect();

    let buffer_views: Vec<BufferView> = gltf
        .buffer_views
        .iter()
        .map(|buffer_view| BufferView {
            byte_offset: buffer_view.byte_offset as u32,
        })
        .collect();

    let accessors: Vec<Accessor> = gltf
        .accessors
        .iter()
        .map(|accessor| Accessor {
            buffer_view: accessor.buffer_view.unwrap() as u32,
            count: accessor.count as _,
        })
        .collect();

    let num_indices = accessors[primitive.indices.unwrap()].count;

    let materials = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "materials",
        data: &materials,
    });

    let buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "buffer",
        data: &buffer,
    });
    let primitives = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "primitives",
        data: &primitives,
    });
    let buffer_views = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "buffer_views",
        data: &buffer_views,
    });
    let accessors = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "accessors",
        data: &accessors,
    });
    let gltf = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "gltf",
        data: &[Gltf {
            buffer_views: *buffer_views,
            accessors: *accessors,
            buffer: *buffer,
            primitives: *primitives,
            materials: *materials,
        }],
    });

    let compute_pipeline = device.create_compute_pipeline(nbn::ComputePipelineDesc {
        name: c"main",
        path: "shader.spv",
    });

    let scene_size = 1024;

    let output = device.create_buffer(nbn::BufferDescriptor {
        name: "output_buffer",
        size: scene_size * scene_size * scene_size * std::mem::size_of::<PackedMaterial>() as u64,
        ty: nbn::BufferType::Download,
    });

    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    unsafe {
        let fence = device.create_fence();

        device
            .begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
            .unwrap();

        device.cmd_fill_buffer(
            *command_buffer,
            *output.buffer,
            0,
            vk::WHOLE_SIZE,
            std::mem::transmute(PackedMaterial::INVALID),
        );

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *compute_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *compute_pipeline.layout,
            0,
            &[device.descriptors.set],
            &[],
        );

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Input {
            output: u64,
            gltf: u64,
            scene_size: u32,
        }

        device.cmd_push_constants(
            *command_buffer,
            *compute_pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            nbn::cast_slice(&[Input {
                output: *output,
                gltf: *gltf,
                scene_size: scene_size as u32,
            }]),
        );

        device.cmd_dispatch(*command_buffer, (num_indices / 3).div_ceil(64), 1, 1);

        device.end_command_buffer(*command_buffer).unwrap();

        let transfer_command_buffers: Vec<vk::CommandBuffer> =
            images.iter().map(|image| *image.command_buffer).collect();

        let semaphore = device.create_semaphore();

        device
            .device
            .queue_submit(
                *device.transfer_queue,
                &[vk::SubmitInfo::default()
                    .signal_semaphores(&[*semaphore])
                    .command_buffers(&transfer_command_buffers)],
                vk::Fence::null(),
            )
            .unwrap();

        device
            .device
            .queue_submit(
                *device.compute_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[*semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                    .command_buffers(&[*command_buffer])],
                *fence,
            )
            .unwrap();

        device.device.wait_for_fences(&[*fence], true, !0).unwrap();
    }

    let tree = tree64::Tree64::new(tree64::FlatArray {
        values: output.try_as_slice::<PackedMaterial>().unwrap(),
        dimensions: [scene_size as u32; 3],
        empty_value: PackedMaterial::INVALID,
    });
    dbg!(
        &tree.stats,
        tree.nodes.len() * std::mem::size_of::<tree64::Node>(),
        tree.data.len() * std::mem::size_of::<PackedMaterial>()
    );
    tree.serialize(std::fs::File::create("out.tree64").unwrap())
        .unwrap();
}
