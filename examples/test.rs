use ash::vk;

#[repr(C)]
#[derive(Clone, Copy)]
struct BufferView {
    byte_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Accessor {
    buffer_view: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Primitive {
    indices: u32,
    positions: u32,
    uvs: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Gltf {
    buffer_views: u64,
    accessors: u64,
    primitives: u64,
    buffer: u64,
}

fn to_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

fn main() {
    env_logger::init();

    let bytes = std::fs::read("Models/DamagedHelmet/glTF/DamagedHelmet.gltf").unwrap();
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
    let buffer = std::fs::read("Models/DamagedHelmet/glTF/DamagedHelmet.bin").unwrap();

    let image_data = image::open("Models/DamagedHelmet/glTF/Default_AO.jpg")
        .unwrap()
        .to_rgba8();

    let primitives: Vec<Primitive> = mesh
        .primitives
        .iter()
        .map(|primitive| Primitive {
            indices: primitive.indices.unwrap() as u32,
            positions: primitive.attributes.position.unwrap() as u32,
            uvs: primitive.attributes.texcoord_0.unwrap() as u32,
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

    let instance = nbn::Instance::new(None);
    let device = instance.create_device();

    let image = device.create_image_with_data(
        nbn::ImageDescriptor {
            name: "wow",
            extent: vk::Extent3D {
                width: image_data.width(),
                height: image_data.height(),
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
        },
        &image_data,
    );

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
        }],
    });

    let compute_pipeline = device.create_compute_pipeline(nbn::ComputePipelineDesc {
        name: c"main",
        path: "shader.spv",
    });

    let scene_size = 512;

    let output = device.create_buffer(nbn::BufferDescriptor {
        name: "output_buffer",
        size: scene_size * scene_size * scene_size,
        ty: nbn::BufferType::Download,
    });

    let command_buffer = device.create_command_buffer();

    unsafe {
        let fence = device.create_fence();

        device
            .begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
            .unwrap();

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
            image: u32,
            //image: glam::UVec2,
            
        }

        device.cmd_push_constants(
            *command_buffer,
            *compute_pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            to_bytes(&[Input {
                output: *output,
                gltf: *gltf,
                scene_size: scene_size as u32,
                image: *image.image
                //image: glam::UVec2::new(*image.image, 0),
            }]),
        );

        device.cmd_dispatch(*command_buffer, 14556_u32.div_ceil(64), 1, 1);

        device.end_command_buffer(*command_buffer).unwrap();

        device
            .device
            .queue_submit(
                device.queue,
                &[vk::SubmitInfo::default()
                    .command_buffers(&[*image.command_buffer, *command_buffer])],
                *fence,
            )
            .unwrap();

        device.device.wait_for_fences(&[*fence], true, !0).unwrap();
    }

    let slice = output.allocation.mapped_slice().unwrap();

    let tree = tree64::Tree64::new((slice, [scene_size as u32; 3]));
    dbg!(slice.iter().filter(|&&x| x != 0).count());
    tree.serialize(std::fs::File::create("out.tree64").unwrap())
        .unwrap();
}
