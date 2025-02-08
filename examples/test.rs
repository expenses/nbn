use ash::vk;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

slang_struct::slang_include!("gltf.slang");

#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    Debug,
    Hash,
    PartialOrd,
    Ord,
    Eq,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
#[repr(C, packed)]
struct PackedMaterial {
    base_colour: [u8; 3],
    ty_and_aux_value: u8,
}

fn create_image(
    device: &nbn::Device,
    filename: &str,
    format: vk::Format,
    transition_to: nbn::QueueType,
) -> nbn::PendingImageUpload {
    if filename.ends_with(".dds") {
        let dds = ddsfile::Dds::read(std::fs::File::open(filename).unwrap()).unwrap();

        let format = match dds.get_dxgi_format().unwrap() {
            ddsfile::DxgiFormat::BC1_UNorm_sRGB => vk::Format::BC1_RGB_SRGB_BLOCK,
            ddsfile::DxgiFormat::BC3_UNorm_sRGB => vk::Format::BC3_SRGB_BLOCK,
            ddsfile::DxgiFormat::BC5_UNorm => vk::Format::BC5_UNORM_BLOCK,
            other => panic!("{:?}", other),
        };
        device.create_image_with_data(
            nbn::ImageDescriptor {
                name: filename,
                extent: vk::Extent3D {
                    width: dds.get_width(),
                    height: dds.get_height(),
                    depth: 1,
                },
                format,
            },
            &dds.get_data(0).unwrap(),
            transition_to,
        )
    } else {
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
}

fn main() {
    env_logger::init();

    let base = "bi/Bistro_v5_2";

    let bytes = std::fs::read(&format!("{}/bistro_combined.gltf", base)).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let mut image_formats = vec![vk::Format::R8G8B8A8_UNORM; gltf.images.len()];

    for material in &gltf.materials {
        if let Some(tex) = &material.emissive_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
        if let Some(tex) = &material.pbr_metallic_roughness.base_color_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
    }

    let device = nbn::Device::new(None);

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

    let primitives_iter = || gltf.meshes.iter().flat_map(|mesh| &mesh.primitives);

    let primitives: Vec<Primitive> = primitives_iter()
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
            base_colour_image: material
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            metallic_roughness_image: material
                .pbr_metallic_roughness
                .metallic_roughness_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            emissive_image: material
                .emissive_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            flags: matches!(material.alpha_mode, goth_gltf::AlphaMode::Mask) as u32,
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
            byte_offset: accessor.byte_offset as u32,
            flags: (accessor.component_type == goth_gltf::ComponentType::UnsignedInt) as u32,
        })
        .collect();

    //panic!();

    let num_indices: u32 = primitives_iter()
        .map(|primitive| accessors[primitive.indices.unwrap()].count)
        .sum();

    dbg!(num_indices);

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
            num_primitives: primitives_iter().count() as u32,
        }],
    });

    let dim_size = 10000;

    let voxelization_pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        vertex: nbn::ShaderDesc {
            path: "voxelize.spv",
            entry_point: c"main",
        },
        fragment: nbn::ShaderDesc {
            path: "voxelize.spv",
            entry_point: c"main",
        },
        color_attachment_formats: &[],
        conservative_rasterization: true,
    });

    let output = device
        .create_buffer(nbn::BufferDescriptor {
            name: "output_buffer",
            size: 3_000_000_000,
            ty: nbn::BufferType::Download,
        })
        .unwrap();

    let num_outputs = device
        .create_buffer(nbn::BufferDescriptor {
            name: "num_outputs",
            size: 4,
            ty: nbn::BufferType::Download,
        })
        .unwrap();
    let command_buffer = device.create_command_buffer(nbn::QueueType::Graphics);

    unsafe {
        let fence = device.create_fence();

        device
            .begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
            .unwrap();

        device.cmd_fill_buffer(*command_buffer, *num_outputs.buffer, 0, vk::WHOLE_SIZE, 0);

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *voxelization_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            **device.pipeline_layout,
            0,
            &[device.descriptors.set],
            &[],
        );

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Input {
            gltf: u64,
            output: u64,
            num_outputs: u64,
            dim_size: u32,
        }

        device.cmd_push_constants(
            *command_buffer,
            **device.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT,
            0,
            nbn::cast_slice(&[Input {
                output: *output,
                gltf: *gltf,
                num_outputs: *num_outputs,
                dim_size,
            }]),
        );

        let render_area =
            vk::Rect2D::default().extent(vk::Extent2D::default().height(dim_size).width(dim_size));

        device.cmd_begin_rendering(
            *command_buffer,
            &vk::RenderingInfo::default()
                .layer_count(1)
                .render_area(render_area),
        );

        device.cmd_set_viewport(
            *command_buffer,
            0,
            &[vk::Viewport::default()
                .height(dim_size as f32)
                .width(dim_size as f32)
                .max_depth(1.0)],
        );
        device.cmd_set_scissor(*command_buffer, 0, &[render_area]);
        device.cmd_draw(*command_buffer, num_indices, 1, 0, 0);

        device.cmd_end_rendering(*command_buffer);

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
                *device.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[*semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                    .command_buffers(&[*command_buffer])],
                *fence,
            )
            .unwrap();

        dbg!("Started waiting for fence");
        device.device.wait_for_fences(&[*fence], true, !0).unwrap();
    }
    dbg!("Finished waiting for fence");
    let size = num_outputs.try_as_slice::<u32>().unwrap()[0] as usize;
    let output = &output
        .try_as_slice::<(glam::U16Vec3, PackedMaterial)>()
        .unwrap()[..size];

    let mut output = output.to_vec();

    fn interleave_u16_3d(pos: glam::U16Vec3) -> (u16, u16, u16) {
        let value = separate_bits_64(pos.x)
            | (separate_bits_64(pos.y) << 1)
            | (separate_bits_64(pos.z) << 2);
        ((value >> 32) as u16, (value >> 16) as u16, value as u16)
    }

    dbg!("Starting sort");
    radsort::sort_by_key(&mut output, |&(pos, _)| interleave_u16_3d(pos));
    dbg!(output.len());
    output.dedup_by_key(|(pos, _)| *pos);
    dbg!(output.len());

    //let pos: Vec<glam::U16Vec3> = output.iter().map(|item| item.0).collect();
    //std::fs::write("out.dat", nbn::cast_slice(&pos)).unwrap();

    let mut tree = tree64::Tree64 {
        nodes: Default::default(),
        data: Default::default(),
        stats: Default::default(),
        edits: Default::default(),
    };

    let mut nodes_a = Vec::new();
    let mut nodes_b = Vec::new();

    dbg!("Starting first");
    collect(&output, &mut nodes_a, |arr, pop_mask| {
        tree64::Node::new(true, tree.insert_values(arr), pop_mask)
    });
    dbg!("Done first");

    collect(&nodes_a, &mut nodes_b, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });

    dbg!(nodes_a.len(), nodes_b.len());

    collect(&nodes_b, &mut nodes_a, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });

    dbg!(nodes_a.len());

    collect(&nodes_a, &mut nodes_b, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });
    dbg!(nodes_b.len());

    collect(&nodes_b, &mut nodes_a, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });

    dbg!(nodes_a.len());

    collect(&nodes_a, &mut nodes_b, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });
    dbg!(nodes_b.len());

    collect(&nodes_b, &mut nodes_a, |arr, pop_mask| {
        tree64::Node::new(false, tree.insert_nodes(arr), pop_mask)
    });
    dbg!(nodes_a.len());
    dbg!(&tree.stats);
    tree.push_new_root_node(nodes_a[0].1, 10, -glam::IVec3::splat(dim_size as i32));

    dbg!(
        std::mem::size_of_val(&tree.nodes[..]),
        std::mem::size_of_val(&tree.data[..])
    );
    tree.serialize(std::fs::File::create("out.tree64").unwrap())
        .unwrap();
}

fn collect<
    T: std::hash::Hash + Clone + Copy + PartialEq + std::fmt::Debug + Default,
    O,
    F: FnMut(&[T], u64) -> O,
>(
    sorted_values: &[(glam::U16Vec3, T)],
    output: &mut Vec<(glam::U16Vec3, O)>,
    mut transform: F,
) {
    output.clear();
    let mut values = tree64::PopMaskedData::default();

    let mut prev_pos = glam::U16Vec3::splat(u16::MAX);

    for (i, &(pos, val)) in sorted_values.iter().enumerate() {
        let key_pos = pos / 4;
        if !(i == 0 || key_pos == prev_pos) {
            output.push((prev_pos, transform(&values.as_compact(), values.pop_mask)));
            values.pop_mask = 0;
        }

        prev_pos = key_pos;

        let index = (pos % 4).dot(glam::U16Vec3::new(1, 4, 16));
        values.set(index as u32, Some(val));
    }

    output.push((prev_pos, transform(&values.as_compact(), values.pop_mask)));
}

fn separate_bits_64(n: u16) -> u64 {
    let mut n = n as u64;
    n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111u64;
    n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111u64;
    n = (n ^ (n << 8)) & 0b1111000000001111000000001111000000001111000000001111000000001111u64;
    n = (n ^ (n << 4)) & 0b0011000011000011000011000011000011000011000011000011000011000011u64;
    n = (n ^ (n << 2)) & 0b1001001001001001001001001001001001001001001001001001001001001001u64;
    n
}
