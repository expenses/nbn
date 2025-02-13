use ash::vk;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

slang_struct::slang_include!("shaders/gltf.slang");

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

    let base = "models/citadel";

    let bytes = std::fs::read(&format!("{}/voxelization.gltf", base)).unwrap();
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

    let dim_size = 10384;

    let voxelization_shader = device.load_shader("shaders/compiled/voxelize.spv");

    let voxelization_pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        vertex: nbn::ShaderDesc {
            module: &voxelization_shader,
            entry_point: c"vertex",
        },
        fragment: nbn::ShaderDesc {
            module: &voxelization_shader,
            entry_point: c"fragment",
        },
        color_attachment_formats: &[],
        conservative_rasterization: true,
    });

    let output_buffer_size = 4_000_000_000;

    let output_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "outputs",
            size: output_buffer_size,
            ty: nbn::BufferType::Download,
        })
        .unwrap();

    let max_outputs = output_buffer_size / std::mem::size_of::<(u32, u32, u32)>() as u64;

    let num_outputs_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: "num_outputs",
        data: &[0_u32],
    });

    let command_buffer = device.create_command_buffer(nbn::QueueType::Graphics);

    let transfer_semaphore = device.create_semaphore();
    let fence = device.create_fence();

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Voxelization {
        gltf: u64,
        output: u64,
        num_outputs: u64,
        dim_size: u32,
        output_size: u32,
    }

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
            .unwrap();

        device.begin_rendering(&command_buffer, dim_size, dim_size, &[]);

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *voxelization_pipeline,
        );
        device.bind_internal_descriptor_sets(&command_buffer);

        device.push_constants(
            &command_buffer,
            Voxelization {
                output: *output_buffer,
                gltf: *gltf,
                num_outputs: *num_outputs_buffer,
                dim_size,
                output_size: max_outputs as u32,
            },
        );

        device.cmd_draw(*command_buffer, num_indices, 1, 0, 0);

        device.cmd_end_rendering(*command_buffer);

        device.end_command_buffer(*command_buffer).unwrap();
    }

    let transfer_command_buffers: Vec<vk::CommandBuffer> =
        images.iter().map(|image| *image.command_buffer).collect();

    unsafe {
        device
            .queue_submit(
                *device.transfer_queue,
                &[vk::SubmitInfo::default()
                    .signal_semaphores(&[*transfer_semaphore])
                    .command_buffers(&transfer_command_buffers)],
                vk::Fence::null(),
            )
            .unwrap();

        device
            .queue_submit(
                *device.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[*transfer_semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                    .command_buffers(&[*command_buffer])],
                *fence,
            )
            .unwrap();

        println!("Waiting for voxelization");
        device.wait_for_fences(&[*fence], true, !0).unwrap();
    }

    println!("{:#.10?}", device.allocator.generate_report());

    drop(images);
    drop(buffer);

    let num_outputs = num_outputs_buffer.try_as_slice::<u32>().unwrap()[0];

    dbg!(num_outputs, max_outputs);

    if num_outputs as u64 > max_outputs {
        panic!("Overflowed. {}/{}", num_outputs, max_outputs);
    }

    let mut output = output_buffer
        .try_as_slice::<(u32, u32, PackedMaterial)>()
        .unwrap()[..num_outputs as usize]
        .to_vec();

    drop(output_buffer);

    dbg!("Starting sort");
    radsort::sort_by_key(&mut output, |&(pos_high, pos_low, _)| (pos_high, pos_low));
    dbg!(output.len());
    output.dedup_by_key(|&mut (pos_high, pos_low, _)| (pos_high, pos_low));
    dbg!(output.len());

    let mut tree: tree64::Tree64<PackedMaterial> = tree64::Tree64 {
        nodes: Default::default(),
        data: Default::default(),
        stats: Default::default(),
        edits: Default::default(),
    };

    let mut nodes_a = Vec::new();
    let mut nodes_b = Vec::new();

    dbg!("Starting first");
    collect(
        |(pos_high, pos_low, val)| {
            (
                undo_morton_encoding(((pos_high as u64) << 32) | (pos_low as u64)),
                val,
            )
        },
        &output,
        &mut nodes_a,
        |arr, pop_mask| tree64::Node::new(true, tree.insert_values(arr), pop_mask),
    );
    dbg!("Done first");

    collect(
        |pos| pos,
        &nodes_a,
        &mut nodes_b,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );

    dbg!(nodes_a.len(), nodes_b.len());

    collect(
        |pos| pos,
        &nodes_b,
        &mut nodes_a,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );

    dbg!(nodes_a.len());

    collect(
        |pos| pos,
        &nodes_a,
        &mut nodes_b,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );
    dbg!(nodes_b.len());

    collect(
        |pos| pos,
        &nodes_b,
        &mut nodes_a,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );

    dbg!(nodes_a.len());

    collect(
        |pos| pos,
        &nodes_a,
        &mut nodes_b,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );
    dbg!(nodes_b.len());

    collect(
        |pos| pos,
        &nodes_b,
        &mut nodes_a,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );
    dbg!(nodes_a.len());
    collect(
        |pos| pos,
        &nodes_a,
        &mut nodes_b,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );
    dbg!(nodes_b.len());
    collect(
        |pos| pos,
        &nodes_b,
        &mut nodes_a,
        |arr, pop_mask| tree64::Node::new(false, tree.insert_nodes(arr), pop_mask),
    );
    dbg!(nodes_a.len());
    dbg!(&tree.stats);
    tree.push_new_root_node(nodes_a[0].1, 11, -glam::IVec3::splat(dim_size as i32) / 2);

    dbg!(
        std::mem::size_of_val(&tree.nodes[..]),
        std::mem::size_of_val(&tree.data[..])
    );
    tree.serialize(std::fs::File::create("out.tree64").unwrap())
        .unwrap();
}

fn collect<
    T: Copy + Default,
    O,
    F: FnMut(&[T], u64) -> O,
    S: Copy + std::fmt::Debug,
    E: Fn(S) -> (glam::UVec3, T),
>(
    extractor: E,
    sorted_values: &[S],
    output: &mut Vec<(glam::UVec3, O)>,
    mut transform: F,
) {
    output.clear();
    let mut values = tree64::PopMaskedData::default();

    let mut prev_pos = glam::UVec3::splat(u32::MAX);

    for (i, (pos, val)) in sorted_values.iter().copied().map(extractor).enumerate() {
        let key_pos = pos / 4;
        if !(i == 0 || key_pos == prev_pos) {
            output.push((prev_pos, transform(&values.as_compact(), values.pop_mask)));
            values.pop_mask = 0;
        }

        prev_pos = key_pos;

        let index = (pos % 4).dot(glam::UVec3::new(1, 4, 16));
        values.set(index, Some(val));
    }

    output.push((prev_pos, transform(&values.as_compact(), values.pop_mask)));
}

fn undo_morton_encoding(encoding: u64) -> glam::UVec3 {
    glam::UVec3::new(
        combine_bits_64(encoding),
        combine_bits_64(encoding >> 1),
        combine_bits_64(encoding >> 2),
    )
}

#[test]
fn morton_undo_testing() {
    assert_eq!(undo_morton_encoding(0b111), glam::UVec3::new(1, 1, 1));
    assert_eq!(undo_morton_encoding(0b000), glam::UVec3::new(0, 0, 0));
    assert_eq!(undo_morton_encoding(0b001), glam::UVec3::new(1, 0, 0));
    assert_eq!(undo_morton_encoding(0b010), glam::UVec3::new(0, 1, 0));
    assert_eq!(undo_morton_encoding(0b100), glam::UVec3::new(0, 0, 1));
    assert_eq!(undo_morton_encoding(0b111000), glam::UVec3::new(2, 2, 2));
    assert_eq!(undo_morton_encoding(0b100000001), glam::UVec3::new(1, 0, 4));
}

fn combine_bits_64(mut x: u64) -> u32 {
    x &= 0b1001001001001001001001001001001001001001001001001001001001001001u64; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0b0011000011000011000011000011000011000011000011000011000011000011u64; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0b1111000000001111000000001111000000001111000000001111000000001111u64; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0b0000000011111111000000000000000011111111000000000000000011111111u64; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0b1111111111111111000000000000000000000000000000001111111111111111u64; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x ^= x >> 32;
    return x as u32;
}

#[test]
fn morton_xyz() {
    fn morton_encoding(pos: glam::UVec3) -> u64 {
        separate_bits_64(pos.x) | (separate_bits_64(pos.y) << 1) | (separate_bits_64(pos.z) << 2)
    }

    fn separate_bits_64(n: u32) -> u64 {
        let mut n = n as u64;
        n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111u64;
        n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111u64;
        n = (n ^ (n << 8)) & 0b1111000000001111000000001111000000001111000000001111000000001111u64;
        n = (n ^ (n << 4)) & 0b0011000011000011000011000011000011000011000011000011000011000011u64;
        n = (n ^ (n << 2)) & 0b1001001001001001001001001001001001001001001001001001001001001001u64;
        n
    }

    assert_eq!(
        combine_bits_64(separate_bits_64(12345 * 2 * 100)),
        12345 * 2 * 100
    );

    for i in 0..100 {
        let x = glam::UVec3::new(12345, 12345 * 2, 12345 * 3) + i;
        assert_eq!(x, undo_morton_encoding(morton_encoding(x)));
    }

    for i in 0..100 {
        let x = glam::UVec3::new(12345, 12345 * 2, 12345 * 3) * 50 + i;
        assert_eq!(x, undo_morton_encoding(morton_encoding(x)));
    }
}
