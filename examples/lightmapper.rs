use ash::vk;
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

slang_struct::slang_include!("shaders/lightmapper_structs.slang");

fn main() {
    env_logger::init();

    let device = Arc::new(nbn::Device::new(None));

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 64 * 1024 * 1024, nbn::QueueType::Compute);

    let mut args = std::env::args().skip(1);

    let (gltf_data, model, lights) = load_gltf(
        &device,
        &mut staging_buffer,
        &std::path::Path::new(&args.next().unwrap()),
    );

    let width = args.next().unwrap().parse::<u32>().unwrap();
    let height = args.next().unwrap().parse::<u32>().unwrap();

    let mut output_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "output",
            size: width as u64 * height as u64 * 4 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let output_image = device.create_image(nbn::ImageDescriptor {
        name: "output image",
        format: vk::Format::R32G32B32A32_SFLOAT,
        extent: [width, height].into(),
        usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_levels: 1,
    });

    let pos_image = device.create_image(nbn::ImageDescriptor {
        name: "pos image",
        format: vk::Format::R32G32B32A32_SFLOAT,
        extent: [width, height].into(),
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_levels: 1,
    });

    let normal_image = device.create_image(nbn::ImageDescriptor {
        name: "normal image",
        format: vk::Format::R32G32B32A32_SFLOAT,
        extent: [width, height].into(),
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_levels: 1,
    });

    //let num_lights = dbg!(lights.len());

    //let lights = staging_buffer.create_buffer_from_slice(&device, "lights", &lights);

    let instance_buffer = staging_buffer.create_buffer_from_slice(
        &device,
        "Instances",
        &[vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: glam::Mat4::IDENTITY.transpose().to_cols_array()[..12]
                    .try_into()
                    .unwrap(),
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                ash::vk::GeometryInstanceFlagsKHR::default().as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: *gltf_data.acceleration_structure,
            },
        }],
    );

    let model_buffer = staging_buffer.create_buffer_from_slice(&device, "models", &[model]);

    let tlas = device.create_acceleration_structure(
        "tlas",
        nbn::AccelerationStructureData::Instances {
            buffer_address: *instance_buffer,
            count: 1,
        },
        &mut staging_buffer,
    );

    let blue_noise_buffers = nbn::blue_noise::BlueNoiseBuffers::new(&device, &mut staging_buffer);

    staging_buffer.finish(&device);

    let shader = device.load_shader("shaders/compiled/lightmapper.spv");

    let pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        name: "lightmapper pipeline",
        shaders: nbn::GraphicsPipelineShaders::Legacy {
            vertex: nbn::ShaderDesc {
                module: &shader,
                entry_point: c"vertex",
            },
            fragment: nbn::ShaderDesc {
                module: &shader,
                entry_point: c"fragment",
            },
        },
        color_attachment_formats: &[vk::Format::R32G32B32A32_SFLOAT; 2],
        blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA); 2],
        flags: Default::default(),
        depth: Default::default(),
    });

    let mut command_buffer = device.create_command_buffer(nbn::QueueType::Graphics);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.insert_image_pipeline_barrier(
            &command_buffer,
            &pos_image,
            None,
            nbn::BarrierOp::ColorAttachmentWrite,
        );
        device.insert_image_pipeline_barrier(
            &command_buffer,
            &normal_image,
            None,
            nbn::BarrierOp::ColorAttachmentWrite,
        );
        device.begin_rendering(
            &command_buffer,
            width,
            height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*pos_image.view)
                .image_layout(vk::ImageLayout::GENERAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE), vk::RenderingAttachmentInfo::default()
                    .image_view(*normal_image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    })
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)],
            None,
        );
        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline);
        device.push_constants::<PushConstants>(
            &command_buffer,
            PushConstants {
                blue_noise_ranking_tile: *blue_noise_buffers.ranking_tile,
                blue_noise_sobol: *blue_noise_buffers.sobol,
                blue_noise_scrambling_tile: *blue_noise_buffers.scrambling_tile,
                extent: [width, height],
                lights: 0,
                model: *model_buffer,
                output: *output_buffer,
                num_lights: 0,
                tlas: *tlas,
            },
        );
        device.cmd_draw(*command_buffer, model.num_indices, 1, 0, 0);

        device.cmd_end_rendering(*command_buffer);
        device.insert_image_pipeline_barrier(
            &command_buffer,
            &output_image,
            None,//Some(nbn::BarrierOp::ColorAttachmentWrite),
            nbn::BarrierOp::TransferWrite,
        );
        device.cmd_copy_image_to_buffer(
            *command_buffer,
            **output_image,
            vk::ImageLayout::GENERAL,
            *output_buffer.buffer,
            &[vk::BufferImageCopy::default()
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .image_subresource(output_image.subresource_layer())],
        );
        device.end_command_buffer(*command_buffer).unwrap();

        let fence = device.create_fence();

        device
            .queue_submit(
                *device.graphics_queue,
                &[vk::SubmitInfo::default().command_buffers(&[*command_buffer])],
                *fence,
            )
            .unwrap();
        device.wait_for_fences(&[*fence], true, !0).unwrap();
    }

    let output_slice = output_buffer.try_as_slice::<f32>().unwrap();

    image::ImageBuffer::<image::Rgba<f32>, &[f32]>::from_raw(width, height, output_slice)
        .unwrap()
        .save("out.exr")
        .unwrap();
}

fn load_gltf(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> (CombinedModel, Model, Vec<Light>) {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    //assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let lights: Vec<_> = gltf
        .nodes
        .iter()
        .filter_map(|node| {
            node.extensions
                .khr_lights_punctual
                .as_ref()
                .map(|ext| (node, ext.light))
        })
        .map(|(node, index)| {
            let (pos, rotation) = if let goth_gltf::NodeTransform::Set {
                translation,
                rotation,
                ..
            } = node.transform()
            {
                (translation, rotation)
            } else {
                panic!()
            };

            let light = &gltf.extensions.khr_lights_punctual.as_ref().unwrap().lights[index];

            let (spotlight_angle_scale, spotlight_angle_offset) = light
                .spot
                .map(|spot| {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
                    let spotlight_angle_scale = 1.0
                        / 0.000001_f32
                            .max(spot.inner_cone_angle.cos() - spot.outer_cone_angle.cos());
                    let spotlight_angle_offset =
                        -spot.outer_cone_angle.cos() * spotlight_angle_scale;
                    (spotlight_angle_scale, spotlight_angle_offset)
                })
                .unwrap_or((0.0, 1.0));

            Light {
                position: pos,
                emission: (glam::Vec3::from(light.color) * light.intensity).into(),
                spotlight_angle_scale,
                spotlight_angle_offset,
                spotlight_direction: (glam::Quat::from_array(rotation) * glam::Vec3::Z).into(),
            }
        })
        .collect();

    let images = gltf
        .images
        .iter()
        //.zip(&images)
        .map(|image| {
            let path = path.with_file_name(image.uri.as_ref().unwrap());
            let data = image::open(&path).unwrap().to_rgba8();
            let image = staging_buffer.create_sampled_image(
                &device,
                nbn::SampledImageDescriptor {
                    name: image.uri.as_ref().unwrap(),
                    extent: vk::Extent3D {
                        width: data.width(),
                        height: data.height(),
                        depth: 1,
                    }
                    .into(),
                    format: vk::Format::R8G8B8A8_SRGB,
                },
                &data,
                nbn::QueueType::Compute,
                &[0],
            );
            nbn::IndexedImage {
                index: device.register_image(*image.view, false),
                image,
            }
        })
        .collect::<Vec<_>>();

    let material_to_image: Vec<u32> = gltf
        .materials
        .iter()
        .map(|mat| {
            let texture_index = mat
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .unwrap()
                .index;
            *images[gltf.textures[texture_index].source.unwrap()].index
        })
        .collect();

    let buffer = buffer.map(|buffer| buffer.to_vec()).unwrap_or_else(|| {
        std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap()
    });
    //let buffer = std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    fn get_slice<'a, T: Copy>(
        buffer: &'a [u8],
        gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        accessor: &goth_gltf::Accessor,
    ) -> &'a [T] {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        &nbn::cast_slice(&buffer[bv.byte_offset + accessor.byte_offset..])
    }

    let mut indices = Vec::new();
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut image_indices = Vec::new();
    let mut uv2s = Vec::new();

    for mesh in gltf.meshes.iter() {
        for primitive in mesh.primitives.iter() {
            let indices_accessor = &gltf.accessors[primitive.indices.unwrap()];
            assert_eq!(
                indices_accessor.component_type,
                goth_gltf::ComponentType::UnsignedInt
            );
            let prim_indices =
                &get_slice::<u32>(&buffer, &gltf, &indices_accessor)[..indices_accessor.count];
            indices.extend(
                prim_indices
                    .iter()
                    .map(|&index| positions.len() as u32 / 3 + index as u32),
            );

            let get = |accessor_index: Option<usize>, size: usize, error: &str| {
                let accessor = &gltf.accessors[accessor_index.expect(error)];
                assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                &get_slice::<f32>(&buffer, &gltf, accessor)[..accessor.count * size]
            };

            let positions_slice = get(primitive.attributes.position, 3, "positions");
            positions.extend_from_slice(positions_slice);
            uvs.extend_from_slice(get(primitive.attributes.texcoord_0, 2, "uvs"));
            uv2s.extend_from_slice(get(primitive.attributes.texcoord_1, 2, "uv2s"));
            normals.extend_from_slice(get(primitive.attributes.normal, 3, "normals"));

            let material_index = primitive.material.unwrap_or(0);

            image_indices.extend(
                (0..prim_indices.len() / 3)
                    .map(|_| material_to_image.get(material_index).cloned().unwrap_or(0)),
            );
        }
    }

    let num_vertices = positions.len() / 3;
    let num_indices = indices.len();
    let indices = staging_buffer.create_buffer_from_slice(device, "indices", &indices);
    let positions = staging_buffer.create_buffer_from_slice(device, "positions", &positions);
    dbg!(image_indices.len());

    let acceleration_structure = device.create_acceleration_structure(
        &format!("{} acceleration structure", path.display(),),
        nbn::AccelerationStructureData::Triangles {
            index_type: vk::IndexType::UINT32,
            opaque: true,
            vertices_buffer_address: *positions,
            indices_buffer_address: *indices,
            num_vertices: num_vertices as _,
            num_indices: num_indices as _,
        },
        staging_buffer,
    );

    let uvs = staging_buffer.create_buffer_from_slice(device, "uvs", &uvs);
    let uv2s = staging_buffer.create_buffer_from_slice(device, "uv2s", &uv2s);
    let normals = staging_buffer.create_buffer_from_slice(device, "normals", &normals);
    let image_indices =
        staging_buffer.create_buffer_from_slice(device, "image_indices", &image_indices);

    let model = Model {
        positions: *positions,
        uvs: *uvs,
        uv2s: *uv2s,
        normals: *normals,
        indices: *indices,
        image_indices: *image_indices,
        flags: 1,
        num_indices: num_indices as _,
    };

    (
        CombinedModel {
            acceleration_structure,
            _positions: positions,
            _indices: indices,
            _uvs: uvs,
            _uv2s: uv2s,
            _normals: normals,
            _image_indices: image_indices,
            _images: images,
        },
        model,
        lights,
    )
}

struct CombinedModel {
    acceleration_structure: nbn::AccelerationStructure,
    _positions: nbn::Buffer,
    _indices: nbn::Buffer,
    _uvs: nbn::Buffer,
    _uv2s: nbn::Buffer,
    _normals: nbn::Buffer,
    _image_indices: nbn::Buffer,
    _images: Vec<nbn::IndexedImage>,
}
