use nbn::vk;

fn main() {
    let device = nbn::Device::new(None);

    let width = 512;
    let height = 512;

    let image = device.create_image(nbn::ImageDescriptor {
        name: "render attachment",
        format: vk::Format::R8G8B8A8_UNORM,
        extent: nbn::ImageExtent::D2 { width, height },
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_levels: 1,
    });

    let depthbuffer = device.create_image(nbn::ImageDescriptor {
        name: "depth attachment",
        format: vk::Format::D32_SFLOAT,
        extent: nbn::ImageExtent::D2 { width, height },
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        mip_levels: 1,
    });

    let buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "buffer",
            size: width as u64 * height as u64 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let shader = device.load_shader("shaders/compiled/compute_example.spv");

    let pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        name: "triangle pipeline",
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
        color_attachment_formats: &[vk::Format::R8G8B8A8_UNORM],
        blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)],
        depth: nbn::GraphicsPipelineDepthDesc {
            write_enable: true,
            test_enable: true,
            compare_op: vk::CompareOp::GREATER,
            format: vk::Format::D32_SFLOAT,
        },
        flags: Default::default(),
    });

    let command_buffer = device.create_command_buffer(nbn::QueueType::Graphics);

    let dragon = std::fs::File::open("dragon.bin").unwrap();
    let dragon_size = dragon.metadata().unwrap().len();
    let dragon_buffer = device.create_buffer_from_reader(dragon_size, "dragon", dragon);
    let size = dragon_buffer.try_as_slice::<u32>().unwrap()[0];

    let staging_buffer = device.create_buffer_with_data::<u32>(nbn::BufferInitDescriptor {
        name: "staging buffer",
        data: &[u32::from_le_bytes([200, 100, 70, 255]); 512 * 512],
    });

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();

        let image2 = device.create_image_with_data_in_command_buffer(
            nbn::SampledImageDescriptor {
                name: "img",
                format: vk::Format::R8G8B8A8_UNORM,
                extent: nbn::ImageExtent::D2 {
                    width: 512,
                    height: 512,
                },
            },
            &staging_buffer,
            nbn::QueueType::Graphics,
            &[0],
            &command_buffer,
        );

        let index = device.register_image(*image2.view, false);

        device.cmd_pipeline_barrier2(
            *command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                nbn::ImageBarrier::<_, nbn::BarrierOp, _> {
                    image: &image,
                    src: None,
                    dst: nbn::BarrierOp::ColorAttachmentWrite,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                }
                .into(),
                nbn::ImageBarrier::<_, nbn::BarrierOp, _> {
                    image: &depthbuffer,
                    src: None,
                    dst: nbn::BarrierOp::DepthStencilAttachmentReadWrite,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                }
                .into(),
            ]),
        );
        device.begin_rendering(
            &command_buffer,
            512,
            512,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*image.view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.2, 0.1, 0.1, 1.0],
                    },
                })],
            Some(
                &vk::RenderingAttachmentInfo::default()
                    .image_view(*depthbuffer.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ),
        );
        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline);
        device.bind_internal_descriptor_sets_to_all(&command_buffer);
        device.push_constants::<(u64, u32)>(&command_buffer, (*dragon_buffer, *index));
        device.cmd_draw(*command_buffer, size, 1, 0, 0);
        device.cmd_end_rendering(*command_buffer);
        device.cmd_pipeline_barrier2(
            *command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[nbn::ImageBarrier::<
                _,
                nbn::BarrierOp,
                _,
            > {
                image: &image,
                src: Some(nbn::BarrierOp::ColorAttachmentWrite),
                dst: nbn::BarrierOp::TransferRead,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
            }
            .into()]),
        );
        device.cmd_copy_image_to_buffer(
            *command_buffer,
            **image,
            vk::ImageLayout::GENERAL,
            *buffer.buffer,
            &[vk::BufferImageCopy::default()
                .image_extent(vk::Extent3D {
                    width: 512,
                    height: 512,
                    depth: 1,
                })
                .image_subresource(image.subresource_layer())],
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

    let slice = buffer.try_as_slice::<u8>().unwrap();

    use std::io::Write;
    let mut output = std::io::BufWriter::new(std::fs::File::create("output.ppm").unwrap());
    write!(output, "P3 {} {} 255", width, height).unwrap();
    for rgba in slice.chunks(4) {
        write!(output, " {} {} {}", rgba[0], rgba[1], rgba[2]).unwrap();
    }
}
