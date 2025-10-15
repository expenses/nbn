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

    let buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "buffer",
            size: width as u64 * height as u64 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let shader = device.load_shader("shaders/compiled/triangle.spv");

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
        conservative_rasterization: false,
        depth: Default::default(),
        cull_mode: Default::default(),
    });

    let command_buffer = device.create_command_buffer(nbn::QueueType::Graphics);

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();
        device.insert_image_barrier2(
            &command_buffer,
            nbn::NewImageBarrier {
                image: &image,
                src: None,
                dst: nbn::BarrierOp::ColorAttachmentWrite,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
            },
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
            None,
        );
        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline);
        device.cmd_draw(*command_buffer, 3, 1, 0, 0);
        device.cmd_end_rendering(*command_buffer);
        device.insert_image_barrier2(
            &command_buffer,
            nbn::NewImageBarrier {
                image: &image,
                src: Some(nbn::BarrierOp::ColorAttachmentWrite),
                dst: nbn::BarrierOp::TransferRead,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
            },
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
                .image_subresource(image.subresource_layers())],
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
