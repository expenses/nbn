use ash::vk;

fn main() {
    env_logger::init();

    let instance = nbn::Instance::new(None);
    let device = instance.create_device();
    let mut buffer = device.create_buffer(nbn::BufferDescriptor {
        name: "buffer",
        size: 1024,
        ty: nbn::BufferType::Staging,
    });

    let data = [55_u8; 64];

    presser::copy_from_slice_to_offset_with_align(&data, &mut buffer.allocation, 0, 0).unwrap();

    let compute_pipeline = device.create_compute_pipeline(nbn::ComputePipelineDesc {
        name: c"main",
        path: "shader.spv",
    });

    let output = device.create_buffer(nbn::BufferDescriptor {
        name: "output_buffer",
        size: 96,
        ty: nbn::BufferType::Readback,
    });

    let command_buffer = device.create_command_buffer();

    unsafe {
        let fence = device.create_fence();

        device
            .begin_command_buffer(
                command_buffer.command_buffer,
                &ash::vk::CommandBufferBeginInfo::default(),
            )
            .unwrap();

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *compute_pipeline,
        );
        device.cmd_push_constants(
            *command_buffer,
            compute_pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &output.to_le_bytes(),
        );

        device.cmd_dispatch(*command_buffer, 1, 1, 1);

        device
            .end_command_buffer(command_buffer.command_buffer)
            .unwrap();

        device
            .device
            .queue_submit(
                device.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer.command_buffer])],
                *fence,
            )
            .unwrap();

        device.device.wait_for_fences(&[*fence], true, !0).unwrap();

        dbg!(output.allocation.mapped_slice());
    }
}
