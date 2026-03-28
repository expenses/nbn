
use nbn::{vk, winit};

slang_struct::slang_include!("shaders/particles_structs.slang");

struct Particles {
    data: nbn::Buffer,
    draw_command: nbn::Buffer,
    dispatch_command: nbn::Buffer,
}

impl Particles {
    fn new(device: &nbn::Device) -> Self {
        Self {
            data: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "data",
                    size: std::mem::size_of::<Particle>() as u64 * 20_000,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            draw_command: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "draw_command",
                    size: std::mem::size_of::<vk::DrawIndirectCommand>() as u64,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            dispatch_command: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "dispatch_command",
                    size: std::mem::size_of::<[u32; 3]>() as u64,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
        }
    }
}

struct State {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    reset: nbn::Pipeline,
    spawn: nbn::Pipeline,
    copy: nbn::Pipeline,
    particles: [Particles; 2],
    frame_index: u32,
    device: nbn::Device,
}

struct App {
    state: Option<State>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
                desire_hdr: false,
            },
        );
        let shader = device.load_shader("../shaders/compiled/particles.spv");
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
            color_attachment_formats: &[swapchain.create_info.image_format],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            flags: nbn::GraphicsPipelineFlags::POINTS,
            depth: Default::default(),
        });

        self.state = Some(State {
            particles: [Particles::new(&device), Particles::new(&device)],
            reset: device.create_compute_pipeline(&shader, c"reset"),
            spawn: device.create_compute_pipeline(&shader, c"spawn"),
            copy: device.create_compute_pipeline(&shader, c"copy"),
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            sync_resources: device.create_sync_resources(),
            swapchain,
            window,
            pipeline,
            frame_index: 0,
            device,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        let device = &state.device;

        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                device.recreate_swapchain(&mut state.swapchain);
            }
            winit::event::WindowEvent::RedrawRequested => unsafe {
                let command_buffer =
                    &state.per_frame_command_buffers[state.sync_resources.current_frame];
                let mut frame = state.sync_resources.wait_for_frame(device);

                let (next_image, _suboptimal) = device
                    .swapchain_loader
                    .acquire_next_image(
                        *state.swapchain,
                        !0,
                        *frame.image_available_semaphore,
                        vk::Fence::null(),
                    )
                    .unwrap();
                let image = &state.swapchain.images[next_image as usize];

                device.reset_command_buffer(command_buffer);
                device
                    .begin_command_buffer(**command_buffer, &vk::CommandBufferBeginInfo::default())
                    .unwrap();

                let current_index = (state.frame_index % 2) as usize;
                let other_index = ((state.frame_index + 1) % 2) as usize;

                device.push_constants(
                    &command_buffer,
                    PushConstants {
                        draw_command: *state.particles[current_index].draw_command,
                        particles: *state.particles[current_index].data,
                        dispatch_command: *state.particles[current_index].dispatch_command,
                        other_particles: *state.particles[other_index].data,
                        other_draw_command: *state.particles[other_index].draw_command,
                        other_dispatch_command: *state.particles[other_index].dispatch_command,
                        frame: state.frame_index,
                    },
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.reset,
                );
                device.cmd_dispatch(**command_buffer, 1, 1, 1);

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.spawn,
                );
                device.cmd_dispatch(**command_buffer, 2, 1, 1);

                device.insert_image_pipeline_barrier(
                    &command_buffer,
                    image,
                    Some(nbn::BarrierOp::Acquire),
                    nbn::BarrierOp::ColorAttachmentWrite,
                );

                device.begin_rendering(
                    command_buffer,
                    state.swapchain.create_info.image_extent.width,
                    state.swapchain.create_info.image_extent.height,
                    &[vk::RenderingAttachmentInfo::default()
                        .image_view(*image.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        })
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)],
                    None,
                );
                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *state.pipeline,
                );
                device.cmd_draw_indirect(
                    **command_buffer,
                    *state.particles[current_index].draw_command.buffer,
                    0,
                    1,
                    0,
                );

                device.cmd_end_rendering(**command_buffer);
                device.insert_image_pipeline_barrier(
                    &command_buffer,
                    image,
                    Some(nbn::BarrierOp::ColorAttachmentWrite),
                    nbn::BarrierOp::Present,
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.copy,
                );
                device.cmd_dispatch_indirect(
                    **command_buffer,
                    *state.particles[current_index].dispatch_command.buffer,
                    0,
                );

                device.end_command_buffer(**command_buffer).unwrap();

                frame.submit(
                    device,
                    &image,
                    &[vk::CommandBufferSubmitInfo::default().command_buffer(**command_buffer)],
                );
                device
                    .swapchain_loader
                    .queue_present(
                        *device.graphics_queue,
                        &vk::PresentInfoKHR::default()
                            .wait_semaphores(&[*image.render_finished_semaphore])
                            .swapchains(&[*state.swapchain])
                            .image_indices(&[next_image]),
                    )
                    .unwrap();

                state.frame_index += 1;
            },
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let device = &self.state.as_ref().unwrap().device;

        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.state = None;
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.run_app(&mut App { state: None }).unwrap();
}
