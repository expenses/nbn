use ash::vk;

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
}

impl WindowState {
    fn render(&mut self, device: &nbn::Device) {
        unsafe {
            let command_buffer = &self.per_frame_command_buffers[self.sync_resources.current_frame];
            let mut frame = self.sync_resources.wait_for_frame(device);

            let (next_image, _suboptimal) = device
                .swapchain_loader
                .acquire_next_image(
                    *self.swapchain,
                    !0,
                    *frame.image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();
            let image = &self.swapchain.images[next_image as usize];

            let image_view = nbn::ImageView::from_raw(
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image.image)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .layer_count(1)
                                    .level_count(1)
                                    .aspect_mask(vk::ImageAspectFlags::COLOR),
                            )
                            .format(vk::Format::B8G8R8A8_SRGB)
                            .view_type(vk::ImageViewType::TYPE_2D),
                        None,
                    )
                    .unwrap(),
                &device,
            );

            device.reset_command_buffer(&command_buffer);
            device
                .begin_command_buffer(**command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            vk_sync::cmd::pipeline_barrier(
                &device,
                **command_buffer,
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::Present],
                    next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: image.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                }],
            );
            device.begin_rendering(
                &command_buffer,
                800,
                600,
                &[vk::RenderingAttachmentInfo::default()
                    .image_view(*image_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [1.0; 4] },
                    })
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)],
            );
            device.cmd_bind_pipeline(
                **command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline,
            );
            device.cmd_draw(**command_buffer, 3, 1, 0, 0);

            device.cmd_end_rendering(**command_buffer);
            vk_sync::cmd::pipeline_barrier(
                &device,
                **command_buffer,
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                    next_accesses: &[vk_sync::AccessType::Present],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: image.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                }],
            );
            device.end_command_buffer(**command_buffer).unwrap();

            let cb_infos =
                &[vk::CommandBufferSubmitInfo::default().command_buffer(**command_buffer)];

            frame.submit(&device, cb_infos);
            device
                .swapchain_loader
                .queue_present(
                    *device.graphics_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[*frame.render_finished_semaphore])
                        .swapchains(&[*self.swapchain])
                        .image_indices(&[next_image]),
                )
                .unwrap();
        }
    }
}

struct State {
    device: nbn::Device,
}

struct App {
    window_state: Option<WindowState>,
    state: Option<State>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let swapchain = device.create_swapchain(&window);
        let shader = device.load_shader("shaders/compiled/triangle.spv");
        let pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            vertex: nbn::ShaderDesc {
                module: &shader,
                entry_point: c"vertex",
            },
            fragment: nbn::ShaderDesc {
                module: &shader,
                entry_point: c"fragment",
            },
            color_attachment_formats: &[swapchain.surface_format],
            conservative_rasterization: false,
        });

        self.window_state = Some(WindowState {
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            sync_resources: device.create_sync_resources(),
            swapchain,
            window,
            pipeline,
        });
        self.state = Some(State { device });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.state.as_ref().unwrap();

                if let Some(state) = self.window_state.as_mut() {
                    state.render(&device.device);
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyQ),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let device = self.state.as_ref().unwrap();

                unsafe {
                    device.device.device_wait_idle().unwrap();
                }

                self.window_state = None;
                self.state = None;

                event_loop.exit();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(app_state) = self.window_state.as_mut() {
            app_state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App {
            state: None,
            window_state: None,
        })
        .unwrap();
}
