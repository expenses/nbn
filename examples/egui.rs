use ash::vk;
use std::sync::Arc;

fn create_pipeline(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    swapchain: &nbn::Swapchain,
) -> nbn::Pipeline {
    device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        vertex: nbn::ShaderDesc {
            module: &shader,
            entry_point: c"vertex",
        },
        fragment: nbn::ShaderDesc {
            module: &shader,
            entry_point: c"fragment",
        },
        color_attachment_formats: &[swapchain.create_info.image_format],
        conservative_rasterization: false,
        depth: Default::default(),
        cull_mode: Default::default(),
        mesh_shader: false,
    })
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    shader: nbn::ReloadableShader,
    integration: egui_winit_ash_integration::Integration<Allocator>,
}

struct App {
    window_state: Option<WindowState>,
    device: Option<Arc<nbn::Device>>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = Arc::new(nbn::Device::new(Some(&window), false));

        let size = window.inner_size();

        let swapchain =
            device.create_swapchain(&window, vk::ImageUsageFlags::COLOR_ATTACHMENT, false);

        let integration = egui_winit_ash_integration::Integration::new(
            event_loop,
            size.width,
            size.height,
            window.scale_factor(),
            4096,
            egui::FontDefinitions::default(),
            egui::Style::default(),
            (*device).clone(),
            Allocator(device.clone()),
            device.graphics_queue.index,
            device.graphics_queue.inner,
            device.swapchain_loader.clone(),
            swapchain.clone(),
            swapchain.surface_format(),
        );

        let shader = device.load_reloadable_shader("shaders/compiled/triangle.spv");
        let pipeline = create_pipeline(&device, &shader, &swapchain);

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
            shader,
            integration,
        });
        self.device = Some(device);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = self.window_state.as_mut().unwrap();

        let _ = state.integration.handle_event(&state.window, &event);

        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                let device = self.device.as_ref().unwrap();
                device.recreate_swapchain(&mut state.swapchain);
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                state.integration.update_swapchain(
                    new_size.width,
                    new_size.height,
                    *state.swapchain,
                    state.swapchain.surface_format(),
                );
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                if state.shader.try_reload(&device) {
                    unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                    state.pipeline = create_pipeline(&device, &state.shader, &state.swapchain);
                }

                state.integration.begin_frame(&state.window);
                egui::Window::new("wow").show(&state.integration.context(), |ui| {
                    ui.label("hello world");
                });
                let output = state.integration.end_frame(&mut state.window);
                let clipped_meshes = state
                    .integration
                    .context()
                    .tessellate(output.shapes, output.pixels_per_point);

                unsafe {
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

                    device.reset_command_buffer(&command_buffer);
                    device
                        .begin_command_buffer(
                            **command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
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
                        state.swapchain.create_info.image_extent.width,
                        state.swapchain.create_info.image_extent.height,
                        &[vk::RenderingAttachmentInfo::default()
                            .image_view(*image.view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue { float32: [1.0; 4] },
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
                    device.cmd_draw(**command_buffer, 3, 1, 0, 0);

                    device.cmd_end_rendering(**command_buffer);

                    state.integration.paint(
                        **command_buffer,
                        next_image as _,
                        clipped_meshes,
                        output.textures_delta,
                    );

                    /*vk_sync::cmd::pipeline_barrier(
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
                    );*/
                    device.end_command_buffer(**command_buffer).unwrap();

                    frame.submit(
                        &device,
                        &[vk::CommandBufferSubmitInfo::default().command_buffer(**command_buffer)],
                    );
                    device
                        .swapchain_loader
                        .queue_present(
                            *device.graphics_queue,
                            &vk::PresentInfoKHR::default()
                                .wait_semaphores(&[*frame.render_finished_semaphore])
                                .swapchains(&[*state.swapchain])
                                .image_indices(&[next_image]),
                        )
                        .unwrap();
                }
            }
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
        let device = self.device.as_ref().unwrap();

        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.window_state = None;
        self.device = None;
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
            device: None,
            window_state: None,
        })
        .unwrap();
}

struct Allocator(Arc<nbn::Device>);

impl egui_winit_ash_integration::AllocatorTrait for Allocator {
    type Allocation = gpu_allocator::vulkan::Allocation;
    type AllocationCreateInfo = gpu_allocator::vulkan::AllocationCreateDesc<'static>;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> anyhow::Result<Self::Allocation> {
        Ok(self.0.allocator.inner.lock().allocate(&desc)?)
    }

    fn free(&self, allocation: Self::Allocation) -> anyhow::Result<()> {
        Ok(self.0.allocator.inner.lock().free(allocation)?)
    }
}
