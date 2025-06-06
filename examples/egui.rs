use ash::vk;
use std::sync::Arc;

fn create_pipeline(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    swapchain: &nbn::Swapchain,
) -> nbn::Pipeline {
    device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        name: "triangle pipeline",
        shaders: nbn::GraphicsPipelineShaders::Legacy {
            vertex: nbn::ShaderDesc {
                module: shader,
                entry_point: c"vertex",
            },
            fragment: nbn::ShaderDesc {
                module: shader,
                entry_point: c"fragment",
            },
        },
        color_attachment_formats: &[swapchain.create_info.image_format],
        blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)],
        conservative_rasterization: false,
        depth: Default::default(),
        cull_mode: Default::default(),
    })
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    shader: nbn::ReloadableShader,
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer,
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

        let swapchain =
            device.create_swapchain(&window, vk::ImageUsageFlags::COLOR_ATTACHMENT, false);

        let shader = device.load_reloadable_shader("shaders/compiled/triangle.spv");
        let pipeline = create_pipeline(&device, &shader, &swapchain);

        let egui_ctx = egui::Context::default();

        self.window_state = Some(WindowState {
            alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer::new(),
            egui_render: nbn::egui::Renderer::new(
                &device,
                swapchain.create_info.image_format,
                16 * 1024 * 1024,
            ),
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            sync_resources: device.create_sync_resources(),
            swapchain,
            pipeline,
            shader,
            egui_winit: egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                event_loop,
                Some(window.scale_factor() as _),
                None,
                None,
            ),
            window,
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

        let _ = state.egui_winit.on_window_event(&state.window, &event);

        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                let device = self.device.as_ref().unwrap();
                device.recreate_swapchain(&mut state.swapchain);
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                if state.shader.try_reload(device) {
                    unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                    state.pipeline = create_pipeline(device, &state.shader, &state.swapchain);
                }

                let raw_input = state.egui_winit.take_egui_input(&state.window);

                let egui_ctx = state.egui_winit.egui_ctx();

                egui_ctx.begin_pass(raw_input);
                {
                    let allocator = device.allocator.inner.read();
                    egui::Window::new("Memory Allocations").show(egui_ctx, |ui| {
                        state.alloc_vis.render_breakdown_ui(ui, &allocator);
                        ui.label(format!("{:?}", &device.descriptors.sampled_image_count));
                        ui.label(format!("{:?}", &device.descriptors.storage_image_count));
                    });
                    egui::Window::new("Memory Blocks").show(egui_ctx, |ui| {
                        state.alloc_vis.render_memory_block_ui(ui, &allocator);
                    });
                    state
                        .alloc_vis
                        .render_memory_block_visualization_windows(egui_ctx, &allocator);
                }
                let output = egui_ctx.end_pass();
                state
                    .egui_winit
                    .handle_platform_output(&state.window, output.platform_output);

                let clipped_meshes = state
                    .egui_winit
                    .egui_ctx()
                    .tessellate(output.shapes, output.pixels_per_point);

                unsafe {
                    let current_frame = state.sync_resources.current_frame;
                    let command_buffer = &state.per_frame_command_buffers[current_frame];

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
                        .begin_command_buffer(
                            **command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();

                    state.egui_render.update_textures(
                        device,
                        command_buffer,
                        current_frame,
                        &output.textures_delta,
                    );

                    vk_sync::cmd::pipeline_barrier(
                        device,
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
                    let extent = state.swapchain.create_info.image_extent;
                    device.begin_rendering(
                        command_buffer,
                        extent.width,
                        extent.height,
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

                    state.egui_render.paint(
                        device,
                        command_buffer,
                        &clipped_meshes,
                        state.window.scale_factor() as _,
                        [extent.width, extent.height],
                        current_frame,
                    );

                    device.cmd_end_rendering(**command_buffer);

                    vk_sync::cmd::pipeline_barrier(
                        device,
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

                    frame.submit(
                        device,
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
