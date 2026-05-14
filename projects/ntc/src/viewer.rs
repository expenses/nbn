use crate::RenderToTexturePushConstants;
use egui_winit::egui;
use nbn::{vk, winit};
use std::path::PathBuf;

struct State {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    device: nbn::Device,
    pipeline: nbn::Pipeline,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    params: nbn::Buffer,
    _latent_textures: [nbn::IndexedImage; 4],
    latent_texture_indices: nbn::Buffer,
    size: u32,
    scale: f32,
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    channel_offset: u32,
    exposure: f32,
}

pub struct App {
    state: Option<State>,
    path: PathBuf,
}

impl App {
    pub fn new(path: PathBuf) -> Self {
        Self { path, state: None }
    }
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
                desire_hdr: false,
            },
        );

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 1024 * 1024, nbn::QueueType::Compute);
        let (latent_textures, params, size) = crate::load_ntc_texture(
            &device,
            &mut staging_buffer,
            &std::fs::read(&self.path).unwrap(),
            true,
        );

        let latent_texture_indices: [u32; 4] = std::array::from_fn(|i| *latent_textures[i]);
        let latent_texture_indices = staging_buffer.create_buffer_from_slice(
            &device,
            "latent_texture_indices",
            &latent_texture_indices,
        );

        staging_buffer.finish(&device);

        let shader = device.load_shader("shaders/compiled/ntc.spv");

        let egui_ctx = egui::Context::default();

        self.state = Some(State {
            egui_render: nbn::egui::Renderer::new(
                &device,
                swapchain.create_info.image_format,
                16 * 1024 * 1024,
            ),
            egui_winit: egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                event_loop,
                Some(window.scale_factor() as _),
                None,
                None,
            ),
            pipeline: device.create_compute_pipeline(&shader, c"render_to_texture"),
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            sync_resources: device.create_sync_resources(),
            swapchain_image_heap_indices: swapchain
                .images
                .iter()
                .map(|image| device.register_image(*image.view, true))
                .collect(),
            latent_texture_indices,
            swapchain,
            window,
            device,
            params,
            _latent_textures: latent_textures,
            size,
            scale: 1.0,
            channel_offset: 0,
            exposure: 0.0,
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

        let _ = state.egui_winit.on_window_event(&state.window, &event);

        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                device.recreate_swapchain(&mut state.swapchain);
                state.swapchain_image_heap_indices.clear();
                state.swapchain_image_heap_indices.extend(
                    state
                        .swapchain
                        .images
                        .iter()
                        .map(|image| device.register_image(*image.view, true)),
                );
            }
            winit::event::WindowEvent::RedrawRequested => {
                let current_frame = state.sync_resources.current_frame;

                unsafe {
                    let raw_input = state.egui_winit.take_egui_input(&state.window);

                    let egui_ctx = state.egui_winit.egui_ctx();

                    egui_ctx.begin_pass(raw_input);

                    egui::Window::new("Xyz").show(egui_ctx, |ui| {
                        ui.add(egui::Slider::new(&mut state.scale, 0.0001..=10.0));
                        ui.add(egui::Slider::new(&mut state.channel_offset, 0..=16 - 3));
                        ui.add(egui::Slider::new(&mut state.exposure, -3.0..=3.0));
                    });

                    let output = egui_ctx.end_pass();
                    state
                        .egui_winit
                        .handle_platform_output(&state.window, output.platform_output);

                    let clipped_meshes = state
                        .egui_winit
                        .egui_ctx()
                        .tessellate(output.shapes, output.pixels_per_point);

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

                    device.bind_internal_descriptor_sets(
                        &command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                    );
                    device.insert_image_pipeline_barrier(
                        &command_buffer,
                        image,
                        Some(nbn::BarrierOp::Acquire),
                        nbn::BarrierOp::ComputeStorageWrite,
                    );
                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *state.pipeline,
                    );
                    let extent = state.swapchain.create_info.image_extent;
                    device.push_constants::<RenderToTexturePushConstants>(
                        &command_buffer,
                        RenderToTexturePushConstants {
                            extent: [extent.width, extent.height],
                            params: *state.params,
                            latent_textures: *state.latent_texture_indices,
                            image: *state.swapchain_image_heap_indices[next_image as usize],
                            texture_size: state.size,
                            scale: state.scale,
                            channel_offset: state.channel_offset,
                            exposure: state.exposure,
                        },
                    );
                    device.cmd_dispatch(
                        **command_buffer,
                        (extent.width).div_ceil(64),
                        extent.height,
                        1,
                    );

                    device.insert_image_pipeline_barrier(
                        command_buffer,
                        image,
                        Some(nbn::BarrierOp::ComputeStorageWrite),
                        nbn::BarrierOp::ColorAttachmentReadWrite,
                    );

                    device.begin_rendering(
                        command_buffer,
                        extent.width,
                        extent.height,
                        &[vk::RenderingAttachmentInfo::default()
                            .image_view(*image.view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)],
                        None,
                    );
                    state.egui_render.paint(
                        device,
                        command_buffer,
                        &clipped_meshes,
                        state.window.scale_factor() as _,
                        [extent.width, extent.height],
                        current_frame,
                        nbn::TransferFunction::Srgb,
                    );

                    device.cmd_end_rendering(**command_buffer);

                    device.insert_image_pipeline_barrier(
                        &command_buffer,
                        image,
                        Some(nbn::BarrierOp::ColorAttachmentReadWrite),
                        nbn::BarrierOp::Present,
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
