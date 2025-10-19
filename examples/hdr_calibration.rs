use ash::vk;
use std::sync::Arc;

fn create_pipelines(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    swapchain: &nbn::Swapchain,
) -> (nbn::Pipeline, nbn::Pipeline) {
    let create_pipeline = |vertex, fragment| {
        device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            name: "triangle pipeline",
            shaders: nbn::GraphicsPipelineShaders::Legacy {
                vertex: nbn::ShaderDesc {
                    module: shader,
                    entry_point: vertex,
                },
                fragment: nbn::ShaderDesc {
                    module: shader,
                    entry_point: fragment,
                },
            },
            color_attachment_formats: &[swapchain.create_info.image_format],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            conservative_rasterization: false,
            depth: Default::default(),
            cull_mode: Default::default(),
        })
    };

    (
        create_pipeline(c"vertex", c"fragment"),
        create_pipeline(c"fullscreen_tri", c"fragment_2"),
    )
}

#[derive(PartialEq)]
enum TriangleState {
    Rgb,
    White,
    BarelyVisible,
    Off,
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    tri_pipeline: nbn::Pipeline,
    image_pipeline: nbn::Pipeline,
    shader: nbn::ReloadableShader,
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    is_hdr: bool,
    calibrated_max_nits: f32,
    exposure: f32,
    clamp_before_transfer_function: bool,
    tonemap: bool,
    tonemapping_image: nbn::IndexedImage,
    exr: Option<nbn::IndexedImage>,
    draw_background: bool,
    triangle_state: TriangleState,
    ui_nits: f32,
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
        let device = Arc::new(nbn::Device::new(Some(&window)));

        let is_hdr = std::env::var("NBN_HDR").is_ok();

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            nbn::SurfaceSelectionCriteria {
                desire_hdr: is_hdr,
                force_8_bit: false,
            },
        );

        let shader = device.load_reloadable_shader("shaders/compiled/triangle_hdr.spv");
        let (tri_pipeline, image_pipeline) = create_pipelines(&device, &shader, &swapchain);

        let egui_ctx = egui::Context::default();

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 1024 * 1024 * 16, nbn::QueueType::Transfer);
        let image = nbn::image_loading::create_image(
            &device,
            &mut staging_buffer,
            "shaders/tony-mc-mapface/shader/tony_mc_mapface.dds",
            nbn::QueueType::Graphics,
        );

        let exr = std::env::args().nth(1).map(|filename| {
            let exr = image::open(filename).unwrap().into_rgba32f();
            let exr = staging_buffer.create_sampled_image(
                &device,
                nbn::SampledImageDescriptor {
                    name: "exr",
                    extent: nbn::ImageExtent::D2 {
                        width: exr.width(),
                        height: exr.height(),
                    },
                    format: vk::Format::R32G32B32A32_SFLOAT,
                },
                nbn::cast_slice::<f32, _>(exr.as_flat_samples().as_slice()),
                nbn::QueueType::Graphics,
                &[0],
            );
            nbn::IndexedImage {
                index: device.register_image_with_sampler(*exr.view, &device.samplers.clamp, false),
                image: exr,
            }
        });

        staging_buffer.finish(&device);

        self.window_state = Some(WindowState {
            tonemapping_image: nbn::IndexedImage {
                index: device.register_image_with_sampler(
                    *image.view,
                    &device.samplers.clamp,
                    false,
                ),
                image,
            },
            triangle_state: if exr.is_some() {
                TriangleState::Off
            } else {
                TriangleState::Rgb
            },
            exr,
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
            tri_pipeline,
            image_pipeline,
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
            is_hdr,
            calibrated_max_nits: 400.0,
            ui_nits: 203.0,
            exposure: 0.0,
            clamp_before_transfer_function: is_hdr,
            tonemap: true,
            draw_background: true,
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
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                device.recreate_swapchain(&mut state.swapchain);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                if state.shader.try_reload(device) {
                    unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                    let (tri_pipeline, image_pipeline) =
                        create_pipelines(device, &state.shader, &state.swapchain);
                    state.tri_pipeline = tri_pipeline;
                    state.image_pipeline = image_pipeline
                }

                let raw_input = state.egui_winit.take_egui_input(&state.window);

                let egui_ctx = state.egui_winit.egui_ctx();

                egui_ctx.begin_pass(raw_input);
                {
                    egui::Window::new("Settings").show(egui_ctx, |ui| {
                        ui.selectable_value(&mut state.triangle_state, TriangleState::Off, "off");
                        ui.selectable_value(&mut state.triangle_state, TriangleState::Rgb, "rgb");
                        ui.selectable_value(
                            &mut state.triangle_state,
                            TriangleState::White,
                            "white",
                        );
                        ui.selectable_value(
                            &mut state.triangle_state,
                            TriangleState::BarelyVisible,
                            "barely visible",
                        );
                        ui.add_enabled(
                            state.is_hdr,
                            egui::Checkbox::new(
                                &mut state.clamp_before_transfer_function,
                                "clamp_before_transfer_function",
                            ),
                        );
                        ui.checkbox(&mut state.tonemap, "tonemap");
                        ui.add_enabled(
                            state.exr.is_some(),
                            egui::Checkbox::new(&mut state.draw_background, "draw_background"),
                        );
                        let max_nits = 1600.0;
                        let max_exposure = 10.0;
                        ui.add_enabled(
                            state.is_hdr,
                            egui::Slider::new(&mut state.calibrated_max_nits, 0.0..=max_nits),
                        );
                        ui.add_enabled(
                            state.is_hdr,
                            egui::Slider::new(&mut state.ui_nits, 0.0..=max_nits),
                        );
                        ui.add(egui::Slider::new(&mut state.exposure, -25.0..=max_exposure));
                        ui.horizontal(|ui| {
                            if ui.button("-").clicked() && state.exposure != max_exposure {
                                state.calibrated_max_nits /= 2.0;
                                state.exposure += 1.0;
                            }
                            if ui.button("+").clicked() && state.calibrated_max_nits != max_nits {
                                state.calibrated_max_nits *= 2.0;
                                state.exposure -= 1.0;
                            }
                        });
                    });
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

                    device.insert_image_pipeline_barrier(
                        command_buffer,
                        image,
                        Some(nbn::BarrierOp::Acquire),
                        nbn::BarrierOp::ColorAttachmentWrite,
                    );
                    let extent = state.swapchain.create_info.image_extent;
                    device.begin_rendering(
                        command_buffer,
                        extent.width,
                        extent.height,
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
                    device.bind_internal_descriptor_sets_to_all(&command_buffer);
                    device.push_constants::<(f32, f32, u32, u32, u8)>(
                        &command_buffer,
                        (
                            state.calibrated_max_nits,
                            state.exposure,
                            *state.tonemapping_image,
                            state.exr.as_deref().copied().unwrap_or_default(),
                            (((state.triangle_state == TriangleState::BarelyVisible) as u8) << 4)
                                | ((state.tonemap as u8) << 3)
                                | ((state.clamp_before_transfer_function as u8) << 2)
                                | (((state.triangle_state == TriangleState::White) as u8) << 1)
                                | (state.is_hdr as u8),
                        ),
                    );
                    if state.exr.is_some() && state.draw_background {
                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *state.image_pipeline,
                        );
                        device.cmd_draw(**command_buffer, 3, 1, 0, 0);
                    }
                    if state.triangle_state != TriangleState::Off {
                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *state.tri_pipeline,
                        );
                        device.cmd_draw(**command_buffer, 3, 1, 0, 0);
                    }

                    state.egui_render.paint(
                        device,
                        command_buffer,
                        &clipped_meshes,
                        state.window.scale_factor() as _,
                        [extent.width, extent.height],
                        current_frame,
                        if state.is_hdr {
                            nbn::TransferFunction::Hdr(state.ui_nits)
                        } else {
                            nbn::TransferFunction::Srgb
                        },
                    );

                    device.cmd_end_rendering(**command_buffer);

                    device.insert_image_pipeline_barrier(
                        command_buffer,
                        image,
                        Some(nbn::BarrierOp::ColorAttachmentWrite),
                        nbn::BarrierOp::Present,
                    );
                    device.end_command_buffer(**command_buffer).unwrap();

                    frame.submit(
                        device,
                        image,
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
