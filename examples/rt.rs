use ash::vk;
use std::sync::Arc;

fn create_pipeline(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    _swapchain: &nbn::Swapchain,
) -> nbn::Pipeline {
    device.create_compute_pipeline(shader, c"main")
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
    _data_buffer: nbn::Buffer,
    _instances_buffer: nbn::Buffer,
    tlas: nbn::AccelerationStructure,
    _accel: nbn::AccelerationStructure,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
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

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 16 * 1024 * 1024, nbn::QueueType::Compute);

        let indices = [0_u16, 1, 2];
        let triangle = [1.0_f32, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0];

        let mut data = nbn::cast_slice::<_, u8>(&triangle).to_vec();
        data.extend_from_slice(nbn::cast_slice(&indices));
        let data_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "data",
            data: &data,
        });

        let accel = device.create_acceleration_structure(
            "triangles",
            nbn::AccelerationStructureData::Triangles {
                index_type: vk::IndexType::UINT16,
                num_vertices: 3,
                indices_buffer_address: *data_buffer + 3 * 4 * 3,
                vertices_buffer_address: *data_buffer,
                opaque: true,
                num_indices: 3,
            },
            &mut staging_buffer,
        );

        let instances = &[vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: glam::Mat4::IDENTITY.transpose().to_cols_array()[..12]
                    .try_into()
                    .unwrap(),
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: *accel,
            },
        }];

        //staging_buffer.finish(&device);
        //let mut staging_buffer = nbn::StagingBuffer::new(&device, 16 * 1024 * 1024);

        let instance_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "Instances",
            data: instances,
        });

        let tlas = device.create_acceleration_structure(
            "tlas",
            nbn::AccelerationStructureData::Instances {
                buffer_address: *instance_buffer,
                count: 1,
            },
            &mut staging_buffer,
        );

        staging_buffer.finish(&device);

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            nbn::SurfaceSelectionCriteria {
                write_via_compute: true,
                desire_hdr: false,
            },
        );

        let shader = device.load_reloadable_shader("shaders/compiled/rt_triangle.spv");
        let pipeline = create_pipeline(&device, &shader, &swapchain);

        let egui_ctx = egui::Context::default();

        self.window_state = Some(WindowState {
            _accel: accel,
            _data_buffer: data_buffer,
            tlas,
            _instances_buffer: instance_buffer,
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
            swapchain_image_heap_indices: swapchain
                .images
                .iter()
                .map(|image| device.register_image(*image.view, true))
                .collect(),

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

                    device.cmd_pipeline_barrier2(
                        **command_buffer,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            nbn::NewImageBarrier {
                                image,
                                src: Some(nbn::BarrierOp::Acquire),
                                dst: nbn::BarrierOp::ComputeStorageWrite,
                                src_queue_family_index: command_buffer.queue_family_index,
                                dst_queue_family_index: command_buffer.queue_family_index,
                            }
                            .into(),
                        ]),
                    );
                    let extent = state.swapchain.create_info.image_extent;

                    device.bind_internal_descriptor_sets_to_all(command_buffer);

                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *state.pipeline,
                    );

                    let view = glam::Mat4::look_at_rh(
                        glam::Vec3::new(0.0, 0.0, 2.5),
                        glam::Vec3::ZERO,
                        glam::Vec3::Y,
                    );
                    let proj = nbn::perspective_reversed_infinite_z_vk(
                        59.0_f32.to_radians(),
                        extent.width as f32 / extent.height as f32,
                        0.001,
                    );

                    device.push_constants::<(glam::Mat4, glam::Mat4, u64, vk::Extent2D, u32)>(
                        command_buffer,
                        (
                            view.inverse(),
                            proj.inverse(),
                            *state.tlas,
                            extent,
                            *state.swapchain_image_heap_indices[next_image as usize],
                        ),
                    );

                    device.cmd_dispatch(
                        **command_buffer,
                        extent.width.div_ceil(8),
                        extent.height.div_ceil(8),
                        1,
                    );

                    device.cmd_pipeline_barrier2(
                        **command_buffer,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            nbn::NewImageBarrier {
                                image,
                                src: Some(nbn::BarrierOp::ComputeStorageWrite),
                                dst: nbn::BarrierOp::ColorAttachmentReadWrite,
                                src_queue_family_index: command_buffer.queue_family_index,
                                dst_queue_family_index: command_buffer.queue_family_index,
                            }
                            .into(),
                        ]),
                    );

                    device.begin_rendering(
                        command_buffer,
                        extent.width,
                        extent.height,
                        &[vk::RenderingAttachmentInfo::default()
                            .image_view(*image.view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
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
                    );

                    device.cmd_end_rendering(**command_buffer);

                    device.cmd_pipeline_barrier2(
                        **command_buffer,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            nbn::NewImageBarrier {
                                image,
                                src: Some(nbn::BarrierOp::ColorAttachmentWrite),
                                dst: nbn::BarrierOp::Present,
                                src_queue_family_index: command_buffer.queue_family_index,
                                dst_queue_family_index: command_buffer.queue_family_index,
                            }
                            .into(),
                        ]),
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
