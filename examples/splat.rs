use ash::vk;

slang_struct::slang_include!("shaders/splats/structs.slang");

struct State {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    splat: nbn::Pipeline,
    output: nbn::Pipeline,
    reset: nbn::Pipeline,
    point: nbn::Pipeline,
    setup_dispatch: nbn::Pipeline,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    device: nbn::Device,
    freecam: nbn::freecam::FreeCam,
    frame_index: u32,
    render_bitmasks: nbn::Buffer,
    splat_chunks: Vec<nbn::Buffer>,
    chunk_addresses: nbn::Buffer,
    dispatch: nbn::Buffer,
    splat_data: nbn::Buffer,
    point_to_splat: nbn::Buffer,
    num_splats: u32,
}

const NEAR_PLANE: f32 = 0.001;

struct App {
    state: Option<State>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let filename = std::env::args().nth(1).unwrap();

        let freecam = nbn::freecam::FreeCam::new([2.0, 10.0, 10.0].into(), NEAR_PLANE);

        let splats = std::fs::read(&filename).unwrap();
        let splats = nbn::cast_slice::<_, Splat>(&splats);

        let window = event_loop
            .create_window(
                winit::window::WindowAttributes::default()
                    .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
                    .with_resizable(true),
            )
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 1024 * 1024 * 1024, nbn::QueueType::Compute);

        let splat_chunks: Vec<nbn::Buffer> = splats.chunks(80_000_000).map(|chunk| {
            staging_buffer.create_buffer_from_slice(&device, "splats chunk", chunk)
        }).collect();

        let num_splats = splats.len() as u32;

        let chunk_addresses = staging_buffer.create_buffer_from_slice(&device, "chunk addresses", &splat_chunks.iter().map(|chunk| **chunk).collect::<Vec<_>>());

        staging_buffer.finish(&device);

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
                desire_hdr: false,
            },
        );

        let shader = device.load_shader("shaders/compiled/splats.spv");
        let size = window.inner_size();

        self.state = Some(State {
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
            freecam,
            swapchain,
            reset: device.create_compute_pipeline(&shader, c"reset"),
            splat: device.create_compute_pipeline(&shader, c"splat"),
            point: device.create_compute_pipeline(&shader, c"point"),
            output: device.create_compute_pipeline(&shader, c"output"),
            setup_dispatch: device.create_compute_pipeline(&shader, c"setup_dispatch"),
            window,
            render_bitmasks: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "render_bitmasks",
                    size: size.width as u64 * size.height as u64 * 8,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            splat_data: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "per_splat_data",
                    size: 30_000_000 * std::mem::size_of::<PerSplatData>() as u64,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            point_to_splat: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "point_to_splat",
                    size: 300_000_000 * 4,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            dispatch: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "dispatch",
                    size: std::mem::size_of::<Dispatch>() as _,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
                chunk_addresses,
            device,
            splat_chunks,
            frame_index: 0,
            num_splats,
        });
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.freecam.handle_device_event(event);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        let device = &state.device;

        if state.freecam.handle_window_event(&state.window, &event) {
            return;
        }

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
                state.render_bitmasks = device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "render_bitmasks",
                        size: new_size.width as u64 * new_size.height as u64 * 8,
                        ty: nbn::MemoryLocation::GpuOnly,
                    })
                    .unwrap();
            }
            winit::event::WindowEvent::RedrawRequested => unsafe {
                let (frame, current_frame) = state.sync_resources.wait_for_frame(device);
                let command_buffer = &state.per_frame_command_buffers[current_frame];

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

                // device.cmd_fill_buffer(
                //     **command_buffer,
                //     *state.buffer.buffer,
                //     0,
                //     vk::WHOLE_SIZE,
                //     0,
                // );

                device.insert_image_pipeline_barrier(
                    command_buffer,
                    image,
                    Some(nbn::BarrierOp::Acquire),
                    nbn::BarrierOp::ComputeStorageWrite,
                );
                let extent = state.swapchain.create_info.image_extent;

                device.bind_internal_descriptor_sets_to_all(command_buffer);

                let (view, proj) =
                    state
                        .freecam
                        .update(extent.width, extent.height, 1.0 / 60.0, 10.0);

                // must match FreeCam's fovy (59.0 deg)
                let tan_y = (59.0_f32.to_radians() * 0.5).tan();

                device.push_constants::<PushConstants>(
                    command_buffer,
                    PushConstants {
                        camera: (proj * view).to_cols_array(),
                        view: view.to_cols_array(),
                        tan_y,
                        extent: [extent.width, extent.height],
                        image: *state.swapchain_image_heap_indices[next_image as usize],
                        frame_index: state.frame_index,
                        bitmasks: *state.render_bitmasks,
                        splats: *state.chunk_addresses,
                        num_splats: state.num_splats,
                        dispatch: *state.dispatch,
                        splat_data: *state.splat_data,
                        point_to_splat: *state.point_to_splat,
                    },
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.reset,
                );

                device.insert_pipeline_barriers(
                    &command_buffer,
                    [],
                    [(
                        &state.splat_data,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.point_to_splat,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.render_bitmasks,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.dispatch,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    )],
                );


                device.cmd_dispatch(**command_buffer, 1, 1, 1);

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.splat,
                );

                device.cmd_dispatch(**command_buffer, state.num_splats.div_ceil(64), 1, 1);

                device.insert_pipeline_barriers(
                    &command_buffer,
                    [],
                    [(
                        &state.splat_data,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.point_to_splat,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.render_bitmasks,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.dispatch,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    )],
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.setup_dispatch,
                );

              device.insert_pipeline_barriers(
                    &command_buffer,
                    [],
                    [(
                        &state.splat_data,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.point_to_splat,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.render_bitmasks,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.dispatch,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    )],
                );

                device.cmd_dispatch(**command_buffer, 1, 1, 1);

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.point,
                );

                device.cmd_dispatch_indirect(**command_buffer, *state.dispatch.buffer, 0);

              device.insert_pipeline_barriers(
                    &command_buffer,
                    [],
                    [(
                        &state.splat_data,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.point_to_splat,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.render_bitmasks,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    ),
                    (
                        &state.dispatch,
                        nbn::BarrierOp::AllCommands,
                        nbn::BarrierOp::AllCommands,
                    )],
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.output,
                );

                device.cmd_dispatch(
                    **command_buffer,
                    extent.width.div_ceil(8),
                    extent.height.div_ceil(8),
                    1,
                );

                device.insert_image_pipeline_barrier(
                    command_buffer,
                    image,
                    Some(nbn::BarrierOp::ComputeStorageWrite),
                    nbn::BarrierOp::Present,
                );
                device.end_command_buffer(**command_buffer).unwrap();

                state.sync_resources.submit_current_frame(
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
        let state = self.state.as_mut().unwrap();
        let device = &state.device;

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
