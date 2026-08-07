use ash::vk;

slang_struct::slang_include!("shaders/splats/structs.slang");

struct State {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    splat: nbn::Pipeline,
    output: nbn::Pipeline,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    device: nbn::Device,
    freecam: nbn::freecam::FreeCam,
    frame_index: u32,
    buffer: nbn::Buffer,
    splats: nbn::Buffer,
    num_splats: u32,
}

const NEAR_PLANE: f32 = 0.001;

struct App {
    state: Option<State>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let filename = std::env::args().nth(1).unwrap();

        let freecam = nbn::freecam::FreeCam::new([10.0, 10.0, 10.0].into(), NEAR_PLANE);

        let splats = {
            let mut buf_read = std::io::BufReader::new(std::fs::File::open(&filename).unwrap());
            let p = ply_rs::parser::Parser::<ply::DefaultElement>::new();
            let header = p.read_header(&mut buf_read).unwrap();
            dbg!(&header);
            let mut remaining = Vec::new();
            buf_read.read_to_end(&mut remaining).unwrap();
            nbn::cast_slice::<_, PlySplat>(&remaining).to_vec()
        };

        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 1024 * 1024 * 1024, nbn::QueueType::Compute);

        let splats: Vec<_> = splats.iter().map(|s| Splat {
            center: s.xyz,
            dc: s.f_dc,
            scale: s.scale,
            rot: s.rot
        }).collect();

        let num_splats = splats.len() as u32;

        let splats = staging_buffer.create_buffer_from_slice(&device,
            "splats",
            &splats,
        );

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
            splat: device.create_compute_pipeline(&shader, c"splat"),
            output: device.create_compute_pipeline(&shader, c"output"),
            window,
            buffer: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "buffer",
                    size: size.width as u64 * size.height as u64 * 8,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            device,
            splats,
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
                state.buffer = device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "buffer",
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

                device.cmd_fill_buffer(**command_buffer, *state.buffer.buffer, 0, vk::WHOLE_SIZE, 0);

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
                        .update(extent.width, extent.height, 1.0 / 60.0, 1.0);

                device.push_constants::<PushConstants>(
                    command_buffer,
                    PushConstants {
                        camera: (proj * view).to_cols_array(),
                        view: view.to_cols_array(),
                        tan_fov: [0.0;2],
                        extent: [extent.width, extent.height],
                        image: *state.swapchain_image_heap_indices[next_image as usize],
                        frame_index: state.frame_index,
                        bitmasks: *state.buffer,
                        splats: *state.splats,
                        num_splats: state.num_splats,
                    },
                );

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.splat,
                );


                device.cmd_dispatch(
                    **command_buffer,
                    state.num_splats.div_ceil(64),
                    1,
                    1,
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

use ply_rs::ply;
use std::io::Read;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PlySplat {
    xyz: [f32; 3],
    f_dc: [f32; 3],
    f_rest: [f32; 45],
    opacity: f32,
    scale: [f32; 3],
    rot: [f32; 4],
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.run_app(&mut App { state: None }).unwrap();
}
