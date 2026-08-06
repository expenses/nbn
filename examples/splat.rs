use ash::vk;
use std::fs;

slang_struct::slang_include!("shaders/splats/structs.slang");

fn create_pipeline(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    _swapchain: &nbn::Swapchain,
) -> nbn::Pipeline {
    device.create_compute_pipeline(shader, c"main")
}

struct State {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    shader: nbn::ReloadableShader,
    tlas: nbn::AccelerationStructure,
    _accel: nbn::AccelerationStructure,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    device: nbn::Device,
    freecam: nbn::freecam::FreeCam,
    colours: nbn::Buffer,
    frame_index: u32,
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
            let x = buf_read.read_to_end(&mut remaining).unwrap();
            nbn::cast_slice::<_, PlySplat>(&remaining).to_vec()
        };

        let mut aabbs = Vec::new();
        let mut colours = Vec::new();

        for splat in splats {
            let pos = nbn::glam::Vec3::from(splat.xyz);
            let scale = nbn::glam::Vec3::from(splat.scale.map(|v| v.exp()));
            let rot = nbn::glam::Quat::from_array(splat.rot).normalize();

            // object-space AABB of the oriented ±1σ ellipsoid
            let rm = nbn::glam::Mat3::from_quat(rot);
            let extent =
                scale.x * rm.col(0).abs() + scale.y * rm.col(1).abs() + scale.z * rm.col(2).abs();
            aabbs.push(vk::AabbPositionsKHR {
                min_x: pos.x - extent.x,
                min_y: pos.y - extent.y,
                min_z: pos.z - extent.z,
                max_x: pos.x + extent.x,
                max_y: pos.y + extent.y,
                max_z: pos.z + extent.z,
            });

            colours.push(Splat {
                dc: splat.f_dc,
                opacity_factor: (1.0 + (-splat.opacity).exp()).recip(),
                sh1_0: [splat.f_rest[0], splat.f_rest[15], splat.f_rest[30]],
                sh1_1: [splat.f_rest[1], splat.f_rest[16], splat.f_rest[31]],
                sh1_2: [splat.f_rest[2], splat.f_rest[17], splat.f_rest[32]],
                center: pos.into(),
                inv_scale_2: (scale * scale).recip().into(),
                rot: rot.conjugate().into(),
            });
        }

        dbg!(
            std::mem::size_of_val(&colours[..]),
            colours.len(),
            std::mem::size_of::<Splat>()
        );

        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 1024 * 1024 * 1024, nbn::QueueType::Compute);

        let aabbs_count = aabbs.len() as u32;

        let aabbs = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "aabbs",
            data: &aabbs,
        });

        let accel = device.create_acceleration_structure(
            "blas",
            nbn::AccelerationStructureData::Aabbs {
                buffer_address: *aabbs,
                count: aabbs_count,
            },
            &mut staging_buffer,
        );

        let mut instances = Vec::new();

        for i in 0..1 {
            //100 * 100 * 100 {
            instances.push(
                nbn::AccelerationStructureInstance {
                    acceleration_structure: *accel,
                    transform: glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(-100.0),
                        glam::Quat::from_rotation_y(i as f32),
                        glam::Vec3::new(
                            ((i / 100) % 100) as f32 * 40.0,
                            (i / 10_000) as f32 * 40.0,
                            (i % 100) as f32 * 40.0,
                        ),
                    ),
                    ..Default::default()
                }
                .to_vk(),
            )
        }

        //staging_buffer.finish(&device);
        //let mut staging_buffer = nbn::StagingBuffer::new(&device, 16 * 1024 * 1024);

        let instance_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "Instances",
            data: &instances,
        });

        let tlas = device.create_acceleration_structure(
            "tlas",
            nbn::AccelerationStructureData::Instances {
                buffer_address: *instance_buffer,
                count: instances.len() as _,
            },
            &mut staging_buffer,
        );

        
        let colours = staging_buffer.create_buffer_from_slice(&device,
            "colours",
            &colours,
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

        let shader = device.load_reloadable_shader("shaders/compiled/splats.spv");
        let pipeline = create_pipeline(&device, &shader, &swapchain);

        self.state = Some(State {
            _accel: accel,
            colours,
            tlas,
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
            pipeline,
            shader,
            window,
            device,
            frame_index: 0,
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
            }
            winit::event::WindowEvent::RedrawRequested => {
                if state.shader.try_reload(device) {
                    unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                    state.pipeline = create_pipeline(device, &state.shader, &state.swapchain);
                }

                unsafe {
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
                        .begin_command_buffer(
                            **command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();

                    device.insert_image_pipeline_barrier(
                        command_buffer,
                        image,
                        Some(nbn::BarrierOp::Acquire),
                        nbn::BarrierOp::ComputeStorageWrite,
                    );
                    let extent = state.swapchain.create_info.image_extent;

                    device.bind_internal_descriptor_sets_to_all(command_buffer);

                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *state.pipeline,
                    );

                    let (view, proj) =
                        state
                            .freecam
                            .update(extent.width, extent.height, 1.0 / 60.0, 1.0);

                    device.push_constants::<PushConstants>(
                        command_buffer,
                        PushConstants {
                            view_inv: view.inverse().to_cols_array(),
                            proj_inv: proj.inverse().to_cols_array(),
                            tlas: *state.tlas,
                            extent: [extent.width, extent.height],
                            image: *state.swapchain_image_heap_indices[next_image as usize],
                            splats: *state.colours,
                            frame_index: state.frame_index,
                        },
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
