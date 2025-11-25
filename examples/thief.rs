use ash::vk;
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

slang_struct::slang_include!("shaders/thief_structs.slang");

struct Images {
    hdr: nbn::IndexedImage,
    prims: nbn::IndexedImage,
    depth: nbn::Image,
}

impl Images {
    fn new(device: &nbn::Device, extent: vk::Extent2D) -> Self {
        Self {
            hdr: device.register_owned_image(
                device.create_image(nbn::ImageDescriptor {
                    name: "hdrbuffer",
                    format: vk::Format::R16G16B16A16_SFLOAT,
                    extent: extent.into(),
                    usage: vk::ImageUsageFlags::STORAGE,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_levels: 1,
                }),
                true,
            ),
            prims: device.register_owned_image(
                device.create_image(nbn::ImageDescriptor {
                    name: "primsbuffer",
                    format: vk::Format::R32_UINT,
                    extent: extent.into(),
                    usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_levels: 1,
                }),
                true,
            ),
            depth: device.create_image(nbn::ImageDescriptor {
                name: "depthbuffer",
                format: vk::Format::D32_SFLOAT,
                extent: extent.into(),
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_levels: 1,
            }),
        }
    }
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer,
    _instances_buffer: nbn::Buffer,
    model_buffer: nbn::Buffer,
    tlas: nbn::AccelerationStructure,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    camera_rig: dolly::rig::CameraRig,
    cursor_grabbed: bool,
    keyboard: KeyboardState,
    _gltf_data: CombinedModel,
    lights: nbn::Buffer,
    num_lights: usize,
    lights_pipeline: nbn::Pipeline,
    mesh_pipeline: nbn::Pipeline,
    images: Images,
    resolve_pipeline: nbn::Pipeline,
    frame_index: u32,
    accum_index: u32,
    accum: bool,
    blue_noise_buffers: nbn::blue_noise::BlueNoiseBuffers,
    combined_uniform_buffer: nbn::Buffer,
    num_indices: u32,
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

        let (gltf_data, model, lights) = load_gltf(
            &device,
            &mut staging_buffer,
            &std::path::Path::new("./models/export/assassins.gltf"),
        );

        let num_lights = dbg!(lights.len());

        let lights = staging_buffer.create_buffer_from_slice(&device, "lights", &lights);

        let instance_buffer = staging_buffer.create_buffer_from_slice(
            &device,
            "Instances",
            &[vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: glam::Mat4::IDENTITY.transpose().to_cols_array()[..12]
                        .try_into()
                        .unwrap(),
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    ash::vk::GeometryInstanceFlagsKHR::default().as_raw() as _,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: *gltf_data.acceleration_structure,
                },
            }],
        );

        let model_buffer = staging_buffer.create_buffer_from_slice(&device, "models", &[model]);

        let tlas = device.create_acceleration_structure(
            "tlas",
            nbn::AccelerationStructureData::Instances {
                buffer_address: *instance_buffer,
                count: 1,
            },
            &mut staging_buffer,
        );

        let blue_noise_buffers =
            nbn::blue_noise::BlueNoiseBuffers::new(&device, &mut staging_buffer);

        staging_buffer.finish(&device);

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
                desire_hdr: false,
            },
        );

        let shader = device.load_shader("shaders/compiled/thief.spv");

        let lights_pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            name: "lights pipeline",
            shaders: nbn::GraphicsPipelineShaders::Legacy {
                vertex: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"lights_vert",
                },
                fragment: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"lights_frag",
                },
            },
            color_attachment_formats: &[swapchain.create_info.image_format],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            flags: nbn::GraphicsPipelineFlags::POINTS,
            depth: Default::default(),
        });
        let mesh_pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            name: "mesh pipeline",
            shaders: nbn::GraphicsPipelineShaders::Legacy {
                vertex: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"mesh_vert",
                },
                fragment: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"mesh_frag",
                },
            },
            color_attachment_formats: &[vk::Format::R32_UINT],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            flags: Default::default(),
            depth: nbn::GraphicsPipelineDepthDesc {
                write_enable: true,
                test_enable: true,
                compare_op: vk::CompareOp::GREATER,
                format: vk::Format::D32_SFLOAT,
            },
        });
        let egui_ctx = egui::Context::default();

        self.window_state = Some(WindowState {
            tlas,
            mesh_pipeline,
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
            combined_uniform_buffer: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "combined_uniform_buffer",
                    size: (std::mem::size_of::<Uniforms>() * nbn::FRAMES_IN_FLIGHT) as _,
                    ty: nbn::MemoryLocation::CpuToGpu,
                })
                .unwrap(),
            sync_resources: device.create_sync_resources(),
            swapchain_image_heap_indices: swapchain
                .images
                .iter()
                .map(|image| device.register_image(*image.view, true))
                .collect(),
            images: Images::new(&device, swapchain.create_info.image_extent),
            resolve_pipeline: device.create_compute_pipeline(&shader, c"resolve"),
            swapchain,
            blue_noise_buffers,
            lights_pipeline,
            model_buffer,
            egui_winit: egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                event_loop,
                Some(window.scale_factor() as _),
                None,
                None,
            ),
            window,
            camera_rig: dolly::rig::CameraRig::builder()
                .with(dolly::drivers::Position::new([-22.8, 62.0, 46.0]))
                .with(dolly::drivers::YawPitch {
                    pitch_degrees: -45.0,
                    yaw_degrees: -60.0,
                })
                .with(dolly::drivers::Smooth::new_position_rotation(1.0, 1.0))
                .build(),
            cursor_grabbed: false,
            keyboard: Default::default(),
            _gltf_data: gltf_data,
            lights,
            num_lights,
            accum_index: 0,
            frame_index: 0,
            accum: false,
            num_indices: model.num_indices,
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
                state.images = Images::new(&device, state.swapchain.create_info.image_extent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                let raw_input = state.egui_winit.take_egui_input(&state.window);

                let egui_ctx = state.egui_winit.egui_ctx();

                egui_ctx.begin_pass(raw_input);
                {
                    let allocator = device.allocator.inner.read();
                    egui::Window::new(".").show(egui_ctx, |ui| {
                        ui.checkbox(&mut state.accum, "accum");
                        //state.alloc_vis.render_memory_block_ui(ui, &allocator);
                        //state.alloc_vis.render_breakdown_ui(ui, &allocator);
                    });
                    //
                    //    ui.label(format!("{:?}", &device.descriptors.sampled_image_count));
                    //    ui.label(format!("{:?}", &device.descriptors.storage_image_count));
                    //});
                    //egui::Window::new("Memory Blocks").show(egui_ctx, |ui| {
                    //    state.alloc_vis.render_memory_block_ui(ui, &allocator);
                    //});
                    //state
                    //    .alloc_vis
                    //    .render_memory_block_visualization_windows(egui_ctx, &allocator);
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
                            nbn::ImageBarrier {
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

                    let forward = state.keyboard.forwards as i32 - state.keyboard.backwards as i32;
                    let right = state.keyboard.right as i32 - state.keyboard.left as i32;

                    let prev_transform = state.camera_rig.final_transform;

                    state
                        .camera_rig
                        .driver_mut::<dolly::drivers::Position>()
                        .translate(
                            ((glam::Vec3::from_array(prev_transform.forward()) * forward as f32
                                + glam::Vec3::from_array(prev_transform.right()) * right as f32)
                                * 1.0)
                                .to_array(),
                        );

                    let transform = state.camera_rig.update(1.0 / 60.0);

                    let view = glam::Mat4::look_to_rh(
                        glam::Vec3::from_array(transform.position.into()),
                        glam::Vec3::from_array(transform.forward()),
                        glam::Vec3::Y,
                    );
                    let proj = nbn::perspective_reversed_infinite_z_vk(
                        59.0_f32.to_radians(),
                        extent.width as f32 / extent.height as f32,
                        0.001,
                    );

                    let uniforms = state
                        .combined_uniform_buffer
                        .try_as_slice_mut::<Uniforms>()
                        .unwrap();

                    uniforms[current_frame] = Uniforms {
                        view_inv: view.inverse().to_cols_array(),
                        proj_inv: proj.inverse().to_cols_array(),
                        proj_view: (proj * view).to_cols_array(),
                        tlas: *state.tlas,
                        model: *state.model_buffer,
                        light_values: *state.lights,
                        blue_noise_sobol: *state.blue_noise_buffers.sobol,
                        blue_noise_ranking_tile: *state.blue_noise_buffers.ranking_tile,
                        blue_noise_scrambling_tile: *state.blue_noise_buffers.scrambling_tile,
                        accum_index: state.accum_index,
                        frame_index: state.frame_index,
                        extent: [extent.width, extent.height],
                        hdr_image: *state.images.hdr,
                        prims_image: *state.images.prims,
                        num_lights: state.num_lights as _,
                        swapchain_image: *state.swapchain_image_heap_indices[next_image as usize],
                    };

                    let uniforms_ptr = *state.combined_uniform_buffer
                        + (std::mem::size_of::<Uniforms>() * current_frame) as u64;

                    device.push_constants(
                        command_buffer,
                        PushConstants {
                            uniforms: uniforms_ptr,
                        },
                    );

                    device.begin_rendering(
                        command_buffer,
                        extent.width,
                        extent.height,
                        &[vk::RenderingAttachmentInfo::default()
                            .image_view(*state.images.prims.image.view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)],
                        Some(
                            &vk::RenderingAttachmentInfo::default()
                                .image_view(*state.images.depth.view)
                                .image_layout(vk::ImageLayout::GENERAL)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE),
                        ),
                    );
                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *state.mesh_pipeline,
                    );
                    device.cmd_draw(**command_buffer, state.num_indices as _, 1, 0, 0);
                    device.cmd_end_rendering(**command_buffer);

                    device.cmd_pipeline_barrier2(
                        **command_buffer,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            nbn::ImageBarrier {
                                image,
                                src: Some(nbn::BarrierOp::ColorAttachmentReadWrite),
                                dst: nbn::BarrierOp::ComputeStorageWrite,
                                src_queue_family_index: command_buffer.queue_family_index,
                                dst_queue_family_index: command_buffer.queue_family_index,
                            }
                            .into(),
                        ]),
                    );

                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *state.resolve_pipeline,
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
                            nbn::ImageBarrier {
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
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)],
                        None,
                    );
                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *state.lights_pipeline,
                    );
                    device.cmd_draw(**command_buffer, state.num_lights as _, 1, 0, 0);
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

                    device.cmd_pipeline_barrier2(
                        **command_buffer,
                        &vk::DependencyInfo::default().image_memory_barriers(&[
                            nbn::ImageBarrier {
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

                    state.frame_index += 1;
                    if state.accum {
                        state.accum_index += 1;
                    } else {
                        state.accum_index = 0;
                    }
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(code),
                        state: element_state,
                        ..
                    },
                ..
            } => {
                let pressed = element_state == ElementState::Pressed;

                match code {
                    KeyCode::KeyW => state.keyboard.forwards = pressed,
                    KeyCode::KeyS => state.keyboard.backwards = pressed,
                    KeyCode::KeyD => state.keyboard.right = pressed,
                    KeyCode::KeyA => state.keyboard.left = pressed,
                    KeyCode::KeyG if pressed => {
                        state.cursor_grabbed = !state.cursor_grabbed;
                        state
                            .window
                            .set_cursor_grab(if state.cursor_grabbed {
                                CursorGrabMode::Confined
                            } else {
                                CursorGrabMode::None
                            })
                            .unwrap();
                        state.window.set_cursor_visible(!state.cursor_grabbed);
                    }
                    KeyCode::Escape if pressed => {
                        event_loop.exit();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(state) = self.window_state.as_mut() {
            if !state.cursor_grabbed {
                return;
            }

            match event {
                winit::event::DeviceEvent::MouseMotion { delta: (x, y) } => {
                    state
                        .camera_rig
                        .driver_mut::<dolly::prelude::YawPitch>()
                        .rotate_yaw_pitch(-x as f32 / 10.0, -y as f32 / 10.0);
                }
                _ => {}
            }
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

#[derive(Default)]
struct KeyboardState {
    pub forwards: bool,
    pub backwards: bool,
    pub left: bool,
    pub right: bool,
}

fn load_gltf(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> (CombinedModel, Model, Vec<Light>) {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let lights: Vec<_> = gltf
        .nodes
        .iter()
        .filter_map(|node| {
            node.extensions
                .khr_lights_punctual
                .as_ref()
                .map(|ext| (node, ext.light))
        })
        .map(|(node, index)| {
            let pos = if let goth_gltf::NodeTransform::Set { translation, .. } = node.transform() {
                translation
            } else {
                panic!()
            };

            let light = &gltf.extensions.khr_lights_punctual.as_ref().unwrap().lights[index];
            Light {
                position: pos,
                color: light.color,
                intensity: light.intensity,
            }
        })
        .collect();

    let images = gltf
        .images
        .iter()
        //.zip(&images)
        .map(|image| {
            let path = path.with_file_name(image.uri.as_ref().unwrap());
            let data = image::open(&path).unwrap().to_rgba8();
            let image = staging_buffer.create_sampled_image(
                &device,
                nbn::SampledImageDescriptor {
                    name: image.uri.as_ref().unwrap(),
                    extent: vk::Extent3D {
                        width: data.width(),
                        height: data.height(),
                        depth: 1,
                    }
                    .into(),
                    format: vk::Format::R8G8B8A8_SRGB,
                },
                &data,
                nbn::QueueType::Compute,
                &[0],
            );
            nbn::IndexedImage {
                index: device.register_image(*image.view, false),
                image,
            }
        })
        .collect::<Vec<_>>();

    let material_to_image: Vec<u32> = gltf
        .materials
        .iter()
        .map(|mat| {
            let texture_index = mat
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .unwrap()
                .index;
            *images[gltf.textures[texture_index].source.unwrap()].index
        })
        .collect();

    let buffer = std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    fn get_slice<'a, T: Copy>(
        buffer: &'a [u8],
        gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        accessor: &goth_gltf::Accessor,
    ) -> &'a [T] {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        &nbn::cast_slice(&buffer[bv.byte_offset + accessor.byte_offset..])
    }

    let mut indices = Vec::new();
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut image_indices = Vec::new();

    for mesh in gltf.meshes.iter() {
        for primitive in mesh.primitives.iter() {
            let indices_accessor = &gltf.accessors[primitive.indices.unwrap()];
            assert_eq!(
                indices_accessor.component_type,
                goth_gltf::ComponentType::UnsignedShort
            );
            let prim_indices =
                &get_slice::<u16>(&buffer, &gltf, &indices_accessor)[..indices_accessor.count];
            indices.extend(
                prim_indices
                    .iter()
                    .map(|&index| positions.len() as u32 / 3 + index as u32),
            );

            let get = |accessor_index: Option<usize>, size: usize| {
                let accessor = &gltf.accessors[accessor_index.unwrap()];
                assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                &get_slice::<f32>(&buffer, &gltf, accessor)[..accessor.count * size]
            };

            let positions_slice = get(primitive.attributes.position, 3);
            positions.extend_from_slice(positions_slice);
            uvs.extend_from_slice(get(primitive.attributes.texcoord_0, 2));
            normals.extend_from_slice(get(primitive.attributes.normal, 3));

            let material_index = primitive.material.unwrap();

            image_indices
                .extend((0..prim_indices.len() / 3).map(|_| material_to_image[material_index]));
        }
    }

    let num_vertices = positions.len() / 3;
    let num_indices = indices.len();
    let indices = staging_buffer.create_buffer_from_slice(device, "indices", &indices);
    let positions = staging_buffer.create_buffer_from_slice(device, "positions", &positions);
    dbg!(image_indices.len());

    let acceleration_structure = device.create_acceleration_structure(
        &format!("{} acceleration structure", path.display(),),
        nbn::AccelerationStructureData::Triangles {
            index_type: vk::IndexType::UINT32,
            opaque: true,
            vertices_buffer_address: *positions,
            indices_buffer_address: *indices,
            num_vertices: num_vertices as _,
            num_indices: num_indices as _,
        },
        staging_buffer,
    );

    let uvs = staging_buffer.create_buffer_from_slice(device, "uvs", &uvs);
    let normals = staging_buffer.create_buffer_from_slice(device, "normals", &normals);
    let image_indices =
        staging_buffer.create_buffer_from_slice(device, "image_indices", &image_indices);

    let model = Model {
        positions: *positions,
        uvs: *uvs,
        normals: *normals,
        indices: *indices,
        image_indices: *image_indices,
        flags: 1,
        num_indices: num_indices as _,
    };

    (
        CombinedModel {
            acceleration_structure,
            _positions: positions,
            _indices: indices,
            _uvs: uvs,
            _normals: normals,
            _image_indices: image_indices,
            _images: images,
        },
        model,
        lights,
    )
}

struct CombinedModel {
    acceleration_structure: nbn::AccelerationStructure,
    _positions: nbn::Buffer,
    _indices: nbn::Buffer,
    _uvs: nbn::Buffer,
    _normals: nbn::Buffer,
    _image_indices: nbn::Buffer,
    _images: Vec<nbn::IndexedImage>,
}
