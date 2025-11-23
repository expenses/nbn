use ash::vk;
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

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
    _accel: Vec<(nbn::AccelerationStructure, u32)>,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    camera_rig: dolly::rig::CameraRig,
    cursor_grabbed: bool,
    keyboard: KeyboardState,
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

        let (accel, temp_buffer) = load_gltf(
            &device,
            &mut staging_buffer,
            &std::path::Path::new("./models/export/training.gltf"),
        );

        let instances: Vec<_> = accel
            .iter()
            .map(
                |(accel, material_index)| vk::AccelerationStructureInstanceKHR {
                    transform: vk::TransformMatrixKHR {
                        matrix: glam::Mat4::IDENTITY.transpose().to_cols_array()[..12]
                            .try_into()
                            .unwrap(),
                    },
                    instance_custom_index_and_mask: vk::Packed24_8::new(*material_index, 0xff),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0,
                        ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                            as _,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: **accel,
                    },
                },
            )
            .collect();

        //let instances = &[];

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

        staging_buffer.finish(&device);

        drop(temp_buffer);

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
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
            camera_rig: dolly::rig::CameraRig::builder()
                .with(dolly::drivers::Position::new([0.0, 0.0, 2.5]))
                .with(dolly::drivers::YawPitch::new())
                .with(dolly::drivers::Smooth::new_position_rotation(1.0, 1.0))
                .build(),
            cursor_grabbed: false,
            keyboard: Default::default(),
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

                    device.cmd_bind_pipeline(
                        **command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *state.pipeline,
                    );

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

pub fn load_gltf(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> (Vec<(nbn::AccelerationStructure, u32)>, nbn::Buffer) {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let mut image_formats = vec![vk::Format::R8G8B8A8_UNORM; gltf.images.len()];

    for material in &gltf.materials {
        if let Some(tex) = &material.emissive_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
        if let Some(tex) = &material.pbr_metallic_roughness.base_color_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
    }

    let buffer_file =
        std::fs::File::open(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();
    let buffer = staging_buffer.create_buffer(
        device,
        &format!("{} buffer", path.display()),
        buffer_file.metadata().unwrap().len() as _,
        buffer_file,
    );

    /*let images = gltf
        .images
        .iter()
        .zip(&image_formats)
        .map(|(image, _format)| {
            let image = create_image(
                device,
                staging_buffer,
                path.with_file_name(image.uri.as_ref().unwrap())
                    .to_str()
                    .unwrap(),
                nbn::QueueType::Graphics,
            );

            nbn::IndexedImage {
                index: device.register_image(*image.view, false),
                image,
            }
        })
        .collect::<Vec<_>>();

    dbg!(images.len());

    let materials: Vec<Material> = gltf
        .materials
        .iter()
        .map(|material| Material {
            base_colour_image: material
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            metallic_roughness_image: material
                .pbr_metallic_roughness
                .metallic_roughness_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            normal_image: material
                .normal_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            emissive_image: material
                .emissive_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            flags: !matches!(material.alpha_mode, goth_gltf::AlphaMode::Opaque) as u32,
        })
        .collect();*/

    let get_buffer_offset = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        (*buffer) + bv.byte_offset as u64 + accessor.byte_offset as u64
    };

    let mut meshes = Vec::new();

    for (mesh_index, mesh) in gltf.meshes.iter().enumerate() {
        for (primitive_index, (primitive)) in mesh.primitives.iter().enumerate() {
            let material_index = primitive.material.unwrap();
            //let material = materials[primitive.material.unwrap()];

            let get = |accessor_index: Option<usize>| {
                let accessor = &gltf.accessors[accessor_index.unwrap()];
                assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                get_buffer_offset(accessor)
            };

            let indices = &gltf.accessors[primitive.indices.unwrap()];
            let is_32_bit = match indices.component_type {
                goth_gltf::ComponentType::UnsignedShort => false,
                goth_gltf::ComponentType::UnsignedInt => true,
                other => unimplemented!("{:?}", other),
            };

            let positions_accessor = &gltf.accessors[primitive.attributes.position.unwrap()];

            let positions = get(primitive.attributes.position);

            let acceleration_structure = device.create_acceleration_structure(
                &format!(
                    "{} mesh {} primitive {} acceleration structure",
                    path.display(),
                    mesh_index,
                    primitive_index
                ),
                nbn::AccelerationStructureData::Triangles {
                    index_type: if is_32_bit {
                        vk::IndexType::UINT32
                    } else {
                        vk::IndexType::UINT16
                    },
                    opaque: true,
                    vertices_buffer_address: positions,
                    indices_buffer_address: get_buffer_offset(indices),
                    num_vertices: positions_accessor.count as _,
                    num_indices: indices.count as _,
                },
                staging_buffer,
            );

            meshes.push(/*GltfModel*/ {
                (acceleration_structure, material_index as u32) /*,
                                                                model: Model {
                                                                    material,
                                                                    positions,
                                                                    uvs: get(primitive.attributes.texcoord_0),
                                                                    normals: get(primitive.attributes.normal),
                                                                    indices: get_buffer_offset(indices),
                                                                    meshlets: *meshlets_buffer + meshlets_offset as u64,
                                                                    triangles: *meshlets_buffer + triangles_offset as u64,
                                                                    vertices: *meshlets_buffer + vertices_offset as u64,
                                                                    flags: is_32_bit as u32,
                                                                    num_meshlets,
                                                                    num_indices: indices.count as u32,
                                                                },
                                                                //meshlets: nbn::cast_slice::<_, Meshlet>(&meshlets.data[meshlets_offset as usize..])
                                                                //    [..num_meshlets as usize]
                                                                //    .to_vec(), */
            });
        }
    }

    //dbg!(&materials.len());

    (meshes, buffer) //GltfData {
                     //    _images: images,
                     //    _buffer: buffer,
                     //    _meshlets_buffer: meshlets_buffer,
                     //    meshes,
                     //}
}
