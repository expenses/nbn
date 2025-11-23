use ash::vk;
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

slang_struct::slang_include!("shaders/thief_models.slang");

fn create_hdr(device: &nbn::Device, extent: vk::Extent2D) -> nbn::IndexedImage {
    let image = device.create_image(nbn::ImageDescriptor {
        name: "hdrbuffer",
        format: vk::Format::R16G16B16A16_SFLOAT,
        extent: extent.into(),
        usage: vk::ImageUsageFlags::STORAGE,
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_levels: 1,
    });

    let index = device.register_image(*image.view, true);

    nbn::IndexedImage { image, index }
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer,
    _data_buffer: nbn::Buffer,
    _instances_buffer: nbn::Buffer,
    model_buffer: nbn::Buffer,
    tlas: nbn::AccelerationStructure,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    camera_rig: dolly::rig::CameraRig,
    cursor_grabbed: bool,
    keyboard: KeyboardState,
    gltf_data: CombinedModel,
    lights: nbn::Buffer,
    num_lights: usize,
    lights_pipeline: nbn::Pipeline,
    hdr: nbn::IndexedImage,
    resolve_pipeline: nbn::Pipeline,
    frame_index: u32,
    accum_index: u32,
    accum: bool,
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

        let (gltf_data, model, lights) = load_gltf(
            &device,
            &mut staging_buffer,
            &std::path::Path::new("./models/export/assassins.gltf"),
        );

        let num_lights = dbg!(lights.len());

        let lights = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "lights",
            data: &lights,
        });

        let instance_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "Instances",
            data: &[vk::AccelerationStructureInstanceKHR {
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
        });

        let model_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "models",
            data: &[model],
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
            cull_mode: Default::default(),
        });
        let pipeline = device.create_compute_pipeline(&shader, c"write");

        let egui_ctx = egui::Context::default();

        self.window_state = Some(WindowState {
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
            hdr: create_hdr(&device, swapchain.create_info.image_extent),
            resolve_pipeline: device.create_compute_pipeline(&shader, c"resolve"),
            swapchain,
            pipeline,
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
                .with(dolly::drivers::Position::new([0.0, 0.0, 2.5]))
                .with(dolly::drivers::YawPitch::new())
                .with(dolly::drivers::Smooth::new_position_rotation(1.0, 1.0))
                .build(),
            cursor_grabbed: false,
            keyboard: Default::default(),
            gltf_data,
            lights,
            num_lights,
            accum_index: 0,
            frame_index: 0,
            accum: false,
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
                state.hdr = create_hdr(&device, state.swapchain.create_info.image_extent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                let raw_input = state.egui_winit.take_egui_input(&state.window);

                let egui_ctx = state.egui_winit.egui_ctx();

                egui_ctx.begin_pass(raw_input);
                {
                    let allocator = device.allocator.inner.read();
                    egui::Window::new("Memory Allocations").show(egui_ctx, |ui| {
                        ui.checkbox(&mut state.accum, "accum");
                    });
                    //    state.alloc_vis.render_breakdown_ui(ui, &allocator);
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

                    device.push_constants::<(
                        glam::Mat4,
                        glam::Mat4,
                        glam::Mat4,
                        u64,
                        u64,
                        u64,
                        vk::Extent2D,
                        u32,
                        u32,
                        u32,
                        u32,
                        u32,
                    )>(
                        command_buffer,
                        (
                            view.inverse(),
                            proj.inverse(),
                            (proj * view),
                            *state.tlas,
                            *state.model_buffer,
                            *state.lights,
                            extent,
                            *state.hdr,
                            *state.swapchain_image_heap_indices[next_image as usize],
                            state.num_lights as _,
                            state.accum_index,
                            state.frame_index,
                        ),
                    );

                    device.cmd_dispatch(
                        **command_buffer,
                        extent.width.div_ceil(8),
                        extent.height.div_ceil(8),
                        1,
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

pub fn load_gltf(
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

    let mut material_to_image: Vec<u32> = gltf
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

    fn get_slice<'a, T: bytemuck::Pod>(
        buffer: &'a [u8],
        gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        accessor: &goth_gltf::Accessor,
    ) -> &'a [T] {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        &bytemuck::cast_slice(&buffer[bv.byte_offset + accessor.byte_offset..])
    };

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

    //let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut image_indices = Vec::new();
    //let mut meshes = Vec::new();

    for (mesh_index, mesh) in gltf.meshes.iter().enumerate() {
        for (primitive_index, (primitive)) in mesh.primitives.iter().enumerate() {
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
    let indices =
        staging_buffer.create_buffer_from_slice(device, "indices", bytemuck::cast_slice(&indices));
    let positions = staging_buffer.create_buffer_from_slice(
        device,
        "positions",
        bytemuck::cast_slice(&positions),
    );
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

    let uvs = staging_buffer.create_buffer_from_slice(device, "uvs", bytemuck::cast_slice(&uvs));
    let normals =
        staging_buffer.create_buffer_from_slice(device, "normals", bytemuck::cast_slice(&normals));
    let image_indices = staging_buffer.create_buffer_from_slice(
        device,
        "image_indices",
        bytemuck::cast_slice(&image_indices),
    );

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
            positions,
            indices,
            uvs,
            normals,
            image_indices,
            images,
        },
        model,
        lights,
    )
}

struct CombinedModel {
    acceleration_structure: nbn::AccelerationStructure,
    positions: nbn::Buffer,
    indices: nbn::Buffer,
    uvs: nbn::Buffer,
    normals: nbn::Buffer,
    image_indices: nbn::Buffer,
    images: Vec<nbn::IndexedImage>,
}
