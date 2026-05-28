use nbn::vk;
use nbn::winit::{self, event::ElementState, keyboard::KeyCode};

slang_struct::slang_include!("shaders/jungle/structs.slang");

fn main() {
    let filename = std::env::args().nth(1).unwrap();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App {
            state: None,
            filepath: filename,
        })
        .unwrap();
}

const NEAR_PLANE: f32 = 0.001;

struct State {
    device: nbn::Device,
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    render_pipeline: nbn::Pipeline,
    reset: nbn::Pipeline,
    resolve: nbn::Pipeline,
    raytrace: nbn::Pipeline,
    prefix_sum_instances: nbn::Pipeline,
    prefix_sum_data: nbn::Buffer,
    visible_meshlets: nbn::Buffer,
    dispatch: nbn::Buffer,
    _data: LoadedData,
    freecam: nbn::freecam::FreeCam,
    frame_index: u32,
    images: Images,
    view: [f32; 16],
    update_view: bool,
    camera_pos: [f32; 3],
    uniform_buffers: [nbn::Buffer; nbn::FRAMES_IN_FLIGHT],
}

struct Images {
    depth: nbn::Image,
    vis: nbn::IndexedImage,
}

impl Images {
    fn new(device: &nbn::Device, width: u32, height: u32) -> Self {
        Self {
            depth: device.create_image(nbn::ImageDescriptor {
                name: "depth attachment",
                format: vk::Format::D32_SFLOAT,
                extent: nbn::ImageExtent::D2 { width, height },
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_levels: 1,
            }),
            vis: device.register_owned_image(
                device.create_image(nbn::ImageDescriptor {
                    name: "vis",
                    format: vk::Format::R32_UINT,
                    extent: nbn::ImageExtent::D2 { width, height },
                    usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_levels: 1,
                }),
                true,
            ),
        }
    }
}

struct App {
    state: Option<State>,
    filepath: String,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window));

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            nbn::SurfaceSelectionCriteria {
                force_8_bit: false,
                desire_hdr: false,
            },
        );

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 64 * 1024 * 1024, nbn::QueueType::Compute);

        let data = load_gltf(&device, &mut staging_buffer, &self.filepath);

        staging_buffer.finish(&device);

        let shader = device.load_shader("shaders/compiled/jungle.spv");

        let render_pipeline = device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            name: "triangle pipeline",
            shaders: nbn::GraphicsPipelineShaders::Task {
                task: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"task",
                },
                mesh: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"mesh",
                },
                fragment: nbn::ShaderDesc {
                    module: &shader,
                    entry_point: c"pixel",
                },
            },
            color_attachment_formats: &[vk::Format::R32_UINT],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            flags: nbn::GraphicsPipelineFlags::BACKFACE_CULLING,
            depth: nbn::GraphicsPipelineDepthDesc {
                write_enable: true,
                test_enable: true,
                compare_op: vk::CompareOp::GREATER,
                format: vk::Format::D32_SFLOAT,
            },
        });

        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        let create_buffer = |name, size| {
            device
                .create_buffer(nbn::BufferDescriptor {
                    name,
                    size,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap()
        };

        self.state = Some(State {
            prefix_sum_data: create_buffer("prefix_sum_data", 8 + 8 * data.num_instances as u64),
            dispatch: create_buffer("dispatch", 12),
            uniform_buffers: std::array::from_fn(|i| {
                device
                    .create_buffer(nbn::BufferDescriptor {
                        name: &format!("uniform_buffer_{}", i),
                        size: std::mem::size_of::<Uniforms>() as _,
                        ty: nbn::MemoryLocation::CpuToGpu,
                    })
                    .unwrap()
            }),
            visible_meshlets: create_buffer("visible_meshlets", 4),
            prefix_sum_instances: device.create_compute_pipeline(&shader, c"prefix_sum_instances"),
            reset: device.create_compute_pipeline(&shader, c"reset"),
            resolve: device.create_compute_pipeline(&shader, c"resolve"),
            raytrace: device.create_compute_pipeline(&shader, c"raytrace"),
            images: Images::new(&device, width, height),
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

            device,
            window,
            render_pipeline,
            swapchain,
            _data: data,
            freecam: nbn::freecam::FreeCam::new([0.01; 3].into(), NEAR_PLANE),
            frame_index: 0,
            view: [0.0; 16],
            update_view: true,
            camera_pos: [0.0; 3],
        })
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
                state.images = Images::new(device, new_size.width, new_size.height);
            }
            winit::event::WindowEvent::RedrawRequested => unsafe {
                let extent = state.swapchain.create_info.image_extent;

                let (view, proj) =
                    state
                        .freecam
                        .update(extent.width, extent.height, 1.0 / 60.0, 5.0);

                let frustum_x = (proj.row(3).truncate() + proj.row(0).truncate()).normalize();
                let frustum_y = (proj.row(3).truncate() + proj.row(1).truncate()).normalize();

                if state.update_view {
                    state.view = view.to_cols_array();
                    state.camera_pos = state.freecam.camera_rig.final_transform.position.into();
                }

                let (frame, frame_index) = state.sync_resources.wait_for_frame(device);

                let (next_image, _suboptimal) = device
                    .swapchain_loader
                    .acquire_next_image(
                        *state.swapchain,
                        !0,
                        *frame.image_available_semaphore,
                        vk::Fence::null(),
                    )
                    .unwrap();

                let command_buffer = &state.per_frame_command_buffers[frame_index];

                let image = &state.swapchain.images[next_image as usize];

                state.uniform_buffers[frame_index]
                    .try_as_slice_mut::<Uniforms>()
                    .unwrap()[0] = Uniforms {
                    camera: (proj * view).to_cols_array(),
                    num_instances: state._data.num_instances,
                    vis: *state.images.vis,
                    extent: [extent.width, extent.height],
                    camera_pos: state.camera_pos,
                    near_plane: NEAR_PLANE,
                    frustum: [frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z],
                    view: state.view,
                    view_inv: view.inverse().to_cols_array(),
                    proj_inv: proj.inverse().to_cols_array(),
                };

                device.reset_command_buffer(command_buffer);
                device
                    .begin_command_buffer(**command_buffer, &vk::CommandBufferBeginInfo::default())
                    .unwrap();

                device.bind_internal_descriptor_sets_to_all(command_buffer);

                raytrace(state, next_image, frame_index);

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
                        physical_key: winit::keyboard::PhysicalKey::Code(code),
                        state: element_state,
                        ..
                    },
                ..
            } => {
                let pressed = element_state == ElementState::Pressed;

                match code {
                    KeyCode::KeyV if pressed => {
                        state.update_view = !state.update_view;
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

fn raster(state: &State, next_image: u32, current_frame: usize) {
    let device = &state.device;
    let command_buffer = &state.per_frame_command_buffers[current_frame];
    let extent = state.swapchain.create_info.image_extent;
    let image = &state.swapchain.images[next_image as usize];

    unsafe {
        device.push_constants::<PushConstants>(
            command_buffer,
            PushConstants {
                tlas: *state._data.tlas.tlas,
                uniforms: *state.uniform_buffers[current_frame],
                instances: *state._data.instances,
                prefix_sum_data: *state.prefix_sum_data,
                swapchain: *state.swapchain_image_heap_indices[next_image as usize],
                visible_meshlets: *state.visible_meshlets,
                dispatch: *state.dispatch,
            },
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.reset,
        );

        device.cmd_dispatch(**command_buffer, 1, 1, 1);

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.prefix_sum_instances,
        );

        device.cmd_dispatch(
            **command_buffer,
            state._data.num_instances.div_ceil(64),
            1,
            1,
        );

        device.insert_pipeline_barriers(
            command_buffer,
            [
                (
                    image.into(),
                    Some(nbn::BarrierOp::Acquire),
                    nbn::BarrierOp::ComputeStorageWrite,
                ),
                (
                    (&state.images.depth).into(),
                    None,
                    nbn::BarrierOp::DepthStencilAttachmentReadWrite,
                ),
                (
                    (&state.images.vis).into(),
                    None,
                    nbn::BarrierOp::ColorAttachmentWrite,
                ),
            ],
            [(
                &state.dispatch,
                nbn::BarrierOp::ComputeStorageWrite,
                nbn::BarrierOp::IndirectParamRead,
            )],
        );

        device.begin_rendering(
            command_buffer,
            extent.width,
            extent.height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*state.images.vis.image.view)
                .image_layout(vk::ImageLayout::GENERAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        uint32: [u32::max_value(); 4],
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)],
            Some(
                &vk::RenderingAttachmentInfo::default()
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    })
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .image_view(*state.images.depth.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ),
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *state.render_pipeline,
        );

        device.mesh_shader_loader.cmd_draw_mesh_tasks_indirect(
            **command_buffer,
            *state.dispatch.buffer,
            0,
            1,
            16,
        );

        device.cmd_end_rendering(**command_buffer);

        device.insert_image_pipeline_barrier(
            command_buffer,
            &state.images.vis,
            Some(nbn::BarrierOp::ColorAttachmentWrite),
            nbn::BarrierOp::ComputeStorageRead,
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.resolve,
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
    }
}

fn raytrace(state: &State, next_image: u32, current_frame: usize) {
    let device = &state.device;
    let command_buffer = &state.per_frame_command_buffers[current_frame];
    let extent = state.swapchain.create_info.image_extent;
    let image = &state.swapchain.images[next_image as usize];

    unsafe {
        device.push_constants::<PushConstants>(
            command_buffer,
            PushConstants {
                tlas: *state._data.tlas.tlas,
                uniforms: *state.uniform_buffers[current_frame],
                instances: *state._data.instances,
                prefix_sum_data: *state.prefix_sum_data,
                swapchain: *state.swapchain_image_heap_indices[next_image as usize],
                visible_meshlets: *state.visible_meshlets,
                dispatch: *state.dispatch,
            },
        );

        device.insert_pipeline_barriers(
            command_buffer,
            [(
                image.into(),
                Some(nbn::BarrierOp::Acquire),
                nbn::BarrierOp::ComputeStorageWrite,
            )],
            [],
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.raytrace,
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
    }
}

fn load_gltf<P: AsRef<std::path::Path>>(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: P,
) -> LoadedData {
    let path = path.as_ref();

    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();

    assert_eq!(buffer, None);

    let buffer_data =
        std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    let buffer = staging_buffer.create_buffer_from_slice(
        device,
        &format!("{} buffer", path.display()),
        &buffer_data,
    );

    let meshlets =
        read_meshlets_file(&device, staging_buffer, &path.with_extension("meshlets")).unwrap();

    let get_buffer_data = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        &buffer_data[bv.byte_offset + accessor.byte_offset..]
    };

    let get_buffer_offset = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        (*buffer) + bv.byte_offset as u64 + accessor.byte_offset as u64
    };

    let mut instances = Vec::new();
    let mut images = Vec::new();

    let mut num_indices = 0;

    let mut parent_mappings = vec![None; gltf.nodes.len()];

    for (i, node) in gltf.nodes.iter().enumerate() {
        for &child in &node.children {
            parent_mappings[child] = Some(i);
        }
    }

    let mut blases = Vec::new();
    let mut tlas_instances = Vec::new();

    gltf.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| node.mesh.map(|mesh_index| (i, node, mesh_index)))
        .for_each(|(i, node, mesh_index)| {
            let mesh = &gltf.meshes[mesh_index];
            let mesh_meshlets = &meshlets.metadata[mesh_index];

            let (mut translation, scale, rotation) = match node.transform() {
                goth_gltf::NodeTransform::Set {
                    translation,
                    scale,
                    rotation,
                } => (translation, scale, rotation),
                other => panic!("bad transform: {:?}", other),
            };

            if let Some(parent) = parent_mappings[i] {
                match gltf.nodes[parent].transform() {
                    goth_gltf::NodeTransform::Set {
                        translation: parent_translation,
                        scale: [1.0, 1.0, 1.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                    } => {
                        translation =
                            std::array::from_fn(|i| translation[i] + parent_translation[i]);
                    }
                    other => panic!("bad parent transform: {:?}", other),
                }
            }

            for (
                primitive,
                &[
                    num_meshlets,
                    vertices_offset,
                    triangles_offset,
                    meshlets_offset,
                ],
            ) in mesh.primitives.iter().zip(mesh_meshlets)
            {
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

                let material = &gltf.materials[primitive.material.unwrap_or(0)];

                let positions_accessor = &gltf.accessors[primitive.attributes.position.unwrap()];
                let positions =
                    &nbn::cast_slice::<_, [f32; 3]>(get_buffer_data(positions_accessor))
                        [..positions_accessor.count];
                let radius = positions
                    .iter()
                    .map(|&p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .sqrt();

                let acceleration_structure = device.create_acceleration_structure(
                    &format!("{} acceleration structure", path.display(),),
                    nbn::AccelerationStructureData::Triangles(&[
                        nbn::AccelerationStructureTriangles {
                            index_type: if is_32_bit {
                                vk::IndexType::UINT32
                            } else {
                                vk::IndexType::UINT16
                            },
                            opaque: true,
                            vertices_buffer_address: get(primitive.attributes.position),
                            indices_buffer_address: get_buffer_offset(indices),
                            num_vertices: positions_accessor.count as _,
                            num_indices: indices.count as _,
                        },
                    ]),
                    staging_buffer,
                );

                let transform = nbn::glam::Mat4::from_scale_rotation_translation(
                    scale.into(),
                    nbn::glam::Quat::from_array(rotation),
                    translation.into(),
                );

                let image_index = if let Some(info) =
                    material.pbr_metallic_roughness.base_color_texture.as_ref()
                {
                    let texture = &gltf.textures[info.index];
                    let sampler = &gltf.samplers[texture.sampler.unwrap()];
                    let image = &gltf.images[texture.source.unwrap()];
                    let path = path.with_file_name(image.uri.as_ref().unwrap());
                    let image = load_dds(device, staging_buffer, path);
                    let image = nbn::IndexedImage {
                        index: device.register_image_with_sampler(
                            *image.view,
                            if sampler.wrap_s == goth_gltf::SamplerWrap::ClampToEdge {
                                &device.samplers.clamp
                            } else {
                                &device.samplers.repeat
                            },
                            false,
                        ),
                        image,
                    };

                    let index = *image.index;

                    images.push(image);

                    index
                } else {
                    u32::max_value()
                };

                let instance = Instance {
                    translation,
                    scale,
                    rotation,
                    indices: get_buffer_offset(indices),
                    positions: get(primitive.attributes.position),
                    uvs: get(primitive.attributes.texcoord_0),
                    flags: is_32_bit as _,
                    image: image_index,
                    radius,
                    meshlets: *meshlets.buffer + meshlets_offset as u64,
                    triangles: *meshlets.buffer + triangles_offset as u64,
                    vertices: *meshlets.buffer + vertices_offset as u64,
                    num_meshlets,
                };

                if let Some(instancing) = node.extensions.ext_mesh_gpu_instancing {
                    assert_eq!(
                        (translation, rotation, scale),
                        ([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3])
                    );

                    let translations = &gltf.accessors[instancing.attributes.translation];
                    let scales = &gltf.accessors[instancing.attributes.scale];
                    let rotations = &gltf.accessors[instancing.attributes.rotation];

                    let scales =
                        &nbn::cast_slice::<_, [f32; 3]>(get_buffer_data(scales))[..scales.count];

                    let translations =
                        &nbn::cast_slice::<_, [f32; 3]>(get_buffer_data(translations))
                            [..translations.count];

                    let rotations = &nbn::cast_slice::<_, [f32; 4]>(get_buffer_data(rotations))
                        [..rotations.count];

                    for ((&translation, &scale), &rotation) in
                        translations.iter().zip(scales).zip(rotations)
                    {
                        let transform = nbn::glam::Mat4::from_scale_rotation_translation(
                            scale.into(),
                            nbn::glam::Quat::from_array(rotation),
                            translation.into(),
                        );
                        tlas_instances.push(
                            nbn::AccelerationStructureInstance {
                                acceleration_structure: *acceleration_structure,
                                transform,
                                custom_index: instances.len() as _,
                                ..Default::default()
                            }
                            .to_vk(),
                        );
                        instances.push(Instance {
                            translation,
                            scale,
                            rotation,
                            ..instance
                        });
                        num_indices += indices.count;
                    }
                } else {
                    tlas_instances.push(
                        nbn::AccelerationStructureInstance {
                            acceleration_structure: *acceleration_structure,
                            transform,
                            custom_index: instances.len() as _,
                            ..Default::default()
                        }
                        .to_vk(),
                    );
                    instances.push(instance);
                    num_indices += indices.count;
                }

                dbg!(num_indices);
                blases.push(acceleration_structure);
            }
        });

    LoadedData {
        _buffer: buffer,
        _images: images,
        _meshlets: meshlets.buffer,
        instances: staging_buffer.create_buffer_from_slice(
            device,
            &format!("{} instances", path.display()),
            &instances,
        ),
        num_instances: instances.len() as _,
        tlas: device.create_tlas_from_instances(staging_buffer, "tlas", &tlas_instances),
        blases,
    }

    // for [num_meshlets, vertices_offset, triangles_offset, meshlets_offset] in meshlets.metadata.iter().flat_map(|x| x) {
    //     dbg!(num_meshlets, vertices_offset, triangles_offset, meshlets_offset);
    // }
}

fn load_dds<P: AsRef<std::path::Path>>(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: P,
) -> nbn::Image {
    let path = path.as_ref();
    let dds = match std::fs::File::open(path) {
        Ok(file) => ddsfile::Dds::read(file).unwrap(),
        Err(error) => {
            panic!("{} failed to load: {}", path.display(), error);
        }
    };

    assert_eq!(dds.get_dxgi_format(), Some(ddsfile::DxgiFormat::BC7_UNorm));
    let extent = vk::Extent3D {
        width: dds.get_width(),
        height: dds.get_height(),
        depth: dds.get_depth(),
    };

    let bits_per_pixel = 8;

    let mut offset = 0;
    let mut offsets = Vec::new();
    for i in 0..dds.get_num_mipmap_levels() {
        offsets.push(offset);
        let level_width = (extent.width >> i).max(1).next_multiple_of(4) as u64;
        let level_height = (extent.height >> i).max(1).next_multiple_of(4) as u64;
        offset += (level_width * level_height) * bits_per_pixel / 8;
    }

    // use the 1st image instead as I don't have enough vram otherwise

    let (offsets, divisor) = if extent.width >= 4096 {
        (nbn::ImageLods::Offsets(&offsets[1..]), 2)
    } else {
        (nbn::ImageLods::Offsets(&offsets), 1)
    };

    staging_buffer.create_sampled_image(
        device,
        nbn::SampledImageDescriptor {
            name: &path.display().to_string(),
            extent: vk::Extent3D {
                width: dds.get_width() / divisor,
                height: dds.get_height() / divisor,
                depth: dds.get_depth(),
            }
            .into(),
            format: vk::Format::BC7_SRGB_BLOCK,
        },
        &dds.data,
        nbn::QueueType::Graphics,
        offsets,
    )
}

struct LoadedData {
    _buffer: nbn::Buffer,
    _meshlets: nbn::Buffer,
    _images: Vec<nbn::IndexedImage>,
    instances: nbn::Buffer,
    num_instances: u32,
    tlas: nbn::TlasWithInstances,
    blases: Vec<nbn::AccelerationStructure>,
}

struct Meshlets {
    buffer: nbn::Buffer,
    metadata: Vec<Vec<[u32; 4]>>,
}

fn read_meshlets_file<P: AsRef<std::path::Path>>(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: P,
) -> std::io::Result<Meshlets> {
    use std::io::Read;

    let path = path.as_ref();
    let mut reader = std::fs::File::open(path)?;
    let mut header = [0; 8];
    reader.read_exact(&mut header)?;
    assert_eq!(b"MESHLETS", &header);
    let mut val = [0; 4];
    reader.read_exact(&mut val)?;

    let num_meshes = u32::from_le_bytes(val);
    let mut meshes = Vec::with_capacity(num_meshes as _);

    for _ in 0..num_meshes {
        reader.read_exact(&mut val)?;
        let num_primitives = u32::from_le_bytes(val);
        let mut primitives = vec![[0_u32; 4]; num_primitives as _];
        reader.read_exact(nbn::cast_slice_mut(&mut primitives))?;
        meshes.push(primitives);
    }

    reader.read_exact(&mut val)?;

    let len_data = u32::from_le_bytes(val);

    let buffer = staging_buffer.create_buffer(
        device,
        &format!("{} staging buffer", path.display()),
        len_data as _,
        reader,
    );

    Ok(Meshlets {
        buffer,
        metadata: meshes,
    })
}
