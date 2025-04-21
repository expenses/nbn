use std::{ffi::CStr, io::Read};

use ash::vk;

slang_struct::slang_include!("shaders/models.slang");
slang_struct::slang_include!("shaders/culling.slang");
slang_struct::slang_include!("shaders/uniforms.slang");

fn create_image(
    device: &nbn::Device,
    filename: &str,
    format: vk::Format,
    transition_to: nbn::QueueType,
) -> nbn::PendingImageUpload {
    if filename.ends_with(".dds") {
        let dds = ddsfile::Dds::read(std::fs::File::open(filename).unwrap()).unwrap();

        let format = match dds.get_dxgi_format().unwrap() {
            ddsfile::DxgiFormat::BC1_UNorm_sRGB => vk::Format::BC1_RGB_SRGB_BLOCK,
            ddsfile::DxgiFormat::BC3_UNorm_sRGB => vk::Format::BC3_SRGB_BLOCK,
            ddsfile::DxgiFormat::BC5_UNorm => vk::Format::BC5_UNORM_BLOCK,
            other => panic!("{:?}", other),
        };
        device.create_sampled_image_with_data(
            nbn::SampledImageDescriptor {
                name: filename,
                extent: vk::Extent3D {
                    width: dds.get_width(),
                    height: dds.get_height(),
                    depth: 1,
                },
                format,
            },
            dds.get_data(0).unwrap(),
            transition_to,
            &[0],
        )
    } else if filename.ends_with(".ktx2") {
        let ktx2 = ktx2::Reader::new(std::fs::read(filename).unwrap()).unwrap();
        let header = ktx2.header();

        let mut data = Vec::with_capacity(
            ktx2.levels()
                .map(|level| level.uncompressed_byte_length)
                .sum::<u64>() as _,
        );
        let mut offsets = Vec::with_capacity(ktx2.levels().len());

        for level in ktx2.levels() {
            offsets.push(data.len() as _);
            data.extend_from_slice(
                &zstd::bulk::decompress(level.data, level.uncompressed_byte_length as _).unwrap(),
            );
        }

        device.create_sampled_image_with_data(
            nbn::SampledImageDescriptor {
                name: filename,
                extent: vk::Extent3D {
                    width: header.pixel_width,
                    height: header.pixel_height,
                    depth: header.pixel_depth.max(1),
                },
                format: vk::Format::from_raw(header.format.unwrap().value() as _),
            },
            &data,
            transition_to,
            &offsets,
        )
    } else {
        let image_data = image::open(filename).unwrap().to_rgba8();

        device.create_sampled_image_with_data(
            nbn::SampledImageDescriptor {
                name: filename,
                extent: vk::Extent3D {
                    width: image_data.width(),
                    height: image_data.height(),
                    depth: 1,
                },
                format,
            },
            &image_data,
            transition_to,
            &[0],
        )
    }
}

fn create_mesh_pipeline(device: &nbn::Device, shader: &nbn::ShaderModule) -> MeshPipelines {
    let create_pipeline = |fragment: &CStr| {
        device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            vertex: nbn::ShaderDesc {
                module: shader,
                entry_point: c"vertex",
            },
            fragment: nbn::ShaderDesc {
                module: shader,
                entry_point: fragment,
            },
            color_attachment_formats: &[vk::Format::R32_UINT],
            conservative_rasterization: false,
            depth: nbn::GraphicsPipelineDepthDesc {
                write_enable: true,
                test_enable: true,
                compare_op: vk::CompareOp::GREATER,
                format: vk::Format::D32_SFLOAT,
            },
            cull_mode: vk::CullModeFlags::BACK,
            mesh_shader: true,
        })
    };

    MeshPipelines {
        opaque: create_pipeline(c"opaque_fragment"),
        alpha_clipped: create_pipeline(c"alpha_clipped_fragment"),
    }
}

struct MeshPipelines {
    opaque: nbn::Pipeline,
    alpha_clipped: nbn::Pipeline,
}

struct ComputePipelines {
    reset_buffers: nbn::Pipeline,
    generate_meshlet_prefix_sums: nbn::Pipeline,
    write_meshlet_instances: nbn::Pipeline,
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    swapchain_image_heap_indices: Vec<u32>,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    combined_uniform_buffer: nbn::Buffer,
    mesh_pipelines: nbn::ReloadablePipeline<MeshPipelines>,
    compute_pipelines: nbn::ReloadablePipeline<ComputePipelines>,
    blit_pipeline: nbn::ReloadablePipeline<nbn::Pipeline>,
    depth_buffer: nbn::Image,
    visbuffer: nbn::IndexedImage,
    time: f32,
    prefix_sum_values: nbn::Buffer,
    _gltf: GltfData,
    instances: nbn::Buffer,
    meshlet_instances: nbn::Buffer,
    dispatches: nbn::Buffer,
    num_instances: u32,
}

struct GltfData {
    _images: Vec<nbn::IndexedImage>,
    _buffer: nbn::Buffer,
    _meshlets_buffer: nbn::Buffer,
    meshes: Vec<Vec<Model>>,
}

struct Meshlets {
    data: Vec<u8>,
    metadata: Vec<Vec<[u32; 4]>>,
}

fn read_meshlets_file(path: &std::path::Path) -> std::io::Result<Meshlets> {
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
    let mut data = vec![0; len_data as _];
    reader.read_exact(&mut data)?;

    Ok(Meshlets {
        data,
        metadata: meshes,
    })
}

fn load_gltf(device: &nbn::Device, path: &std::path::Path) -> GltfData {
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

    let meshlets = read_meshlets_file(&path.with_extension("meshlets")).unwrap();

    let meshlets_buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: &format!("{} meshlets buffer", path.display()),
        data: &meshlets.data,
    });

    let buffer = std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    let buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
        name: &format!("{} buffer", path.display()),
        data: &buffer,
    });

    let images = gltf
        .images
        .iter()
        .zip(&image_formats)
        .map(|(image, format)| {
            create_image(
                device,
                path.with_file_name(image.uri.as_ref().unwrap())
                    .to_str()
                    .unwrap(),
                *format,
                nbn::QueueType::Graphics,
            )
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
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            metallic_roughness_image: material
                .pbr_metallic_roughness
                .metallic_roughness_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            normal_image: material
                .normal_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            emissive_image: material
                .emissive_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()].image)
                .unwrap_or(u32::MAX),
            flags: !matches!(material.alpha_mode, goth_gltf::AlphaMode::Opaque) as u32,
        })
        .collect();

    let get_buffer_offset = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        (*buffer) + bv.byte_offset as u64 + accessor.byte_offset as u64
    };

    let mut meshes = Vec::new();

    for (mesh, meshlets_mesh) in gltf.meshes.iter().zip(&meshlets.metadata) {
        let mut primitives = Vec::new();

        for (primitive, &[num_meshlets, meshlets_offset, vertices_offset, triangles_offset]) in
            mesh.primitives.iter().zip(meshlets_mesh)
        {
            let material = materials[primitive.material.unwrap()];

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
            primitives.push(Model {
                material,
                positions: get(primitive.attributes.position),
                uvs: get(primitive.attributes.texcoord_0),
                normals: get(primitive.attributes.normal),
                indices: get_buffer_offset(indices),
                meshlets: *meshlets_buffer + meshlets_offset as u64,
                triangles: *meshlets_buffer + triangles_offset as u64,
                vertices: *meshlets_buffer + vertices_offset as u64,
                flags: is_32_bit as u32,
                num_meshlets,
            });
        }

        meshes.push(primitives);
    }

    dbg!(&materials.len());

    let transfer_fence = device.create_fence();

    let transfer_command_buffers: Vec<vk::CommandBuffer> =
        images.iter().map(|image| *image.command_buffer).collect();

    unsafe {
        device
            .queue_submit(
                *device.transfer_queue,
                &[vk::SubmitInfo::default().command_buffers(&transfer_command_buffers)],
                *transfer_fence,
            )
            .unwrap();

        device
            .wait_for_fences(&[*transfer_fence], true, !0)
            .unwrap();
    }

    let images: Vec<_> = images.into_iter().map(|image| image.into_inner()).collect();

    GltfData {
        _images: images,
        _buffer: buffer,
        _meshlets_buffer: meshlets_buffer,
        meshes,
    }
}

// copied from https://docs.rs/ultraviolet/latest/src/ultraviolet/projection/rh_yup.rs.html#350-365
// The glam version only works for opengl/wgpu I think :(
fn perspective_reversed_infinite_z_vk(
    vertical_fov: f32,
    aspect_ratio: f32,
    z_near: f32,
) -> glam::Mat4 {
    let t = (vertical_fov / 2.0).tan();

    let sy = 1.0 / t;

    let sx = sy / aspect_ratio;

    glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
        glam::Vec4::new(0.0, 0.0, z_near, 0.0),
    )
}

struct App {
    window_state: Option<WindowState>,
    device: Option<nbn::Device>,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::WindowAttributes::default().with_resizable(true))
            .unwrap();
        let device = nbn::Device::new(Some(&window), false);

        let gltf = load_gltf(
            &device,
            std::path::Path::new("models/citadel/voxelization_ktx2.gltf"),
        );

        let mut instances = Vec::new();

        for mesh in &gltf.meshes {
            for &model in mesh {
                instances.push(Instance {
                    model,
                    position: [0.0; 4],
                    _padding: [0; 2],
                });
            }
        }

        for mesh in gltf.meshes.iter().rev() {
            for &model in mesh {
                instances.push(Instance {
                    model,
                    position: [0.0, 1000.0, 0.0, 1.0],
                    _padding: [0; 2],
                });
            }
        }

        let num_instances = instances.len() as u32;

        let instances = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "instances",
            data: &instances,
        });

        let swapchain = device.create_swapchain(&window, vk::ImageUsageFlags::STORAGE, true);

        let size = window.inner_size();

        let visbuffer = device.create_image(nbn::ImageDescriptor {
            name: "visbuffer",
            format: vk::Format::R32_UINT,
            extent: vk::Extent3D {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            ty: vk::ImageViewType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_levels: 1,
        });

        self.window_state = Some(WindowState {
            prefix_sum_values: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "prefix_sum_values",
                    size: 8 * 10_000,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            dispatches: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "dispatches",
                    size: (std::mem::size_of::<[vk::DrawMeshTasksIndirectCommandEXT; 4]>() + 8 * 2)
                        as _,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            combined_uniform_buffer: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "combined_uniform_buffer",
                    size: (std::mem::size_of::<UniformBuffer>() * nbn::FRAMES_IN_FLIGHT) as _,
                    ty: nbn::MemoryLocation::CpuToGpu,
                })
                .unwrap(),
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            swapchain_image_heap_indices: swapchain
                .images
                .iter()
                .map(|image| device.register_image(*image.view, true))
                .collect(),
            sync_resources: device.create_sync_resources(),
            swapchain,
            window,
            mesh_pipelines: nbn::ReloadablePipeline::new(
                &device,
                device.load_reloadable_shader("shaders/compiled/mesh_shaders.spv"),
                &create_mesh_pipeline,
            ),
            compute_pipelines: nbn::ReloadablePipeline::new(
                &device,
                device.load_reloadable_shader("shaders/compiled/compute.spv"),
                &|device: &nbn::Device, shader| ComputePipelines {
                    reset_buffers: device.create_compute_pipeline(shader, c"reset_buffers"),
                    generate_meshlet_prefix_sums: device
                        .create_compute_pipeline(shader, c"generate_meshlet_prefix_sums"),
                    write_meshlet_instances: device
                        .create_compute_pipeline(shader, c"write_meshlet_instances"),
                },
            ),
            blit_pipeline: nbn::ReloadablePipeline::new(
                &device,
                device.load_reloadable_shader("shaders/compiled/resolve_visbuffer.spv"),
                &|device: &nbn::Device, shader| device.create_compute_pipeline(shader, c"main"),
            ),
            depth_buffer: device.create_image(nbn::ImageDescriptor {
                name: "depth_buffer",
                format: vk::Format::D32_SFLOAT,
                extent: vk::Extent3D {
                    width: size.width,
                    height: size.height,
                    depth: 1,
                },
                ty: vk::ImageViewType::TYPE_2D,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_levels: 1,
            }),
            visbuffer: nbn::IndexedImage {
                index: device.register_image(*visbuffer.view, true),
                image: visbuffer,
            },
            _gltf: gltf,
            instances,
            meshlet_instances: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "meshlet_instances",
                    size: (std::mem::size_of::<([u32; 4], u32)>() * 100_000) as _,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            num_instances,
            time: 0.0,
        });
        self.device = Some(device);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                let window_state = self.window_state.as_mut().unwrap();

                window_state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };

                let device = self.device.as_ref().unwrap();

                device.recreate_swapchain(&mut window_state.swapchain);
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                window_state.depth_buffer = device.create_image(nbn::ImageDescriptor {
                    name: "depth_buffer",
                    format: vk::Format::D32_SFLOAT,
                    extent: vk::Extent3D {
                        width: new_size.width,
                        height: new_size.height,
                        depth: 1,
                    },
                    ty: vk::ImageViewType::TYPE_2D,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    mip_levels: 1,
                });
                device.deregister_image(window_state.visbuffer.index, true);
                window_state.visbuffer.image = device.create_image(nbn::ImageDescriptor {
                    name: "visbuffer",
                    format: vk::Format::R32_UINT,
                    extent: vk::Extent3D {
                        width: new_size.width,
                        height: new_size.height,
                        depth: 1,
                    },
                    ty: vk::ImageViewType::TYPE_2D,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_levels: 1,
                });
                window_state.visbuffer.index =
                    device.register_image(*window_state.visbuffer.image.view, true);

                for index in window_state.swapchain_image_heap_indices.drain(..) {
                    device.deregister_image(index, true);
                }
                window_state.swapchain_image_heap_indices.extend(
                    window_state
                        .swapchain
                        .images
                        .iter()
                        .map(|image| device.register_image(*image.view, true)),
                );
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                if let Some(state) = self.window_state.as_mut() {
                    state.blit_pipeline.refresh(device);
                    state.mesh_pipelines.refresh(device);
                    state.compute_pipelines.refresh(device);

                    let extent = state.swapchain.create_info.image_extent;

                    let scale = 4000.0;
                    let perspective = perspective_reversed_infinite_z_vk(
                        45.0_f32.to_radians(),
                        extent.width as f32 / extent.height as f32,
                        0.0001,
                    );
                    let view = glam::Mat4::look_at_rh(
                        glam::Vec3::new(scale * state.time.cos(), scale, scale * state.time.sin()),
                        glam::Vec3::ZERO,
                        glam::Vec3::Y,
                    );
                    let mat = perspective * view;

                    let uniforms = state
                        .combined_uniform_buffer
                        .try_as_slice_mut::<UniformBuffer>()
                        .unwrap();

                    uniforms[state.sync_resources.current_frame] = UniformBuffer {
                        mat: mat.to_cols_array(),
                        view: view.to_cols_array(),
                        perspective: perspective.to_cols_array(),
                        near_plane: 0.0001,
                        instances: *state.instances,
                        meshlet_instances: *state.meshlet_instances,
                        extent: [extent.width, extent.height],
                        num_instances: state.num_instances,
                        visbuffer: state.visbuffer.index,
                        dispatches: *state.dispatches,
                    };

                    let uniforms_ptr = *state.combined_uniform_buffer
                        + (std::mem::size_of::<UniformBuffer>()
                            * state.sync_resources.current_frame) as u64;

                    unsafe {
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

                        device.bind_internal_descriptor_sets_to_all(command_buffer);

                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            *state.compute_pipelines.reset_buffers,
                        );
                        device.push_constants(command_buffer, *state.dispatches);
                        device.cmd_dispatch(**command_buffer, 1, 1, 1);

                        device.insert_global_barrier(
                            command_buffer,
                            &[vk_sync::AccessType::ComputeShaderWrite],
                            &[vk_sync::AccessType::ComputeShaderReadOther],
                        );

                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            *state.compute_pipelines.generate_meshlet_prefix_sums,
                        );
                        device.push_constants(
                            command_buffer,
                            (
                                uniforms_ptr,
                                *state.prefix_sum_values,
                                *state.prefix_sum_values + (8 * 5_000),
                            ),
                        );
                        device.cmd_dispatch(
                            **command_buffer,
                            state.num_instances.div_ceil(64),
                            1,
                            1,
                        );

                        device.insert_global_barrier(
                            command_buffer,
                            &[vk_sync::AccessType::ComputeShaderWrite],
                            &[vk_sync::AccessType::ComputeShaderReadOther],
                        );

                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            *state.compute_pipelines.write_meshlet_instances,
                        );
                        device.push_constants(
                            command_buffer,
                            (uniforms_ptr, *state.prefix_sum_values, 0_u32),
                        );
                        device.cmd_dispatch_indirect(
                            **command_buffer,
                            *state.dispatches.buffer,
                            4 * 3 * 2,
                        );
                        device.push_constants(
                            command_buffer,
                            (uniforms_ptr, *state.prefix_sum_values + (8 * 5_000), 1_u32),
                        );
                        device.cmd_dispatch_indirect(
                            **command_buffer,
                            *state.dispatches.buffer,
                            4 * 3 * 3,
                        );

                        vk_sync::cmd::pipeline_barrier(
                            device,
                            **command_buffer,
                            Some(vk_sync::GlobalBarrier {
                                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                                next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
                            }),
                            &[],
                            &[
                                vk_sync::ImageBarrier {
                                    previous_accesses: &[],
                                    next_accesses: &[
                                        vk_sync::AccessType::DepthStencilAttachmentWrite,
                                    ],
                                    previous_layout: vk_sync::ImageLayout::Optimal,
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    discard_contents: true,
                                    src_queue_family_index: device.graphics_queue.index,
                                    dst_queue_family_index: device.graphics_queue.index,
                                    image: **state.depth_buffer,
                                    range: vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::DEPTH),
                                },
                                vk_sync::ImageBarrier {
                                    previous_accesses: &[],
                                    next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                                    previous_layout: vk_sync::ImageLayout::Optimal,
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    discard_contents: true,
                                    src_queue_family_index: device.graphics_queue.index,
                                    dst_queue_family_index: device.graphics_queue.index,
                                    image: **state.visbuffer.image,
                                    range: vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                },
                            ],
                        );

                        device.begin_rendering(
                            command_buffer,
                            extent.width,
                            extent.height,
                            &[vk::RenderingAttachmentInfo::default()
                                .image_view(*state.visbuffer.image.view)
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        uint32: [u32::MAX; 4],
                                    },
                                })
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)],
                            Some(
                                &vk::RenderingAttachmentInfo::default()
                                    .image_view(*state.depth_buffer.view)
                                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                    .load_op(vk::AttachmentLoadOp::CLEAR)
                                    .store_op(vk::AttachmentStoreOp::STORE)
                                    .clear_value(vk::ClearValue {
                                        depth_stencil: vk::ClearDepthStencilValue {
                                            depth: 0.0,
                                            stencil: 0,
                                        },
                                    }),
                            ),
                        );

                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *state.mesh_pipelines.opaque,
                        );
                        device.push_constants(command_buffer, (uniforms_ptr, 0_u32));
                        device.mesh_shader_loader.cmd_draw_mesh_tasks_indirect(
                            **command_buffer,
                            *state.dispatches.buffer,
                            0,
                            1,
                            std::mem::size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as _,
                        );
                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *state.mesh_pipelines.alpha_clipped,
                        );
                        device.push_constants(command_buffer, (uniforms_ptr, 1_u32));
                        device.mesh_shader_loader.cmd_draw_mesh_tasks_indirect(
                            **command_buffer,
                            *state.dispatches.buffer,
                            std::mem::size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as _,
                            1,
                            std::mem::size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as _,
                        );

                        state.time += 0.005;
                        device.cmd_end_rendering(**command_buffer);
                        vk_sync::cmd::pipeline_barrier(
                            device,
                            **command_buffer,
                            None,
                            &[],
                            &[
                                vk_sync::ImageBarrier {
                                    previous_accesses: &[vk_sync::AccessType::Present],
                                    next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                                    previous_layout: vk_sync::ImageLayout::Optimal,
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    discard_contents: true,
                                    src_queue_family_index: device.graphics_queue.index,
                                    dst_queue_family_index: device.graphics_queue.index,
                                    image: image.image,
                                    range: vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                },
                                vk_sync::ImageBarrier {
                                    previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                                    next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
                                    previous_layout: vk_sync::ImageLayout::Optimal,
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    discard_contents: false,
                                    src_queue_family_index: device.graphics_queue.index,
                                    dst_queue_family_index: device.graphics_queue.index,
                                    image: **state.visbuffer.image,
                                    range: vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                },
                            ],
                        );
                        device.cmd_bind_pipeline(
                            **command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            **state.blit_pipeline,
                        );
                        device.push_constants(
                            command_buffer,
                            (
                                uniforms_ptr,
                                state.swapchain_image_heap_indices[next_image as usize],
                            ),
                        );
                        device.cmd_dispatch(
                            **command_buffer,
                            extent.width.div_ceil(8),
                            extent.height.div_ceil(8),
                            1,
                        );
                        vk_sync::cmd::pipeline_barrier(
                            device,
                            **command_buffer,
                            None,
                            &[],
                            &[vk_sync::ImageBarrier {
                                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                                next_accesses: &[vk_sync::AccessType::Present],
                                previous_layout: vk_sync::ImageLayout::Optimal,
                                next_layout: vk_sync::ImageLayout::Optimal,
                                discard_contents: false,
                                src_queue_family_index: device.graphics_queue.index,
                                dst_queue_family_index: device.graphics_queue.index,
                                image: image.image,
                                range: vk::ImageSubresourceRange::default()
                                    .layer_count(1)
                                    .level_count(1)
                                    .aspect_mask(vk::ImageAspectFlags::COLOR),
                            }],
                        );
                        device.end_command_buffer(**command_buffer).unwrap();

                        frame.submit(
                            device,
                            &[vk::CommandBufferSubmitInfo::default()
                                .command_buffer(**command_buffer)],
                        );
                        device
                            .swapchain_loader
                            .queue_present(
                                *device.graphics_queue,
                                &vk::PresentInfoKHR::default()
                                    .wait_semaphores(&[*frame.render_finished_semaphore])
                                    .swapchains(&[*state.swapchain])
                                    .image_indices(&[next_image]),
                            )
                            .unwrap();
                    }
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App {
            device: None,
            window_state: None,
        })
        .unwrap();
}
