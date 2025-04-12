use ash::vk;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

slang_struct::slang_include!("shaders/gltf.slang");

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
            &dds.get_data(0).unwrap(),
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
                &zstd::bulk::decompress(&level.data, level.uncompressed_byte_length as _).unwrap(),
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

fn create_pipeline(
    device: &nbn::Device,
    shader: &nbn::ShaderModule,
    swapchain: &nbn::Swapchain,
) -> nbn::Pipeline {
    device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
        vertex: nbn::ShaderDesc {
            module: &shader,
            entry_point: c"vertex",
        },
        fragment: nbn::ShaderDesc {
            module: &shader,
            entry_point: c"fragment",
        },
        color_attachment_formats: &[swapchain.create_info.image_format],
        conservative_rasterization: false,
        depth: nbn::GraphicsPipelineDepthDesc {
            write_enable: true,
            test_enable: true,
            compare_op: vk::CompareOp::GREATER,
            format: vk::Format::D32_SFLOAT,
        },
        cull_mode: vk::CullModeFlags::FRONT,
    })
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    pipeline: nbn::Pipeline,
    shader: nbn::ReloadableShader,
    depth_buffer: nbn::Image,
    time: f32,
    gltf: nbn::Buffer,
    aux: Vec<nbn::Buffer>,
    num_indices: u32,
    draws: nbn::Buffer,
    images: Vec<nbn::SampledImage>,
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
        let device = nbn::Device::new(Some(&window));

        let (gltf, images, draws, num_indices, aux) = {
            let base = "models/citadel";

            let bytes = std::fs::read(&format!("{}/voxelization_ktx2.gltf", base)).unwrap();
            let (gltf, buffer): (
                goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
                _,
            ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
            assert!(buffer.is_none());
            dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

            let mut image_formats = vec![vk::Format::R8G8B8A8_UNORM; gltf.images.len()];

            for material in &gltf.materials {
                if let Some(tex) = &material.emissive_texture {
                    image_formats[gltf.textures[tex.index].source.unwrap()] =
                        vk::Format::R8G8B8A8_SRGB;
                }
                if let Some(tex) = &material.pbr_metallic_roughness.base_color_texture {
                    image_formats[gltf.textures[tex.index].source.unwrap()] =
                        vk::Format::R8G8B8A8_SRGB;
                }
            }

            let buffer = std::fs::read(&format!(
                "{}/{}",
                base,
                gltf.buffers[0].uri.as_ref().unwrap()
            ))
            .unwrap();

            let images = gltf
                .images
                .iter()
                .zip(&image_formats)
                .map(|(image, format)| {
                    let path = format!("{}/{}", base, image.uri.as_ref().unwrap());
                    create_image(&device, &path, *format, nbn::QueueType::Graphics)
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

            dbg!(&materials.len());

            let primitives_iter = || gltf.meshes.iter().flat_map(|mesh| &mesh.primitives);

            let primitives: Vec<Primitive> = primitives_iter()
                .map(|primitive| Primitive {
                    indices: primitive.indices.unwrap() as u32,
                    positions: primitive.attributes.position.unwrap() as u32,
                    uvs: primitive.attributes.texcoord_0.unwrap() as u32,
                    normals: primitive.attributes.normal.unwrap() as u32,
                    material: primitive.material.unwrap() as u32,
                })
                .collect();

            let buffer_views: Vec<BufferView> = gltf
                .buffer_views
                .iter()
                .map(|buffer_view| BufferView {
                    byte_offset: buffer_view.byte_offset as u32,
                })
                .collect();

            let accessors: Vec<Accessor> = gltf
                .accessors
                .iter()
                .map(|accessor| Accessor {
                    buffer_view: accessor.buffer_view.unwrap() as u32,
                    count: accessor.count as _,
                    byte_offset: accessor.byte_offset as u32,
                    flags: (accessor.component_type == goth_gltf::ComponentType::UnsignedInt)
                        as u32,
                })
                .collect();

            //panic!();

            let draws: Vec<_> = primitives_iter()
                .enumerate()
                .map(|(i, primitive)| vk::DrawIndirectCommand {
                    vertex_count: accessors[primitive.indices.unwrap()].count,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: i as _,
                })
                .collect();

            let draws = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "draws",
                data: &draws,
            });

            let num_indices: u32 = primitives.len() as _;

            dbg!(num_indices);

            let materials = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "materials",
                data: &materials,
            });

            let buffer = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "buffer",
                data: &buffer,
            });
            let primitives = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "primitives",
                data: &primitives,
            });
            let buffer_views = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "buffer_views",
                data: &buffer_views,
            });
            let accessors = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "accessors",
                data: &accessors,
            });
            let gltf = device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "gltf",
                data: &[Gltf {
                    buffer_views: *buffer_views,
                    accessors: *accessors,
                    buffer: *buffer,
                    primitives: *primitives,
                    materials: *materials,
                    num_primitives: primitives_iter().count() as u32,
                }],
            });
            (
                gltf,
                images,
                draws,
                num_indices,
                vec![accessors, buffer_views, primitives, buffer, materials],
            )
        };

        dbg!(&images.len());

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

        let swapchain = device.create_swapchain(&window);
        let shader = device.load_reloadable_shader("shaders/compiled/gltf_raster.spv");
        let pipeline = create_pipeline(&device, &shader, &swapchain);

        let size = window.inner_size();

        self.window_state = Some(WindowState {
            per_frame_command_buffers: [
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
                device.create_command_buffer(nbn::QueueType::Graphics),
            ],
            sync_resources: device.create_sync_resources(),
            swapchain,
            window,
            pipeline,
            shader,
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
            gltf,
            draws,
            images,
            aux,
            num_indices,
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
            }
            winit::event::WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();

                if let Some(state) = self.window_state.as_mut() {
                    if state.shader.try_reload(&device) {
                        unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };
                        state.pipeline = create_pipeline(&device, &state.shader, &state.swapchain);
                    }

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

                        device.reset_command_buffer(&command_buffer);
                        device
                            .begin_command_buffer(
                                **command_buffer,
                                &vk::CommandBufferBeginInfo::default(),
                            )
                            .unwrap();
                        vk_sync::cmd::pipeline_barrier(
                            &device,
                            **command_buffer,
                            None,
                            &[],
                            &[
                                vk_sync::ImageBarrier {
                                    previous_accesses: &[vk_sync::AccessType::Present],
                                    next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
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
                            ],
                        );

                        let extent = state.swapchain.create_info.image_extent;

                        device.begin_rendering(
                            &command_buffer,
                            extent.width,
                            extent.height,
                            &[vk::RenderingAttachmentInfo::default()
                                .image_view(*image.view)
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue {
                                    color: vk::ClearColorValue { float32: [1.0; 4] },
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
                            *state.pipeline,
                        );
                        device.bind_internal_descriptor_sets(&command_buffer);
                        let scale = 10000.0;
                        let mat = glam::Mat4::perspective_infinite_reverse_rh(
                            54.0_f32.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            0.0001,
                        ) * glam::Mat4::look_at_rh(
                            glam::Vec3::new(
                                scale * state.time.cos(),
                                scale,
                                scale * state.time.sin(),
                            ),
                            glam::Vec3::ZERO,
                            -glam::Vec3::Y,
                        );

                        device.push_constants(&command_buffer, (mat, *state.gltf));
                        state.time += 0.005;
                        device.cmd_draw_indirect(
                            **command_buffer,
                            *state.draws.buffer,
                            0,
                            state.num_indices,
                            std::mem::size_of::<vk::DrawIndirectCommand>() as _,
                        );

                        device.cmd_end_rendering(**command_buffer);
                        vk_sync::cmd::pipeline_barrier(
                            &device,
                            **command_buffer,
                            None,
                            &[],
                            &[vk_sync::ImageBarrier {
                                previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
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
                            &device,
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
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App {
            device: None,
            window_state: None,
        })
        .unwrap();
}
