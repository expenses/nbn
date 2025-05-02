use std::{ffi::CStr, io::Read};

mod assets;
mod rendering;

use assets::*;

use dolly::prelude::YawPitch;
use nbn::vk;
use winit::{event::ElementState, keyboard::KeyCode, window::CursorGrabMode};

slang_struct::slang_include!("shaders/models.slang");
slang_struct::slang_include!("shaders/uniforms.slang");

const TOTAL_NUM_INSTANCES_OF_TYPE: u64 = 10_000;
const TOTAL_NUM_VISIBLE_MESHLETS: usize = 1_000_000;

fn create_mesh_pipeline(device: &nbn::Device, shader: &nbn::ShaderModule) -> MeshPipelines {
    let create_pipeline = |fragment: &CStr, cull_mode| {
        device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
            name: "mesh pipeline",
            shaders: nbn::GraphicsPipelineShaders::Task {
                task: nbn::ShaderDesc {
                    module: shader,
                    entry_point: c"task",
                },
                mesh: nbn::ShaderDesc {
                    module: shader,
                    entry_point: c"vertex",
                },
                fragment: nbn::ShaderDesc {
                    module: shader,
                    entry_point: fragment,
                },
            },
            color_attachment_formats: &[vk::Format::R32_UINT],
            blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            conservative_rasterization: false,
            depth: nbn::GraphicsPipelineDepthDesc {
                write_enable: true,
                test_enable: true,
                compare_op: vk::CompareOp::GREATER,
                format: vk::Format::D32_SFLOAT,
            },
            cull_mode,
        })
    };

    MeshPipelines {
        opaque: create_pipeline(c"opaque_fragment", vk::CullModeFlags::BACK),
        alpha_clipped: create_pipeline(c"alpha_clipped_fragment", vk::CullModeFlags::NONE),
    }
}

struct MeshPipelines {
    opaque: nbn::Pipeline,
    alpha_clipped: nbn::Pipeline,
}

struct ComputePipelines {
    reset_buffers: nbn::Pipeline,
    generate_meshlet_prefix_sums: nbn::Pipeline,
}

struct BlitPipelines {
    resolve_visbuffer: nbn::Pipeline,
    tonemap: nbn::Pipeline,
}

#[derive(Default)]
struct KeyboardState {
    pub forwards: bool,
    pub backwards: bool,
    pub left: bool,
    pub right: bool,
}

struct Framebuffers {
    vis: nbn::Image,
    vis_index: nbn::ImageIndex,
    depth: nbn::Image,
    depth_index: nbn::ImageIndex,
    hdr: nbn::Image,
    hdr_sampled_index: nbn::ImageIndex,
    hdr_storage_index: nbn::ImageIndex,
}

impl Framebuffers {
    fn new(device: &nbn::Device, extent: vk::Extent3D) -> Self {
        let vis = device.create_image(nbn::ImageDescriptor {
            name: "visbuffer",
            format: vk::Format::R32_UINT,
            extent,
            ty: vk::ImageViewType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_levels: 1,
        });

        let hdr = device.create_image(nbn::ImageDescriptor {
            name: "hdrbuffer",
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent,
            ty: vk::ImageViewType::TYPE_2D,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_levels: 1,
        });

        let depth = device.create_image(nbn::ImageDescriptor {
            name: "depth_buffer",
            format: vk::Format::D32_SFLOAT,
            extent,
            ty: vk::ImageViewType::TYPE_2D,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            mip_levels: 1,
        });

        Self {
            vis_index: device.register_image(*vis.view, true),
            depth_index: device.register_image_with_sampler(
                *depth.view,
                &device.clamp_sampler,
                false,
            ),
            hdr_sampled_index: device.register_image_with_sampler(
                *hdr.view,
                &device.clamp_sampler,
                false,
            ),
            hdr_storage_index: device.register_image(*hdr.view, true),
            depth,
            vis,
            hdr,
        }
    }
}

struct WindowState {
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    combined_uniform_buffer: nbn::Buffer,
    mesh_pipelines: nbn::ReloadablePipeline<MeshPipelines>,
    compute_pipelines: nbn::ReloadablePipeline<ComputePipelines>,
    blit_pipelines: nbn::ReloadablePipeline<BlitPipelines>,
    framebuffers: Framebuffers,
    time: f32,
    prefix_sum_values: nbn::Buffer,
    _gltf: GltfData,
    instances: nbn::Buffer,
    meshlet_instances: nbn::Buffer,
    dispatches: nbn::Buffer,
    num_instances: u32,
    models: nbn::Buffer,
    camera_rig: dolly::rig::CameraRig,
    keyboard: KeyboardState,
    cursor_grabbed: bool,
    egui_winit: egui_winit::State,
    egui_render: nbn::egui::Renderer,
    alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer,
    debugging_pipeline: nbn::Pipeline,
    meshlet_debugging_pipeline: nbn::Pipeline,
    num_debug_meshlet_instances: u32,
    debug_meshlet_instances: nbn::Buffer,
    tonemap_lut: nbn::IndexedImage,
    tlas: nbn::AccelerationStructure,
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

        let mut staging_buffer =
            nbn::StagingBuffer::new(&device, 64 * 1024 * 1024, nbn::QueueType::Transfer);

        let tonemap_lut = assets::create_image(
            &device,
            &mut staging_buffer,
            "shaders/tony-mc-mapface/shader/tony_mc_mapface.dds",
            nbn::QueueType::Graphics,
        );
        let tonemap_lut = nbn::IndexedImage {
            index: device.register_image_with_sampler(
                *tonemap_lut.view,
                &device.clamp_sampler,
                false,
            ),
            image: tonemap_lut,
        };

        let gltf = load_gltf(
            &device,
            &mut staging_buffer,
            std::path::Path::new("models/citadel/voxelization_ktx2.gltf"),
            //std::path::Path::new("models/Bistro_v5_2/bistro_combined.gltf"),
        );

        let mut models = Vec::new();
        let mut instances = Vec::new();

        let mut debug_meshlet_instances = Vec::new();

        let mut acceleration_structure_instances: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::new();

        for model in &gltf.meshes {
            acceleration_structure_instances.push(
                nbn::AccelerationStructureInstance {
                    acceleration_structure_address: *model.acceleration_structure,
                    custom_index: models.len() as u32,
                    mask: 0xff,
                    shader_binding_table_record_offset: 0,
                    flags: vk::GeometryInstanceFlagsKHR::empty(),
                    transform: nbn::transform_from_mat4(glam::Mat4::IDENTITY),
                }
                .into(),
            );

            for &meshlet in &model.meshlets {
                debug_meshlet_instances.push(MeshletInstance {
                    meshlet,
                    instance: Instance {
                        _model_index_and_padding: [models.len() as u32; 4],
                        position: [0.0; 4],
                    },
                });
            }

            instances.push(Instance {
                _model_index_and_padding: [models.len() as u32; 4],
                position: [0.0; 4],
            });
            models.push(model.model);
        }

        let acceleration_structure_instances =
            device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "acceleration_structure_instances",
                data: &acceleration_structure_instances,
            });

        let tlas = device.create_acceleration_structure(
            "tlas",
            nbn::AccelerationStructureData::Instances {
                buffer_address: *acceleration_structure_instances,
                count: instances.len() as _,
            },
            &mut staging_buffer,
        );

        staging_buffer.finish(&device);

        let num_debug_meshlet_instances = debug_meshlet_instances.len() as u32;
        let debug_meshlet_instances = device.create_buffer_with_data(nbn::BufferInitDescriptor {
            name: "debug_meshlet_instances",
            data: &debug_meshlet_instances,
        });

        let swapchain = device.create_swapchain(
            &window,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            true,
        );

        let size = window.inner_size();

        let camera_rig: dolly::rig::CameraRig = dolly::rig::CameraRig::builder()
            .with(dolly::drivers::Position::new([1000.0, 1000.0, 0.0]))
            .with(dolly::drivers::YawPitch::new())
            .with(dolly::drivers::Smooth::new_position_rotation(1.0, 1.0))
            .build();

        let debugging_module = device.load_shader("shaders/compiled/debugging.spv");

        self.window_state = Some(WindowState {
            tlas,
            tonemap_lut,
            num_debug_meshlet_instances,
            debug_meshlet_instances,
            debugging_pipeline: device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
                name: "model debugging pipeline",
                shaders: nbn::GraphicsPipelineShaders::Legacy {
                    vertex: nbn::ShaderDesc {
                        module: &debugging_module,
                        entry_point: c"model_debug_vertex",
                    },
                    fragment: nbn::ShaderDesc {
                        module: &debugging_module,
                        entry_point: c"model_debug_fragment",
                    },
                },
                conservative_rasterization: false,
                cull_mode: vk::CullModeFlags::BACK,
                blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)],
                depth: nbn::GraphicsPipelineDepthDesc {
                    write_enable: true,
                    test_enable: true,
                    compare_op: vk::CompareOp::GREATER,
                    format: vk::Format::D32_SFLOAT,
                },
                color_attachment_formats: &[swapchain.create_info.image_format],
            }),
            meshlet_debugging_pipeline: device.create_graphics_pipeline(
                nbn::GraphicsPipelineDesc {
                    name: "meshlet debugging pipeline",
                    shaders: nbn::GraphicsPipelineShaders::Legacy {
                        vertex: nbn::ShaderDesc {
                            module: &debugging_module,
                            entry_point: c"meshlet_debug_vertex",
                        },
                        fragment: nbn::ShaderDesc {
                            module: &debugging_module,
                            entry_point: c"model_debug_fragment",
                        },
                    },
                    conservative_rasterization: false,
                    cull_mode: vk::CullModeFlags::BACK,
                    blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA)],
                    depth: nbn::GraphicsPipelineDepthDesc {
                        write_enable: true,
                        test_enable: true,
                        compare_op: vk::CompareOp::GREATER,
                        format: vk::Format::D32_SFLOAT,
                    },
                    color_attachment_formats: &[swapchain.create_info.image_format],
                },
            ),
            alloc_vis: gpu_allocator::vulkan::AllocatorVisualizer::new(),
            egui_render: nbn::egui::Renderer::new(
                &device,
                swapchain.create_info.image_format,
                1024 * 1024,
            ),
            egui_winit: egui_winit::State::new(
                egui::Context::default(),
                egui::ViewportId::ROOT,
                event_loop,
                Some(window.scale_factor() as _),
                None,
                None,
            ),
            camera_rig,
            cursor_grabbed: false,
            prefix_sum_values: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "prefix_sum_values",
                    size: 8 * TOTAL_NUM_INSTANCES_OF_TYPE * 2,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            dispatches: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "dispatches",
                    size: (4 * 4 * 2 + 8 * 2 + 4),
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
                },
            ),
            blit_pipelines: nbn::ReloadablePipeline::new(
                &device,
                device.load_reloadable_shader("shaders/compiled/resolve_visbuffer.spv"),
                &|device: &nbn::Device, shader| BlitPipelines {
                    resolve_visbuffer: device.create_compute_pipeline(shader, c"resolve_visbuffer"),
                    tonemap: device.create_compute_pipeline(shader, c"tonemap"),
                },
            ),
            framebuffers: Framebuffers::new(
                &device,
                vk::Extent3D {
                    width: size.width,
                    height: size.height,
                    depth: 1,
                },
            ),
            _gltf: gltf,
            instances: device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "instances",
                data: &instances,
            }),
            models: device.create_buffer_with_data(nbn::BufferInitDescriptor {
                name: "models",
                data: &models,
            }),
            meshlet_instances: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "meshlet_instances",
                    size: (std::mem::size_of::<MeshletInstance>() * TOTAL_NUM_VISIBLE_MESHLETS)
                        as _,
                    ty: nbn::MemoryLocation::GpuOnly,
                })
                .unwrap(),
            num_instances: instances.len() as u32,
            time: 0.0,
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
        let device = self.device.as_ref().unwrap();
        let state = self.window_state.as_mut().unwrap();

        let response = state.egui_winit.on_window_event(&state.window, &event);

        if response.consumed {
            return;
        }

        match event {
            winit::event::WindowEvent::Resized(new_size) => {
                state.swapchain.create_info.image_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };

                device.recreate_swapchain(&mut state.swapchain);
                unsafe { device.queue_wait_idle(*device.graphics_queue).unwrap() };

                state.framebuffers = Framebuffers::new(
                    &device,
                    vk::Extent3D {
                        width: new_size.width,
                        height: new_size.height,
                        depth: 1,
                    },
                );

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
                rendering::render(&device, state);
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
                        .driver_mut::<YawPitch>()
                        .rotate_yaw_pitch(-x as f32 / 10.0, -y as f32 / 10.0);
                }
                _ => {}
            }
        }
    }

    fn exiting(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        unsafe {
            let device = self.device.as_ref().unwrap();
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
