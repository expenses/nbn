use nbn::vk;
use std::sync::Arc;
use winit::{event::ElementState, keyboard::KeyCode};

slang_struct::slang_include!("shaders/lightmapper_structs.slang");

mod least_squares;

use clap::{Args, Parser};

#[derive(Args)]
struct CommonArgs {
    path: std::path::PathBuf,
    #[arg(short, long)]
    envmap: std::path::PathBuf,
    #[arg(long, default_value_t = 1.0)]
    envmap_strength: f32,
}

#[derive(Args)]
struct LightmapperArgs {
    width: u32,
    height: u32,
    #[arg(short, default_value_t = 1024)]
    num_samples: u32,
    #[arg(short, long, default_value = "out.exr")]
    output: String,
}

#[derive(Args)]
struct ViewerArgs {
    #[arg(short, long)]
    lightmap: Option<std::path::PathBuf>,
}

#[derive(clap::Subcommand)]
enum Mode {
    Lightmapper(#[command(flatten)] LightmapperArgs),
    Viewer(#[command(flatten)] ViewerArgs),
}

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    mode: Mode,
    #[command(flatten)]
    common: CommonArgs,
}

fn write_output(width: u32, height: u32, slice: &[f32], filename: &str, num_samples: f32) {
    let mut output_vec = slice.to_vec();

    output_vec.chunks_mut(4).for_each(|chunk| {
        for i in 0..3 {
            chunk[i] /= num_samples;
        }
    });

    image::ImageBuffer::<image::Rgba<f32>, &[f32]>::from_raw(width, height, &output_vec)
        .unwrap()
        .save(filename)
        .unwrap();
}

fn main() {
    env_logger::init();

    let args = Arguments::parse();

    match args.mode {
        Mode::Viewer(viewer_args) => {
            let event_loop = winit::event_loop::EventLoop::new().unwrap();
            event_loop
                .run_app(&mut App {
                    state: None,
                    args: args.common,
                    viewer_args,
                })
                .unwrap();
        }
        Mode::Lightmapper(lightmapper_args) => {
            lightmap(&args.common, &lightmapper_args);
        }
    }
}

fn lightmap(args: &CommonArgs, lightmapper_args: &LightmapperArgs) {
    let device = Arc::new(nbn::Device::new(None));

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 64 * 1024 * 1024, nbn::QueueType::Compute);

    let mut envmap = image::open(&args.envmap).unwrap().to_rgba32f();
    envmap.pixels_mut().for_each(|p| {
        p[0] *= args.envmap_strength;
        p[1] *= args.envmap_strength;
        p[2] *= args.envmap_strength;
    });

    let envmap_size = dbg!([envmap.width(), envmap.height()]);

    let alias_table =
        alias_table::construct(envmap.rows().enumerate().flat_map(|(row, pixels)| {
            let height = envmap.height() as usize;
            let theta = std::f32::consts::PI * (row as f32 + 0.5) / height as f32;
            let sin_theta = theta.sin();
            pixels.map(move |p| {
                use image::Pixel;
                (p.to_luma()[0] * sin_theta, sin_theta)
            })
        }));

    let alias_table = staging_buffer.create_buffer_from_slice(&device, "alias_table", &alias_table);

    let envmap = staging_buffer.create_sampled_image(
        &device,
        nbn::SampledImageDescriptor {
            name: "envmap",
            format: vk::Format::R32G32B32A32_SFLOAT,
            extent: [envmap.width(), envmap.height()].into(),
        },
        nbn::cast_slice(&*envmap),
        nbn::QueueType::Graphics,
        &[0],
    );
    let envmap = device.register_owned_image(envmap, false);

    let (gltf_data, model, lights) = load_gltf(&device, &mut staging_buffer, &args.path);

    let width = lightmapper_args.width;
    let height = lightmapper_args.height;
    let total_samples = lightmapper_args.num_samples;

    let mut output_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "output",
            size: width as u64 * height as u64 * 4 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let mut temp_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "temp buffer",
            size: width as u64 * height as u64 * 4 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    let location_bitmasks_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "location_bitmasks_buffer",
            size: width as u64 * height as u64 * 8,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let triangle_indices_buffer = device
        .create_buffer(nbn::BufferDescriptor {
            name: "triangle_indices_buffer",
            size: width as u64 * height as u64 * 4 * 64,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let model_buffer = staging_buffer.create_buffer_from_slice(&device, "models", &[model]);
    let num_lights = dbg!(lights.len() as _);
    let lights = staging_buffer.create_buffer_from_slice(&device, "lights", &lights);

    let tlas = device.create_tlas_from_instances(
        &mut staging_buffer,
        "model",
        &[nbn::AccelerationStructureInstance {
            acceleration_structure: *gltf_data.acceleration_structure,
            ..Default::default()
        }
        .to_vk()],
    );
    let uv_tlas = device.create_tlas_from_instances(
        &mut staging_buffer,
        "uv_tlas",
        &[nbn::AccelerationStructureInstance {
            acceleration_structure: *gltf_data.uv_acceleration_structure,
            ..Default::default()
        }
        .to_vk()],
    );

    let push_constants = PushConstants {
        extent: [width, height],
        lights: *lights,
        model: *model_buffer,
        output: *output_buffer,
        temp: *temp_buffer,
        num_lights,
        tlas_: *tlas.tlas,
        uv_tlas_: *uv_tlas.tlas,
        location_bitmasks: *location_bitmasks_buffer,
        triangle_indices: *triangle_indices_buffer,
        sample_index: 0,
        envmap: *envmap,
        envmap_size,
        alias_table: *alias_table,
    };

    let shader = device.load_shader("../shaders/compiled/lightmapper.spv");

    let compute_pipeline = device.create_compute_pipeline(&shader, c"lightmap");
    let dilation_pipeline = device.create_compute_pipeline(&shader, c"dilation");
    let check_locations_pipeline = device.create_compute_pipeline(&shader, c"check_locations");

    unsafe {
        device.cmd_bind_pipeline(
            *staging_buffer.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *check_locations_pipeline,
        );
        device.push_constants::<PushConstants>(&staging_buffer.command_buffer, push_constants);
        device.cmd_dispatch(
            *staging_buffer.command_buffer,
            width.div_ceil(8),
            height.div_ceil(8),
            1,
        );
    }

    staging_buffer.finish(&device);

    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);

    for sample_index in 0..total_samples {
        unsafe {
            device
                .begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                *compute_pipeline,
            );
            device.push_constants::<PushConstants>(
                &command_buffer,
                PushConstants {
                    sample_index,
                    ..push_constants
                },
            );
            device.cmd_dispatch(*command_buffer, width.div_ceil(8), height.div_ceil(8), 1);
            device.end_command_buffer(*command_buffer).unwrap();
        }

        dbg!(sample_index);
        device.submit_and_wait_on_command_buffer(&command_buffer);
        device.reset_command_buffer(&command_buffer);

        //if sample_index % 128 == 0 {
        //    write_output(width, height, temp_buffer.try_as_slice::<f32>().unwrap(), &format!("{}.exr", sample_index), (sample_index+1) as _);
        //}
    }

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.bind_internal_descriptor_sets(&command_buffer, vk::PipelineBindPoint::COMPUTE);
        device.push_constants::<PushConstants>(&command_buffer, push_constants);
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *dilation_pipeline,
        );
        device.cmd_dispatch(*command_buffer, width.div_ceil(8), height.div_ceil(8), 1);
        device.end_command_buffer(*command_buffer).unwrap();
    }

    device.submit_and_wait_on_command_buffer(&command_buffer);

    write_output(
        width,
        height,
        output_buffer.try_as_slice::<f32>().unwrap(),
        &lightmapper_args.output,
        total_samples as _,
    );

    /*
    let (solution_r, solution_g, solution_b, pixel_info) = least_squares::get_solutions(
        &least_squares::Coverage {
            coverage: temp_slice,
            width,
            height,
        },
        &gltf_data.seams,
        &output_slice,
    );

    for (i, pi) in pixel_info.iter().enumerate() {
        let index = (pi.y * width + pi.x) as usize;
        output_slice[index * 4] = solution_r[i];
        output_slice[index * 4 + 1] = solution_g[i];
        output_slice[index * 4 + 2] = solution_b[i];
        output_slice[index * 4 + 3] = 1.0;
    }

    image::ImageBuffer::<image::Rgba<f32>, &[f32]>::from_raw(width, height, output_slice)
        .unwrap()
        .save("seamless.exr")
        .unwrap();
    */
}

struct State {
    device: nbn::Device,
    window: winit::window::Window,
    swapchain: nbn::Swapchain,
    sync_resources: nbn::SyncResources,
    per_frame_command_buffers: [nbn::CommandBuffer; nbn::FRAMES_IN_FLIGHT],
    swapchain_image_heap_indices: Vec<nbn::ImageIndex>,
    render_pipeline: nbn::Pipeline,
    tlas: nbn::TlasWithInstances,
    _model: CombinedModel,
    freecam: nbn::freecam::FreeCam,
    envmap: nbn::IndexedImage,
    envmap_size: [u32; 2],
    model: nbn::Buffer,
    num_lights: u32,
    lights: nbn::Buffer,
    alias_table: nbn::Buffer,
    uniform_buffers: [nbn::Buffer; nbn::FRAMES_IN_FLIGHT],
    frame_index: u32,
    lightmap: Option<nbn::IndexedImage>,
}

struct App {
    state: Option<State>,
    args: CommonArgs,
    viewer_args: ViewerArgs,
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
            nbn::StagingBuffer::new(&device, 128 * 1024 * 1024, nbn::QueueType::Compute);

        let envmap = image::open(&self.args.envmap).unwrap().to_rgba32f();

        let alias_table =
            alias_table::construct(envmap.rows().enumerate().flat_map(|(row, pixels)| {
                let height = envmap.height() as usize;
                let theta = std::f32::consts::PI * (row as f32 + 0.5) / height as f32;
                let sin_theta = theta.sin();
                pixels.map(move |p| {
                    use image::Pixel;
                    (p.to_luma()[0] * sin_theta, sin_theta)
                })
            }));

        let alias_table =
            staging_buffer.create_buffer_from_slice(&device, "alias_table", &alias_table);

        let envmap_size = [envmap.width(), envmap.height()];
        let envmap = staging_buffer.create_sampled_image(
            &device,
            nbn::SampledImageDescriptor {
                name: "envmap",
                format: vk::Format::R32G32B32A32_SFLOAT,
                extent: [envmap.width(), envmap.height()].into(),
            },
            nbn::cast_slice(&*envmap),
            nbn::QueueType::Graphics,
            &[0],
        );
        let envmap = device.register_owned_image(envmap, false);

        let (gltf_data, model, lights) = load_gltf(&device, &mut staging_buffer, &self.args.path);

        let num_lights = dbg!(lights.len() as _);
        let lights = staging_buffer.create_buffer_from_slice(&device, "lights", &lights);

        let model_buffer = staging_buffer.create_buffer_from_slice(&device, "models", &[model]);

        let tlas = device.create_tlas_from_instances(
            &mut staging_buffer,
            "model",
            &[nbn::AccelerationStructureInstance {
                acceleration_structure: *gltf_data.acceleration_structure,
                ..Default::default()
            }
            .to_vk()],
        );

        let lightmap = self.viewer_args.lightmap.as_ref().map(|lightmap| {
            let lightmap = image::open(&lightmap).unwrap().to_rgba32f();
            device.register_owned_image(
                staging_buffer.create_sampled_image(
                    &device,
                    nbn::SampledImageDescriptor {
                        name: "lightmap",
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        extent: [lightmap.width(), lightmap.height()].into(),
                    },
                    nbn::cast_slice(&*lightmap),
                    nbn::QueueType::Graphics,
                    &[0],
                ),
                false,
            )
        });

        staging_buffer.finish(&device);

        let shader = device.load_shader("../shaders/compiled/lightmapper.spv");

        let render_pipeline = device.create_compute_pipeline(&shader, c"render");

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
            uniform_buffers: [
                device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "uniform_buffer_0",
                        size: std::mem::size_of::<PushConstants>() as _,
                        ty: nbn::MemoryLocation::CpuToGpu,
                    })
                    .unwrap(),
                device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "uniform_buffer_1",
                        size: std::mem::size_of::<PushConstants>() as _,
                        ty: nbn::MemoryLocation::CpuToGpu,
                    })
                    .unwrap(),
                device
                    .create_buffer(nbn::BufferDescriptor {
                        name: "uniform_buffer_2",
                        size: std::mem::size_of::<PushConstants>() as _,
                        ty: nbn::MemoryLocation::CpuToGpu,
                    })
                    .unwrap(),
            ],
            device,
            window,
            swapchain,
            render_pipeline,
            tlas,
            _model: gltf_data,
            freecam: nbn::freecam::FreeCam::new(Default::default()),
            envmap,
            envmap_size,
            model: model_buffer,
            num_lights,
            lights,
            alias_table,
            frame_index: 0,
            lightmap,
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
            }
            winit::event::WindowEvent::RedrawRequested => unsafe {
                let current_frame = state.sync_resources.current_frame;
                let command_buffer = &state.per_frame_command_buffers[current_frame];

                let extent = state.swapchain.create_info.image_extent;

                state.uniform_buffers[current_frame]
                    .try_as_slice_mut::<PushConstants>()
                    .unwrap()[0] = PushConstants {
                    extent: [extent.width, extent.height],
                    lights: *state.lights,
                    model: *state.model,
                    output: 0,
                    temp: 0,
                    num_lights: state.num_lights,
                    tlas_: *state.tlas.tlas,
                    uv_tlas_: 0,
                    location_bitmasks: 0,
                    sample_index: 0,
                    triangle_indices: 0,
                    envmap: *state.envmap,
                    envmap_size: state.envmap_size,
                    alias_table: *state.alias_table,
                };

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
                    .begin_command_buffer(**command_buffer, &vk::CommandBufferBeginInfo::default())
                    .unwrap();

                device.insert_image_pipeline_barrier(
                    command_buffer,
                    image,
                    Some(nbn::BarrierOp::Acquire),
                    nbn::BarrierOp::ColorAttachmentWrite,
                );

                device.bind_internal_descriptor_sets_to_all(command_buffer);

                device.cmd_bind_pipeline(
                    **command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *state.render_pipeline,
                );

                let (view, proj) = state
                    .freecam
                    .update(extent.width, extent.height, 1.0 / 60.0);

                device.push_constants(
                    command_buffer,
                    ViewerConstants {
                        base: *state.uniform_buffers[current_frame],
                        image: *state.swapchain_image_heap_indices[next_image as usize],
                        view_inv: view.inverse().to_cols_array(),
                        proj_inv: proj.inverse().to_cols_array(),
                        combined: (proj * view).to_cols_array(),
                        frame_index: state.frame_index,
                        lightmap: state
                            .lightmap
                            .as_ref()
                            .map(|lightmap| **lightmap)
                            .unwrap_or(u32::max_value()),
                    },
                );

                state.frame_index += 1;

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
    //assert!(buffer.is_none());
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
            let (pos, rotation) = if let goth_gltf::NodeTransform::Set {
                translation,
                rotation,
                ..
            } = node.transform()
            {
                (translation, rotation)
            } else {
                panic!()
            };

            let light = &gltf.extensions.khr_lights_punctual.as_ref().unwrap().lights[index];

            let (spotlight_angle_scale, spotlight_angle_offset) = light
                .spot
                .map(|spot| {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
                    let spotlight_angle_scale = 1.0
                        / 0.000001_f32
                            .max(spot.inner_cone_angle.cos() - spot.outer_cone_angle.cos());
                    let spotlight_angle_offset =
                        -spot.outer_cone_angle.cos() * spotlight_angle_scale;
                    (spotlight_angle_scale, spotlight_angle_offset)
                })
                .unwrap_or((0.0, 1.0));

            Light {
                position: pos,
                emission: (glam::Vec3::from(light.color) * light.intensity).into(),
                spotlight_angle_scale,
                spotlight_angle_offset,
                spotlight_direction: (glam::Quat::from_array(rotation) * glam::Vec3::Z).into(),
            }
        })
        .collect();

    let buffer = buffer.map(|buffer| buffer.to_vec()).unwrap_or_else(|| {
        std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap()
    });

    let images = gltf
        .images
        .iter()
        .map(|image| {
            let data = match image.buffer_view {
                Some(index) => {
                    let view = &gltf.buffer_views[index];
                    let slice = &buffer[view.byte_offset..view.byte_offset + view.byte_length];
                    image::load_from_memory(slice).unwrap().to_rgba8()
                }
                None => {
                    let path = path.with_file_name(image.uri.as_ref().unwrap());
                    image::open(&path).unwrap().to_rgba8()
                }
            };
            let image = staging_buffer.create_sampled_image(
                &device,
                nbn::SampledImageDescriptor {
                    name: image.uri.as_deref().unwrap_or("inline"),
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
                index: device.register_image_with_sampler(
                    *image.view,
                    &device.samplers.nearest_repeat,
                    false,
                ),
                image,
            }
        })
        .collect::<Vec<_>>();

    let materials: Vec<Material> = gltf
        .materials
        .iter()
        .map(|material| {
            dbg!(material);
            Material {
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
                emissive_image: material
                    .emissive_texture
                    .as_ref()
                    .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                    .unwrap_or(u32::MAX),
                flags: matches!(material.alpha_mode, goth_gltf::AlphaMode::Mask) as u32,
                base_color_factor: material.pbr_metallic_roughness.base_color_factor,
                normal_image: 0,
            }
        })
        .collect();

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
    let mut material_indices = Vec::new();
    let mut uv2s = Vec::new();
    let mut uv2s_3d = Vec::new();

    gltf.nodes
        .iter()
        .filter_map(|node| node.mesh.map(|mesh_index| (node, &gltf.meshes[mesh_index])))
        .for_each(|(_node, mesh)| {
            for primitive in mesh.primitives.iter() {
                let indices_accessor = &gltf.accessors[primitive.indices.unwrap()];
                match indices_accessor.component_type {
                    goth_gltf::ComponentType::UnsignedInt => {
                        indices.extend(
                            get_slice::<u32>(&buffer, &gltf, &indices_accessor)
                                [..indices_accessor.count]
                                .iter()
                                .map(|&index| positions.len() as u32 / 3 + index as u32),
                        );
                    }
                    _ => {
                        indices.extend(
                            get_slice::<u16>(&buffer, &gltf, &indices_accessor)
                                [..indices_accessor.count]
                                .iter()
                                .map(|&index| positions.len() as u32 / 3 + index as u32),
                        );
                    }
                }

                let get = |accessor_index: Option<usize>, size: usize, error: &str| {
                    let accessor = &gltf.accessors[accessor_index.expect(error)];
                    assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                    &get_slice::<f32>(&buffer, &gltf, accessor)[..accessor.count * size]
                };

                let positions_slice = get(primitive.attributes.position, 3, "positions");

                let uv2s_vec = match primitive.attributes.texcoord_1 {
                    Some(_) => get(primitive.attributes.texcoord_1, 2, "uv2s").to_vec(),
                    _ => vec![0.0; (positions_slice.len() / 3) * 2],
                };

                positions.extend_from_slice(positions_slice);
                uvs.extend_from_slice(get(primitive.attributes.texcoord_0, 2, "uvs"));
                uv2s.extend_from_slice(&uv2s_vec);
                uv2s_3d.extend(uv2s_vec.chunks(2).flat_map(|c| [c[0], c[1], 0.0]));
                normals.extend_from_slice(get(primitive.attributes.normal, 3, "normals"));

                material_indices.extend(
                    (0..indices_accessor.count / 3).map(|_| primitive.material.unwrap_or(0) as u32),
                );
            }
        });

    let seams = least_squares::find_seams(&indices, &positions, &normals, &uv2s);

    let num_vertices = positions.len() / 3;
    let num_indices = indices.len();
    let indices = staging_buffer.create_buffer_from_slice(device, "indices", &indices);
    let positions = staging_buffer.create_buffer_from_slice(device, "positions", &positions);
    let materials = staging_buffer.create_buffer_from_slice(device, "materials", &materials);

    let uv2s_3d = staging_buffer.create_buffer_from_slice(device, "uv2s_3d", &uv2s_3d);

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

    let uv_acceleration_structure = device.create_acceleration_structure(
        &format!("{} uv acceleration structure", path.display(),),
        nbn::AccelerationStructureData::Triangles {
            index_type: vk::IndexType::UINT32,
            opaque: true,
            vertices_buffer_address: *uv2s_3d,
            indices_buffer_address: *indices,
            num_vertices: num_vertices as _,
            num_indices: num_indices as _,
        },
        staging_buffer,
    );

    let uvs = staging_buffer.create_buffer_from_slice(device, "uvs", &uvs);
    let uv2s = staging_buffer.create_buffer_from_slice(device, "uv2s", &uv2s);
    let normals = staging_buffer.create_buffer_from_slice(device, "normals", &normals);
    let material_indices =
        staging_buffer.create_buffer_from_slice(device, "material_indices", &material_indices);

    let model = Model {
        positions: *positions,
        uvs: *uvs,
        uv2s: *uv2s,
        normals: *normals,
        indices: *indices,
        material_indices: *material_indices,
        flags: 1,
        num_indices: num_indices as _,
        materials: *materials,
    };

    (
        CombinedModel {
            acceleration_structure,
            uv_acceleration_structure,
            seams,
            _positions: positions,
            _indices: indices,
            _uvs: uvs,
            _uv2s: uv2s,
            _uv2s_3d: uv2s_3d,
            _normals: normals,
            _material_indices: material_indices,
            _images: images,
            _materials: materials,
        },
        model,
        lights,
    )
}

struct CombinedModel {
    acceleration_structure: nbn::AccelerationStructure,
    uv_acceleration_structure: nbn::AccelerationStructure,
    seams: Vec<least_squares::Seam>,
    _positions: nbn::Buffer,
    _indices: nbn::Buffer,
    _uvs: nbn::Buffer,
    _uv2s: nbn::Buffer,
    _uv2s_3d: nbn::Buffer,
    _normals: nbn::Buffer,
    _material_indices: nbn::Buffer,
    _images: Vec<nbn::IndexedImage>,
    _materials: nbn::Buffer,
}
