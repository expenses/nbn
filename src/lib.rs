use ash::prelude::VkResult;
pub use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::c_char;
use std::ffi::{self, CStr};
use std::io::Read;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
pub use vk_sync::{
    cmd::pipeline_barrier, AccessType, BufferBarrier, GlobalBarrier, ImageBarrier, ImageLayout,
};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub mod descriptors;
pub mod egui;
mod hot_reloading;
mod surface;
mod util;

pub use descriptors::ImageIndex;
pub use hot_reloading::*;
pub use surface::*;
pub use util::*;

#[derive(Clone, Copy)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

pub struct Queue {
    pub inner: vk::Queue,
    pub index: u32,
}

impl Queue {
    fn new(device: &ash::Device, index: u32) -> Self {
        Self {
            inner: unsafe { device.get_device_queue(index, 0) },
            index,
        }
    }
}

impl std::ops::Deref for Queue {
    type Target = vk::Queue;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct Allocator {
    pub inner: RwLock<gpu_allocator::vulkan::Allocator>,
    deallocation_mutex: Mutex<()>,
    deallocation_notifier: parking_lot::Condvar,
}

impl Allocator {
    pub fn wait_for_next_deallocation(&self) {
        self.deallocation_notifier
            .wait(&mut self.deallocation_mutex.lock());
    }

    pub fn generate_report(&self) -> gpu_allocator::AllocatorReport {
        self.inner.read().generate_report()
    }
}

pub use gpu_allocator::MemoryLocation;

pub struct BufferDescriptor<'a> {
    pub name: &'a str,
    pub size: u64,
    pub ty: MemoryLocation,
}

pub struct SampledImageDescriptor<'a> {
    pub name: &'a str,
    pub format: vk::Format,
    pub extent: ImageExtent,
}

#[derive(Clone, Copy)]
pub enum ImageExtent {
    D2 {
        width: u32,
        height: u32,
    },
    D2Layered {
        width: u32,
        height: u32,
        num_layers: u32,
    },
    D3 {
        width: u32,
        height: u32,
        depth: u32,
    },
}

impl From<vk::Extent2D> for ImageExtent {
    fn from(extent: vk::Extent2D) -> Self {
        ImageExtent::D2 {
            width: extent.width,
            height: extent.height,
        }
    }
}

impl From<vk::Extent3D> for ImageExtent {
    fn from(extent: vk::Extent3D) -> Self {
        if extent.depth > 1 {
            ImageExtent::D3 {
                width: extent.width,
                height: extent.height,
                depth: extent.depth,
            }
        } else {
            ImageExtent::D2 {
                width: extent.width,
                height: extent.height,
            }
        }
    }
}

impl From<ImageExtent> for vk::Extent3D {
    fn from(extent: ImageExtent) -> vk::Extent3D {
        match extent {
            ImageExtent::D2 { width, height } => vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            ImageExtent::D3 {
                width,
                height,
                depth,
            } => vk::Extent3D {
                width,
                height,
                depth,
            },
            ImageExtent::D2Layered {
                width,
                height,
                num_layers,
            } => vk::Extent3D {
                width,
                height,
                depth: num_layers,
            },
        }
    }
}

pub struct ImageDescriptor<'a> {
    pub name: &'a str,
    pub format: vk::Format,
    pub extent: ImageExtent,
    pub usage: vk::ImageUsageFlags,
    pub aspect_mask: vk::ImageAspectFlags,
    pub mip_levels: u32,
}

pub struct BufferInitDescriptor<'a, T> {
    pub name: &'a str,
    pub data: &'a [T],
}

pub struct Samplers {
    pub repeat: Sampler,
    pub clamp: Sampler,
    pub nearest_clamp: Sampler,
}

pub struct Device {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
    debug_utils_loader: ash::ext::debug_utils::Instance,
    surface_loader: ash::khr::surface::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub descriptors: ManuallyDrop<descriptors::Descriptors>,
    pub allocator: ManuallyDrop<Arc<Allocator>>,
    pub graphics_queue: Queue,
    pub compute_queue: Queue,
    pub transfer_queue: Queue,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub debug_utils_device_loader: ash::ext::debug_utils::Device,
    pub pipeline_layout: ManuallyDrop<PipelineLayout>,
    pub properties: vk::PhysicalDeviceProperties,
    pub mesh_shader_loader: ash::ext::mesh_shader::Device,
    pub acceleration_structure_loader: ash::khr::acceleration_structure::Device,
    pub samplers: ManuallyDrop<Samplers>,
}

impl Device {
    pub fn new(window: Option<&Window>, enable_debug_printf: bool) -> Self {
        let entry = ash::Entry::linked();

        let app_name = c"nbn";

        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut required_instance_extensions = Vec::new();
        if let Some(window) = window {
            required_instance_extensions.extend_from_slice(
                ash_window::enumerate_required_extensions(
                    window.display_handle().unwrap().as_raw(),
                )
                .unwrap(),
            );
        }
        required_instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        required_instance_extensions.push(ash::khr::surface::NAME.as_ptr());
        if enable_debug_printf {
            // Needed in order to direct debug printf statements to
            // `vulkan_debug_callback`. Use `VK_LAYER_PRINTF_TO_STDOUT=1` otherwise.
            required_instance_extensions.push(ash::ext::layer_settings::NAME.as_ptr());
        }

        let available_instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
        let available_instance_extensions: HashSet<_> = available_instance_extensions
            .iter()
            .filter_map(|ext| ext.extension_name_as_c_str().ok())
            .collect();

        let create_flags = vk::InstanceCreateFlags::default();

        let mut validation_features = vk::ValidationFeaturesEXT::default();
        if enable_debug_printf {
            validation_features = validation_features
                .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF]);
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&required_instance_extensions)
            .flags(create_flags)
            .push_next(&mut validation_features);

        let instance: ash::Instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(instance) => instance,
            Err(error) => {
                for extension in required_instance_extensions {
                    let extension = unsafe { CStr::from_ptr(extension) };
                    if !available_instance_extensions.contains(extension) {
                        dbg!(extension);
                    }
                }
                panic!("Instance creation error: {}", error);
            }
        };

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_callback = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };

        let (pdevice, properties) = physical_devices
            .into_iter()
            .map(|pdevice| {
                let properties = unsafe { instance.get_physical_device_properties(pdevice) };

                (pdevice, properties)
            })
            .min_by_key(|(_, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::OTHER => 3,
                vk::PhysicalDeviceType::CPU => 4,
                _ => 5,
            })
            .unwrap();

        let queues = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };

        let graphics_queue_family_index = queues
            .iter()
            .position(|info| info.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap();

        let compute_queue_family_index = queues
            .iter()
            .enumerate()
            .position(|(i, info)| {
                info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && i != graphics_queue_family_index
            })
            .unwrap_or(graphics_queue_family_index);

        let transfer_queue_family_index = queues
            .iter()
            .enumerate()
            .position(|(i, info)| {
                info.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && i != graphics_queue_family_index
                    && i != compute_queue_family_index
            })
            .unwrap_or(compute_queue_family_index);

        let graphics_queue_family_index = graphics_queue_family_index as u32;
        let compute_queue_family_index = compute_queue_family_index as u32;
        let transfer_queue_family_index = transfer_queue_family_index as u32;

        let mut enabled_vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::default();
        let mut enabled_vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut enabled_vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
        let mut enabled_mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut enabled_mutable_descriptor_features =
            vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();

        let mut enabled_acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();

        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();

        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();

        let mut enabled_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enabled_vulkan_1_1_features)
            .push_next(&mut enabled_vulkan_1_2_features)
            .push_next(&mut enabled_vulkan_1_3_features)
            .push_next(&mut enabled_mesh_shader_features)
            .push_next(&mut enabled_mutable_descriptor_features)
            .push_next(&mut enabled_acceleration_structure_features)
            .push_next(&mut ray_tracing_pipeline_features)
            .push_next(&mut ray_query_features);

        unsafe { instance.get_physical_device_features2(pdevice, &mut enabled_features) };
        assert!(enabled_features.features.shader_int16 > 0);
        assert!(enabled_features.features.shader_int64 > 0);
        assert!(enabled_features.features.geometry_shader > 0);
        assert!(enabled_features.features.multi_draw_indirect > 0);
        assert!(enabled_features.features.sampler_anisotropy > 0);
        assert!(enabled_vulkan_1_1_features.shader_draw_parameters > 0);
        assert!(enabled_vulkan_1_2_features.buffer_device_address > 0);
        assert!(enabled_vulkan_1_2_features.shader_int8 > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_sampled_image_update_after_bind > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_storage_image_update_after_bind > 0);
        assert!(enabled_vulkan_1_2_features.runtime_descriptor_array > 0);
        assert!(enabled_vulkan_1_2_features.timeline_semaphore > 0);
        assert!(enabled_vulkan_1_2_features.shader_buffer_int64_atomics > 0);
        assert!(enabled_vulkan_1_2_features.shader_float16 > 0);
        assert!(enabled_vulkan_1_3_features.dynamic_rendering > 0);
        assert!(enabled_vulkan_1_3_features.synchronization2 > 0);
        assert!(enabled_mesh_shader_features.mesh_shader > 0);
        assert!(enabled_mesh_shader_features.task_shader > 0);
        assert!(enabled_mutable_descriptor_features.mutable_descriptor_type > 0);
        assert!(enabled_acceleration_structure_features.acceleration_structure > 0);
        assert!(ray_tracing_pipeline_features.ray_tracing_pipeline > 0);
        assert!(ray_query_features.ray_query > 0);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(true)
            .shader_int64(true)
            .geometry_shader(true)
            .sampler_anisotropy(true)
            .multi_draw_indirect(true)
            // needed for debug printf
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true);

        let mut vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::default()
            // not strictly required, but means SV_InstanceID doesn't seem to do what we want
            // so we use SV_StartInstanceLocation.
            .shader_draw_parameters(true);

        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true)
            .shader_buffer_int64_atomics(true)
            .shader_float16(true)
            // Needed for debug printf.
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);

        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .mesh_shader(true)
            .task_shader(true);

        let mut mutable_descriptor_features =
            vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default()
                .mutable_descriptor_type(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let mut ray_query_features =
            vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);

        let mut queue_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&[1.0])];

        if compute_queue_family_index != graphics_queue_family_index {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_queue_family_index)
                    .queue_priorities(&[1.0]),
            );
        }

        if transfer_queue_family_index != compute_queue_family_index {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_queue_family_index)
                    .queue_priorities(&[1.0]),
            );
        }

        let device: ash::Device = unsafe {
            instance
                .create_device(
                    pdevice,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_infos)
                        .enabled_extension_names(&[
                            ash::khr::swapchain::NAME.as_ptr(),
                            ash::ext::conservative_rasterization::NAME.as_ptr(),
                            // Needed for debug printf.
                            ash::khr::shader_non_semantic_info::NAME.as_ptr(),
                            // ;)
                            ash::ext::mesh_shader::NAME.as_ptr(),
                            ash::ext::mutable_descriptor_type::NAME.as_ptr(),
                            // Ray tracing
                            ash::khr::acceleration_structure::NAME.as_ptr(),
                            ash::khr::deferred_host_operations::NAME.as_ptr(),
                            ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
                            ash::khr::ray_query::NAME.as_ptr(),
                        ])
                        .enabled_features(&vulkan_1_0_features)
                        .push_next(&mut vulkan_1_1_features)
                        .push_next(&mut vulkan_1_2_features)
                        .push_next(&mut vulkan_1_3_features)
                        .push_next(&mut mesh_shader_features)
                        .push_next(&mut mutable_descriptor_features)
                        .push_next(&mut acceleration_structure_features)
                        .push_next(&mut ray_tracing_pipeline_features)
                        .push_next(&mut ray_query_features),
                    None,
                )
                .unwrap()
        };

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                buffer_device_address: true,
                debug_settings: Default::default(),
                allocation_sizes: Default::default(),
            })
            .unwrap();

        let descriptors = ManuallyDrop::new(descriptors::Descriptors::new(&device));

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[*descriptors.layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::ALL)
                        .offset(0)
                        .size(properties.limits.max_push_constants_size)]),
                None,
            )
        }
        .unwrap();

        let create_sampler = |address_mode, filter| {
            Sampler::from_raw(
                unsafe {
                    device.create_sampler(
                        &vk::SamplerCreateInfo::default()
                            .anisotropy_enable(true)
                            .max_anisotropy(properties.limits.max_sampler_anisotropy)
                            .mag_filter(filter)
                            .min_filter(filter)
                            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                            .min_lod(0.0)
                            .max_lod(f32::MAX)
                            .address_mode_u(address_mode)
                            .address_mode_v(address_mode)
                            .address_mode_w(address_mode),
                        None,
                    )
                }
                .unwrap(),
                &device,
            )
        };

        let this = Self {
            physical_device: pdevice,
            swapchain_loader: ash::khr::swapchain::Device::new(&instance, &device),
            debug_utils_device_loader: ash::ext::debug_utils::Device::new(&instance, &device),
            mesh_shader_loader: ash::ext::mesh_shader::Device::new(&instance, &device),
            acceleration_structure_loader: ash::khr::acceleration_structure::Device::new(
                &instance, &device,
            ),
            instance,
            entry,
            debug_callback,
            debug_utils_loader,
            surface_loader,
            pipeline_layout: ManuallyDrop::new(PipelineLayout::from_raw(pipeline_layout, &device)),
            descriptors,
            graphics_queue: Queue::new(&device, graphics_queue_family_index),
            compute_queue: Queue::new(&device, compute_queue_family_index),
            transfer_queue: Queue::new(&device, transfer_queue_family_index),
            samplers: ManuallyDrop::new(Samplers {
                repeat: create_sampler(vk::SamplerAddressMode::REPEAT, vk::Filter::LINEAR),
                clamp: create_sampler(vk::SamplerAddressMode::CLAMP_TO_EDGE, vk::Filter::LINEAR),
                nearest_clamp: create_sampler(
                    vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    vk::Filter::NEAREST,
                ),
            }),

            device,
            allocator: ManuallyDrop::new(Arc::new(Allocator {
                inner: RwLock::new(allocator),
                deallocation_notifier: Default::default(),
                deallocation_mutex: Default::default(),
            })),
            properties,
        };

        this.set_object_name(*this.samplers.repeat, "repeat_sampler");
        this.set_object_name(*this.samplers.clamp, "clamp_sampler");
        this.set_object_name(*this.samplers.nearest_clamp, "nearest_clamp_sampler");

        this
    }

    pub fn get_memory_budget(&self) -> vk::PhysicalDeviceMemoryBudgetPropertiesEXT<'_> {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical_device, &mut props)
        };
        budget
    }

    pub fn get_remaining_memory(&self) -> Vec<i64> {
        let budget = self.get_memory_budget();
        let mut vec = Vec::with_capacity(vk::MAX_MEMORY_HEAPS);
        for i in 0..vk::MAX_MEMORY_HEAPS {
            if budget.heap_budget[i] != 0 {
                vec.push(budget.heap_budget[i] as i64 - budget.heap_usage[i] as i64);
            }
        }

        vec
    }

    pub fn create_swapchain(
        &self,
        window: &Window,
        image_usage: vk::ImageUsageFlags,
        require_non_srgb: bool,
    ) -> Swapchain {
        let surface = self.create_surface(window);
        let size = window.inner_size();
        let surface_format = self.select_surface_format(*surface, require_non_srgb);

        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, *surface)
        }
        .unwrap();

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .image_usage(image_usage)
            .image_array_layers(1)
            .min_image_count((FRAMES_IN_FLIGHT as u32).max(surface_caps.min_image_count))
            .present_mode(vk::PresentModeKHR::FIFO)
            .image_extent(vk::Extent2D {
                width: size.width,
                height: size.height,
            })
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .surface(*surface);

        let swapchain =
            unsafe { self.swapchain_loader.create_swapchain(&create_info, None) }.unwrap();

        let swapchain_images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap()
        };

        let swapchain_images: Vec<_> = swapchain_images
            .into_iter()
            .map(|image| {
                let view = ImageView::from_raw(
                    unsafe {
                        self.create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(image)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                )
                                .format(surface_format.format)
                                .view_type(vk::ImageViewType::TYPE_2D),
                            None,
                        )
                        .unwrap()
                    },
                    self,
                );

                SwapchainImage { image, view }
            })
            .collect();

        let swapchain = WrappedSwapchain {
            swapchain,
            loader: self.swapchain_loader.clone(),
        };

        Swapchain {
            images: swapchain_images,
            surface: ManuallyDrop::new(surface),
            swapchain: ManuallyDrop::new(swapchain),
            create_info,
        }
    }

    pub fn recreate_swapchain(&self, swapchain: &mut Swapchain) {
        swapchain.create_info.old_swapchain = **swapchain.swapchain;

        let raw_swapchain = WrappedSwapchain {
            swapchain: unsafe {
                self.swapchain_loader
                    .create_swapchain(&swapchain.create_info, None)
            }
            .unwrap(),
            loader: self.swapchain_loader.clone(),
        };

        let swapchain_images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(*raw_swapchain)
                .unwrap()
        };

        swapchain.images = swapchain_images
            .into_iter()
            .map(|image| {
                let view = ImageView::from_raw(
                    unsafe {
                        self.create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(image)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .layer_count(1)
                                        .level_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                )
                                .format(swapchain.create_info.image_format)
                                .view_type(vk::ImageViewType::TYPE_2D),
                            None,
                        )
                        .unwrap()
                    },
                    self,
                );

                SwapchainImage { image, view }
            })
            .collect();

        unsafe {
            ManuallyDrop::drop(&mut swapchain.swapchain);
        }
        swapchain.swapchain = ManuallyDrop::new(raw_swapchain);
    }

    pub fn create_surface(&self, window: &Window) -> Surface {
        Surface {
            surface: unsafe {
                ash_window::create_surface(
                    &self.entry,
                    &self.instance,
                    window.display_handle().unwrap().as_raw(),
                    window.window_handle().unwrap().as_raw(),
                    None,
                )
            }
            .unwrap(),
            loader: self.surface_loader.clone(),
        }
    }

    pub fn select_surface_format(
        &self,
        surface: vk::SurfaceKHR,
        require_non_srgb: bool,
    ) -> vk::SurfaceFormatKHR {
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
        }
        .unwrap();

        formats
            .into_iter()
            .max_by_key(|format| match format.format {
                vk::Format::B8G8R8A8_SRGB => {
                    if require_non_srgb {
                        0
                    } else {
                        2
                    }
                }

                vk::Format::B8G8R8A8_UNORM => 1,
                _ => 0,
            })
            .unwrap()
    }

    pub fn create_image(&self, desc: ImageDescriptor) -> Image {
        let image = WrappedImage::from_raw(
            unsafe {
                self.device.create_image(
                    &vk::ImageCreateInfo::default()
                        .format(desc.format)
                        .extent(desc.extent.into())
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .image_type(match desc.extent {
                            ImageExtent::D2 { .. } => vk::ImageType::TYPE_2D,
                            ImageExtent::D3 { .. } | ImageExtent::D2Layered { .. } => {
                                vk::ImageType::TYPE_3D
                            }
                        })
                        .array_layers(1)
                        .mip_levels(desc.mip_levels)
                        .usage(desc.usage),
                    None,
                )
            }
            .unwrap(),
            &self.device,
        );

        self.set_object_name(*image, desc.name);

        let memory_requirements = unsafe { self.device.get_image_memory_requirements(*image) };

        let allocator = (*self.allocator).clone();

        let allocation = allocator
            .inner
            .write()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: desc.name,
                linear: false,
                requirements: memory_requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            self.device
                .bind_image_memory(*image, allocation.memory(), allocation.offset())
        }
        .unwrap();

        let subresource_range = vk::ImageSubresourceRange::default()
            .layer_count(1)
            .level_count(desc.mip_levels)
            .aspect_mask(desc.aspect_mask);

        let image_view = ImageView::from_raw(
            unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*image)
                        .subresource_range(subresource_range)
                        .format(desc.format)
                        .view_type(match desc.extent {
                            ImageExtent::D2 { .. } => vk::ImageViewType::TYPE_2D,
                            ImageExtent::D3 { .. } => vk::ImageViewType::TYPE_3D,
                            ImageExtent::D2Layered { .. } => vk::ImageViewType::TYPE_2D_ARRAY,
                        }),
                    None,
                )
            }
            .unwrap(),
            &self.device,
        );

        self.set_object_name(*image_view, desc.name);

        Image {
            image,
            view: image_view,
            extent: desc.extent.into(),
            subresource_range,
            allocator,
            allocation,
        }
    }

    fn get_image_tracker(&self, is_storage: bool) -> descriptors::ImageCountTracker {
        if is_storage {
            self.descriptors.storage_image_count.clone()
        } else {
            self.descriptors.sampled_image_count.clone()
        }
    }

    pub fn register_image_with_sampler(
        &self,
        view: vk::ImageView,
        sampler: &Sampler,
        is_storage: bool,
    ) -> ImageIndex {
        let index = ImageIndex::new(self.get_image_tracker(is_storage));

        if is_storage {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view);

            unsafe {
                self.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptors.set)
                        .dst_binding(1)
                        .dst_array_element(*index)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .image_info(&[image_info])],
                    &[],
                );
            }
        } else {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(**sampler);

            unsafe {
                self.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptors.set)
                        .dst_binding(0)
                        .dst_array_element(*index)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .image_info(&[image_info])],
                    &[],
                );
            }
        }

        index
    }

    pub fn register_image(&self, view: vk::ImageView, is_storage: bool) -> ImageIndex {
        self.register_image_with_sampler(view, &self.samplers.repeat, is_storage)
    }

    pub fn create_image_with_data_in_command_buffer(
        &self,
        desc: SampledImageDescriptor,
        bytes: &[u8],
        transition_to: QueueType,
        lod_offsets: &[u64],
        command_buffer: &CommandBuffer,
    ) -> (Buffer, Image) {
        let mip_levels = lod_offsets.len() as u32;

        let buffer = self.create_buffer_with_data(BufferInitDescriptor {
            name: desc.name,
            data: bytes,
        });
        let image = self.create_image(ImageDescriptor {
            name: desc.name,
            format: desc.format,
            extent: desc.extent,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_levels,
        });

        unsafe {
            self.insert_pipeline_barrier(
                command_buffer,
                None,
                &[],
                &[ImageBarrier2 {
                    previous_accesses: &[],
                    next_accesses: &[AccessType::TransferWrite],
                    discard_contents: true,
                    image: &image,
                    src_queue_family_index: command_buffer.queue_family_index,
                    dst_queue_family_index: command_buffer.queue_family_index,
                }
                .into()],
            );

            let copies: Vec<_> = lod_offsets
                .iter()
                .enumerate()
                .map(|(i, &offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_extent(vk::Extent3D {
                            width: image.extent.width >> i,
                            height: image.extent.height >> i,
                            depth: image.extent.depth,
                        })
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .layer_count(1)
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(i as _),
                        )
                })
                .collect();

            self.cmd_copy_buffer_to_image(
                **command_buffer,
                *buffer.buffer,
                *image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );

            self.insert_pipeline_barrier(
                command_buffer,
                None,
                &[],
                &[ImageBarrier2 {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer],
                    discard_contents: false,
                    image: &image,
                    src_queue_family_index: command_buffer.queue_family_index,
                    dst_queue_family_index: self.get_queue(transition_to).index,
                }
                .into()],
            );
        }

        (buffer, image)
    }

    pub fn create_sampled_image_with_data(
        &self,
        desc: SampledImageDescriptor,
        bytes: &[u8],
        transition_to: QueueType,
        lod_offsets: &[u64],
    ) -> PendingImageUpload {
        let command_buffer = self.create_command_buffer(QueueType::Transfer);

        unsafe {
            self.begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
                .unwrap();
        }

        let (buffer, image) = self.create_image_with_data_in_command_buffer(
            desc,
            bytes,
            transition_to,
            lod_offsets,
            &command_buffer,
        );

        unsafe {
            self.end_command_buffer(*command_buffer).unwrap();
        }

        let index = self.register_image(*image.view, false);

        PendingImageUpload {
            image: IndexedImage { image, index },
            _upload_buffer: buffer,
            command_buffer,
        }
    }

    pub fn insert_pipeline_barrier(
        &self,
        command_buffer: &CommandBuffer,
        global_barrier: Option<GlobalBarrier>,
        buffer_barriers: &[BufferBarrier],
        image_barriers: &[ImageBarrier],
    ) {
        vk_sync::cmd::pipeline_barrier(
            self,
            **command_buffer,
            global_barrier,
            buffer_barriers,
            image_barriers,
        );
    }

    pub fn insert_global_barrier(
        &self,
        command_buffer: &CommandBuffer,
        previous_accesses: &[vk_sync::AccessType],
        next_accesses: &[vk_sync::AccessType],
    ) {
        self.insert_pipeline_barrier(
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses,
                next_accesses,
            }),
            &[],
            &[],
        );
    }

    pub fn insert_image_barrier<T: Into<ImageInfo> + Copy>(
        &self,
        command_buffer: &CommandBuffer,
        barrier: ImageBarrier2<T>,
    ) {
        self.insert_pipeline_barrier(command_buffer, None, &[], &[barrier.into()]);
    }

    pub fn bind_internal_descriptor_sets(
        &self,
        command_buffer: &CommandBuffer,
        bind_point: vk::PipelineBindPoint,
    ) {
        unsafe {
            self.cmd_bind_descriptor_sets(
                **command_buffer,
                bind_point,
                **self.pipeline_layout,
                0,
                &[self.descriptors.set],
                &[],
            );
        }
    }

    pub fn bind_internal_descriptor_sets_to_all(&self, command_buffer: &CommandBuffer) {
        self.bind_internal_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS);
        self.bind_internal_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE);
    }

    pub fn push_constants<T: Copy>(&self, command_buffer: &CommandBuffer, data: T) {
        unsafe {
            self.cmd_push_constants(
                **command_buffer,
                **self.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                cast_slice(&[data]),
            );
        }
    }

    pub fn begin_rendering(
        &self,
        command_buffer: &CommandBuffer,
        width: u32,
        height: u32,
        color_attachments: &[vk::RenderingAttachmentInfo],
        depth_attachment: Option<&vk::RenderingAttachmentInfo>,
    ) {
        let render_area =
            vk::Rect2D::default().extent(vk::Extent2D::default().width(width).height(height));

        let mut rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(render_area)
            .color_attachments(color_attachments);

        if let Some(depth_attachment) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth_attachment);
        }

        unsafe {
            self.cmd_begin_rendering(**command_buffer, &rendering_info);

            self.cmd_set_viewport(
                **command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(width as f32)
                    .height(height as f32)
                    .max_depth(1.0)],
            );
            self.cmd_set_scissor(**command_buffer, 0, &[render_area]);
        }
    }

    pub fn create_buffer(&self, desc: BufferDescriptor) -> VkResult<Buffer> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(desc.size).usage(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                None,
            )
        }?;

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocator = (*self.allocator).clone();

        let desc = &gpu_allocator::vulkan::AllocationCreateDesc {
            name: desc.name,
            linear: true,
            requirements: memory_requirements,
            location: desc.ty,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = match allocator.inner.write().allocate(desc) {
            Ok(allocation) => allocation,
            Err(error) => {
                panic!("{}: {:?}", error, desc);
            }
        };

        self.set_object_name(unsafe { allocation.memory() }, desc.name);
        self.set_object_name(buffer, desc.name);

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .unwrap();

        let address = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Ok(Buffer {
            buffer: WrappedBuffer::from_raw(buffer, &self.device),
            allocation,
            allocator,
            address,
        })
    }

    pub fn create_buffer_from_reader<R: Read>(
        &self,
        size: u64,
        name: &str,
        mut reader: R,
    ) -> Buffer {
        let mut buffer = self
            .create_buffer(BufferDescriptor {
                name,
                size,
                ty: MemoryLocation::CpuToGpu,
            })
            .unwrap();

        let slice = buffer.try_as_slice_mut().unwrap();
        reader.read_exact(&mut slice[..size as usize]).unwrap();
        buffer
    }

    pub fn create_buffer_with_data<T: Copy>(&self, desc: BufferInitDescriptor<T>) -> Buffer {
        let mut buffer = self
            .create_buffer(BufferDescriptor {
                name: desc.name,
                size: std::mem::size_of_val(desc.data)
                    // Required min size for AMDVLK
                    .next_multiple_of(16) as u64,
                ty: MemoryLocation::CpuToGpu,
            })
            .unwrap();

        presser::copy_from_slice_to_offset_with_align(
            desc.data,
            &mut buffer.allocation,
            0,
            0, // buffer.alignment as usize,
        )
        .unwrap();

        buffer
    }

    pub fn load_reloadable_shader<P: AsRef<Path>>(&self, path: P) -> ReloadableShader {
        let path = path.as_ref();
        let dirty = Arc::new(AtomicBool::new(false));

        use notify::Watcher;

        let mut watcher = notify::RecommendedWatcher::new(
            {
                let dirty = dirty.clone();
                move |res| {
                    if let Ok(notify::Event {
                        kind: notify::EventKind::Modify(..),
                        ..
                    }) = res
                    {
                        dirty.store(true, Ordering::Relaxed);
                    }
                }
            },
            Default::default(),
        )
        .unwrap();

        watcher
            .watch(path, notify::RecursiveMode::NonRecursive)
            .unwrap();

        ReloadableShader {
            inner: self.load_shader(path),
            dirty,
            _watcher: watcher,
        }
    }

    pub fn load_shader<P: AsRef<Path>>(&self, path: P) -> ShaderModule {
        let path = path.as_ref();
        let bytes = std::fs::read(path).unwrap();
        let spirv_slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };

        let shader_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv_slice),
                None,
            )
        }
        .unwrap();

        self.set_object_name(shader_module, path.to_str().unwrap());

        ShaderModule {
            inner: WrappedShaderModule::from_raw(shader_module, &self.device),
            path: path.into(),
        }
    }

    pub fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> Pipeline {
        let mut dynamic_rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(desc.color_attachment_formats)
            .depth_attachment_format(desc.depth.format);

        let mut conservative_rasterization =
            vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
                .conservative_rasterization_mode(if desc.conservative_rasterization {
                    vk::ConservativeRasterizationModeEXT::OVERESTIMATE
                } else {
                    vk::ConservativeRasterizationModeEXT::DISABLED
                });

        let stages: &[vk::PipelineShaderStageCreateInfo] = match desc.shaders {
            GraphicsPipelineShaders::Legacy { vertex, fragment } => &[
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .name(vertex.entry_point)
                    .module(**vertex.module),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(fragment.entry_point)
                    .module(**fragment.module),
            ],
            GraphicsPipelineShaders::Mesh { mesh, fragment } => &[
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::MESH_EXT)
                    .name(mesh.entry_point)
                    .module(**mesh.module),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(fragment.entry_point)
                    .module(**fragment.module),
            ],
            GraphicsPipelineShaders::Task {
                task,
                mesh,
                fragment,
            } => &[
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::TASK_EXT)
                    .name(task.entry_point)
                    .module(**task.module),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::MESH_EXT)
                    .name(mesh.entry_point)
                    .module(**mesh.module),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(fragment.entry_point)
                    .module(**fragment.module),
            ],
        };

        let pipelines = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(stages)
                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                    .input_assembly_state(
                        &vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                    )
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::default()
                            .line_width(1.0)
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(desc.cull_mode)
                            .push_next(&mut conservative_rasterization),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .scissor_count(1)
                            .viewport_count(1),
                    )
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                            vk::DynamicState::VIEWPORT,
                            vk::DynamicState::SCISSOR,
                        ]),
                    )
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default()
                            .attachments(desc.blend_attachments),
                    )
                    .depth_stencil_state(
                        &vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_compare_op(desc.depth.compare_op)
                            .depth_write_enable(desc.depth.write_enable)
                            .depth_test_enable(desc.depth.test_enable),
                    )
                    .layout(**self.pipeline_layout)
                    .push_next(&mut dynamic_rendering)],
                None,
            )
        }
        .unwrap();

        let pipeline = pipelines[0];

        self.set_object_name(pipeline, desc.name);

        Pipeline::from_raw(pipeline, &self.device)
    }

    pub fn create_compute_pipeline(&self, module: &ShaderModule, entry_point: &CStr) -> Pipeline {
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .name(entry_point)
                    .module(**module),
            )
            .layout(**self.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .unwrap();

        let pipeline = pipelines[0];

        self.set_object_name(
            pipeline,
            &format!(
                "{} - {}",
                module.path.display(),
                entry_point.to_str().unwrap()
            ),
        );

        Pipeline::from_raw(pipeline, &self.device)
    }

    pub fn get_queue(&self, ty: QueueType) -> &Queue {
        match ty {
            QueueType::Graphics => &self.graphics_queue,
            QueueType::Compute => &self.compute_queue,
            QueueType::Transfer => &self.transfer_queue,
        }
    }

    pub fn create_command_buffer(&self, ty: QueueType) -> CommandBuffer {
        let queue_family_index = self.get_queue(ty).index;

        let pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

        let pool = CommandPool::from_raw(
            unsafe {
                self.device
                    .create_command_pool(&pool_create_info, None)
                    .unwrap()
            },
            &self.device,
        );

        let command_buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .command_buffer_count(1),
            )
        }
        .unwrap();

        CommandBuffer {
            command_buffer: command_buffers[0],
            device: self.device.clone(),
            pool,
            queue_family_index,
        }
    }

    pub fn create_fence(&self) -> Fence {
        Fence::from_raw(
            unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            },
            &self.device,
        )
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> Semaphore {
        let mut semaphore_type = vk::SemaphoreTypeCreateInfo::default()
            .initial_value(initial_value)
            .semaphore_type(vk::SemaphoreType::TIMELINE);

        Semaphore::from_raw(
            unsafe {
                self.device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type),
                        None,
                    )
                    .unwrap()
            },
            &self.device,
        )
    }

    pub fn create_semaphore(&self) -> Semaphore {
        Semaphore::from_raw(
            unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            },
            &self.device,
        )
    }

    pub fn reset_command_buffer(&self, command_buffer: &CommandBuffer) {
        unsafe {
            self.device
                .reset_command_pool(*command_buffer.pool, vk::CommandPoolResetFlags::empty())
        }
        .unwrap();
    }

    pub fn create_sync_resources(&self) -> SyncResources {
        SyncResources {
            timeline_semaphore: self.create_timeline_semaphore(2),
            current_frame: 0,
            frames: [
                FrameResources::new(self, 0),
                FrameResources::new(self, 1),
                FrameResources::new(self, 2),
            ],
        }
    }

    fn set_object_name<T: vk::Handle>(&self, object: T, name: &str) {
        let name = format!("{} ({})", name, std::any::type_name_of_val(&object));

        unsafe {
            self.debug_utils_device_loader
                .set_debug_utils_object_name(
                    &vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(object)
                        .object_name(&std::ffi::CString::new(name).unwrap()),
                )
                .unwrap();
        }
    }

    pub fn create_acceleration_structure(
        &self,
        name: &str,
        data: AccelerationStructureData,
        staging_buffer: &mut StagingBuffer,
    ) -> AccelerationStructure {
        let (geometry, ty, num_primitives) = match data {
            AccelerationStructureData::Triangles {
                index_type,
                vertices_buffer_address,
                indices_buffer_address,
                num_vertices,
                num_indices,
                opaque,
            } => (
                vk::AccelerationStructureGeometryKHR::default()
                    .flags(if opaque {
                        vk::GeometryFlagsKHR::OPAQUE
                    } else {
                        Default::default()
                    })
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .index_type(index_type)
                            .max_vertex(num_vertices - 1)
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: indices_buffer_address,
                            })
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: vertices_buffer_address,
                            })
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .vertex_stride(3 * 4),
                    }),
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                num_indices / 3,
            ),
            AccelerationStructureData::Instances {
                buffer_address,
                count,
            } => (
                vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                            .data(vk::DeviceOrHostAddressConstKHR {
                                device_address: buffer_address,
                            }),
                    }),
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                count,
            ),
        };

        let geometries = &[geometry];

        let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .ty(ty)
            .geometries(geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            self.acceleration_structure_loader
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    &[num_primitives],
                    &mut size_info,
                )
        };

        let scratch_data_offset = (*staging_buffer.staging_buffer)
            + staging_buffer.allocate_with_alignment(
                self,
                size_info.build_scratch_size as _,
                BufferAlignmentType::Address(128),
            ) as u64;

        let buffer = self
            .create_buffer(BufferDescriptor {
                name: &format!("{} backing buffer", name),
                size: size_info.acceleration_structure_size,
                ty: MemoryLocation::GpuOnly,
            })
            .unwrap();

        let acceleration_structure = WrappedAccelerationStructure {
            inner: unsafe {
                self.acceleration_structure_loader
                    .create_acceleration_structure(
                        &vk::AccelerationStructureCreateInfoKHR::default()
                            .size(size_info.acceleration_structure_size)
                            .buffer(*buffer.buffer)
                            .ty(ty),
                        None,
                    )
            }
            .unwrap(),
            loader: self.acceleration_structure_loader.clone(),
        };

        self.set_object_name(*acceleration_structure, name);

        let geometry_info = geometry_info
            .dst_acceleration_structure(*acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_data_offset,
            });

        unsafe {
            self.acceleration_structure_loader
                .cmd_build_acceleration_structures(
                    *staging_buffer.command_buffer,
                    &[geometry_info],
                    &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(num_primitives)]],
                );
        }

        self.insert_global_barrier(
            &staging_buffer.command_buffer,
            &[AccessType::AccelerationStructureBuildWrite],
            &[AccessType::AccelerationStructureBuildRead],
        );

        AccelerationStructure {
            address: unsafe {
                self.acceleration_structure_loader
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(*acceleration_structure),
                    )
            },
            _inner: acceleration_structure,
            _buffer: buffer,
        }
    }

    pub fn dispatch_command_pipeline<T: Copy>(
        &self,
        command_buffer: &CommandBuffer,
        pipeline: &Pipeline,
        push_constants: T,
        dispatch_width: u32,
        dispatch_height: u32,
        dispatch_depth: u32,
    ) {
        unsafe {
            self.cmd_bind_pipeline(**command_buffer, vk::PipelineBindPoint::COMPUTE, **pipeline)
        };
        self.push_constants(command_buffer, push_constants);
        unsafe {
            self.cmd_dispatch(
                **command_buffer,
                dispatch_width,
                dispatch_height,
                dispatch_depth,
            )
        };
    }
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.allocator
            .inner
            .read()
            .report_memory_leaks(log::Level::Error);

        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
            ManuallyDrop::drop(&mut self.descriptors);
            ManuallyDrop::drop(&mut self.pipeline_layout);
            ManuallyDrop::drop(&mut self.samplers);

            self.device.destroy_device(None);

            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

macro_rules! wrap_raii_struct {
    ($name : ident, $ty : ty, $func : expr) => {
        pub struct $name {
            pub inner: $ty,
            device: ash::Device,
        }

        impl $name {
            pub fn from_raw(inner: $ty, device: &ash::Device) -> Self {
                Self {
                    inner,
                    device: device.clone(),
                }
            }
        }

        impl std::ops::Deref for $name {
            type Target = $ty;

            fn deref(&self) -> &$ty {
                &self.inner
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    $func(&self.device, self.inner, None);
                }
            }
        }
    };
}

pub struct ShaderModule {
    inner: WrappedShaderModule,
    path: PathBuf,
}

impl Deref for ShaderModule {
    type Target = vk::ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

wrap_raii_struct!(Fence, vk::Fence, ash::Device::destroy_fence);
wrap_raii_struct!(
    WrappedShaderModule,
    vk::ShaderModule,
    ash::Device::destroy_shader_module
);
wrap_raii_struct!(Pipeline, vk::Pipeline, ash::Device::destroy_pipeline);
wrap_raii_struct!(WrappedBuffer, vk::Buffer, ash::Device::destroy_buffer);
wrap_raii_struct!(WrappedImage, vk::Image, ash::Device::destroy_image);
wrap_raii_struct!(ImageView, vk::ImageView, ash::Device::destroy_image_view);
wrap_raii_struct!(
    PipelineLayout,
    vk::PipelineLayout,
    ash::Device::destroy_pipeline_layout
);
wrap_raii_struct!(
    CommandPool,
    vk::CommandPool,
    ash::Device::destroy_command_pool
);
wrap_raii_struct!(
    DescriptorPool,
    vk::DescriptorPool,
    ash::Device::destroy_descriptor_pool
);
wrap_raii_struct!(
    DescriptorSetLayout,
    vk::DescriptorSetLayout,
    ash::Device::destroy_descriptor_set_layout
);
wrap_raii_struct!(Sampler, vk::Sampler, ash::Device::destroy_sampler);
wrap_raii_struct!(Semaphore, vk::Semaphore, ash::Device::destroy_semaphore);

struct WrappedAccelerationStructure {
    inner: vk::AccelerationStructureKHR,
    loader: ash::khr::acceleration_structure::Device,
}

impl Deref for WrappedAccelerationStructure {
    type Target = vk::AccelerationStructureKHR;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for WrappedAccelerationStructure {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_acceleration_structure(self.inner, None) }
    }
}

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
    pub pool: CommandPool,
    device: ash::Device,
    pub queue_family_index: u32,
}

impl std::ops::Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.command_buffer
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(*self.pool, &[self.command_buffer]);
        }
    }
}

pub struct ShaderDesc<'a> {
    pub entry_point: &'a ffi::CStr,
    pub module: &'a ShaderModule,
}

pub struct GraphicsPipelineDesc<'a> {
    pub name: &'a str,
    pub shaders: GraphicsPipelineShaders<'a>,
    pub color_attachment_formats: &'a [vk::Format],
    pub blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    pub conservative_rasterization: bool,
    pub cull_mode: vk::CullModeFlags,
    pub depth: GraphicsPipelineDepthDesc,
}

pub enum GraphicsPipelineShaders<'a> {
    Legacy {
        vertex: ShaderDesc<'a>,
        fragment: ShaderDesc<'a>,
    },
    Mesh {
        mesh: ShaderDesc<'a>,
        fragment: ShaderDesc<'a>,
    },
    Task {
        task: ShaderDesc<'a>,
        mesh: ShaderDesc<'a>,
        fragment: ShaderDesc<'a>,
    },
}

#[derive(Default)]
pub struct GraphicsPipelineDepthDesc {
    pub write_enable: bool,
    pub test_enable: bool,
    pub compare_op: vk::CompareOp,
    pub format: vk::Format,
}

pub struct Buffer {
    pub buffer: WrappedBuffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Arc<Allocator>,
    address: u64,
}

impl Buffer {
    pub fn try_as_slice<T: Copy>(&self) -> Option<&[T]> {
        self.allocation.mapped_slice().map(cast_slice)
    }

    pub fn try_as_slice_mut<T: Copy>(&mut self) -> Option<&mut [T]> {
        self.allocation.mapped_slice_mut().map(cast_slice_mut)
    }
}

impl std::ops::Deref for Buffer {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.address
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let allocation = std::mem::take(&mut self.allocation);
        self.allocator.inner.write().free(allocation).unwrap();
        self.allocator.deallocation_notifier.notify_all();
    }
}

pub struct IndexedImage {
    pub image: Image,
    pub index: ImageIndex,
}

impl std::ops::Deref for IndexedImage {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

pub struct Image {
    pub image: WrappedImage,
    pub view: ImageView,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Arc<Allocator>,
    pub extent: vk::Extent3D,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl std::ops::Deref for Image {
    type Target = WrappedImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let allocation = std::mem::take(&mut self.allocation);
        self.allocator.inner.write().free(allocation).unwrap();
        self.allocator.deallocation_notifier.notify_all();
    }
}

pub struct PendingImageUpload {
    _upload_buffer: Buffer,
    pub command_buffer: CommandBuffer,
    pub image: IndexedImage,
}

impl PendingImageUpload {
    pub fn into_inner(self) -> IndexedImage {
        self.image
    }
}

pub struct StagingBuffer {
    offset: usize,
    staging_buffer: Buffer,
    command_buffer: CommandBuffer,
    buffer_size: u64,
    queue_type: QueueType,
}

impl StagingBuffer {
    pub fn new(device: &Device, buffer_size: u64, queue_type: QueueType) -> Self {
        let command_buffer = device.create_command_buffer(queue_type);
        unsafe {
            device
                .begin_command_buffer(*command_buffer, &Default::default())
                .unwrap()
        };

        Self {
            queue_type,
            offset: 0,
            buffer_size,
            staging_buffer: device
                .create_buffer(BufferDescriptor {
                    size: buffer_size,
                    name: "staging buffer",
                    ty: MemoryLocation::CpuToGpu,
                })
                .unwrap(),
            command_buffer,
        }
    }

    pub fn flush(&mut self, device: &Device) {
        let fence = device.create_fence();
        unsafe {
            device.end_command_buffer(*self.command_buffer).unwrap();
            device
                .queue_submit(
                    **device.get_queue(self.queue_type),
                    &[vk::SubmitInfo::default().command_buffers(&[*self.command_buffer])],
                    *fence,
                )
                .unwrap();

            device.wait_for_fences(&[*fence], true, !0).unwrap();

            device.reset_command_buffer(&self.command_buffer);
            device
                .begin_command_buffer(*self.command_buffer, &Default::default())
                .unwrap()
        }
        self.offset = 0;
    }

    pub fn finish(self, device: &Device) {
        let fence = device.create_fence();
        unsafe {
            device.end_command_buffer(*self.command_buffer).unwrap();
            device
                .queue_submit(
                    **device.get_queue(self.queue_type),
                    &[vk::SubmitInfo::default().command_buffers(&[*self.command_buffer])],
                    *fence,
                )
                .unwrap();

            device.wait_for_fences(&[*fence], true, !0).unwrap();
        }
    }

    fn allocate_with_alignment(
        &mut self,
        device: &Device,
        size: usize,
        alignment: BufferAlignmentType,
    ) -> usize {
        self.offset = match alignment {
            BufferAlignmentType::Offset(offset) => self.offset.next_multiple_of(offset),
            BufferAlignmentType::Address(address_offset) => {
                (*self.staging_buffer as usize + self.offset).next_multiple_of(address_offset)
                    - *self.staging_buffer as usize
            }
        };
        if size > self.buffer_size as usize {
            self.flush(device);
            self.buffer_size = (self.buffer_size * 2).max(size as u64);
            self.staging_buffer = device
                .create_buffer(BufferDescriptor {
                    size: self.buffer_size,
                    name: "staging buffer",
                    ty: MemoryLocation::CpuToGpu,
                })
                .unwrap();
        } else if size > (self.buffer_size as usize).saturating_sub(self.offset) {
            self.flush(device);
        }
        let offset = self.offset;
        self.offset += size;
        offset
    }

    fn allocate<R: Read>(&mut self, device: &Device, size: usize, mut reader: R) -> usize {
        let offset = self.allocate_with_alignment(device, size, BufferAlignmentType::Offset(16));
        reader
            .read_exact(&mut self.staging_buffer.try_as_slice_mut().unwrap()[offset..offset + size])
            .unwrap();
        offset
    }

    pub fn create_buffer<R: Read>(
        &mut self,
        device: &Device,
        name: &str,
        size: usize,
        reader: R,
    ) -> Buffer {
        let offset = self.allocate(device, size, reader);
        let buffer = device
            .create_buffer(BufferDescriptor {
                name,
                size: size as _,
                ty: MemoryLocation::GpuOnly,
            })
            .unwrap();
        unsafe {
            device.cmd_copy_buffer(
                *self.command_buffer,
                *self.staging_buffer.buffer,
                *buffer.buffer,
                &[vk::BufferCopy {
                    src_offset: offset as _,
                    dst_offset: 0,
                    size: size as _,
                }],
            );
        }
        buffer
    }

    pub fn create_buffer_from_slice(&mut self, device: &Device, name: &str, data: &[u8]) -> Buffer {
        self.create_buffer(device, name, data.len(), std::io::Cursor::new(data))
    }

    pub fn create_sampled_image(
        &mut self,
        device: &Device,
        desc: SampledImageDescriptor,
        bytes: &[u8],
        transition_to: QueueType,
        lod_offsets: &[u64],
    ) -> Image {
        let offset = self.allocate(device, bytes.len() as _, std::io::Cursor::new(bytes));
        let mip_levels = lod_offsets.len() as u32;
        let image = device.create_image(ImageDescriptor {
            name: desc.name,
            format: desc.format,
            extent: desc.extent,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,

            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_levels,
        });
        device.insert_image_barrier(
            &self.command_buffer,
            ImageBarrier2 {
                image: &image,
                previous_accesses: &[],
                next_accesses: &[AccessType::TransferWrite],
                discard_contents: true,
                src_queue_family_index: device.get_queue(QueueType::Transfer).index,
                dst_queue_family_index: device.get_queue(QueueType::Transfer).index,
            },
        );
        unsafe {
            let copies: Vec<_> = lod_offsets
                .iter()
                .enumerate()
                .map(|(i, &lod_offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset as u64 + lod_offset)
                        .image_extent(vk::Extent3D {
                            width: image.extent.width >> i,
                            height: image.extent.height >> i,
                            depth: image.extent.depth,
                        })
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .layer_count(1)
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(i as _),
                        )
                })
                .collect();

            device.cmd_copy_buffer_to_image(
                *self.command_buffer,
                *self.staging_buffer.buffer,
                *image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
        }

        device.insert_image_barrier(
            &self.command_buffer,
            ImageBarrier2 {
                image: &image,
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer],
                discard_contents: false,
                src_queue_family_index: self.command_buffer.queue_family_index,
                dst_queue_family_index: device.get_queue(transition_to).index,
            },
        );
        image
    }
}

enum BufferAlignmentType {
    Address(usize),
    Offset(usize),
}

pub struct ImageInfo {
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
}

impl From<&Image> for ImageInfo {
    fn from(image: &Image) -> Self {
        Self {
            image: *image.image,
            subresource_range: image.subresource_range,
        }
    }
}

pub struct ImageBarrier2<'a, T> {
    pub image: T,
    pub previous_accesses: &'a [AccessType],
    pub next_accesses: &'a [AccessType],
    pub discard_contents: bool,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
}

impl<'a, T: Into<ImageInfo> + Copy> From<ImageBarrier2<'a, T>> for ImageBarrier<'a> {
    fn from(barrier: ImageBarrier2<'a, T>) -> Self {
        let image_info: ImageInfo = barrier.image.into();
        Self {
            image: image_info.image,
            range: image_info.subresource_range,
            discard_contents: barrier.discard_contents,
            previous_accesses: barrier.previous_accesses,
            next_accesses: barrier.next_accesses,
            src_queue_family_index: barrier.src_queue_family_index,
            dst_queue_family_index: barrier.dst_queue_family_index,
            previous_layout: ImageLayout::Optimal,
            next_layout: ImageLayout::Optimal,
        }
    }
}

pub enum AccelerationStructureData {
    Instances {
        buffer_address: u64,
        count: u32,
    },
    Triangles {
        index_type: vk::IndexType,
        vertices_buffer_address: u64,
        indices_buffer_address: u64,
        num_vertices: u32,
        num_indices: u32,
        opaque: bool,
    },
}

pub struct AccelerationStructure {
    _inner: WrappedAccelerationStructure,
    _buffer: Buffer,
    address: u64,
}

impl Deref for AccelerationStructure {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.address
    }
}
