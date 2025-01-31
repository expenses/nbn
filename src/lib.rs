use ash::vk::DebugUtilsMessengerEXT;
use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use parking_lot::Mutex;
use std::borrow::Cow;
use std::ffi;
use std::ffi::c_char;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        _ => panic!(),
    };

    log::log!(
        target: "vulkan",
        level,
        "{message_type:?} [{message_id_name} ({message_id_number})] : {message}"
    );

    vk::FALSE
}

pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

pub struct Queue {
    inner: vk::Queue,
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

type Allocator = Arc<Mutex<gpu_allocator::vulkan::Allocator>>;

pub struct Descriptors {
    _pool: DescriptorPool,
    layout: DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    index: AtomicU32,
    linear_sampler: Sampler,
}

pub struct Device {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_callback: DebugUtilsMessengerEXT,
    debug_utils_loader: debug_utils::Instance,
    surface_loader: surface::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub descriptors: std::mem::ManuallyDrop<Descriptors>,
    allocator: std::mem::ManuallyDrop<Allocator>,
    pub graphics_queue: Queue,
    pub compute_queue: Queue,
    pub transfer_queue: Queue,
    pub swapchain_loader: ash::khr::swapchain::Device,
}

pub enum BufferType {
    Gpu,
    Upload,
    Download,
}

pub struct BufferDescriptor<'a> {
    pub name: &'a str,
    pub size: u64,
    pub ty: BufferType,
}

pub struct ImageDescriptor<'a> {
    pub name: &'a str,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

pub struct BufferInitDescriptor<'a, T> {
    pub name: &'a str,
    pub data: &'a [T],
}

impl Device {
    pub fn new(window: Option<&Window>) -> Self {
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

        let mut extension_names = Vec::new();
        if let Some(window) = window {
            extension_names.extend_from_slice(
                ash_window::enumerate_required_extensions(
                    window.display_handle().unwrap().as_raw(),
                )
                .unwrap(),
            );
        }
        extension_names.push(debug_utils::NAME.as_ptr());
        extension_names.push(surface::NAME.as_ptr());

        let create_flags = vk::InstanceCreateFlags::default();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error")
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

        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
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

        let (pdevice, _) = physical_devices
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

        let mut enabled_vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut enabled_vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();

        let mut enabled_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut enabled_vulkan_1_2_features)
            .push_next(&mut enabled_vulkan_1_3_features);

        unsafe { instance.get_physical_device_features2(pdevice, &mut enabled_features) };
        assert!(enabled_features.features.shader_int16 > 0);
        assert!(enabled_features.features.shader_int64 > 0);
        assert!(enabled_vulkan_1_2_features.buffer_device_address > 0);
        assert!(enabled_vulkan_1_2_features.shader_int8 > 0);
        assert!(enabled_vulkan_1_2_features.descriptor_binding_sampled_image_update_after_bind > 0);
        assert!(enabled_vulkan_1_2_features.runtime_descriptor_array > 0);
        assert!(enabled_vulkan_1_2_features.timeline_semaphore > 0);
        assert!(enabled_vulkan_1_3_features.dynamic_rendering > 0);
        assert!(enabled_vulkan_1_3_features.synchronization2 > 0);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(true)
            .shader_int64(true);

        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true);

        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let queue_infos = &[
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_queue_family_index)
                .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_queue_family_index)
                .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(transfer_queue_family_index)
                .queue_priorities(&[1.0]),
        ];

        let device: ash::Device = unsafe {
            instance
                .create_device(
                    pdevice,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(queue_infos)
                        .enabled_extension_names(&[swapchain::NAME.as_ptr()])
                        .enabled_features(&vulkan_1_0_features)
                        .push_next(&mut vulkan_1_2_features)
                        .push_next(&mut vulkan_1_3_features),
                    None,
                )
                .unwrap()
        };

        let descriptor_pool = DescriptorPool::from_raw(
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024)]),
                    None,
                )
            }
            .unwrap(),
            &device,
        );

        let descriptor_set_layout = DescriptorSetLayout::from_raw(
            unsafe {
                let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&[vk::DescriptorBindingFlags::UPDATE_AFTER_BIND]);

                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(&[vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024)])
                        .push_next(&mut flags),
                    None,
                )
            }
            .unwrap(),
            &device,
        );

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&[*descriptor_set_layout]),
            )
        }
        .unwrap()[0];

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

        Self {
            physical_device: pdevice,
            swapchain_loader: ash::khr::swapchain::Device::new(&instance, &device),
            instance,
            entry,
            debug_callback,
            debug_utils_loader,
            surface_loader,
            descriptors: std::mem::ManuallyDrop::new(Descriptors {
                _pool: descriptor_pool,
                set: descriptor_set,
                layout: descriptor_set_layout,
                index: AtomicU32::new(0),
                linear_sampler: Sampler::from_raw(
                    unsafe {
                        device.create_sampler(
                            &vk::SamplerCreateInfo::default()
                                .mag_filter(vk::Filter::LINEAR)
                                .min_filter(vk::Filter::LINEAR)
                                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                                .address_mode_w(vk::SamplerAddressMode::REPEAT),
                            None,
                        )
                    }
                    .unwrap(),
                    &device,
                ),
            }),
            graphics_queue: Queue::new(&device, graphics_queue_family_index),
            compute_queue: Queue::new(&device, compute_queue_family_index),
            transfer_queue: Queue::new(&device, transfer_queue_family_index),
            device,
            allocator: std::mem::ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
        }
    }

    pub fn create_swapchain(&self, window: &Window) -> Swapchain {
        let surface = self.create_surface(window);
        let size = window.inner_size();
        let surface_format = self.select_surface_format(*surface);

        let swapchain = unsafe {
            self.swapchain_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_array_layers(1)
                    .min_image_count(FRAMES_IN_FLIGHT as u32)
                    .present_mode(vk::PresentModeKHR::MAILBOX)
                    .image_extent(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    })
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .surface(*surface),
                None,
            )
        }
        .unwrap();

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
            surface: std::mem::ManuallyDrop::new(surface),
            swapchain: std::mem::ManuallyDrop::new(swapchain),
        }
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

    pub fn select_surface_format(&self, surface: vk::SurfaceKHR) -> vk::SurfaceFormatKHR {
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
        }
        .unwrap();

        formats
            .into_iter()
            .max_by_key(|format| match format.format {
                vk::Format::B8G8R8A8_SRGB => 2,
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
                        .extent(desc.extent)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .image_type(vk::ImageType::TYPE_2D)
                        .array_layers(1)
                        .mip_levels(1)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
                    None,
                )
            }
            .unwrap(),
            &self.device,
        );

        let memory_requirements = unsafe { self.device.get_image_memory_requirements(*image) };

        let allocator = (*self.allocator).clone();

        let allocation = allocator
            .lock()
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
            .level_count(1)
            .aspect_mask(vk::ImageAspectFlags::COLOR);

        let image_view = ImageView::from_raw(
            unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*image)
                        .subresource_range(subresource_range)
                        .format(desc.format)
                        .view_type(vk::ImageViewType::TYPE_2D),
                    None,
                )
            }
            .unwrap(),
            &self.device,
        );

        let index = self.descriptors.index.fetch_add(1, Ordering::Relaxed);

        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptors.set)
                    .dst_binding(0)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .sampler(*self.descriptors.linear_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(*image_view)])],
                &[],
            );
        }

        Image {
            image,
            extent: desc.extent,
            subresource_range,
            index,
            allocator,
            allocation,
        }
    }

    pub fn create_image_with_data(
        &self,
        desc: ImageDescriptor,
        bytes: &[u8],
        transition_to: QueueType,
    ) -> PendingImageUpload {
        let buffer = self.create_buffer_with_data(BufferInitDescriptor {
            name: desc.name,
            data: bytes,
        });
        let image = self.create_image(desc);
        let command_buffer = self.create_command_buffer(QueueType::Transfer);

        unsafe {
            self.begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
                .unwrap();

            vk_sync::cmd::pipeline_barrier(
                self,
                *command_buffer,
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[],
                    next_accesses: &[vk_sync::AccessType::TransferWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    range: image.subresource_range,
                    discard_contents: true,
                    image: *image.image,
                    src_queue_family_index: self.transfer_queue.index,
                    dst_queue_family_index: self.transfer_queue.index,
                }],
            );

            self.cmd_copy_buffer_to_image(
                *command_buffer,
                *buffer.buffer,
                *image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_extent(image.extent)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .layer_count(1)
                            .aspect_mask(vk::ImageAspectFlags::COLOR),
                    )],
            );

            vk_sync::cmd::pipeline_barrier(
                self,
                *command_buffer,
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::TransferWrite],
                    next_accesses: &[
                        vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                    ],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    range: image.subresource_range,
                    discard_contents: false,
                    image: *image.image,
                    src_queue_family_index: self.transfer_queue.index,
                    dst_queue_family_index: self.get_queue(transition_to).index,
                }],
            );

            self.end_command_buffer(*command_buffer).unwrap();
        }

        PendingImageUpload {
            image,
            _upload_buffer: buffer,
            command_buffer,
        }
    }

    pub fn create_buffer(&self, desc: BufferDescriptor) -> Buffer {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(desc.size).usage(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                ),
                None,
            )
        }
        .unwrap();

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocator = (*self.allocator).clone();

        let allocation = allocator
            .lock()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: desc.name,
                linear: true,
                requirements: memory_requirements,
                location: match desc.ty {
                    BufferType::Upload => gpu_allocator::MemoryLocation::CpuToGpu,
                    BufferType::Gpu => gpu_allocator::MemoryLocation::GpuOnly,
                    BufferType::Download => gpu_allocator::MemoryLocation::GpuToCpu,
                },
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .unwrap();

        let address = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Buffer {
            buffer: WrappedBuffer::from_raw(buffer, &self.device),
            allocation,
            allocator,
            address,
        }
    }

    pub fn create_buffer_with_data<T: Copy>(&self, desc: BufferInitDescriptor<T>) -> Buffer {
        let mut buffer = self.create_buffer(BufferDescriptor {
            name: desc.name,
            size: std::mem::size_of_val(desc.data)
                // Required min size for AMDVLK
                .next_multiple_of(16) as u64,
            ty: BufferType::Upload,
        });

        presser::copy_from_slice_to_offset_with_align(
            desc.data,
            &mut buffer.allocation,
            0,
            0, //buffer.alignment as usize,
        )
        .unwrap();

        buffer
    }

    pub fn create_compute_pipeline(&self, desc: ComputePipelineDesc) -> Pipeline {
        let bytes = std::fs::read(desc.path).unwrap();
        let spirv_slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };

        let shader_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv_slice),
                None,
            )
        }
        .unwrap();

        let pipeline_layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[*self.descriptors.layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        // todo: use device max.
                        .size(256)]),
                None,
            )
        }
        .unwrap();

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .name(desc.name)
                    .module(shader_module),
            )
            .layout(pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .unwrap();

        Pipeline {
            layout: PipelineLayout::from_raw(pipeline_layout, &self.device),
            pipeline: WrappedPipeline::from_raw(pipelines[0], &self.device),
            _modules: vec![ShaderModule::from_raw(shader_module, &self.device)],
        }
    }

    fn get_queue(&self, ty: QueueType) -> &Queue {
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
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

macro_rules! wrap_raii_struct {
    ($name:ident, $ty:ty, $func:expr) => {
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

wrap_raii_struct!(Fence, vk::Fence, ash::Device::destroy_fence);
wrap_raii_struct!(
    ShaderModule,
    vk::ShaderModule,
    ash::Device::destroy_shader_module
);
wrap_raii_struct!(WrappedPipeline, vk::Pipeline, ash::Device::destroy_pipeline);
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

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
    pub pool: CommandPool,
    device: ash::Device,
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(*self.pool, &[self.command_buffer]);
        }
    }
}

impl std::ops::Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.command_buffer
    }
}

pub struct ComputePipelineDesc<'a> {
    pub name: &'a ffi::CStr,
    pub path: &'a str,
}

pub struct Pipeline {
    pub layout: PipelineLayout,
    pub pipeline: WrappedPipeline,
    _modules: Vec<ShaderModule>,
}

impl std::ops::Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.allocator.lock().report_memory_leaks(log::Level::Error);

        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            std::mem::ManuallyDrop::drop(&mut self.descriptors);

            self.device.destroy_device(None);

            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Buffer {
    pub buffer: WrappedBuffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Allocator,
    address: u64,
}

impl Buffer {
    pub fn try_as_slice<T: Copy>(&self) -> Option<&[T]> {
        self.allocation.mapped_slice().map(cast_slice)
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
        self.allocator.lock().free(allocation).unwrap();
    }
}

pub struct Image {
    pub image: WrappedImage,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Allocator,
    index: u32,
    pub extent: vk::Extent3D,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl std::ops::Deref for Image {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let allocation = std::mem::take(&mut self.allocation);
        self.allocator.lock().free(allocation).unwrap();
    }
}

pub struct PendingImageUpload {
    _upload_buffer: Buffer,
    pub command_buffer: CommandBuffer,
    pub image: Image,
}

impl PendingImageUpload {
    pub fn into_inner(self) -> Image {
        self.image
    }
}

pub fn cast_slice<I: Copy, O: Copy>(slice: &[I]) -> &[O] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct CurrentFrame<'a> {
    frame: &'a mut FrameResources,
    timeline_semaphore: &'a Semaphore,
}

impl std::ops::Deref for CurrentFrame<'_> {
    type Target = FrameResources;

    fn deref(&self) -> &Self::Target {
        self.frame
    }
}

impl CurrentFrame<'_> {
    pub fn submit(&mut self, device: &Device, command_buffers: &[vk::CommandBufferSubmitInfo]) {
        self.frame.number += FRAMES_IN_FLIGHT as u64;
        unsafe {
            device.queue_submit2(
                *device.graphics_queue,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(command_buffers)
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(*self.frame.image_available_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*self.frame.render_finished_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                        vk::SemaphoreSubmitInfo::default()
                            .value(self.frame.number)
                            .semaphore(**self.timeline_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                    ])],
                vk::Fence::null(),
            )
        }
        .unwrap();
    }
}

pub struct SyncResources {
    frames: [FrameResources; FRAMES_IN_FLIGHT],
    timeline_semaphore: Semaphore,
    pub current_frame: usize,
}

impl SyncResources {
    pub fn wait_for_frame(&mut self, device: &Device) -> CurrentFrame {
        let frame = &mut self.frames[self.current_frame];

        unsafe {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[*self.timeline_semaphore])
                        .values(&[frame.number]),
                    !0,
                )
                .unwrap();
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        CurrentFrame {
            frame,
            timeline_semaphore: &self.timeline_semaphore,
        }
    }
}

pub struct FrameResources {
    pub image_available_semaphore: Semaphore,
    pub render_finished_semaphore: Semaphore,
    number: u64,
}

impl FrameResources {
    fn new(device: &Device, number: u64) -> Self {
        Self {
            number,
            image_available_semaphore: device.create_semaphore(),
            render_finished_semaphore: device.create_semaphore(),
        }
    }
}

pub struct Surface {
    surface: vk::SurfaceKHR,
    loader: surface::Instance,
}
impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

impl std::ops::Deref for Surface {
    type Target = vk::SurfaceKHR;

    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

pub struct WrappedSwapchain {
    swapchain: vk::SwapchainKHR,
    loader: swapchain::Device,
}
impl Drop for WrappedSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl std::ops::Deref for WrappedSwapchain {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

pub struct Swapchain {
    surface: std::mem::ManuallyDrop<Surface>,
    swapchain: std::mem::ManuallyDrop<WrappedSwapchain>,
    pub images: Vec<SwapchainImage>,
}

impl std::ops::Deref for Swapchain {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.swapchain);
            std::mem::ManuallyDrop::drop(&mut self.surface);
        }
    }
}

pub struct SwapchainImage {
    pub image: vk::Image,
    pub view: ImageView,
}
