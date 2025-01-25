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
use winit::raw_window_handle::HasDisplayHandle;
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

pub struct Instance {
    instance: ash::Instance,
    debug_callback: DebugUtilsMessengerEXT,
    debug_utils_loader: debug_utils::Instance,
}

impl Instance {
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
            extension_names.extend(
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

        Self {
            instance,
            debug_callback,
            debug_utils_loader,
        }
    }

    pub fn create_device(&self) -> Device {
        let pdevices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };

        let (pdevice, queue_family_index) = pdevices
            .iter()
            .find_map(|pdevice| {
                unsafe {
                    self.instance
                        .get_physical_device_queue_family_properties(*pdevice)
                }
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    log::info!("{:?}", info);
                    let supports_graphic_and_surface =
                        info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    if supports_graphic_and_surface {
                        Some((*pdevice, index))
                    } else {
                        None
                    }
                })
            })
            .expect("Couldn't find suitable device.");

        let queue_family_index = queue_family_index as u32;
        let device_extension_names_raw = [swapchain::NAME.as_ptr()];

        let priorities = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let mut enabled_vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default();

        let mut enabled_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut enabled_vulkan_1_2_features);

        unsafe {
            self.instance
                .get_physical_device_features2(pdevice, &mut enabled_features)
        };
        assert!(enabled_features.features.shader_int16 > 0);
        assert!(enabled_features.features.shader_int64 > 0);
        assert!(enabled_vulkan_1_2_features.buffer_device_address > 0);
        assert!(enabled_vulkan_1_2_features.shader_int8 > 0);
        //assert!(enabled_vulkan_1_2_features.storage_push_constant8 > 0);
        assert!(enabled_vulkan_1_2_features.runtime_descriptor_array > 0);

        let vulkan_1_0_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(true)
            .shader_int64(true);

        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            //.storage_push_constant8(true)
            .runtime_descriptor_array(true);
        //.descriptor_binding_partially_bound(true);

        let mut vulkan_1_3_features =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&vulkan_1_0_features)
            .push_next(&mut vulkan_1_2_features)
            .push_next(&mut vulkan_1_3_features);

        let device: ash::Device = unsafe {
            self.instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap()
        };

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = CommandPool::from_raw(
            unsafe { device.create_command_pool(&pool_create_info, None).unwrap() },
            &device,
        );

        let descriptor_pool = DescriptorPool::from_raw(
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
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
                //let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                //    .binding_flags(&[vk::DescriptorBindingFlags::PARTIALLY_BOUND]);

                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1024),
                    ]),
                    // .push_next(&mut flags),
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
                instance: self.instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                buffer_device_address: true,
                debug_settings: Default::default(),
                allocation_sizes: Default::default(),
            })
            .unwrap();

        Device {
            descriptors: std::mem::ManuallyDrop::new(Descriptors {
                pool: descriptor_pool,
                set: descriptor_set,
                layout: descriptor_set_layout,
                index: AtomicU32::new(0),
                linear_sampler: Sampler::from_raw(
                    unsafe { device.create_sampler(&vk::SamplerCreateInfo::default(), None) }
                        .unwrap(),
                    &device,
                ),
            }),
            device,

            allocator: Some(Arc::new(Mutex::new(allocator))),
            queue: present_queue,
            command_pool: std::mem::ManuallyDrop::new(pool),
            queue_family_index,
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

type Allocator = Arc<Mutex<gpu_allocator::vulkan::Allocator>>;

pub struct Descriptors {
    pool: DescriptorPool,
    layout: DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    index: AtomicU32,
    linear_sampler: Sampler,
}

pub struct Device {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub descriptors: std::mem::ManuallyDrop<Descriptors>,
    command_pool: std::mem::ManuallyDrop<CommandPool>,
    allocator: Option<Allocator>,
    pub queue_family_index: u32,
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

        let allocator = self.allocator.clone().unwrap();

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
            image: image,
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
    ) -> PendingImageUpload {
        let buffer = self.create_buffer_with_data(BufferInitDescriptor {
            name: desc.name,
            data: bytes,
        });
        let image = self.create_image(desc);
        let command_buffer = self.create_command_buffer();

        unsafe {
            self.begin_command_buffer(*command_buffer, &ash::vk::CommandBufferBeginInfo::default())
                .unwrap();

            self.insert_image_barrier(
                &command_buffer,
                &image,
                &[],
                &[vk_sync::AccessType::TransferWrite],
                true,
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

            self.insert_image_barrier(
                &command_buffer,
                &image,
                &[vk_sync::AccessType::TransferWrite],
                &[vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer],
                false,
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
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                ),
                None,
            )
        }
        .unwrap();

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocator = self.allocator.clone().unwrap();

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
            alignment: memory_requirements.alignment,
        }
    }

    pub fn create_buffer_with_data<T: Copy>(&self, desc: BufferInitDescriptor<T>) -> Buffer {
        let mut buffer = self.create_buffer(BufferDescriptor {
            name: desc.name,
            size: ((desc.data.len() * std::mem::size_of::<T>()) as u64)
                // Required min size for AMDVLK
                .next_multiple_of(16),
            ty: BufferType::Upload,
        });

        presser::copy_from_slice_to_offset_with_align(
            &desc.data,
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

    pub fn create_command_buffer(&self) -> CommandBuffer {
        let command_buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(**self.command_pool)
                    .command_buffer_count(1),
            )
        }
        .unwrap();

        CommandBuffer {
            command_buffer: command_buffers[0],
            device: self.device.clone(),
            pool: **self.command_pool,
        }
    }

    pub fn create_fence(&self) -> Fence {
        Fence {
            device: self.device.clone(),
            inner: unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            },
        }
    }

    pub fn insert_image_barrier(
        &self,
        command_buffer: &CommandBuffer,
        image: &Image,
        previous_accesses: &[vk_sync::AccessType],
        next_accesses: &[vk_sync::AccessType],
        discard_contents: bool,
    ) {
        vk_sync::cmd::pipeline_barrier(
            &*self,
            **command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses,
                next_accesses,
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                range: image.subresource_range,
                discard_contents,
                image: *image.image,
                src_queue_family_index: self.queue_family_index,
                dst_queue_family_index: self.queue_family_index,
            }],
        );
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
            fn from_raw(inner: $ty, device: &ash::Device) -> Self {
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

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
    pool: vk::CommandPool,
    device: ash::Device,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(self.pool, &[self.command_buffer]);
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
        let allocator = self.allocator.take().unwrap();
        allocator.lock().report_memory_leaks(log::Level::Error);
        drop(allocator);

        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.command_pool);
            std::mem::ManuallyDrop::drop(&mut self.descriptors);

            self.device.destroy_device(None);
        }
    }
}

pub struct Buffer {
    pub buffer: WrappedBuffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Allocator,
    address: u64,
    alignment: u64,
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
    pub image: Image,
    pub command_buffer: CommandBuffer,
}

impl PendingImageUpload {
    pub fn into_inner(self) -> Image {
        self.image
    }
}
