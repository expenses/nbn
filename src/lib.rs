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

        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .storage_push_constant8(true);

        let mut vulkan_1_3_features =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
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

        let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

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
            device,
            allocator: Some(Arc::new(Mutex::new(allocator))),
            queue: present_queue,
            command_pool: pool,
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

pub struct Device {
    pub device: ash::Device,
    pub queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: Option<Allocator>,
}

pub enum BufferType {
    Gpu,
    Staging,
    Readback,
}

pub struct BufferDescriptor<'a> {
    pub name: &'a str,
    pub size: u64,
    pub ty: BufferType,
}

impl Device {
    pub fn create_buffer(&self, desc: BufferDescriptor) -> Buffer {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(desc.size).usage(
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
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
                    BufferType::Staging => gpu_allocator::MemoryLocation::CpuToGpu,
                    BufferType::Gpu => gpu_allocator::MemoryLocation::GpuOnly,
                    BufferType::Readback => gpu_allocator::MemoryLocation::GpuToCpu,
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
            buffer,
            allocation,
            allocator,
            device: self.device.clone(),
            address,
        }
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
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        // todo: use device max.
                        .size(256),
                ]),
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
            layout: pipeline_layout,
            device: self.device.clone(),
            pipeline: pipelines[0],
            modules: vec![shader_module],
        }
    }

    pub fn create_command_buffer(&self) -> CommandBuffer {
        let command_buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .command_buffer_count(1),
            )
        }
        .unwrap();

        CommandBuffer {
            command_buffer: command_buffers[0],
            device: self.device.clone(),
            pool: self.command_pool,
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
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    device: ash::Device,
    modules: Vec<vk::ShaderModule>,
}

impl std::ops::Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            for module in self.modules.drain(..) {
                self.device.destroy_shader_module(module, None);
            }

            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let allocator = self.allocator.take().unwrap();
        allocator.lock().report_memory_leaks(log::Level::Error);

        drop(allocator);

        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

pub struct Buffer {
    buffer: vk::Buffer,
    device: ash::Device,
    pub allocation: gpu_allocator::vulkan::Allocation,
    allocator: Allocator,
    address: u64,
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

        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}
