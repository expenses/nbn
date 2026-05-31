struct Device {
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    vk::Device device;
    vk::Queue graphics_queue;
    vk::Queue compute_queue;
    vk::Queue transfer_queue;
    
    Device();
};
