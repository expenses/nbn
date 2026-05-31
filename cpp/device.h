#include "pch.h"

struct ShaderModule {
    vk::raii::ShaderModule module = {nullptr};
    std::string path;
};

struct Device {
    vk::raii::Context context;
    vk::raii::Instance instance = {nullptr};
    vk::raii::PhysicalDevice physical_device = {nullptr};
    vk::PhysicalDeviceProperties properties;
    vk::raii::Device device = {nullptr};
    vk::raii::Queue graphics_queue = {nullptr};
    vk::raii::Queue compute_queue = {nullptr};
    vk::raii::Queue transfer_queue = {nullptr};
    vk::raii::PipelineLayout pipeline_layout = {nullptr};

    Device();
    ShaderModule load_shader(const std::string& path);
    auto create_compute_pipeline(
        const ShaderModule& module,
        const std::string& entry_point
    ) -> vk::raii::Pipeline;
};
