#include "pch.h"

struct Descriptors {
    vk::raii::DescriptorPool pool = {nullptr};
    vk::raii::DescriptorSetLayout layout = {nullptr};
    vk::DescriptorSet set {};

    Descriptors() = default;
    explicit Descriptors(const vk::raii::Device& device);
};

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
    Descriptors descriptors {};
    vk::raii::PipelineLayout pipeline_layout = {nullptr};

    Device();
    ShaderModule load_shader(const std::string& path);
    auto create_compute_pipeline(
        const ShaderModule& module,
        const std::string& entry_point
    ) -> vk::raii::Pipeline;
};
