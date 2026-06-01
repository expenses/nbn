#include "pch.h"

enum class QueueType { Graphics, Compute, Transfer };

struct Queue {
    vk::raii::Queue handle = {nullptr};
    uint32_t index = 0;
};

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

struct Buffer {
    vma::raii::Buffer buffer;
    uint64_t addr;
    void* ptr;
};

struct CommandBuffer {
    vk::raii::CommandPool pool = {nullptr};
    vk::raii::CommandBuffer buffer = {nullptr};
    QueueType ty;
};

struct Device {
    vk::raii::Context context;
    vk::raii::Instance instance = {nullptr};
    vk::raii::PhysicalDevice physical_device = {nullptr};
    vk::PhysicalDeviceProperties properties;
    vk::raii::Device device = {nullptr};
    Queue graphics_queue;
    Queue compute_queue;
    Queue transfer_queue;
    vma::raii::Allocator allocator = {nullptr};
    Descriptors descriptors {};
    vk::raii::PipelineLayout pipeline_layout = {nullptr};

    Device();
    ShaderModule load_shader(const std::string& path);
    vk::raii::Pipeline create_compute_pipeline(
        const ShaderModule& module,
        const std::string& entry_point
    );

    Buffer create_buffer(
        uint64_t size,
        const std::string& name,
        vma::MemoryUsage usage
    );

    Queue& get_queue(QueueType ty);

    CommandBuffer create_command_buffer(QueueType ty);
};

void check_vk_result(vk::Result err);
