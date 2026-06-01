#include "pch.h"

enum class QueueType { Graphics, Compute, Transfer };

struct Queue {
    vk::raii::Queue queue = {nullptr};
    uint32_t index = 0;

    vk::raii::Queue* operator->() {
        return &queue;
    }
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

struct ImageDescriptor {
    std::string name;
    vk::Format format;
    vk::Extent3D extent;
    vk::ImageUsageFlags usage;
    vk::ImageAspectFlags aspect_mask;
    uint32_t mip_levels;
    vk::ImageViewType view_type;
};

struct Image {
    vma::raii::Image image = {nullptr};
    vk::raii::ImageView view = {nullptr};
    vk::Extent3D extent;
    vk::ImageSubresourceRange subresource_range;
};

struct CommandBuffer {
    vk::raii::CommandPool pool = {nullptr};
    vk::raii::CommandBuffer buffer = {nullptr};
    vk::raii::PipelineLayout& pipeline_layout;
    QueueType ty;

    vk::raii::CommandBuffer* operator->() {
        return &buffer;
    }

    template<class T>
    void push_constants(T value) {
        buffer.pushConstants<T>(
            pipeline_layout,
            vk::ShaderStageFlagBits::eAll,
            0,
            value
        );
    }
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

    Image create_image(const ImageDescriptor& desc);

    Queue& get_queue(QueueType ty);

    CommandBuffer create_command_buffer(QueueType ty);

    vk::raii::Device* operator->() {
        return &device;
    }
};

void check_vk_result(vk::Result err);
