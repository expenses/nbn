#include "core.h"
#include "command_buffer.h"
#include "descriptors.h"

struct Queue {
    vk::raii::Queue queue = {nullptr};
    uint32_t index = 0;

    vk::raii::Queue* operator->() {
        return &queue;
    }
};

struct ShaderModule {
    vk::raii::ShaderModule module = {nullptr};
    std::string path;
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

struct ImageIndex {
    uint32_t index = 0;
    std::shared_ptr<CountTracker> tracker;

    ImageIndex(uint32_t idx, std::shared_ptr<CountTracker> tr) {
        index = idx;
        tracker = std::move(tr);
    }

    ~ImageIndex() {
        if (tracker)
            tracker->remove(index);
    }
};

struct IndexedImage {
    ImageIndex index;
    Image image;
};

struct Samplers {
    vk::raii::Sampler repeat = {nullptr};
    vk::raii::Sampler clamp = {nullptr};
    vk::raii::Sampler nearest_clamp = {nullptr};
    vk::raii::Sampler nearest_repeat = {nullptr};
};

using GraphicsPipelineFlags = uint32_t;
inline constexpr GraphicsPipelineFlags
    GRAPHICS_PIPELINE_CONSERVATIVE_RASTERIZATION = 1;
inline constexpr GraphicsPipelineFlags GRAPHICS_PIPELINE_POINTS = 2;
inline constexpr GraphicsPipelineFlags GRAPHICS_PIPELINE_BACKFACE_CULLING = 4;

struct GraphicsPipelineDepthDesc {
    bool write_enable = false;
    bool test_enable = false;
    vk::CompareOp compare_op = vk::CompareOp::eLess;
    vk::Format format = vk::Format::eUndefined;
};

struct GraphicsPipelineDesc {
    std::string name;
    std::span<const vk::PipelineShaderStageCreateInfo> stages;
    std::span<const vk::Format> color_attachment_formats;
    std::span<const vk::PipelineColorBlendAttachmentState> blend_attachments;
    GraphicsPipelineFlags flags = 0;
    GraphicsPipelineDepthDesc depth;
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
    Samplers samplers {};

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

    vk::raii::Pipeline
    create_graphics_pipeline(const GraphicsPipelineDesc& desc);

    ImageIndex register_image(vk::ImageView view, bool is_storage);
    ImageIndex register_image_with_sampler(
        vk::ImageView view,
        vk::Sampler sampler,
        bool is_storage
    );
    IndexedImage register_owned_image(Image image, bool is_storage);

    Queue& get_queue(QueueType ty);

    CommandBuffer create_command_buffer(QueueType ty);

    vk::raii::Device* operator->() {
        return &device;
    }
};

void check_vk_result(vk::Result err);
