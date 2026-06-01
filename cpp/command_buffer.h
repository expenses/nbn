
enum class BarrierOp {
    ColorAttachmentWrite,
    ColorAttachmentReadWrite,
    ComputeStorageWrite,
    ComputeStorageRead,
    ComputeStorageReadWrite,
    DepthStencilAttachmentReadWrite,
    TransferRead,
    TransferWrite,
    AllCommandsSampledRead,
    TransferOrBlitRead,
    TransferOrBlitWrite,
    IndirectParamRead,
    Present,
    Acquire,
    AllCommands,
};

struct BarrierOpInfo {
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2 access;
    vk::ImageLayout layout;
};

inline BarrierOpInfo barrier_op_info(BarrierOp op) {
    switch (op) {
        using enum BarrierOp;
        case ColorAttachmentWrite:
            return {
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::ImageLayout::eGeneral
            };
        case ColorAttachmentReadWrite:
            return {
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentRead
                    | vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::ImageLayout::eGeneral
            };
        case ComputeStorageWrite:
            return {
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageWrite,
                vk::ImageLayout::eGeneral
            };
        case ComputeStorageRead:
            return {
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageRead,
                vk::ImageLayout::eGeneral
            };
        case ComputeStorageReadWrite:
            return {
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageRead
                    | vk::AccessFlagBits2::eShaderStorageWrite,
                vk::ImageLayout::eGeneral
            };
        case DepthStencilAttachmentReadWrite:
            return {
                vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentRead
                    | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::ImageLayout::eGeneral
            };
        case TransferRead:
            return {
                vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead,
                vk::ImageLayout::eGeneral
            };
        case TransferWrite:
            return {
                vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite,
                vk::ImageLayout::eGeneral
            };
        case AllCommandsSampledRead:
            return {
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderSampledRead,
                vk::ImageLayout::eGeneral
            };
        case TransferOrBlitRead:
            return {
                vk::PipelineStageFlagBits2::eCopy
                    | vk::PipelineStageFlagBits2::eBlit,
                vk::AccessFlagBits2::eTransferRead,
                vk::ImageLayout::eGeneral
            };
        case TransferOrBlitWrite:
            return {
                vk::PipelineStageFlagBits2::eCopy
                    | vk::PipelineStageFlagBits2::eBlit,
                vk::AccessFlagBits2::eTransferWrite,
                vk::ImageLayout::eGeneral
            };
        case IndirectParamRead:
            return {
                vk::PipelineStageFlagBits2::eDrawIndirect,
                vk::AccessFlagBits2::eIndirectCommandRead,
                vk::ImageLayout::eGeneral
            };
        case Present:
            return {
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                {},
                vk::ImageLayout::ePresentSrcKHR
            };
        case Acquire:
            return {
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                {},
                vk::ImageLayout::eUndefined
            };
        case AllCommands:
            return {
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderRead
                    | vk::AccessFlagBits2::eShaderWrite
                    | vk::AccessFlagBits2::eColorAttachmentRead
                    | vk::AccessFlagBits2::eColorAttachmentWrite
                    | vk::AccessFlagBits2::eTransferRead
                    | vk::AccessFlagBits2::eTransferWrite
                    | vk::AccessFlagBits2::eDepthStencilAttachmentRead
                    | vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                    | vk::AccessFlagBits2::eShaderStorageRead
                    | vk::AccessFlagBits2::eShaderStorageWrite
                    | vk::AccessFlagBits2::eShaderSampledRead
                    | vk::AccessFlagBits2::eIndirectCommandRead,
                vk::ImageLayout::eGeneral
            };
    }
}

struct ImageInfo {
    vk::Image image;
    vk::ImageSubresourceRange subresource_range;
};

struct ImageBarrier {
    ImageInfo image;
    std::optional<BarrierOp> src;
    BarrierOp dst;
    uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
};

struct BufferBarrier {
    const Buffer* buffer;
    BarrierOp src;
    BarrierOp dst;
};

struct CommandBuffer {
    vk::raii::CommandPool pool = {nullptr};
    vk::raii::CommandBuffer buffer = {nullptr};
    vk::raii::PipelineLayout& pipeline_layout;
    QueueType ty;
    uint32_t queue_index = 0;

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

    template<size_t I, size_t B>
    void insert_pipeline_barriers(
        const std::array<ImageBarrier, I>& images,
        const std::array<BufferBarrier, B>& buffers
    ) {
        std::array<vk::ImageMemoryBarrier2, I> img_barrs {};
        for (size_t i = 0; i < I; i++) {
            auto& b = images[i];
            auto dst = barrier_op_info(b.dst);
            auto& barr = img_barrs[i];
            barr.setImage(b.image.image)
                .setSubresourceRange(b.image.subresource_range)
                .setDstStageMask(dst.stages)
                .setDstAccessMask(dst.access)
                .setSrcQueueFamilyIndex(b.src_queue_family_index)
                .setDstQueueFamilyIndex(b.dst_queue_family_index)
                .setNewLayout(dst.layout);

            if (b.src.has_value()) {
                auto src = barrier_op_info(*b.src);
                barr.setSrcStageMask(src.stages)
                    .setSrcAccessMask(src.access)
                    .setOldLayout(src.layout);
            }
        }

        std::array<vk::BufferMemoryBarrier2, B> buf_barrs {};
        for (size_t i = 0; i < B; i++) {
            auto& b = buffers[i];
            auto src = barrier_op_info(b.src);
            auto dst = barrier_op_info(b.dst);
            buf_barrs[i]
                .setSrcQueueFamilyIndex(queue_index)
                .setDstQueueFamilyIndex(queue_index)
                .setSrcStageMask(src.stages)
                .setSrcAccessMask(src.access)
                .setDstStageMask(dst.stages)
                .setDstAccessMask(dst.access)
                .setBuffer(b.buffer->buffer)
                .setSize(VK_WHOLE_SIZE);
        }

        buffer.pipelineBarrier2(
            vk::DependencyInfo {}
                .setImageMemoryBarriers(img_barrs)
                .setBufferMemoryBarriers(buf_barrs)
        );
    }
};
