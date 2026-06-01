#include "../../shaders/compute_ex_structs.slang"
#include "../nbn.h"

int main() {
    auto device = Device();
    auto shader = device.load_shader("../shaders/compiled/compute_ex.spv");
    auto pipeline = device.create_compute_pipeline(shader, "main");
    auto buffer =
        device.create_buffer(64 * 4, "floats", vma::MemoryUsage::eCpuToGpu);
    assert(buffer.ptr);
    auto ptr = (float*)buffer.ptr;
    for (uint i = 0; i < 64; i++) {
        ptr[i] = float(i) * 1.5;
    }

    auto image = device.create_image({
        .name = "image",
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D {.width=64,.height=64,.depth=1},
        .usage = vk::ImageUsageFlagBits::eStorage,
        .aspect_mask = vk::ImageAspectFlagBits::eColor,
        .mip_levels=1,
        .view_type = vk::ImageViewType::e2D
    });

    auto command_buffer = device.create_command_buffer(QueueType::Compute);

    auto fence = device->createFence({});

    command_buffer->begin({});

    command_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    command_buffer.push_constants(
        PushConstants {
            .buffer = reinterpret_cast<float*>(buffer.addr),
            .num_items = 64,
        }
    );

    command_buffer->dispatch(1, 1, 1);

    command_buffer->end();

    auto command_buffers = std::array {*command_buffer.buffer};

    device.compute_queue->submit(
        vk::SubmitInfo {}.setCommandBuffers(command_buffers),
        fence
    );

    check_vk_result(device->waitForFences({*fence}, true, 1000000));

    for (uint i = 0; i < 64; i++) {
        printf("%f\n", ptr[i]);
    }
}
