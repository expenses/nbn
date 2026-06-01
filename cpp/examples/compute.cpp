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
    auto command_buffer = device.create_command_buffer(QueueType::Compute);

    auto fence = device.device.createFence({});

    command_buffer.buffer.begin({});

    command_buffer.buffer.bindPipeline(
        vk::PipelineBindPoint::eCompute,
        pipeline
    );

    command_buffer.buffer.pushConstants<PushConstants>(
        device.pipeline_layout,
        vk::ShaderStageFlagBits::eAll,
        0,
        PushConstants {
            .buffer = reinterpret_cast<float*>(buffer.addr),
            .num_items = 64,
        }
    );

    command_buffer.buffer.dispatch(1, 1, 1);

    command_buffer.buffer.end(

    );

    auto command_buffers = std::array {*command_buffer.buffer};

    device.compute_queue.handle.submit(
        vk::SubmitInfo {}.setCommandBuffers(command_buffers),
        fence
    );

    check_vk_result(device.device.waitForFences({*fence}, true, 1000000));

    for (uint i = 0; i < 64; i++) {
        printf("%f\n", ptr[i]);
    }
}
