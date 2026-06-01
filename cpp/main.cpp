#include "nbn.h"

int main() {
    auto device = Device();
    auto shader = device.load_shader("../shaders/compiled/jungle.spv");
    auto pipeline = device.create_compute_pipeline(shader, "raytrace");
    auto a = device.create_buffer(1024, "wow", vma::MemoryUsage::eCpuToGpu);
    auto b = device.create_buffer(1024, "wow", vma::MemoryUsage::eGpuOnly);
}
