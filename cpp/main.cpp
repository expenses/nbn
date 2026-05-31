#include "device.h"

int main() {
    auto device = Device();
    auto shader = device.load_shader("../shaders/compiled/jungle.spv");
    auto pipeline = device.create_compute_pipeline(shader, "raytrace");
}
