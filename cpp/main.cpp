#include "device.h"

int main() {
    auto device = Device();
    device.load_shader("../shaders/compiled/jungle.spv");
}
