void check_vk_result(vk::Result err) {
    if (err != vk::Result::eSuccess) {
        printf("%s\n", vk::to_string(err).c_str());
        if (err < vk::Result::eSuccess) {
            abort();
        }
    }
}