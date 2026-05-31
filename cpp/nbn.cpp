struct Device {
    Device() {
        const vk::ApplicationInfo appInfo = {
            .pApplicationName = "nbn",
            .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName = "nbn",
            .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4};

        VULKAN_HPP_DEFAULT_DISPATCHER.init();

        vk::raii::Context context;

        std::array<const char*, 0> layers;
        std::array<const char*, 0> instance_extensions;

        vk::raii::Instance instance(
            context,
            vk::InstanceCreateInfo {
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<uint32_t>(layers.size()),
                .ppEnabledLayerNames = layers.data(),
                .enabledExtensionCount =
                    static_cast<uint32_t>(instance_extensions.size()),
                .ppEnabledExtensionNames = instance_extensions.data()}
        );

        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance, vkGetInstanceProcAddr);

    }

};
