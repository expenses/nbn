struct Device {
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    vk::Device device;
    vk::Queue graphics_queue;
    vk::Queue compute_queue;
    vk::Queue transfer_queue;

    static Device new_device() {
        VULKAN_HPP_DEFAULT_DISPATCHER.init();

        vk::ApplicationInfo appInfo {
            .pApplicationName = "nbn",
            .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName = "nbn",
            .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3,
        };
        const auto instance = vk::createInstance(
            vk::InstanceCreateInfo {.pApplicationInfo = &appInfo}
        );

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance, vkGetInstanceProcAddr);

        auto devices = instance.enumeratePhysicalDevices();

        auto score = [](vk::PhysicalDeviceType ty) {
            switch (ty) {
                case vk::PhysicalDeviceType::eDiscreteGpu:
                    return 0;
                case vk::PhysicalDeviceType::eIntegratedGpu:
                    return 1;
                case vk::PhysicalDeviceType::eVirtualGpu:
                    return 2;
                case vk::PhysicalDeviceType::eOther:
                    return 3;
                case vk::PhysicalDeviceType::eCpu:
                    return 4;
                default:
                    return 5;
            }
        };

        const auto best_physical_device = std::min_element(
            devices.begin(),
            devices.end(),
            [&](auto& a, auto& b) {
                return score(a.getProperties().deviceType)
                    < score(b.getProperties().deviceType);
            }
        );

        vk::PhysicalDevice physical_device = std::move(*best_physical_device);

        const auto properties = physical_device.getProperties();

        // auto feat = physical_device.getFeatures2<
        //     vk::PhysicalDeviceFeatures2,
        //     vk::PhysicalDeviceVulkan11Features,
        //     vk::PhysicalDeviceVulkan12Features,
        //     vk::PhysicalDeviceVulkan13Features>();

        // auto const& features_10 =
        //     feat.get<vk::PhysicalDeviceFeatures2>().features;
        // auto const& features_11 =
        //     feat.get<vk::PhysicalDeviceVulkan11Features>().features;
        // auto const& features_12 =
        //     feat.get<vk::PhysicalDeviceVulkan12Features>().features;
        // auto const& features_13 =
        //     feat.get<vk::PhysicalDeviceVulkan13Features>().features;

        // assert(features_10.shaderInt16);
        // assert(features_10.shaderInt64);
        // assert(features_10.geometryShader);
        // assert(features_10.samplerAnisotropy);
        // assert(features_10.multiDrawIndirect);
        // assert(features_10.fragmentStoresAndAtomics);
        // assert(features_10.vertexPipelineStoresAndAtomics);
        // assert(features_11.shaderDrawParameters);
        // assert(features_12.bufferDeviceAddress);
        // assert(features_12.shaderInt8);
        // assert(features_12.descriptorBindingSampledImageUpdateAfterBind);
        // assert(features_12.descriptorBindingStorageImageUpdateAfterBind);
        // assert(features_12.runtimeDescriptorArray);
        // assert(features_12.timelineSemaphore);
        // assert(features_12.shaderBufferInt64Atomics);
        // assert(features_12.shaderFloat16);
        // assert(features_12.vulkanMemoryModel);
        // assert(features_12.vulkanMemoryModelDeviceScope);
        // assert(features_12.drawIndirectCount);

        // assert(features_13.dynamicRendering);
        // assert(features_13.synchronization2);

        auto families = physical_device.getQueueFamilyProperties();

        uint32_t graphics_queue_family = 0;
        for (auto i = 0u; i < families.size(); i++)
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphics_queue_family = i;
                break;
            }

        uint32_t compute_queue_family = graphics_queue_family;
        for (auto i = 0u; i < families.size(); i++)
            if (i != graphics_queue_family
                && families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                compute_queue_family = i;
                break;
            }

        uint32_t transfer_queue_family = compute_queue_family;
        for (auto i = 0u; i < families.size(); i++)
            if (i != graphics_queue_family && i != compute_queue_family
                && families[i].queueFlags & vk::QueueFlagBits::eTransfer) {
                transfer_queue_family = i;
                break;
            }

        auto f13 = vk::PhysicalDeviceVulkan13Features{}
            .setSynchronization2(true)
            .setDynamicRendering(true);
        auto f12 = vk::PhysicalDeviceVulkan12Features{}
            .setPNext(&f13)
            .setBufferDeviceAddress(true)
            .setDescriptorBindingSampledImageUpdateAfterBind(true)
            .setDescriptorBindingStorageImageUpdateAfterBind(true)
            .setRuntimeDescriptorArray(true)
            .setTimelineSemaphore(true)
            .setShaderBufferInt64Atomics(true)
            .setShaderInt8(true)
            .setShaderFloat16(true)
            .setVulkanMemoryModel(true)
            .setVulkanMemoryModelDeviceScope(true)
            .setDrawIndirectCount(true);
        auto f11 = vk::PhysicalDeviceVulkan11Features{}
            .setPNext(&f12)
            .setShaderDrawParameters(true);
        auto enabled_features = vk::PhysicalDeviceFeatures{}
            .setShaderInt64(true)
            .setShaderInt16(true)
            .setSamplerAnisotropy(true)
            .setMultiDrawIndirect(true)
            .setFragmentStoresAndAtomics(true)
            .setVertexPipelineStoresAndAtomics(true)
            .setGeometryShader(true);
        float priority = 1.0f;
        std::array<vk::DeviceQueueCreateInfo, 3> queue_infos = {
            vk::DeviceQueueCreateInfo {
                .queueFamilyIndex = graphics_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            },
            vk::DeviceQueueCreateInfo {
                .queueFamilyIndex = compute_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            },
            vk::DeviceQueueCreateInfo {
                .queueFamilyIndex = transfer_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            }
        };

        const auto device = physical_device.createDevice(
            vk::DeviceCreateInfo{}
                .setQueueCreateInfoCount(static_cast<uint32_t>(queue_infos.size()))
                .setPQueueCreateInfos(queue_infos.data())
                .setEnabledExtensionCount(1)
                .setPpEnabledExtensionNames(
                    std::array{"VK_KHR_swapchain"}.data())
                .setPEnabledFeatures(&enabled_features)
                .setPNext(&f11)
        );

        const auto graphics_queue = device.getQueue(graphics_queue_family, 0);
        const auto compute_queue = device.getQueue(compute_queue_family, 0);
        const auto transfer_queue = device.getQueue(transfer_queue_family, 0);

        return {
            .instance = std::move(instance),
            .physical_device = physical_device,
            .properties = properties,
            .device = std::move(device),
            .graphics_queue = graphics_queue,
            .compute_queue = compute_queue,
            .transfer_queue = transfer_queue
        };
    }
};
