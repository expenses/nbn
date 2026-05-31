#include "nbn.h"

Device::Device() {
    auto instance_extensions =
        std::array {"VK_KHR_surface", "VK_EXT_debug_utils"};

    vk::ApplicationInfo appInfo {
        .pApplicationName = "nbn",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName = "nbn",
        .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    instance = vk::raii::Instance(
        context,
        vk::InstanceCreateInfo {
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = instance_extensions.size(),
            .ppEnabledExtensionNames = instance_extensions.data()
        }
    );

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
        }
    };

    const auto best_physical_device =
        std::min_element(devices.begin(), devices.end(), [&](auto& a, auto& b) {
            return score(a.getProperties().deviceType)
                < score(b.getProperties().deviceType);
        });

    physical_device = std::move(*best_physical_device);

    properties = physical_device.getProperties();

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

    auto vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features {}
                                   .setSynchronization2(true)
                                   .setDynamicRendering(true);
    auto vulkan_1_2_features =
        vk::PhysicalDeviceVulkan12Features {}
            .setPNext(&vulkan_1_3_features)
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
    auto vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features {}
                                   .setPNext(&vulkan_1_2_features)
                                   .setShaderDrawParameters(true);
    auto mutable_descriptor_features =
        vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT {}
            .setMutableDescriptorType(true);

    mutable_descriptor_features.setPNext(&vulkan_1_1_features);

    auto ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR {}
                                  .setPNext(&mutable_descriptor_features)
                                  .setRayQuery(true);
    auto enabled_features = vk::PhysicalDeviceFeatures {}
                                .setShaderInt64(true)
                                .setShaderInt16(true)
                                .setSamplerAnisotropy(true)
                                .setMultiDrawIndirect(true)
                                .setFragmentStoresAndAtomics(true)
                                .setVertexPipelineStoresAndAtomics(true)
                                .setGeometryShader(true);
    float priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queue_infos;
    queue_infos.push_back(
        vk::DeviceQueueCreateInfo {
            .queueFamilyIndex = graphics_queue_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        }
    );
    if (compute_queue_family != graphics_queue_family) {
        queue_infos.push_back(
            vk::DeviceQueueCreateInfo {
                .queueFamilyIndex = compute_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            }
        );
    }
    if (transfer_queue_family != compute_queue_family) {
        queue_infos.push_back(
            vk::DeviceQueueCreateInfo {
                .queueFamilyIndex = transfer_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            }
        );
    }

    auto extensions = std::array {
        "VK_KHR_swapchain",
        "VK_EXT_mesh_shader",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_ray_query",
        "VK_KHR_acceleration_structure",
        "VK_KHR_deferred_host_operations",
        "VK_EXT_mutable_descriptor_type"
    };

    device = vk::raii::Device(
        physical_device,
        vk::DeviceCreateInfo {}
            .setQueueCreateInfoCount(static_cast<uint32_t>(queue_infos.size()))
            .setPQueueCreateInfos(queue_infos.data())
            .setEnabledExtensionCount(extensions.size())
            .setPpEnabledExtensionNames(extensions.data())
            .setPEnabledFeatures(&enabled_features)
            .setPNext(&ray_query_features)
    );

    graphics_queue = device.getQueue(graphics_queue_family, 0);
    compute_queue = device.getQueue(compute_queue_family, 0);
    transfer_queue = device.getQueue(transfer_queue_family, 0);

    descriptors = Descriptors(device);

    auto set_layouts = std::array {*descriptors.layout};

    auto push_constant_ranges =
        std::array {vk::PushConstantRange {}
                        .setStageFlags(vk::ShaderStageFlagBits::eAll)
                        .setOffset(0)
                        .setSize(properties.limits.maxPushConstantsSize)};

    pipeline_layout = vk::raii::PipelineLayout(
        device,
        vk::PipelineLayoutCreateInfo {}
            .setSetLayouts(set_layouts)
            .setPushConstantRanges(push_constant_ranges)
    );
}

ShaderModule Device::load_shader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to read " + path);
    }
    auto size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(
        reinterpret_cast<char*>(code.data()),
        static_cast<std::streamsize>(size)
    );

    return ShaderModule {
        device.createShaderModule(
            vk::ShaderModuleCreateInfo {}.setCodeSize(size).setPCode(
                code.data()
            )
        ),
        path,
    };
}

auto Device::create_compute_pipeline(
    const ShaderModule& module,
    const std::string& entry_point
) -> vk::raii::Pipeline {
    auto pipeline = device.createComputePipeline(
        VK_NULL_HANDLE,
        vk::ComputePipelineCreateInfo {}
            .setStage(
                vk::PipelineShaderStageCreateInfo {}
                    .setStage(vk::ShaderStageFlagBits::eCompute)
                    .setModule(*module.module)
                    .setPName(entry_point.c_str())
            )
            .setLayout(*pipeline_layout)
    );

    auto vk_pipeline = static_cast<VkPipeline>(*pipeline);
    device.setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT {}
            .setObjectType(vk::ObjectType::ePipeline)
            .setObjectHandle(reinterpret_cast<uint64_t>(vk_pipeline))
            .setPObjectName((module.path + " - " + entry_point).c_str())
    );

    return pipeline;
}
