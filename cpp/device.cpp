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

    graphics_queue.queue = device.getQueue(graphics_queue_family, 0);
    graphics_queue.index = graphics_queue_family;
    compute_queue.queue = device.getQueue(compute_queue_family, 0);
    compute_queue.index = compute_queue_family;
    transfer_queue.queue = device.getQueue(transfer_queue_family, 0);
    transfer_queue.index = transfer_queue_family;

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

    allocator = vma::raii::Allocator {
        instance,
        device,
        vma::AllocatorCreateInfo {
            .flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress,
            .physicalDevice = physical_device
        }
    };

    auto make_sampler =
        [&](vk::Filter mag, vk::Filter min, vk::SamplerAddressMode addr) {
            return vk::raii::Sampler(
                device,
                vk::SamplerCreateInfo {}
                    .setMagFilter(mag)
                    .setMinFilter(min)
                    .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                    .setAddressModeU(addr)
                    .setAddressModeV(addr)
                    .setAddressModeW(addr)
                    .setMinLod(0)
                    .setMaxLod(VK_LOD_CLAMP_NONE)
            );
        };

    samplers = Samplers {
        .repeat = make_sampler(
            vk::Filter::eLinear,
            vk::Filter::eLinear,
            vk::SamplerAddressMode::eRepeat
        ),
        .clamp = make_sampler(
            vk::Filter::eLinear,
            vk::Filter::eLinear,
            vk::SamplerAddressMode::eClampToEdge
        ),
        .nearest_clamp = make_sampler(
            vk::Filter::eNearest,
            vk::Filter::eNearest,
            vk::SamplerAddressMode::eClampToEdge
        ),
        .nearest_repeat = make_sampler(
            vk::Filter::eNearest,
            vk::Filter::eNearest,
            vk::SamplerAddressMode::eRepeat
        ),
    };
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

vk::raii::Pipeline Device::create_compute_pipeline(
    const ShaderModule& module,
    const std::string& entry_point
) {
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

Buffer Device::create_buffer(
    uint64_t size,
    const std::string& name,
    vma::MemoryUsage usage
) {
    auto buffer = allocator.createBuffer(
        vk::BufferCreateInfo {}.setSize(size).setUsage(
            vk::BufferUsageFlagBits::eTransferSrc
            | vk::BufferUsageFlagBits::eShaderDeviceAddress
        ),
        vma::AllocationCreateInfo {
            .flags = vma::AllocationCreateFlagBits::eMapped,
            .usage = usage
        }
    );

    auto addr = device.getBufferAddress({.buffer = buffer});

    auto vk_buffer = static_cast<VkBuffer>(*buffer);
    device.setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT {}
            .setObjectType(vk::ObjectType::eBuffer)
            .setObjectHandle(reinterpret_cast<uint64_t>(vk_buffer))
            .setPObjectName(name.c_str())
    );

    auto buffer_info = buffer.getAllocation().getInfo();

    return {
        .buffer = std::move(buffer),
        .addr = addr,
        .ptr = buffer_info.pMappedData
    };
}

static vk::ImageType view_type_to_image_type(vk::ImageViewType ty) {
    switch (ty) {
        case vk::ImageViewType::e1D:
        case vk::ImageViewType::e1DArray:
            return vk::ImageType::e1D;
        case vk::ImageViewType::e2D:
        case vk::ImageViewType::eCube:
        case vk::ImageViewType::eCubeArray:
            return vk::ImageType::e2D;
        case vk::ImageViewType::e2DArray:
        case vk::ImageViewType::e3D:
            return vk::ImageType::e3D;
    }
}

Image Device::create_image(const ImageDescriptor& desc) {
    auto img = allocator.createImage(
        vk::ImageCreateInfo {}
            .setFormat(desc.format)
            .setExtent(desc.extent)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setImageType(view_type_to_image_type(desc.view_type))
            .setArrayLayers(1)
            .setMipLevels(desc.mip_levels)
            .setUsage(desc.usage),
        vma::AllocationCreateInfo {}.setUsage(vma::MemoryUsage::eGpuOnly)
    );

    device.setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT {}
            .setObjectType(vk::ObjectType::eImage)
            .setObjectHandle(
                reinterpret_cast<uint64_t>(static_cast<VkImage>(*img))
            )
            .setPObjectName(desc.name.c_str())
    );

    auto subresource_range = vk::ImageSubresourceRange {}
                                 .setAspectMask(desc.aspect_mask)
                                 .setLevelCount(desc.mip_levels)
                                 .setLayerCount(1);

    auto view = vk::raii::ImageView(
        device,
        vk::ImageViewCreateInfo {}
            .setImage(*img)
            .setSubresourceRange(subresource_range)
            .setFormat(desc.format)
            .setViewType(desc.view_type)
    );

    device.setDebugUtilsObjectNameEXT(
        vk::DebugUtilsObjectNameInfoEXT {}
            .setObjectType(vk::ObjectType::eImageView)
            .setObjectHandle(
                reinterpret_cast<uint64_t>(static_cast<VkImageView>(*view))
            )
            .setPObjectName(desc.name.c_str())
    );

    return Image {
        .image = std::move(img),
        .view = std::move(view),
        .extent = desc.extent,
        .subresource_range = subresource_range,
    };
}

ImageIndex Device::register_image(vk::ImageView view, bool is_storage) {
    return register_image_with_sampler(view, *samplers.repeat, is_storage);
}

ImageIndex Device::register_image_with_sampler(
    vk::ImageView view,
    vk::Sampler sampler,
    bool is_storage
) {
    auto& tracker = is_storage ? descriptors.storage_image_count
                               : descriptors.sampled_image_count;
    auto index = tracker->add();

    auto idx = ImageIndex(index, tracker);

    auto image_info = vk::DescriptorImageInfo {}
                          .setImageLayout(vk::ImageLayout::eGeneral)
                          .setImageView(view);

    if (!is_storage) {
        image_info.setSampler(sampler);
    }

    device.updateDescriptorSets(
        vk::WriteDescriptorSet {}
            .setDstSet(descriptors.set)
            .setDstBinding(is_storage ? 0 : 1)
            .setDstArrayElement(index)
            .setDescriptorType(
                is_storage ? vk::DescriptorType::eStorageImage
                           : vk::DescriptorType::eCombinedImageSampler
            )
            .setDescriptorCount(1)
            .setImageInfo(image_info),
        {}
    );

    return idx;
}

IndexedImage Device::register_owned_image(Image image, bool is_storage) {
    return IndexedImage {
        .index = register_image(*image.view, is_storage),
        .image = std::move(image),
    };
}

Queue& Device::get_queue(QueueType ty) {
    switch (ty) {
        case QueueType::Graphics:
            return graphics_queue;
        case QueueType::Compute:
            return compute_queue;
        case QueueType::Transfer:
            return transfer_queue;
    }
}

CommandBuffer Device::create_command_buffer(QueueType ty) {
    auto pool = vk::raii::CommandPool(
        device,
        vk::CommandPoolCreateInfo {}.setQueueFamilyIndex(get_queue(ty).index)
    );

    auto cmd = std::move(device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo {}
            .setCommandPool(*pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1)
    )[0]);

    return CommandBuffer {
        .pool = std::move(pool),
        .buffer = std::move(cmd),
        .pipeline_layout = pipeline_layout,
        .ty = ty,
    };
}
