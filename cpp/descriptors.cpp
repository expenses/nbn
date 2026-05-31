#include "nbn.h"

Descriptors::Descriptors(const vk::raii::Device& device) {
    uint32_t desc_count = 1024;

    auto pool_sizes = std::array {
        vk::DescriptorPoolSize {}
            .setType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(desc_count),
        vk::DescriptorPoolSize {}
            .setType(vk::DescriptorType::eMutableEXT)
            .setDescriptorCount(desc_count),
    };

    pool = vk::raii::DescriptorPool(
        device,
        vk::DescriptorPoolCreateInfo {}
            .setFlags(
                vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind
                | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
            )
            .setMaxSets(1)
            .setPoolSizes(pool_sizes)
    );

    auto binding_flag_array = std::array {
        vk::DescriptorBindingFlags {
            vk::DescriptorBindingFlagBits::eUpdateAfterBind
        },
        vk::DescriptorBindingFlags {
            vk::DescriptorBindingFlagBits::eUpdateAfterBind
        },
    };

    auto binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo {}.setBindingFlags(
            binding_flag_array
        );

    auto storage_image_types = std::array {vk::DescriptorType::eStorageImage};

    auto allowed_type_0 =
        vk::MutableDescriptorTypeListEXT {}.setDescriptorTypes(
            storage_image_types
        );
    auto allowed_type_1 = vk::MutableDescriptorTypeListEXT {};
    auto allowed_types = std::array {allowed_type_0, allowed_type_1};

    auto mutable_create_info =
        vk::MutableDescriptorTypeCreateInfoEXT {}.setMutableDescriptorTypeLists(
            allowed_types
        );

    binding_flags.setPNext(&mutable_create_info);

    auto bindings = std::array {
        vk::DescriptorSetLayoutBinding {}
            .setBinding(0)
            .setStageFlags(vk::ShaderStageFlagBits::eAll)
            .setDescriptorType(vk::DescriptorType::eMutableEXT)
            .setDescriptorCount(desc_count),
        vk::DescriptorSetLayoutBinding {}
            .setBinding(1)
            .setStageFlags(vk::ShaderStageFlagBits::eAll)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(desc_count),
    };

    layout = vk::raii::DescriptorSetLayout(
        device,
        vk::DescriptorSetLayoutCreateInfo {}
            .setFlags(
                vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool
            )
            .setBindings(bindings)
            .setPNext(&binding_flags)
    );

    auto descriptor_sets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo {}.setDescriptorPool(*pool).setSetLayouts(
            *layout
        )
    );

    set = *descriptor_sets[0];
}
