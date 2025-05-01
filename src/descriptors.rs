use crate::*;

#[derive(Default, Debug)]
pub struct CountTracker {
    next: u32,
    unused: Vec<u32>,
}

impl CountTracker {
    pub(crate) fn add(&mut self) -> u32 {
        if let Some(index) = self.unused.pop() {
            return index;
        }

        let index = self.next;
        self.next += 1;
        index
    }

    pub(crate) fn remove(&mut self, index: u32) {
        self.unused.push(index);
    }
}

pub type ImageCountTracker = Arc<parking_lot::Mutex<CountTracker>>;

pub struct ImageIndex {
    index: u32,
    tracker: ImageCountTracker,
}

impl ImageIndex {
    pub fn new(tracker: ImageCountTracker) -> Self {
        let index = tracker.lock().add();
        Self { index, tracker }
    }
}

impl Deref for ImageIndex {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Drop for ImageIndex {
    fn drop(&mut self) {
        self.tracker.lock().remove(self.index);
    }
}

pub struct Descriptors {
    _pool: DescriptorPool,
    pub(crate) layout: DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    pub sampled_image_count: ImageCountTracker,
    pub storage_image_count: ImageCountTracker,
}

impl Descriptors {
    pub fn new(device: &ash::Device) -> Self {
        let counts_of_each_descriptor_type = 1024;

        let descriptor_pool = DescriptorPool::from_raw(
            unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(counts_of_each_descriptor_type),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::MUTABLE_EXT)
                                .descriptor_count(counts_of_each_descriptor_type),
                        ]),
                    None,
                )
            }
            .unwrap(),
            device,
        );

        let descriptor_set_layout = DescriptorSetLayout::from_raw(
            unsafe {
                let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&[vk::DescriptorBindingFlags::UPDATE_AFTER_BIND; 2]);

                let allowed_descriptor_types_per_binding = [
                    vk::MutableDescriptorTypeListEXT::default(),
                    vk::MutableDescriptorTypeListEXT::default()
                        .descriptor_types(&[vk::DescriptorType::STORAGE_IMAGE]),
                ];

                let mut mutable_descriptor_create_info =
                    vk::MutableDescriptorTypeCreateInfoEXT::default()
                        .mutable_descriptor_type_lists(&allowed_descriptor_types_per_binding);

                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(&[
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(0)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(counts_of_each_descriptor_type),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(1)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                                .descriptor_type(vk::DescriptorType::MUTABLE_EXT)
                                .descriptor_count(counts_of_each_descriptor_type),
                        ])
                        .push_next(&mut flags)
                        .push_next(&mut mutable_descriptor_create_info),
                    None,
                )
            }
            .unwrap(),
            device,
        );

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&[*descriptor_set_layout]),
            )
        }
        .unwrap()[0];

        Self {
            _pool: descriptor_pool,
            set: descriptor_set,
            layout: descriptor_set_layout,
            sampled_image_count: Default::default(),
            storage_image_count: Default::default(),
        }
    }
}
