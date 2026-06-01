
struct CountTracker {
    uint32_t next = 0;
    std::vector<uint32_t> unused;

    uint32_t add() {
        if (!unused.empty()) {
            auto idx = unused.back();
            unused.pop_back();
            return idx;
        }
        return next++;
    }
    void remove(uint32_t index) {
        unused.push_back(index);
    }
};

struct Descriptors {
    vk::raii::DescriptorPool pool = {nullptr};
    vk::raii::DescriptorSetLayout layout = {nullptr};
    vk::raii::DescriptorSet set = {nullptr};
    std::shared_ptr<CountTracker> sampled_image_count;
    std::shared_ptr<CountTracker> storage_image_count;

    Descriptors() = default;
    explicit Descriptors(const vk::raii::Device& device);
};
