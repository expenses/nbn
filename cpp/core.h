enum class QueueType { Graphics, Compute, Transfer };

struct Buffer {
    vma::raii::Buffer buffer;
    uint64_t addr;
    void* ptr;
};
