use crate as nbn;

pub struct BlueNoiseBuffers {
    pub sobol: nbn::Buffer,
    pub ranking_tile: nbn::Buffer,
    pub scrambling_tile: nbn::Buffer,
}

impl BlueNoiseBuffers {
    pub fn new(device: &nbn::Device, staging_buffer: &mut nbn::StagingBuffer) -> Self {
        Self {
            sobol: staging_buffer.create_buffer_from_slice(
                device,
                "sobol",
                blue_noise_sampler::spp64::SOBOL,
            ),
            ranking_tile: staging_buffer.create_buffer_from_slice(
                device,
                "ranking_tile",
                blue_noise_sampler::spp64::RANKING_TILE,
            ),
            scrambling_tile: staging_buffer.create_buffer_from_slice(
                device,
                "scrambling_tile",
                blue_noise_sampler::spp64::SCRAMBLING_TILE,
            ),
        }
    }
}
