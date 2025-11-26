use crate as nbn;

use blue_noise_sampler::spp256::*;

pub struct BlueNoiseBuffers {
    pub sobol: nbn::Buffer,
    pub ranking_tile: nbn::Buffer,
    pub scrambling_tile: nbn::Buffer,
}

impl BlueNoiseBuffers {
    pub fn new(device: &nbn::Device, staging_buffer: &mut nbn::StagingBuffer) -> Self {
        Self {
            sobol: staging_buffer.create_buffer_from_slice(device, "sobol", SOBOL),
            ranking_tile: staging_buffer.create_buffer_from_slice(
                device,
                "ranking_tile",
                RANKING_TILE,
            ),
            scrambling_tile: staging_buffer.create_buffer_from_slice(
                device,
                "scrambling_tile",
                SCRAMBLING_TILE,
            ),
        }
    }
}
