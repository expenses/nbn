use crate as nbn;
use ash::vk;

pub fn create_image(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    filename: &str,
    transition_to: nbn::QueueType,
) -> nbn::Image {
    if filename.ends_with(".dds") {
        let dds = match std::fs::File::open(filename) {
            Ok(file) => ddsfile::Dds::read(file).unwrap(),
            Err(error) => {
                panic!("{} failed to load: {}", filename, error);
            }
        };

        // See for bpp values.
        // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression
        let (format, bits_per_pixel) = match dds.get_dxgi_format().unwrap() {
            ddsfile::DxgiFormat::BC1_UNorm_sRGB => (vk::Format::BC1_RGB_SRGB_BLOCK, 4),
            ddsfile::DxgiFormat::BC3_UNorm_sRGB => (vk::Format::BC3_SRGB_BLOCK, 8),
            ddsfile::DxgiFormat::BC5_UNorm => (vk::Format::BC5_UNORM_BLOCK, 8),
            ddsfile::DxgiFormat::R9G9B9E5_SharedExp => {
                (vk::Format::E5B9G9R9_UFLOAT_PACK32, 9 + 9 + 5)
            }
            other => panic!("{:?}", other),
        };
        let extent = vk::Extent3D {
            width: dds.get_width(),
            height: dds.get_height(),
            depth: dds.get_depth(),
        };
        let mut offset = 0;
        let mut offsets = Vec::new();
        for i in 0..dds.get_num_mipmap_levels() {
            offsets.push(offset);
            let level_width = (extent.width >> i).max(1).next_multiple_of(4) as u64;
            let level_height = (extent.height >> i).max(1).next_multiple_of(4) as u64;
            offset += (level_width * level_height) * bits_per_pixel / 8;
        }

        staging_buffer.create_sampled_image(
            device,
            nbn::SampledImageDescriptor {
                name: filename,
                extent: extent.into(),
                format,
            },
            &dds.data,
            transition_to,
            &offsets,
        )
    } else {
        assert!(filename.ends_with(".ktx2"));

        let ktx2 = ktx2::Reader::new(std::fs::read(filename).unwrap()).unwrap();
        let header = ktx2.header();

        let mut data = Vec::with_capacity(
            ktx2.levels()
                .map(|level| level.uncompressed_byte_length)
                .sum::<u64>() as _,
        );
        let mut offsets = Vec::with_capacity(ktx2.levels().len());

        for level in ktx2.levels() {
            offsets.push(data.len() as _);
            data.extend_from_slice(
                &zstd::bulk::decompress(level.data, level.uncompressed_byte_length as _).unwrap(),
            );
        }

        staging_buffer.create_sampled_image(
            device,
            nbn::SampledImageDescriptor {
                name: filename,
                extent: vk::Extent3D {
                    width: header.pixel_width,
                    height: header.pixel_height,
                    depth: header.pixel_depth.max(1),
                }
                .into(),
                format: vk::Format::from_raw(header.format.unwrap().value() as _),
            },
            &data,
            transition_to,
            &offsets,
        )
    }
}
