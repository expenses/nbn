use crate::*;

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

pub struct GltfModel {
    pub model: Model,
    pub acceleration_structure: nbn::AccelerationStructure,
    // for debugging
    pub meshlets: Vec<Meshlet>,
}

pub struct GltfData {
    _images: Vec<nbn::IndexedImage>,
    pub(crate) _buffer: nbn::Buffer,
    _meshlets_buffer: nbn::Buffer,
    pub meshes: Vec<GltfModel>,
}

struct Meshlets {
    buffer: nbn::Buffer,
    metadata: Vec<Vec<[u32; 4]>>,
    // for debugging
    data: Vec<u8>,
}

fn read_meshlets_file(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> std::io::Result<Meshlets> {
    let mut reader = std::fs::File::open(path)?;
    let mut header = [0; 8];
    reader.read_exact(&mut header)?;
    assert_eq!(b"MESHLETS", &header);
    let mut val = [0; 4];
    reader.read_exact(&mut val)?;

    let num_meshes = u32::from_le_bytes(val);
    let mut meshes = Vec::with_capacity(num_meshes as _);

    for _ in 0..num_meshes {
        reader.read_exact(&mut val)?;
        let num_primitives = u32::from_le_bytes(val);
        let mut primitives = vec![[0_u32; 4]; num_primitives as _];
        reader.read_exact(nbn::cast_slice_mut(&mut primitives))?;
        meshes.push(primitives);
    }

    reader.read_exact(&mut val)?;

    let len_data = u32::from_le_bytes(val);

    // for debugging
    let mut data = vec![0; len_data as usize];
    reader.read_exact(&mut data).unwrap();

    let buffer = staging_buffer.create_buffer(
        device,
        &format!("{} staging buffer", path.display()),
        len_data as _,
        std::io::Cursor::new(&data),
    );

    Ok(Meshlets {
        buffer,
        metadata: meshes,
        data,
    })
}

pub fn load_gltf(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> GltfData {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let mut image_formats = vec![vk::Format::R8G8B8A8_UNORM; gltf.images.len()];

    for material in &gltf.materials {
        if let Some(tex) = &material.emissive_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
        if let Some(tex) = &material.pbr_metallic_roughness.base_color_texture {
            image_formats[gltf.textures[tex.index].source.unwrap()] = vk::Format::R8G8B8A8_SRGB;
        }
    }

    let meshlets =
        read_meshlets_file(&device, staging_buffer, &path.with_extension("meshlets")).unwrap();
    let meshlets_buffer = meshlets.buffer;

    let buffer_file =
        std::fs::File::open(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();
    let buffer = staging_buffer.create_buffer(
        device,
        &format!("{} buffer", path.display()),
        buffer_file.metadata().unwrap().len() as _,
        buffer_file,
    );

    let images = gltf
        .images
        .iter()
        .zip(&image_formats)
        .map(|(image, _format)| {
            let image = create_image(
                device,
                staging_buffer,
                path.with_file_name(image.uri.as_ref().unwrap())
                    .to_str()
                    .unwrap(),
                nbn::QueueType::Graphics,
            );

            nbn::IndexedImage {
                index: device.register_image(*image.view, false),
                image,
            }
        })
        .collect::<Vec<_>>();

    dbg!(images.len());

    let materials: Vec<Material> = gltf
        .materials
        .iter()
        .map(|material| Material {
            base_colour_image: material
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            metallic_roughness_image: material
                .pbr_metallic_roughness
                .metallic_roughness_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            normal_image: material
                .normal_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            emissive_image: material
                .emissive_texture
                .as_ref()
                .map(|tex| *images[gltf.textures[tex.index].source.unwrap()])
                .unwrap_or(u32::MAX),
            flags: !matches!(material.alpha_mode, goth_gltf::AlphaMode::Opaque) as u32,
        })
        .collect();

    let get_buffer_offset = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        (*buffer) + bv.byte_offset as u64 + accessor.byte_offset as u64
    };

    let mut meshes = Vec::new();

    for (mesh_index, (mesh, meshlets_mesh)) in
        gltf.meshes.iter().zip(&meshlets.metadata).enumerate()
    {
        for (
            primitive_index,
            (
                primitive,
                &[
                    num_meshlets,
                    vertices_offset,
                    triangles_offset,
                    meshlets_offset,
                ],
            ),
        ) in mesh.primitives.iter().zip(meshlets_mesh).enumerate()
        {
            let material = materials[primitive.material.unwrap()];

            let get = |accessor_index: Option<usize>| {
                let accessor = &gltf.accessors[accessor_index.unwrap()];
                assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                get_buffer_offset(accessor)
            };

            let indices = &gltf.accessors[primitive.indices.unwrap()];
            let is_32_bit = match indices.component_type {
                goth_gltf::ComponentType::UnsignedShort => false,
                goth_gltf::ComponentType::UnsignedInt => true,
                other => unimplemented!("{:?}", other),
            };

            let positions_accessor = &gltf.accessors[primitive.attributes.position.unwrap()];

            let positions = get(primitive.attributes.position);

            let acceleration_structure = device.create_acceleration_structure(
                &format!(
                    "{} mesh {} primitive {} acceleration structure",
                    path.display(),
                    mesh_index,
                    primitive_index
                ),
                nbn::AccelerationStructureData::Triangles {
                    index_type: if is_32_bit {
                        vk::IndexType::UINT32
                    } else {
                        vk::IndexType::UINT16
                    },
                    opaque: material.flags == 0,
                    vertices_buffer_address: positions,
                    indices_buffer_address: get_buffer_offset(indices),
                    num_vertices: positions_accessor.count as _,
                    num_indices: indices.count as _,
                },
                staging_buffer,
            );

            meshes.push(GltfModel {
                acceleration_structure,
                model: Model {
                    material,
                    positions,
                    uvs: get(primitive.attributes.texcoord_0),
                    normals: get(primitive.attributes.normal),
                    indices: get_buffer_offset(indices),
                    meshlets: *meshlets_buffer + meshlets_offset as u64,
                    triangles: *meshlets_buffer + triangles_offset as u64,
                    vertices: *meshlets_buffer + vertices_offset as u64,
                    flags: is_32_bit as u32,
                    num_meshlets,
                    num_indices: indices.count as u32,
                },
                meshlets: nbn::cast_slice::<_, Meshlet>(&meshlets.data[meshlets_offset as usize..])
                    [..num_meshlets as usize]
                    .to_vec(),
            });
        }
    }

    dbg!(&materials.len());

    GltfData {
        _images: images,
        _buffer: buffer,
        _meshlets_buffer: meshlets_buffer,
        meshes,
    }
}
