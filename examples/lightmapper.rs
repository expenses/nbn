use ash::vk;
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

slang_struct::slang_include!("shaders/lightmapper.slang");

fn main() {
    env_logger::init();

    let device = Arc::new(nbn::Device::new(None));

    let mut staging_buffer =
        nbn::StagingBuffer::new(&device, 16 * 1024 * 1024, nbn::QueueType::Compute);

    let (gltf_data, model, lights) = load_gltf(
        &device,
        &mut staging_buffer,
        &std::path::Path::new("./models/export/assassins.gltf"),
    );

    let num_lights = dbg!(lights.len());

    let lights = staging_buffer.create_buffer_from_slice(&device, "lights", &lights);

    let instance_buffer = staging_buffer.create_buffer_from_slice(
        &device,
        "Instances",
        &[vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: glam::Mat4::IDENTITY.transpose().to_cols_array()[..12]
                    .try_into()
                    .unwrap(),
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                ash::vk::GeometryInstanceFlagsKHR::default().as_raw() as _,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: *gltf_data.acceleration_structure,
            },
        }],
    );

    let model_buffer = staging_buffer.create_buffer_from_slice(&device, "models", &[model]);

    let tlas = device.create_acceleration_structure(
        "tlas",
        nbn::AccelerationStructureData::Instances {
            buffer_address: *instance_buffer,
            count: 1,
        },
        &mut staging_buffer,
    );

    let blue_noise_buffers = nbn::blue_noise::BlueNoiseBuffers::new(&device, &mut staging_buffer);

    staging_buffer.finish(&device);
}

fn load_gltf(
    device: &nbn::Device,
    staging_buffer: &mut nbn::StagingBuffer,
    path: &std::path::Path,
) -> (CombinedModel, Model, Vec<Light>) {
    let bytes = std::fs::read(path).unwrap();
    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();
    assert!(buffer.is_none());
    dbg!(gltf.meshes.len(), gltf.meshes[0].primitives.len());

    let lights: Vec<_> = gltf
        .nodes
        .iter()
        .filter_map(|node| {
            node.extensions
                .khr_lights_punctual
                .as_ref()
                .map(|ext| (node, ext.light))
        })
        .map(|(node, index)| {
            let (pos, rotation) = if let goth_gltf::NodeTransform::Set {
                translation,
                rotation,
                ..
            } = node.transform()
            {
                (translation, rotation)
            } else {
                panic!()
            };

            let light = &gltf.extensions.khr_lights_punctual.as_ref().unwrap().lights[index];

            let (spotlight_angle_scale, spotlight_angle_offset) = light
                .spot
                .map(|spot| {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
                    let spotlight_angle_scale = 1.0
                        / 0.000001_f32
                            .max(spot.inner_cone_angle.cos() - spot.outer_cone_angle.cos());
                    let spotlight_angle_offset =
                        -spot.outer_cone_angle.cos() * spotlight_angle_scale;
                    (spotlight_angle_scale, spotlight_angle_offset)
                })
                .unwrap_or((0.0, 1.0));

            Light {
                position: pos,
                emission: (glam::Vec3::from(light.color) * light.intensity).into(),
                spotlight_angle_scale,
                spotlight_angle_offset,
                spotlight_direction: (glam::Quat::from_array(rotation) * glam::Vec3::Z).into(),
            }
        })
        .collect();

    let images = gltf
        .images
        .iter()
        //.zip(&images)
        .map(|image| {
            let path = path.with_file_name(image.uri.as_ref().unwrap());
            let data = image::open(&path).unwrap().to_rgba8();
            let image = staging_buffer.create_sampled_image(
                &device,
                nbn::SampledImageDescriptor {
                    name: image.uri.as_ref().unwrap(),
                    extent: vk::Extent3D {
                        width: data.width(),
                        height: data.height(),
                        depth: 1,
                    }
                    .into(),
                    format: vk::Format::R8G8B8A8_SRGB,
                },
                &data,
                nbn::QueueType::Compute,
                &[0],
            );
            nbn::IndexedImage {
                index: device.register_image(*image.view, false),
                image,
            }
        })
        .collect::<Vec<_>>();

    let material_to_image: Vec<u32> = gltf
        .materials
        .iter()
        .map(|mat| {
            let texture_index = mat
                .pbr_metallic_roughness
                .base_color_texture
                .as_ref()
                .unwrap()
                .index;
            *images[gltf.textures[texture_index].source.unwrap()].index
        })
        .collect();

    let buffer = std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    fn get_slice<'a, T: Copy>(
        buffer: &'a [u8],
        gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        accessor: &goth_gltf::Accessor,
    ) -> &'a [T] {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        &nbn::cast_slice(&buffer[bv.byte_offset + accessor.byte_offset..])
    }

    let mut indices = Vec::new();
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut normals = Vec::new();
    let mut image_indices = Vec::new();
    let mut uv2s = Vec::new();

    for mesh in gltf.meshes.iter() {
        for primitive in mesh.primitives.iter() {
            let indices_accessor = &gltf.accessors[primitive.indices.unwrap()];
            assert_eq!(
                indices_accessor.component_type,
                goth_gltf::ComponentType::UnsignedShort
            );
            let prim_indices =
                &get_slice::<u16>(&buffer, &gltf, &indices_accessor)[..indices_accessor.count];
            indices.extend(
                prim_indices
                    .iter()
                    .map(|&index| positions.len() as u32 / 3 + index as u32),
            );

            let get = |accessor_index: Option<usize>, size: usize, error: &str| {
                let accessor = &gltf.accessors[accessor_index.expect(error)];
                assert_eq!(accessor.component_type, goth_gltf::ComponentType::Float);
                &get_slice::<f32>(&buffer, &gltf, accessor)[..accessor.count * size]
            };

            let positions_slice = get(primitive.attributes.position, 3, "positions");
            positions.extend_from_slice(positions_slice);
            uvs.extend_from_slice(get(primitive.attributes.texcoord_0, 2, "uvs"));
            uv2s.extend_from_slice(get(primitive.attributes.texcoord_1, 2, "uv2s"));
            normals.extend_from_slice(get(primitive.attributes.normal, 3, "normals"));

            let material_index = primitive.material.unwrap();

            image_indices
                .extend((0..prim_indices.len() / 3).map(|_| material_to_image[material_index]));
        }
    }

    let num_vertices = positions.len() / 3;
    let num_indices = indices.len();
    let indices = staging_buffer.create_buffer_from_slice(device, "indices", &indices);
    let positions = staging_buffer.create_buffer_from_slice(device, "positions", &positions);
    dbg!(image_indices.len());

    let acceleration_structure = device.create_acceleration_structure(
        &format!("{} acceleration structure", path.display(),),
        nbn::AccelerationStructureData::Triangles {
            index_type: vk::IndexType::UINT32,
            opaque: true,
            vertices_buffer_address: *positions,
            indices_buffer_address: *indices,
            num_vertices: num_vertices as _,
            num_indices: num_indices as _,
        },
        staging_buffer,
    );

    let uvs = staging_buffer.create_buffer_from_slice(device, "uvs", &uvs);
    let uv2s = staging_buffer.create_buffer_from_slice(device, "uv2s", &uv2s);
    let normals = staging_buffer.create_buffer_from_slice(device, "normals", &normals);
    let image_indices =
        staging_buffer.create_buffer_from_slice(device, "image_indices", &image_indices);

    let model = Model {
        positions: *positions,
        uvs: *uvs,
        uv2s: *uv2s,
        normals: *normals,
        indices: *indices,
        image_indices: *image_indices,
        flags: 1,
        num_indices: num_indices as _,
    };

    (
        CombinedModel {
            acceleration_structure,
            _positions: positions,
            _indices: indices,
            _uvs: uvs,
            _uv2s: uv2s,
            _normals: normals,
            _image_indices: image_indices,
            _images: images,
        },
        model,
        lights,
    )
}

struct CombinedModel {
    acceleration_structure: nbn::AccelerationStructure,
    _positions: nbn::Buffer,
    _indices: nbn::Buffer,
    _uvs: nbn::Buffer,
    _uv2s: nbn::Buffer,
    _normals: nbn::Buffer,
    _image_indices: nbn::Buffer,
    _images: Vec<nbn::IndexedImage>,
}
