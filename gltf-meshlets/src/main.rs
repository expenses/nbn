use std::{
    io::{BufWriter, Write},
    path::Path,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Meshlet {
    center: [f32; 3],
    radius: f32,
    cone_axis: [i8; 3],
    cone_cutoff: i8,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u8,
    triangle_count: u8,
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let path = Path::new(&path);

    let (gltf, _): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&std::fs::read(&path).unwrap()).unwrap();

    let buffer = std::fs::read(path.with_file_name(gltf.buffers[0].uri.as_ref().unwrap())).unwrap();

    let get_buffer_slice = |accessor: &goth_gltf::Accessor| {
        let bv = &gltf.buffer_views[accessor.buffer_view.unwrap()];
        assert_eq!(bv.byte_stride, None);
        &buffer[bv.byte_offset..bv.byte_offset + bv.byte_length][accessor.byte_offset..]
    };

    let output = path.with_extension("meshlets");
    let mut output = BufWriter::new(std::fs::File::create(output).unwrap());

    output.write_all(b"MESHLETS").unwrap();

    let write_val = |output: &mut BufWriter<std::fs::File>, val: u32| {
        //dbg!(val);
        output.write_all(&val.to_le_bytes()).unwrap();
    };

    write_val(&mut output, gltf.meshes.len() as _);

    let mut data: Vec<u8> = Vec::new();

    for mesh in gltf.meshes {
        write_val(&mut output, mesh.primitives.len() as _);

        for primitive in mesh.primitives {
            let indices = &gltf.accessors[primitive.indices.unwrap()];
            let slice = get_buffer_slice(indices);
            let indices = match indices.component_type {
                goth_gltf::ComponentType::UnsignedShort => cast_slice::<_, u16>(slice)
                    [..indices.count]
                    .iter()
                    .map(|&index| index as u32)
                    .collect(),
                goth_gltf::ComponentType::UnsignedInt => {
                    cast_slice::<_, u32>(slice)[..indices.count].to_vec()
                }
                other => unimplemented!("{:?}", other),
            };

            let positions = &gltf.accessors[primitive.attributes.position.unwrap()];
            assert_eq!(positions.component_type, goth_gltf::ComponentType::Float);
            let slice = get_buffer_slice(positions);
            let adapter = meshopt::utilities::VertexDataAdapter::new(slice, 4 * 3, 0).unwrap();
            let max_vertices = 64;
            let max_triangles = 124;
            let meshlets = meshopt::clusterize::build_meshlets(
                &indices,
                &adapter,
                max_vertices,
                max_triangles,
                0.0,
            );

            write_val(&mut output, meshlets.meshlets.len() as _);

            write_val(&mut output, data.len() as _);
            data.extend_from_slice(cast_slice(&meshlets.vertices));

            write_val(&mut output, data.len() as _);
            data.extend_from_slice(cast_slice(&meshlets.triangles));

            let meshlets: Vec<Meshlet> = meshlets
                .iter()
                .zip(&meshlets.meshlets)
                .map(|(meshlet_data, meshlet)| {
                    let bounds =
                        meshopt::clusterize::compute_meshlet_bounds(meshlet_data, &adapter);
                    Meshlet {
                        center: bounds.center,
                        radius: bounds.radius,
                        cone_axis: bounds.cone_axis_s8,
                        cone_cutoff: bounds.cone_cutoff_s8,
                        vertex_offset: meshlet.vertex_offset,
                        triangle_offset: meshlet.triangle_offset,
                        triangle_count: meshlet.triangle_count as _,
                        vertex_count: meshlet.vertex_count as _,
                    }
                })
                .collect();

            //dbg!(std::mem::size_of::<Meshlet>());

            write_val(&mut output, data.len() as _);
            data.extend_from_slice(cast_slice(&meshlets));
        }
    }

    write_val(&mut output, data.len() as _);
    output.write_all(&data).unwrap();
}
fn cast_slice<I: Copy, O: Copy>(slice: &[I]) -> &[O] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}
