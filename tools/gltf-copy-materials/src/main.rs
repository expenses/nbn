use nanoserde::SerJson;

fn main() {
    let mut args = std::env::args().skip(1);
    let from = args.next().unwrap();
    let to = args.next().unwrap();

    let from_data = goth_gltf::Gltf::<goth_gltf::default_extensions::Extensions>::from_json_string(
        &std::fs::read_to_string(&from).unwrap(),
    )
    .unwrap();

    let mut to_data =
        goth_gltf::Gltf::<goth_gltf::default_extensions::Extensions>::from_json_string(
            &std::fs::read_to_string(&to).unwrap(),
        )
        .unwrap();

    let mut data: std::collections::HashMap<_, _> = from_data
        .materials
        .into_iter()
        .filter(|mat| mat.pbr_metallic_roughness.base_color_texture.is_some())
        .map(|mat| (mat.name.clone().unwrap(), mat))
        .collect();

    for mat in to_data.materials.iter_mut() {
        let name = mat.name.as_ref().unwrap();
        if let Some(m) = data.remove(name) {
            *mat = m;
        } else {
            dbg!(name);
        }
    }

    to_data.images = from_data.images;
    to_data.textures = from_data.textures;

    std::fs::write(&to, to_data.serialize_json()).unwrap();
}
