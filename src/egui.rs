use crate as nbn;
use nbn::vk;
use std::collections::HashMap;

pub struct Renderer {
    pipeline: nbn::Pipeline,
    textures: HashMap<egui::TextureId, EguiTexture>,
    sampler: nbn::Sampler,
    buffer: nbn::Buffer,
    temporary_buffers: [Vec<nbn::Buffer>; nbn::FRAMES_IN_FLIGHT],
    temporary_textures: [Vec<nbn::Image>; nbn::FRAMES_IN_FLIGHT],
}

impl Renderer {
    pub fn new(
        device: &nbn::Device,
        color_attachment_format: vk::Format,
        initial_buffer_size: u64,
    ) -> Self {
        let egui_shader = device.load_shader("shaders/compiled/egui.spv");

        let fragment_entry_point = match color_attachment_format {
            vk::Format::B8G8R8A8_UNORM => c"fragment_non_srgb",
            vk::Format::B8G8R8A8_SRGB => c"fragment_srgb",
            other => todo!("{:?}", other),
        };

        Self {
            temporary_buffers: Default::default(),
            temporary_textures: Default::default(),
            textures: Default::default(),
            pipeline: device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
                vertex: nbn::ShaderDesc {
                    entry_point: c"vertex",
                    module: &egui_shader,
                },
                fragment: nbn::ShaderDesc {
                    entry_point: fragment_entry_point,
                    module: &egui_shader,
                },
                color_attachment_formats: &[color_attachment_format],
                blend_attachments: &[vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(true)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE)],

                mesh_shader: false,
                conservative_rasterization: false,
                depth: nbn::GraphicsPipelineDepthDesc::default(),
                cull_mode: vk::CullModeFlags::NONE,
            }),
            sampler: nbn::Sampler::from_raw(
                unsafe {
                    device.create_sampler(
                        &vk::SamplerCreateInfo::default()
                            .anisotropy_enable(true)
                            .max_anisotropy(device.properties.limits.max_sampler_anisotropy)
                            .mag_filter(vk::Filter::LINEAR)
                            .min_filter(vk::Filter::LINEAR)
                            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                            .min_lod(0.0)
                            .max_lod(f32::MAX)
                            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                        None,
                    )
                }
                .unwrap(),
                device,
            ),
            buffer: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "egui buffer",
                    size: initial_buffer_size,
                    ty: nbn::MemoryLocation::CpuToGpu,
                })
                .unwrap(),
        }
    }

    pub fn remove_texture(
        &mut self,
        device: &nbn::Device,
        id: egui::TextureId,
        current_frame: usize,
    ) {
        if let Some(texture) = self.textures.remove(&id) {
            device.deregister_image(texture.id, false);
            self.temporary_textures[current_frame].push(texture.image);
        }
    }

    pub fn update_textures(
        &mut self,
        device: &nbn::Device,
        command_buffer: &nbn::CommandBuffer,
        current_frame: usize,
        textures: &egui::TexturesDelta,
    ) {
        self.temporary_buffers[current_frame].clear();
        self.temporary_textures[current_frame].clear();

        for &id in &textures.free {
            self.remove_texture(device, id, current_frame);
        }

        for (id, data) in &textures.set {
            self.remove_texture(device, *id, current_frame);

            let (staging_buffer, image) = match &data.image {
                egui::ImageData::Color(image) => device.create_image_with_data_in_command_buffer(
                    nbn::SampledImageDescriptor {
                        name: &format!("egui image ({:?}) (rgba8)", id),
                        format: vk::Format::R8G8B8A8_SRGB,
                        extent: vk::Extent3D {
                            width: image.width() as _,
                            height: image.height() as _,
                            depth: 1,
                        },
                    },
                    nbn::cast_slice(&image.pixels),
                    nbn::QueueType::Graphics,
                    &[0],
                    command_buffer,
                ),
                egui::ImageData::Font(image) => device.create_image_with_data_in_command_buffer(
                    nbn::SampledImageDescriptor {
                        name: &format!("egui image ({:?}) (r32f)", id),
                        format: vk::Format::R32_SFLOAT,
                        extent: vk::Extent3D {
                            width: image.width() as _,
                            height: image.height() as _,
                            depth: 1,
                        },
                    },
                    nbn::cast_slice(&image.pixels),
                    nbn::QueueType::Graphics,
                    &[0],
                    command_buffer,
                ),
            };

            let registered_id =
                device.register_image_with_sampler(*image.view, &self.sampler, false);

            self.textures.insert(
                *id,
                EguiTexture {
                    id: registered_id,
                    image,
                },
            );

            self.temporary_buffers[current_frame].push(staging_buffer);
        }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn paint(
        &mut self,
        device: &nbn::Device,
        command_buffer: &nbn::CommandBuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
        scale_factor: f32,
        extent: [u32; 2],
    ) {
        let mut offset = 0;
        let buffer_ptr = *self.buffer;
        let slice: &mut [u8] = self.buffer.try_as_slice_mut().unwrap();

        device.bind_internal_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS);
        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *self.pipeline,
        );

        for primitive in clipped_primitives {
            device.cmd_set_scissor(
                **command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: (primitive.clip_rect.min.x * scale_factor) as _,
                        y: (primitive.clip_rect.min.y * scale_factor) as _,
                    },
                    extent: vk::Extent2D {
                        width: (primitive.clip_rect.width() * scale_factor) as _,
                        height: (primitive.clip_rect.height() * scale_factor) as _,
                    },
                }],
            );
            if let egui::epaint::Primitive::Mesh(mesh) = &primitive.primitive {
                let texture = self.textures.get(&mesh.texture_id).unwrap();

                let vertices_offset = offset;
                let vertices_slice: &[u8] = nbn::cast_slice(&mesh.vertices);
                slice[offset..offset + vertices_slice.len()].copy_from_slice(vertices_slice);
                offset += vertices_slice.len();

                let indices_offset = offset;
                let indices_slice: &[u8] = nbn::cast_slice(&mesh.indices);
                slice[offset..offset + indices_slice.len()].copy_from_slice(indices_slice);
                offset += indices_slice.len();

                device.push_constants(
                    command_buffer,
                    (
                        buffer_ptr + vertices_offset as u64,
                        buffer_ptr + indices_offset as u64,
                        extent,
                        scale_factor,
                        texture.id,
                    ),
                );
                device.cmd_draw(**command_buffer, mesh.indices.len() as _, 1, 0, 0);
            }
        }
    }
}

struct EguiTexture {
    id: u32,
    image: nbn::Image,
}
