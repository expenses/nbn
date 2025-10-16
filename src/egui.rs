use crate as nbn;
use nbn::vk;
use std::collections::HashMap;

pub struct Renderer {
    pipeline: nbn::Pipeline,
    textures: HashMap<egui::TextureId, nbn::IndexedImage>,
    buffer: nbn::Buffer,
    temporary_buffers: [Vec<nbn::Buffer>; nbn::FRAMES_IN_FLIGHT],
    temporary_textures: [Vec<nbn::Image>; nbn::FRAMES_IN_FLIGHT],
    buffer_section_size: u64,
}

impl Renderer {
    pub fn new(
        device: &nbn::Device,
        color_attachment_format: vk::Format,
        initial_buffer_size: u64,
    ) -> Self {
        let egui_shader = device.load_shader("shaders/compiled/egui.spv");

        Self {
            buffer_section_size: initial_buffer_size,
            temporary_buffers: Default::default(),
            temporary_textures: Default::default(),
            textures: Default::default(),
            pipeline: device.create_graphics_pipeline(nbn::GraphicsPipelineDesc {
                name: "egui pipeline",
                shaders: nbn::GraphicsPipelineShaders::Legacy {
                    vertex: nbn::ShaderDesc {
                        entry_point: c"vertex",
                        module: &egui_shader,
                    },
                    fragment: nbn::ShaderDesc {
                        entry_point: c"fragment",
                        module: &egui_shader,
                    },
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

                conservative_rasterization: false,
                depth: nbn::GraphicsPipelineDepthDesc::default(),
                cull_mode: vk::CullModeFlags::NONE,
            }),

            buffer: device
                .create_buffer(nbn::BufferDescriptor {
                    name: "egui buffer",
                    size: initial_buffer_size * nbn::FRAMES_IN_FLIGHT as u64,
                    ty: nbn::MemoryLocation::CpuToGpu,
                })
                .unwrap(),
        }
    }

    pub fn remove_texture(&mut self, id: egui::TextureId, current_frame: usize) {
        if let Some(texture) = self.textures.remove(&id) {
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
            self.remove_texture(id, current_frame);
        }

        for (id, data) in &textures.set {
            match data.pos {
                Some([x, y]) => {
                    let texture = self.textures.get(id).unwrap();

                    let (width, height, data) = match &data.image {
                        egui::ImageData::Color(image) => (
                            image.width(),
                            image.height(),
                            nbn::cast_slice::<_, u8>(&image.pixels),
                        ),
                        egui::ImageData::Font(image) => (
                            image.width(),
                            image.height(),
                            nbn::cast_slice::<_, u8>(&image.pixels),
                        ),
                    };

                    let staging_buffer =
                        device.create_buffer_with_data(nbn::BufferInitDescriptor {
                            name: "egui update buffer",
                            data,
                        });

                    device.insert_pipeline_barrier(
                        command_buffer,
                        None,
                        &[],
                        &[nbn::ImageBarrier {
                            previous_accesses: &[
                                nbn::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                            ],
                            next_accesses: &[nbn::AccessType::TransferWrite],
                            previous_layout: nbn::ImageLayout::Optimal,
                            next_layout: nbn::ImageLayout::Optimal,
                            discard_contents: false,
                            src_queue_family_index: device.graphics_queue.index,
                            dst_queue_family_index: device.graphics_queue.index,
                            image: *texture.image.image,
                            range: texture.image.subresource_range,
                        }],
                    );
                    unsafe {
                        device.cmd_copy_buffer_to_image(
                            **command_buffer,
                            *staging_buffer.buffer,
                            *texture.image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::BufferImageCopy::default()
                                .buffer_offset(0)
                                .image_offset(vk::Offset3D {
                                    x: x as _,
                                    y: y as _,
                                    z: 0,
                                })
                                .image_extent(vk::Extent3D {
                                    width: width as _,
                                    height: height as _,
                                    depth: 1,
                                })
                                .image_subresource(
                                    vk::ImageSubresourceLayers::default()
                                        .layer_count(1)
                                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                                )],
                        )
                    }
                    device.insert_pipeline_barrier(
                        command_buffer,
                        None,
                        &[],
                        &[nbn::ImageBarrier {
                            previous_accesses: &[nbn::AccessType::TransferWrite],
                            next_accesses: &[
                                nbn::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                            ],
                            previous_layout: nbn::ImageLayout::Optimal,
                            next_layout: nbn::ImageLayout::Optimal,
                            discard_contents: false,
                            src_queue_family_index: device.graphics_queue.index,
                            dst_queue_family_index: device.graphics_queue.index,
                            image: *texture.image.image,
                            range: texture.image.subresource_range,
                        }],
                    );

                    self.temporary_buffers[current_frame].push(staging_buffer);
                }
                None => {
                    self.remove_texture(*id, current_frame);

                    let (staging_buffer, name, format, extent) = match &data.image {
                        egui::ImageData::Color(image) => {
                            let name = format!("egui image ({:?}) (rgba8)", id);

                            (
                                device.create_buffer_with_data(nbn::BufferInitDescriptor {
                                    name: &name,
                                    data: &image.pixels,
                                }),
                                name,
                                vk::Format::R8G8B8A8_SRGB,
                                vk::Extent2D {
                                    width: image.width() as _,
                                    height: image.height() as _,
                                },
                            )
                        }
                        egui::ImageData::Font(image) => {
                            let name = format!("egui image ({:?}) (r32f)", id);

                            (
                                device.create_buffer_with_data(nbn::BufferInitDescriptor {
                                    name: &name,
                                    data: &image.pixels,
                                }),
                                name,
                                vk::Format::R32_SFLOAT,
                                vk::Extent2D {
                                    width: image.width() as _,
                                    height: image.height() as _,
                                },
                            )
                        }
                    };

                    let image = device.create_image_with_data_in_command_buffer(
                        nbn::SampledImageDescriptor {
                            name: &name,
                            format,
                            extent: extent.into(),
                        },
                        &staging_buffer,
                        nbn::QueueType::Graphics,
                        &[0],
                        command_buffer,
                    );

                    let registered_id = device.register_image_with_sampler(
                        *image.view,
                        &device.samplers.clamp,
                        false,
                    );

                    self.textures.insert(
                        *id,
                        nbn::IndexedImage {
                            index: registered_id,
                            image,
                        },
                    );

                    self.temporary_buffers[current_frame].push(staging_buffer);
                }
            }
        }
    }

    #[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
    pub unsafe fn paint(
        &mut self,
        device: &nbn::Device,
        command_buffer: &nbn::CommandBuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
        scale_factor: f32,
        extent: [u32; 2],
        frame_index: usize,
        transfer_function: nbn::TransferFunction,
    ) {
        let buffer_ptr = *self.buffer + self.buffer_section_size * frame_index as u64;
        let slice: &mut [u8] = self.buffer.try_as_slice_mut().unwrap();
        let slice = &mut slice[self.buffer_section_size as usize * frame_index
            ..self.buffer_section_size as usize * (frame_index + 1)];
        let mut offset = 0;

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

                let (transfer_function_index, calibrated_nits) =
                    transfer_function.as_push_constants();
                device.push_constants::<(u64, u64, [u32; 2], f32, u32, u32, f32)>(
                    command_buffer,
                    (
                        buffer_ptr + vertices_offset as u64,
                        buffer_ptr + indices_offset as u64,
                        extent,
                        scale_factor,
                        **texture,
                        transfer_function_index,
                        calibrated_nits,
                    ),
                );
                device.cmd_draw(**command_buffer, mesh.indices.len() as _, 1, 0, 0);
            }
        }
    }
}
