use crate::*;

pub fn render(device: &nbn::Device, state: &mut WindowState) {
    state.blit_pipelines.refresh(device);
    state.mesh_pipelines.refresh(device);
    state.compute_pipelines.refresh(device);

    let forward = state.keyboard.forwards as i32 - state.keyboard.backwards as i32;
    let right = state.keyboard.right as i32 - state.keyboard.left as i32;

    let prev_transform = state.camera_rig.final_transform;

    state
        .camera_rig
        .driver_mut::<dolly::drivers::Position>()
        .translate(
            ((glam::Vec3::from_array(prev_transform.forward()) * forward as f32
                + glam::Vec3::from_array(prev_transform.right()) * right as f32)
                * 100.0)
                .to_array(),
        );

    let transform = state.camera_rig.update(1.0 / 60.0);
    let camera_position = glam::Vec3::from_array(transform.position.into());

    let extent = state.swapchain.create_info.image_extent;

    let scale = 500.0;
    let projection = perspective_reversed_infinite_z_vk(
        45.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.0001,
    );
    let start_camera_pos = glam::Vec3::new(scale, scale, 0.0);
    let camera_pos = glam::Vec3::from_array(transform.position.into());
    let view = glam::Mat4::look_to_rh(
        camera_pos,
        glam::Vec3::from_array(transform.forward()),
        glam::Vec3::Y,
    );
    let mat = projection * view;

    let uniforms = state
        .combined_uniform_buffer
        .try_as_slice_mut::<UniformBuffer>()
        .unwrap();

    let frustum_x = (projection.row(3).truncate() + projection.row(0).truncate()).normalize();
    let frustum_y = (projection.row(3).truncate() + projection.row(1).truncate()).normalize();

    let raw_input = state.egui_winit.take_egui_input(&state.window);

    let egui_ctx = state.egui_winit.egui_ctx();

    egui_ctx.begin_pass(raw_input);
    if !state.cursor_grabbed {
        let allocator = device.allocator.inner.read();
        egui::Window::new("Memory Allocations").show(&egui_ctx, |ui| {
            state.alloc_vis.render_breakdown_ui(ui, &allocator);
        });
        egui::Window::new("Memory Blocks").show(&egui_ctx, |ui| {
            state.alloc_vis.render_memory_block_ui(ui, &allocator);
        });
        state
            .alloc_vis
            .render_memory_block_visualization_windows(&egui_ctx, &allocator);
    }
    let output = egui_ctx.end_pass();
    state
        .egui_winit
        .handle_platform_output(&state.window, output.platform_output);

    let clipped_meshes = state
        .egui_winit
        .egui_ctx()
        .tessellate(output.shapes, output.pixels_per_point);

    let current_frame = state.sync_resources.current_frame;

    unsafe {
        let command_buffer = &state.per_frame_command_buffers[current_frame];
        let mut frame = state.sync_resources.wait_for_frame(device);

        let (next_image, _suboptimal) = device
            .swapchain_loader
            .acquire_next_image(
                *state.swapchain,
                !0,
                *frame.image_available_semaphore,
                vk::Fence::null(),
            )
            .unwrap();

        uniforms[current_frame] = UniformBuffer {
            mat: mat.to_cols_array(),
            view: view.to_cols_array(),
            perspective: projection.to_cols_array(),
            near_plane: 0.0001,
            instances: *state.instances,
            meshlet_instances: *state.meshlet_instances,
            extent: [extent.width, extent.height],
            num_instances: state.num_instances,
            visbuffer: state.framebuffers.vis.index,
            hdrbuffer: state.framebuffers.hdr.index,
            swapchain_image: state.swapchain_image_heap_indices[next_image as usize],
            dispatches: *state.dispatches,
            models: *state.models,
            camera_position: camera_pos.into(),
            frustum: [frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z],
            opaque_prefix_sum_values: *state.prefix_sum_values,
            alpha_clip_prefix_sum_values: *state.prefix_sum_values
                + (8 * TOTAL_NUM_INSTANCES_OF_TYPE),
            tonemap_lut_image: *state.tonemap_lut,
            depthbuffer: state.framebuffers.depth.index,
        };

        let uniforms_ptr = *state.combined_uniform_buffer
            + (std::mem::size_of::<UniformBuffer>() * current_frame) as u64;

        let image = &state.swapchain.images[next_image as usize];

        device.reset_command_buffer(command_buffer);
        device
            .begin_command_buffer(**command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();

        state.egui_render.update_textures(
            device,
            command_buffer,
            current_frame,
            &output.textures_delta,
        );

        device.bind_internal_descriptor_sets_to_all(command_buffer);

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.compute_pipelines.reset_buffers,
        );
        device.push_constants(command_buffer, *state.dispatches);
        device.cmd_dispatch(**command_buffer, 1, 1, 1);

        device.insert_global_barrier(
            command_buffer,
            &[nbn::AccessType::ComputeShaderReadWrite],
            &[nbn::AccessType::ComputeShaderReadWrite],
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.compute_pipelines.generate_meshlet_prefix_sums,
        );
        device.push_constants(command_buffer, (uniforms_ptr,));
        device.cmd_dispatch(**command_buffer, state.num_instances.div_ceil(64), 1, 1);

        nbn::pipeline_barrier(
            device,
            **command_buffer,
            Some(nbn::GlobalBarrier {
                previous_accesses: &[nbn::AccessType::ComputeShaderReadWrite],
                next_accesses: &[
                    nbn::AccessType::ComputeShaderReadWrite,
                    nbn::AccessType::TaskShaderReadOther,
                    nbn::AccessType::MeshShaderReadOther,
                ],
            }),
            &[],
            &[
                nbn::ImageBarrier {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::DepthStencilAttachmentWrite],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: **state.framebuffers.depth.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::DEPTH),
                },
                nbn::ImageBarrier {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::ColorAttachmentWrite],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: **state.framebuffers.vis.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                },
            ],
        );

        device.begin_rendering(
            command_buffer,
            extent.width,
            extent.height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*state.framebuffers.vis.image.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        uint32: [u32::MAX; 4],
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)],
            Some(
                &vk::RenderingAttachmentInfo::default()
                    .image_view(*state.framebuffers.depth.image.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    }),
            ),
        );

        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *state.mesh_pipelines.opaque,
        );
        device.push_constants(command_buffer, (uniforms_ptr, 0_u32));
        device.mesh_shader_loader.cmd_draw_mesh_tasks_indirect(
            **command_buffer,
            *state.dispatches.buffer,
            0,
            1,
            4 * 4,
        );
        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *state.mesh_pipelines.alpha_clipped,
        );
        device.push_constants(command_buffer, (uniforms_ptr, 1_u32));
        device.mesh_shader_loader.cmd_draw_mesh_tasks_indirect(
            **command_buffer,
            *state.dispatches.buffer,
            4 * 4,
            1,
            4 * 4,
        );

        state.time += 0.005;
        device.cmd_end_rendering(**command_buffer);
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[
                nbn::ImageBarrier {
                    previous_accesses: &[nbn::AccessType::ColorAttachmentWrite],
                    next_accesses: &[nbn::AccessType::ComputeShaderReadOther],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: **state.framebuffers.vis.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                },
                nbn::ImageBarrier {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: **state.framebuffers.hdr.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                },
            ],
        );
        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.blit_pipelines.resolve_visbuffer,
        );
        device.push_constants(command_buffer, uniforms_ptr);
        device.cmd_dispatch(
            **command_buffer,
            extent.width.div_ceil(8),
            extent.height.div_ceil(8),
            1,
        );
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[
                nbn::ImageBarrier {
                    previous_accesses: &[nbn::AccessType::Present],
                    next_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: image.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                },
                nbn::ImageBarrier {
                    previous_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    next_accesses: &[nbn::AccessType::ComputeShaderReadOther],
                    previous_layout: nbn::ImageLayout::Optimal,
                    next_layout: nbn::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: **state.framebuffers.hdr.image,
                    range: vk::ImageSubresourceRange::default()
                        .layer_count(1)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR),
                },
            ],
        );
        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *state.blit_pipelines.tonemap,
        );
        device.push_constants(command_buffer, uniforms_ptr);
        device.cmd_dispatch(
            **command_buffer,
            extent.width.div_ceil(8),
            extent.height.div_ceil(8),
            1,
        );
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[nbn::ImageBarrier {
                previous_accesses: &[nbn::AccessType::ComputeShaderWrite],
                next_accesses: &[nbn::AccessType::ColorAttachmentReadWrite],
                previous_layout: nbn::ImageLayout::Optimal,
                next_layout: nbn::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
                image: image.image,
                range: vk::ImageSubresourceRange::default()
                    .layer_count(1)
                    .level_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            }],
        );
        /*nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[nbn::ImageBarrier {
                previous_accesses: &[nbn::AccessType::Present],
                next_accesses: &[nbn::AccessType::ColorAttachmentReadWrite],
                previous_layout: nbn::ImageLayout::Optimal,
                next_layout: nbn::ImageLayout::Optimal,
                discard_contents: true,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
                image: image.image,
                range: vk::ImageSubresourceRange::default()
                    .layer_count(1)
                    .level_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            }],
        );
        device.begin_rendering(
            command_buffer,
            extent.width,
            extent.height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*image.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.5; 4] },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)],
            Some(
                &vk::RenderingAttachmentInfo::default()
                    .image_view(*state.depth_buffer.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    }),
            ),
        );
        device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *state.meshlet_debugging_pipeline,
        );
        device.bind_internal_descriptor_sets_to_all(command_buffer);

        let ptr = *state.models;
        let mut i = 0;
        for models in state._gltf.meshes.iter().take(1) {
            for model in models.iter().take(1) {
                device.push_constants::<(u64, u64)>(
                    &command_buffer,
                    (
                        ptr + (i * std::mem::size_of::<Model>()) as u64,
                        uniforms_ptr,
                    ),
                );
                for (meshlet_index, meshlet) in model.meshlets.iter().enumerate() {
                    device.cmd_draw(
                        **command_buffer,
                        meshlet.triangle_count as u32 * 3,
                        1,
                        0,
                        meshlet_index as _,
                    );
                }

                /*device.push_constants::<(u64, u64)>(
                    &command_buffer,
                    (
                        ptr + (i * std::mem::size_of::<Model>()) as u64,
                        uniforms_ptr,
                    ),
                );*/
                //device.cmd_draw(**command_buffer, model.model.num_indices, 1, 0, 0);
                i += 1;
            }
        }
        device.cmd_end_rendering(**command_buffer);*/

        device.begin_rendering(
            command_buffer,
            extent.width,
            extent.height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*image.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)],
            None,
        );
        state.egui_render.paint(
            device,
            command_buffer,
            &clipped_meshes,
            state.window.scale_factor() as _,
            [extent.width, extent.height],
            current_frame,
        );
        device.cmd_end_rendering(**command_buffer);
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[nbn::ImageBarrier {
                previous_accesses: &[nbn::AccessType::ColorAttachmentReadWrite],
                next_accesses: &[nbn::AccessType::Present],
                previous_layout: nbn::ImageLayout::Optimal,
                next_layout: nbn::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
                image: image.image,
                range: vk::ImageSubresourceRange::default()
                    .layer_count(1)
                    .level_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            }],
        );
        device.end_command_buffer(**command_buffer).unwrap();

        frame.submit(
            device,
            &[vk::CommandBufferSubmitInfo::default().command_buffer(**command_buffer)],
        );
        device
            .swapchain_loader
            .queue_present(
                *device.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[*frame.render_finished_semaphore])
                    .swapchains(&[*state.swapchain])
                    .image_indices(&[next_image]),
            )
            .unwrap();
    }
}
