use crate::*;

pub fn render(device: &nbn::Device, state: &mut WindowState) {
    state.blit_pipelines.refresh(device);
    state.mesh_pipelines.refresh(device);
    state.compute_pipelines.refresh(device);
    state.shadow_pipeline.refresh(device);
    state.shadow_denoising_pipelines.refresh(device);

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

    let near_depth = 0.0001;

    let projection = nbn::perspective_reversed_infinite_z_vk(
        45.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        near_depth,
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
        egui::Window::new("Images").show(&egui_ctx, |ui| {
            ui.label(format!("{:?}", &device.descriptors.sampled_image_count));
            ui.label(format!("{:?}", &device.descriptors.storage_image_count));
            let mut radio_button = |mode| {
                ui.radio_value(&mut state.debug_mode, mode, format!("{:?}", &mode));
            };

            radio_button(DebugMode::None);
            radio_button(DebugMode::Triangles);
            radio_button(DebugMode::Model);
            radio_button(DebugMode::BaseColour);
            radio_button(DebugMode::Normals);
            radio_button(DebugMode::BaseNormals);
            radio_button(DebugMode::MapNormals);
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
            mat_inv: mat.inverse().to_cols_array(),
            view: view.to_cols_array(),
            perspective: projection.to_cols_array(),
            near_plane: near_depth,
            instances: *state.instances,
            meshlet_instances: *state.meshlet_instances,
            extent: [extent.width, extent.height],
            num_instances: state.num_instances,
            visbuffer: *state.framebuffers.vis_index,
            hdrbuffer: *state.framebuffers.hdr_storage_index,
            hdrbuffer_sampled: *state.framebuffers.hdr_sampled_index,
            swapchain_image: *state.swapchain_image_heap_indices[next_image as usize],
            dispatches: *state.dispatches,
            models: *state.models,
            camera_position: camera_pos.into(),
            frustum: [frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z],
            opaque_prefix_sum_values: *state.prefix_sum_values,
            alpha_clip_prefix_sum_values: *state.prefix_sum_values
                + (8 * TOTAL_NUM_INSTANCES_OF_TYPE),
            tonemap_lut_image: *state.tonemap_lut,
            depthbuffer: **state.framebuffers.depth,
            prev_depthbuffer: **state.framebuffers.depth.other(),
            _acceleration_structure: *state.tlas,
            _blue_noise_sobol: *state.blue_noise.sobol,
            _blue_noise_scrambling_tile: *state.blue_noise.scrambling_tile,
            _blue_noise_ranking_tile: *state.blue_noise.ranking_tile,
            frame_index: state.frame_index,
            debug_mode: state.debug_mode as u32,
            half_size_shadow_buffer: *state.framebuffers.half_size_shadow_buffer,
        };

        state.frame_index += 1;

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

        device.dispatch_command_pipeline(
            command_buffer,
            &state.compute_pipelines.reset_buffers,
            *state.dispatches,
            1,
            1,
            1,
        );

        device.insert_global_barrier(
            command_buffer,
            &[nbn::AccessType::ComputeShaderReadWrite],
            &[nbn::AccessType::ComputeShaderReadWrite],
        );

        device.dispatch_command_pipeline(
            command_buffer,
            &state.compute_pipelines.generate_meshlet_prefix_sums,
            uniforms_ptr,
            state.num_instances.div_ceil(64),
            1,
            1,
        );
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
                nbn::ImageBarrier2 {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::DepthStencilAttachmentWrite],
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.depth.image,
                }
                .into(),
                nbn::ImageBarrier2 {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::ColorAttachmentWrite],
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.vis,
                }
                .into(),
            ],
        );

        device.begin_rendering(
            command_buffer,
            extent.width,
            extent.height,
            &[vk::RenderingAttachmentInfo::default()
                .image_view(*state.framebuffers.vis.view)
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
                nbn::ImageBarrier2 {
                    previous_accesses: &[nbn::AccessType::ColorAttachmentWrite],
                    next_accesses: &[nbn::AccessType::ComputeShaderReadOther],
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.vis,
                }
                .into(),
                nbn::ImageBarrier2 {
                    previous_accesses: &[],
                    next_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.hdr,
                }
                .into(),
                nbn::ImageBarrier2 {
                    previous_accesses: &[nbn::AccessType::DepthStencilAttachmentWrite],
                    next_accesses: &[
                        nbn::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                    ],
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.depth.image,
                }
                .into(),
            ],
        );

        device.dispatch_command_pipeline(
            command_buffer,
            &state.shadow_pipeline,
            uniforms_ptr,
            extent.width.div_ceil(2).div_ceil(8),
            extent.height.div_ceil(2).div_ceil(8),
            1,
        );
        device.insert_global_barrier(
            command_buffer,
            &[nbn::AccessType::ComputeShaderReadWrite],
            &[nbn::AccessType::ComputeShaderReadWrite],
        );
        device.dispatch_command_pipeline(
            command_buffer,
            &state.shadow_denoising_pipelines.tile_classification,
            uniforms_ptr,
            extent.width.div_ceil(2).div_ceil(8),
            extent.height.div_ceil(2).div_ceil(8),
            1,
        );

        device.dispatch_command_pipeline(
            command_buffer,
            &state.blit_pipelines.resolve_visbuffer,
            uniforms_ptr,
            extent.width.div_ceil(8),
            extent.height.div_ceil(8),
            1,
        );
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            Some(nbn::GlobalBarrier {
                previous_accesses: &[nbn::AccessType::DepthStencilAttachmentWrite],
                next_accesses: &[nbn::AccessType::DepthStencilAttachmentRead],
            }),
            &[],
            &[
                nbn::ImageBarrier2 {
                    previous_accesses: &[nbn::AccessType::Present],
                    next_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image,
                }
                .into(),
                nbn::ImageBarrier2 {
                    previous_accesses: &[nbn::AccessType::ComputeShaderWrite],
                    next_accesses: &[nbn::AccessType::ComputeShaderReadOther],
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.index,
                    dst_queue_family_index: device.graphics_queue.index,
                    image: &state.framebuffers.hdr,
                }
                .into(),
            ],
        );
        device.dispatch_command_pipeline(
            command_buffer,
            &state.blit_pipelines.tonemap,
            uniforms_ptr,
            extent.width.div_ceil(8),
            extent.height.div_ceil(8),
            1,
        );
        nbn::pipeline_barrier(
            device,
            **command_buffer,
            None,
            &[],
            &[nbn::ImageBarrier2 {
                previous_accesses: &[nbn::AccessType::ComputeShaderWrite],
                next_accesses: &[nbn::AccessType::ColorAttachmentReadWrite],
                discard_contents: false,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
                image,
            }
            .into()],
        );
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
            &[nbn::ImageBarrier2 {
                previous_accesses: &[nbn::AccessType::ColorAttachmentReadWrite],
                next_accesses: &[nbn::AccessType::Present],
                discard_contents: false,
                src_queue_family_index: device.graphics_queue.index,
                dst_queue_family_index: device.graphics_queue.index,
                image,
            }
            .into()],
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

        state.framebuffers.depth.flip();
    }
}
