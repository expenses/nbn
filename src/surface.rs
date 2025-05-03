use crate::*;

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct CurrentFrame<'a> {
    frame: &'a mut FrameResources,
    timeline_semaphore: &'a Semaphore,
}

impl std::ops::Deref for CurrentFrame<'_> {
    type Target = FrameResources;

    fn deref(&self) -> &Self::Target {
        self.frame
    }
}

impl CurrentFrame<'_> {
    pub fn submit(&mut self, device: &Device, command_buffers: &[vk::CommandBufferSubmitInfo]) {
        self.frame.number += FRAMES_IN_FLIGHT as u64;
        unsafe {
            device.queue_submit2(
                *device.graphics_queue,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(command_buffers)
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(*self.frame.image_available_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                    .signal_semaphore_infos(&[
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*self.frame.render_finished_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                        vk::SemaphoreSubmitInfo::default()
                            .value(self.frame.number)
                            .semaphore(**self.timeline_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                    ])],
                vk::Fence::null(),
            )
        }
        .unwrap();
    }
}

pub struct SyncResources {
    pub(crate) frames: [FrameResources; FRAMES_IN_FLIGHT],
    pub(crate) timeline_semaphore: Semaphore,
    pub current_frame: usize,
}

impl SyncResources {
    pub fn wait_for_frame(&mut self, device: &Device) -> CurrentFrame {
        let frame = &mut self.frames[self.current_frame];

        unsafe {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[*self.timeline_semaphore])
                        .values(&[frame.number]),
                    !0,
                )
                .unwrap();
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        CurrentFrame {
            frame,
            timeline_semaphore: &self.timeline_semaphore,
        }
    }
}

pub struct FrameResources {
    pub image_available_semaphore: Semaphore,
    pub render_finished_semaphore: Semaphore,
    number: u64,
}

impl FrameResources {
    pub(crate) fn new(device: &Device, number: u64) -> Self {
        Self {
            number,
            image_available_semaphore: device.create_semaphore(),
            render_finished_semaphore: device.create_semaphore(),
        }
    }
}

pub struct Surface {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) loader: ash::khr::surface::Instance,
}
impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

impl std::ops::Deref for Surface {
    type Target = vk::SurfaceKHR;

    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

pub struct WrappedSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) loader: ash::khr::swapchain::Device,
}
impl Drop for WrappedSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl std::ops::Deref for WrappedSwapchain {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

pub struct Swapchain {
    pub(crate) surface: std::mem::ManuallyDrop<Surface>,
    pub(crate) swapchain: std::mem::ManuallyDrop<WrappedSwapchain>,
    pub images: Vec<SwapchainImage>,
    pub create_info: vk::SwapchainCreateInfoKHR<'static>,
}

impl Swapchain {
    pub fn surface_format(&self) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format: self.create_info.image_format,
            color_space: self.create_info.image_color_space,
        }
    }
}

impl std::ops::Deref for Swapchain {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.swapchain);
            std::mem::ManuallyDrop::drop(&mut self.surface);
        }
    }
}

pub struct SwapchainImage {
    pub image: vk::Image,
    pub view: ImageView,
}

impl From<&SwapchainImage> for ImageInfo {
    fn from(image: &SwapchainImage) -> Self {
        Self {
            image: image.image,
            subresource_range: vk::ImageSubresourceRange::default()
                .level_count(1)
                .layer_count(1)
                .aspect_mask(vk::ImageAspectFlags::COLOR),
        }
    }
}
