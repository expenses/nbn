use crate::*;

pub(crate) unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let mut level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        _ => panic!(),
    };

    if message_id_name == "Loader Message" {
        level = log::Level::Debug;
    }

    // Known error: https://github.com/shader-slang/slang/issues/6218
    if message.contains("VUID-PrimitiveTriangleIndicesEXT-PrimitiveTriangleIndicesEXT-07054") {
        level = log::Level::Debug;
    }

    if message_id_name == "VVL-DEBUG-PRINTF" {
        if let Some((_, message)) = message.rsplit_once('|') {
            log::log!(target: "shader", log::Level::Warn, "{message}");
        } else {
            log::log!(target: "shader", log::Level::Warn, "{message}");
        }
    } else {
        log::log!(
            target: "vulkan",
            level,
            "{message_type:?} [{message_id_name} ({message_id_number})] : {message}"
        );
    }

    vk::FALSE
}

pub fn cast_slice<I: Copy, O: Copy>(slice: &[I]) -> &[O] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}

pub fn cast_slice_mut<I: Copy, O: Copy>(slice: &mut [I]) -> &mut [O] {
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr() as *mut O,
            std::mem::size_of_val(slice) / std::mem::size_of::<O>(),
        )
    }
}

// copied from https://docs.rs/ultraviolet/latest/src/ultraviolet/projection/rh_yup.rs.html#350-365
// The glam version only works for opengl/wgpu I think :(
pub fn perspective_reversed_infinite_z_vk(
    vertical_fov: f32,
    aspect_ratio: f32,
    z_near: f32,
) -> glam::Mat4 {
    let t = (vertical_fov / 2.0).tan();

    let sy = 1.0 / t;

    let sx = sy / aspect_ratio;

    glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
        glam::Vec4::new(0.0, 0.0, z_near, 0.0),
    )
}

pub fn transform_from_mat4(transform: glam::Mat4) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: transform.transpose().to_cols_array()[..12]
            .try_into()
            .unwrap(),
    }
}

pub struct AccelerationStructureInstance {
    pub acceleration_structure_address: u64,
    pub custom_index: u32,
    pub mask: u8,
    pub shader_binding_table_record_offset: u32,
    pub flags: vk::GeometryInstanceFlagsKHR,
    pub transform: vk::TransformMatrixKHR,
}

impl From<AccelerationStructureInstance> for vk::AccelerationStructureInstanceKHR {
    fn from(instance: AccelerationStructureInstance) -> Self {
        Self {
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: instance.acceleration_structure_address,
            },
            transform: instance.transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(
                instance.custom_index,
                instance.mask,
            ),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                instance.shader_binding_table_record_offset,
                instance.flags.as_raw() as _,
            ),
        }
    }
}

pub struct PingPong<T> {
    items: [T; 2],
    flipped: bool,
}

impl<T> PingPong<T> {
    pub fn new(items: [T; 2]) -> Self {
        Self {
            items,
            flipped: false,
        }
    }

    pub fn flip(&mut self) {
        self.flipped = !self.flipped;
    }

    pub fn other(&self) -> &T {
        &self.items[(!self.flipped) as usize]
    }
}

impl<T> std::ops::Deref for PingPong<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.items[self.flipped as usize]
    }
}
