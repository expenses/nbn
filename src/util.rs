use crate::*;

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
