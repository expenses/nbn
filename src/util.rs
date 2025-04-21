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

    if message_id_name == "VVL-DEBUG-PRINTF" {
        let (_, message) = message.rsplit_once('|').unwrap();
        log::log!(target: "shader", log::Level::Warn, "{message}");
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

#[derive(Default)]
pub(crate) struct CountTracker {
    next: u32,
    unused: Vec<u32>,
}

impl CountTracker {
    pub(crate) fn add(&mut self) -> u32 {
        if let Some(index) = self.unused.pop() {
            return index;
        }

        let index = self.next;
        self.next += 1;
        index
    }

    pub(crate) fn remove(&mut self, index: u32) {
        self.unused.push(index);
    }
}
