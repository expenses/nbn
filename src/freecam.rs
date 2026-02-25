use dolly::prelude::*;
use winit::{event::ElementState, keyboard::KeyCode, window::CursorGrabMode};

pub struct FreeCam {
    pub keyboard: KeyboardState,
    pub camera_rig: dolly::rig::CameraRig,
    pub cursor_grabbed: bool,
}

impl FreeCam {
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            camera_rig: dolly::rig::CameraRig::builder()
                .with(dolly::drivers::Position::new(position))
                .with(dolly::drivers::YawPitch::new())
                .with(dolly::drivers::Smooth::new_position_rotation(1.0, 1.0))
                .build(),
            keyboard: Default::default(),
            cursor_grabbed: false,
        }
    }

    pub fn handle_device_event(&mut self, event: winit::event::DeviceEvent) -> bool {
        if !self.cursor_grabbed {
            return false;
        }

        match event {
            winit::event::DeviceEvent::MouseMotion { delta: (x, y) } => {
                self.camera_rig
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch(-x as f32 / 10.0, -y as f32 / 10.0);
                true
            }
            _ => false,
        }
    }

    pub fn handle_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(code),
                        state: element_state,
                        ..
                    },
                ..
            } => {
                let pressed = *element_state == ElementState::Pressed;

                match code {
                    KeyCode::KeyW => self.keyboard.forwards = pressed,
                    KeyCode::KeyS => self.keyboard.backwards = pressed,
                    KeyCode::KeyD => self.keyboard.right = pressed,
                    KeyCode::KeyA => self.keyboard.left = pressed,
                    KeyCode::KeyG if pressed => {
                        self.cursor_grabbed = !self.cursor_grabbed;
                        window
                            .set_cursor_grab(if self.cursor_grabbed {
                                CursorGrabMode::Confined
                            } else {
                                CursorGrabMode::None
                            })
                            .unwrap();
                        window.set_cursor_visible(!self.cursor_grabbed);
                    }
                    _ => return false,
                }

                return true;
            }
            _ => {}
        }

        false
    }

    pub fn update(&mut self, width: u32, height: u32, delta_time: f32) -> (glam::Mat4, glam::Mat4) {
        let forward = self.keyboard.forwards as i32 - self.keyboard.backwards as i32;
        let right = self.keyboard.right as i32 - self.keyboard.left as i32;

        let prev_transform = self.camera_rig.final_transform;

        self.camera_rig
            .driver_mut::<dolly::drivers::Position>()
            .translate(
                ((glam::Vec3::from_array(prev_transform.forward()) * forward as f32
                    + glam::Vec3::from_array(prev_transform.right()) * right as f32)
                    * 0.25)
                    .to_array(),
            );

        let transform = self.camera_rig.update(delta_time);
        let camera_pos = glam::Vec3::from_array(transform.position.into());

        let view = glam::Mat4::look_to_rh(
            camera_pos,
            glam::Vec3::from_array(transform.forward()),
            glam::Vec3::Y,
        );
        let proj = crate::perspective_reversed_infinite_z_vk(
            59.0_f32.to_radians(),
            width as f32 / height as f32,
            0.001,
        );
        (view, proj)
    }
}

#[derive(Default)]
pub struct KeyboardState {
    pub forwards: bool,
    pub backwards: bool,
    pub left: bool,
    pub right: bool,
}
