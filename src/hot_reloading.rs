pub use crate::*;

pub struct ReloadableShader {
    pub(crate) inner: ShaderModule,
    pub(crate) dirty: Arc<AtomicBool>,
    pub(crate) path: PathBuf,
    pub(crate) _watcher: notify::RecommendedWatcher,
}

impl ReloadableShader {
    pub fn try_reload(&mut self, device: &Device) -> bool {
        let dirty = self.dirty.swap(false, Ordering::Relaxed);

        if dirty {
            self.inner = device.load_shader(&self.path);
        }

        dirty
    }
}

impl std::ops::Deref for ReloadableShader {
    type Target = ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct ReloadablePipeline<T: 'static> {
    shader: ReloadableShader,
    pipeline: T,
    create_fn: &'static dyn Fn(&Device, &ShaderModule) -> T,
}

impl<T> ReloadablePipeline<T> {
    pub fn new(
        device: &Device,
        shader: ReloadableShader,
        create_fn: &'static dyn Fn(&Device, &ShaderModule) -> T,
    ) -> Self {
        Self {
            pipeline: (create_fn)(device, &shader),
            create_fn,
            shader,
        }
    }

    pub fn refresh(&mut self, device: &Device) {
        if self.shader.try_reload(device) {
            self.pipeline = (self.create_fn)(device, &self.shader);
        }
    }
}

impl<T> std::ops::Deref for ReloadablePipeline<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}
