use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Model load error: {0}")]
    Load(String),
    #[error("Inference error: {0}")]
    Infer(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// Standardized tensor input/output to decouple from specific frameworks.
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub trait SegmentationBackend {
    /// Load a model from the given path.
    fn load_model(&mut self, path: &Path) -> Result<()>;
    /// Perform inference on the provided input tensor, returning the output tensor.
    fn infer(&self, input: &TensorData) -> Result<TensorData>;
}

/// The default, 100% pure-Rust CPU inference engine using tract-onnx.
pub struct TractBackend {
    // TODO: Store the tract model plan here
}

impl Default for TractBackend {
    fn default() -> Self {
        Self {}
    }
}

impl SegmentationBackend for TractBackend {
    fn load_model(&mut self, _path: &Path) -> Result<()> {
        // TODO: implement tract model loading
        Ok(())
    }

    fn infer(&self, _input: &TensorData) -> Result<TensorData> {
        // TODO: implement tract inference
        Err(BackendError::Infer("Tract inference not yet implemented".into()))
    }
}

/// The optional GPU-accelerated inference engine using ONNX Runtime.
#[cfg(feature = "gpu-inference")]
pub struct OrtBackend {
    // TODO: Store the ort session here
}

#[cfg(feature = "gpu-inference")]
impl Default for OrtBackend {
    fn default() -> Self {
        Self {}
    }
}

#[cfg(feature = "gpu-inference")]
impl SegmentationBackend for OrtBackend {
    fn load_model(&mut self, _path: &Path) -> Result<()> {
        // TODO: implement ort model loading
        Ok(())
    }

    fn infer(&self, _input: &TensorData) -> Result<TensorData> {
        // TODO: implement ort inference
        Err(BackendError::Infer("Ort inference not yet implemented".into()))
    }
}
