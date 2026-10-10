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

use tract_onnx::prelude::*;

/// The default, 100% pure-Rust CPU inference engine using tract-onnx.
#[derive(Default)]
pub struct TractBackend {
    plan: Option<std::sync::Arc<TypedSimplePlan>>,
}

impl SegmentationBackend for TractBackend {
    fn load_model(&mut self, path: &Path) -> Result<()> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| BackendError::Load(e.to_string()))?
            .into_optimized()
            .map_err(|e| BackendError::Load(e.to_string()))?
            .into_runnable()
            .map_err(|e| BackendError::Load(e.to_string()))?;
        self.plan = Some(model);
        Ok(())
    }

    fn infer(&self, input: &TensorData) -> Result<TensorData> {
        let plan = self.plan.as_ref().ok_or_else(|| BackendError::Infer("Model not loaded".into()))?;
        let tensor = tract_onnx::prelude::Tensor::from_shape(&input.shape, &input.data)
            .map_err(|e| BackendError::Infer(e.to_string()))?;
        let result = plan
            .run(tvec!(tensor.into()))
            .map_err(|e| BackendError::Infer(e.to_string()))?;
        let output_tensor = result
            .first()
            .ok_or_else(|| BackendError::Infer("No output tensor returned".into()))?;
        let shape: Vec<usize> = output_tensor.shape().to_vec();
        let data: Vec<f32> = output_tensor
            .to_plain_array_view::<f32>()
            .map_err(|e| BackendError::Infer(e.to_string()))?
            .iter()
            .copied()
            .collect();
        Ok(TensorData { shape, data })
    }
}

/// The optional GPU-accelerated inference engine using ONNX Runtime.
#[cfg(feature = "gpu-inference")]
#[derive(Default)]
pub struct OrtBackend {
    // TODO: Store the ort session here
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
