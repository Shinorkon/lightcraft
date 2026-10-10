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

#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::*;

/// The default, 100% pure-Rust CPU inference engine using tract-onnx.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct TractBackend {
    plan: Option<std::sync::Arc<TypedSimplePlan>>,
}

#[cfg(not(target_arch = "wasm32"))]
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
        let tensor = tract_onnx::prelude::Tensor::from_shape(&input.shape, &input.data).map_err(|e| BackendError::Infer(e.to_string()))?;
        let result = plan.run(tvec!(tensor.into())).map_err(|e| BackendError::Infer(e.to_string()))?;
        let output_tensor = result.first().ok_or_else(|| BackendError::Infer("No output tensor returned".into()))?;
        let shape: Vec<usize> = output_tensor.shape().to_vec();
        let data: Vec<f32> = output_tensor.to_plain_array_view::<f32>().map_err(|e| BackendError::Infer(e.to_string()))?.iter().copied().collect();
        Ok(TensorData { shape, data })
    }
}

/// Fallback dummy backend for wasm32 where AI models are disabled.
#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub struct DummyBackend;

#[cfg(target_arch = "wasm32")]
impl SegmentationBackend for DummyBackend {
    fn load_model(&mut self, _path: &Path) -> Result<()> {
        Err(BackendError::Load("Segmentation not supported on wasm32".into()))
    }

    fn infer(&self, _input: &TensorData) -> Result<TensorData> {
        Err(BackendError::Infer("Segmentation not supported on wasm32".into()))
    }
}

/// The optional GPU-accelerated inference engine using ONNX Runtime.
#[cfg(all(feature = "gpu-inference", not(target_arch = "wasm32")))]
#[derive(Default)]
pub struct OrtBackend {
    session: Option<std::sync::Arc<std::sync::Mutex<ort::session::Session>>>,
}

#[cfg(all(feature = "gpu-inference", not(target_arch = "wasm32")))]
impl SegmentationBackend for OrtBackend {
    fn load_model(&mut self, path: &Path) -> Result<()> {
        let mut builder = ort::session::Session::builder().map_err(|e| BackendError::Load(e.to_string()))?;
        let session = builder.commit_from_file(path).map_err(|e| BackendError::Load(e.to_string()))?;
        self.session = Some(std::sync::Arc::new(std::sync::Mutex::new(session)));
        Ok(())
    }

    fn infer(&self, input: &TensorData) -> Result<TensorData> {
        let session_arc = self.session.as_ref().ok_or_else(|| BackendError::Infer("Model not loaded".into()))?;
        let mut session = session_arc.lock().map_err(|e| BackendError::Infer(e.to_string()))?;
        let shape: Vec<i64> = input.shape.iter().map(|&x| x as i64).collect();
        let tensor = ort::value::Tensor::from_array((shape.as_slice(), input.data.clone().into_boxed_slice()))
            .map_err(|e| BackendError::Infer(e.to_string()))?;
        let outputs = session.run(ort::inputs![tensor]).map_err(|e| BackendError::Infer(e.to_string()))?;
        let output_tensor = outputs.values().next().ok_or_else(|| BackendError::Infer("No output tensor returned".into()))?;
        let (out_shape, out_data) = output_tensor.try_extract_tensor::<f32>().map_err(|e| BackendError::Infer(e.to_string()))?;
        Ok(TensorData { shape: out_shape.iter().map(|&x| x as usize).collect(), data: out_data.to_vec() })
    }
}

/// Instantiate the preferred segmentation backend: `OrtBackend` when compiled with
/// `gpu-inference`, otherwise `TractBackend`.
pub fn default_backend() -> Box<dyn SegmentationBackend + Send + Sync> {
    #[cfg(all(feature = "gpu-inference", not(target_arch = "wasm32")))]
    {
        Box::new(OrtBackend::default())
    }
    #[cfg(all(not(feature = "gpu-inference"), not(target_arch = "wasm32")))]
    {
        Box::new(TractBackend::default())
    }
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(DummyBackend::default())
    }
}
