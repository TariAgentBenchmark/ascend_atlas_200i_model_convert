# ONNX Model Conversion for PIMFuse Model

This directory contains scripts to convert the trained PIMFuse model to ONNX format for deployment and inference.

## Files

- `convert_to_onnx.py` - Main conversion script  
- `export_onnx.py` - Test script to validate the ONNX model
- `model.onnx` - The exported ONNX model (created after running conversion)

## Model Architecture

The converted ONNX model now uses the **complete original PIMFuse model** with all components preserved:

### Model Components:
- ✅ Vibration signal processing (CNN_1D_V1) - **Fully Preserved**
- ✅ Pressure signal processing (CNN_1D_P1) - **Fully Preserved**
- ✅ Physical model with pre-computed FFT data - **Fully Preserved**
- ✅ Multi-modal fusion with attention mechanism - **Fully Preserved**

### FFT Processing

The FFT operations have been moved to the data preprocessing stage:

1. FFT is computed during data processing and stored as `S_P1_fft`
2. The model accepts pre-computed FFT data as input
3. No FFT operations are performed within the model itself
4. Full accuracy is maintained compared to the original model

## Usage

### Converting the Model

```bash
python convert_to_onnx.py
```

This will:
1. Load the checkpoint from `lightning_logs/version_0/checkpoints/epoch=150-step=150.ckpt`
2. Use the original model directly (no wrapper needed)
3. Export to `model.onnx`

### Testing the Converted Model

```bash
python export_onnx.py
```

This will:
1. Load and validate the ONNX model
2. Run test inference
3. Compare outputs with the original PyTorch model

### Using the ONNX Model for Inference

```python
import numpy as np
import onnxruntime as ort

# Load the model
session = ort.InferenceSession("model.onnx")

# Prepare inputs (including pre-computed FFT data)
batch_size = 1
sequence_length = 5120

pairs = np.ones((batch_size,), dtype=np.float32)
S_V = np.random.randn(batch_size, sequence_length, 3).astype(np.float32)
S_P = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
S_P1 = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
S_P1_fft = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)

# Run inference
outputs = session.run(None, {
    'pairs': pairs,
    'S_V': S_V,
    'S_P': S_P,
    'S_P1': S_P1,
    'S_P1_fft': S_P1_fft
})

prediction = outputs[0]  # pred_final
```

### Input Data Format

- `pairs`: Binary flag (1 if vibration data available, 0 otherwise) - Shape: [batch_size]
- `S_V`: Vibration signal data - Shape: [batch_size, sequence_length, 3]
- `S_P`: Pressure signal data - Shape: [batch_size, sequence_length, 1]  
- `S_P1`: Physical model pressure data - Shape: [batch_size, sequence_length, 1]
- `S_P1_fft`: Pre-computed FFT magnitude of S_P1 - Shape: [batch_size, sequence_length, 1] (float32)

### Output Data Format

- `pred_final`: Final classification predictions - Shape: [batch_size, 9] (9 classes)

## Requirements

```bash
pip install torch onnx onnxruntime numpy
```

## Dynamic Axes

The model supports dynamic batch sizes and sequence lengths:
- `batch_size`: Can vary at inference time
- `sequence_length`: Can vary at inference time (though model was trained on 5120)

## Model Performance

The ONNX model maintains perfect fidelity to the original PyTorch model since no simplification is needed.

## Deployment Considerations

### Advantages of ONNX Model:
- ✅ Framework-agnostic deployment
- ✅ Optimized inference performance
- ✅ Cross-platform compatibility
- ✅ Integration with ONNX Runtime optimizations
- ✅ Full model accuracy preserved

### Performance Optimization:
- Use ONNX Runtime with appropriate execution providers (CPU, CUDA, etc.)
- Consider quantization for further size/speed improvements
- Batch multiple samples for better throughput

### Example Deployment:
```python
# Optimized inference session
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# Enable optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model.onnx", sess_options, providers=providers)
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **Shape Mismatches**: Verify input shapes match the expected format including S_P1_fft
3. **Performance Issues**: Check if appropriate execution providers are available

### Validation:
Always run `export_onnx.py` after conversion to ensure the model works correctly. 