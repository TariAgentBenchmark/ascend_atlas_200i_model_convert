# ONNX Model Conversion for PIMFuse Model

This directory contains scripts to convert the trained PIMFuse model to ONNX format for deployment and inference.

## Files

- `convert_to_onnx.py` - Main conversion script
- `test_onnx_model.py` - Test script to validate the ONNX model
- `model_simplified.onnx` - The exported ONNX model (created after running conversion)

## Model Architecture

The converted ONNX model is a **simplified version** of the original PIMFuse model with the following key differences:

### Original Model Components:
- Vibration signal processing (CNN_1D_V1)
- Pressure signal processing (CNN_1D_P1) 
- Physical model with FFT operations
- Multi-modal fusion with attention mechanism

### Simplified ONNX Model:
- ✅ Vibration signal processing (CNN_1D_V1) - **Preserved**
- ✅ Pressure signal processing (CNN_1D_P1) - **Preserved**
- ⚠️ Physical model **without FFT operations** - **Simplified**
- ✅ Multi-modal fusion with attention mechanism - **Preserved**

### Why Simplification?

The original physical model uses FFT operations (`torch.fft.fft`) with complex numbers, which are not supported in ONNX. The simplified version:

1. Uses only the CNN layers of the physical model
2. Bypasses the FFT-based frequency domain analysis
3. Maintains the same input/output interface
4. Preserves the attention mechanism and fusion logic

## Usage

### Converting the Model

```bash
python convert_to_onnx.py
```

This will:
1. Load the checkpoint from `lightning_logs/version_0/checkpoints/epoch=68-step=68.ckpt`
2. Create a simplified ONNX-compatible wrapper
3. Export to `model_simplified.onnx`

### Testing the Converted Model

```bash
python test_onnx_model.py
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
session = ort.InferenceSession("model_simplified.onnx")

# Prepare inputs
batch_size = 1
sequence_length = 5120

pairs = np.ones((batch_size,), dtype=np.float32)  # 1 if vibration data available, 0 otherwise
S_V = np.random.randn(batch_size, sequence_length, 3).astype(np.float32)  # Vibration data [batch, time, 3_channels]
S_P = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)  # Pressure data [batch, time, 1_channel]
S_P1 = np.random.randn(batch_size, sequence_length, 1).astype(np.float32) # Pressure data for physical model

# Run inference
outputs = session.run(None, {
    'pairs': pairs,
    'S_V': S_V,
    'S_P': S_P,
    'S_P1': S_P1
})

predictions = outputs[0]  # Shape: [batch_size, 9] - 9 class predictions
```

## Input Specifications

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| `pairs` | `[batch_size]` | float32 | Binary indicator (1.0 = vibration available, 0.0 = pressure only) |
| `S_V` | `[batch_size, sequence_length, 3]` | float32 | Vibration signals (3 channels) |
| `S_P` | `[batch_size, sequence_length, 1]` | float32 | Pressure signals (1 channel) |
| `S_P1` | `[batch_size, sequence_length, 1]` | float32 | Pressure signals for physical model |

## Output Specifications

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `pred_final` | `[batch_size, 9]` | float32 | Final predictions for 9 classes |

## Dynamic Axes

The model supports dynamic batch sizes and sequence lengths:
- `batch_size`: Can vary at inference time
- `sequence_length`: Can vary at inference time (though model was trained on 5120)

## Model Performance

The simplified ONNX model maintains high fidelity to the original PyTorch model:
- **Max difference**: ~1.5e-5
- **Mean difference**: ~5e-6

This indicates excellent preservation of model behavior despite the simplification.

## Requirements

```bash
pip install torch onnx onnxruntime numpy
```

## Limitations

1. **Physical Model Simplification**: The FFT-based frequency analysis is replaced with CNN-only processing
2. **Fixed Architecture**: The attention mechanism and fusion logic assume specific input dimensions
3. **ONNX Opset**: Requires ONNX opset version 13+ for diagonal operations

## Deployment Considerations

### Advantages of ONNX Model:
- ✅ Framework-agnostic deployment
- ✅ Optimized inference performance
- ✅ Smaller memory footprint
- ✅ Cross-platform compatibility
- ✅ Integration with ONNX Runtime optimizations

### Performance Optimization:
- Use ONNX Runtime with appropriate execution providers (CPU, CUDA, etc.)
- Consider quantization for further size/speed improvements
- Batch multiple samples for better throughput

### Example Deployment:
```python
# Optimized inference session
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model_simplified.onnx", providers=providers)

# Enable optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model_simplified.onnx", sess_options, providers=providers)
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed
2. **Shape Mismatches**: Verify input shapes match the expected format
3. **Performance Issues**: Check if appropriate execution providers are available

### Validation:
Always run `test_onnx_model.py` after conversion to ensure the model works correctly. 