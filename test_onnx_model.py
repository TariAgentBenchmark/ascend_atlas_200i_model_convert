import numpy as np
import onnxruntime as ort
import torch

def test_onnx_model():
    """Test the exported ONNX model"""
    
    # Load the ONNX model
    try:
        session = ort.InferenceSession("model_simplified.onnx")
        print("✓ ONNX model loaded successfully")
        
        # Print model info
        print(f"Model inputs: {[input.name for input in session.get_inputs()]}")
        print(f"Model outputs: {[output.name for output in session.get_outputs()]}")
        
        # Print input shapes
        for input_meta in session.get_inputs():
            print(f"Input '{input_meta.name}': {input_meta.shape} ({input_meta.type})")
        
        # Print output shapes  
        for output_meta in session.get_outputs():
            print(f"Output '{output_meta.name}': {output_meta.shape} ({output_meta.type})")
            
    except Exception as e:
        print(f"✗ Failed to load ONNX model: {e}")
        return False
    
    # Test inference
    try:
        # Create test inputs
        batch_size = 1
        sequence_length = 5120
        
        pairs = np.ones((batch_size,), dtype=np.float32)
        S_V = np.random.randn(batch_size, sequence_length, 3).astype(np.float32)
        S_P = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
        S_P1 = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
        
        # Run inference
        outputs = session.run(
            None,  # Return all outputs
            {
                'pairs': pairs,
                'S_V': S_V, 
                'S_P': S_P,
                'S_P1': S_P1
            }
        )
        
        print(f"✓ ONNX inference successful")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output sample: {outputs[0][0][:5]}...")  # Show first 5 values
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX inference failed: {e}")
        return False

def compare_pytorch_vs_onnx():
    """Compare PyTorch model output with ONNX model output"""
    try:
        # Load ONNX model
        session = ort.InferenceSession("model_simplified.onnx")
        
        # Load PyTorch model
        from convert_to_onnx import SimplifiedONNXWrapper, PIMFuseTrainer
        from argparse import Namespace
        
        checkpoint = torch.load("lightning_logs/version_0/checkpoints/epoch=68-step=68.ckpt", 
                               map_location=torch.device('cpu'))
        hparams = checkpoint['hyper_parameters']
        args = Namespace(**hparams)
        
        model = PIMFuseTrainer(args=args, label_names=["0","1","2","3","4","5","6","7","8"])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        pytorch_model = SimplifiedONNXWrapper(model.model)
        pytorch_model.eval()
        
        # Create test inputs
        batch_size = 1
        sequence_length = 5120
        
        pairs_torch = torch.ones((batch_size,), dtype=torch.float32)
        S_V_torch = torch.randn(batch_size, sequence_length, 3, dtype=torch.float32)
        S_P_torch = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)
        S_P1_torch = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)
        
        # Convert to numpy for ONNX
        pairs_np = pairs_torch.numpy()
        S_V_np = S_V_torch.numpy()
        S_P_np = S_P_torch.numpy()
        S_P1_np = S_P1_torch.numpy()
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(pairs_torch, S_V_torch, S_P_torch, S_P1_torch)
        
        # ONNX inference
        onnx_output = session.run(None, {
            'pairs': pairs_np,
            'S_V': S_V_np,
            'S_P': S_P_np, 
            'S_P1': S_P1_np
        })[0]
        
        # Compare outputs
        pytorch_np = pytorch_output.numpy()
        max_diff = np.max(np.abs(pytorch_np - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_np - onnx_output))
        
        print(f"✓ Model comparison completed")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ Models match very closely!")
        elif max_diff < 1e-3:
            print("✓ Models match reasonably well")
        else:
            print("⚠ Models have significant differences")
            
        return True
        
    except Exception as e:
        print(f"✗ Model comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ONNX model...")
    print("=" * 50)
    
    success = test_onnx_model()
    
    if success:
        print("\n" + "=" * 50)
        print("Comparing PyTorch vs ONNX outputs...")
        compare_pytorch_vs_onnx() 