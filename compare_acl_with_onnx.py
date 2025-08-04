import numpy as np
import onnxruntime as ort
from acl_model import CustomModelInference


def test_acl_model(model_path):
    """Test the ACL model"""
    try:
        # Create ACL model instance
        model = CustomModelInference(model_path)
        print("✓ ACL model loaded successfully")
        
        # Create test inputs (ACL model expects 4 inputs: pairs, S_V, S_P, S_P1)
        batch_size = 1
        sequence_length = 5120
        
        pairs = np.ones((batch_size,), dtype=np.float32)
        S_V = np.random.randn(batch_size, sequence_length, 3).astype(np.float32)
        S_P = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
        S_P1 = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
        
        print(f"Input shapes:")
        print(f"  pairs: {pairs.shape}")
        print(f"  S_V: {S_V.shape}")
        print(f"  S_P: {S_P.shape}")
        print(f"  S_P1: {S_P1.shape}")
        
        # Run inference
        outputs = model.forward(pairs, S_V, S_P, S_P1)
        
        print(f"✓ ACL inference successful")
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
            print(f"  Output {i} sample: {output.flatten()[:5]}...")
        
        # Clean up
        del model
        return outputs, (pairs, S_V, S_P, S_P1)
        
    except Exception as e:
        print(f"✗ ACL model test failed: {e}")
        return None, None


def test_onnx_model(inputs=None):
    """Test the ONNX model"""
    try:
        # Load the ONNX model
        session = ort.InferenceSession("model.onnx")
        print("✓ ONNX model loaded successfully")
        
        # Print model info
        print(f"Model inputs: {[input.name for input in session.get_inputs()]}")
        print(f"Model outputs: {[output.name for output in session.get_outputs()]}")
        
        # Print input shapes
        for input_meta in session.get_inputs():
            print(f"  Input '{input_meta.name}': {input_meta.shape} ({input_meta.type})")
        
        # Print output shapes  
        for output_meta in session.get_outputs():
            print(f"  Output '{output_meta.name}': {output_meta.shape} ({output_meta.type})")
        
        # Create test inputs if not provided
        if inputs is None:
            batch_size = 1
            sequence_length = 5120
            
            pairs = np.ones((batch_size,), dtype=np.float32)
            S_V = np.random.randn(batch_size, sequence_length, 3).astype(np.float32)
            S_P = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
            S_P1 = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
            S_P1_fft = np.random.randn(batch_size, sequence_length, 1).astype(np.float32)
        else:
            pairs, S_V, S_P, S_P1 = inputs
            # Create S_P1_fft for ONNX model if it requires it
            S_P1_fft = np.random.randn(*S_P1.shape).astype(np.float32)
        
        # Prepare inputs dictionary based on model requirements
        input_names = [input.name for input in session.get_inputs()]
        input_dict = {
            'pairs': pairs,
            'S_V': S_V,
            'S_P': S_P,
            'S_P1': S_P1
        }
        
        # Add S_P1_fft if the model expects it
        if 'S_P1_fft' in input_names:
            input_dict['S_P1_fft'] = S_P1_fft
            print(f"  S_P1_fft: {S_P1_fft.shape}")
        
        # Run inference
        outputs = session.run(None, input_dict)
        
        print(f"✓ ONNX inference successful")
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
            print(f"  Output {i} sample: {output.flatten()[:5]}...")
        
        return outputs
        
    except Exception as e:
        print(f"✗ ONNX model test failed: {e}")
        return None


def compare_acl_vs_onnx(acl_model_path, onnx_model_path="model.onnx"):
    """Compare ACL model output with ONNX model output"""
    print("Comparing ACL vs ONNX models...")
    print("=" * 60)
    
    # Test ACL model first
    print("\n1. Testing ACL model:")
    print("-" * 30)
    acl_outputs, test_inputs = test_acl_model(acl_model_path)
    
    if acl_outputs is None:
        print("✗ ACL model test failed, cannot proceed with comparison")
        return False
    
    # Test ONNX model with same inputs
    print("\n2. Testing ONNX model:")
    print("-" * 30)
    onnx_outputs = test_onnx_model(test_inputs)
    
    if onnx_outputs is None:
        print("✗ ONNX model test failed, cannot proceed with comparison")
        return False
    
    # Compare outputs
    print("\n3. Comparing outputs:")
    print("-" * 30)
    
    try:
        # Check if we have the same number of outputs
        if len(acl_outputs) != len(onnx_outputs):
            print(f"⚠ Different number of outputs: ACL={len(acl_outputs)}, ONNX={len(onnx_outputs)}")
            min_outputs = min(len(acl_outputs), len(onnx_outputs))
            print(f"Comparing first {min_outputs} outputs...")
        else:
            min_outputs = len(acl_outputs)
        
        overall_max_diff = 0
        overall_mean_diff = 0
        
        for i in range(min_outputs):
            acl_out = acl_outputs[i]
            onnx_out = onnx_outputs[i]
            
            # Reshape if necessary to match dimensions
            if acl_out.shape != onnx_out.shape:
                print(f"  Output {i}: Shape mismatch - ACL: {acl_out.shape}, ONNX: {onnx_out.shape}")
                # Try to reshape to match
                if acl_out.size == onnx_out.size:
                    acl_out = acl_out.reshape(onnx_out.shape)
                    print(f"  Output {i}: Reshaped ACL output to match ONNX shape")
                else:
                    print(f"  Output {i}: Cannot compare due to different sizes")
                    continue
            
            # Calculate differences
            max_diff = np.max(np.abs(acl_out - onnx_out))
            mean_diff = np.mean(np.abs(acl_out - onnx_out))
            rel_diff = mean_diff / (np.mean(np.abs(onnx_out)) + 1e-8)
            
            print(f"  Output {i}:")
            print(f"    Max difference: {max_diff:.8f}")
            print(f"    Mean difference: {mean_diff:.8f}")
            print(f"    Relative difference: {rel_diff:.8f}")
            
            # Update overall statistics
            overall_max_diff = max(overall_max_diff, max_diff)
            overall_mean_diff += mean_diff
        
        overall_mean_diff /= min_outputs
        
        print(f"\nOverall comparison:")
        print(f"  Max difference: {overall_max_diff:.8f}")
        print(f"  Mean difference: {overall_mean_diff:.8f}")
        
        # Provide assessment
        if overall_max_diff < 1e-6:
            print("✓ Models match very closely!")
        elif overall_max_diff < 1e-4:
            print("✓ Models match reasonably well")
        elif overall_max_diff < 1e-2:
            print("⚠ Models have moderate differences")
        else:
            print("✗ Models have significant differences")
        
        return True
        
    except Exception as e:
        print(f"✗ Output comparison failed: {e}")
        return False


def main():
    """Main function to run the comparison"""
    print("ACL vs ONNX Model Comparison")
    print("=" * 60)
    
    # Model paths - update these according to your setup
    acl_model_path = "model.om"  # Update this path
    onnx_model_path = "model.onnx"
    
    print(f"ACL model path: {acl_model_path}")
    print(f"ONNX model path: {onnx_model_path}")
    print()
    
    # Run comparison
    success = compare_acl_vs_onnx(acl_model_path, onnx_model_path)
    
    if success:
        print("\n✓ Comparison completed successfully!")
    else:
        print("\n✗ Comparison failed!")


if __name__ == "__main__":
    main()