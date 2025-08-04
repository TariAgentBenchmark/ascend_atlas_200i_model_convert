#!/usr/bin/env python3

import argparse
import time
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score
import sys
sys.path.append('.')

from data_processing import dataProcessing, standardize_with_train
from data_processing_3 import dataProcessing_3

# Import ACL inference class
try:
    from scripts.acl_inference import ACLModelInference, prepare_acl_inputs
    ACL_AVAILABLE = True
except ImportError:
    print("Warning: ACL not available. ACL comparison will be skipped.")
    ACL_AVAILABLE = False


def load_test_data(data_path, max_samples=None):
    """Load and process test data for comparison"""
    print(f"Loading test data from: {data_path}")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist. Creating dummy data for testing.")
        # Create dummy data with 7 features for ONNX compatibility
        dummy_data = np.random.randn(40, 5120, 7).astype(np.float32)
        dummy_labels = np.random.randint(0, 9, 40)
        filenames = [f"dummy_{i:04d}" for i in range(40)]
        return dummy_data, dummy_labels, filenames
    
    # Load data_2_S1 data
    data_2_path = os.path.join(data_path, "data_2_S1")
    if os.path.exists(data_2_path):
        Train_P, Train_V, Train_Yf, Test_P, Test_V, Test_Yf = dataProcessing(file_path=data_2_path)
        Train_V_std, Test_V_std = standardize_with_train(Train_V, Test_V)
        Train_P_std, Test_P_std = standardize_with_train(Train_P, Test_P)
        
        # Load data_1_S1 data
        data_1_path = os.path.join(data_path, "data_1_S1")
        if os.path.exists(data_1_path):
            train_x, test_x, train_y, test_y, train_x_fft, test_x_fft = dataProcessing_3(file_path=data_1_path)
            test_x = test_x.reshape(-1, test_x.shape[2])
            test_x = np.expand_dims(test_x, axis=-1)
            test_x_fft = test_x_fft.reshape(-1, test_x_fft.shape[2])
            test_x_fft = np.expand_dims(test_x_fft, axis=-1)
            
            # Separate FFT real and imaginary parts
            test_x_fft_real = np.real(test_x_fft).astype(np.float32)
            test_x_fft_imag = np.imag(test_x_fft).astype(np.float32)
            
            # Combine data (includes FFT real and imaginary parts)
            Test_x = np.concatenate((Test_P_std, Test_V_std, test_x, test_x_fft_real, test_x_fft_imag), axis=2)
            Test_Y = Test_Yf
            
            # Ensure correct data type
            Test_x = Test_x.astype(np.float32)
            
            # Limit samples if specified
            if max_samples and len(Test_x) > max_samples:
                Test_x = Test_x[:max_samples]
                Test_Y = Test_Y[:max_samples]
            
            filenames = [f"sample_{i:04d}" for i in range(len(Test_x))]
            print(f"Data loaded successfully. Test samples: {len(Test_x)}")
            return Test_x, Test_Y, filenames
    
    # Fallback to dummy data
    print("Warning: Required data folders not found. Creating dummy data for testing.")
    dummy_data = np.random.randn(40, 5120, 7).astype(np.float32)
    dummy_labels = np.random.randint(0, 9, 40)
    filenames = [f"dummy_{i:04d}" for i in range(40)]
    return dummy_data, dummy_labels, filenames


def prepare_onnx_inputs(data_batch):
    """Prepare inputs for ONNX model (5 inputs including S_P1_fft)"""
    batch_size = data_batch.shape[0]
    
    # Separate data components
    S_P = data_batch[:, :, 0:1]  # Pressure data [batch, seq_len, 1]
    S_V = data_batch[:, :, 1:4]  # Vibration data [batch, seq_len, 3]
    S_P1 = data_batch[:, :, 4:5]  # Physical data [batch, seq_len, 1]
    
    # Handle FFT data - if 7 dimensions, it includes real and imaginary parts
    if data_batch.shape[2] >= 7:
        S_P1_fft_real = data_batch[:, :, 5:6]  # FFT real part [batch, seq_len, 1]
        S_P1_fft_imag = data_batch[:, :, 6:7]  # FFT imaginary part [batch, seq_len, 1]
        
        # Compute FFT magnitude: |complex| = sqrt(real^2 + imag^2)
        S_P1_fft = np.sqrt(S_P1_fft_real**2 + S_P1_fft_imag**2)
    elif data_batch.shape[2] >= 6:
        S_P1_fft_real = data_batch[:, :, 5:6]  # FFT real part [batch, seq_len, 1]
        S_P1_fft_imag = np.zeros_like(S_P1_fft_real)  # If no imaginary part, set to 0
        
        # Compute FFT magnitude
        S_P1_fft = np.sqrt(S_P1_fft_real**2 + S_P1_fft_imag**2)
    else:
        # Compatibility with old format
        S_P1_fft = np.zeros((batch_size, data_batch.shape[1], 1))
    
    # Check if vibration data exists (determine pairs)
    s_zero = np.array([0, 0, 0])
    s_zero_expanded = np.broadcast_to(s_zero, (S_V.shape[1], 3))
    
    pairs = []
    for i in range(batch_size):
        is_zero = np.allclose(S_V[i], s_zero_expanded)
        pairs.append(0.0 if is_zero else 1.0)
    
    pairs = np.array(pairs, dtype=np.float32)
    
    return pairs, S_V.astype(np.float32), S_P.astype(np.float32), S_P1.astype(np.float32), S_P1_fft.astype(np.float32)


def run_onnx_inference(model_path, data, batch_size):
    """Run ONNX model inference"""
    print(f"Loading ONNX model from: {model_path}")
    
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    
    print("ONNX Model inputs:")
    for input_meta in session.get_inputs():
        print(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")
    
    print("ONNX Model outputs:")
    for output_meta in session.get_outputs():
        print(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")
    
    all_predictions = []
    all_logits = []
    inference_times = []
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print("Starting ONNX inference...")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Prepare batch data
        batch_data = data[start_idx:end_idx]
        
        # Prepare ONNX inputs (5 inputs)
        pairs, S_V, S_P, S_P1, S_P1_fft = prepare_onnx_inputs(batch_data)
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, {
            'pairs': pairs,
            'S_V': S_V,
            'S_P': S_P,
            'S_P1': S_P1,
            'S_P1_fft': S_P1_fft
        })
        end_time = time.time()
        
        # Process output
        pred_final = outputs[0]
        if len(pred_final.shape) == 1:
            pred_final = pred_final.reshape(batch_data.shape[0], -1)
        
        # Save raw logits
        all_logits.extend(pred_final)
        
        # Get predicted classes
        pred_classes = np.argmax(pred_final, axis=1)
        
        # Record results
        all_predictions.extend(pred_classes)
        inference_times.append(end_time - start_time)
        
        if batch_idx % 10 == 0:
            print(f"  Processed batch {batch_idx}/{num_batches}")
    
    return all_predictions, all_logits, inference_times


def run_acl_inference_wrapper(model_path, data, batch_size):
    """Wrapper for ACL model inference"""
    if not ACL_AVAILABLE:
        print("ACL not available, skipping ACL inference")
        return None, None, None
    
    print(f"Loading ACL model from: {model_path}")
    
    # Load ACL model
    try:
        model = ACLModelInference(model_path)
    except Exception as e:
        print(f"Error loading ACL model: {e}")
        return None, None, None
    
    all_predictions = []
    all_logits = []
    inference_times = []
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print("Starting ACL inference...")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Prepare batch data
        batch_data = data[start_idx:end_idx]
        
        # Prepare ACL inputs (4 inputs - without S_P1_fft)
        pairs, S_V, S_P, S_P1, _ = prepare_acl_inputs(batch_data)
        
        # Run inference
        start_time = time.time()
        outputs = model.forward(pairs, S_V, S_P, S_P1)
        end_time = time.time()
        
        # Process output
        pred_final = outputs[0]
        if len(pred_final.shape) == 1:
            pred_final = pred_final.reshape(batch_data.shape[0], -1)
        
        # Save raw logits
        all_logits.extend(pred_final)
        
        # Get predicted classes
        pred_classes = np.argmax(pred_final, axis=1)
        
        # Record results
        all_predictions.extend(pred_classes)
        inference_times.append(end_time - start_time)
        
        if batch_idx % 10 == 0:
            print(f"  Processed batch {batch_idx}/{num_batches}")
    
    # Clean up
    del model
    
    return all_predictions, all_logits, inference_times


def compare_results(onnx_preds, onnx_logits, onnx_times, acl_preds, acl_logits, acl_times, labels):
    """Compare results between ONNX and ACL models"""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Basic metrics
    if acl_preds is not None:
        onnx_accuracy = accuracy_score(labels, onnx_preds)
        acl_accuracy = accuracy_score(labels, acl_preds)
        
        print(f"ONNX Accuracy: {onnx_accuracy:.4f}")
        print(f"ACL Accuracy:  {acl_accuracy:.4f}")
        print(f"Accuracy Difference: {abs(onnx_accuracy - acl_accuracy):.4f}")
        
        # Prediction agreement
        agreement = np.mean(np.array(onnx_preds) == np.array(acl_preds))
        print(f"Prediction Agreement: {agreement:.4f} ({agreement*100:.1f}%)")
        
        # Logits comparison
        if onnx_logits and acl_logits:
            onnx_logits_array = np.array(onnx_logits)
            acl_logits_array = np.array(acl_logits)
            
            # Ensure same shape
            min_samples = min(len(onnx_logits_array), len(acl_logits_array))
            onnx_logits_array = onnx_logits_array[:min_samples]
            acl_logits_array = acl_logits_array[:min_samples]
            
            if onnx_logits_array.shape == acl_logits_array.shape:
                max_logit_diff = np.max(np.abs(onnx_logits_array - acl_logits_array))
                mean_logit_diff = np.mean(np.abs(onnx_logits_array - acl_logits_array))
                
                print(f"Max Logit Difference: {max_logit_diff:.6f}")
                print(f"Mean Logit Difference: {mean_logit_diff:.6f}")
                
                if max_logit_diff < 1e-5:
                    print("✓ Logits match very closely!")
                elif max_logit_diff < 1e-3:
                    print("✓ Logits match reasonably well")
                elif max_logit_diff < 0.1:
                    print("⚠ Logits have moderate differences")
                else:
                    print("⚠ Logits have significant differences")
            else:
                print(f"⚠ Logit shapes don't match: ONNX {onnx_logits_array.shape} vs ACL {acl_logits_array.shape}")
        
        # Performance comparison
        onnx_total_time = sum(onnx_times)
        acl_total_time = sum(acl_times)
        onnx_avg_time = onnx_total_time / len(onnx_preds)
        acl_avg_time = acl_total_time / len(acl_preds)
        
        print(f"\nPerformance Comparison:")
        print(f"ONNX Total Time: {onnx_total_time:.4f}s")
        print(f"ACL Total Time:  {acl_total_time:.4f}s")
        print(f"ONNX Avg Time per Sample: {onnx_avg_time:.6f}s")
        print(f"ACL Avg Time per Sample:  {acl_avg_time:.6f}s")
        print(f"Speedup (ACL vs ONNX): {onnx_avg_time/acl_avg_time:.2f}x")
        
    else:
        print("ACL inference not available - showing ONNX results only:")
        onnx_accuracy = accuracy_score(labels, onnx_preds)
        print(f"ONNX Accuracy: {onnx_accuracy:.4f}")
        
        onnx_total_time = sum(onnx_times)
        onnx_avg_time = onnx_total_time / len(onnx_preds)
        print(f"ONNX Total Time: {onnx_total_time:.4f}s")
        print(f"ONNX Avg Time per Sample: {onnx_avg_time:.6f}s")


def save_comparison_results(onnx_preds, onnx_logits, acl_preds, acl_logits, labels, filenames, output_path):
    """Save detailed comparison results to Excel"""
    print(f"Saving comparison results to: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Detailed comparison results
        comparison_data = {
            'Sample': filenames,
            'True_Label': labels,
            'ONNX_Prediction': onnx_preds,
        }
        
        if acl_preds is not None:
            comparison_data['ACL_Prediction'] = acl_preds
            comparison_data['Predictions_Match'] = [1 if o == a else 0 for o, a in zip(onnx_preds, acl_preds)]
            comparison_data['ONNX_Correct'] = [1 if p == l else 0 for p, l in zip(onnx_preds, labels)]
            comparison_data['ACL_Correct'] = [1 if p == l else 0 for p, l in zip(acl_preds, labels)]
        else:
            comparison_data['ONNX_Correct'] = [1 if p == l else 0 for p, l in zip(onnx_preds, labels)]
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Comparison_Results', index=False)
        
        # ONNX logits
        if onnx_logits:
            onnx_logits_array = np.array(onnx_logits)
            onnx_logits_df = pd.DataFrame(onnx_logits_array, columns=[f'ONNX_Logit_Class_{i}' for i in range(onnx_logits_array.shape[1])])
            onnx_logits_df['Sample'] = filenames
            onnx_logits_df['True_Label'] = labels
            # Reorder columns
            cols = ['Sample', 'True_Label'] + [f'ONNX_Logit_Class_{i}' for i in range(onnx_logits_array.shape[1])]
            onnx_logits_df = onnx_logits_df[cols]
            onnx_logits_df.to_excel(writer, sheet_name='ONNX_Logits', index=False)
        
        # ACL logits
        if acl_logits:
            acl_logits_array = np.array(acl_logits)
            acl_logits_df = pd.DataFrame(acl_logits_array, columns=[f'ACL_Logit_Class_{i}' for i in range(acl_logits_array.shape[1])])
            acl_logits_df['Sample'] = filenames
            acl_logits_df['True_Label'] = labels
            # Reorder columns
            cols = ['Sample', 'True_Label'] + [f'ACL_Logit_Class_{i}' for i in range(acl_logits_array.shape[1])]
            acl_logits_df = acl_logits_df[cols]
            acl_logits_df.to_excel(writer, sheet_name='ACL_Logits', index=False)
        
        # Logits difference (if both available)
        if onnx_logits and acl_logits:
            onnx_array = np.array(onnx_logits)
            acl_array = np.array(acl_logits)
            min_samples = min(len(onnx_array), len(acl_array))
            
            if onnx_array.shape == acl_array.shape:
                diff_array = onnx_array[:min_samples] - acl_array[:min_samples]
                diff_df = pd.DataFrame(diff_array, columns=[f'Diff_Class_{i}' for i in range(diff_array.shape[1])])
                diff_df['Sample'] = filenames[:min_samples]
                diff_df['Max_Abs_Diff'] = np.max(np.abs(diff_array), axis=1)
                # Reorder columns
                cols = ['Sample', 'Max_Abs_Diff'] + [f'Diff_Class_{i}' for i in range(diff_array.shape[1])]
                diff_df = diff_df[cols]
                diff_df.to_excel(writer, sheet_name='Logits_Difference', index=False)
    
    print(f"Comparison results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare ACL and ONNX Model Inference')
    parser.add_argument('--onnx-model', type=str, required=True, help='Path to ONNX model (.onnx) file')
    parser.add_argument('--acl-model', type=str, help='Path to ACL model (.om) file (optional)')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to test data directory')
    parser.add_argument('--output', type=str, default='acl_onnx_comparison.xlsx', help='Output Excel file path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max-samples', type=int, default=80, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Check model files
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model file not found: {args.onnx_model}")
        return
    
    if args.acl_model and not os.path.exists(args.acl_model):
        print(f"Error: ACL model file not found: {args.acl_model}")
        return
    
    # Load test data
    test_data, test_labels, filenames = load_test_data(args.data_path, args.max_samples)
    
    print(f"Loaded {len(test_data)} test samples")
    print(f"Data shape: {test_data.shape}")
    
    # Run ONNX inference
    onnx_preds, onnx_logits, onnx_times = run_onnx_inference(
        args.onnx_model, test_data, args.batch_size
    )
    
    # Run ACL inference (if available and model provided)
    acl_preds, acl_logits, acl_times = None, None, None
    if args.acl_model and ACL_AVAILABLE:
        acl_preds, acl_logits, acl_times = run_acl_inference_wrapper(
            args.acl_model, test_data, args.batch_size
        )
    elif args.acl_model and not ACL_AVAILABLE:
        print("Warning: ACL model path provided but ACL is not available")
    
    # Compare results
    compare_results(onnx_preds, onnx_logits, onnx_times, acl_preds, acl_logits, acl_times, test_labels)
    
    # Save detailed results
    save_comparison_results(onnx_preds, onnx_logits, acl_preds, acl_logits, test_labels, filenames, args.output)


if __name__ == "__main__":
    main()