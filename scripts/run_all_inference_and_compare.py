#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """运行命令并处理结果"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # 显示最后500个字符
    else:
        print(f"✗ {description} failed with return code {result.returncode}")
        if result.stderr:
            print("Error:", result.stderr)
        if result.stdout:
            print("Output:", result.stdout)
        return False
    
    return True


def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ Found {description}: {file_path}")
        return True
    else:
        print(f"✗ Missing {description}: {file_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all model inference and compare logits')
    
    # 模型文件路径
    parser.add_argument('--pytorch-model', type=str, help='Path to PyTorch model checkpoint (.ckpt)')
    parser.add_argument('--onnx-model', type=str, help='Path to ONNX model (.onnx)')
    parser.add_argument('--acl-model', type=str, help='Path to ACL model (.om)')
    
    # 数据路径
    parser.add_argument('--data-path', type=str, default='./data', help='Path to test data directory')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max-batches', type=int, default=10, help='Maximum number of batches to process')
    parser.add_argument('--device', type=str, default='cpu', help='Device for PyTorch inference (cpu/cuda)')
    
    # 输出设置
    parser.add_argument('--output-dir', type=str, default='inference_results', help='Output directory for results')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip logits comparison step')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查模型文件
    models_to_run = []
    
    if args.pytorch_model:
        if check_file_exists(args.pytorch_model, "PyTorch model"):
            models_to_run.append(('pytorch', args.pytorch_model))
        else:
            print("Warning: PyTorch model not found, skipping PyTorch inference")
    
    if args.onnx_model:
        if check_file_exists(args.onnx_model, "ONNX model"):
            models_to_run.append(('onnx', args.onnx_model))
        else:
            print("Warning: ONNX model not found, skipping ONNX inference")
    
    if args.acl_model:
        if check_file_exists(args.acl_model, "ACL model"):
            models_to_run.append(('acl', args.acl_model))
        else:
            print("Warning: ACL model not found, skipping ACL inference")
    
    if len(models_to_run) == 0:
        print("Error: No valid model files provided. Please specify at least one model:")
        print("  --pytorch-model for PyTorch .ckpt file")
        print("  --onnx-model for ONNX .onnx file")
        print("  --acl-model for ACL .om file")
        return 1
    
    print(f"\nWill run inference for {len(models_to_run)} model(s): {[model[0] for model in models_to_run]}")
    
    # 运行推理
    result_files = []
    
    for model_type, model_path in models_to_run:
        output_file = os.path.join(args.output_dir, f"{model_type}_results.xlsx")
        
        if model_type == 'pytorch':
            command = f"python scripts/pytorch_inference.py --model-path \"{model_path}\" --data-path \"{args.data_path}\" --output \"{output_file}\" --batch-size {args.batch_size} --max-batches {args.max_batches} --device {args.device}"
            
        elif model_type == 'onnx':
            command = f"python scripts/onnx_inference.py --model-path \"{model_path}\" --data-path \"{args.data_path}\" --output \"{output_file}\" --batch-size {args.batch_size} --max-batches {args.max_batches}"
            
        elif model_type == 'acl':
            command = f"python scripts/acl_inference.py --model-path \"{model_path}\" --data-path \"{args.data_path}\" --output \"{output_file}\" --batch-size {args.batch_size} --max-batches {args.max_batches}"
        
        success = run_command(command, f"{model_type.upper()} inference")
        
        if success and os.path.exists(output_file):
            result_files.append((model_type, output_file))
            print(f"✓ {model_type.upper()} results saved to: {output_file}")
        else:
            print(f"✗ Failed to generate {model_type.upper()} results")
    
    if len(result_files) == 0:
        print("Error: No inference results were generated successfully")
        return 1
    
    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully completed inference for {len(result_files)} model(s):")
    for model_type, result_file in result_files:
        print(f"  {model_type.upper()}: {result_file}")
    
    # 运行logits对比（如果有多个模型且未跳过）
    if len(result_files) >= 2 and not args.skip_comparison:
        print(f"\n{'='*60}")
        print("RUNNING LOGITS COMPARISON")
        print(f"{'='*60}")
        
        comparison_output = os.path.join(args.output_dir, "logits_comparison.xlsx")
        plots_dir = os.path.join(args.output_dir, "comparison_plots")
        
        # 构建对比命令
        comparison_args = []
        for model_type, result_file in result_files:
            comparison_args.append(f"--{model_type}-results \"{result_file}\"")
        
        comparison_command = f"python scripts/compare_logits.py {' '.join(comparison_args)} --output \"{comparison_output}\" --plots-dir \"{plots_dir}\""
        
        success = run_command(comparison_command, "Logits comparison")
        
        if success:
            print(f"✓ Logits comparison completed successfully")
            print(f"  Comparison results: {comparison_output}")
            print(f"  Visualization plots: {plots_dir}")
        else:
            print(f"✗ Logits comparison failed")
    
    elif len(result_files) < 2:
        print(f"\nSkipping logits comparison: Need at least 2 models, but only {len(result_files)} available")
    
    else:
        print(f"\nSkipping logits comparison as requested (--skip-comparison)")
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"All results saved in: {args.output_dir}")
    print("Individual inference results:")
    for model_type, result_file in result_files:
        print(f"  {model_type.upper()}: {result_file}")
    
    if len(result_files) >= 2 and not args.skip_comparison:
        print(f"Logits comparison: {os.path.join(args.output_dir, 'logits_comparison.xlsx')}")
        print(f"Plots directory: {os.path.join(args.output_dir, 'comparison_plots')}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 