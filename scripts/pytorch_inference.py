#!/usr/bin/env python3

import argparse
import time
import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from data_processing import dataProcessing, standardize_with_train, collect_data1
from data_processing_3 import valid_test_slice, CustomTensorDataset, dataProcessing_3
from trainer import PIMFuseTrainer
from Net import PIMFuseModel


def load_model(model_path, device='cpu'):
    """加载PyTorch模型"""
    print(f"Loading PyTorch model from: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取超参数
    hparams = checkpoint['hyper_parameters']
    
    # 创建模型
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    
    # 创建argparse.Namespace对象
    from argparse import Namespace
    args = Namespace(**hparams)
    
    # 创建训练器
    model = PIMFuseTrainer(args=args, label_names=label_names)
    
    # 加载状态字典
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    return model, label_names


def load_and_process_data(data_path):
    """加载和处理数据"""
    print(f"Loading data from: {data_path}")
    
    # 检查数据路径
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist. Creating dummy data for testing.")
        # 创建虚拟数据
        dummy_data = np.random.randn(80, 5120, 5)  # 80个样本，5120长度，5个特征
        dummy_labels = np.random.randint(0, 9, 80)  # 80个标签
        return dummy_data, dummy_labels, ["dummy_file"] * 80
    
    # 加载data_2_S1数据
    data_2_path = os.path.join(data_path, "data_2_S1")
    if os.path.exists(data_2_path):
        Train_P, Train_V, Train_Yf, Test_P, Test_V, Test_Yf = dataProcessing(file_path=data_2_path)
        Train_V_std, Test_V_std = standardize_with_train(Train_V, Test_V)
        Train_P_std, Test_P_std = standardize_with_train(Train_P, Test_P)
        
        # 加载data_1_S1数据
        data_1_path = os.path.join(data_path, "data_1_S1")
        if os.path.exists(data_1_path):
            train_x, test_x, train_y, test_y = dataProcessing_3(file_path=data_1_path)
            train_x = train_x.reshape(-1, train_x.shape[2])
            train_x = np.expand_dims(train_x, axis=-1)
            test_x = test_x.reshape(-1, test_x.shape[2])
            test_x = np.expand_dims(test_x, axis=-1)
            
            # 合并数据
            Test_x = np.concatenate((Test_P_std, Test_V_std, test_x), axis=2)
            Test_Y = Test_Yf
            
            print(f"Data loaded successfully. Test samples: {len(Test_x)}")
            filenames = [f"sample_{i:04d}" for i in range(len(Test_x))]
            return Test_x, Test_Y, filenames
    
    # 如果找不到数据，创建虚拟数据
    print("Warning: Required data folders not found. Creating dummy data for testing.")
    dummy_data = np.random.randn(80, 5120, 5)
    dummy_labels = np.random.randint(0, 9, 80)
    filenames = [f"dummy_{i:04d}" for i in range(80)]
    return dummy_data, dummy_labels, filenames


def run_inference(model, data_loader, device, label_names):
    """运行推理"""
    print("Starting inference...")
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            start_time = time.time()
            
            # 获取批次数据
            y, S_P, S_V, pairs, S_P1 = model._get_batch_data(batch)
            
            # 前向传播
            output = model.model(pairs, S_V, S_P, S_P1)
            
            # 获取预测结果
            pred_final = output['pred_final']
            pred_classes = torch.argmax(pred_final, dim=1)
            confidence = torch.softmax(pred_final, dim=1)
            
            # 记录结果
            all_predictions.extend(pred_classes.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            
            # 记录推理时间
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
    
    return all_predictions, all_labels, all_confidences, inference_times


def calculate_metrics(predictions, labels, confidences, inference_times, label_names):
    """计算性能指标"""
    print("Calculating metrics...")
    
    # 基本指标
    accuracy = accuracy_score(labels, predictions)
    total_time = sum(inference_times)
    avg_time_per_sample = total_time / len(predictions)
    
    # 分类报告 - 只对存在的类别生成报告
    unique_labels = sorted(list(set(labels) | set(predictions)))
    actual_label_names = [label_names[i] for i in unique_labels if i < len(label_names)]
    report = classification_report(labels, predictions, labels=unique_labels, target_names=actual_label_names, output_dict=True, zero_division=0)
    
    # 每个类别的准确率 - 只对存在的类别计算
    class_accuracies = {}
    for class_idx in unique_labels:
        class_name = label_names[class_idx] if class_idx < len(label_names) else str(class_idx)
        class_mask = np.array(labels) == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(np.array(labels)[class_mask], np.array(predictions)[class_mask])
            class_accuracies[class_name] = class_acc
        else:
            class_accuracies[class_name] = 0.0
    
    # 置信度统计 - 只对存在的类别计算
    confidence_stats = {}
    for i, class_idx in enumerate(unique_labels):
        class_name = label_names[class_idx] if class_idx < len(label_names) else str(class_idx)
        if i < len(confidences[0]) if confidences else 0:
            class_confidences = [conf[i] if i < len(conf) else 0.0 for conf in confidences]
            confidence_stats[class_name] = {
                'mean': np.mean(class_confidences),
                'std': np.std(class_confidences),
                'max': np.max(class_confidences),
                'min': np.min(class_confidences)
            }
    
    metrics = {
        'accuracy': accuracy,
        'total_inference_time': total_time,
        'avg_inference_time_per_sample': avg_time_per_sample,
        'samples_per_second': len(predictions) / total_time,
        'class_accuracies': class_accuracies,
        'confidence_stats': confidence_stats,
        'classification_report': report
    }
    
    return metrics


def save_results_to_excel(predictions, labels, confidences, filenames, metrics, label_names, output_path):
    """保存结果到Excel文件"""
    print(f"Saving results to: {output_path}")
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 总体统计表
        overall_stats = pd.DataFrame({
            'Metric': ['总体准确率', '推理总时间(s)', '平均每样本推理时间(s)', '样本处理速度(samples/s)', '总样本数'],
            'Value': [
                metrics['accuracy'],
                metrics['total_inference_time'],
                metrics['avg_inference_time_per_sample'],
                metrics['samples_per_second'],
                len(predictions)
            ]
        })
        overall_stats.to_excel(writer, sheet_name='总体统计', index=False)
        
        # 详细结果表
        detailed_results = pd.DataFrame({
            'Sample': filenames,
            'True_Label': labels,
            'Predicted_Label': predictions,
            'Correct': [1 if p == l else 0 for p, l in zip(predictions, labels)]
        })
        
        # 获取实际存在的类别
        existing_classes = list(metrics['class_accuracies'].keys())
        
        # 添加每个类别的置信度 - 只包含实际存在的类别
        for i, class_name in enumerate(existing_classes):
            if i < len(confidences[0]) if confidences else 0:  # 确保置信度数组有足够的元素
                detailed_results[f'Confidence_Class_{class_name}'] = [conf[i] if i < len(conf) else 0.0 for conf in confidences]
        
        detailed_results.to_excel(writer, sheet_name='详细结果', index=False)
        
        # 类别准确率表 - 只包含实际存在的类别
        class_acc_data = []
        for class_name in existing_classes:
            if class_name in metrics['classification_report']:
                class_acc_data.append({
                    'Class': class_name,
                    'Accuracy': metrics['class_accuracies'][class_name],
                    'Precision': metrics['classification_report'][class_name]['precision'],
                    'Recall': metrics['classification_report'][class_name]['recall'],
                    'F1-Score': metrics['classification_report'][class_name]['f1-score']
                })
        class_acc_df = pd.DataFrame(class_acc_data)
        class_acc_df.to_excel(writer, sheet_name='类别统计', index=False)
        
        # 置信度统计表 - 只包含实际存在的类别
        confidence_data = []
        for class_name in existing_classes:
            if class_name in metrics['confidence_stats']:
                stats = metrics['confidence_stats'][class_name]
                confidence_data.append({
                    'Class': class_name,
                    'Mean_Confidence': stats['mean'],
                    'Std_Confidence': stats['std'],
                    'Max_Confidence': stats['max'],
                    'Min_Confidence': stats['min']
                })
        
        confidence_df = pd.DataFrame(confidence_data)
        confidence_df.to_excel(writer, sheet_name='置信度统计', index=False)
    
    print(f"Results saved successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Model Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to PyTorch model checkpoint')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to test data directory')
    parser.add_argument('--output', type=str, default='pytorch_results.xlsx', help='Output Excel file path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max-batches', type=int, default=10, help='Maximum number of batches to process')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # 加载模型
    model, label_names = load_model(args.model_path, device)
    
    # 加载数据
    test_data, test_labels, filenames = load_and_process_data(args.data_path)
    
    # 限制数据量
    max_samples = args.max_batches * args.batch_size
    if len(test_data) > max_samples:
        test_data = test_data[:max_samples]
        test_labels = test_labels[:max_samples]
        filenames = filenames[:max_samples]
    
    # 创建数据集和数据加载器
    test_dataset = CustomTensorDataset(
        torch.tensor(test_data, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collect_data1,
        shuffle=False,
        num_workers=0
    )
    
    # 运行推理
    predictions, labels, confidences, inference_times = run_inference(
        model, test_loader, device, label_names
    )
    
    # 计算指标
    metrics = calculate_metrics(predictions, labels, confidences, inference_times, label_names)
    
    # 显示结果摘要
    print("\n=== PyTorch Model Inference Results ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Inference Time: {metrics['total_inference_time']:.4f}s")
    print(f"Average Time per Sample: {metrics['avg_inference_time_per_sample']:.6f}s")
    print(f"Samples per Second: {metrics['samples_per_second']:.2f}")
    print(f"Total Samples: {len(predictions)}")
    
    print("\nClass Accuracies:")
    for class_name, acc in metrics['class_accuracies'].items():
        print(f"  Class {class_name}: {acc:.4f}")
    
    # 保存结果
    save_results_to_excel(predictions, labels, confidences, filenames, metrics, label_names, args.output)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main() 