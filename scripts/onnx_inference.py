#!/usr/bin/env python3

import argparse
import time
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append('.')

from data_processing import dataProcessing, standardize_with_train
from data_processing_3 import dataProcessing_3


def load_onnx_model(model_path):
    """加载ONNX模型"""
    print(f"Loading ONNX model from: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    
    # 创建推理会话
    session = ort.InferenceSession(model_path)
    
    # 获取输入输出信息
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    print(f"Model loaded successfully")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    return session, input_names, output_names


def load_and_process_data(data_path):
    """加载和处理数据"""
    print(f"Loading data from: {data_path}")
    
    # 检查数据路径
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist. Creating dummy data for testing.")
        # 创建虚拟数据
        dummy_data = np.random.randn(80, 5120, 5).astype(np.float32)
        dummy_labels = np.random.randint(0, 9, 80)
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
            
            # 确保数据类型正确
            Test_x = Test_x.astype(np.float32)
            
            print(f"Data loaded successfully. Test samples: {len(Test_x)}")
            filenames = [f"sample_{i:04d}" for i in range(len(Test_x))]
            return Test_x, Test_Y, filenames
    
    # 如果找不到数据，创建虚拟数据
    print("Warning: Required data folders not found. Creating dummy data for testing.")
    dummy_data = np.random.randn(80, 5120, 5).astype(np.float32)
    dummy_labels = np.random.randint(0, 9, 80)
    filenames = [f"dummy_{i:04d}" for i in range(80)]
    return dummy_data, dummy_labels, filenames


def prepare_onnx_inputs(data_batch):
    """准备ONNX模型输入"""
    # 输入数据shape: [batch, sequence_length, features]
    # features: [pressure(1), vibration(3), physical(1)]
    
    batch_size = data_batch.shape[0]
    
    # 分离数据
    S_P = data_batch[:, :, 0:1]  # 压力数据 [batch, seq_len, 1]
    S_V = data_batch[:, :, 1:4]  # 振动数据 [batch, seq_len, 3]
    S_P1 = data_batch[:, :, 4:5]  # 物理数据 [batch, seq_len, 1]
    
    # 检查是否有振动数据（判断pairs）
    # 如果振动数据全为0，则pairs为0，否则为1
    s_zero = np.array([0, 0, 0])
    s_zero_expanded = np.broadcast_to(s_zero, (S_V.shape[1], 3))
    
    pairs = []
    for i in range(batch_size):
        is_zero = np.allclose(S_V[i], s_zero_expanded)
        pairs.append(0.0 if is_zero else 1.0)
    
    pairs = np.array(pairs, dtype=np.float32)
    
    return {
        'pairs': pairs,
        'S_V': S_V.astype(np.float32),
        'S_P': S_P.astype(np.float32),
        'S_P1': S_P1.astype(np.float32)
    }


def run_onnx_inference(session, input_names, output_names, data, batch_size):
    """运行ONNX推理"""
    print("Starting ONNX inference...")
    
    all_predictions = []
    all_confidences = []
    all_logits = []
    inference_times = []
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # 准备批次数据
        batch_data = data[start_idx:end_idx]
        
        # 准备ONNX输入
        onnx_inputs = prepare_onnx_inputs(batch_data)
        
        # 创建输入字典
        input_dict = {name: onnx_inputs[name] for name in input_names}
        
        # 运行推理
        start_time = time.time()
        outputs = session.run(output_names, input_dict)
        end_time = time.time()
        
        # 处理输出
        pred_final = outputs[0]  # 假设第一个输出是预测结果
        
        # 保存原始logits
        all_logits.extend(pred_final)
        
        # 获取预测类别
        pred_classes = np.argmax(pred_final, axis=1)
        
        # 计算置信度（softmax）
        exp_logits = np.exp(pred_final - np.max(pred_final, axis=1, keepdims=True))
        confidence = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 记录结果
        all_predictions.extend(pred_classes)
        all_confidences.extend(confidence)
        inference_times.append(end_time - start_time)
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{num_batches}")
    
    return all_predictions, all_confidences, all_logits, inference_times


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


def save_results_to_excel(predictions, labels, confidences, logits, filenames, metrics, label_names, output_path):
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
        
        # 添加logits数据
        logits_array = np.array(logits)
        for i in range(logits_array.shape[1]):
            detailed_results[f'Logit_Class_{i}'] = logits_array[:, i]
        
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
        
        # Logits数据表
        logits_df = pd.DataFrame(logits, columns=[f'Class_{i}_Logit' for i in range(len(logits[0]))])
        logits_df['Sample'] = filenames
        logits_df['True_Label'] = labels
        logits_df['Predicted_Label'] = predictions
        # 重新排列列的顺序
        cols = ['Sample', 'True_Label', 'Predicted_Label'] + [f'Class_{i}_Logit' for i in range(len(logits[0]))]
        logits_df = logits_df[cols]
        logits_df.to_excel(writer, sheet_name='原始Logits', index=False)
    
    print(f"Results saved successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ONNX Model Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to test data directory')
    parser.add_argument('--output', type=str, default='onnx_results.xlsx', help='Output Excel file path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max-batches', type=int, default=10, help='Maximum number of batches to process')
    
    args = parser.parse_args()
    
    # 标签名称
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    
    # 加载ONNX模型
    session, input_names, output_names = load_onnx_model(args.model_path)
    
    # 加载数据
    test_data, test_labels, filenames = load_and_process_data(args.data_path)
    
    # 限制数据量
    max_samples = args.max_batches * args.batch_size
    if len(test_data) > max_samples:
        test_data = test_data[:max_samples]
        test_labels = test_labels[:max_samples]
        filenames = filenames[:max_samples]
    
    # 运行推理
    predictions, confidences, logits, inference_times = run_onnx_inference(
        session, input_names, output_names, test_data, args.batch_size
    )
    
    # 计算指标
    metrics = calculate_metrics(predictions, test_labels, confidences, inference_times, label_names)
    
    # 显示结果摘要
    print("\n=== ONNX Model Inference Results ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Inference Time: {metrics['total_inference_time']:.4f}s")
    print(f"Average Time per Sample: {metrics['avg_inference_time_per_sample']:.6f}s")
    print(f"Samples per Second: {metrics['samples_per_second']:.2f}")
    print(f"Total Samples: {len(predictions)}")
    
    print("\nClass Accuracies:")
    for class_name, acc in metrics['class_accuracies'].items():
        print(f"  Class {class_name}: {acc:.4f}")
    
    # 保存结果
    save_results_to_excel(predictions, test_labels, confidences, logits, filenames, metrics, label_names, args.output)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main() 