#!/usr/bin/env python3

import argparse
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.path.append('.')

from data_processing import dataProcessing, standardize_with_train
from data_processing_3 import dataProcessing_3

# 导入ACL模型推理类
import acl


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


class ACLModelInference:
    """ACL模型推理类"""
    
    def __init__(self, model_path):
        self.device_id = 0
        self.model_path = model_path
        
        # 初始化ACL
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        
        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        
        # 创建输入输出数据集
        self.input_dataset, self.input_data = self.prepare_dataset("input")
        self.output_dataset, self.output_data = self.prepare_dataset("output")
        
        print(f"ACL model loaded successfully from: {model_path}")
    
    def prepare_dataset(self, io_type):
        """准备数据集"""
        if io_type == "input":
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        
        dataset = acl.mdl.create_dataset()
        datas = []
        
        for i in range(io_num):
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        
        return dataset, datas
    
    def forward(self, pairs, S_V, S_P, S_P1):
        """执行推理"""
        inputs = [pairs, S_V, S_P, S_P1]
        
        # 将输入数据拷贝到设备内存
        for i in range(len(inputs)):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(
                self.input_data[i]["buffer"],
                self.input_data[i]["size"],
                bytes_ptr,
                len(bytes_data),
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
        
        # 执行模型推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        
        # 处理输出数据
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            ret = acl.rt.memcpy(
                buffer_host,
                self.output_data[i]["size"],
                self.output_data[i]["buffer"],
                self.output_data[i]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data)
        
        return inference_result
    
    def __del__(self):
        """析构函数"""
        # 释放资源
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"])
                ret = acl.rt.free(item["buffer"])
        
        ret = acl.mdl.destroy_dataset(self.input_dataset)
        ret = acl.mdl.destroy_dataset(self.output_dataset)
        ret = acl.mdl.destroy_desc(self.model_desc)
        ret = acl.mdl.unload(self.model_id)
        ret = acl.rt.reset_device(self.device_id)
        ret = acl.finalize()


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


def prepare_acl_inputs(data_batch):
    """准备ACL模型输入"""
    batch_size = data_batch.shape[0]
    
    # 分离数据
    S_P = data_batch[:, :, 0:1]  # 压力数据
    S_V = data_batch[:, :, 1:4]  # 振动数据
    S_P1 = data_batch[:, :, 4:5]  # 物理数据
    
    # 检查是否有振动数据（判断pairs）
    s_zero = np.array([0, 0, 0])
    s_zero_expanded = np.broadcast_to(s_zero, (S_V.shape[1], 3))
    
    pairs = []
    for i in range(batch_size):
        is_zero = np.allclose(S_V[i], s_zero_expanded)
        pairs.append(0.0 if is_zero else 1.0)
    
    pairs = np.array(pairs, dtype=np.float32)
    
    return pairs, S_V.astype(np.float32), S_P.astype(np.float32), S_P1.astype(np.float32)


def run_acl_inference(model, data, batch_size, label_names):
    """运行ACL推理"""
    print("Starting ACL inference...")
    
    all_predictions = []
    all_confidences = []
    inference_times = []
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # 准备批次数据
        batch_data = data[start_idx:end_idx]
        
        # 准备ACL输入
        pairs, S_V, S_P, S_P1 = prepare_acl_inputs(batch_data)
        
        # 运行推理
        start_time = time.time()
        outputs = model.forward(pairs, S_V, S_P, S_P1)
        end_time = time.time()
        
        # 处理输出
        # 假设第一个输出是预测结果，重新调整形状
        pred_final = outputs[0]
        if len(pred_final.shape) == 1:
            pred_final = pred_final.reshape(batch_data.shape[0], -1)
        
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
    
    return all_predictions, all_confidences, inference_times


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
    parser = argparse.ArgumentParser(description='ACL Model Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ACL model (.om) file')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to test data directory')
    parser.add_argument('--output', type=str, default='acl_results.xlsx', help='Output Excel file path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max-batches', type=int, default=10, help='Maximum number of batches to process')
    
    args = parser.parse_args()
    
    # 标签名称
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    
    # 检查ACL模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Error: ACL model file not found: {args.model_path}")
        print("Please ensure the model file exists and is accessible.")
        return
    
    # 加载ACL模型
    model = None
    try:
        model = ACLModelInference(args.model_path)
    except Exception as e:
        print(f"Error loading ACL model: {e}")
        print("Please ensure ACL runtime is properly installed and configured.")
        return
    
    # 加载数据
    test_data, test_labels, filenames = load_and_process_data(args.data_path)
    
    # 限制数据量
    max_samples = args.max_batches * args.batch_size
    if len(test_data) > max_samples:
        test_data = test_data[:max_samples]
        test_labels = test_labels[:max_samples]
        filenames = filenames[:max_samples]
    
    # 运行推理
    predictions, confidences, inference_times = run_acl_inference(
        model, test_data, args.batch_size, label_names
    )
    
    # 计算指标
    metrics = calculate_metrics(predictions, test_labels, confidences, inference_times, label_names)
    
    # 显示结果摘要
    print("\n=== ACL Model Inference Results ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Inference Time: {metrics['total_inference_time']:.4f}s")
    print(f"Average Time per Sample: {metrics['avg_inference_time_per_sample']:.6f}s")
    print(f"Samples per Second: {metrics['samples_per_second']:.2f}")
    print(f"Total Samples: {len(predictions)}")
    
    print("\nClass Accuracies:")
    for class_name, acc in metrics['class_accuracies'].items():
        print(f"  Class {class_name}: {acc:.4f}")
    
    # 保存结果
    save_results_to_excel(predictions, test_labels, confidences, filenames, metrics, label_names, args.output)
    
    print(f"\nResults saved to: {args.output}")
    
    # 清理资源
    if model:
        del model


if __name__ == "__main__":
    main() 