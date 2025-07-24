#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine


def load_logits_from_excel(file_path):
    """从Excel文件中加载logits数据"""
    print(f"Loading logits from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    # 读取原始Logits表
    logits_df = pd.read_excel(file_path, sheet_name='原始Logits')
    
    # 提取模型名称
    model_name = Path(file_path).stem
    if model_name.endswith('_results'):
        model_name = model_name[:-8]
    
    # 分离logits数据和标签
    logit_columns = [col for col in logits_df.columns if col.startswith('Class_') and col.endswith('_Logit')]
    logits_data = logits_df[logit_columns].values
    
    # 提取其他信息
    samples = logits_df['Sample'].values
    true_labels = logits_df['True_Label'].values
    predicted_labels = logits_df['Predicted_Label'].values
    
    result = {
        'model_name': model_name,
        'logits': logits_data,
        'samples': samples,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'num_classes': len(logit_columns)
    }
    
    print(f"Successfully loaded {model_name} logits: {logits_data.shape}")
    return result


def calculate_logits_similarity(logits1, logits2, model1_name, model2_name):
    """计算两个模型logits之间的相似性指标"""
    print(f"Calculating similarity between {model1_name} and {model2_name}...")
    
    # 确保数据形状匹配
    if logits1.shape != logits2.shape:
        print(f"Warning: Shape mismatch - {model1_name}: {logits1.shape}, {model2_name}: {logits2.shape}")
        min_samples = min(logits1.shape[0], logits2.shape[0])
        min_classes = min(logits1.shape[1], logits2.shape[1])
        logits1 = logits1[:min_samples, :min_classes]
        logits2 = logits2[:min_samples, :min_classes]
    
    # 扁平化用于整体相关性计算
    flat1 = logits1.flatten()
    flat2 = logits2.flatten()
    
    # 计算各种相似性指标
    similarity_metrics = {}
    
    # 均方误差
    mse = mean_squared_error(flat1, flat2)
    similarity_metrics['MSE'] = mse
    similarity_metrics['RMSE'] = np.sqrt(mse)
    
    # 平均绝对误差
    mae = mean_absolute_error(flat1, flat2)
    similarity_metrics['MAE'] = mae
    
    # 皮尔逊相关系数
    pearson_corr, pearson_p = pearsonr(flat1, flat2)
    similarity_metrics['Pearson_Correlation'] = pearson_corr
    similarity_metrics['Pearson_P_Value'] = pearson_p
    
    # 斯皮尔曼相关系数
    spearman_corr, spearman_p = spearmanr(flat1, flat2)
    similarity_metrics['Spearman_Correlation'] = spearman_corr
    similarity_metrics['Spearman_P_Value'] = spearman_p
    
    # 余弦相似度
    cosine_sim = 1 - cosine(flat1, flat2)
    similarity_metrics['Cosine_Similarity'] = cosine_sim
    
    # 计算每个样本的相似性
    sample_similarities = []
    for i in range(logits1.shape[0]):
        sample_cosine = 1 - cosine(logits1[i], logits2[i])
        sample_similarities.append(sample_cosine)
    
    similarity_metrics['Mean_Sample_Cosine_Similarity'] = np.mean(sample_similarities)
    similarity_metrics['Std_Sample_Cosine_Similarity'] = np.std(sample_similarities)
    similarity_metrics['Min_Sample_Cosine_Similarity'] = np.min(sample_similarities)
    similarity_metrics['Max_Sample_Cosine_Similarity'] = np.max(sample_similarities)
    
    # 计算每个类别的相似性
    class_similarities = {}
    for class_idx in range(logits1.shape[1]):
        class_logits1 = logits1[:, class_idx]
        class_logits2 = logits2[:, class_idx]
        
        class_mse = mean_squared_error(class_logits1, class_logits2)
        class_mae = mean_absolute_error(class_logits1, class_logits2)
        class_pearson, _ = pearsonr(class_logits1, class_logits2)
        class_cosine = 1 - cosine(class_logits1, class_logits2)
        
        class_similarities[f'Class_{class_idx}'] = {
            'MSE': class_mse,
            'MAE': class_mae,
            'Pearson_Correlation': class_pearson,
            'Cosine_Similarity': class_cosine
        }
    
    similarity_metrics['Class_Similarities'] = class_similarities
    similarity_metrics['Sample_Similarities'] = sample_similarities
    
    return similarity_metrics


def analyze_prediction_consistency(data_list):
    """分析三个模型预测结果的一致性"""
    print("Analyzing prediction consistency...")
    
    # 确保所有模型的样本数量一致
    min_samples = min([len(data['samples']) for data in data_list])
    
    consistency_analysis = {}
    
    # 预测结果一致性
    predictions = []
    for data in data_list:
        predictions.append(data['predicted_labels'][:min_samples])
    
    # 计算三个模型完全一致的样本比例
    pred_array = np.array(predictions)
    all_same = np.all(pred_array == pred_array[0], axis=0)
    consistency_analysis['All_Models_Agree_Ratio'] = np.mean(all_same)
    consistency_analysis['All_Models_Agree_Count'] = np.sum(all_same)
    
    # 计算两两一致性
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            model1_name = data_list[i]['model_name']
            model2_name = data_list[j]['model_name']
            
            agreement = np.mean(predictions[i] == predictions[j])
            consistency_analysis[f'{model1_name}_vs_{model2_name}_Agreement'] = agreement
    
    # 分析不一致的样本
    inconsistent_samples = []
    for idx, (sample_name, true_label) in enumerate(zip(data_list[0]['samples'][:min_samples], 
                                                        data_list[0]['true_labels'][:min_samples])):
        if not all_same[idx]:
            sample_info = {
                'Sample': sample_name,
                'True_Label': true_label,
            }
            for data in data_list:
                sample_info[f'{data["model_name"]}_Prediction'] = data['predicted_labels'][idx]
            inconsistent_samples.append(sample_info)
    
    consistency_analysis['Inconsistent_Samples'] = inconsistent_samples
    consistency_analysis['Total_Samples'] = min_samples
    
    return consistency_analysis


def create_visualization_plots(data_list, similarity_results, output_dir):
    """创建可视化图表"""
    print("Creating visualization plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 相关性热力图
    if len(data_list) >= 2:
        plt.figure(figsize=(12, 8))
        
        correlation_matrix = []
        model_names = []
        
        for i, data1 in enumerate(data_list):
            model_names.append(data1['model_name'])
            row = []
            for j, data2 in enumerate(data_list):
                if i == j:
                    row.append(1.0)
                elif i < j:
                    key = f"{data1['model_name']}_vs_{data2['model_name']}"
                    if key in similarity_results:
                        row.append(similarity_results[key]['Pearson_Correlation'])
                    else:
                        row.append(0.0)
                else:
                    key = f"{data2['model_name']}_vs_{data1['model_name']}"
                    if key in similarity_results:
                        row.append(similarity_results[key]['Pearson_Correlation'])
                    else:
                        row.append(0.0)
            correlation_matrix.append(row)
        
        correlation_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Pearson Correlation'})
        plt.title('Model Logits Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logits_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 每个类别的logits分布对比
    if len(data_list) >= 2:
        num_classes = data_list[0]['num_classes']
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for class_idx in range(min(num_classes, 9)):
            ax = axes[class_idx]
            
            for data in data_list:
                class_logits = data['logits'][:, class_idx]
                ax.hist(class_logits, alpha=0.6, label=data['model_name'], bins=30)
            
            ax.set_title(f'Class {class_idx} Logits Distribution')
            ax.set_xlabel('Logit Value')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # 隐藏多余的子图
        for idx in range(num_classes, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_logits_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def save_comparison_results(data_list, similarity_results, consistency_analysis, output_path):
    """保存对比结果到Excel"""
    print(f"Saving comparison results to: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 1. 模型基本信息
        model_info_data = []
        for data in data_list:
            model_info_data.append({
                'Model_Name': data['model_name'],
                'Num_Samples': len(data['samples']),
                'Num_Classes': data['num_classes'],
                'Accuracy': np.mean(data['predicted_labels'] == data['true_labels'])
            })
        model_info_df = pd.DataFrame(model_info_data)
        model_info_df.to_excel(writer, sheet_name='模型基本信息', index=False)
        
        # 2. 整体相似性对比
        if similarity_results:
            similarity_data = []
            for comparison_name, metrics in similarity_results.items():
                if not comparison_name.endswith('_class_similarities'):
                    row = {'Comparison': comparison_name}
                    for metric_name, value in metrics.items():
                        if not isinstance(value, (dict, list)):
                            row[metric_name] = value
                    similarity_data.append(row)
            
            similarity_df = pd.DataFrame(similarity_data)
            similarity_df.to_excel(writer, sheet_name='整体相似性', index=False)
        
        # 3. 预测一致性分析
        consistency_data = []
        for key, value in consistency_analysis.items():
            if not isinstance(value, list):
                consistency_data.append({'Metric': key, 'Value': value})
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_df.to_excel(writer, sheet_name='预测一致性', index=False)
        
        # 4. 不一致样本详情
        if consistency_analysis.get('Inconsistent_Samples'):
            inconsistent_df = pd.DataFrame(consistency_analysis['Inconsistent_Samples'])
            inconsistent_df.to_excel(writer, sheet_name='不一致样本', index=False)
        
        # 5. 类别级别的相似性
        if similarity_results:
            for comparison_name, metrics in similarity_results.items():
                if 'Class_Similarities' in metrics:
                    class_sim_data = []
                    for class_name, class_metrics in metrics['Class_Similarities'].items():
                        row = {'Class': class_name}
                        row.update(class_metrics)
                        class_sim_data.append(row)
                    
                    class_sim_df = pd.DataFrame(class_sim_data)
                    sheet_name = f'类别相似性_{comparison_name}'[:31]  # Excel sheet name限制
                    class_sim_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Comparison results saved successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Logits from Different Model Formats')
    parser.add_argument('--pytorch-results', type=str, help='Path to PyTorch results Excel file')
    parser.add_argument('--onnx-results', type=str, help='Path to ONNX results Excel file')
    parser.add_argument('--acl-results', type=str, help='Path to ACL results Excel file')
    parser.add_argument('--output', type=str, default='logits_comparison.xlsx', help='Output comparison file')
    parser.add_argument('--plots-dir', type=str, default='comparison_plots', help='Directory for visualization plots')
    
    args = parser.parse_args()
    
    # 收集所有提供的结果文件
    result_files = []
    if args.pytorch_results:
        result_files.append(args.pytorch_results)
    if args.onnx_results:
        result_files.append(args.onnx_results)
    if args.acl_results:
        result_files.append(args.acl_results)
    
    if len(result_files) < 2:
        print("Error: At least 2 result files are required for comparison")
        print("Please provide at least 2 of: --pytorch-results, --onnx-results, --acl-results")
        return
    
    # 加载所有logits数据
    data_list = []
    for file_path in result_files:
        data = load_logits_from_excel(file_path)
        if data is not None:
            data_list.append(data)
    
    if len(data_list) < 2:
        print("Error: Failed to load sufficient data for comparison")
        return
    
    print(f"Successfully loaded data from {len(data_list)} models")
    
    # 计算两两相似性
    similarity_results = {}
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            data1 = data_list[i]
            data2 = data_list[j]
            
            comparison_key = f"{data1['model_name']}_vs_{data2['model_name']}"
            similarity_metrics = calculate_logits_similarity(
                data1['logits'], data2['logits'], 
                data1['model_name'], data2['model_name']
            )
            similarity_results[comparison_key] = similarity_metrics
    
    # 分析预测一致性
    consistency_analysis = analyze_prediction_consistency(data_list)
    
    # 打印关键结果
    print("\n=== Logits Comparison Results ===")
    for comparison_name, metrics in similarity_results.items():
        print(f"\n{comparison_name}:")
        print(f"  Pearson Correlation: {metrics['Pearson_Correlation']:.6f}")
        print(f"  Cosine Similarity: {metrics['Cosine_Similarity']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
    
    print(f"\nPrediction Consistency:")
    print(f"  All models agree: {consistency_analysis['All_Models_Agree_Ratio']:.4f} "
          f"({consistency_analysis['All_Models_Agree_Count']}/{consistency_analysis['Total_Samples']})")
    
    # 创建可视化图表
    create_visualization_plots(data_list, similarity_results, args.plots_dir)
    
    # 保存结果
    save_comparison_results(data_list, similarity_results, consistency_analysis, args.output)
    
    print(f"\nComparison completed. Results saved to: {args.output}")
    print(f"Visualization plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main() 