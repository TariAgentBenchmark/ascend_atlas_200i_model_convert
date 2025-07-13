#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_result_file(file_path):
    """加载推理结果文件"""
    print(f"Loading result file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        # 读取所有sheet
        sheets = pd.read_excel(file_path, sheet_name=None)
        
        # 提取模型名称（从文件名）
        model_name = Path(file_path).stem
        if model_name.endswith('_results'):
            model_name = model_name[:-8]  # 去掉 '_results' 后缀
        
        result = {
            'model_name': model_name,
            'file_path': file_path,
            'sheets': sheets
        }
        
        print(f"Successfully loaded {model_name} results")
        return result
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_metrics(result_data):
    """从结果数据中提取关键指标"""
    if not result_data:
        return None
    
    sheets = result_data['sheets']
    model_name = result_data['model_name']
    
    metrics = {'model_name': model_name}
    
    # 提取总体统计
    if '总体统计' in sheets:
        overall_stats = sheets['总体统计']
        for _, row in overall_stats.iterrows():
            metric_name = row['Metric']
            value = row['Value']
            if metric_name == '总体准确率':
                metrics['accuracy'] = value
            elif metric_name == '推理总时间(s)':
                metrics['total_time'] = value
            elif metric_name == '平均每样本推理时间(s)':
                metrics['avg_time_per_sample'] = value
            elif metric_name == '样本处理速度(samples/s)':
                metrics['samples_per_second'] = value
            elif metric_name == '总样本数':
                metrics['total_samples'] = value
    
    # 提取类别统计
    if '类别统计' in sheets:
        class_stats = sheets['类别统计']
        class_accuracies = {}
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        
        for _, row in class_stats.iterrows():
            class_name = str(row['Class'])
            class_accuracies[class_name] = row['Accuracy']
            class_precision[class_name] = row['Precision']
            class_recall[class_name] = row['Recall']
            class_f1[class_name] = row['F1-Score']
        
        metrics['class_accuracies'] = class_accuracies
        metrics['class_precision'] = class_precision
        metrics['class_recall'] = class_recall
        metrics['class_f1'] = class_f1
    
    # 提取置信度统计
    if '置信度统计' in sheets:
        confidence_stats = sheets['置信度统计']
        confidence_means = {}
        confidence_stds = {}
        
        for _, row in confidence_stats.iterrows():
            class_name = str(row['Class'])
            confidence_means[class_name] = row['Mean_Confidence']
            confidence_stds[class_name] = row['Std_Confidence']
        
        metrics['confidence_means'] = confidence_means
        metrics['confidence_stds'] = confidence_stds
    
    return metrics


def create_comparison_tables(all_metrics):
    """创建对比表格"""
    model_names = [m['model_name'] for m in all_metrics]
    
    # 总体指标对比
    overall_comparison = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [m.get('accuracy', 0) for m in all_metrics],
        'Total_Time(s)': [m.get('total_time', 0) for m in all_metrics],
        'Avg_Time_Per_Sample(s)': [m.get('avg_time_per_sample', 0) for m in all_metrics],
        'Samples_Per_Second': [m.get('samples_per_second', 0) for m in all_metrics],
        'Total_Samples': [m.get('total_samples', 0) for m in all_metrics]
    })
    
    # 速度对比
    speed_comparison = pd.DataFrame({
        'Model': model_names,
        'Samples_Per_Second': [m.get('samples_per_second', 0) for m in all_metrics],
        'Avg_Time_Per_Sample(ms)': [m.get('avg_time_per_sample', 0) * 1000 for m in all_metrics],
        'Total_Time(s)': [m.get('total_time', 0) for m in all_metrics]
    })
    
    # 类别准确率对比
    class_names = list(all_metrics[0]['class_accuracies'].keys()) if all_metrics and 'class_accuracies' in all_metrics[0] else []
    class_accuracy_data = {'Class': class_names}
    
    for metric in all_metrics:
        model_name = metric['model_name']
        class_accuracies = metric.get('class_accuracies', {})
        class_accuracy_data[model_name] = [class_accuracies.get(cls, 0) for cls in class_names]
    
    class_accuracy_comparison = pd.DataFrame(class_accuracy_data)
    
    # 模型排名
    ranking_data = []
    for metric in all_metrics:
        ranking_data.append({
            'Model': metric['model_name'],
            'Accuracy': metric.get('accuracy', 0),
            'Speed_Rank': 0,  # 将在后面计算
            'Accuracy_Rank': 0,  # 将在后面计算
            'Combined_Score': 0  # 将在后面计算
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # 计算排名
    ranking_df['Accuracy_Rank'] = ranking_df['Accuracy'].rank(ascending=False, method='min')
    ranking_df['Speed_Rank'] = pd.DataFrame(speed_comparison.set_index('Model')['Samples_Per_Second']).rank(ascending=False, method='min')
    ranking_df['Combined_Score'] = (ranking_df['Accuracy_Rank'] + ranking_df['Speed_Rank']) / 2
    ranking_df = ranking_df.sort_values('Combined_Score')
    
    return {
        'overall_comparison': overall_comparison,
        'speed_comparison': speed_comparison,
        'class_accuracy_comparison': class_accuracy_comparison,
        'ranking': ranking_df
    }


def create_detailed_statistics(all_metrics):
    """创建详细统计信息"""
    detailed_stats = []
    
    for metric in all_metrics:
        model_name = metric['model_name']
        
        # 基本统计
        basic_stats = {
            'Model': model_name,
            'Overall_Accuracy': metric.get('accuracy', 0),
            'Total_Time(s)': metric.get('total_time', 0),
            'Avg_Time_Per_Sample(s)': metric.get('avg_time_per_sample', 0),
            'Samples_Per_Second': metric.get('samples_per_second', 0),
            'Total_Samples': metric.get('total_samples', 0)
        }
        
        # 类别统计
        class_accuracies = metric.get('class_accuracies', {})
        for class_name, acc in class_accuracies.items():
            basic_stats[f'Class_{class_name}_Accuracy'] = acc
        
        # 置信度统计
        confidence_means = metric.get('confidence_means', {})
        for class_name, conf in confidence_means.items():
            basic_stats[f'Class_{class_name}_Confidence_Mean'] = conf
        
        detailed_stats.append(basic_stats)
    
    return pd.DataFrame(detailed_stats)


def generate_summary_report(all_metrics, comparison_tables):
    """生成摘要报告"""
    print("\n" + "="*80)
    print("                      MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # 总体对比
    overall_df = comparison_tables['overall_comparison']
    print("\n1. OVERALL PERFORMANCE COMPARISON")
    print("-" * 50)
    for _, row in overall_df.iterrows():
        print(f"{row['Model']:12} | Accuracy: {row['Accuracy']:.4f} | Speed: {row['Samples_Per_Second']:.2f} samples/s")
    
    # 最佳性能模型
    best_accuracy_model = overall_df.loc[overall_df['Accuracy'].idxmax()]['Model']
    best_speed_model = overall_df.loc[overall_df['Samples_Per_Second'].idxmax()]['Model']
    
    print(f"\n2. BEST PERFORMING MODELS")
    print("-" * 50)
    print(f"Best Accuracy: {best_accuracy_model} ({overall_df['Accuracy'].max():.4f})")
    print(f"Best Speed: {best_speed_model} ({overall_df['Samples_Per_Second'].max():.2f} samples/s)")
    
    # 速度对比
    speed_df = comparison_tables['speed_comparison']
    print(f"\n3. SPEED COMPARISON")
    print("-" * 50)
    for _, row in speed_df.iterrows():
        print(f"{row['Model']:12} | {row['Samples_Per_Second']:6.2f} samples/s | {row['Avg_Time_Per_Sample(ms)']:6.2f} ms/sample")
    
    # 类别准确率对比
    class_df = comparison_tables['class_accuracy_comparison']
    print(f"\n4. CLASS ACCURACY COMPARISON")
    print("-" * 50)
    if not class_df.empty:
        for _, row in class_df.iterrows():
            class_name = row['Class']
            print(f"Class {class_name}:")
            for col in class_df.columns[1:]:  # 跳过'Class'列
                print(f"  {col:12}: {row[col]:.4f}")
    
    # 模型排名
    ranking_df = comparison_tables['ranking']
    print(f"\n5. MODEL RANKING")
    print("-" * 50)
    for i, (_, row) in enumerate(ranking_df.iterrows(), 1):
        print(f"{i}. {row['Model']:12} | Accuracy Rank: {int(row['Accuracy_Rank'])} | Speed Rank: {int(row['Speed_Rank'])} | Combined Score: {row['Combined_Score']:.1f}")
    
    print("\n" + "="*80)


def save_comparison_excel(comparison_tables, detailed_stats, output_path):
    """保存对比结果到Excel文件"""
    print(f"Saving comparison results to: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 总体指标对比
        comparison_tables['overall_comparison'].to_excel(writer, sheet_name='总体指标对比', index=False)
        
        # 速度对比
        comparison_tables['speed_comparison'].to_excel(writer, sheet_name='速度对比', index=False)
        
        # 类别准确率对比
        comparison_tables['class_accuracy_comparison'].to_excel(writer, sheet_name='类别准确率对比', index=False)
        
        # 模型排名
        comparison_tables['ranking'].to_excel(writer, sheet_name='模型排名', index=False)
        
        # 详细统计汇总
        detailed_stats.to_excel(writer, sheet_name='详细统计汇总', index=False)
    
    print(f"Comparison results saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Compare inference results from multiple models')
    parser.add_argument('--files', nargs='+', required=True, help='List of Excel result files to compare')
    parser.add_argument('--output', type=str, default='comparison_results.xlsx', help='Output comparison file')
    parser.add_argument('--summary-only', action='store_true', help='Only show summary, do not generate Excel file')
    
    args = parser.parse_args()
    
    # 验证输入文件
    valid_files = []
    for file_path in args.files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if len(valid_files) < 2:
        print("Error: Need at least 2 valid result files for comparison")
        return
    
    print(f"Comparing {len(valid_files)} result files:")
    for file_path in valid_files:
        print(f"  - {file_path}")
    
    # 加载结果文件
    all_results = []
    for file_path in valid_files:
        result = load_result_file(file_path)
        if result:
            all_results.append(result)
    
    if len(all_results) < 2:
        print("Error: Could not load at least 2 valid result files")
        return
    
    # 提取指标
    all_metrics = []
    for result in all_results:
        metrics = extract_metrics(result)
        if metrics:
            all_metrics.append(metrics)
    
    if len(all_metrics) < 2:
        print("Error: Could not extract metrics from at least 2 files")
        return
    
    # 创建对比表格
    comparison_tables = create_comparison_tables(all_metrics)
    
    # 创建详细统计
    detailed_stats = create_detailed_statistics(all_metrics)
    
    # 生成摘要报告
    generate_summary_report(all_metrics, comparison_tables)
    
    # 保存Excel文件（如果需要）
    if not args.summary_only:
        save_comparison_excel(comparison_tables, detailed_stats, args.output)
        print(f"\nDetailed comparison results saved to: {args.output}")
    else:
        print("\nSummary-only mode: No Excel file generated")


if __name__ == "__main__":
    main() 