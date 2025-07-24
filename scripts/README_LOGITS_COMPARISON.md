# Logits Comparison Tools

本文档介绍如何使用修改后的推理脚本来输出和对比三种模型格式的logits精度。

## 功能特性

### 1. 增强的推理脚本
- **pytorch_inference.py**: PyTorch模型推理，现在输出原始logits
- **onnx_inference.py**: ONNX模型推理，现在输出原始logits  
- **acl_inference.py**: ACL模型推理，现在输出原始logits

### 2. 新增的对比工具
- **compare_logits.py**: 对比不同模型格式的logits精度
- **run_all_inference_and_compare.py**: 一键运行所有推理并自动对比

## 输出内容

每个推理脚本现在会输出：
1. **分类结果**: 预测类别和准确率
2. **置信度**: Softmax后的概率分布
3. **原始Logits**: 模型的原始输出（未经过softmax）
4. **详细Excel报告**: 包含以上所有信息的Excel文件

Excel文件包含以下表格：
- `总体统计`: 整体性能指标
- `详细结果`: 每个样本的预测结果、置信度和logits
- `类别统计`: 每个类别的准确率、精确度、召回率等
- `置信度统计`: 每个类别的置信度统计
- `原始Logits`: 专门的logits数据表

## 使用方法

### 方式1: 单独运行推理脚本

```bash
# PyTorch推理
python scripts/pytorch_inference.py \
    --model-path path/to/model.ckpt \
    --data-path ./data \
    --output pytorch_results.xlsx \
    --batch-size 8 \
    --max-batches 10

# ONNX推理
python scripts/onnx_inference.py \
    --model-path path/to/model.onnx \
    --data-path ./data \
    --output onnx_results.xlsx \
    --batch-size 8 \
    --max-batches 10

# ACL推理
python scripts/acl_inference.py \
    --model-path path/to/model.om \
    --data-path ./data \
    --output acl_results.xlsx \
    --batch-size 8 \
    --max-batches 10
```

### 方式2: 对比logits

```bash
# 对比两个或三个模型的logits
python scripts/compare_logits.py \
    --pytorch-results pytorch_results.xlsx \
    --onnx-results onnx_results.xlsx \
    --acl-results acl_results.xlsx \
    --output logits_comparison.xlsx \
    --plots-dir comparison_plots
```

### 方式3: 一键运行所有推理和对比

```bash
# 运行所有可用模型的推理并自动对比
python scripts/run_all_inference_and_compare.py \
    --pytorch-model path/to/model.ckpt \
    --onnx-model path/to/model.onnx \
    --acl-model path/to/model.om \
    --data-path ./data \
    --output-dir inference_results \
    --batch-size 8 \
    --max-batches 10
```

## Logits对比指标

对比脚本会计算以下精度指标：

### 整体相似性指标
- **均方误差 (MSE)**: 衡量logits数值差异
- **平均绝对误差 (MAE)**: 平均绝对差异
- **皮尔逊相关系数**: 线性相关性 (-1到1)
- **斯皮尔曼相关系数**: 单调相关性
- **余弦相似度**: 向量角度相似性 (0到1)

### 样本级别分析
- **每个样本的余弦相似度**: 单个样本的logits相似性
- **相似度统计**: 均值、标准差、最小值、最大值

### 类别级别分析
- **每个类别的相似性指标**: 分别计算每个类别的MSE、MAE、相关性等
- **类别logits分布**: 可视化每个类别的logits分布差异

### 预测一致性分析
- **完全一致比例**: 三个模型预测完全相同的样本比例
- **两两一致性**: 任意两个模型的预测一致性
- **不一致样本详情**: 列出所有预测不一致的样本

## 输出文件

### 推理结果文件
每个推理脚本输出一个Excel文件，包含：
- 分类结果和准确率
- 每个样本的详细信息
- 原始logits数据

### 对比结果文件
`logits_comparison.xlsx` 包含：
- **模型基本信息**: 各模型的样本数、类别数、准确率
- **整体相似性**: 模型间的各种相似性指标
- **预测一致性**: 预测结果的一致性分析
- **不一致样本**: 预测不一致样本的详细信息
- **类别相似性**: 每个类别的详细相似性分析

### 可视化图表
`comparison_plots/` 目录包含：
- **相关性热力图**: 模型间logits相关性的热力图
- **类别分布图**: 每个类别logits分布的对比图

## 示例命令

### 快速开始（推荐）
```bash
# 一键运行所有推理和对比
python scripts/run_all_inference_and_compare.py \
    --pytorch-model lightning_logs/version_0/checkpoints/epoch=68-step=68.ckpt \
    --onnx-model model.onnx \
    --acl-model model.om \
    --data-path ./data \
    --output-dir results
```

### 只对比已有结果
```bash
# 如果已经有推理结果，只运行对比
python scripts/compare_logits.py \
    --pytorch-results pytorch_results.xlsx \
    --onnx-results onnx_results.xlsx \
    --output logits_comparison.xlsx
```

## 注意事项

1. **数据一致性**: 确保所有模型使用相同的测试数据
2. **批次大小**: 建议使用相同的批次大小以保证结果可比性
3. **设备一致性**: PyTorch推理可指定CPU或CUDA设备
4. **文件路径**: 使用绝对路径或确保相对路径正确
5. **依赖项**: 确保安装了所需的Python包（pandas, numpy, sklearn, scipy, matplotlib, seaborn）

## 结果解读

### 高精度一致性指标
- **皮尔逊相关系数 > 0.99**: 表示logits高度线性相关
- **余弦相似度 > 0.999**: 表示logits向量方向极其相似
- **RMSE < 0.01**: 表示logits数值差异很小

### 预测一致性
- **完全一致比例 > 95%**: 表示模型预测高度一致
- **两两一致性 > 98%**: 表示任意两个模型预测基本一致

这些工具可以帮助你全面评估不同模型格式转换后的精度保持情况。 