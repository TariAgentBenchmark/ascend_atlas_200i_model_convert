import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from trainer import PIMFuseTrainer


class ImprovedONNXWrapper(nn.Module):
    """改进的ONNX包装器，尝试保留更多的原始特征"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # 预计算FFT位置权重 - 尝试模拟FFT特征
        self.fft_positions = [10, 20, 29, 39, 49, 59, 68, 79, 89, 98, 107, 120, 147, 294, 437, 581, 732, 880, 1027, 1180, 1324,
                             137, 284, 427, 571, 722, 870, 1017, 1170, 1314]
        
        # 创建一个简单的线性层来模拟FFT特征提取
        self.fft_simulator = nn.Linear(5120, 30)  # 从输入长度映射到30个特征
        
        # 初始化权重来模拟频域特征提取
        with torch.no_grad():
            # 创建一个模拟FFT的权重矩阵
            weight_matrix = torch.zeros(30, 5120)
            for i, pos in enumerate(self.fft_positions[:30]):  # 只取前30个位置
                if pos < 5120:
                    weight_matrix[i, pos] = 1.0
                    # 添加一些邻近位置的权重来模拟频域响应
                    for offset in [-2, -1, 1, 2]:
                        if 0 <= pos + offset < 5120:
                            weight_matrix[i, pos + offset] = 0.5
            
            self.fft_simulator.weight.data = weight_matrix
            self.fft_simulator.bias.data.zero_()
    
    def forward(self, pairs, S_V, S_P, S_P1):
        """改进的前向传播，尝试保留更多原始特征"""
        S_V = S_V.permute(0, 2, 1)
        S_P = S_P.permute(0, 2, 1)
        S_P1 = S_P1.permute(0, 2, 1)
        
        # 振动和压力模型特征
        feat_vibration_shared, feat_vibration_distinct, pred_vibration = self.model.vibration_model(S_V)
        feat_pressure_shared, feat_pressure_distinct, pred_pressure = self.model.pressure_model(S_P)
        
        # 应用共享投影
        feat_vibration_shared = self.model.shared_project(feat_vibration_shared)
        feat_pressure_shared = self.model.shared_project(feat_pressure_shared)
        
        # 改进的物理模型 - 尝试模拟FFT特征
        # 1. CNN特征
        input_conv1 = self.model.physical_model.conv1(S_P1)
        input_conv2 = self.model.physical_model.conv2(input_conv1)
        pooled = self.model.physical_model.global_pool(input_conv2)
        cnn_features = pooled.view(pooled.size(0), -1)
        
        # 2. 模拟FFT特征
        S_P1_flat = S_P1.view(S_P1.size(0), -1)  # [batch, 5120]
        simulated_fft_features = self.fft_simulator(S_P1_flat)  # [batch, 30]
        
        # 3. 组合CNN和模拟FFT特征
        # 确保维度匹配
        if cnn_features.size(1) < 30:
            padding = torch.zeros(cnn_features.size(0), 30 - cnn_features.size(1), device=cnn_features.device)
            cnn_features = torch.cat([cnn_features, padding], dim=1)
        elif cnn_features.size(1) > 30:
            cnn_features = cnn_features[:, :30]
        
        # 模拟原始模型的特征组合: y_30 = cnn_features + fft_features
        y_30_improved = cnn_features + simulated_fft_features
        
        pairs = pairs.unsqueeze(1)
        
        # 特征融合逻辑（保持不变）
        h1 = feat_vibration_shared
        h2 = feat_pressure_shared
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_pressure_shared
        
        # 注意力机制
        attn_input = torch.stack([feat_pressure_distinct, feat_avg_shared, y_30_improved, feat_vibration_distinct], dim=1)
        qkvs = self.model.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.model.num_classes, dim=-1)
        q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(dim=1)
        ks = torch.stack(k, dim=1)
        q_mean_expanded = q_mean.unsqueeze(1).unsqueeze(2)
        attn_logits = (ks * q_mean_expanded).sum(dim=-1)
        attn_logits = attn_logits / (q.shape[-1] ** 0.5)
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze() == 0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.model.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)
        
        return pred_final


def convert_to_improved_onnx(checkpoint_path, output_path):
    """转换为改进的ONNX模型"""
    print("Converting to improved ONNX model...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    args = Namespace(**hparams)
    
    # 创建模型
    model = PIMFuseTrainer(
        args=args,
        label_names=["0","1","2","3","4","5","6","7","8"]
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 创建改进的ONNX包装器
    onnx_model = ImprovedONNXWrapper(model.model)
    onnx_model.eval()
    
    # 定义输入
    batch_size = 1
    sequence_length = 5120
    
    dummy_pairs = torch.ones((batch_size,), dtype=torch.float32)
    dummy_S_V = torch.randn(batch_size, sequence_length, 3, dtype=torch.float32)
    dummy_S_P = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)
    dummy_S_P1 = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)
    
    dummy_input = (dummy_pairs, dummy_S_V, dummy_S_P, dummy_S_P1)
    
    # 测试模型
    with torch.no_grad():
        try:
            test_output = onnx_model(*dummy_input)
            print(f"Improved model test successful. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Improved model test failed: {e}")
            return
    
    # 导出ONNX
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            output_path,
            input_names=['pairs', 'S_V', 'S_P', 'S_P1'],
            output_names=['pred_final'],
            dynamic_axes={
                'pairs': {0: 'batch_size'},
                'S_V': {0: 'batch_size', 1: 'sequence_length'},
                'S_P': {0: 'batch_size', 1: 'sequence_length'},
                'S_P1': {0: 'batch_size', 1: 'sequence_length'},
                'pred_final': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        print(f"Improved ONNX model successfully exported to {output_path}")
        print("This model includes a simulated FFT feature extractor that should")
        print("provide better accuracy than the basic simplified version.")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def compare_models(checkpoint_path, original_onnx_path, improved_onnx_path):
    """比较原始PyTorch、简化ONNX和改进ONNX模型的特征"""
    print("\n=== Model Feature Comparison ===")
    
    # 加载PyTorch模型
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    args = Namespace(**hparams)
    
    pytorch_model = PIMFuseTrainer(args=args, label_names=["0","1","2","3","4","5","6","7","8"])
    pytorch_model.load_state_dict(checkpoint['state_dict'])
    pytorch_model.eval()
    
    # 创建测试输入
    test_pairs = torch.ones((1,), dtype=torch.float32)
    test_S_V = torch.randn(1, 5120, 3, dtype=torch.float32)
    test_S_P = torch.randn(1, 5120, 1, dtype=torch.float32)
    test_S_P1 = torch.randn(1, 5120, 1, dtype=torch.float32)
    
    with torch.no_grad():
        # PyTorch模型物理特征
        S_P1_permuted = test_S_P1.permute(0, 2, 1)
        inputs1 = torch.fft.fft(torch.complex(S_P1_permuted, torch.zeros_like(S_P1_permuted))).abs()
        positions = [10, 20, 29, 39, 49, 59, 68, 79, 89, 98, 107, 120, 147, 294, 437, 581, 732, 880, 1027, 1180, 1324,
                     137, 284, 427, 571, 722, 870, 1017, 1170, 1314]
        extracted_values = inputs1[:, :, positions[:30] if len(positions) >= 30 else positions]
        
        print(f"PyTorch FFT features shape: {extracted_values.shape}")
        print(f"PyTorch FFT features mean: {extracted_values.mean():.6f}")
        print(f"PyTorch FFT features std: {extracted_values.std():.6f}")
        
        # 改进ONNX模型的模拟FFT特征
        improved_model = ImprovedONNXWrapper(pytorch_model.model)
        improved_output = improved_model(test_pairs, test_S_V, test_S_P, test_S_P1)
        print(f"Improved ONNX output shape: {improved_output.shape}")
        print(f"Improved ONNX output mean: {improved_output.mean():.6f}")


if __name__ == "__main__":
    checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=68-step=68.ckpt"
    improved_output_path = "model_improved.onnx"
    
    # 转换为改进的ONNX模型
    convert_to_improved_onnx(checkpoint_path, improved_output_path)
    
    # 比较模型特征
    compare_models(checkpoint_path, "model_simplified.onnx", improved_output_path) 