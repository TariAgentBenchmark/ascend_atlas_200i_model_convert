import torch
import torch.nn as nn
from argparse import Namespace
from trainer import PIMFuseTrainer

class SimplifiedONNXWrapper(nn.Module):
    """Simplified wrapper that bypasses FFT operations for ONNX compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pairs, S_V, S_P, S_P1):
        # Manually implement the forward pass without the physical model FFT operations
        S_V = S_V.permute(0, 2, 1)
        S_P = S_P.permute(0, 2, 1)
        S_P1 = S_P1.permute(0, 2, 1)
        
        # Get features from vibration and pressure models
        feat_vibration_shared, feat_vibration_distinct, pred_vibration = self.model.vibration_model(S_V)
        feat_pressure_shared, feat_pressure_distinct, pred_pressure = self.model.pressure_model(S_P)
        
        # Apply shared projection
        feat_vibration_shared = self.model.shared_project(feat_vibration_shared)
        feat_pressure_shared = self.model.shared_project(feat_pressure_shared)
        
        # Simplified physical model - just use CNN features without FFT
        # This bypasses the FFT operations that cause ONNX export issues
        input_47 = self.model.physical_model.conv1(S_P1)
        input_51 = torch.max_pool1d(input_47, 4, 4)
        input_53 = self.model.physical_model.conv2(input_51)
        pooled = torch.max_pool1d(input_53, 4, 4)
        global_pooled = torch.adaptive_avg_pool1d(pooled, 1)
        y_30_simplified = global_pooled.view(global_pooled.size(0), -1)
        
        # Ensure y_30_simplified has exactly 30 features - pad with zeros if needed
        # Use torch operations instead of Python conditionals to avoid tracing warnings
        current_size = y_30_simplified.size(1)
        target_size = 30
        
        # If we have fewer than 30 features, pad with zeros
        if current_size < target_size:
            padding_needed = target_size - current_size
            padding = torch.zeros(y_30_simplified.size(0), padding_needed, device=y_30_simplified.device)
            y_30_simplified = torch.cat([y_30_simplified, padding], dim=1)
        
        # If we have more than 30 features, truncate to 30
        y_30_simplified = y_30_simplified[:, :target_size]
        
        pairs = pairs.unsqueeze(1)
        
        # Fusion logic
        h1 = feat_vibration_shared
        h2 = feat_pressure_shared
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_pressure_shared
        
        # Attention mechanism
        attn_input = torch.stack([feat_pressure_distinct, feat_avg_shared, y_30_simplified, feat_vibration_distinct], dim=1)
        qkvs = self.model.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.model.num_classes, dim=-1)
        q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(dim=1)
        ks = torch.stack(k, dim=1)
        q_mean_expanded = q_mean.unsqueeze(1).unsqueeze(2)
        attn_logits = (ks * q_mean_expanded).sum(dim=-1)
        attn_logits = attn_logits / (q.shape[-1] ** 0.5)  # Use constant instead of math.sqrt
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze() == 0, :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.model.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)
        
        return pred_final

def convert_to_onnx(checkpoint_path, output_path):
    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    
    # Convert hyperparameters to Namespace object
    args = Namespace(**hparams)
    
    # Create model with proper arguments
    model = PIMFuseTrainer(
        args=args,
        label_names=["0","1","2","3","4","5","6","7","8"]
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Wrap the model for ONNX export
    onnx_model = SimplifiedONNXWrapper(model.model)
    onnx_model.eval()
    
    # Define input dimensions (the model expects [batch, length, channels] format)
    batch_size = 1
    sequence_length = 5120
    
    # Create dummy inputs with correct shapes
    dummy_pairs = torch.ones((batch_size,), dtype=torch.float32)  # Shape: [batch_size]
    dummy_S_V = torch.randn(batch_size, sequence_length, 3, dtype=torch.float32)  # [batch, length, channels]
    dummy_S_P = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)  # [batch, length, channels]
    dummy_S_P1 = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)  # [batch, length, channels]
    
    dummy_input = (dummy_pairs, dummy_S_V, dummy_S_P, dummy_S_P1)
    
    # Test the model with dummy input first
    with torch.no_grad():
        try:
            test_output = onnx_model(*dummy_input)
            print(f"Simplified model test successful. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Simplified model test failed: {e}")
            return
    
    # Export to ONNX
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
            opset_version=13,  # Use opset 13 to support diagonal operator
            do_constant_folding=True
        )
        print(f"Simplified model successfully exported to {output_path}")
        print("Note: This ONNX model uses a simplified version of the physical model")
        print("that doesn't include FFT operations for ONNX compatibility.")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=68-step=68.ckpt"
    output_path = "model_simplified.onnx"
    convert_to_onnx(checkpoint_path, output_path) 
