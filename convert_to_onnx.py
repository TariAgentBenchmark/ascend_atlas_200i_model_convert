import torch
from argparse import Namespace
from trainer import PIMFuseTrainer

def convert_to_onnx(checkpoint_path, output_path):
    """Convert PyTorch model to ONNX format - no wrapper needed since FFT is removed"""
    print("Converting model to ONNX format...")
    
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
    
    # Use the original model directly - no wrapper needed since FFT is removed
    onnx_model = model.model
    onnx_model.eval()
    
    # Define input dimensions (the model expects [batch, length, channels] format)
    batch_size = 1
    sequence_length = 5120
    
    # Create dummy inputs with correct shapes (now including S_P1_fft)
    dummy_pairs = torch.ones((batch_size,), dtype=torch.float32)  # Shape: [batch_size]
    dummy_S_V = torch.randn(batch_size, sequence_length, 3, dtype=torch.float32)  # [batch, length, channels]
    dummy_S_P = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)  # [batch, length, channels]
    dummy_S_P1 = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)  # [batch, length, channels]
    dummy_S_P1_fft = torch.randn(batch_size, sequence_length, 1, dtype=torch.float32)  # FFT magnitude data
    
    dummy_input = (dummy_pairs, dummy_S_V, dummy_S_P, dummy_S_P1, dummy_S_P1_fft)
    
    # Create a ONNX-compatible wrapper that handles magnitude data directly
    class ONNXCompatibleWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, pairs, S_V, S_P, S_P1, S_P1_fft):
            # ONNX doesn't support complex numbers, so S_P1_fft is already magnitude data
            # We replicate the original model's forward pass but skip the .abs() operation
            
            S_V = S_V.permute(0, 2, 1)
            S_P = S_P.permute(0, 2, 1)
            S_P1 = S_P1.permute(0, 2, 1)
            S_P1_fft_abs = S_P1_fft.permute(0, 2, 1)  # Already magnitude, no .abs() needed
            
            feat_vibration_shared, feat_vibration_distinct, pred_vibration = self.model.vibration_model(S_V)
            feat_pressure_shared, feat_pressure_distinct, pred_pressure = self.model.pressure_model(S_P)
            feat_vibration_shared = self.model.shared_project(feat_vibration_shared)
            feat_pressure_shared = self.model.shared_project(feat_pressure_shared)
            y_30, y_pred_phy = self.model.physical_model(S_P1, S_P1_fft_abs)

            pairs = pairs.unsqueeze(1)

            h1 = feat_vibration_shared
            h2 = feat_pressure_shared
            term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
            term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
            feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
            feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_pressure_shared

            attn_input = torch.stack([feat_pressure_distinct, feat_avg_shared, y_30, feat_vibration_distinct], dim=1)
            qkvs = self.model.attn_proj(attn_input)
            q, v, *k = qkvs.chunk(2 + self.model.num_classes, dim=-1)
            q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(dim=1)
            ks = torch.stack(k, dim=1)
            attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
            attn_logits = attn_logits / (q.shape[-1] ** 0.5)  # Use constant instead of math.sqrt
            attn_mask = torch.ones_like(attn_logits)
            attn_mask[pairs.squeeze() == 0, :, -1] = 0
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_logits, dim=-1)

            feat_final = torch.matmul(attn_weights, v)
            pred_final = self.model.final_pred_fc(feat_final)
            pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)
            
            return pred_final
    
    onnx_wrapper = ONNXCompatibleWrapper(onnx_model)
    
    # Test the wrapper with dummy input first
    with torch.no_grad():
        test_output = onnx_wrapper(*dummy_input)
        print(f"ONNX wrapper test successful. Output shape: {test_output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        output_path,
        input_names=['pairs', 'S_V', 'S_P', 'S_P1', 'S_P1_fft'],
        output_names=['pred_final'],
        dynamic_axes={
            'pairs': {0: 'batch_size'},
            'S_V': {0: 'batch_size', 1: 'sequence_length'},
            'S_P': {0: 'batch_size', 1: 'sequence_length'},
            'S_P1': {0: 'batch_size', 1: 'sequence_length'},
            'S_P1_fft': {0: 'batch_size', 1: 'sequence_length'},
            'pred_final': {0: 'batch_size'}
        },
        opset_version=14,  # Use opset 14 for better complex number support
        do_constant_folding=True
    )
    print(f"Model successfully exported to {output_path}")
    print("Note: This ONNX model uses the original model structure with pre-computed FFT data as input.")

if __name__ == "__main__":
    checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=150-step=150.ckpt"
    output_path = "model.onnx"
    convert_to_onnx(checkpoint_path, output_path) 
