#!/usr/bin/env python3
# run_acl_infer.py
import numpy as np
import acl
import argparse
import os
import time
from data_processing_3 import dataProcessing_3, scalar_stand

class AscendInfer:
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.model_id = None
        self.input_dataset = None
        self.output_dataset = None
        self._init_resource(model_path)

    def _init_resource(self, model_path):
        # Initialize ACL
        ret = acl.init()
        assert ret == 0, f"ACL init failed: {ret}"
        
        # Set device and create context
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"Set device failed: {ret}"
        
        self.context = acl.rt.create_context(0)
        self.stream = acl.rt.create_stream()
        
        # Load OM model
        model_path = os.path.realpath(model_path)
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"Model load failed: {ret}"
        
        # Prepare model I/O
        self._prepare_model_io()

    def _prepare_model_io(self):
        # Create input dataset
        self.input_dataset = acl.mdl.create_dataset()
        num_inputs = acl.mdl.get_num_inputs(self.model_id)
        
        for i in range(num_inputs):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_id, i)
            buffer, ret = acl.rt.malloc(buffer_size, 0)
            assert ret == 0, f"Input malloc failed: {ret}"
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)

        # Create output dataset
        self.output_dataset = acl.mdl.create_dataset()
        num_outputs = acl.mdl.get_num_outputs(self.model_id)
        
        for i in range(num_outputs):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_id, i)
            buffer, ret = acl.rt.malloc(buffer_size, 0)
            assert ret == 0, f"Output malloc failed: {ret}"
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)

    def run(self, input_data):
        # Copy input data to device
        for i, data in enumerate(input_data):
            buffer = acl.mdl.get_dataset_buffer(self.input_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buffer)
            data_size = acl.get_data_buffer_size(buffer)
            
            # Convert numpy data to bytes
            bytes_data = data.tobytes()
            assert len(bytes_data) <= data_size, "Input data too large"
            
            # Copy to device
            ret = acl.rt.memcpy(
                data_ptr, data_size, 
                bytes_data, len(bytes_data), 
                2  # ACL_MEMCPY_HOST_TO_DEVICE
            )
            assert ret == 0, f"Input copy failed: {ret}"

        # Execute inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        assert ret == 0, f"Inference failed: {ret}"

        # Process outputs
        outputs = []
        for i in range(acl.mdl.get_dataset_num_buffers(self.output_dataset)):
            buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buffer)
            data_size = acl.get_data_buffer_size(buffer)
            
            # Allocate host memory
            host_buffer, ret = acl.rt.malloc_host(data_size)
            assert ret == 0, f"Host malloc failed: {ret}"
            
            # Copy from device to host
            ret = acl.rt.memcpy(
                host_buffer, data_size,
                data_ptr, data_size,
                4  # ACL_MEMCPY_DEVICE_TO_HOST
            )
            assert ret == 0, f"Output copy failed: {ret}"
            
            # Convert to numpy array (assuming float32 output)
            np_out = np.frombuffer(host_buffer, dtype=np.float32)
            outputs.append(np_out.copy())
            
            # Free host buffer
            acl.rt.free_host(host_buffer)
        
        return outputs

    def __del__(self):
        # Release resources in reverse order
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.input_dataset:
            acl.mdl.destroy_dataset(self.input_dataset)
        if self.output_dataset:
            acl.mdl.destroy_dataset(self.output_dataset)
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

def generate_random_data():
    """Generate random input data for inference"""
    # Create one sample of random data
    num_samples = 1
    seq_length = 5120
    
    # Generate random vibration data (3 channels)
    S_V = np.random.randn(num_samples, 3, seq_length).astype(np.float32)
    # Generate random pressure data (1 channel)
    S_P = np.random.randn(num_samples, 1, seq_length).astype(np.float32)
    S_P1 = S_P.copy()  # Physical model input same as pressure
    # Pairs are zeros (not used in this model)
    pairs = np.zeros((num_samples, 2), dtype=np.int64)
    
    # Add batch dimension
    S_V = np.expand_dims(S_V, axis=0)
    S_P = np.expand_dims(S_P, axis=0)
    S_P1 = np.expand_dims(S_P1, axis=0)
    pairs = np.expand_dims(pairs, axis=0)
    
    return pairs, S_V, S_P, S_P1, None

def prepare_data(input_dir):
    """Process and prepare test data"""
    if input_dir is None:
        return generate_random_data()
    
    # Get test data
    _, Test_X, _, Test_Y = dataProcessing_3(input_dir)
    
    # Standardize data
    Test_X = Test_X.astype(np.float32)
    Test_X = scalar_stand(Test_X, Test_X)[1]  # Use test stats for standardization
    
    # Split into model inputs
    S_V = Test_X[:, 0:3, :]  # Vibration data (3 channels)
    S_P = Test_X[:, 3:4, :]  # Pressure data (1 channel)
    S_P1 = S_P.copy()        # Physical model input
    pairs = np.zeros((Test_X.shape[0], 2), dtype=np.int64)
    
    # Add batch dimension
    S_V = np.expand_dims(S_V, axis=0)
    S_P = np.expand_dims(S_P, axis=0)
    S_P1 = np.expand_dims(S_P1, axis=0)
    pairs = np.expand_dims(pairs, axis=0)
    
    return pairs, S_V, S_P, S_P1, Test_Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on Ascend AI processor')
    parser.add_argument('model_path', type=str, help='Path to the OM model file')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing input data (optional)')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID (default: 0)')
    args = parser.parse_args()
    
    # Initialize inference engine
    ascend_net = AscendInfer(args.model_path, args.device_id)
    
    # Prepare data
    pairs, S_V, S_P, S_P1, labels = prepare_data(args.data_dir)
    input_data = [pairs, S_V, S_P, S_P1]
    
    # Run inference
    start_time = time.time()
    outputs = ascend_net.run(input_data)
    inference_time = time.time() - start_time
    
    # Process outputs
    logits = outputs[0]  # First output is final predictions
    predictions = np.argmax(logits, axis=-1)
    
    # Print results
    print(f"Inference time: {inference_time:.4f}s")
    print(f"Predictions: {predictions}")
    
    if labels is not None:
        print(f"True labels: {labels}")
        print(f"Accuracy: {np.mean(predictions == labels):.4f}")
    else:
        print("Using randomly generated input data - no true labels available")
