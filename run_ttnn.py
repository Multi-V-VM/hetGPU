#!/usr/bin/env python3
"""
Run TTNN binary with visible inputs and outputs
Usage: python run_ttnn_.py <binary_file.ttnn> [fill_value] [expected_value] [rows] [cols]
  fill_value: Optional float to fill all input elements (default: random)
  expected_value: Optional expected output value for verification
  rows: Optional number of rows (default: 1)
  cols: Optional number of columns (default: 32)
"""

import numpy as np
import ttrt
import ttrt.runtime
import sys
import os

def test_relu_with_io(binary_path, fill_value=None, expected_value=None, rows=1, cols=32):
    """Run TTNN binary and return True if verification passed, False otherwise."""
    # Check if file exists
    if not os.path.exists(binary_path):
        print(f"Error: Binary file '{binary_path}' not found")
        sys.exit(1)
    
    print(f"Loading binary: {binary_path}")
    
    try:
        # Load the binary
        binary = ttrt.binary.load_binary_from_path(binary_path)
        
        # Get the binary info
        print("\n=== Binary Information ===")
        print(f"Binary loaded successfully")
        
        # Create test input
        input_shape = (rows, cols)
        print(f"\n=== Creating Input ===")
        print(f"Input shape: {input_shape}")
        
        # Create input data based on fill_value parameter
        if fill_value is not None:
            # Fill entire array with the specified value
            input_data = np.full(input_shape, fill_value, dtype=np.float32)
            print(f"Filled all elements with: {fill_value}")
        else:
            # Create input with both positive and negative values for testing
            input_data = np.random.randn(*input_shape).astype(np.float32)
            # Make some values clearly negative for testing
            input_data[0:10, 0:10] = -5.0  # Top-left corner negative
            input_data[20:30, 20:30] = 10.0  # Some positive values
            print("Using random values with some fixed regions")
        
        print(f"Input sample (first 5x5):")
        print(input_data[:5, :5])
        print(f"Input min: {input_data.min():.4f}, max: {input_data.max():.4f}")
        
        # Get number of available devices
        num_devices = ttrt.runtime.get_num_available_devices()
        print(f"\n=== Device Information ===")
        print(f"Number of available devices: {num_devices}")
        
        if num_devices == 0:
            print("No devices available!")
            return
        
        # Open device
        mesh_shape = (1, 1)
        print(f"Opening mesh device with shape: {mesh_shape}")
        
        # Create mesh device options
        options = ttrt.runtime.MeshDeviceOptions()
        options.device_ids = [0]
        
        mesh_device = ttrt.runtime.open_mesh_device(
            mesh_shape,
            options
        )
        
        # Set runtime
        ttrt.runtime.set_compatible_runtime(binary)
        
        # Create input tensor
        print("\n=== Running Inference ===")
        input_tensor = ttrt.runtime.create_borrowed_host_tensor(
            input_data.ctypes.data,  # data pointer
            list(input_data.shape),  # shape
            [s // input_data.itemsize for s in input_data.strides],  # strides in elements
            input_data.itemsize,  # element size in bytes
            ttrt.runtime.DataType.Float32
        )
        
        # Get input layout from binary
        input_layout = ttrt.runtime.get_layout(binary, 0, 0)  # program 0, input 0
        
        # Convert tensor to proper layout and move to device
        device_tensor = ttrt.runtime.to_layout(
            input_tensor, 
            mesh_device,
            input_layout,
            True  # blocking
        )
        
        # Submit for execution
        output_tensors = ttrt.runtime.submit(
            mesh_device,  # device
            binary,       # executable
            0,            # program_index
            [device_tensor]  # inputs
        )
        
        # Get output - to_host returns a tuple, we want the first element
        output_data = ttrt.runtime.to_host(output_tensors[0], untilize=True)[0]
        
        # Debug: Check what type of data we got
        print(f"Output type: {type(output_data)}")
        if isinstance(output_data, list):
            print(f"Output list length: {len(output_data)}")
            if len(output_data) > 0:
                print(f"First element type: {type(output_data[0])}")
        
        # Convert ttrt Tensor to numpy array
        if hasattr(output_data, 'get_data_buffer'):
            # It's a ttrt Tensor object
            output_shape = output_data.get_shape()
            output_buffer = output_data.get_data_buffer()
            output_data = np.frombuffer(output_buffer, dtype=np.float32).reshape(output_shape)
        elif hasattr(output_data, 'numpy'):
            # If it's a torch tensor or similar
            output_data = output_data.numpy()
        elif isinstance(output_data, list):
            output_data = np.array(output_data).reshape(input_shape)
        
        print("\n=== Output Information ===")
        print(f"Output shape: {output_data.shape}")
        print(f"Output sample (first 5x5):")
        print(output_data[:5, :5])
        print(f"Output min: {output_data.min():.4f}, max: {output_data.max():.4f}")
        
        # Verification
        print("\n=== Verification ===")
        
        if expected_value is not None:
            # Use provided expected value
            print(f"Checking if all outputs equal {expected_value}")
            expected_output = np.full_like(output_data, expected_value, dtype=np.float32)
            tolerance = abs(expected_value) * 1e-5 if expected_value != 0 else 1e-5
            is_correct = np.allclose(output_data, expected_output, atol=tolerance, rtol=1e-5)
            
            if is_correct:
                print(f"✓ All outputs match expected value {expected_value}")
            else:
                # Check how many values match
                matches = np.isclose(output_data, expected_output, atol=tolerance, rtol=1e-5)
                match_count = matches.sum()
                total_count = output_data.size
                print(f"✗ Output mismatch: {match_count}/{total_count} values match expected {expected_value}")
                
                # Show some mismatches
                if not matches.all():
                    mismatch_indices = np.where(~matches)
                    num_examples = min(5, len(mismatch_indices[0]))
                    print(f"\nFirst {num_examples} mismatches:")
                    for i in range(num_examples):
                        idx = (mismatch_indices[0][i], mismatch_indices[1][i])
                        print(f"  [{idx[0]},{idx[1]}]: expected {expected_value}, got {output_data[idx]:.6f}")
        else:
            # Default ReLU verification
            print("Performing ReLU verification (no expected value provided)")
            expected_output = np.maximum(input_data, 0)
            
            # Check specific regions
            print(f"Input mean: {input_data.mean():.4f}")
            print(f"Output mean: {output_data.mean():.4f}")
            
            # Check if ReLU worked correctly
            is_correct = np.allclose(output_data, expected_output, atol=1e-5)
            
            if is_correct:
                print("✓ ReLU operation verified successfully!")
                print(f"  - All negative values were set to 0")
                print(f"  - All positive values were preserved")
                
                # Count negative inputs that became 0
                negative_count = (input_data < 0).sum()
                zeros_in_output = (output_data == 0).sum()
                print(f"  - {negative_count} negative values in input")
                print(f"  - {zeros_in_output} zero values in output")
            else:
                print("✗ ReLU operation failed verification")
                max_diff = np.max(np.abs(output_data - expected_output))
                print(f"  - Maximum difference: {max_diff}")
                
                # Find where differences occur
                diff_mask = ~np.isclose(output_data, expected_output, atol=1e-5)
                if diff_mask.any():
                    diff_indices = np.where(diff_mask)
                    print(f"  - First difference at index: {diff_indices[0][0]}, {diff_indices[1][0]}")
                    print(f"    Input: {input_data[diff_indices[0][0], diff_indices[1][0]]}")
                    print(f"    Expected: {expected_output[diff_indices[0][0], diff_indices[1][0]]}")
                    print(f"    Got: {output_data[diff_indices[0][0], diff_indices[1][0]]}")
        
        # Performance info
        print("\n=== Performance ===")
        # Note: Actual performance metrics would come from ttrt profiling
        print("Check run_results.json for detailed performance metrics")
        
        # Cleanup
        ttrt.runtime.close_mesh_device(mesh_device)
        
        # Return verification result
        return is_correct
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print("Usage: python run_ttnn_.py <binary_file.ttnn> [fill_value] [expected_value] [rows] [cols]")
        print("Examples:")
        print("  python run_ttnn_.py relu.ttnn                      # Random 1x32, ReLU verification")
        print("  python run_ttnn_.py relu.ttnn -5.0                 # 1x32 filled with -5.0")
        print("  python run_ttnn_.py relu.ttnn -5.0 0.0             # 1x32, expect all 0.0")
        print("  python run_ttnn_.py relu.ttnn -5.0 0.0 64 128      # 64x128, expect all 0.0")
        print("  python run_ttnn_.py ceil.ttnn 3.14 4.0 32 32       # 32x32, expect all 4.0")
        sys.exit(1)
    
    binary_path = sys.argv[1]
    fill_value = None
    expected_value = None
    rows = 1
    cols = 32
    
    if len(sys.argv) >= 3:
        try:
            fill_value = float(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid fill value '{sys.argv[2]}'. Must be a float.")
            sys.exit(1)
    
    if len(sys.argv) >= 4:
        try:
            expected_value = float(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid expected value '{sys.argv[3]}'. Must be a float.")
            sys.exit(1)
    
    if len(sys.argv) >= 5:
        try:
            rows = int(sys.argv[4])
            if rows <= 0:
                raise ValueError("Rows must be positive")
        except ValueError as e:
            print(f"Error: Invalid rows value '{sys.argv[4]}'. Must be a positive integer.")
            sys.exit(1)
    
    if len(sys.argv) == 6:
        try:
            cols = int(sys.argv[5])
            if cols <= 0:
                raise ValueError("Columns must be positive")
        except ValueError as e:
            print(f"Error: Invalid cols value '{sys.argv[5]}'. Must be a positive integer.")
            sys.exit(1)
    
    success = test_relu_with_io(binary_path, fill_value, expected_value, rows, cols)
    sys.exit(0 if success else 1)