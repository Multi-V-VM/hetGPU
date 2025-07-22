#!/usr/bin/env python3
"""
Run TTNN binary with matrix inputs and outputs
Usage: python run_ttnn_matrix.py <binary_file.ttnn> <input_matrix1> [input_matrix2 ...] <expected_output_matrix>
  input_matrix: Matrix as string, e.g., "[[1,2,3,4]]" or "[1,2,3,4]" for 1D
  expected_output_matrix: Expected output matrix as string (always the last argument)
  
Data types:
  - Default: int32 for integers (42), float32 for decimals (42.0)
  - Rust-style literals: 42i32, 42u32, 42i64, 42u64, 42f32, 42f64
  - Special float values: NaNf32, Inff32, -Inff32 (or NaN, Inf, -Inf for default f32)
  - Supported types: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64
  
Examples:
  python run_ttnn_matrix.py vector4.ttnn "[1u32,2u32,3u32,4u32]" "[4u32]"
  python run_ttnn_matrix.py sub.ttnn "[2u64]" "[1u64]"
  python run_ttnn_matrix.py add.ttnn "[1,2]" "[3,4]" "[4,6]"  # Multiple inputs
  python run_ttnn_matrix.py test.ttnn "[1.5f32,2.5f32]" "[4.0f32]"
"""

import numpy as np
import ttrt
import ttrt.runtime
import sys
import os
import re

# Map of type suffixes to numpy dtypes
TYPE_MAP = {
    'i8': np.int8,
    'i16': np.int16,
    'i32': np.int32,
    'i64': np.int64,
    'u8': np.uint8,
    'u16': np.uint16,
    'u32': np.uint32,
    'u64': np.uint64,
    'f32': np.float32,
    'f64': np.float64,
}

# Map numpy dtypes to TTNN DataTypes
TTNN_DTYPE_MAP = {
    np.dtype(np.int8): ttrt.runtime.DataType.Int8,
    np.dtype(np.int16): ttrt.runtime.DataType.Int16,
    np.dtype(np.int32): ttrt.runtime.DataType.Int32,
    np.dtype(np.int64): ttrt.runtime.DataType.Int64,
    np.dtype(np.uint8): ttrt.runtime.DataType.UInt8,
    np.dtype(np.uint16): ttrt.runtime.DataType.UInt16,
    np.dtype(np.uint32): ttrt.runtime.DataType.UInt32,
    np.dtype(np.uint64): ttrt.runtime.DataType.UInt64,
    np.dtype(np.float32): ttrt.runtime.DataType.Float32,
    np.dtype(np.float64): ttrt.runtime.DataType.Float64,
}

def parse_rust_literal(literal_str):
    """Parse a Rust-style literal like '42i32' or '3.14f32' or special values like 'NaNf32'."""
    literal_str = literal_str.strip()
    
    # Check for special float values with type suffix
    special_match = re.match(r'^(NaN|nan|Inf|inf|-Inf|-inf)([f]\d+)?$', literal_str)
    if special_match:
        special_value, type_suffix = special_match.groups()
        
        # Default to f32 if no suffix
        if not type_suffix:
            type_suffix = 'f32'
            
        if type_suffix not in TYPE_MAP:
            raise ValueError(f"Unknown type suffix: {type_suffix}")
        dtype = TYPE_MAP[type_suffix]
        
        # Parse special value
        special_value_lower = special_value.lower()
        if special_value_lower == 'nan':
            value = float('nan')
        elif special_value_lower == 'inf':
            value = float('inf')
        elif special_value_lower == '-inf':
            value = float('-inf')
        else:
            raise ValueError(f"Unknown special value: {special_value}")
            
        return dtype(value), dtype
    
    # Check for regular Rust-style type suffix
    match = re.match(r'^(-?[\d.]+)([iuf]\d+)?$', literal_str)
    if not match:
        raise ValueError(f"Invalid literal format: {literal_str}")
    
    value_str, type_suffix = match.groups()
    
    if type_suffix:
        # Has explicit type suffix
        if type_suffix not in TYPE_MAP:
            raise ValueError(f"Unknown type suffix: {type_suffix}")
        dtype = TYPE_MAP[type_suffix]
        
        # Parse value based on type
        if type_suffix.startswith('f'):
            value = float(value_str)
        else:
            value = int(float(value_str))  # Allow "42.0i32"
            
        return dtype(value), dtype
    else:
        # No type suffix - use default rules
        if '.' in value_str:
            # Has decimal point -> float32
            return np.float32(float(value_str)), np.float32
        else:
            # Integer -> int32
            return np.int32(int(value_str)), np.int32

def parse_matrix_string(matrix_str):
    """Parse a matrix string into a numpy array with proper dtype.
    Returns (array, dtype)."""
    matrix_str = matrix_str.strip()
    
    # First, try to parse as a single value
    if not matrix_str.startswith('['):
        value, dtype = parse_rust_literal(matrix_str)
        return np.array([[value]], dtype=dtype), dtype
    
    # Remove outer brackets and split by commas
    # Handle nested arrays
    if matrix_str.startswith('[['):
        # 2D array
        # This is a simplified parser - for production, use proper JSON parsing
        rows = []
        dtype = None
        
        # Remove outer brackets
        inner = matrix_str[1:-1]
        
        # Split by '],['
        row_strs = inner.split('],[')
        for row_str in row_strs:
            row_str = row_str.strip('[]')
            row = []
            for val_str in row_str.split(','):
                value, val_dtype = parse_rust_literal(val_str.strip())
                row.append(value)
                if dtype is None:
                    dtype = val_dtype
                elif dtype != val_dtype:
                    print(f"Warning: Mixed types in array, using {dtype}")
            rows.append(row)
        
        return np.array(rows, dtype=dtype), dtype
    else:
        # 1D array
        inner = matrix_str[1:-1]
        values = []
        dtype = None
        
        for val_str in inner.split(','):
            value, val_dtype = parse_rust_literal(val_str.strip())
            values.append(value)
            if dtype is None:
                dtype = val_dtype
            elif dtype != val_dtype:
                print(f"Warning: Mixed types in array, using {dtype}")
        
        # Return as row vector (1xN)
        return np.array([values], dtype=dtype), dtype

def run_ttnn_with_matrix_io(binary_path, input_matrices, expected_output_matrix):
    """Run TTNN binary with matrix inputs and verify against expected outputs.
    
    Args:
        binary_path: Path to the TTNN binary file
        input_matrices: List of input matrix strings
        expected_output_matrix: Expected output matrix string
    """
    # Check if file exists
    if not os.path.exists(binary_path):
        print(f"Error: Binary file '{binary_path}' not found")
        sys.exit(1)
    
    print(f"Loading binary: {binary_path}")
    
    try:
        # Load the binary
        binary = ttrt.binary.load_binary_from_path(binary_path)
        
        print("\n=== Binary Information ===")
        print(f"Binary loaded successfully")
        
        # Parse inputs and expected output
        input_tensors = []
        input_data_list = []
        for i, input_matrix in enumerate(input_matrices):
            input_data, input_dtype = parse_matrix_string(input_matrix)
            input_data_list.append((input_data, input_dtype))
            print(f"\n=== Input {i} ===")
            print(f"Input shape: {input_data.shape}")
            print(f"Input data: {input_data}")
            print(f"Input dtype: {input_dtype}")
        
        expected_output, output_dtype = parse_matrix_string(expected_output_matrix)
        
        print(f"\n=== Expected Output ===")
        print(f"Expected shape: {expected_output.shape}")
        print(f"Expected data: {expected_output}")
        print(f"Expected dtype: {output_dtype}")
        
        # Get number of available devices
        num_devices = ttrt.runtime.get_num_available_devices()
        print(f"\n=== Device Information ===")
        print(f"Number of available devices: {num_devices}")
        
        if num_devices == 0:
            print("No devices available!")
            return False
        
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
        
        # Create input tensors
        print("\n=== Running Inference ===")
        
        device_tensors = []
        for i, (input_data, input_dtype) in enumerate(input_data_list):
            # Get TTNN data type
            ttnn_dtype = TTNN_DTYPE_MAP.get(input_data.dtype, ttrt.runtime.DataType.Float32)
            
            input_tensor = ttrt.runtime.create_borrowed_host_tensor(
                input_data.ctypes.data,  # data pointer
                list(input_data.shape),  # shape
                [s // input_data.itemsize for s in input_data.strides],  # strides in elements
                input_data.itemsize,  # element size in bytes
                ttnn_dtype
            )
            
            # Get input layout from binary
            input_layout = ttrt.runtime.get_layout(binary, 0, i)  # program 0, input i
            
            # Convert tensor to proper layout and move to device
            device_tensor = ttrt.runtime.to_layout(
                input_tensor, 
                mesh_device,
                input_layout,
                True  # blocking
            )
            device_tensors.append(device_tensor)
        
        # Submit for execution
        output_tensors = ttrt.runtime.submit(
            mesh_device,  # device
            binary,       # executable
            0,            # program_index
            device_tensors  # inputs
        )
        
        # Get output - to_host returns a tuple, we want the first element
        output_data = ttrt.runtime.to_host(output_tensors[0], untilize=True)[0]
        
        # Convert ttrt Tensor to numpy array with proper dtype
        if hasattr(output_data, 'get_data_buffer'):
            # It's a ttrt Tensor object
            output_shape = output_data.get_shape()
            output_buffer = output_data.get_data_buffer()
            # Try to use the expected output dtype
            output_data = np.frombuffer(output_buffer, dtype=output_dtype).reshape(output_shape)
        elif hasattr(output_data, 'numpy'):
            # If it's a torch tensor or similar
            output_data = output_data.numpy().astype(output_dtype)
        elif isinstance(output_data, list):
            output_data = np.array(output_data, dtype=output_dtype)
            # Try to match expected output shape if possible
            if output_data.size == expected_output.size:
                output_data = output_data.reshape(expected_output.shape)
        
        print("\n=== Output Information ===")
        print(f"Output shape: {output_data.shape}")
        print(f"Output data: {output_data}")
        print(f"Output dtype: {output_data.dtype}")
        
        # Verification
        print("\n=== Verification ===")
        
        # Flatten both arrays for comparison if shapes don't match
        output_flat = output_data.flatten()
        expected_flat = expected_output.flatten()
        
        # Allow for some shape flexibility - compare flattened arrays
        if output_flat.size != expected_flat.size:
            print(f"✗ Size mismatch: output has {output_flat.size} elements, expected {expected_flat.size}")
            success = False
        else:
            # Compare values based on data type
            if np.issubdtype(output_dtype, np.floating):
                # Floating point comparison
                # Handle NaN values specially
                nan_mask_output = np.isnan(output_flat)
                nan_mask_expected = np.isnan(expected_flat)
                
                # Check if NaN positions match
                if not np.array_equal(nan_mask_output, nan_mask_expected):
                    is_correct = False
                else:
                    # Compare non-NaN values
                    non_nan_mask = ~nan_mask_output
                    if np.any(non_nan_mask):
                        # Use a more relaxed tolerance for approximations (especially for division)
                        # For reciprocal-based division, errors can be around 1e-4 to 1e-3
                        tolerance = 1e-3
                        is_correct = np.allclose(output_flat[non_nan_mask], expected_flat[non_nan_mask], atol=tolerance, rtol=1e-3)
                    else:
                        # All values are NaN and match
                        is_correct = True
            else:
                # Integer comparison - exact match
                is_correct = np.array_equal(output_flat, expected_flat)
            
            if is_correct:
                print(f"✓ Output matches expected values")
                success = True
            else:
                # Show mismatches
                if np.issubdtype(output_dtype, np.floating):
                    matches = np.isclose(output_flat, expected_flat, atol=tolerance, rtol=1e-3)
                else:
                    matches = output_flat == expected_flat
                    
                match_count = matches.sum()
                total_count = output_flat.size
                print(f"✗ Output mismatch: {match_count}/{total_count} values match")
                
                # Show specific mismatches
                mismatch_indices = np.where(~matches)[0]
                num_examples = min(10, len(mismatch_indices))
                print(f"\nFirst {num_examples} mismatches:")
                for i in range(num_examples):
                    idx = mismatch_indices[i]
                    print(f"  [{idx}]: expected {expected_flat[idx]}, got {output_flat[idx]}")
                
                success = False
        
        # Cleanup
        ttrt.runtime.close_mesh_device(mesh_device)
        
        return success
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_ttnn_matrix.py <binary_file.ttnn> <input_matrix1> [input_matrix2 ...] <expected_output_matrix>")
        print("\nExamples:")
        print('  python run_ttnn_matrix.py vector4.ttnn "[1u32,2u32,3u32,4u32]" "[4u32]"')
        print('  python run_ttnn_matrix.py sub.ttnn "[2u64]" "[1u64]" "[1u64]"')
        print('  python run_ttnn_matrix.py add.ttnn "[1,2]" "[3,4]" "[4,6]"  # Multiple inputs')
        print('  python run_ttnn_matrix.py test.ttnn "[1.0,2.0]" "[3.0]"  # defaults to float32')
        print('  python run_ttnn_matrix.py matrix.ttnn "[[1i32,2i32],[3i32,4i32]]" "[[5i32,6i32],[7i32,8i32]]"')
        sys.exit(1)
    
    binary_path = sys.argv[1]
    # All arguments except the first (binary) and last (expected output) are inputs
    input_matrices = sys.argv[2:-1]
    expected_output_matrix = sys.argv[-1]
    
    success = run_ttnn_with_matrix_io(binary_path, input_matrices, expected_output_matrix)
    sys.exit(0 if success else 1)