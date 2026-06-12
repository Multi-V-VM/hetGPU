//! CuTile to TOSA Transformation Module
//!
//! This module handles the transformation from CuTile MLIR dialect
//! to TOSA MLIR dialect using pattern-based rewriting.

use crate::{
    cutile_comgr_data_kind_s, cutile_comgr_data_set_t, cutile_comgr_status_s,
    cutile_comgr_status_t, get_next_handle, DataContent, DATA_SET_STORE, DATA_STORE,
};
use std::collections::HashMap;

/// Operation mapping from CuTile to TOSA
struct OpMapping {
    cutile_op: &'static str,
    tosa_op: &'static str,
    requires_reshape: bool,
}

lazy_static::lazy_static! {
    /// Mapping table for CuTile to TOSA operations
    static ref OP_MAPPINGS: Vec<OpMapping> = vec![
        // Arithmetic operations
        OpMapping { cutile_op: "cuda_tile.addf", tosa_op: "tosa.add", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.addi", tosa_op: "tosa.add", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.subf", tosa_op: "tosa.sub", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.subi", tosa_op: "tosa.sub", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.mulf", tosa_op: "tosa.mul", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.muli", tosa_op: "tosa.mul", requires_reshape: false },

        // Unary operations
        OpMapping { cutile_op: "cuda_tile.absf", tosa_op: "tosa.abs", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.absi", tosa_op: "tosa.abs", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.negf", tosa_op: "tosa.negate", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.ceil", tosa_op: "tosa.ceil", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.floor", tosa_op: "tosa.floor", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.exp", tosa_op: "tosa.exp", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.log", tosa_op: "tosa.log", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.rsqrt", tosa_op: "tosa.rsqrt", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.tanh", tosa_op: "tosa.tanh", requires_reshape: false },

        // Bitwise operations
        OpMapping { cutile_op: "cuda_tile.andi", tosa_op: "tosa.bitwise_and", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.ori", tosa_op: "tosa.bitwise_or", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.xori", tosa_op: "tosa.bitwise_xor", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.noti", tosa_op: "tosa.bitwise_not", requires_reshape: false },

        // Comparison operations
        OpMapping { cutile_op: "cuda_tile.cmpf", tosa_op: "tosa.greater", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.cmpi", tosa_op: "tosa.greater", requires_reshape: false },

        // Tensor operations
        OpMapping { cutile_op: "cuda_tile.constant", tosa_op: "tosa.const", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.reshape", tosa_op: "tosa.reshape", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.broadcast", tosa_op: "tosa.tile", requires_reshape: true },
        OpMapping { cutile_op: "cuda_tile.cat", tosa_op: "tosa.concat", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.select", tosa_op: "tosa.select", requires_reshape: false },

        // Reduction operations
        OpMapping { cutile_op: "cuda_tile.reduce_add", tosa_op: "tosa.reduce_sum", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.reduce_max", tosa_op: "tosa.reduce_max", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.reduce_min", tosa_op: "tosa.reduce_min", requires_reshape: false },

        // Min/Max operations
        OpMapping { cutile_op: "cuda_tile.maxf", tosa_op: "tosa.maximum", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.minf", tosa_op: "tosa.minimum", requires_reshape: false },
        OpMapping { cutile_op: "cuda_tile.clamp", tosa_op: "tosa.clamp", requires_reshape: false },

        // Matrix operations
        OpMapping { cutile_op: "cuda_tile.matmul", tosa_op: "tosa.matmul", requires_reshape: false },
    ];
}

/// Transform CuTile MLIR to TOSA MLIR
pub fn transform_to_tosa(
    input_handles: &[u64],
    output_set: cutile_comgr_data_set_t,
) -> cutile_comgr_status_t {
    eprintln!(
        "CuTile->TOSA: Transforming {} input(s)",
        input_handles.len()
    );

    for &handle in input_handles {
        // Get input data
        let input_content = {
            let store = DATA_STORE.lock().unwrap();
            store.get(&handle).cloned()
        };

        let input_content = match input_content {
            Some(content) => content,
            None => {
                eprintln!("CuTile->TOSA: Invalid input handle {}", handle);
                return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
            }
        };

        // Verify input is CuTile MLIR
        if input_content.kind != cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_MLIR_CUTILE {
            eprintln!("CuTile->TOSA: Skipping non-CuTile-MLIR input");
            continue;
        }

        // Parse and transform MLIR
        let cutile_mlir = match std::str::from_utf8(&input_content.content) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("CuTile->TOSA: Invalid UTF-8 in MLIR content");
                return Err(cutile_comgr_status_s::CUTILE_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
            }
        };

        let tosa_mlir = transform_mlir_text(cutile_mlir)?;

        // Create output data
        let output_handle = get_next_handle();
        {
            let mut store = DATA_STORE.lock().unwrap();
            store.insert(
                output_handle,
                DataContent {
                    kind: cutile_comgr_data_kind_s::CUTILE_COMGR_DATA_KIND_MLIR_TOSA,
                    content: tosa_mlir.into_bytes(),
                    name: input_content.name.map(|n| n.replace(".mlir", "_tosa.mlir")),
                },
            );
        }

        // Add to output set
        {
            let mut store = DATA_SET_STORE.lock().unwrap();
            if let Some(handles) = store.get_mut(&output_set.handle) {
                handles.push(output_handle);
            }
        }

        eprintln!("CuTile->TOSA: Successfully transformed to TOSA");
    }

    Ok(())
}

/// Transform MLIR text from CuTile dialect to TOSA dialect
fn transform_mlir_text(input: &str) -> Result<String, cutile_comgr_status_s> {
    let mut output = String::new();
    let mut type_mappings: HashMap<String, String> = HashMap::new();

    for line in input.lines() {
        let transformed = transform_line(line, &mut type_mappings);
        output.push_str(&transformed);
        output.push('\n');
    }

    Ok(output)
}

/// Transform a single line of MLIR
fn transform_line(line: &str, type_mappings: &mut HashMap<String, String>) -> String {
    let mut result = line.to_string();

    // Transform CuTile operations to TOSA operations
    for mapping in OP_MAPPINGS.iter() {
        if result.contains(mapping.cutile_op) {
            result = result.replace(mapping.cutile_op, mapping.tosa_op);
        }
    }

    // Transform tile types to tensor types
    // !cuda_tile.tile<4xf32> -> tensor<4xf32>
    result = transform_types(&result, type_mappings);

    // Handle special cases
    result = handle_special_ops(&result);

    result
}

/// Transform CuTile types to TOSA-compatible types
fn transform_types(line: &str, _type_mappings: &mut HashMap<String, String>) -> String {
    let mut result = line.to_string();

    // Replace cuda_tile.tile types with tensor types
    // Pattern: !cuda_tile.tile<SHAPE x TYPE> -> tensor<SHAPE x TYPE>
    while let Some(start) = result.find("!cuda_tile.tile<") {
        if let Some(end) = result[start..].find('>') {
            let end_pos = start + end + 1;
            let tile_type = &result[start..end_pos];

            // Extract shape and element type
            let inner = &tile_type[16..tile_type.len() - 1]; // Remove "!cuda_tile.tile<" and ">"
            let tensor_type = format!("tensor<{}>", inner);

            result = format!("{}{}{}", &result[..start], tensor_type, &result[end_pos..]);
        } else {
            break;
        }
    }

    // Replace cuda_tile.ptr types with memref or i64
    result = result.replace("!cuda_tile.ptr<", "memref<");

    result
}

/// Handle special operations that require more complex transformations
fn handle_special_ops(line: &str) -> String {
    let mut result = line.to_string();

    // Handle tosa.mul which requires a shift parameter for integers
    if result.contains("tosa.mul") && !result.contains("shift") {
        // Add shift = 0 for integer multiplications if not present
        if let Some(pos) = result.find("tosa.mul") {
            // Check if this looks like an integer operation
            if result.contains("i32")
                || result.contains("i64")
                || result.contains("i16")
                || result.contains("i8")
            {
                // Find the position after the operands to insert shift
                if let Some(colon_pos) = result[pos..].find(':') {
                    let insert_pos = pos + colon_pos;
                    result = format!(
                        "{} {{shift = 0 : i8}} {}",
                        &result[..insert_pos],
                        &result[insert_pos..]
                    );
                }
            }
        }
    }

    // Handle broadcast -> tile transformation
    // tosa.tile requires multiples attribute
    if result.contains("tosa.tile") && !result.contains("multiples") {
        // This is a simplified handling - in practice, we'd need to compute
        // the multiples from the source and destination shapes
        if let Some(arrow_pos) = result.find("->") {
            // Extract destination shape and compute multiples
            // For now, add a placeholder
            result = result.replace(
                "tosa.tile",
                "tosa.tile {multiples = array<i64: 1, 1, 1, 1>}",
            );
        }
    }

    result
}

/// High-level MLIR transformer that handles module structure
pub struct MlirTransformer {
    /// Collected function definitions
    functions: Vec<String>,
    /// Global constants
    constants: Vec<String>,
}

impl MlirTransformer {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            constants: Vec::new(),
        }
    }

    /// Transform a complete CuTile module to TOSA
    pub fn transform_module(&mut self, input: &str) -> Result<String, String> {
        let mut type_mappings = HashMap::new();
        let mut output = String::new();

        // Write module header
        output.push_str("// TOSA module generated from CuTile IR\n");
        output.push_str("// Transformation performed by cutile_comgr_sys\n\n");

        for line in input.lines() {
            let transformed = transform_line(line, &mut type_mappings);
            output.push_str(&transformed);
            output.push('\n');
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_transform() {
        let mut mappings = HashMap::new();
        let input = "!cuda_tile.tile<4xf32>";
        let output = transform_types(input, &mut mappings);
        assert_eq!(output, "tensor<4xf32>");
    }

    #[test]
    fn test_op_transform() {
        let mut mappings = HashMap::new();
        let input = "%0 = cuda_tile.addf %1, %2 : !cuda_tile.tile<4xf32>";
        let output = transform_line(input, &mut mappings);
        assert!(output.contains("tosa.add"));
        assert!(output.contains("tensor<4xf32>"));
    }

    #[test]
    fn test_full_module_transform() {
        let input = r#"
module {
  func.func @test(%arg0: !cuda_tile.tile<4xf32>) -> !cuda_tile.tile<4xf32> {
    %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<4xf32>
    return %0 : !cuda_tile.tile<4xf32>
  }
}
"#;
        let result = transform_mlir_text(input);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("tosa.abs"));
        assert!(output.contains("tensor<4xf32>"));
    }
}
