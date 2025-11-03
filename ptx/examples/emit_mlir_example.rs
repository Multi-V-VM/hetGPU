// Example showing how to use the emit_linalg_mlir pass
// This converts PTX directly to MLIR using memref, arith, and linalg dialects

use ptx::pass;
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input.ptx>", args[0]);
        return Ok(());
    }

    let input_file = &args[1];

    // Read PTX source
    let ptx_source = fs::read_to_string(input_file)?;

    // Parse PTX
    let ast = ptx_parser::parse_module_checked(&ptx_source)
        .map_err(|e| format!("Failed to parse PTX: {:?}", e))?;

    println!("Converting PTX directly to Linalg MLIR...");

    // Convert to MLIR using the direct PTX-to-MLIR conversion
    match pass::to_mlir_module(ast) {
        Ok(mlir_code) => {
            println!("Generated MLIR:");
            println!("{}", mlir_code);

            // Write to file
            let output_file = input_file.replace(".ptx", "_linalg.mlir");
            fs::write(&output_file, &mlir_code)?;
            println!("MLIR written to: {}", output_file);
        }
        Err(e) => {
            eprintln!("Error converting to MLIR: {:?}", e);
        }
    }

    Ok(())
}
