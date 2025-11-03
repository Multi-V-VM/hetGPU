use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <ptx_file>", args[0]);
        std::process::exit(1);
    }

    let ptx_file = &args[1];
    let ptx_source = fs::read_to_string(ptx_file)?;

    // Parse PTX source
    let ast =
        ptx_parser::parse_module_checked(&ptx_source).map_err(|_| format!("PTX parsing failed"))?;

    // Use the PTX library to compile with debug info
    match ptx::to_llvm_module_with_debug_round_trip(ast) {
        Ok((module, _bitcode, _mappings)) => {
            // Print LLVM IR
            let llvm_ir = module.llvm_ir.print_module_to_string();
            println!("{}", llvm_ir.to_str());
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
