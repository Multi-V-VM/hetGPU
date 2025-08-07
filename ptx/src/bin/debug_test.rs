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
        Ok(module) => match module.0.print_to_string() {
            Ok(llvm_ir) => println!("{}", llvm_ir),
            Err(e) => {
                eprintln!("Error generating LLVM IR: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
