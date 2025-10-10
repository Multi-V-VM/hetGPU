// PTX to TMatmul Hardware Integration
// Compiles PTX code to TMatmul assembly for hardware execution

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "ptx_to_tmatmul_hw")]
#[command(about = "Compile PTX to TMatmul assembly for hardware execution", long_about = None)]
struct Args {
    /// Input PTX file
    #[arg(short, long)]
    input: PathBuf,

    /// Output assembly file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Hardware assembly directory (default: /root/matmulfreellm/hardware/ternary_matmul/asm)
    #[arg(long, default_value = "/root/matmulfreellm/hardware/ternary_matmul/asm")]
    hw_asm_dir: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Memory map file (JSON format)
    #[arg(long)]
    memory_map: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read PTX input
    if args.verbose {
        println!("Reading PTX from: {}", args.input.display());
    }
    let ptx_source = fs::read_to_string(&args.input)?;

    // Compile PTX to TMatmul
    if args.verbose {
        println!("Compiling PTX to TMatmul assembly...");
    }

    let tmatmul_asm = ptx::pass::ptx_to_tmatmul_assembly(&ptx_source)
        .map_err(|e| format!("Compilation error: {}", e))?;

    // Determine output path
    let output_path = if let Some(out) = args.output {
        out
    } else {
        // Generate default output path based on input filename
        let input_stem = args.input.file_stem()
            .ok_or("Invalid input filename")?
            .to_string_lossy();
        args.hw_asm_dir.join(format!("{}.S", input_stem))
    };

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write output
    if args.verbose {
        println!("Writing TMatmul assembly to: {}", output_path.display());
    }
    let mut output_file = fs::File::create(&output_path)?;
    output_file.write_all(tmatmul_asm.as_bytes())?;

    println!("✓ Successfully compiled to: {}", output_path.display());
    println!("\nTo run on hardware simulator:");
    println!("  cd {}", args.hw_asm_dir.parent().unwrap().join("cocotb").display());
    println!("  make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs");

    Ok(())
}
