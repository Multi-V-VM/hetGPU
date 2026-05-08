//! Standalone SASS Inliner Binary
//!
//! A CLI tool for testing SASS → LLVM IR inlining functionality.
//!
//! Usage:
//!   sass_inliner [OPTIONS] <input>
//!
//! Examples:
//!   # Inline SASS from CUBIN file
//!   sass_inliner kernel.cubin -o output.ll
//!
//!   # Use cuobjdump text output
//!   cuobjdump -sass -lineinfo kernel.cubin | sass_inliner --stdin -o output.ll
//!
//!   # Specify inlining strategy
//!   sass_inliner kernel.cubin --strategy hybrid -o output.ll
//!
//!   # Dump SASS instructions without LLVM conversion
//!   sass_inliner kernel.cubin --dump-sass

use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use llvm_zluda::core::*;
use llvm_zluda::{
    LLVMVerifierFailureAction, LLVMVerifyModule, LLVMWriteBitcodeToMemoryBuffer,
    LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos,
    LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets,
};

use ptx::sass::{
    ControlFlowAnalyzer, CubinParser, EnhancedSassInstruction, PtxReconstructor, PtxRecoveryEngine,
    SassDisassembler, SassInlineConfig, SassInlineStrategy, SassInliner, SassKernelBuilder,
    TextDisassemblyParser,
};

/// CLI arguments
struct Args {
    /// Input file (CUBIN or - for stdin with cuobjdump output)
    input: String,
    /// Output file (defaults to stdout)
    output: Option<PathBuf>,
    /// Read cuobjdump output from stdin
    stdin: bool,
    /// Inlining strategy
    strategy: SassInlineStrategy,
    /// Preserve control codes
    preserve_control_codes: bool,
    /// Target SM version
    target_sm: u32,
    /// Dump SASS instructions without LLVM conversion
    dump_sass: bool,
    /// Dump parsed CUBIN structure
    dump_cubin: bool,
    /// Emit LLVM bitcode instead of text IR
    emit_bitcode: bool,
    /// Verify LLVM module after generation
    verify: bool,
    /// Verbose output
    verbose: bool,
    /// PTX source file (for mapping)
    ptx_source: Option<PathBuf>,
    /// Recover PTX from SASS using debug info
    recover_ptx: bool,
    /// Output file for recovered PTX
    ptx_output: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: None,
            stdin: false,
            strategy: SassInlineStrategy::Hybrid,
            preserve_control_codes: true,
            target_sm: 75,
            dump_sass: false,
            dump_cubin: false,
            emit_bitcode: false,
            verify: true,
            verbose: false,
            ptx_source: None,
            recover_ptx: false,
            ptx_output: None,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv: Vec<String> = std::env::args().collect();
    argv.remove(0); // Remove program name

    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-o" | "--output" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--output requires an argument".to_string());
                }
                args.output = Some(PathBuf::from(&argv[i]));
            }
            "--stdin" => {
                args.stdin = true;
            }
            "-s" | "--strategy" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--strategy requires an argument".to_string());
                }
                args.strategy = match argv[i].as_str() {
                    "asm" | "inline-asm" | "assembly" => SassInlineStrategy::InlineAssembly,
                    "ptx" | "reconstruct" | "reconstruction" | "ptx-reconstruction" => {
                        SassInlineStrategy::PtxReconstruction
                    }
                    "meta" | "metadata" | "metadata-only" => SassInlineStrategy::MetadataOnly,
                    "hybrid" => SassInlineStrategy::Hybrid,
                    other => return Err(format!("Unknown strategy: {}", other)),
                };
            }
            "--sm" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--sm requires an argument".to_string());
                }
                args.target_sm = argv[i].parse().map_err(|_| "Invalid SM version")?;
            }
            "--dump-sass" => {
                args.dump_sass = true;
            }
            "--dump-cubin" => {
                args.dump_cubin = true;
            }
            "--emit-bitcode" | "-c" => {
                args.emit_bitcode = true;
            }
            "--no-verify" => {
                args.verify = false;
            }
            "--no-control-codes" => {
                args.preserve_control_codes = false;
            }
            "-v" | "--verbose" => {
                args.verbose = true;
            }
            "--ptx" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--ptx requires an argument".to_string());
                }
                args.ptx_source = Some(PathBuf::from(&argv[i]));
            }
            "--recover-ptx" => {
                args.recover_ptx = true;
            }
            "--ptx-output" => {
                i += 1;
                if i >= argv.len() {
                    return Err("--ptx-output requires an argument".to_string());
                }
                args.ptx_output = Some(PathBuf::from(&argv[i]));
            }
            "-" => {
                // '-' is a valid input meaning stdin
                if args.input.is_empty() {
                    args.input = "-".to_string();
                    args.stdin = true; // Automatically enable stdin mode
                } else {
                    return Err("Unexpected argument: -".to_string());
                }
            }
            other if other.starts_with('-') => {
                return Err(format!("Unknown option: {}", other));
            }
            _ => {
                if args.input.is_empty() {
                    args.input = arg.clone();
                } else {
                    return Err(format!("Unexpected argument: {}", arg));
                }
            }
        }
        i += 1;
    }

    if args.input.is_empty() && !args.stdin {
        return Err("No input file specified".to_string());
    }

    Ok(args)
}

fn print_help() {
    println!(
        r#"SASS Inliner - Convert SASS to LLVM IR

USAGE:
    sass_inliner [OPTIONS] <input>

ARGS:
    <input>     Input CUBIN file, or use --stdin for cuobjdump output

OPTIONS:
    -h, --help              Show this help message
    -o, --output <file>     Output file (default: stdout)
    --stdin                 Read cuobjdump -sass output from stdin
    -s, --strategy <type>   Inlining strategy:
                              asm, inline-asm    - Preserve exact SASS encoding
                              ptx, reconstruct   - Convert to PTX-equivalent IR
                              meta, metadata     - Metadata only, no IR changes
                              hybrid (default)   - Balance precision/compatibility
    --sm <version>          Target SM version (default: 75)
    --ptx <file>            PTX source file for mapping
    --recover-ptx           Recover PTX from SASS using debug info
    --ptx-output <file>     Output file for recovered PTX (default: stdout)
    --dump-sass             Dump parsed SASS instructions (no LLVM)
    --dump-cubin            Dump parsed CUBIN structure
    -c, --emit-bitcode      Emit LLVM bitcode instead of text IR
    --no-verify             Skip LLVM module verification
    --no-control-codes      Don't preserve SASS control codes
    -v, --verbose           Verbose output

PTX RECOVERY:
    The --recover-ptx option extracts PTX source from CUBIN files using
    DWARF debug information (.debug_line section). If no debug info is
    available, it reconstructs approximate PTX from SASS instruction
    semantics.

EXAMPLES:
    # Inline SASS from CUBIN file
    sass_inliner kernel.cubin -o output.ll

    # Use cuobjdump output
    cuobjdump -sass -lineinfo kernel.cubin | sass_inliner --stdin -o output.ll

    # Dump SASS instructions
    sass_inliner kernel.cubin --dump-sass

    # Use PTX reconstruction strategy
    sass_inliner kernel.cubin --strategy ptx -o output.ll

    # Recover PTX from CUBIN with debug info
    sass_inliner kernel.cubin --recover-ptx --ptx-output recovered.ptx

    # Recover PTX with original PTX source for better mapping
    sass_inliner kernel.cubin --recover-ptx --ptx original.ptx -o recovered.ptx
"#
    );
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Load PTX source if provided
    let _ptx_source = if let Some(ref path) = args.ptx_source {
        Some(fs::read_to_string(path).map_err(|e| format!("Failed to read PTX source: {}", e))?)
    } else {
        None
    };

    // Get instructions based on input type
    let (instructions, kernel_name) = if args.stdin {
        load_from_stdin(&args)?
    } else {
        load_from_cubin(&args)?
    };

    if args.verbose {
        eprintln!(
            "Loaded {} instructions from kernel '{}'",
            instructions.len(),
            kernel_name
        );
    }

    // Dump SASS if requested
    if args.dump_sass {
        dump_sass_instructions(&instructions);
        return Ok(());
    }

    // Recover PTX if requested
    if args.recover_ptx {
        if args.verbose {
            eprintln!("Recovering PTX from SASS...");
        }

        // Check if we have a CUBIN for full recovery or need reconstruction
        if !args.stdin && fs::metadata(&args.input).is_ok() {
            // Try full PTX recovery from CUBIN with debug info
            let cubin_data = fs::read(&args.input)
                .map_err(|e| format!("Failed to read CUBIN for PTX recovery: {}", e))?;

            if cubin_data.starts_with(&[0x7f, b'E', b'L', b'F']) {
                let mut engine = PtxRecoveryEngine::new(cubin_data)
                    .map_err(|e| format!("Failed to create PTX recovery engine: {}", e))?;

                // Add PTX source if provided
                if let Some(ref ptx_path) = args.ptx_source {
                    engine
                        .add_ptx_source(ptx_path)
                        .map_err(|e| format!("Failed to add PTX source: {}", e))?;
                }

                let result = engine
                    .recover_all()
                    .map_err(|e| format!("Failed to recover PTX: {}", e))?;

                // Output recovered PTX
                let mut ptx_content = result.ptx_header.clone();
                for func in &result.functions {
                    if let Some(ref reconstructed) = func.reconstructed_ptx {
                        ptx_content.push_str("\n");
                        ptx_content.push_str(reconstructed);
                    }
                }

                if let Some(ref ptx_out) = args.ptx_output {
                    fs::write(ptx_out, &ptx_content)
                        .map_err(|e| format!("Failed to write PTX output: {}", e))?;
                    eprintln!("Recovered PTX written to {:?}", ptx_out);
                    eprintln!("  Functions: {}", result.functions.len());
                    eprintln!(
                        "  Total SASS instructions: {}",
                        result.stats.total_sass_instructions
                    );
                    eprintln!("  Mapped to PTX: {}", result.stats.mapped_instructions);
                } else {
                    println!("{}", ptx_content);
                }

                return Ok(());
            }
        }

        // Fallback: reconstruct PTX from SASS semantics
        let mut reconstructor = PtxReconstructor::new(args.target_sm);
        let mut ptx_content = format!(
            ".version 7.0\n.target sm_{}\n.address_size 64\n\n",
            args.target_sm
        );
        ptx_content.push_str(&format!(".visible .entry {} ()\n{{\n", kernel_name));
        ptx_content.push_str("    .reg .b32 %r<256>;\n");
        ptx_content.push_str("    .reg .pred %p<8>;\n\n");

        for inst in &instructions {
            let ptx = reconstructor.reconstruct_instruction(inst);
            ptx_content.push_str(&format!("    // 0x{:04x}: {}\n", inst.address, inst.opcode));
            ptx_content.push_str(&format!("    {}\n", ptx));
        }
        ptx_content.push_str("}\n");

        if let Some(ref ptx_out) = args.ptx_output {
            fs::write(ptx_out, &ptx_content)
                .map_err(|e| format!("Failed to write PTX output: {}", e))?;
            eprintln!("Reconstructed PTX written to {:?}", ptx_out);
        } else {
            println!("{}", ptx_content);
        }

        return Ok(());
    }

    // Create LLVM module
    unsafe {
        // Initialize LLVM
        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmPrinters();
        LLVM_InitializeAllAsmParsers();

        let context = LLVMContextCreate();
        let module_name = CString::new("sass_inlined").unwrap();
        let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);

        // Set target triple for NVPTX
        let triple = CString::new("nvptx64-nvidia-cuda").unwrap();
        LLVMSetTarget(module, triple.as_ptr());

        // Create kernel function
        let config = SassInlineConfig {
            strategy: args.strategy,
            preserve_control_codes: args.preserve_control_codes,
            emit_debug_info: true,
            track_registers: true,
            target_sm: args.target_sm,
            function_prefix: "_sass_".to_string(),
        };

        if args.verbose {
            eprintln!("Using strategy: {:?}", args.strategy);
            eprintln!("Target SM: {}", args.target_sm);
        }

        // Build kernel
        let builder = SassKernelBuilder::new(config.clone());
        let _function = builder.build_kernel(
            context,
            module,
            &kernel_name,
            &instructions,
            &[], // No parameters for now
        )?;

        if args.verbose {
            eprintln!("Created kernel function: _sass_{}", kernel_name);
        }

        // Verify module
        if args.verify {
            let mut error_msg: *mut std::os::raw::c_char = std::ptr::null_mut();
            if LLVMVerifyModule(
                module,
                LLVMVerifierFailureAction::LLVMReturnStatusAction,
                &mut error_msg,
            ) != 0
            {
                let error = if !error_msg.is_null() {
                    let s = std::ffi::CStr::from_ptr(error_msg)
                        .to_string_lossy()
                        .to_string();
                    LLVMDisposeMessage(error_msg);
                    s
                } else {
                    "Unknown verification error".to_string()
                };
                // Don't fail on verification errors, just warn
                eprintln!("Warning: LLVM verification failed: {}", error);
            } else if args.verbose {
                eprintln!("LLVM module verified successfully");
            }
        }

        // Get statistics from inliner
        let _inliner = SassInliner::new(context, module, config);
        // Stats would be accumulated during actual inlining

        // Output the module
        let output_data = if args.emit_bitcode {
            let mem_buf = LLVMWriteBitcodeToMemoryBuffer(module);
            let data_ptr = LLVMGetBufferStart(mem_buf);
            let data_len = LLVMGetBufferSize(mem_buf);
            let data = std::slice::from_raw_parts(data_ptr as *const u8, data_len).to_vec();
            LLVMDisposeMemoryBuffer(mem_buf);
            data
        } else {
            let ir_str = LLVMPrintModuleToString(module);
            let ir = std::ffi::CStr::from_ptr(ir_str).to_bytes().to_vec();
            LLVMDisposeMessage(ir_str);
            ir
        };

        // Write output
        if let Some(ref path) = args.output {
            fs::write(path, &output_data).map_err(|e| format!("Failed to write output: {}", e))?;
            if args.verbose {
                eprintln!("Wrote output to {:?}", path);
            }
        } else {
            io::stdout()
                .write_all(&output_data)
                .map_err(|e| format!("Failed to write to stdout: {}", e))?;
        }

        // Cleanup
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }

    Ok(())
}

fn load_from_cubin(args: &Args) -> Result<(Vec<EnhancedSassInstruction>, String), String> {
    let cubin_data =
        fs::read(&args.input).map_err(|e| format!("Failed to read CUBIN file: {}", e))?;

    if args.verbose {
        eprintln!("Read {} bytes from {}", cubin_data.len(), args.input);
    }

    // Parse CUBIN
    let parsed = CubinParser::new(cubin_data)
        .parse()
        .map_err(|e| format!("Failed to parse CUBIN: {:?}", e))?;

    if args.dump_cubin {
        dump_cubin_structure(&parsed);
        std::process::exit(0);
    }

    if parsed.kernels.is_empty() {
        return Err("No kernels found in CUBIN".to_string());
    }

    // Use first kernel
    let kernel = &parsed.kernels[0];
    let kernel_name = kernel.name.clone();

    if args.verbose {
        eprintln!(
            "Processing kernel '{}' (SM {}, {} bytes code)",
            kernel.name,
            kernel.sm_version,
            kernel.code.len()
        );
    }

    // Disassemble
    let disasm = SassDisassembler::new(kernel.sm_version)
        .map_err(|e| format!("Failed to create disassembler: {:?}", e))?;

    let mut instructions = disasm.disassemble(&kernel.code, kernel.address);

    // Apply debug info
    for inst in &mut instructions {
        if let Some(debug_info) = parsed.debug_lines.get(&inst.address) {
            inst.ptx_file = Some(debug_info.file.clone());
            inst.ptx_line = Some(debug_info.line);
            inst.ptx_column = Some(debug_info.column);
        }
        inst.function_name = Some(kernel_name.clone());
    }

    // Analyze control flow
    ControlFlowAnalyzer::find_basic_blocks(&mut instructions);
    ControlFlowAnalyzer::analyze_data_flow(&mut instructions);

    Ok((instructions, kernel_name))
}

fn load_from_stdin(args: &Args) -> Result<(Vec<EnhancedSassInstruction>, String), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("Failed to read stdin: {}", e))?;

    if args.verbose {
        eprintln!("Read {} bytes from stdin", input.len());
    }

    // Parse cuobjdump output
    let mut instructions = TextDisassemblyParser::parse_cuobjdump_output(&input);

    if instructions.is_empty() {
        return Err("No instructions parsed from input".to_string());
    }

    // Get kernel name from first instruction
    let kernel_name = instructions[0]
        .function_name
        .clone()
        .unwrap_or_else(|| "kernel".to_string());

    // Analyze control flow
    ControlFlowAnalyzer::find_basic_blocks(&mut instructions);
    ControlFlowAnalyzer::analyze_data_flow(&mut instructions);

    Ok((instructions, kernel_name))
}

fn dump_sass_instructions(instructions: &[EnhancedSassInstruction]) {
    println!("=== SASS Instructions ({} total) ===\n", instructions.len());

    for inst in instructions {
        // Address and opcode
        print!("{:08x}: {:12}", inst.address, inst.opcode);

        // Operands
        let dest_str: Vec<String> = inst
            .dest_operands
            .iter()
            .map(|op| format!("{:?}", op))
            .collect();
        let src_str: Vec<String> = inst
            .src_operands
            .iter()
            .map(|op| format!("{:?}", op))
            .collect();

        if !dest_str.is_empty() || !src_str.is_empty() {
            print!("  ");
            if !dest_str.is_empty() {
                print!("{}", dest_str.join(", "));
            }
            if !src_str.is_empty() {
                if !dest_str.is_empty() {
                    print!(", ");
                }
                print!("{}", src_str.join(", "));
            }
        }

        println!();

        // Control codes
        let cc = &inst.control_codes;
        if cc.stall_count > 0
            || cc.write_barrier > 0
            || cc.read_barrier > 0
            || cc.wait_barrier_mask > 0
        {
            println!(
                "          [stall:{} wb:{} rb:{} wait:0x{:x}]",
                cc.stall_count, cc.write_barrier, cc.read_barrier, cc.wait_barrier_mask
            );
        }

        // PTX mapping
        if let (Some(file), Some(line)) = (&inst.ptx_file, inst.ptx_line) {
            println!("          -> {}:{}", file, line);
        }

        // Basic block boundary
        if inst.basic_block_id.is_some() {
            // Check if this starts a new basic block
        }
    }

    println!("\n=== Summary ===");
    println!("Total instructions: {}", instructions.len());

    // Count by opcode class
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for inst in instructions {
        let class = format!("{:?}", inst.opcode_class);
        *class_counts.entry(class).or_insert(0) += 1;
    }

    println!("\nBy opcode class:");
    let mut sorted: Vec<_> = class_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (class, count) in sorted.iter().take(10) {
        println!("  {:30} {:5}", class, count);
    }
}

fn dump_cubin_structure(parsed: &ptx::sass::ParsedCubin) {
    println!("=== CUBIN Structure ===\n");

    println!("Architecture: SM {}", parsed.sm_version);
    println!("Kernels: {}", parsed.kernels.len());
    println!("Constants: {}", parsed.constants.len());
    println!("Debug lines: {}", parsed.debug_lines.len());

    println!("\n--- Kernels ---");
    for kernel in &parsed.kernels {
        println!("  {} (SM {})", kernel.name, kernel.sm_version);
        println!("    Address: 0x{:x}", kernel.address);
        println!("    Size: {} bytes", kernel.size);
        println!("    Code: {} bytes", kernel.code.len());
        println!("    Shared memory: {} bytes", kernel.shared_mem_size);
        println!("    Registers: {}", kernel.num_registers);
        println!("    Max threads: {}", kernel.max_threads_per_block);
    }

    if !parsed.constants.is_empty() {
        println!("\n--- Constants ---");
        for constant in &parsed.constants {
            println!(
                "  {} (bank {})",
                constant.name.as_deref().unwrap_or("<unnamed>"),
                constant.bank
            );
            println!(
                "    Offset: 0x{:x}, Size: {} bytes",
                constant.offset, constant.size
            );
        }
    }

    if !parsed.debug_lines.is_empty() {
        println!("\n--- Debug Line Mappings (first 10) ---");
        let mut lines: Vec<_> = parsed.debug_lines.iter().collect();
        lines.sort_by_key(|(addr, _)| *addr);
        for (addr, info) in lines.iter().take(10) {
            println!("  0x{:08x} -> {}:{}", addr, info.file, info.line);
        }
        if lines.len() > 10 {
            println!("  ... and {} more", lines.len() - 10);
        }
    }
}
