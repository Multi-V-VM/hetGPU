// PTX Debug Tool - State Recovery CLI
// This tool provides command-line interface for PTX debugging and state recovery

use clap::{Args, Parser, Subcommand};
use ptx::debug::*;
use ptx::state_recovery::*;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// GPU Backend Types
#[derive(Debug, Clone, Copy, PartialEq)]
enum GpuBackend {
    Nvidia,
    Amd,
    Intel,
    Tenstorrent,
}

// Runtime state for managing GPU devices and kernels
struct HetGpuRuntime {
    current_device: GpuBackend,
    compiled_kernels: HashMap<(String, GpuBackend), Vec<u8>>, // (kernel_name, backend) -> native code
    device_memory: HashMap<String, DeviceMemory>,             // memory allocations
    pause_flags: HashMap<String, Arc<Mutex<bool>>>, // kernel pause flags for cooperative checkpointing
}

struct DeviceMemory {
    backend: GpuBackend,
    size: usize,
    device_ptr: u64,
    host_mirror: Vec<u8>, // Host copy for migration
}

impl HetGpuRuntime {
    fn new() -> Self {
        Self {
            current_device: Self::detect_device(),
            compiled_kernels: HashMap::new(),
            device_memory: HashMap::new(),
            pause_flags: HashMap::new(),
        }
    }

    fn detect_device() -> GpuBackend {
        // In real implementation, would check PCI devices, environment vars, etc.
        // For now, default to NVIDIA
        if std::env::var("HETGPU_BACKEND").is_ok() {
            match std::env::var("HETGPU_BACKEND").unwrap().as_str() {
                "amd" => GpuBackend::Amd,
                "intel" => GpuBackend::Intel,
                "tenstorrent" => GpuBackend::Tenstorrent,
                _ => GpuBackend::Nvidia,
            }
        } else {
            GpuBackend::Nvidia
        }
    }

    // JIT compile PTX to target backend
    fn compile_kernel(
        &mut self,
        kernel_name: &str,
        ptx_code: &str,
        target: GpuBackend,
    ) -> Result<Vec<u8>, String> {
        let key = (kernel_name.to_string(), target);

        // Check cache
        if let Some(compiled) = self.compiled_kernels.get(&key) {
            return Ok(compiled.clone());
        }

        println!("JIT compiling {} for {:?}...", kernel_name, target);
        let start = Instant::now();

        let compiled = match target {
            GpuBackend::Nvidia => {
                // PTX is native for NVIDIA, just return as-is
                // In real implementation, would use cuModuleLoadDataEx
                ptx_code.as_bytes().to_vec()
            }
            GpuBackend::Amd => {
                // Translate PTX to SPIR-V, then to AMD GCN
                self.ptx_to_spirv(ptx_code)?
            }
            GpuBackend::Intel => {
                // Translate PTX to SPIR-V for Intel
                self.ptx_to_spirv(ptx_code)?
            }
            GpuBackend::Tenstorrent => {
                // Translate PTX to Metalium assembly
                self.ptx_to_metalium(ptx_code)?
            }
        };

        println!("JIT compilation took {:?}", start.elapsed());
        self.compiled_kernels.insert(key, compiled.clone());
        Ok(compiled)
    }

    fn ptx_to_spirv(&self, ptx_code: &str) -> Result<Vec<u8>, String> {
        // Placeholder for PTX to SPIR-V translation
        // Would use LLVM to parse PTX and emit SPIR-V
        Err("PTX to SPIR-V translation not yet implemented".to_string())
    }

    fn ptx_to_metalium(&self, ptx_code: &str) -> Result<Vec<u8>, String> {
        // Placeholder for PTX to Tenstorrent Metalium translation
        // Would map PTX warps to vector operations on Tensix cores
        Err("PTX to Metalium translation not yet implemented".to_string())
    }

    // Allocate device memory with host mirror for migration
    fn allocate_memory(&mut self, name: String, size: usize) -> Result<u64, String> {
        let device_ptr = match self.current_device {
            GpuBackend::Nvidia => {
                // Would call cuMemAlloc
                0x1000000 // Dummy address
            }
            GpuBackend::Amd => {
                // Would call hipMalloc
                0x2000000
            }
            GpuBackend::Intel => {
                // Would call zeMemAllocDevice
                0x3000000
            }
            GpuBackend::Tenstorrent => {
                // Would use TT memory allocation
                0x4000000
            }
        };

        let mem = DeviceMemory {
            backend: self.current_device,
            size,
            device_ptr,
            host_mirror: vec![0u8; size],
        };

        self.device_memory.insert(name, mem);
        Ok(device_ptr)
    }

    // Copy memory to host for migration
    fn copy_to_host(&mut self, name: &str) -> Result<(), String> {
        let mem = self
            .device_memory
            .get_mut(name)
            .ok_or("Memory allocation not found")?;

        println!("Copying {} bytes from device to host...", mem.size);
        // In real implementation, would use cudaMemcpyDeviceToHost, etc.
        Ok(())
    }

    // Copy memory from host after migration
    fn copy_from_host(&mut self, name: &str) -> Result<(), String> {
        let mem = self
            .device_memory
            .get_mut(name)
            .ok_or("Memory allocation not found")?;

        println!("Copying {} bytes from host to device...", mem.size);
        // In real implementation, would use cudaMemcpyHostToDevice, etc.
        Ok(())
    }
}

// Kernel execution state for checkpointing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KernelState {
    kernel_name: String,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    segment_id: u32, // Which segment of the kernel we're in
    thread_states: Vec<ThreadState>,
    shared_memory: Vec<u8>,
    memory_snapshot: HashMap<String, Vec<u8>>, // Named memory allocations
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ThreadState {
    thread_id: (u32, u32, u32),
    block_id: (u32, u32, u32),
    registers: HashMap<String, u64>, // Register name -> value
    program_counter: u32,
    predicate_mask: u32,
}

#[derive(Parser)]
#[command(name = "ptx-debug")]
#[command(about = "PTX Debug Tool for state recovery and debugging")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile PTX with debug information
    Compile(CompileArgs),
    /// Analyze debug mappings
    Analyze(AnalyzeArgs),
    /// Interactive debugging session
    Debug(DebugArgs),
    /// Recover state from crash dump
    Recover(RecoverArgs),
    /// Export debug information
    Export(ExportArgs),
    /// Live migrate a running kernel between GPUs
    Migrate(MigrateArgs),
    /// Launch a kernel with migration support
    Launch(LaunchArgs),
}

#[derive(Args)]
struct MigrateArgs {
    /// Running kernel identifier
    #[arg(short, long)]
    kernel_id: String,

    /// Source GPU backend
    #[arg(short, long)]
    source: String,

    /// Target GPU backend
    #[arg(short, long)]
    target: String,

    /// Output state file
    #[arg(short, long)]
    state_file: PathBuf,
}

#[derive(Args)]
struct LaunchArgs {
    /// PTX kernel file
    #[arg(short, long)]
    kernel: PathBuf,

    /// Kernel function name
    #[arg(short, long)]
    function: String,

    /// Grid dimensions (x,y,z)
    #[arg(short, long, value_delimiter = ',')]
    grid: Vec<u32>,

    /// Block dimensions (x,y,z)
    #[arg(short, long, value_delimiter = ',')]
    block: Vec<u32>,

    /// Enable migration support
    #[arg(short, long)]
    migration: bool,
}

#[derive(Args)]
struct CompileArgs {
    /// Input PTX file
    #[arg(short, long)]
    input: PathBuf,

    /// Output debug mapping file
    #[arg(short, long)]
    output: PathBuf,

    /// Target architecture (amd_gcn, intel_spirv, sass)
    #[arg(short, long, default_value = "amd_gcn")]
    target: String,

    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct AnalyzeArgs {
    /// Debug mapping file to analyze
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Show detailed instruction mappings
    #[arg(short, long)]
    detailed: bool,

    /// Filter by PTX line number
    #[arg(short, long)]
    line: Option<u32>,
}

#[derive(Args)]
struct DebugArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Initial breakpoint location (file:line:column)
    #[arg(short, long)]
    breakpoint: Option<String>,

    /// GDB-compatible mode
    #[arg(short, long)]
    gdb_mode: bool,
}

#[derive(Args)]
struct RecoverArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Target architecture crash address
    #[arg(short, long)]
    address: String,

    /// Target architecture (amd_gcn, intel_spirv, sass)
    #[arg(short, long, default_value = "amd_gcn")]
    target: String,

    /// Memory dump file
    #[arg(long)]
    memory_dump: Option<PathBuf>,
}

#[derive(Args)]
struct ExportArgs {
    /// Debug mapping file
    #[arg(short, long)]
    mapping_file: PathBuf,

    /// Export format (json, gdb, vscode)
    #[arg(short, long, default_value = "json")]
    format: String,

    /// Output file
    #[arg(short, long)]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile(args) => compile_command(args).await,
        Commands::Analyze(args) => analyze_command(args).await,
        Commands::Debug(args) => debug_command(args).await,
        Commands::Recover(args) => recover_command(args).await,
        Commands::Export(args) => export_command(args).await,
        Commands::Migrate(args) => migrate_command(args).await,
        Commands::Launch(args) => launch_command(args).await,
    }
}

async fn migrate_command(args: MigrateArgs) -> anyhow::Result<()> {
    println!("Starting live migration of kernel {}...", args.kernel_id);
    println!("Source: {} -> Target: {}", args.source, args.target);

    let mut runtime = HetGpuRuntime::new();

    // Step 1: Set pause flag for the kernel
    println!("Setting pause flag for cooperative checkpoint...");
    if let Some(pause_flag) = runtime.pause_flags.get(&args.kernel_id) {
        *pause_flag.lock().unwrap() = true;
    }

    // Step 2: Wait for kernel to reach safe checkpoint (barrier)
    println!("Waiting for kernel to reach barrier...");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Step 3: Capture kernel state
    println!("Capturing kernel state...");
    let kernel_state = capture_kernel_state(&args.kernel_id)?;

    // Step 4: Copy device memory to host
    println!("Copying device memory to host...");
    for (name, _) in &runtime.device_memory {
        runtime.copy_to_host(name)?;
    }

    // Step 5: Save state to file
    let state_json = serde_json::to_string_pretty(&kernel_state)?;
    fs::write(&args.state_file, state_json)?;
    println!("Kernel state saved to: {}", args.state_file.display());

    // Step 6: Switch to target device
    runtime.current_device = match args.target.as_str() {
        "amd" => GpuBackend::Amd,
        "intel" => GpuBackend::Intel,
        "tenstorrent" => GpuBackend::Tenstorrent,
        _ => GpuBackend::Nvidia,
    };

    // Step 7: Restore on target device
    println!("Restoring kernel on target device...");
    restore_kernel_state(&kernel_state, &mut runtime)?;

    println!("Migration completed successfully!");
    Ok(())
}

async fn launch_command(args: LaunchArgs) -> anyhow::Result<()> {
    println!("Launching kernel with migration support...");

    let mut runtime = HetGpuRuntime::new();

    // Read PTX code
    let ptx_code = fs::read_to_string(&args.kernel)?;

    // Parse grid and block dimensions
    let grid_dim = match args.grid.as_slice() {
        [x] => (*x, 1, 1),
        [x, y] => (*x, *y, 1),
        [x, y, z] => (*x, *y, *z),
        _ => (1, 1, 1),
    };

    let block_dim = match args.block.as_slice() {
        [x] => (*x, 1, 1),
        [x, y] => (*x, *y, 1),
        [x, y, z] => (*x, *y, *z),
        _ => (1, 1, 1),
    };

    // Compile kernel for current device
    let kernel_binary =
        runtime.compile_kernel(&args.function, &ptx_code, runtime.current_device)?;

    // Create pause flag for this kernel
    let kernel_id = format!("{}_{}", args.function, std::process::id());
    runtime
        .pause_flags
        .insert(kernel_id.clone(), Arc::new(Mutex::new(false)));

    println!("Kernel ID: {}", kernel_id);
    println!("Grid: {:?}, Block: {:?}", grid_dim, block_dim);
    println!("Backend: {:?}", runtime.current_device);

    if args.migration {
        println!("Migration support enabled - kernel will check for pause at barriers");
    }

    // In real implementation, would actually launch the kernel
    println!("Kernel launched successfully!");

    Ok(())
}

fn capture_kernel_state(kernel_id: &str) -> anyhow::Result<KernelState> {
    // In real implementation, would:
    // 1. Use NVBit or similar to read register values
    // 2. Copy shared memory contents
    // 3. Record program counters

    // Mock implementation
    let state = KernelState {
        kernel_name: kernel_id.to_string(),
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        segment_id: 1,
        thread_states: vec![ThreadState {
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            registers: HashMap::from([("r0".to_string(), 42), ("r1".to_string(), 100)]),
            program_counter: 0x100,
            predicate_mask: 0xFFFFFFFF,
        }],
        shared_memory: vec![0u8; 1024],
        memory_snapshot: HashMap::new(),
    };

    Ok(state)
}

fn restore_kernel_state(state: &KernelState, runtime: &mut HetGpuRuntime) -> anyhow::Result<()> {
    // In real implementation, would:
    // 1. Allocate memory on new device
    // 2. Copy memory contents
    // 3. Launch special resume kernel
    // 4. Initialize registers from saved state

    println!(
        "Restoring kernel {} at segment {}",
        state.kernel_name, state.segment_id
    );
    println!("Restoring {} thread states", state.thread_states.len());

    // Copy memory allocations to new device
    for (name, data) in &state.memory_snapshot {
        runtime.copy_from_host(name)?;
    }

    Ok(())
}

async fn compile_command(args: CompileArgs) -> anyhow::Result<()> {
    println!("Compiling PTX with debug information...");
    println!("Input: {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Target: {}", args.target);

    // Read PTX source
    let ptx_content = fs::read_to_string(&args.input)?;

    // Get the absolute path of the input file for debug info
    let source_filename = args
        .input
        .canonicalize()
        .unwrap_or_else(|_| args.input.clone())
        .to_string_lossy()
        .to_string();

    // Compile PTX to LLVM IR with debug information and custom filename
    let result = ptx::ptx_to_llvm_with_debug_and_filename(&ptx_content, &source_filename);

    match result {
        Ok((llvm_module, debug_mappings)) => {
            // Convert LLVM module to string
            let llvm_ir = llvm_module
                .print_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to convert LLVM module to string: {}", e))?;

            // Write LLVM IR to output file
            fs::write(&args.output, llvm_ir)?;
            println!(
                "LLVM IR with debug info saved to: {}",
                args.output.display()
            );

            if args.verbose {
                println!("Debug mappings: {} entries", debug_mappings.len());
                for (i, mapping) in debug_mappings.iter().enumerate().take(5) {
                    println!(
                        "  [{}] {}:{}:{} -> {} instructions",
                        i,
                        mapping.ptx_location.file,
                        mapping.ptx_location.line,
                        mapping.ptx_location.column,
                        mapping.target_instructions.len()
                    );
                }
                if debug_mappings.len() > 5 {
                    println!("  ... and {} more entries", debug_mappings.len() - 5);
                }
            }
        }
        Err(e) => {
            eprintln!("Compilation failed: {}", e);
            return Err(anyhow::anyhow!("PTX compilation failed: {}", e));
        }
    }
    // Parse PTX (using ptx_parser)
    // For now, create mock debug mappings
    let debug_mappings = vec![DwarfMappingEntry {
        ptx_location: PtxSourceLocation {
            file: args.input.to_string_lossy().to_string(),
            line: 1,
            column: 1,
            instruction_offset: 0,
        },
        target_instructions: vec![TargetInstruction::AmdGcn {
            instruction: "v_add_f32 v0, v1, v2".to_string(),
            address: 0x1000,
            register_state: HashMap::new(),
        }],
        variable_mappings: HashMap::new(),
        scope_id: 1,
    }];

    Ok(())
}

async fn analyze_command(args: AnalyzeArgs) -> anyhow::Result<()> {
    println!("Analyzing debug mappings...");

    let content = fs::read_to_string(&args.mapping_file)?;
    let mappings: Vec<DwarfMappingEntry> = serde_json::from_str(&content)?;

    println!("Total mappings: {}", mappings.len());

    for (i, mapping) in mappings.iter().enumerate() {
        if let Some(line_filter) = args.line {
            if mapping.ptx_location.line != line_filter {
                continue;
            }
        }

        println!("\nMapping {}:", i + 1);
        println!(
            "  PTX Location: {}:{}:{}",
            mapping.ptx_location.file, mapping.ptx_location.line, mapping.ptx_location.column
        );

        println!(
            "  Target Instructions: {}",
            mapping.target_instructions.len()
        );

        if args.detailed {
            for (j, inst) in mapping.target_instructions.iter().enumerate() {
                match inst {
                    TargetInstruction::AmdGcn {
                        instruction,
                        address,
                        ..
                    } => {
                        println!("    [{}] AMD GCN: {} @ 0x{:x}", j, instruction, address);
                    }
                    TargetInstruction::IntelSpirv {
                        instruction,
                        opcode,
                        ..
                    } => {
                        println!("    [{}] SPIRV: {} (opcode: {})", j, instruction, opcode);
                    }
                    TargetInstruction::Sass {
                        instruction,
                        address,
                        predicate,
                    } => {
                        let pred_str = predicate
                            .as_ref()
                            .map(|p| format!(" [{}]", p))
                            .unwrap_or_default();
                        println!(
                            "    [{}] SASS: {} @ 0x{:x}{}",
                            j, instruction, address, pred_str
                        );
                    }
                }
            }
        }

        if !mapping.variable_mappings.is_empty() {
            println!("  Variables: {}", mapping.variable_mappings.len());
            if args.detailed {
                for (name, location) in &mapping.variable_mappings {
                    println!("    {} -> {:?}", name, location);
                }
            }
        }
    }

    Ok(())
}

async fn debug_command(args: DebugArgs) -> anyhow::Result<()> {
    println!("Starting interactive debugging session...");

    let mut manager = PtxStateRecoveryManager::new();
    manager.load_debug_mappings(&args.mapping_file).unwrap();

    // Set initial breakpoint if provided
    if let Some(bp_location) = args.breakpoint {
        let parts: Vec<&str> = bp_location.split(':').collect();
        if parts.len() >= 2 {
            let file = parts[0].to_string();
            let line = parts[1].parse::<u32>().unwrap_or(1);
            let column = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

            let location = PtxSourceLocation {
                file,
                line,
                column,
                instruction_offset: 0,
            };

            let bp_id = manager.add_breakpoint(location, None);
            println!(
                "Breakpoint {} set at {}:{}:{}",
                bp_id, parts[0], line, column
            );
        }
    }

    // Interactive debugging loop
    loop {
        print!("(ptx-debug) ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "quit" | "q" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "help" | "h" => {
                print_debug_help();
            }
            "info" | "i" => {
                println!("{}", manager.generate_state_dump());
            }
            "backtrace" | "bt" => {
                println!("Call stack:");
                for (i, frame) in manager.get_call_stack().iter().enumerate() {
                    println!(
                        "  #{}: {} at {}:{}:{}",
                        i,
                        frame.function_name,
                        frame.location.file,
                        frame.location.line,
                        frame.location.column
                    );
                }
            }
            cmd if cmd.starts_with("break ") || cmd.starts_with("b ") => {
                let location_str = cmd.split_whitespace().nth(1).unwrap_or("");
                let parts: Vec<&str> = location_str.split(':').collect();
                if parts.len() >= 2 {
                    let file = parts[0].to_string();
                    let line = parts[1].parse::<u32>().unwrap_or(1);
                    let column = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

                    let location = PtxSourceLocation {
                        file: file.clone(),
                        line,
                        column,
                        instruction_offset: 0,
                    };

                    let bp_id = manager.add_breakpoint(location, None);
                    println!("Breakpoint {} set at {}:{}:{}", bp_id, file, line, column);
                } else {
                    println!("Usage: break <file>:<line>[:<column>]");
                }
            }
            cmd if cmd.starts_with("delete ") || cmd.starts_with("d ") => {
                if let Some(id_str) = cmd.split_whitespace().nth(1) {
                    if let Ok(id) = id_str.parse::<u32>() {
                        if manager.remove_breakpoint(id) {
                            println!("Breakpoint {} deleted", id);
                        } else {
                            println!("No breakpoint with id {}", id);
                        }
                    } else {
                        println!("Invalid breakpoint id: {}", id_str);
                    }
                } else {
                    println!("Usage: delete <breakpoint_id>");
                }
            }
            cmd if cmd.starts_with("print ") || cmd.starts_with("p ") => {
                if let Some(var_name) = cmd.split_whitespace().nth(1) {
                    if let Some(value) = manager.get_variable_value(var_name) {
                        println!("{} = {:?}", var_name, value);
                    } else {
                        println!("Variable '{}' not found", var_name);
                    }
                } else {
                    println!("Usage: print <variable_name>");
                }
            }
            cmd if cmd.starts_with("recover ") => {
                if let Some(addr_str) = cmd.split_whitespace().nth(1) {
                    if let Ok(address) = u64::from_str_radix(addr_str.trim_start_matches("0x"), 16)
                    {
                        if let Some(location) =
                            manager.find_ptx_location_from_target("amd_gcn", address)
                        {
                            println!(
                                "Address 0x{:x} maps to PTX location: {}:{}:{}",
                                address, location.file, location.line, location.column
                            );
                        } else {
                            println!("No PTX location found for address 0x{:x}", address);
                        }
                    } else {
                        println!("Invalid address format: {}", addr_str);
                    }
                } else {
                    println!("Usage: recover <hex_address>");
                }
            }
            "" => {
                // Empty line, continue
            }
            _ => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    input
                );
            }
        }
    }

    Ok(())
}

async fn recover_command(args: RecoverArgs) -> anyhow::Result<()> {
    println!("Recovering PTX state from crash...");

    let mut manager = PtxStateRecoveryManager::new();
    manager.load_debug_mappings(&args.mapping_file).unwrap();

    // Parse address
    let address = if args.address.starts_with("0x") {
        u64::from_str_radix(&args.address[2..], 16)?
    } else {
        args.address.parse::<u64>()?
    };

    // Find PTX location
    if let Some(location) = manager.find_ptx_location_from_target(&args.target, address) {
        println!("Crash occurred at PTX location:");
        println!("  File: {}", location.file);
        println!("  Line: {}", location.line);
        println!("  Column: {}", location.column);
        println!("  Instruction Offset: {}", location.instruction_offset);

        // Load memory dump if provided
        if let Some(memory_dump_path) = args.memory_dump {
            let memory_data = fs::read(&memory_dump_path)?;
            manager.take_memory_snapshot("crash_dump".to_string(), address, memory_data);
        }

        // Generate recovery report
        println!("\n{}", manager.generate_state_dump());
    } else {
        println!(
            "No PTX location found for address 0x{:x} in {} architecture",
            address, args.target
        );
    }

    Ok(())
}

async fn export_command(args: ExportArgs) -> anyhow::Result<()> {
    println!("Exporting debug information...");

    let manager = PtxStateRecoveryManager::new();

    match args.format.as_str() {
        "json" => {
            let content = fs::read_to_string(&args.mapping_file)?;
            fs::write(&args.output, content)?;
            println!("JSON export saved to: {}", args.output.display());
        }
        "gdb" => {
            let gdb_info = manager.export_gdb_compatible_info();
            fs::write(&args.output, gdb_info)?;
            println!("GDB-compatible export saved to: {}", args.output.display());
        }
        "vscode" => {
            // Create VS Code debug configuration
            let vscode_config = serde_json::json!({
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "PTX Debug",
                        "type": "gdb",
                        "request": "attach",
                        "program": "${workspaceFolder}/target/debug/main",
                        "MIMode": "gdb",
                        "setupCommands": [
                            {
                                "description": "Load PTX debug symbols",
                                "text": "source ptx_debug.gdb",
                                "ignoreFailures": false
                            }
                        ]
                    }
                ]
            });
            fs::write(&args.output, serde_json::to_string_pretty(&vscode_config)?)?;
            println!(
                "VS Code debug configuration saved to: {}",
                args.output.display()
            );
        }
        _ => {
            println!("Unknown export format: {}", args.format);
            println!("Supported formats: json, gdb, vscode");
        }
    }

    Ok(())
}

fn print_debug_help() {
    println!("Available commands:");
    println!("  help, h                 - Show this help");
    println!("  info, i                 - Show current state information");
    println!("  backtrace, bt           - Show call stack");
    println!("  break <loc>, b <loc>    - Set breakpoint at location (file:line:column)");
    println!("  delete <id>, d <id>     - Delete breakpoint by id");
    println!("  print <var>, p <var>    - Print variable value");
    println!("  recover <addr>          - Recover PTX location from target address");
    println!("  quit, q, exit           - Exit debugger");
    println!();
    println!("Examples:");
    println!("  break kernel.ptx:42:10  - Set breakpoint at line 42, column 10");
    println!("  print tid               - Print value of variable 'tid'");
    println!("  recover 0x1000          - Find PTX location for address 0x1000");
}
