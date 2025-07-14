use std::env;
use std::process;
use std::path::Path;
use gemmini_runtime_sys::{Device, CoreCoord};
use regex::Regex;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <mlir_file> [input_data] [output_data]", args[0]);
        eprintln!("  mlir_file:   Path to the MLIR file to execute on Gemmini");
        eprintln!("  input_data:  Optional input data file (binary format)");
        eprintln!("  output_data: Optional output data file (binary format)");
        process::exit(1);
    }
    
    let mlir_file = &args[1];
    let input_file = args.get(2);
    let output_file = args.get(3);
    
    println!("Gemmini MLIR Runner");
    println!("MLIR file: {}", mlir_file);
    
    if !Path::new(mlir_file).exists() {
        eprintln!("Error: MLIR file '{}' does not exist", mlir_file);
        process::exit(1);
    }
    
    // Initialize Gemmini device
    let device = match Device::new(0) {
        Ok(dev) => {
            println!("✓ Gemmini device initialized");
            dev
        }
        Err(e) => {
            eprintln!("✗ Failed to initialize Gemmini device: {}", e);
            process::exit(1);
        }
    };
    
    // Get device info
    match device.get_name() {
        Ok(name) => println!("Device: {}", name),
        Err(e) => eprintln!("Warning: Could not get device name: {}", e),
    }
    
    // Create program
    let program = match device.create_program() {
        Ok(prog) => {
            println!("✓ Program created");
            prog
        }
        Err(e) => {
            eprintln!("✗ Failed to create program: {}", e);
            process::exit(1);
        }
    };
    
    // Read MLIR file
    let mlir_content = match std::fs::read_to_string(mlir_file) {
        Ok(content) => {
            println!("✓ MLIR file loaded ({} bytes)", content.len());
            content
        }
        Err(e) => {
            eprintln!("✗ Failed to read MLIR file: {}", e);
            process::exit(1);
        }
    };
    
    // Extract kernel name from MLIR
    let kernel_name = match extract_kernel_name(&mlir_content) {
        Ok(name) => {
            println!("✓ Found kernel: {}", name);
            name
        }
        Err(e) => {
            eprintln!("✗ Failed to extract kernel name: {}", e);
            process::exit(1);
        }
    };

    // Load MLIR into program
    if let Err(e) = program.load_from_mlir(&mlir_content) {
        eprintln!("✗ Failed to load MLIR: {}", e);
        process::exit(1);
    }
    println!("✓ MLIR loaded into program");
    
    // Create input/output buffers
    let buffer_size = 32 * 32 * 4; // 32x32 float32 matrix
    
    let input_buffer = match device.create_buffer(buffer_size) {
        Ok(buf) => {
            println!("✓ Input buffer created ({} bytes)", buffer_size);
            buf
        }
        Err(e) => {
            eprintln!("✗ Failed to create input buffer: {}", e);
            process::exit(1);
        }
    };
    
    let output_buffer = match device.create_buffer(buffer_size) {
        Ok(buf) => {
            println!("✓ Output buffer created ({} bytes)", buffer_size);
            buf
        }
        Err(e) => {
            eprintln!("✗ Failed to create output buffer: {}", e);
            process::exit(1);
        }
    };
    
    // Initialize input buffer
    let mut input_data = vec![0u8; buffer_size as usize];
    
    if let Some(input_path) = input_file {
        // Read input from file
        match std::fs::read(input_path) {
            Ok(file_data) => {
                let copy_size = std::cmp::min(file_data.len(), input_data.len());
                input_data[..copy_size].copy_from_slice(&file_data[..copy_size]);
                println!("✓ Input data loaded from {} ({} bytes)", input_path, copy_size);
            }
            Err(e) => {
                eprintln!("Warning: Could not read input file '{}': {}", input_path, e);
                // Fall back to test pattern
                initialize_test_pattern(&mut input_data);
                println!("✓ Using test pattern as input");
            }
        }
    } else {
        // Generate test pattern
        initialize_test_pattern(&mut input_data);
        println!("✓ Using test pattern as input");
    }
    
    // Write data to input buffer
    if let Err(e) = input_buffer.write(&input_data) {
        eprintln!("✗ Failed to write to input buffer: {}", e);
        process::exit(1);
    }
    println!("✓ Input data written to buffer");
    
    // Create kernel and set runtime arguments
    let core = CoreCoord { x: 0, y: 0 };
    let kernel = match program.create_kernel(&kernel_name, core) {
        Ok(k) => {
            println!("✓ Kernel created");
            k
        }
        Err(e) => {
            eprintln!("✗ Failed to create kernel: {}", e);
            process::exit(1);
        }
    };
    
    // Set runtime arguments
    if let Err(e) = program.set_runtime_args(&kernel_name, &[&input_buffer, &output_buffer]) {
        eprintln!("✗ Failed to set runtime args: {}", e);
        process::exit(1);
    }
    println!("✓ Runtime arguments set");
    
    // Launch the program
    println!("▶ Launching Gemmini program...");
    if let Err(e) = program.launch(&device) {
        eprintln!("✗ Failed to launch program: {}", e);
        process::exit(1);
    }
    
    // Wait for completion
    if let Err(e) = program.wait_for_completion() {
        eprintln!("✗ Failed to wait for completion: {}", e);
        process::exit(1);
    }
    println!("✓ Program execution completed");
    
    // Read output data
    let mut output_data = vec![0u8; buffer_size as usize];
    if let Err(e) = output_buffer.read(&mut output_data) {
        eprintln!("✗ Failed to read output buffer: {}", e);
        process::exit(1);
    }
    println!("✓ Output data read from buffer");
    
    // Save output if requested
    if let Some(output_path) = output_file {
        if let Err(e) = std::fs::write(output_path, &output_data) {
            eprintln!("✗ Failed to write output file '{}': {}", output_path, e);
            process::exit(1);
        }
        println!("✓ Output saved to {}", output_path);
    }
    
    // Display sample output values
    print_sample_output(&output_data);
    
    println!("🎉 Gemmini execution completed successfully!");
}

fn initialize_test_pattern(data: &mut [u8]) {
    // Initialize as float32 test pattern
    let float_data = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len() / 4)
    };
    
    for (i, val) in float_data.iter_mut().enumerate() {
        *val = (i % 100) as f32 / 10.0;
    }
}

fn extract_kernel_name(mlir_content: &str) -> Result<String, String> {
    // Parse MLIR to find func.func definitions
    let func_regex = Regex::new(r"func\.func\s+@([a-zA-Z_][a-zA-Z0-9_]*)")
        .map_err(|e| format!("Regex compilation error: {}", e))?;
    
    let mut kernel_names = Vec::new();
    
    for cap in func_regex.captures_iter(mlir_content) {
        if let Some(func_name) = cap.get(1) {
            kernel_names.push(func_name.as_str().to_string());
        }
    }
    
    match kernel_names.len() {
        0 => Err("No function definitions found in MLIR file".to_string()),
        1 => Ok(kernel_names[0].clone()),
        _ => {
            println!("Multiple functions found: {:?}", kernel_names);
            println!("Using the first function: {}", kernel_names[0]);
            Ok(kernel_names[0].clone())
        }
    }
}

fn print_sample_output(data: &[u8]) {
    // Display first few values as float32
    let float_data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, std::cmp::min(8, data.len() / 4))
    };
    
    print!("Output sample: [");
    for (i, val) in float_data.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.2}", val);
    }
    println!("]");
}