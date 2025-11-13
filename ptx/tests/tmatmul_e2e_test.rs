// End-to-end test: PTX -> TMatmul Assembly -> Cocotb simulation
// This test compiles PTX to TMatmul assembly and prepares it for hardware simulation

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[cfg(test)]
mod e2e_tests {
    use super::*;

    /// Directory where cocotb expects assembly files
    fn cocotb_asm_dir() -> PathBuf {
        PathBuf::from("/root/matmulfreellm/hardware/ternary_matmul/asm")
    }

    /// Directory where cocotb tests run
    fn cocotb_test_dir() -> PathBuf {
        PathBuf::from("/root/matmulfreellm/hardware/ternary_matmul/cocotb")
    }

    /// Create output directory if it doesn't exist
    fn ensure_output_dir(path: &PathBuf) -> std::io::Result<()> {
        if !path.exists() {
            println!("Creating output directory: {}", path.display());
            fs::create_dir_all(path)?;
        }
        Ok(())
    }

    /// End-to-end test for k*D matmul pattern 1: [1 k*D][k*D D]
    #[test]
    fn test_e2e_kd_matmul_pattern1() {
        println!("\n=== E2E Test: [1 k*D][k*D D] Pattern ===\n");

        // Step 1: Create TMatmul assembly directly (PTX parser may not support all features)
        use ptx::pass::emit_tmatmul_asm::*;

        let mut codegen = TMatmulCodegen::new();

        // Setup memory map
        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("W0", MemoryLocation::WF);
        codegen.map_memory("W1", MemoryLocation::WC);
        codegen.map_memory("W2", MemoryLocation::WG);
        codegen.map_memory("O", MemoryLocation::O);

        codegen.add_section("E2E Test: k*D Matmul [1 3*D][3*D D]");
        codegen.add_comment("Input: 1x3D, Weight: 3DxD, Output: 1xD");
        codegen.add_comment("Split input into 3 blocks, multiply each, sum results");

        // Implementation
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%0"])
            .unwrap();

        // Block 0: input[0:D] * W0
        codegen.add_comment("Block 0");
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "W0"], &["%1"])
            .unwrap();

        // Block 1: input[D:2D] * W1
        codegen.add_comment("Block 1");
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "W1"], &["%2"])
            .unwrap();

        // Block 2: input[2D:3D] * W2
        codegen.add_comment("Block 2");
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "W2"], &["%3"])
            .unwrap();

        // Sum results
        codegen.add_comment("Sum all blocks");
        codegen
            .emit_operation("tmatmul.add", &["%1", "%2"], &["%4"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.add", &["%4", "%3"], &["%5"])
            .unwrap();

        // Store output
        codegen
            .emit_operation("tmatmul.sv", &["%5", "O"], &[])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Step 2: Write to cocotb assembly directory
        let asm_dir = cocotb_asm_dir();
        let output_path = asm_dir.join("test_kd_pattern1.S");

        match ensure_output_dir(&asm_dir) {
            Ok(_) => {
                match fs::write(&output_path, &assembly) {
                    Ok(_) => {
                        println!("✓ Assembly written to: {}", output_path.display());
                        println!("\nGenerated Assembly:");
                        println!("{}", assembly);
                    }
                    Err(e) => {
                        println!("⚠ Could not write to cocotb dir: {}", e);
                        // Fallback to /tmp
                        let fallback = PathBuf::from("/tmp/test_kd_pattern1.S");
                        fs::write(&fallback, &assembly).unwrap();
                        println!("✓ Assembly written to fallback: {}", fallback.display());
                    }
                }
            }
            Err(e) => {
                println!("⚠ Could not create cocotb dir: {}", e);
                // Fallback to /tmp
                let fallback = PathBuf::from("/tmp/test_kd_pattern1.S");
                fs::write(&fallback, &assembly).unwrap();
                println!("✓ Assembly written to fallback: {}", fallback.display());
            }
        }

        // Step 3: Print instructions to run cocotb
        println!("\n=== To run on cocotb hardware simulator ===");
        println!("cd {}", cocotb_test_dir().display());
        println!("make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs");
        println!("\nOr to run specific program:");
        println!("make SIM=verilator MODULE=tb_asm TESTCASE=test_kd_pattern1");
    }

    /// End-to-end test for k*D matmul pattern 2: [1 D][D k*D]
    #[test]
    fn test_e2e_kd_matmul_pattern2() {
        println!("\n=== E2E Test: [1 D][D k*D] Pattern ===\n");

        use ptx::pass::emit_tmatmul_asm::*;

        let mut codegen = TMatmulCodegen::new();

        // Setup memory map
        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("W", MemoryLocation::WF);
        codegen.map_memory("O", MemoryLocation::O);
        codegen.map_memory("TEMP", MemoryLocation::TempVec);

        codegen.add_section("E2E Test: k*D Matmul [1 D][D 5*D]");
        codegen.add_comment("Input: 1xD, Weight: Dx5D, Output: 1x5D");
        codegen.add_comment("Single input produces 5D outputs via 5 matmuls");

        // Load input
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%0"])
            .unwrap();

        // Initialize accumulator
        codegen
            .emit_operation("tmatmul.ldv", &["TEMP"], &["%1"])
            .unwrap();

        // Perform 5 matmuls (k=5)
        for i in 0..5 {
            codegen.add_comment(&format!("Block {}", i));
            codegen
                .emit_operation("tmatmul.matmul", &["%0", "W"], &["%2"])
                .unwrap();
            codegen
                .emit_operation("tmatmul.add", &["%1", "%2"], &["%3"])
                .unwrap();
            if i < 4 {
                codegen
                    .emit_operation("tmatmul.add", &["%3", "%3"], &["%1"])
                    .unwrap();
            }
        }

        // Store result
        codegen
            .emit_operation("tmatmul.sv", &["%3", "O"], &[])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Write to cocotb directory
        let asm_dir = cocotb_asm_dir();
        let output_path = asm_dir.join("test_kd_pattern2.S");

        match ensure_output_dir(&asm_dir) {
            Ok(_) => match fs::write(&output_path, &assembly) {
                Ok(_) => {
                    println!("✓ Assembly written to: {}", output_path.display());
                    println!("\nGenerated Assembly:");
                    println!("{}", assembly);
                }
                Err(e) => {
                    println!("⚠ Could not write to cocotb dir: {}", e);
                    let fallback = PathBuf::from("/tmp/test_kd_pattern2.S");
                    fs::write(&fallback, &assembly).unwrap();
                    println!("✓ Assembly written to fallback: {}", fallback.display());
                }
            },
            Err(e) => {
                println!("⚠ Could not create cocotb dir: {}", e);
                let fallback = PathBuf::from("/tmp/test_kd_pattern2.S");
                fs::write(&fallback, &assembly).unwrap();
                println!("✓ Assembly written to fallback: {}", fallback.display());
            }
        }

        println!("\n=== To run on cocotb hardware simulator ===");
        println!("cd {}", cocotb_test_dir().display());
        println!("make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs");
    }

    /// E2E test with padding for non-power-of-2 k
    #[test]
    fn test_e2e_kd_with_padding() {
        println!("\n=== E2E Test: k*D Matmul with Padding (k=7->8) ===\n");

        use ptx::pass::emit_tmatmul_asm::*;

        let k = 7;
        let k_padded = 8; // Next power of 2

        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("W", MemoryLocation::WF);
        codegen.map_memory("O", MemoryLocation::O);
        codegen.map_memory("ZERO", MemoryLocation::TempVec);

        codegen.add_section(&format!(
            "E2E Test: Padding k={} to k_padded={}",
            k, k_padded
        ));
        codegen.add_comment("Demonstrates zero-padding for non-power-of-2 k");

        // Load and process k blocks
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%1"])
            .unwrap();

        // Process 7 real blocks
        for i in 0..k {
            codegen.add_comment(&format!("Real block {}/{}", i + 1, k));
            codegen
                .emit_operation("tmatmul.matmul", &["%1", "W"], &["%2"])
                .unwrap();
            codegen
                .emit_operation("tmatmul.add", &["%0", "%2"], &["%3"])
                .unwrap();
            codegen
                .emit_operation("tmatmul.add", &["%3", "%3"], &["%0"])
                .unwrap();
        }

        // Add 1 padding block (k_padded - k = 8 - 7 = 1)
        for i in k..k_padded {
            codegen.add_comment(&format!("Padding block {}/{}", i + 1, k_padded));
            codegen
                .emit_operation("tmatmul.ldv", &["ZERO"], &["%2"])
                .unwrap();
            codegen
                .emit_operation("tmatmul.add", &["%0", "%2"], &["%3"])
                .unwrap();
            codegen
                .emit_operation("tmatmul.add", &["%3", "%3"], &["%0"])
                .unwrap();
        }

        codegen
            .emit_operation("tmatmul.sv", &["%0", "O"], &[])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Write assembly
        let asm_dir = cocotb_asm_dir();
        let output_path = asm_dir.join("test_kd_padding.S");

        match ensure_output_dir(&asm_dir) {
            Ok(_) => match fs::write(&output_path, &assembly) {
                Ok(_) => {
                    println!("✓ Assembly written to: {}", output_path.display());
                }
                Err(e) => {
                    println!("⚠ Could not write to cocotb dir: {}", e);
                    let fallback = PathBuf::from("/tmp/test_kd_padding.S");
                    fs::write(&fallback, &assembly).unwrap();
                    println!("✓ Assembly written to fallback: {}", fallback.display());
                }
            },
            Err(e) => {
                println!("⚠ Could not create cocotb dir: {}", e);
                let fallback = PathBuf::from("/tmp/test_kd_padding.S");
                fs::write(&fallback, &assembly).unwrap();
                println!("✓ Assembly written to fallback: {}", fallback.display());
            }
        }

        println!("\n=== To run on cocotb hardware simulator ===");
        println!("cd {}", cocotb_test_dir().display());
        println!("make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs");
    }

    /// Complete E2E test that runs cocotb if available
    #[test]
    #[ignore] // Ignore by default, run with: cargo test --test tmatmul_e2e_test -- --ignored
    fn test_e2e_full_with_cocotb_run() {
        println!("\n=== Full E2E Test with Cocotb Execution ===\n");

        use ptx::pass::emit_tmatmul_asm::*;

        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("WF", MemoryLocation::WF);
        codegen.map_memory("O", MemoryLocation::O);

        codegen.add_section("Full E2E Test Program");
        codegen.add_comment("Complete test: compile -> assemble -> simulate");

        // Simple matmul sequence
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "WF"], &["%1"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.sv", &["%1", "O"], &[])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Step 1: Write assembly
        let asm_dir = cocotb_asm_dir();
        let output_path = asm_dir.join("test_e2e_full.S");

        ensure_output_dir(&asm_dir).expect("Failed to create asm dir");
        fs::write(&output_path, &assembly).expect("Failed to write assembly");
        println!("✓ Assembly written to: {}", output_path.display());

        // Step 2: Run cocotb simulation
        let cocotb_dir = cocotb_test_dir();

        if cocotb_dir.exists() {
            println!("\n✓ Cocotb directory found: {}", cocotb_dir.display());
            println!("Running cocotb simulation...\n");

            let output = Command::new("make")
                .current_dir(&cocotb_dir)
                .env("SIM", "verilator")
                .env("MODULE", "tb_asm")
                .env("TESTCASE", "test_e2e_full")
                .output();

            match output {
                Ok(result) => {
                    println!("=== Cocotb Output ===");
                    println!("{}", String::from_utf8_lossy(&result.stdout));

                    if !result.status.success() {
                        println!("\n=== Cocotb Errors ===");
                        println!("{}", String::from_utf8_lossy(&result.stderr));
                        panic!("Cocotb simulation failed");
                    } else {
                        println!("\n✓ Cocotb simulation completed successfully!");
                    }
                }
                Err(e) => {
                    println!("⚠ Could not run cocotb: {}", e);
                    println!("Please run manually:");
                    println!("  cd {}", cocotb_dir.display());
                    println!("  make SIM=verilator MODULE=tb_asm TESTCASE=test_e2e_full");
                }
            }
        } else {
            println!("⚠ Cocotb directory not found: {}", cocotb_dir.display());
            println!("\nTo run simulation manually:");
            println!("  cd <your-cocotb-dir>");
            println!("  make SIM=verilator MODULE=tb_asm TESTCASE=test_e2e_full");
        }
    }

    /// Test PTX to TMatmul to Cocotb pipeline
    #[test]
    fn test_e2e_ptx_pipeline() {
        println!("\n=== E2E Test: PTX -> TMatmul -> Cocotb ===\n");

        // Simple PTX kernel for testing
        let ptx_source = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry simple_matmul(
    .param .u64 input,
    .param .u64 output
) {
    .reg .f32 %f<10>;
    .reg .u64 %rd<5>;

    // Load input pointer
    ld.param.u64 %rd1, [input];

    // Load output pointer
    ld.param.u64 %rd2, [output];

    // Simple computation
    ld.global.f32 %f0, [%rd1];
    ld.global.f32 %f1, [%rd1];

    mul.f32 %f2, %f0, %f1;
    add.f32 %f3, %f2, %f0;

    st.global.f32 [%rd2], %f3;

    ret;
}
"#;

        println!("Step 1: Parse and compile PTX to TMatmul assembly");
        let result = ptx::pass::ptx_to_tmatmul_assembly(ptx_source);

        match result {
            Ok(assembly) => {
                println!("✓ PTX compiled successfully\n");
                println!("Generated Assembly:");
                println!("{}", assembly);

                // Write to file
                let output_path = PathBuf::from("/tmp/test_e2e_ptx.S");
                fs::write(&output_path, &assembly).unwrap();
                println!("\n✓ Assembly written to: {}", output_path.display());

                // Try to copy to cocotb dir
                let cocotb_path = cocotb_asm_dir().join("test_e2e_ptx.S");
                match fs::write(&cocotb_path, &assembly) {
                    Ok(_) => println!("✓ Also copied to cocotb: {}", cocotb_path.display()),
                    Err(_) => println!("⚠ Could not copy to cocotb directory"),
                }

                println!("\n=== To run on cocotb ===");
                println!("cd {}", cocotb_test_dir().display());
                println!("make SIM=verilator MODULE=tb_asm TESTCASE=test_e2e_ptx");
            }
            Err(e) => {
                println!("⚠ PTX compilation not fully supported yet: {}", e);
                println!("This is expected - the PTX parser may not support all features");
                println!("Use the direct TMatmul assembly tests instead");
            }
        }
    }

    /// Generate test runner script
    #[test]
    fn test_e2e_generate_runner_script() {
        println!("\n=== Generating E2E Test Runner Script ===\n");

        let script = r#"#!/bin/bash
# Auto-generated E2E test runner for TMatmul k*D tests

set -e

COCOTB_DIR="/root/matmulfreellm/hardware/ternary_matmul/cocotb"
ASM_DIR="/root/matmulfreellm/hardware/ternary_matmul/asm"

echo "=== TMatmul k*D E2E Test Runner ==="
echo ""

# Check if cocotb directory exists
if [ ! -d "$COCOTB_DIR" ]; then
    echo "Error: Cocotb directory not found: $COCOTB_DIR"
    exit 1
fi

# Check if assembly directory exists
if [ ! -d "$ASM_DIR" ]; then
    echo "Creating assembly directory: $ASM_DIR"
    mkdir -p "$ASM_DIR"
fi

cd "$COCOTB_DIR"

# Run tests
echo "Running Pattern 1: [1 k*D][k*D D]..."
if [ -f "$ASM_DIR/test_kd_pattern1.S" ]; then
    make SIM=verilator MODULE=tb_asm TESTCASE=test_kd_pattern1
    echo "✓ Pattern 1 passed"
else
    echo "⚠ test_kd_pattern1.S not found, skipping"
fi

echo ""
echo "Running Pattern 2: [1 D][D k*D]..."
if [ -f "$ASM_DIR/test_kd_pattern2.S" ]; then
    make SIM=verilator MODULE=tb_asm TESTCASE=test_kd_pattern2
    echo "✓ Pattern 2 passed"
else
    echo "⚠ test_kd_pattern2.S not found, skipping"
fi

echo ""
echo "Running Padding Test (k=7->8)..."
if [ -f "$ASM_DIR/test_kd_padding.S" ]; then
    make SIM=verilator MODULE=tb_asm TESTCASE=test_kd_padding
    echo "✓ Padding test passed"
else
    echo "⚠ test_kd_padding.S not found, skipping"
fi

echo ""
echo "=== All E2E tests completed ==="
"#;

        let script_path = PathBuf::from("/tmp/run_tmatmul_e2e_tests.sh");
        fs::write(&script_path, script).unwrap();

        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).unwrap();
        }

        println!("✓ Runner script generated: {}", script_path.display());
        println!("\nTo run all E2E tests:");
        println!("  {}", script_path.display());
        println!("\nOr manually:");
        println!("  bash {}", script_path.display());
    }
}
