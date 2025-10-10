// ptx_to_tmatmul_example.rs - Complete PTX to TMatmul compilation example

use std::fs;

fn main() {
    println!("PTX to TMatmul Complete Compilation Example");
    println!("============================================\n");

    // Example 1: Simple PTX kernel
    example_simple_kernel();

    // Example 2: Vector addition kernel
    example_vector_add();

    // Example 3: Matrix multiply hint
    example_matmul_hint();

    // Example 4: Reading from file
    if let Some(filename) = std::env::args().nth(1) {
        example_from_file(&filename);
    }
}

fn example_simple_kernel() {
    println!("Example 1: Simple PTX Kernel");
    println!("-----------------------------\n");

    let ptx_source = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry simple_kernel(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .f32 %r1, %r2, %r3;

    // Load from input
    ld.param.u64 %r1, [input];

    // Perform computation
    add.f32 %r2, %r1, %r1;
    mul.f32 %r3, %r2, %r1;

    // Store to output
    st.param.u64 [output], %r3;

    ret;
}
"#;

    match ptx::pass::ptx_to_tmatmul_assembly(ptx_source) {
        Ok(assembly) => {
            println!("Generated TMatmul Assembly:");
            println!("{}\n", assembly);

            // Save to file
            if let Err(e) = fs::write("/tmp/simple_kernel.tmatmul", &assembly) {
                eprintln!("Warning: Could not save assembly: {}", e);
            } else {
                println!("✓ Saved to /tmp/simple_kernel.tmatmul\n");
            }
        }
        Err(e) => {
            eprintln!("Compilation error: {}\n", e);
        }
    }
}

fn example_vector_add() {
    println!("\nExample 2: Vector Addition Kernel");
    println!("-----------------------------------\n");

    let ptx_source = r#"
.version 7.0
.target sm_80

.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .f32 %f<10>;

    // Load from a
    ld.global.f32 %f1, [a];

    // Load from b
    ld.global.f32 %f2, [b];

    // Add
    add.f32 %f3, %f1, %f2;

    // Store to c
    st.global.f32 [c], %f3;

    ret;
}
"#;

    match ptx::pass::ptx_to_tmatmul_assembly(ptx_source) {
        Ok(assembly) => {
            println!("Generated TMatmul Assembly:");
            println!("{}\n", assembly);

            fs::write("/tmp/vector_add.tmatmul", &assembly).ok();
            println!("✓ Saved to /tmp/vector_add.tmatmul\n");
        }
        Err(e) => {
            eprintln!("Compilation error: {}\n", e);
        }
    }
}

fn example_matmul_hint() {
    println!("\nExample 3: Matrix Multiply Pattern");
    println!("------------------------------------\n");

    let ptx_source = r#"
.version 7.0
.target sm_80

.visible .entry matmul_kernel(
    .param .u64 weight_matrix,
    .param .u64 input_vector,
    .param .u64 output_vector
)
{
    .reg .f32 %f<20>;

    // This pattern hints at matrix multiplication
    // Load input vector
    ld.global.f32 %f1, [input_vector];

    // Multiple MAD operations (matrix-vector multiply pattern)
    mad.f32 %f2, %f1, %f1, 0.0;
    mad.f32 %f3, %f2, %f1, %f2;
    mad.f32 %f4, %f3, %f1, %f3;

    // Store result
    st.global.f32 [output_vector], %f4;

    ret;
}
"#;

    match ptx::pass::ptx_to_tmatmul_assembly(ptx_source) {
        Ok(assembly) => {
            println!("Generated TMatmul Assembly:");
            println!("{}\n", assembly);

            // Check if tmatmul operations were generated
            if assembly.contains("tmatmul") {
                println!("✓ TMatmul accelerator operations detected!");
            } else {
                println!("ℹ Note: Pattern-based optimization could map this to tmatmul");
            }

            fs::write("/tmp/matmul.tmatmul", &assembly).ok();
            println!("\n✓ Saved to /tmp/matmul.tmatmul\n");
        }
        Err(e) => {
            eprintln!("Compilation error: {}\n", e);
        }
    }
}

fn example_from_file(filename: &str) {
    println!("\nExample 4: Compiling from file: {}", filename);
    println!("-------------------------------------------\n");

    match fs::read_to_string(filename) {
        Ok(ptx_source) => {
            println!("Read {} bytes of PTX source\n", ptx_source.len());

            match ptx::pass::ptx_to_tmatmul_assembly(&ptx_source) {
                Ok(assembly) => {
                    println!("Generated TMatmul Assembly:");
                    println!("{}\n", assembly);

                    let output_filename = filename.replace(".ptx", ".tmatmul");
                    if let Err(e) = fs::write(&output_filename, &assembly) {
                        eprintln!("Warning: Could not save assembly: {}", e);
                    } else {
                        println!("✓ Saved to {}\n", output_filename);
                    }
                }
                Err(e) => {
                    eprintln!("Compilation error: {}\n", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error reading file: {}\n", e);
            println!("Usage: cargo run --example ptx_to_tmatmul_example <file.ptx>");
        }
    }
}