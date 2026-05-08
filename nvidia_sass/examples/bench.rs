// bench.rs - Benchmark for nvidia_sass SM120 PTX assembler pipeline
//
// Benchmarks each stage:
// 1. Instruction selection (isel)
// 2. Register allocation
// 3. Instruction scheduling
// 4. Instruction encoding
// 5. CUBIN ELF generation
// 6. Full pipeline (isel → regalloc → schedule → encode → CUBIN)
// 7. Scaling: 1, 10, 100, 1000 kernel instructions
//
// Usage:
//   cargo run -p nvidia_sass --example bench

use nvidia_sass::cubin_builder;
use nvidia_sass::encoding;
use nvidia_sass::isel;
use nvidia_sass::regalloc;
use nvidia_sass::scheduler;
use nvidia_sass::types::*;
use std::time::Instant;

fn main() {
    println!("=== nvidia_sass SM120 Assembler Benchmark ===\n");

    let iterations = 1000;

    // Build a representative vector_add kernel (virtual regs)
    let virtual_insts = build_vector_add_virtual();
    println!(
        "Kernel: vector_add ({} instructions, {} iterations per bench)\n",
        virtual_insts.len(),
        iterations
    );

    // --- Bench 1: Instruction selection ---
    bench_isel(iterations);

    // --- Bench 2: Register allocation ---
    let (physical_insts, num_regs) = bench_regalloc(&virtual_insts, iterations);

    // --- Bench 3: Instruction scheduling ---
    let scheduled_insts = bench_scheduler(&physical_insts, iterations);

    // --- Bench 4: Instruction encoding ---
    bench_encoding(&scheduled_insts, iterations);

    // --- Bench 5: CUBIN generation ---
    bench_cubin_generation(&scheduled_insts, num_regs, iterations);

    // --- Bench 6: Full pipeline ---
    bench_full_pipeline(iterations);

    // --- Bench 7: Scaling ---
    println!("\n--- Scaling (full pipeline) ---");
    for n_insts in [10, 50, 100, 500, 1000] {
        bench_scaling(n_insts, 100);
    }

    println!("\n=== Benchmark complete ===");
}

/// Build a vector_add kernel using virtual registers.
fn build_vector_add_virtual() -> Vec<SassInst> {
    vec![
        isel::select_special_reg(200, SpecialReg::TidX),
        isel::select_load_global(201, 200, 0),
        isel::select_load_global(202, 200, 4),
        isel::select_add_f32(203, 201, 202),
        isel::select_store_global(200, 8, 203),
        isel::select_exit(),
    ]
}

/// Build a larger kernel with N instructions for scaling tests.
fn build_scaled_kernel(n: usize) -> Vec<SassInst> {
    let mut insts = Vec::with_capacity(n + 2);

    // Prologue: read tid
    insts.push(isel::select_special_reg(200, SpecialReg::TidX));

    // Body: chain of FADD instructions
    insts.push(isel::select_load_global(201, 200, 0));
    for i in 0..n.saturating_sub(3) {
        let dst = (202 + i as u16).min(254) as u8;
        let src1 = (201 + i as u16).min(254) as u8;
        insts.push(SassInst {
            opcode: Opcode {
                mnemonic: "FADD",
                class: OpcodeClass::Alu3,
            },
            dst: Some(Reg::R(dst)),
            srcs: vec![Operand::Reg(Reg::R(src1)), Operand::Reg(Reg::R(200))],
            pred: None,
            modifiers: vec![],
            control: ControlCodes::default(),
        });
    }

    // Epilogue: store + exit
    insts.push(isel::select_store_global(200, 0, 201));
    insts.push(isel::select_exit());
    insts
}

fn bench_isel(iterations: usize) {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = build_vector_add_virtual();
    }
    let elapsed = start.elapsed();
    println!(
        "[1] Instruction selection (vector_add, {}x): {:.2}ms total, {:.2}us/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
}

fn bench_regalloc(virtual_insts: &[SassInst], iterations: usize) -> (Vec<SassInst>, u32) {
    let start = Instant::now();
    let mut result = (vec![], 0u32);
    for _ in 0..iterations {
        result = regalloc::allocate(virtual_insts).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "[2] Register allocation ({}x): {:.2}ms total, {:.2}us/iter, {} regs used",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
        result.1,
    );
    result
}

fn bench_scheduler(physical_insts: &[SassInst], iterations: usize) -> Vec<SassInst> {
    let start = Instant::now();
    let mut result = vec![];
    for _ in 0..iterations {
        result = scheduler::schedule(physical_insts);
    }
    let elapsed = start.elapsed();
    println!(
        "[3] Instruction scheduling ({}x): {:.2}ms total, {:.2}us/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
    result
}

fn bench_encoding(scheduled_insts: &[SassInst], iterations: usize) {
    let start = Instant::now();
    for _ in 0..iterations {
        for inst in scheduled_insts {
            let _ = encoding::encode(inst, 120).unwrap();
        }
    }
    let elapsed = start.elapsed();
    let total_insts = scheduled_insts.len() * iterations;
    println!(
        "[4] Instruction encoding ({}x, {} insts total): {:.2}ms total, {:.0}ns/inst",
        iterations,
        total_insts,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000_000.0 / total_insts as f64,
    );
}

fn bench_cubin_generation(scheduled_insts: &[SassInst], num_regs: u32, iterations: usize) {
    let module = SassModule {
        kernels: vec![SassKernel {
            name: "vector_add".to_string(),
            instructions: scheduled_insts.to_vec(),
            num_registers: num_regs,
            shared_mem_bytes: 0,
            const_mem_bytes: 0,
            local_mem_bytes: 0,
            max_threads: 1024,
            params: vec![
                ("a".to_string(), 8, 0),
                ("b".to_string(), 8, 8),
                ("c".to_string(), 8, 16),
                ("n".to_string(), 4, 24),
            ],
        }],
        sm_version: 120,
        global_constants: vec![],
    };

    let start = Instant::now();
    let mut cubin_size = 0;
    for _ in 0..iterations {
        let cubin = cubin_builder::build_cubin_from_module(&module).unwrap();
        cubin_size = cubin.len();
    }
    let elapsed = start.elapsed();
    println!(
        "[5] CUBIN ELF generation ({}x): {:.2}ms total, {:.2}us/iter, {} bytes output",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
        cubin_size,
    );
}

fn bench_full_pipeline(iterations: usize) {
    let start = Instant::now();
    let mut cubin_size = 0;
    for _ in 0..iterations {
        // isel
        let virtual_insts = build_vector_add_virtual();
        // regalloc
        let (physical_insts, num_regs) = regalloc::allocate(&virtual_insts).unwrap();
        // schedule
        let scheduled_insts = scheduler::schedule(&physical_insts);
        // cubin
        let module = SassModule {
            kernels: vec![SassKernel {
                name: "vector_add".to_string(),
                instructions: scheduled_insts,
                num_registers: num_regs,
                shared_mem_bytes: 0,
                const_mem_bytes: 0,
                local_mem_bytes: 0,
                max_threads: 1024,
                params: vec![
                    ("a".to_string(), 8, 0),
                    ("b".to_string(), 8, 8),
                    ("c".to_string(), 8, 16),
                    ("n".to_string(), 4, 24),
                ],
            }],
            sm_version: 120,
            global_constants: vec![],
        };
        let cubin = cubin_builder::build_cubin_from_module(&module).unwrap();
        cubin_size = cubin.len();
    }
    let elapsed = start.elapsed();
    println!(
        "[6] Full pipeline isel→regalloc→sched→cubin ({}x): {:.2}ms total, {:.2}us/iter",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
    println!("    Output CUBIN: {} bytes", cubin_size);
}

fn bench_scaling(n_insts: usize, iterations: usize) {
    let virtual_insts = build_scaled_kernel(n_insts);

    let start = Instant::now();
    for _ in 0..iterations {
        let (physical_insts, num_regs) = regalloc::allocate(&virtual_insts).unwrap();
        let scheduled_insts = scheduler::schedule(&physical_insts);
        let module = SassModule {
            kernels: vec![SassKernel {
                name: "bench_kernel".to_string(),
                instructions: scheduled_insts,
                num_registers: num_regs,
                shared_mem_bytes: 0,
                const_mem_bytes: 0,
                local_mem_bytes: 0,
                max_threads: 1024,
                params: vec![],
            }],
            sm_version: 120,
            global_constants: vec![],
        };
        let _ = cubin_builder::build_cubin_from_module(&module).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "  {} instructions ({}x): {:.2}ms total, {:.2}us/iter, {:.0}ns/inst",
        n_insts,
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
        elapsed.as_secs_f64() * 1_000_000_000.0 / (iterations as f64 * n_insts as f64),
    );
}
