// nvidia_sass/tests/e2e_vector_add.rs

use nvidia_sass::cubin_builder;
use nvidia_sass::isel;
use nvidia_sass::regalloc;
use nvidia_sass::scheduler;
use nvidia_sass::types::*;

/// Build a vector_add kernel through the full pipeline.
///
/// Kernel: vector_add(float *a, float *b, float *c, int n)
///   tid = threadIdx.x
///   c[tid] = a[tid] + b[tid]
#[test]
fn test_e2e_vector_add_cubin() {
    // Step 1: Build instruction sequence with virtual regs (128+).
    // The register allocator treats R(n) where n >= 128 as virtual.
    let virtual_insts = vec![
        isel::select_special_reg(200, SpecialReg::TidX), // R200 = tid.x
        isel::select_load_global(201, 200, 0),           // R201 = *(R200 + 0) [a[tid]]
        isel::select_load_global(202, 200, 4),           // R202 = *(R200 + 4) [b[tid]]
        isel::select_add_f32(203, 201, 202),             // R203 = R201 + R202
        isel::select_store_global(200, 8, 203),          // *(R200 + 8) = R203 [c[tid]]
        isel::select_exit(),
    ];

    // Step 2: Register allocation
    let (physical_insts, num_regs) = regalloc::allocate(&virtual_insts).unwrap();
    assert!(num_regs <= 255);

    // Verify that all virtual registers were mapped to physical registers < 128
    for inst in &physical_insts {
        if let Some(Reg::R(n)) = inst.dst {
            assert!(
                n < 128,
                "virtual reg should be mapped to physical, got R{}",
                n
            );
        }
    }

    // Step 3: Instruction scheduling
    let scheduled_insts = scheduler::schedule(&physical_insts);

    // Step 4: Build SassModule
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
                // (name, offset, size)
                ("a".to_string(), 0, 8),
                ("b".to_string(), 8, 8),
                ("c".to_string(), 16, 8),
                ("n".to_string(), 24, 4),
            ],
        }],
        sm_version: 120,
        global_constants: vec![],
    };

    // Step 5: Generate CUBIN
    let cubin = cubin_builder::build_cubin_from_module(&module).unwrap();

    // Validate ELF structure
    assert!(cubin.len() > 64, "CUBIN should be larger than ELF header");
    assert_eq!(&cubin[0..4], b"\x7fELF", "valid ELF magic");
    assert_eq!(cubin[4], 2, "64-bit ELF");
    assert_eq!(u16::from_le_bytes([cubin[18], cubin[19]]), 190, "EM_CUDA");

    // Kernel name should appear in string tables
    let cubin_str = String::from_utf8_lossy(&cubin);
    assert!(cubin_str.contains("vector_add"), "contains kernel name");

    eprintln!("Generated CUBIN size: {} bytes", cubin.len());
    eprintln!("Registers used: {}", num_regs);
}

/// Test round-trip for each instruction in vector_add.
#[test]
fn test_e2e_vector_add_roundtrip() {
    // Use physical registers (< 128) since roundtrip validates encoding directly.
    let insts = vec![
        isel::select_special_reg(0, SpecialReg::TidX),
        isel::select_load_global(1, 0, 0),
        isel::select_load_global(2, 0, 4),
        isel::select_add_f32(3, 1, 2),
        isel::select_store_global(0, 8, 3),
        isel::select_exit(),
    ];

    for inst in &insts {
        nvidia_sass::roundtrip::validate_roundtrip(inst, 120).unwrap();
    }
}

/// Test that scheduling produces valid control codes.
#[test]
fn test_e2e_vector_add_scheduling() {
    let insts = vec![
        isel::select_special_reg(0, SpecialReg::TidX),
        isel::select_load_global(1, 0, 0),
        isel::select_load_global(2, 0, 4),
        isel::select_add_f32(3, 1, 2),
        isel::select_store_global(0, 8, 3),
        isel::select_exit(),
    ];

    let scheduled = scheduler::schedule(&insts);

    // FADD (idx 3) reads R1 and R2 from LDG (high latency).
    // Should have barrier waits set because LDG latency exceeds 15.
    let fadd = &scheduled[3];
    assert!(
        fadd.control.wait_mask != 0 || fadd.control.stall > 1,
        "FADD should wait for LDG results"
    );

    // All stall counts should be valid (1-15)
    for inst in &scheduled {
        assert!(
            inst.control.stall >= 1 && inst.control.stall <= 15,
            "stall count should be 1-15, got {}",
            inst.control.stall
        );
    }
}
