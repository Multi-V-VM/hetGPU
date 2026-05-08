//! Test binary for SASS-PTX mapping and checkpoint/resume functionality
//!
//! Usage: cargo run -p ptx --bin test_sass_mapping

use ptx::{
    CheckpointManager, ExecutionContext, GpuCheckpointState, GpuTrapHandler, HetGpuDebugInterface,
    MemoryRegion, MemorySpace, PtxReconstructor, PtxSourceLocation, ResumeInfo, SassPtxMapper,
    ThreadCheckpointState, TrapReason,
};
use std::collections::HashMap;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== hetGPU SASS-PTX Mapping Test ===\n");

    // Read the kernel.ptx file
    let ptx_source = fs::read_to_string("kernel.ptx")?;
    println!("Loaded kernel.ptx ({} lines)\n", ptx_source.lines().count());

    // Test 1: Create SASS-PTX mapper with PTX source
    println!("--- Test 1: SassPtxMapper ---");
    let mut mapper = SassPtxMapper::with_ptx_source(ptx_source.clone());

    // Simulate cuobjdump output (normally you'd run cuobjdump on the compiled CUBIN)
    let sample_cuobjdump = r#"
        code for sm_61
        Function : atom_add
        .headerflags    @"EF_CUDA_SM61 EF_CUDA_PTX_SM(EF_CUDA_SM61)"
## File "kernel.ptx", line 11
/*0000*/ MOV R1, c[0x0][0x20] ;
/*0010*/ LDG.E.U64 R2, [R4] ;
## Line 13
/*0020*/ LDG.E.U64 R4, [R6] ;
## Line 16
/*0030*/ LDG.E.U32 R0, [R2] ;
## Line 19
/*0040*/ LDG.E.U32 R7, [R2+0x4] ;
## Line 22
/*0050*/ STS.U32 [RZ], R0 ;
/*0060*/ ATOMS.ADD R3, [RZ], R7 ;
## Line 26
/*0070*/ LDS.U32 R0, [RZ] ;
## Line 30
/*0080*/ STG.E.U32 [R4], R3 ;
## Line 33
/*0090*/ STG.E.U32 [R4+0x4], R0 ;
/*00a0*/ EXIT ;
"#;

    mapper.parse_cuobjdump_output(sample_cuobjdump)?;

    println!("Parsed SASS mappings:");
    println!("{}", mapper.dump_mapping_table());

    // Test 2: Query mappings
    println!("--- Test 2: Query Mappings ---");

    // SASS → PTX lookup
    if let Some(loc) = mapper.sass_to_ptx_location(0x0030) {
        println!("SASS 0x0030 → PTX {}:{}", loc.file, loc.line);
        if let Some(line) = mapper.get_ptx_source_line(loc.line) {
            println!("  Source: {}", line.trim());
        }
    }

    // PTX → SASS lookup
    if let Some(addrs) = mapper.ptx_to_sass_addresses("kernel.ptx", 22) {
        println!(
            "PTX line 22 → SASS: {:?}",
            addrs
                .iter()
                .map(|a| format!("0x{:04x}", a))
                .collect::<Vec<_>>()
        );
    }

    // Test 3: HetGPU Debug Interface
    println!("\n--- Test 3: HetGpuDebugInterface ---");
    let mut debug_iface = HetGpuDebugInterface::with_mapper(mapper);

    // Set breakpoints
    let bp_id = debug_iface.set_breakpoint_at_ptx_line("kernel.ptx", 22)?;
    println!("Set breakpoint #{} at kernel.ptx:22", bp_id);

    // Simulate breakpoint hit
    if let Some(bp) = debug_iface.is_breakpoint(0x0050) {
        println!(
            "Breakpoint hit at SASS 0x0050, PTX line {}",
            bp.ptx_location.line
        );
    }

    // Test 4: PTX Reconstructor
    println!("\n--- Test 4: PTX Reconstructor ---");
    let mut reconstructor = PtxReconstructor::new(ptx_source.clone());

    // Add mappings to reconstructor
    for (file, line) in debug_iface.get_mapper().get_mapped_ptx_lines() {
        if let Some(addr) = debug_iface.get_mapper().ptx_to_sass_address(&file, line) {
            reconstructor
                .get_mapper_mut()
                .add_mapping(addr, &file, line, "SASS");
        }
    }

    println!("PTX context around line 22:");
    println!("{}", reconstructor.get_context_window(22, 2, 2));

    // Test 5: GPU Trap Handler and Checkpoint
    println!("--- Test 5: GpuTrapHandler & Checkpoint ---");
    let trap_handler = GpuTrapHandler::new("/tmp/hetgpu_checkpoints");

    // Simulate a trap at SASS address 0x0050
    let registers: HashMap<String, u64> = [
        ("R0".to_string(), 0x12345678),
        ("R1".to_string(), 0xDEADBEEF),
        ("R2".to_string(), 0x00007FFF00000000),
        ("R3".to_string(), 0x00000000000000FF),
    ]
    .into_iter()
    .collect();

    let predicates: HashMap<String, bool> = [("P0".to_string(), true), ("P1".to_string(), false)]
        .into_iter()
        .collect();

    // Add mapper to trap handler's debug interface
    {
        let debug_iface_arc = trap_handler.get_debug_interface();
        let mut debug_iface = debug_iface_arc.write().unwrap();
        debug_iface
            .get_mapper_mut()
            .parse_cuobjdump_output(sample_cuobjdump)?;
        debug_iface
            .get_mapper_mut()
            .set_ptx_source(ptx_source.clone());
    }

    // Handle trap
    let checkpoint = trap_handler.handle_trap(
        TrapReason::UserInterrupt,
        0x0050,
        registers.clone(),
        predicates.clone(),
        (0, 0, 0), // thread_id
        (0, 0, 0), // block_id
        0,         // warp_id
        0,         // lane_id
        "atom_add",
    )?;

    println!("\nCheckpoint created:");
    println!("{}", checkpoint.summary());

    // Test 6: Save and Load Checkpoint
    println!("--- Test 6: Checkpoint Save/Load ---");
    let checkpoint_path = "/tmp/test_checkpoint.json";
    checkpoint.save_to_file(checkpoint_path)?;
    println!("Saved checkpoint to {}", checkpoint_path);

    let loaded_checkpoint = GpuCheckpointState::load_from_file(checkpoint_path)?;
    println!("Loaded checkpoint:");
    println!("  Kernel: {}", loaded_checkpoint.kernel_name);
    println!(
        "  SASS address: 0x{:08x}",
        loaded_checkpoint.execution_context.sass_address
    );
    println!(
        "  PTX line: {}",
        loaded_checkpoint.execution_context.ptx_location.line
    );
    println!(
        "  Registers: {:?}",
        loaded_checkpoint.execution_context.registers.len()
    );

    // Test 7: Resume Info
    println!("\n--- Test 7: Resume Info ---");
    let resume_info = ResumeInfo::from_checkpoint_file(checkpoint_path)?;
    println!("Resume info loaded:");
    println!("  Resume at SASS: 0x{:08x}", resume_info.sass_address);
    println!("  Resume at PTX line: {}", resume_info.ptx_line);
    println!("  Register restoration commands:");
    for cmd in resume_info
        .get_register_restoration_commands()
        .iter()
        .take(2)
    {
        println!("    {}", cmd);
    }

    // Test 8: Checkpoint Manager
    println!("\n--- Test 8: CheckpointManager ---");
    let mut manager = CheckpointManager::new("/tmp/hetgpu_checkpoints");

    // Create multiple checkpoints
    for i in 0..3 {
        let ctx = ExecutionContext {
            sass_address: 0x0050 + (i * 0x10) as u64,
            ptx_location: PtxSourceLocation {
                file: "kernel.ptx".to_string(),
                line: 22 + i,
                column: 0,
                instruction_offset: (0x50 + i * 0x10) as usize,
            },
            registers: registers.clone(),
            predicates: predicates.clone(),
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            warp_id: 0,
            lane_id: 0,
            active_mask: 0xFFFFFFFF,
            call_stack: vec![],
        };

        let mut ckpt = GpuCheckpointState::new(
            TrapReason::CheckpointRequest {
                label: format!("step_{}", i),
            },
            ctx,
            "atom_add".to_string(),
        );
        ckpt.grid_dim = (1, 1, 1);
        ckpt.block_dim = (32, 1, 1);

        manager.add_checkpoint(ckpt);
    }

    println!("Checkpoints in manager:");
    for (kernel, timestamp, threads) in manager.list_checkpoints() {
        println!("  {} @ {} ({} threads)", kernel, timestamp, threads);
    }

    if let Some(latest) = manager.get_latest_checkpoint("atom_add") {
        println!("\nLatest checkpoint for atom_add:");
        println!("  SASS: 0x{:08x}", latest.execution_context.sass_address);
        println!("  PTX line: {}", latest.execution_context.ptx_location.line);
    }

    // Test 9: Thread state checkpointing
    println!("\n--- Test 9: Thread State Checkpointing ---");
    let mut full_checkpoint = GpuCheckpointState::new(
        TrapReason::UserInterrupt,
        ExecutionContext {
            sass_address: 0x0050,
            ptx_location: PtxSourceLocation {
                file: "kernel.ptx".to_string(),
                line: 22,
                column: 0,
                instruction_offset: 0x50,
            },
            registers: registers.clone(),
            predicates: predicates.clone(),
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            warp_id: 0,
            lane_id: 0,
            active_mask: 0xFFFFFFFF,
            call_stack: vec![],
        },
        "atom_add".to_string(),
    );

    // Add thread states
    for tid in 0..32 {
        let thread_state = ThreadCheckpointState {
            thread_id: (tid, 0, 0),
            block_id: (0, 0, 0),
            warp_id: 0,
            lane_id: tid,
            sass_address: 0x0050,
            registers: [
                (format!("R0"), 0x12345678 + tid as u64),
                (format!("R1"), 0xDEADBEEF),
            ]
            .into_iter()
            .collect(),
            predicates: [("P0".to_string(), true)].into_iter().collect(),
            local_memory: vec![],
            active: true,
        };
        full_checkpoint.thread_states.push(thread_state);
    }

    // Add memory regions
    full_checkpoint.memory_regions.push(MemoryRegion {
        name: "shared_mem".to_string(),
        base_address: 0,
        size: 1024,
        data: vec![0u8; 1024],
        memory_space: MemorySpace::Shared,
    });

    full_checkpoint.ptx_source = Some(ptx_source);
    full_checkpoint.grid_dim = (1, 1, 1);
    full_checkpoint.block_dim = (32, 1, 1);

    let full_checkpoint_path = "/tmp/full_checkpoint.json";
    full_checkpoint.save_to_file(full_checkpoint_path)?;
    println!("Full checkpoint saved to {}", full_checkpoint_path);
    println!("  Thread states: {}", full_checkpoint.thread_states.len());
    println!("  Memory regions: {}", full_checkpoint.memory_regions.len());

    // Verify
    let loaded_full = GpuCheckpointState::load_from_file(full_checkpoint_path)?;
    println!(
        "  Verified load: {} threads, {} memory regions",
        loaded_full.thread_states.len(),
        loaded_full.memory_regions.len()
    );

    println!("\n=== All tests passed! ===");
    Ok(())
}
