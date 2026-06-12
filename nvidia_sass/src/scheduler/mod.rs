use crate::types::*;
use std::collections::HashMap;

/// Instruction latency estimates for SM120.
fn instruction_latency(class: OpcodeClass) -> u8 {
    match class {
        OpcodeClass::Alu3 | OpcodeClass::Alu2 => 4,
        OpcodeClass::Fma => 4,
        OpcodeClass::Load => 200,
        OpcodeClass::Store => 1,
        OpcodeClass::Branch => 1,
        OpcodeClass::Comparison => 4,
        OpcodeClass::Sync => 1,
        OpcodeClass::Special => 8,
        OpcodeClass::Nop => 1,
    }
}

/// Schedule instructions by computing control codes.
///
/// Tracks register write latencies and assigns:
/// - Stall counts for short-latency dependencies
/// - Write barriers for long-latency ops (loads)
/// - Wait masks when consuming barrier-protected results
pub fn schedule(insts: &[SassInst]) -> Vec<SassInst> {
    let mut result = Vec::with_capacity(insts.len());
    let mut pending_writes: HashMap<u8, (usize, u8)> = HashMap::new();
    let mut next_barrier: u8 = 0;
    let mut barrier_map: HashMap<u8, u8> = HashMap::new();

    for (idx, inst) in insts.iter().enumerate() {
        let mut ctrl = ControlCodes {
            stall: 1,
            yield_flag: false,
            write_barrier: 7,
            read_barrier: 7,
            wait_mask: 0,
            reuse: 0,
        };

        let mut max_stall_needed: u8 = 0;
        for src_reg in source_registers(inst) {
            if let Some(&barrier_id) = barrier_map.get(&src_reg) {
                ctrl.wait_mask |= 1 << barrier_id;
            }
            if let Some(&(write_idx, latency)) = pending_writes.get(&src_reg) {
                let distance = (idx - write_idx) as u8;
                if distance < latency {
                    let needed = latency.saturating_sub(distance);
                    max_stall_needed = max_stall_needed.max(needed);
                }
            }
        }

        ctrl.stall = max_stall_needed.min(15).max(1);

        if ctrl.wait_mask != 0 {
            ctrl.stall = 1;
        }

        if let Some(ref dst) = inst.dst {
            if let Reg::R(n) = dst {
                let latency = instruction_latency(inst.opcode.class);
                pending_writes.insert(*n, (idx, latency));

                if latency > 15 {
                    let barrier_id = next_barrier % 6;
                    ctrl.write_barrier = barrier_id;
                    barrier_map.insert(*n, barrier_id);
                    next_barrier += 1;
                }
            }
        }

        let mut scheduled_inst = inst.clone();
        scheduled_inst.control = ctrl;
        result.push(scheduled_inst);
    }

    result
}

fn source_registers(inst: &SassInst) -> Vec<u8> {
    let mut regs = Vec::new();
    for src in &inst.srcs {
        match src {
            Operand::Reg(Reg::R(n)) => regs.push(*n),
            Operand::Memory {
                base: Reg::R(n), ..
            } => regs.push(*n),
            _ => {}
        }
    }
    regs
}
