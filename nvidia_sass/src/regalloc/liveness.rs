use crate::types::*;
use std::collections::HashMap;

pub fn compute_live_ranges(insts: &[SassInst]) -> HashMap<u8, (usize, usize)> {
    let mut ranges: HashMap<u8, (usize, usize)> = HashMap::new();

    for (idx, inst) in insts.iter().enumerate() {
        if let Some(Reg::R(n)) = inst.dst {
            ranges.entry(n).or_insert((idx, idx)).1 = idx;
        }
        for src in &inst.srcs {
            for reg in extract_regs(src) {
                if let Reg::R(n) = reg {
                    let entry = ranges.entry(n).or_insert((idx, idx));
                    if idx < entry.0 {
                        entry.0 = idx;
                    }
                    if idx > entry.1 {
                        entry.1 = idx;
                    }
                }
            }
        }
    }
    ranges
}

fn extract_regs(op: &Operand) -> Vec<Reg> {
    match op {
        Operand::Reg(r) => vec![*r],
        Operand::Memory { base, .. } => vec![*base],
        _ => vec![],
    }
}
