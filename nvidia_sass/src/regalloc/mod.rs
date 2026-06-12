pub mod liveness;

use crate::types::*;
use std::collections::HashMap;

/// Allocate physical registers for instructions with virtual registers.
/// Returns (physical instructions, number of registers used).
/// Virtual registers are R(n) where n >= 128.
pub fn allocate(insts: &[SassInst]) -> Result<(Vec<SassInst>, u32), NvSassError> {
    let live_ranges = liveness::compute_live_ranges(insts);
    let mapping = linear_scan(&live_ranges)?;
    let num_regs = mapping.values().copied().max().unwrap_or(0) as u32 + 1;

    let physical_insts = insts
        .iter()
        .map(|inst| {
            let mut new_inst = inst.clone();
            if let Some(ref mut dst) = new_inst.dst {
                *dst = remap_reg(dst, &mapping);
            }
            new_inst.srcs = new_inst
                .srcs
                .iter()
                .map(|op| remap_operand(op, &mapping))
                .collect();
            new_inst
        })
        .collect();

    Ok((physical_insts, num_regs.max(8)))
}

fn linear_scan(live_ranges: &HashMap<u8, (usize, usize)>) -> Result<HashMap<u8, u8>, NvSassError> {
    let mut mapping: HashMap<u8, u8> = HashMap::new();

    let mut ranges: Vec<(u8, usize, usize)> = live_ranges
        .iter()
        .map(|(&vreg, &(start, end))| (vreg, start, end))
        .collect();
    ranges.sort_by_key(|&(_, start, _)| start);

    let mut active: Vec<(u8, u8, usize)> = Vec::new(); // (vreg, phys, end)

    for (vreg, start, end) in ranges {
        if vreg < 128 {
            mapping.insert(vreg, vreg);
            continue;
        }

        // Expire old intervals
        active.retain(|&(_, _, active_end)| active_end >= start);

        let used: std::collections::HashSet<u8> = active.iter().map(|&(_, phys, _)| phys).collect();
        let phys = (0..=254u8)
            .find(|r| !used.contains(r))
            .ok_or_else(|| NvSassError::RegAllocError("out of registers".to_string()))?;

        mapping.insert(vreg, phys);
        active.push((vreg, phys, end));
    }

    Ok(mapping)
}

fn remap_reg(reg: &Reg, mapping: &HashMap<u8, u8>) -> Reg {
    match reg {
        Reg::R(n) => {
            if let Some(&phys) = mapping.get(n) {
                Reg::R(phys)
            } else {
                *reg
            }
        }
        other => *other,
    }
}

fn remap_operand(op: &Operand, mapping: &HashMap<u8, u8>) -> Operand {
    match op {
        Operand::Reg(r) => Operand::Reg(remap_reg(r, mapping)),
        Operand::Memory { base, offset } => Operand::Memory {
            base: remap_reg(base, mapping),
            offset: *offset,
        },
        other => other.clone(),
    }
}
