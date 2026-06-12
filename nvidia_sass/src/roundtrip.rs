use crate::encoding;
use crate::types::*;

pub fn validate_roundtrip(inst: &SassInst, sm_version: u32) -> Result<(), NvSassError> {
    let encoded = encoding::encode(inst, sm_version)?;
    let lo = u64::from_le_bytes(encoded[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(encoded[8..].try_into().unwrap());

    // 1. Control codes round-trip
    let decoded_cc = encoding::control_codes::decode(hi);
    if decoded_cc != inst.control {
        return Err(NvSassError::EncodingError {
            opcode: inst.opcode.mnemonic.to_string(),
            msg: format!(
                "control code mismatch: encoded {:?}, decoded {:?}",
                inst.control, decoded_cc
            ),
        });
    }

    // 2. Opcode bits match
    let opcode_bits = ((lo >> 52) & 0xFFF) as u16;
    let expected_bits = encoding::sm120::lookup_opcode_bits(inst.opcode.mnemonic)?;
    if opcode_bits != expected_bits {
        return Err(NvSassError::EncodingError {
            opcode: inst.opcode.mnemonic.to_string(),
            msg: format!(
                "opcode bits mismatch: got 0x{:03x}, expected 0x{:03x}",
                opcode_bits, expected_bits
            ),
        });
    }

    // 3. Destination register
    if let Some(ref dst) = inst.dst {
        match dst {
            Reg::R(_) | Reg::RZ => {
                let dst_bits = (lo & 0xFF) as u8;
                let expected_dst = dst.encode_gpr();
                if dst_bits != expected_dst {
                    return Err(NvSassError::EncodingError {
                        opcode: inst.opcode.mnemonic.to_string(),
                        msg: format!(
                            "dst register mismatch: got {}, expected {}",
                            dst_bits, expected_dst
                        ),
                    });
                }
            }
            _ => {} // predicate dst validated separately
        }
    }

    // 4. Predicate
    let pred_reg = ((lo >> 16) & 0x7) as u8;
    let pred_neg = ((lo >> 19) & 0x1) != 0;
    match &inst.pred {
        Some(p) => {
            if pred_reg != p.reg.encode_pred() || pred_neg != p.negated {
                return Err(NvSassError::EncodingError {
                    opcode: inst.opcode.mnemonic.to_string(),
                    msg: "predicate mismatch".to_string(),
                });
            }
        }
        None => {
            if pred_reg != 7 {
                // PT = 7
                return Err(NvSassError::EncodingError {
                    opcode: inst.opcode.mnemonic.to_string(),
                    msg: format!("expected PT (7), got {}", pred_reg),
                });
            }
        }
    }

    Ok(())
}
