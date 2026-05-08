use crate::types::ControlCodes;

pub fn encode(cc: &ControlCodes) -> u64 {
    let mut bits: u64 = 0;
    bits |= (cc.stall as u64) & 0xF;
    bits |= ((cc.yield_flag as u64) & 1) << 4;
    bits |= ((cc.write_barrier as u64) & 0x7) << 5;
    bits |= ((cc.read_barrier as u64) & 0x7) << 8;
    bits |= ((cc.wait_mask as u64) & 0x3F) << 11;
    bits |= ((cc.reuse as u64) & 0xF) << 17;
    bits
}

pub fn decode(bits: u64) -> ControlCodes {
    ControlCodes {
        stall: (bits & 0xF) as u8,
        yield_flag: ((bits >> 4) & 1) != 0,
        write_barrier: ((bits >> 5) & 0x7) as u8,
        read_barrier: ((bits >> 8) & 0x7) as u8,
        wait_mask: ((bits >> 11) & 0x3F) as u8,
        reuse: ((bits >> 17) & 0xF) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_default() {
        let cc = ControlCodes::default();
        let encoded = encode(&cc);
        let decoded = decode(encoded);
        assert_eq!(cc, decoded);
    }

    #[test]
    fn test_roundtrip_full() {
        let cc = ControlCodes {
            stall: 5,
            yield_flag: true,
            write_barrier: 3,
            read_barrier: 2,
            wait_mask: 0b101010,
            reuse: 0b1100,
        };
        let encoded = encode(&cc);
        let decoded = decode(encoded);
        assert_eq!(cc, decoded);
    }

    #[test]
    fn test_encode_bits() {
        let cc = ControlCodes {
            stall: 0xF,
            yield_flag: false,
            write_barrier: 7,
            read_barrier: 7,
            wait_mask: 0,
            reuse: 0,
        };
        let bits = encode(&cc);
        assert_eq!(bits & 0xF, 0xF);
        assert_eq!((bits >> 5) & 0x7, 7);
        assert_eq!((bits >> 8) & 0x7, 7);
    }
}
