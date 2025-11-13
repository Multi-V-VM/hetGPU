// Integration tests for TMatmul backend

#[cfg(test)]
mod tests {
    use ptx::pass::emit_tmatmul_asm::*;

    #[test]
    fn test_register_allocator() {
        let mut allocator = RegisterAllocator::new();

        // Allocate first register
        let r0 = allocator.allocate("%0").unwrap();
        assert_eq!(r0, "v0");

        // Allocate second register
        let r1 = allocator.allocate("%1").unwrap();
        assert_eq!(r1, "v1");

        // Idempotent allocation
        let r0_again = allocator.allocate("%0").unwrap();
        assert_eq!(r0_again, "v0");

        // Free and reallocate
        allocator.free("%0");
        let r_new = allocator.allocate("%2").unwrap();
        assert_eq!(r_new, "v0"); // Should reuse freed register
    }

    #[test]
    fn test_register_exhaustion() {
        let mut allocator = RegisterAllocator::new();

        // Allocate all 8 registers
        for i in 0..8 {
            let reg = allocator.allocate(&format!("%{}", i)).unwrap();
            assert_eq!(reg, format!("v{}", i));
        }

        // Should fail on 9th allocation
        let result = allocator.allocate("%8");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_load_store() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("input", MemoryLocation::X);
        codegen.map_memory("output", MemoryLocation::O);

        codegen
            .emit_operation("tmatmul.ldv", &["input"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.sv", &["%0", "output"], &[])
            .unwrap();

        let assembly = codegen.get_assembly();

        assert!(assembly.contains("ldv"));
        assert!(assembly.contains("v0"));
        assert!(assembly.contains("X"));
        assert!(assembly.contains("sv"));
        assert!(assembly.contains("O"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);

        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%1"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.add", &["%0", "%1"], &["%2"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.mul", &["%2", "%0"], &["%3"])
            .unwrap();

        let assembly = codegen.get_assembly();

        assert!(assembly.contains("add"));
        assert!(assembly.contains("mul"));
        assert!(assembly.contains("v2"));
        assert!(assembly.contains("v3"));
    }

    #[test]
    fn test_activation_functions() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);

        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.sig", &["%0"], &["%1"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.csig", &["%0"], &["%2"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.silu", &["%0"], &["%3"])
            .unwrap();

        let assembly = codegen.get_assembly();

        assert!(assembly.contains("sig"));
        assert!(assembly.contains("csig"));
        assert!(assembly.contains("silu"));
    }

    #[test]
    fn test_matmul_sequence() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("W", MemoryLocation::WF);

        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "W"], &["%1"])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Should contain the three-instruction sequence
        assert!(assembly.contains("tmatmul_import"));
        assert!(assembly.contains("tmatmul_go"));
        assert!(assembly.contains("tmatmul_export"));
        assert!(assembly.contains("WF"));
    }

    #[test]
    fn test_normalization() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);

        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%1"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%1"], &["%2"])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Check for double normalization (count lines starting with "    norm")
        let norm_count = assembly
            .lines()
            .filter(|line| line.trim_start().starts_with("norm"))
            .count();
        assert_eq!(norm_count, 2);
    }

    #[test]
    fn test_complete_mlgru_sequence() {
        let mut codegen = TMatmulCodegen::new();

        // Setup memory map
        codegen.map_memory("X", MemoryLocation::X);
        codegen.map_memory("oH", MemoryLocation::OH);
        codegen.map_memory("WF", MemoryLocation::WF);
        codegen.map_memory("WC", MemoryLocation::WC);
        codegen.map_memory("TEMP_VEC", MemoryLocation::TempVec);

        // MLGRU sequence
        codegen.add_section("MLGRU Test");

        // Load input and hidden state
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.sv", &["%0", "TEMP_VEC"], &[])
            .unwrap();
        codegen
            .emit_operation("tmatmul.norm", &["%0"], &["%0"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.ldv", &["oH"], &["%1"])
            .unwrap();

        // Compute gates
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "WF"], &["%2"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.matmul", &["%0", "WC"], &["%3"])
            .unwrap();

        // Activations
        codegen
            .emit_operation("tmatmul.sig", &["%2"], &["%7"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.silu", &["%3"], &["%3"])
            .unwrap();

        // Hidden state update
        codegen
            .emit_operation("tmatmul.mul", &["%7", "%1"], &["%5"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.csig", &["%2"], &["%6"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.mul", &["%6", "%3"], &["%6"])
            .unwrap();
        codegen
            .emit_operation("tmatmul.add", &["%5", "%6"], &["%1"])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Verify key operations present
        assert!(assembly.contains("MLGRU Test"));
        assert!(assembly.contains("ldv"));
        assert!(assembly.contains("sv"));
        assert!(assembly.contains("norm"));
        assert!(assembly.contains("tmatmul_import"));
        assert!(assembly.contains("tmatmul_go"));
        assert!(assembly.contains("sig"));
        assert!(assembly.contains("silu"));
        assert!(assembly.contains("csig"));
        assert!(assembly.contains("WF"));
        assert!(assembly.contains("WC"));
    }

    #[test]
    fn test_assembly_format() {
        let mut codegen = TMatmulCodegen::new();

        codegen.map_memory("X", MemoryLocation::X);

        codegen.add_section("TEST SECTION");
        codegen.add_comment("This is a test");
        codegen
            .emit_operation("tmatmul.ldv", &["X"], &["%0"])
            .unwrap();

        let assembly = codegen.get_assembly();

        // Check header format
        assert!(assembly.contains("vector registers: v0..v7"));
        assert!(assembly.contains("DDR: X, WG, WF"));
        assert!(assembly.contains("instructions:"));

        // Check section format
        assert!(assembly.contains("-------------------------------------------------------"));
        assert!(assembly.contains("TEST SECTION"));

        // Check comment format
        assert!(assembly.contains("; This is a test"));
    }

    #[test]
    fn test_error_handling_unallocated_operand() {
        let mut codegen = TMatmulCodegen::new();

        // Try to use an operand that hasn't been allocated
        let result = codegen.emit_operation("tmatmul.add", &["%999", "%1000"], &["%2"]);

        assert!(result.is_err());
    }

    #[test]
    fn test_memory_location_display() {
        assert_eq!(format!("{}", MemoryLocation::X), "X");
        assert_eq!(format!("{}", MemoryLocation::WF), "WF");
        assert_eq!(format!("{}", MemoryLocation::OH), "oH");
        assert_eq!(format!("{}", MemoryLocation::TempVec), "TEMP_VEC");
        assert_eq!(
            format!("{}", MemoryLocation::Custom("MY_MEM".to_string())),
            "MY_MEM"
        );
    }

    #[test]
    fn test_instruction_formatting() {
        let inst1 = TMatmulInstruction::LoadVector {
            dst: "v0".to_string(),
            src: MemoryLocation::X,
        };
        assert_eq!(format!("{}", inst1), "    ldv    v0,X");

        let inst2 = TMatmulInstruction::Add {
            dst: "v2".to_string(),
            src1: "v0".to_string(),
            src2: "v1".to_string(),
        };
        assert_eq!(format!("{}", inst2), "    add    v2,v0,v1");

        let inst3 = TMatmulInstruction::Comment("Test comment".to_string());
        assert_eq!(format!("{}", inst3), "    ; Test comment");
    }
}
