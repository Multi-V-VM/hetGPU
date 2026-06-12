use nvidia_sass::cubin_builder::build_cubin_from_module;
use nvidia_sass::types::*;

fn make_simple_kernel() -> SassModule {
    SassModule {
        sm_version: 120,
        global_constants: vec![],
        kernels: vec![SassKernel {
            name: "test_kernel".to_string(),
            instructions: vec![
                // NOP
                SassInst {
                    opcode: Opcode {
                        mnemonic: "NOP",
                        class: OpcodeClass::Nop,
                    },
                    dst: None,
                    srcs: vec![],
                    pred: None,
                    modifiers: vec![],
                    control: ControlCodes::default(),
                },
                // EXIT
                SassInst {
                    opcode: Opcode {
                        mnemonic: "EXIT",
                        class: OpcodeClass::Branch,
                    },
                    dst: None,
                    srcs: vec![],
                    pred: None,
                    modifiers: vec![],
                    control: ControlCodes::default(),
                },
            ],
            num_registers: 8,
            shared_mem_bytes: 0,
            const_mem_bytes: 160,
            local_mem_bytes: 0,
            max_threads: 1024,
            params: vec![],
        }],
    }
}

#[test]
fn test_cubin_has_elf_magic() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    assert!(cubin.len() >= 64, "CUBIN too small to contain ELF header");
    assert_eq!(&cubin[0..4], b"\x7fELF", "CUBIN must start with ELF magic");
}

#[test]
fn test_cubin_is_64bit_le() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    assert_eq!(cubin[4], 2, "ELF class should be ELFCLASS64 (2)");
    assert_eq!(
        cubin[5], 1,
        "ELF data should be ELFDATA2LSB (1) for little-endian"
    );
}

#[test]
fn test_cubin_machine_type() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let e_machine = u16::from_le_bytes([cubin[18], cubin[19]]);
    assert_eq!(e_machine, 190, "e_machine should be EM_CUDA (190)");
}

#[test]
fn test_cubin_sm_version_in_flags() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let e_flags = u32::from_le_bytes([cubin[48], cubin[49], cubin[50], cubin[51]]);
    assert_eq!(e_flags, 120, "e_flags should contain SM version 120");
}

#[test]
fn test_cubin_contains_text_section() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let text_name = b".text.test_kernel";
    assert!(
        contains_bytes(&cubin, text_name),
        "CUBIN should contain the string '.text.test_kernel'"
    );
}

#[test]
fn test_cubin_contains_nv_info_section() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    assert!(
        contains_bytes(&cubin, b".nv.info"),
        "CUBIN should contain the string '.nv.info'"
    );
}

#[test]
fn test_cubin_contains_kernel_symbol() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    assert!(
        contains_bytes(&cubin, b"test_kernel"),
        "CUBIN should contain the kernel name 'test_kernel' in the symbol table"
    );
}

#[test]
fn test_cubin_elf_header_size() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let e_ehsize = u16::from_le_bytes([cubin[52], cubin[53]]);
    assert_eq!(e_ehsize, 64, "ELF header size should be 64 for ELF64");
}

#[test]
fn test_cubin_section_header_entry_size() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let e_shentsize = u16::from_le_bytes([cubin[58], cubin[59]]);
    assert_eq!(
        e_shentsize, 64,
        "Section header entry size should be 64 for ELF64"
    );
}

#[test]
fn test_cubin_section_count() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    let e_shnum = u16::from_le_bytes([cubin[60], cubin[61]]);
    // NULL + shstrtab + strtab + symtab + nv.info_global + 3 per kernel = 8
    assert_eq!(e_shnum, 8, "Should have 8 sections for 1-kernel CUBIN");
}

#[test]
fn test_cubin_text_data_present() {
    let module = make_simple_kernel();
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");
    // Two instructions at 16 bytes each = 32 bytes of text data.
    // The text section is 128-byte aligned, so find the aligned region.
    // Just verify the CUBIN is large enough to contain the text + headers.
    assert!(
        cubin.len() > 64 + 32,
        "CUBIN should be large enough to hold ELF header + text data"
    );
}

#[test]
fn test_cubin_multiple_kernels() {
    let module = SassModule {
        sm_version: 120,
        global_constants: vec![],
        kernels: vec![
            SassKernel {
                name: "kernel_a".to_string(),
                instructions: vec![SassInst {
                    opcode: Opcode {
                        mnemonic: "EXIT",
                        class: OpcodeClass::Branch,
                    },
                    dst: None,
                    srcs: vec![],
                    pred: None,
                    modifiers: vec![],
                    control: ControlCodes::default(),
                }],
                num_registers: 8,
                shared_mem_bytes: 0,
                const_mem_bytes: 160,
                local_mem_bytes: 0,
                max_threads: 512,
                params: vec![],
            },
            SassKernel {
                name: "kernel_b".to_string(),
                instructions: vec![SassInst {
                    opcode: Opcode {
                        mnemonic: "NOP",
                        class: OpcodeClass::Nop,
                    },
                    dst: None,
                    srcs: vec![],
                    pred: None,
                    modifiers: vec![],
                    control: ControlCodes::default(),
                }],
                num_registers: 16,
                shared_mem_bytes: 256,
                const_mem_bytes: 160,
                local_mem_bytes: 0,
                max_threads: 1024,
                params: vec![],
            },
        ],
    };
    let cubin = build_cubin_from_module(&module).expect("CUBIN build should succeed");

    assert!(
        contains_bytes(&cubin, b".text.kernel_a"),
        "should contain .text.kernel_a"
    );
    assert!(
        contains_bytes(&cubin, b".text.kernel_b"),
        "should contain .text.kernel_b"
    );
    assert!(
        contains_bytes(&cubin, b"kernel_a"),
        "should contain symbol kernel_a"
    );
    assert!(
        contains_bytes(&cubin, b"kernel_b"),
        "should contain symbol kernel_b"
    );

    let e_shnum = u16::from_le_bytes([cubin[60], cubin[61]]);
    // NULL + shstrtab + strtab + symtab + nv.info_global + 3*2 kernels = 11
    assert_eq!(e_shnum, 11, "Should have 11 sections for 2-kernel CUBIN");
}

/// Search for a byte subsequence in the binary.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}
