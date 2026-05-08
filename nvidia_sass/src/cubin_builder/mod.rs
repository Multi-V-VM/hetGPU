use crate::types::*;

// ---------------------------------------------------------------------------
// ELF64 constants
// ---------------------------------------------------------------------------

// ELF identification
const ELFMAG: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ELFOSABI_NONE: u8 = 0;

// ELF type
const ET_EXEC: u16 = 2;

// CUDA machine type
const EM_CUDA: u16 = 190;

// Section header types
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_CUDA_INFO: u32 = 0x7000_0000;

// Section header flags
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

// Symbol binding and type
const STB_GLOBAL: u8 = 1;
const STB_LOCAL: u8 = 0;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;

// ELF header size and section header entry size for ELF64
const ELF64_EHDR_SIZE: u16 = 64;
const ELF64_SHDR_SIZE: u16 = 64;
const ELF64_SYM_SIZE: usize = 24;

// .nv.info attribute types (format: u8 format | u8 attribute, u16 size)
const EIATTR_REGCOUNT: u16 = 0x2F04;
const EIATTR_MAX_THREADS: u16 = 0x0504;
const EIATTR_SMEM_SIZE: u16 = 0x0808;
const EIATTR_LMEM_SIZE: u16 = 0x0A04;
const EIATTR_EXIT_INSTR_OFFSETS: u16 = 0x1C04;

// Text section alignment
const TEXT_ALIGNMENT: u64 = 128;

// ---------------------------------------------------------------------------
// ELF builder helpers
// ---------------------------------------------------------------------------

/// A string table builder that deduplicates entries.
struct StringTable {
    data: Vec<u8>,
}

impl StringTable {
    fn new() -> Self {
        // String table starts with a null byte.
        StringTable { data: vec![0] }
    }

    /// Add a string and return its offset.
    fn add(&mut self, s: &str) -> u32 {
        // Check if the string already exists
        let needle = s.as_bytes();
        if let Some(pos) = self.find(needle) {
            return pos as u32;
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
        offset
    }

    fn find(&self, needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        // Search for existing string in table (must be null-terminated at matching position)
        let len = needle.len();
        for i in 0..self.data.len() {
            if i + len + 1 <= self.data.len()
                && &self.data[i..i + len] == needle
                && self.data[i + len] == 0
            {
                return Some(i);
            }
        }
        None
    }

    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Helper to write a little-endian u16 to a byte vector.
fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Helper to write a little-endian u32 to a byte vector.
fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Helper to write a little-endian u64 to a byte vector.
fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Align buffer to given alignment by appending zero bytes.
fn align_to(buf: &mut Vec<u8>, alignment: usize) {
    let rem = buf.len() % alignment;
    if rem != 0 {
        buf.resize(buf.len() + (alignment - rem), 0);
    }
}

// ---------------------------------------------------------------------------
// Section header descriptor (built up, then serialized at the end)
// ---------------------------------------------------------------------------

struct SectionHeader {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

impl SectionHeader {
    fn write_to(&self, buf: &mut Vec<u8>) {
        write_u32(buf, self.sh_name);
        write_u32(buf, self.sh_type);
        write_u64(buf, self.sh_flags);
        write_u64(buf, self.sh_addr);
        write_u64(buf, self.sh_offset);
        write_u64(buf, self.sh_size);
        write_u32(buf, self.sh_link);
        write_u32(buf, self.sh_info);
        write_u64(buf, self.sh_addralign);
        write_u64(buf, self.sh_entsize);
    }
}

// ---------------------------------------------------------------------------
// .nv.info builder
// ---------------------------------------------------------------------------

fn build_nv_info_for_kernel(kernel: &SassKernel, text_section_idx: u16) -> Vec<u8> {
    let mut data = Vec::new();

    // Each attribute: u16 type_id, u16 size, payload (u32 value)
    // The format byte and attribute byte are packed into the u16 type_id.

    // EIATTR_REGCOUNT
    write_u16(&mut data, EIATTR_REGCOUNT);
    write_u16(&mut data, 4);
    write_u32(&mut data, kernel.num_registers);

    // EIATTR_MAX_THREADS
    write_u16(&mut data, EIATTR_MAX_THREADS);
    write_u16(&mut data, 4);
    write_u32(&mut data, kernel.max_threads);

    // EIATTR_SMEM_SIZE (only if nonzero)
    if kernel.shared_mem_bytes > 0 {
        write_u16(&mut data, EIATTR_SMEM_SIZE);
        write_u16(&mut data, 4);
        write_u32(&mut data, kernel.shared_mem_bytes);
    }

    // EIATTR_LMEM_SIZE (only if nonzero)
    if kernel.local_mem_bytes > 0 {
        write_u16(&mut data, EIATTR_LMEM_SIZE);
        write_u16(&mut data, 4);
        write_u32(&mut data, kernel.local_mem_bytes);
    }

    // EIATTR_EXIT_INSTR_OFFSETS: scan instructions to find EXIT opcodes
    let exit_offsets = find_exit_offsets(&kernel.instructions);
    for off in &exit_offsets {
        write_u16(&mut data, EIATTR_EXIT_INSTR_OFFSETS);
        write_u16(&mut data, 4);
        write_u32(&mut data, *off);
    }

    // Encode the target text section index for per-kernel info.
    // NVIDIA encodes this as a 2-byte section index prefix for certain attributes,
    // but for basic CUBIN generation we embed the attributes directly.
    let _ = text_section_idx;

    data
}

/// Scan instructions for EXIT opcodes and return their byte offsets.
fn find_exit_offsets(instructions: &[SassInst]) -> Vec<u32> {
    let mut offsets = Vec::new();
    for (i, inst) in instructions.iter().enumerate() {
        if inst.opcode.mnemonic == "EXIT" {
            // Each instruction is 16 bytes
            offsets.push((i as u32) * 16);
        }
    }
    offsets
}

// ---------------------------------------------------------------------------
// .nv.constant0 builder
// ---------------------------------------------------------------------------

fn build_constant0_for_kernel(kernel: &SassKernel) -> Vec<u8> {
    // The constant0 section holds kernel parameters.
    // Layout: parameter data at their declared offsets.
    // We allocate const_mem_bytes or at least enough to hold all params.
    let min_size = kernel
        .params
        .iter()
        .map(|(_, off, sz)| (off + sz) as usize)
        .max()
        .unwrap_or(0);
    let size = std::cmp::max(kernel.const_mem_bytes as usize, min_size);
    if size == 0 {
        // Even if no params, provide a minimal constant0 section (NVIDIA does 160 bytes min).
        return vec![0u8; 160];
    }
    let data = vec![0u8; std::cmp::max(size, 160)];
    // Parameters could be filled in by the driver at launch; leave as zero for now.
    data
}

// ---------------------------------------------------------------------------
// Symbol table builder
// ---------------------------------------------------------------------------

struct SymbolEntry {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}

impl SymbolEntry {
    fn write_to(&self, buf: &mut Vec<u8>) {
        write_u32(buf, self.st_name);
        buf.push(self.st_info);
        buf.push(self.st_other);
        write_u16(buf, self.st_shndx);
        write_u64(buf, self.st_value);
        write_u64(buf, self.st_size);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a CUBIN (CUDA Binary) ELF file from a `SassModule`.
///
/// The result is a `Vec<u8>` containing a valid ELF64 binary with:
/// - Machine type EM_CUDA (190)
/// - `.text.<kernel>` sections with encoded SASS
/// - `.nv.info.<kernel>` sections with kernel metadata
/// - `.nv.constant0.<kernel>` sections for parameters
/// - A global `.nv.info` section
/// - Symbol table and string tables
pub fn build_cubin_from_module(module: &SassModule) -> Result<Vec<u8>, NvSassError> {
    // 1. Encode all kernels.
    let mut encoded_kernels: Vec<Vec<u8>> = Vec::new();
    for kernel in &module.kernels {
        let encoded_insts = crate::encoding::encode_kernel(kernel, module.sm_version)?;
        let mut text_data: Vec<u8> = Vec::new();
        for inst_bytes in &encoded_insts {
            text_data.extend_from_slice(inst_bytes);
        }
        encoded_kernels.push(text_data);
    }

    // 2. Build section name string table (.shstrtab).
    let mut shstrtab = StringTable::new();
    let shstrtab_name_idx = shstrtab.add(".shstrtab");
    let strtab_name_idx = shstrtab.add(".strtab");
    let symtab_name_idx = shstrtab.add(".symtab");
    let nv_info_global_name_idx = shstrtab.add(".nv.info");

    // Per-kernel section name indices
    let mut text_name_indices: Vec<u32> = Vec::new();
    let mut nv_info_name_indices: Vec<u32> = Vec::new();
    let mut const0_name_indices: Vec<u32> = Vec::new();
    for kernel in &module.kernels {
        text_name_indices.push(shstrtab.add(&format!(".text.{}", kernel.name)));
        nv_info_name_indices.push(shstrtab.add(&format!(".nv.info.{}", kernel.name)));
        const0_name_indices.push(shstrtab.add(&format!(".nv.constant0.{}", kernel.name)));
    }

    // 3. Build symbol name string table (.strtab).
    let mut strtab = StringTable::new();
    // Add kernel names as symbols
    let mut kernel_name_indices: Vec<u32> = Vec::new();
    for kernel in &module.kernels {
        kernel_name_indices.push(strtab.add(&kernel.name));
    }

    // 4. Plan sections.
    //
    // Section layout:
    //   [0] NULL
    //   [1] .shstrtab
    //   [2] .strtab
    //   [3] .symtab
    //   [4] .nv.info  (global)
    //   For each kernel i (0-based):
    //     [5 + 3*i + 0] .text.<name>
    //     [5 + 3*i + 1] .nv.info.<name>
    //     [5 + 3*i + 2] .nv.constant0.<name>

    let num_kernels = module.kernels.len();
    let num_sections = 5 + 3 * num_kernels; // NULL + shstrtab + strtab + symtab + nv.info_global + 3*kernels
    let shstrtab_idx: u16 = 1;
    let strtab_idx: u16 = 2;
    let symtab_idx: u16 = 3;
    let _nv_info_global_idx: u16 = 4;

    fn text_section_idx(kernel_idx: usize) -> u16 {
        (5 + 3 * kernel_idx) as u16
    }
    #[allow(dead_code)]
    fn nv_info_section_idx(kernel_idx: usize) -> u16 {
        (5 + 3 * kernel_idx + 1) as u16
    }
    #[allow(dead_code)]
    fn const0_section_idx(kernel_idx: usize) -> u16 {
        (5 + 3 * kernel_idx + 2) as u16
    }

    // 5. Build .nv.info data per kernel.
    let mut nv_info_datas: Vec<Vec<u8>> = Vec::new();
    for (i, kernel) in module.kernels.iter().enumerate() {
        nv_info_datas.push(build_nv_info_for_kernel(kernel, text_section_idx(i)));
    }

    // Build per-kernel constant0 data.
    let mut const0_datas: Vec<Vec<u8>> = Vec::new();
    for kernel in &module.kernels {
        const0_datas.push(build_constant0_for_kernel(kernel));
    }

    // Global .nv.info (can be empty for minimal CUBIN).
    let nv_info_global_data: Vec<u8> = Vec::new();

    // 6. Build symbol table.
    let mut symbols: Vec<SymbolEntry> = Vec::new();

    // Symbol 0: null symbol (required by ELF)
    symbols.push(SymbolEntry {
        st_name: 0,
        st_info: 0,
        st_other: 0,
        st_shndx: 0,
        st_value: 0,
        st_size: 0,
    });

    // Section symbols for each text section (local)
    for i in 0..num_kernels {
        symbols.push(SymbolEntry {
            st_name: 0,
            st_info: (STB_LOCAL << 4) | STT_SECTION,
            st_other: 0,
            st_shndx: text_section_idx(i),
            st_value: 0,
            st_size: 0,
        });
    }

    let first_global = symbols.len() as u32;

    // Kernel function symbols (global)
    for (i, _kernel) in module.kernels.iter().enumerate() {
        symbols.push(SymbolEntry {
            st_name: kernel_name_indices[i],
            st_info: (STB_GLOBAL << 4) | STT_FUNC,
            st_other: 0,
            st_shndx: text_section_idx(i),
            st_value: 0,
            st_size: encoded_kernels[i].len() as u64,
        });
    }

    let symtab_data = {
        let mut buf = Vec::new();
        for sym in &symbols {
            sym.write_to(&mut buf);
        }
        buf
    };

    // 7. Now lay out the file.
    //
    // File structure:
    //   [ELF header]                (64 bytes)
    //   [section data, packed]      (variable)
    //   [section header table]      (num_sections * 64 bytes)
    //
    // We write data sequentially after the ELF header, tracking offsets.

    let mut output = Vec::new();

    // Reserve space for ELF header (64 bytes).
    output.resize(ELF64_EHDR_SIZE as usize, 0);

    // Section headers to fill in.
    let mut section_headers: Vec<SectionHeader> = Vec::new();

    // [0] NULL section header
    section_headers.push(SectionHeader {
        sh_name: 0,
        sh_type: SHT_NULL,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: 0,
        sh_size: 0,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 0,
        sh_entsize: 0,
    });

    // [1] .shstrtab — we'll finalize data at the end, but write placeholder offset
    let shstrtab_offset = output.len() as u64;
    let shstrtab_bytes = shstrtab.as_bytes();
    output.extend_from_slice(shstrtab_bytes);
    let shstrtab_size = shstrtab_bytes.len() as u64;
    section_headers.push(SectionHeader {
        sh_name: shstrtab_name_idx,
        sh_type: SHT_STRTAB,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: shstrtab_offset,
        sh_size: shstrtab_size,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 1,
        sh_entsize: 0,
    });

    // [2] .strtab
    let strtab_offset = output.len() as u64;
    let strtab_bytes = strtab.as_bytes();
    output.extend_from_slice(strtab_bytes);
    let strtab_size = strtab_bytes.len() as u64;
    section_headers.push(SectionHeader {
        sh_name: strtab_name_idx,
        sh_type: SHT_STRTAB,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: strtab_offset,
        sh_size: strtab_size,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 1,
        sh_entsize: 0,
    });

    // [3] .symtab
    align_to(&mut output, 8);
    let symtab_offset = output.len() as u64;
    output.extend_from_slice(&symtab_data);
    let symtab_size = symtab_data.len() as u64;
    section_headers.push(SectionHeader {
        sh_name: symtab_name_idx,
        sh_type: SHT_SYMTAB,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: symtab_offset,
        sh_size: symtab_size,
        sh_link: strtab_idx as u32, // link to .strtab
        sh_info: first_global,      // index of first global symbol
        sh_addralign: 8,
        sh_entsize: ELF64_SYM_SIZE as u64,
    });

    // [4] .nv.info (global)
    let nv_info_global_offset = output.len() as u64;
    output.extend_from_slice(&nv_info_global_data);
    let nv_info_global_size = nv_info_global_data.len() as u64;
    section_headers.push(SectionHeader {
        sh_name: nv_info_global_name_idx,
        sh_type: SHT_CUDA_INFO,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: nv_info_global_offset,
        sh_size: nv_info_global_size,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 4,
        sh_entsize: 0,
    });

    // Per-kernel sections
    for i in 0..num_kernels {
        // .text.<kernel>
        align_to(&mut output, TEXT_ALIGNMENT as usize);
        let text_offset = output.len() as u64;
        let text_data = &encoded_kernels[i];
        output.extend_from_slice(text_data);
        let text_size = text_data.len() as u64;
        section_headers.push(SectionHeader {
            sh_name: text_name_indices[i],
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addr: 0,
            sh_offset: text_offset,
            sh_size: text_size,
            sh_link: symtab_idx as u32,
            // sh_info for .text sections: the symbol table index of the kernel.
            // The kernel global symbol starts at `first_global + i`.
            sh_info: first_global + i as u32,
            sh_addralign: TEXT_ALIGNMENT,
            sh_entsize: 0,
        });

        // .nv.info.<kernel>
        align_to(&mut output, 4);
        let nv_info_offset = output.len() as u64;
        let nv_info_data = &nv_info_datas[i];
        output.extend_from_slice(nv_info_data);
        let nv_info_size = nv_info_data.len() as u64;
        section_headers.push(SectionHeader {
            sh_name: nv_info_name_indices[i],
            sh_type: SHT_CUDA_INFO,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: nv_info_offset,
            sh_size: nv_info_size,
            sh_link: 0,
            sh_info: text_section_idx(i) as u32,
            sh_addralign: 4,
            sh_entsize: 0,
        });

        // .nv.constant0.<kernel>
        align_to(&mut output, 4);
        let const0_offset = output.len() as u64;
        let const0_data = &const0_datas[i];
        output.extend_from_slice(const0_data);
        let const0_size = const0_data.len() as u64;
        section_headers.push(SectionHeader {
            sh_name: const0_name_indices[i],
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC,
            sh_addr: 0,
            sh_offset: const0_offset,
            sh_size: const0_size,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 4,
            sh_entsize: 0,
        });
    }

    // 8. Write section header table.
    align_to(&mut output, 8);
    let shdr_offset = output.len() as u64;
    for sh in &section_headers {
        sh.write_to(&mut output);
    }

    // 9. Patch the ELF header.
    {
        let hdr = &mut output[..ELF64_EHDR_SIZE as usize];

        // e_ident
        hdr[0..4].copy_from_slice(&ELFMAG);
        hdr[4] = ELFCLASS64;
        hdr[5] = ELFDATA2LSB;
        hdr[6] = EV_CURRENT;
        hdr[7] = ELFOSABI_NONE;
        // EI_PAD [8..16] already zero

        // e_type (offset 16, 2 bytes)
        hdr[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());

        // e_machine (offset 18, 2 bytes)
        hdr[18..20].copy_from_slice(&EM_CUDA.to_le_bytes());

        // e_version (offset 20, 4 bytes)
        hdr[20..24].copy_from_slice(&1u32.to_le_bytes()); // EV_CURRENT

        // e_entry (offset 24, 8 bytes) — 0 for CUBIN
        hdr[24..32].copy_from_slice(&0u64.to_le_bytes());

        // e_phoff (offset 32, 8 bytes) — no program headers
        hdr[32..40].copy_from_slice(&0u64.to_le_bytes());

        // e_shoff (offset 40, 8 bytes)
        hdr[40..48].copy_from_slice(&shdr_offset.to_le_bytes());

        // e_flags (offset 48, 4 bytes) — SM version
        hdr[48..52].copy_from_slice(&module.sm_version.to_le_bytes());

        // e_ehsize (offset 52, 2 bytes)
        hdr[52..54].copy_from_slice(&ELF64_EHDR_SIZE.to_le_bytes());

        // e_phentsize (offset 54, 2 bytes) — 0 (no program headers)
        hdr[54..56].copy_from_slice(&0u16.to_le_bytes());

        // e_phnum (offset 56, 2 bytes)
        hdr[56..58].copy_from_slice(&0u16.to_le_bytes());

        // e_shentsize (offset 58, 2 bytes)
        hdr[58..60].copy_from_slice(&ELF64_SHDR_SIZE.to_le_bytes());

        // e_shnum (offset 60, 2 bytes)
        hdr[60..62].copy_from_slice(&(num_sections as u16).to_le_bytes());

        // e_shstrndx (offset 62, 2 bytes)
        hdr[62..64].copy_from_slice(&shstrtab_idx.to_le_bytes());
    }

    Ok(output)
}
