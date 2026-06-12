// DWARF debug information generation for PTX to target architecture mapping
// This module provides functionality to maintain mappings from PTX source to
// compiled target code (SASS/AMD GCN/Intel SPIRV) for program state recovery

use super::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use llvm_zluda::core::*;
use llvm_zluda::debuginfo::*;
use llvm_zluda::prelude::*;
use llvm_zluda::*;
use serde::{Deserialize, Serialize};

/// PTX source location information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtxSourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub instruction_offset: usize,
}

/// Target architecture instruction mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetInstruction {
    AmdGcn {
        instruction: String,
        address: u64,
        register_state: HashMap<String, String>,
    },
    IntelSpirv {
        instruction: String,
        opcode: u32,
        operands: Vec<String>,
    },
    Sass {
        instruction: String,
        address: u64,
        predicate: Option<String>,
    },
}

/// DWARF mapping entry that connects PTX source to target instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwarfMappingEntry {
    pub ptx_location: PtxSourceLocation,
    pub target_instructions: Vec<TargetInstruction>,
    pub variable_mappings: HashMap<String, VariableLocation>,
    pub scope_id: u64,
}

/// Variable location in target architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableLocation {
    Register(String),
    Memory { address: u64, size: u32 },
    Constant(String),
}

/// DWARF debug info builder for PTX compilation
pub struct PtxDwarfBuilder {
    pub context: LLVMContextRef,
    pub module: LLVMModuleRef,
    di_builder: *mut llvm_zluda::LLVMOpaqueDIBuilder,
    pub compile_unit: LLVMMetadataRef,
    pub file: LLVMMetadataRef,
    source_mappings: Vec<DwarfMappingEntry>,
    current_scope: LLVMMetadataRef,
    variable_counter: u64,
}

impl PtxDwarfBuilder {
    /// Create a new DWARF builder for PTX compilation
    pub unsafe fn new(
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
    ) -> Result<Self, String> {
        // Check environment variable first
        if std::env::var("PTX_DISABLE_DEBUG_INFO").is_ok() {
            return Err("Debug information disabled via environment variable".into());
        }

        let di_builder = LLVMCreateDIBuilder(module);
        if di_builder.is_null() {
            return Err("Failed to create DIBuilder".to_string());
        }

        let producer_cstr = CString::new(producer).map_err(|_| "Invalid producer string")?;
        let filename_cstr = CString::new(filename).unwrap();
        let directory_cstr = CString::new(".").unwrap();
        let di_file = LLVMDIBuilderCreateFile(
            di_builder,
            filename_cstr.as_ptr(),
            filename_cstr.as_bytes().len(),
            directory_cstr.as_ptr(),
            directory_cstr.as_bytes().len(),
        );

        // Create compile unit with proper parameters
        let di_compile_unit = LLVMDIBuilderCreateCompileUnit(
            di_builder,
            llvm_zluda::debuginfo::LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC99, // Use C99 for compatibility like the example
            di_file,
            producer_cstr.as_ptr(),
            producer_cstr.as_bytes().len(),
            1,           // isOptimized: true (like the working example)
            ptr::null(), // no flags
            0,           // flags length
            0,           // runtime version
            ptr::null(), // split name
            0,           // split name length
            llvm_zluda::debuginfo::LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
            0,           // DWO ID
            1,           // split debug inlining
            0,           // debug info for profiling
            ptr::null(), // sysroot
            0,           // sysroot length
            ptr::null(), // SDK
            0,           // SDK length
        );

        let debug_context = Self {
            context,
            module,
            di_builder,
            compile_unit: di_compile_unit,
            file: di_file,
            source_mappings: Vec::new(),
            current_scope: di_compile_unit, // Use compile unit as initial scope
            variable_counter: 0,
        };

        // Set DWARF version metadata to fix "invalid version (0)" error
        // Only add module flags if they don't already exist
        let dwarf_version_str = CString::new("Dwarf Version").unwrap();
        let existing_dwarf_flag = LLVMGetModuleFlag(
            module,
            dwarf_version_str.as_ptr(),
            dwarf_version_str.as_bytes().len(),
        );
        if existing_dwarf_flag.is_null() {
            let version_val = LLVMConstInt(LLVMInt32TypeInContext(context), 2, 0); // DWARF version 2
            let version_metadata = LLVMValueAsMetadata(version_val);
            LLVMAddModuleFlag(
                module,
                LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning, // Use Warning instead of Error
                dwarf_version_str.as_ptr(),
                dwarf_version_str.as_bytes().len(),
                version_metadata,
            );
        }

        // Also set Debug Info Version
        let debug_info_version_str = CString::new("Debug Info Version").unwrap();
        let existing_debug_flag = LLVMGetModuleFlag(
            module,
            debug_info_version_str.as_ptr(),
            debug_info_version_str.as_bytes().len(),
        );
        if existing_debug_flag.is_null() {
            let debug_version_val = LLVMConstInt(LLVMInt32TypeInContext(context), 3, 0); // Debug Info Version 3
            let debug_version_metadata = LLVMValueAsMetadata(debug_version_val);
            LLVMAddModuleFlag(
                module,
                LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError,
                debug_info_version_str.as_ptr(),
                debug_info_version_str.as_bytes().len(),
                debug_version_metadata,
            );
        }

        // Skip llvm.ident metadata creation to avoid LLVM validation errors
        // Producer information is already included in the compile unit

        Ok(debug_context)
    }

    /// Add a PTX source to target instruction mapping
    pub fn add_mapping(&mut self, mapping: DwarfMappingEntry) {
        self.source_mappings.push(mapping);
    }

    /// Create debug types for PTX scalar types
    pub unsafe fn create_ptx_debug_type(
        &self,
        ptx_type: &ptx_parser::ScalarType,
    ) -> LLVMMetadataRef {
        let (name, size_bits, encoding) = match ptx_type {
            ptx_parser::ScalarType::U8 => ("u8", 8, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U16 => ("u16", 16, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U32 => ("u32", 32, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U64 => ("u64", 64, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::S8 => ("s8", 8, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S16 => ("s16", 16, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S32 => ("s32", 32, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S64 => ("s64", 64, 5), // DW_ATE_signed
            ptx_parser::ScalarType::F16 => ("f16", 16, 4), // DW_ATE_float
            ptx_parser::ScalarType::F32 => ("f32", 32, 4), // DW_ATE_float
            ptx_parser::ScalarType::F64 => ("f64", 64, 4), // DW_ATE_float
            ptx_parser::ScalarType::Pred => ("pred", 1, 2), // DW_ATE_boolean
            ptx_parser::ScalarType::B8 => ("b8", 8, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B16 => ("b16", 16, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B32 => ("b32", 32, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B64 => ("b64", 64, 7), // DW_ATE_unsigned
            _ => ("unknown", 32, 7),                    // Default fallback
        };

        let name_cstr = CString::new(name).unwrap();
        LLVMDIBuilderCreateBasicType(
            self.di_builder,
            name_cstr.as_ptr(),
            name.len(),
            size_bits,
            encoding,
            0, // flags
        )
    }

    /// Create debug info for function parameters
    pub unsafe fn create_parameter_debug_info(
        &self,
        function_scope: LLVMMetadataRef,
        param_name: &str,
        param_type: &ptx_parser::ScalarType,
        arg_num: u32,
        line: u32,
    ) -> LLVMMetadataRef {
        let param_name_cstr = CString::new(param_name).unwrap();
        let param_debug_type = self.create_ptx_debug_type(param_type);

        LLVMDIBuilderCreateParameterVariable(
            self.di_builder,
            function_scope,
            param_name_cstr.as_ptr(),
            param_name.len(),
            arg_num,
            self.file,
            line,
            param_debug_type,
            1, // alwaysPreserve
            0, // flags
        )
    }

    /// Create debug info for local variables
    pub unsafe fn create_local_variable_debug_info(
        &self,
        scope: LLVMMetadataRef,
        var_name: &str,
        var_type: &ptx_parser::ScalarType,
        line: u32,
    ) -> LLVMMetadataRef {
        let var_name_cstr = CString::new(var_name).unwrap();
        let var_debug_type = self.create_ptx_debug_type(var_type);

        LLVMDIBuilderCreateAutoVariable(
            self.di_builder,
            scope,
            var_name_cstr.as_ptr(),
            var_name.len(),
            self.file,
            line,
            var_debug_type,
            1, // alwaysPreserve
            0, // flags
            0, // alignInBits
        )
    }

    /// Create lexical block for improved scope tracking
    pub unsafe fn create_lexical_block(
        &self,
        parent_scope: LLVMMetadataRef,
        line: u32,
        column: u32,
    ) -> LLVMMetadataRef {
        LLVMDIBuilderCreateLexicalBlock(self.di_builder, parent_scope, self.file, line, column)
    }

    /// Create function debug info with parameters
    pub unsafe fn create_function_debug_info(
        &self,
        function_name: &str,
        linkage_name: &str,
        line: u32,
        is_definition: bool,
        param_types: &[ptx_parser::ScalarType],
    ) -> LLVMMetadataRef {
        let function_name_cstr = CString::new(function_name).unwrap();
        let linkage_name_cstr = CString::new(linkage_name).unwrap();

        // Create function type
        let void_type = LLVMDIBuilderCreateBasicType(self.di_builder, c"void".as_ptr(), 4, 0, 0, 0);

        // Create parameter types array
        let mut param_debug_types = vec![void_type]; // Return type first
        for param_type in param_types {
            param_debug_types.push(self.create_ptx_debug_type(param_type));
        }

        let function_type = LLVMDIBuilderCreateSubroutineType(
            self.di_builder,
            self.file,
            param_debug_types.as_mut_ptr(),
            param_debug_types.len() as u32,
            0, // flags
        );

        LLVMDIBuilderCreateFunction(
            self.di_builder,
            self.file, // scope
            function_name_cstr.as_ptr(),
            function_name.len(),
            linkage_name_cstr.as_ptr(),
            linkage_name.len(),
            self.file,
            line,
            function_type,
            0, // isLocalToUnit (isLocal: false)
            is_definition as i32,
            line, // scopeLine
            0,    // flags
            1,    // isOptimized (true like the working example)
        )
    }

    /// Create debug location for PTX source line
    pub unsafe fn create_debug_location(
        &self,
        line: u32,
        column: u32,
        scope: Option<LLVMMetadataRef>,
    ) -> Result<LLVMMetadataRef, String> {
        let scope_ref = scope.unwrap_or(self.current_scope);

        let debug_loc = LLVMDIBuilderCreateDebugLocation(
            self.context,
            line,
            column,
            scope_ref,
            ptr::null_mut(), // no inlined_at
        );

        if debug_loc.is_null() {
            return Err("Failed to create debug location".to_string());
        }

        Ok(debug_loc)
    }

    /// Create variable debug info with enhanced PTX variable and memory address tracking
    pub unsafe fn create_variable_debug_info(
        &mut self,
        name: &str,
        line: u32,
        var_type: LLVMMetadataRef,
        location: &VariableLocation,
        function_scope: Option<LLVMMetadataRef>,
    ) -> Result<LLVMMetadataRef, String> {
        let name_cstr = CString::new(name).map_err(|_| "Invalid variable name")?;

        // Use function scope if provided, otherwise try to use current_scope only if it's a valid local scope
        let valid_scope = function_scope.unwrap_or_else(|| {
            // If no function scope provided, we can't create local variables safely
            // Return null to indicate this variable should be skipped
            ptr::null_mut()
        });

        if valid_scope.is_null() {
            return Err("No valid function scope available for local variable".to_string());
        }

        // Create enhanced variable with PTX-specific attributes based on location
        let di_variable = match location {
            VariableLocation::Memory { address, size } => {
                // Create memory-based variable with address annotation
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for memory variable
                self.add_memory_variable_mapping(name, line, *address, *size)?;
                var
            }
            VariableLocation::Register(reg_name) => {
                // Create register-based variable with register annotation
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for register variable
                self.add_register_variable_mapping(name, line, reg_name)?;
                var
            }
            VariableLocation::Constant(value) => {
                // Create constant variable
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for constant variable
                self.add_constant_variable_mapping(name, line, value)?;
                var
            }
        };

        self.variable_counter += 1;
        Ok(di_variable)
    }

    /// Add memory variable mapping with address tracking
    fn add_memory_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        address: u64,
        size: u32,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Memory { address, size },
        );

        // Create target instruction with memory address information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpVariable_{}", var_name),
            opcode: 0x3B, // OpVariable in SPIR-V
            operands: vec![
                format!("ptr_0x{:016x}", address),
                format!("size_{}", size),
                format!("type_memory"),
                format!("storage_class_function"),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Add register variable mapping with register tracking
    fn add_register_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        reg_name: &str,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Register(reg_name.to_string()),
        );

        // Create target instruction with register information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpLoad_{}", var_name),
            opcode: 0x3D, // OpLoad in SPIR-V
            operands: vec![
                format!("reg_{}", reg_name),
                format!("type_register"),
                self.parse_register_type(reg_name),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Add constant variable mapping
    fn add_constant_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        value: &str,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Constant(value.to_string()),
        );

        // Create target instruction with constant information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpConstant_{}", var_name),
            opcode: 0x2B, // OpConstant in SPIR-V
            operands: vec![
                format!("value_{}", value),
                format!("type_constant"),
                self.parse_constant_type(value),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Parse PTX register type from register name
    fn parse_register_type(&self, reg_name: &str) -> String {
        if reg_name.starts_with("%r") {
            "int32".to_string()
        } else if reg_name.starts_with("%f") {
            "float32".to_string()
        } else if reg_name.starts_with("%d") {
            "float64".to_string()
        } else if reg_name.starts_with("%p") {
            "predicate".to_string()
        } else if reg_name.starts_with("%h") {
            "int16".to_string()
        } else if reg_name.starts_with("%c") {
            "int8".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Parse constant type from value
    fn parse_constant_type(&self, value: &str) -> String {
        if value.contains('.') {
            "float".to_string()
        } else if value.starts_with('-') || value.parse::<i64>().is_ok() {
            "integer".to_string()
        } else {
            "string".to_string()
        }
    }

    /// Create basic type debug info (for PTX types)
    pub unsafe fn create_basic_type(
        &self,
        name: &str,
        size_in_bits: u64,
        encoding: u32,
    ) -> Result<LLVMMetadataRef, String> {
        let name_cstr = CString::new(name).map_err(|_| "Invalid type name")?;

        Ok(LLVMDIBuilderCreateBasicType(
            self.di_builder,
            name_cstr.as_ptr(),
            name_cstr.as_bytes().len(),
            size_in_bits,
            encoding,
            0, // flags
        ))
    }

    /// Create a function/subroutine type for debug info
    pub unsafe fn create_function_type(
        &self,
        return_type: Option<LLVMMetadataRef>,
        parameter_types: &[LLVMMetadataRef],
    ) -> Result<LLVMMetadataRef, String> {
        // Create array of parameter types
        let mut all_types = Vec::new();

        // Add return type as first element (LLVM convention)
        if let Some(ret_type) = return_type {
            all_types.push(ret_type);
        } else {
            // Create void type for no return
            let void_type = self.create_basic_type("void", 0, 0)?;
            all_types.push(void_type);
        }

        // Add parameter types
        all_types.extend_from_slice(parameter_types);

        Ok(LLVMDIBuilderCreateSubroutineType(
            self.di_builder,
            self.file, // file
            all_types.as_mut_ptr(),
            all_types.len() as u32,
            0, // flags
        ))
    }

    /// Create a compile unit for the module
    pub unsafe fn create_compile_unit(&mut self) -> Result<(), String> {
        // The compile unit is already created in the constructor, so this is a no-op
        Ok(())
    }

    /// Get all source mappings for state recovery
    pub fn get_mappings(&self) -> &[DwarfMappingEntry] {
        &self.source_mappings
    }

    /// Find mapping by PTX source location
    pub fn find_mapping_by_location(&self, line: u32, column: u32) -> Option<&DwarfMappingEntry> {
        self.source_mappings.iter().find(|mapping| {
            mapping.ptx_location.line == line && mapping.ptx_location.column == column
        })
    }

    /// Export mappings for external debugger integration
    pub fn export_mapping_table(&self) -> String {
        let mut output = String::new();
        output.push_str("# PTX to Target Architecture Debug Mapping\n");
        output.push_str("# Format: ptx_line:ptx_col -> target_instructions\n\n");

        for mapping in &self.source_mappings {
            output.push_str(&format!(
                "{}:{}:{} -> [\n",
                mapping.ptx_location.file, mapping.ptx_location.line, mapping.ptx_location.column
            ));

            for (i, target_inst) in mapping.target_instructions.iter().enumerate() {
                match target_inst {
                    TargetInstruction::AmdGcn {
                        instruction,
                        address,
                        ..
                    } => {
                        output.push_str(&format!(
                            "  AMD_GCN[{}]: {} @ 0x{:x}\n",
                            i, instruction, address
                        ));
                    }
                    TargetInstruction::IntelSpirv {
                        instruction,
                        opcode,
                        ..
                    } => {
                        output.push_str(&format!(
                            "  SPIRV[{}]: {} (opcode: {})\n",
                            i, instruction, opcode
                        ));
                    }
                    TargetInstruction::Sass {
                        instruction,
                        address,
                        predicate,
                    } => {
                        let pred_str = predicate
                            .as_ref()
                            .map(|p| format!(" [{}]", p))
                            .unwrap_or_default();
                        output.push_str(&format!(
                            "  SASS[{}]: {} @ 0x{:x}{}\n",
                            i, instruction, address, pred_str
                        ));
                    }
                }
            }
            output.push_str("]\n\n");
        }

        output
    }

    /// Get the underlying DIBuilder (deprecated - use get_builder_ref instead)
    pub fn get_builder(&self) -> LLVMDIBuilderRef {
        self.di_builder
    }

    /// Finalize debug information generation
    pub unsafe fn finalize(&self) {
        if !self.di_builder.is_null() {
            LLVMDIBuilderFinalize(self.di_builder);
        }
    }

    /// Finalize the compilation unit
    pub unsafe fn finalize_compile_unit(&self) -> Result<(), String> {
        // For LLVM, we don't need to do anything specific to finalize the compile unit
        // The debug info is finalized automatically when the module is compiled
        Ok(())
    }

    /// Clear all debug locations to prevent invalid records
    pub unsafe fn clear_debug_locations(&self) -> Result<(), String> {
        // Set null debug location on all DIBuilders
        LLVMDIBuilderFinalize(self.di_builder);

        // Clear any pending nodes by finalizing
        LLVMDIBuilderFinalize(self.di_builder);

        Ok(())
    }

    /// Track PTX variable assignment with llvm.dbg.value calls
    pub unsafe fn track_ptx_variable_assignment(
        &mut self,
        builder: LLVMBuilderRef,
        variable_name: &str,
        value: LLVMValueRef,
        line: u32,
    ) -> Result<(), String> {
        // Generate unique variable name based on counter
        self.variable_counter += 1;
        let var_name = format!("var_{}", self.variable_counter);

        // Create debug variable info
        let var_name_cstr = CString::new(var_name.clone()).unwrap();
        let value_type = LLVMTypeOf(value);

        // Determine type size based on LLVM type
        let (type_name, type_size, encoding) =
            if LLVMGetTypeKind(value_type) == LLVMTypeKind::LLVMIntegerTypeKind {
                let bit_width = LLVMGetIntTypeWidth(value_type);
                match bit_width {
                    32 => ("i32", 32, 5), // DW_ATE_signed
                    64 => ("i64", 64, 5), // DW_ATE_signed
                    _ => ("i32", 32, 5),  // DW_ATE_signed
                }
            } else {
                ("i32", 32, 5) // DW_ATE_signed
            };

        let type_name_cstr = CString::new(type_name).unwrap();
        let debug_type = LLVMDIBuilderCreateBasicType(
            self.di_builder,
            type_name_cstr.as_ptr(),
            type_name_cstr.as_bytes().len(),
            type_size,
            encoding,
            0,
        );

        // Create debug variable
        let debug_var = LLVMDIBuilderCreateAutoVariable(
            self.di_builder,
            self.current_scope,
            var_name_cstr.as_ptr(),
            var_name_cstr.as_bytes().len(),
            self.file,
            line,
            debug_type,
            1, // always preserve
            0, // no flags
            0, // alignment
        );

        // Create debug location
        let debug_loc = LLVMDIBuilderCreateDebugLocation(
            self.context,
            line,
            1, // column
            self.current_scope,
            ptr::null_mut(),
        );

        // Get or declare llvm.dbg.value function
        let dbg_value_name = CString::new("llvm.dbg.value").unwrap();
        let mut dbg_value_fn = LLVMGetNamedFunction(self.module, dbg_value_name.as_ptr());

        if dbg_value_fn.is_null() {
            // Declare llvm.dbg.value function
            let void_type = LLVMVoidTypeInContext(self.context);
            let metadata_type = LLVMMetadataTypeInContext(self.context);
            let param_types = [metadata_type, metadata_type, metadata_type];
            let function_type = LLVMFunctionType(
                void_type,
                param_types.as_ptr() as *mut _,
                param_types.len() as u32,
                0, // not variadic
            );

            dbg_value_fn = LLVMAddFunction(self.module, dbg_value_name.as_ptr(), function_type);
        }

        // Validate the input value first
        if value.is_null() {
            return Err("Cannot create debug metadata for null value".to_string());
        }

        // Create call to llvm.dbg.value
        let value_metadata = LLVMValueAsMetadata(value);
        let var_metadata = debug_var;
        let expr_metadata = LLVMDIBuilderCreateExpression(self.di_builder, ptr::null_mut(), 0);

        // Validate that all metadata is non-null before proceeding
        if value_metadata.is_null() || var_metadata.is_null() || expr_metadata.is_null() {
            return Err("Failed to create valid metadata for llvm.dbg.value call".to_string());
        }

        let args = [
            LLVMMetadataAsValue(self.context, value_metadata),
            LLVMMetadataAsValue(self.context, var_metadata),
            LLVMMetadataAsValue(self.context, expr_metadata),
        ];

        // Use the same function type that was used to declare the function
        let void_type = LLVMVoidTypeInContext(self.context);
        let metadata_type = LLVMMetadataTypeInContext(self.context);
        let param_types = [metadata_type, metadata_type, metadata_type];
        let function_type = LLVMFunctionType(
            void_type,
            param_types.as_ptr() as *mut _,
            param_types.len() as u32,
            0, // not variadic
        );
        let call = LLVMBuildCall2(
            builder,
            function_type,
            dbg_value_fn,
            args.as_ptr() as *mut _,
            args.len() as u32,
            CString::new("").unwrap().as_ptr(),
        );

        // Set debug location for the call
        LLVMSetCurrentDebugLocation2(builder, debug_loc);

        println!(
            "DEBUG_VALUE: {}:{} <- loaded_value (line {}) [llvm.dbg.value created]",
            variable_name, var_name, line
        );

        Ok(())
    }

    /// Set function debug info (adds !dbg metadata to function)
    pub unsafe fn set_function_debug_info(
        &self,
        function: LLVMValueRef,
        function_di: LLVMMetadataRef,
    ) -> Result<(), String> {
        // Set the function's debug info using LLVMSetSubprogram
        LLVMSetSubprogram(function, function_di);
        Ok(())
    }

    /// Get the context for external use
    pub fn get_context(&self) -> LLVMContextRef {
        self.context
    }

    /// Get the module for external use
    pub fn get_module(&self) -> LLVMModuleRef {
        self.module
    }
}

impl Drop for PtxDwarfBuilder {
    fn drop(&mut self) {
        unsafe {
            if !self.di_builder.is_null() {
                LLVMDisposeDIBuilder(self.di_builder);
            }
        }
    }
}

/// State recovery mechanism using DWARF mappings
pub struct PtxStateRecovery {
    mappings: Vec<DwarfMappingEntry>,
    current_execution_point: Option<PtxSourceLocation>,
}

impl PtxStateRecovery {
    pub fn new(mappings: Vec<DwarfMappingEntry>) -> Self {
        Self {
            mappings,
            current_execution_point: None,
        }
    }

    /// Set current execution point in PTX source
    pub fn set_execution_point(&mut self, location: PtxSourceLocation) {
        self.current_execution_point = Some(location);
    }

    /// Recover PTX state from target architecture debugging information
    pub fn recover_ptx_state(&self, target_address: u64) -> Option<PtxSourceLocation> {
        for mapping in &self.mappings {
            for target_inst in &mapping.target_instructions {
                match target_inst {
                    TargetInstruction::AmdGcn { address, .. }
                    | TargetInstruction::Sass { address, .. } => {
                        if *address == target_address {
                            return Some(mapping.ptx_location.clone());
                        }
                    }
                    TargetInstruction::IntelSpirv { .. } => {
                        // SPIRV doesn't have direct address mapping, use opcode matching
                        // This would need runtime integration for proper address translation
                    }
                }
            }
        }
        None
    }

    /// Get variable locations at current execution point
    pub fn get_variable_state(&self) -> Option<&HashMap<String, VariableLocation>> {
        if let Some(ref current_location) = self.current_execution_point {
            for mapping in &self.mappings {
                if mapping.ptx_location == *current_location {
                    return Some(&mapping.variable_mappings);
                }
            }
        }
        None
    }

    /// Export current state for debugging
    pub fn export_state_dump(&self) -> String {
        let mut dump = String::new();

        if let Some(ref location) = self.current_execution_point {
            dump.push_str(&format!(
                "Current PTX execution point: {}:{}:{}\n",
                location.file, location.line, location.column
            ));

            if let Some(var_state) = self.get_variable_state() {
                dump.push_str("Variable state:\n");
                for (name, location) in var_state {
                    match location {
                        VariableLocation::Register(reg) => {
                            dump.push_str(&format!("  {} -> register {}\n", name, reg));
                        }
                        VariableLocation::Memory { address, size } => {
                            dump.push_str(&format!(
                                "  {} -> memory 0x{:x} (size: {})\n",
                                name, address, size
                            ));
                        }
                        VariableLocation::Constant(value) => {
                            dump.push_str(&format!("  {} -> constant {}\n", name, value));
                        }
                    }
                }
            }
        } else {
            dump.push_str("No current execution point set\n");
        }

        dump
    }
}

/// Integration point for adding debug info to PTX compilation pipeline
pub fn integrate_debug_info_generation(
    context: LLVMContextRef,
    module: LLVMModuleRef,
    filename: &str,
) -> Result<PtxDwarfBuilder, String> {
    unsafe { PtxDwarfBuilder::new(context, module, filename, "ZLUDA PTX Compiler") }
}

// ============================================================================
// SASS ↔ PTX Bidirectional Mapping for Runtime Debugging
// ============================================================================

/// Parsed SASS instruction with full context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SassInstruction {
    /// SASS opcode (e.g., "LDG", "STG", "FADD", "IMAD")
    pub opcode: String,
    /// Full instruction text
    pub instruction: String,
    /// Address/offset in the compiled binary
    pub address: u64,
    /// Predicate register if any (e.g., "@P0", "@!P1")
    pub predicate: Option<String>,
    /// Destination operands
    pub dest_operands: Vec<String>,
    /// Source operands
    pub src_operands: Vec<String>,
    /// Control codes (e.g., ".REUSE", ".YIELD")
    pub control_codes: Vec<String>,
    /// Stall counts from control codes
    pub stall_count: Option<u8>,
    /// Wait barrier mask
    pub wait_barrier: Option<u8>,
    /// Read barrier
    pub read_barrier: Option<u8>,
    /// Write barrier
    pub write_barrier: Option<u8>,
}

/// Line mapping entry from SASS to PTX source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SassLineMapping {
    /// SASS instruction address
    pub sass_address: u64,
    /// SASS instruction offset string (e.g., "0x0050")
    pub sass_offset: String,
    /// Full SASS instruction
    pub sass_instruction: SassInstruction,
    /// PTX source file name (e.g., "kernel.ptx")
    pub ptx_file: String,
    /// PTX source line number
    pub ptx_line: u32,
    /// PTX source column (if available)
    pub ptx_column: u32,
    /// Function name containing this instruction
    pub function_name: Option<String>,
}

/// Bidirectional SASS ↔ PTX mapping table for runtime debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SassPtxMapper {
    /// SASS address → PTX location mapping
    sass_to_ptx: HashMap<u64, PtxSourceLocation>,
    /// PTX location → SASS addresses mapping (key is "file:line")
    ptx_to_sass: HashMap<String, Vec<u64>>,
    /// All line mappings for detailed inspection
    line_mappings: Vec<SassLineMapping>,
    /// Function address ranges (key is function name, value is (start_addr, end_addr))
    function_ranges: HashMap<String, (u64, u64)>,
    /// Original PTX source content for reconstruction
    ptx_source: Option<String>,
    /// CUBIN file path (if loaded from file)
    cubin_path: Option<String>,
}

impl SassPtxMapper {
    /// Create a new empty mapper
    pub fn new() -> Self {
        Self {
            sass_to_ptx: HashMap::new(),
            ptx_to_sass: HashMap::new(),
            line_mappings: Vec::new(),
            function_ranges: HashMap::new(),
            ptx_source: None,
            cubin_path: None,
        }
    }

    /// Create mapper with PTX source for reconstruction
    pub fn with_ptx_source(ptx_source: String) -> Self {
        Self {
            sass_to_ptx: HashMap::new(),
            ptx_to_sass: HashMap::new(),
            line_mappings: Vec::new(),
            function_ranges: HashMap::new(),
            ptx_source: Some(ptx_source),
            cubin_path: None,
        }
    }

    /// Parse cuobjdump -sass -lineinfo output and build mappings
    pub fn parse_cuobjdump_output(&mut self, output: &str) -> Result<(), String> {
        let mut current_file = String::from("kernel.ptx");
        let mut current_function = String::new();
        let mut pending_line: Option<u32> = None;

        for line in output.lines() {
            let line = line.trim();

            // Parse function header: "Function : kernel_name"
            if line.starts_with("Function :") || line.starts_with("function :") {
                if let Some(name) = line.split(':').nth(1) {
                    current_function = name.trim().to_string();
                }
                continue;
            }

            // Parse file reference: ## File "kernel.ptx", line 16
            if line.contains("## File") {
                if let Some(file_part) = line.split('"').nth(1) {
                    current_file = file_part.to_string();
                }
                // Also extract line number from same line
                if let Some(line_part) = line.split("line ").nth(1) {
                    if let Ok(line_num) = line_part
                        .trim_end_matches(|c: char| !c.is_ascii_digit())
                        .parse::<u32>()
                    {
                        pending_line = Some(line_num);
                    }
                }
                continue;
            }

            // Parse line marker: ## Line 16
            if let Some(line_marker) = line.strip_prefix("## Line ") {
                if let Ok(line_num) = line_marker.trim().parse::<u32>() {
                    pending_line = Some(line_num);
                }
                continue;
            }

            // Parse SASS instruction line: /*0050*/ LDG.E R0, [R2.64] ;
            if line.starts_with("/*") {
                if let Some(sass_inst) = self.parse_sass_instruction_line(line) {
                    let ptx_line = pending_line.unwrap_or(0);

                    // Create line mapping
                    let mapping = SassLineMapping {
                        sass_address: sass_inst.address,
                        sass_offset: format!("0x{:04x}", sass_inst.address),
                        sass_instruction: sass_inst.clone(),
                        ptx_file: current_file.clone(),
                        ptx_line,
                        ptx_column: 0,
                        function_name: if current_function.is_empty() {
                            None
                        } else {
                            Some(current_function.clone())
                        },
                    };

                    // Add to bidirectional maps
                    let ptx_loc = PtxSourceLocation {
                        file: current_file.clone(),
                        line: ptx_line,
                        column: 0,
                        instruction_offset: sass_inst.address as usize,
                    };

                    self.sass_to_ptx.insert(sass_inst.address, ptx_loc);
                    let key = format!("{}:{}", current_file, ptx_line);
                    self.ptx_to_sass
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(sass_inst.address);
                    self.line_mappings.push(mapping);
                }
            }
        }

        Ok(())
    }

    /// Parse a single SASS instruction line
    fn parse_sass_instruction_line(&self, line: &str) -> Option<SassInstruction> {
        // Format: /*0050*/ @P0 LDG.E.U32 R0, [R2.64+0x10] ;

        // Extract address from /*xxxx*/
        let addr_end = line.find("*/")?;
        let addr_str = line.get(2..addr_end)?.trim();
        let address = u64::from_str_radix(addr_str, 16).ok()?;

        // Get rest of line after address
        let rest = line
            .get(addr_end + 2..)?
            .trim()
            .trim_end_matches(';')
            .trim();

        // Check for predicate (@P0, @!P1, etc.)
        let (predicate, instruction_part) = if rest.starts_with('@') {
            let space_idx = rest.find(' ').unwrap_or(rest.len());
            let pred = rest.get(..space_idx)?;
            (Some(pred.to_string()), rest.get(space_idx..)?.trim())
        } else {
            (None, rest)
        };

        // Split into opcode and operands
        let parts: Vec<&str> = instruction_part.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        let opcode_full = parts[0];
        // Split opcode from modifiers (e.g., "LDG.E.U32" -> "LDG", [".E", ".U32"])
        let opcode_parts: Vec<&str> = opcode_full.split('.').collect();
        let opcode = opcode_parts[0].to_string();
        let control_codes: Vec<String> = opcode_parts[1..]
            .iter()
            .map(|s| format!(".{}", s))
            .collect();

        // Parse operands
        let operands_str = parts[1..].join(" ");
        let (dest_operands, src_operands) = self.parse_operands(&operands_str);

        Some(SassInstruction {
            opcode,
            instruction: line.to_string(),
            address,
            predicate,
            dest_operands,
            src_operands,
            control_codes,
            stall_count: None,
            wait_barrier: None,
            read_barrier: None,
            write_barrier: None,
        })
    }

    /// Parse SASS operands into destination and source
    fn parse_operands(&self, operands: &str) -> (Vec<String>, Vec<String>) {
        let parts: Vec<&str> = operands.split(',').map(|s| s.trim()).collect();

        if parts.is_empty() {
            return (vec![], vec![]);
        }

        // First operand is typically destination, rest are sources
        let dest = vec![parts[0].to_string()];
        let src: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        (dest, src)
    }

    /// Query PTX location from SASS address (for breakpoint hit)
    pub fn sass_to_ptx_location(&self, sass_address: u64) -> Option<&PtxSourceLocation> {
        self.sass_to_ptx.get(&sass_address)
    }

    /// Query PTX location from SASS address with address range tolerance
    pub fn sass_to_ptx_location_nearest(
        &self,
        sass_address: u64,
        tolerance: u64,
    ) -> Option<&PtxSourceLocation> {
        // First try exact match
        if let Some(loc) = self.sass_to_ptx.get(&sass_address) {
            return Some(loc);
        }

        // Find nearest address within tolerance
        let mut best_match: Option<(u64, &PtxSourceLocation)> = None;
        for (addr, loc) in &self.sass_to_ptx {
            let diff = if *addr > sass_address {
                *addr - sass_address
            } else {
                sass_address - *addr
            };
            if diff <= tolerance {
                match best_match {
                    None => best_match = Some((*addr, loc)),
                    Some((best_addr, _)) => {
                        let best_diff = if best_addr > sass_address {
                            best_addr - sass_address
                        } else {
                            sass_address - best_addr
                        };
                        if diff < best_diff {
                            best_match = Some((*addr, loc));
                        }
                    }
                }
            }
        }

        best_match.map(|(_, loc)| loc)
    }

    /// Query SASS addresses from PTX line (for setting breakpoints)
    pub fn ptx_to_sass_addresses(&self, file: &str, line: u32) -> Option<&Vec<u64>> {
        let key = format!("{}:{}", file, line);
        self.ptx_to_sass.get(&key)
    }

    /// Get the first SASS address for a PTX line (primary breakpoint address)
    pub fn ptx_to_sass_address(&self, file: &str, line: u32) -> Option<u64> {
        let key = format!("{}:{}", file, line);
        self.ptx_to_sass
            .get(&key)
            .and_then(|addrs| addrs.first().copied())
    }

    /// Get all mappings for a function
    pub fn get_function_mappings(&self, function_name: &str) -> Vec<&SassLineMapping> {
        self.line_mappings
            .iter()
            .filter(|m| m.function_name.as_deref() == Some(function_name))
            .collect()
    }

    /// Get PTX source line content
    pub fn get_ptx_source_line(&self, line: u32) -> Option<&str> {
        self.ptx_source
            .as_ref()
            .and_then(|source| source.lines().nth((line.saturating_sub(1)) as usize))
    }

    /// Get context around a PTX line (for displaying in debugger)
    pub fn get_ptx_context(&self, line: u32, context_lines: u32) -> Vec<(u32, String)> {
        let mut result = Vec::new();
        if let Some(source) = &self.ptx_source {
            let lines: Vec<&str> = source.lines().collect();
            let start = (line.saturating_sub(context_lines).saturating_sub(1)) as usize;
            let end = std::cmp::min((line + context_lines) as usize, lines.len());

            for (i, line_content) in lines[start..end].iter().enumerate() {
                result.push(((start + i + 1) as u32, line_content.to_string()));
            }
        }
        result
    }

    /// Export mappings to JSON for external tools
    pub fn export_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Import mappings from JSON
    pub fn import_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Add a mapping entry directly
    pub fn add_mapping(
        &mut self,
        sass_addr: u64,
        ptx_file: &str,
        ptx_line: u32,
        instruction: &str,
    ) {
        let ptx_loc = PtxSourceLocation {
            file: ptx_file.to_string(),
            line: ptx_line,
            column: 0,
            instruction_offset: sass_addr as usize,
        };

        self.sass_to_ptx.insert(sass_addr, ptx_loc);
        let key = format!("{}:{}", ptx_file, ptx_line);
        self.ptx_to_sass
            .entry(key)
            .or_insert_with(Vec::new)
            .push(sass_addr);

        // Create minimal SASS instruction
        let sass_inst = SassInstruction {
            opcode: instruction
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string(),
            instruction: instruction.to_string(),
            address: sass_addr,
            predicate: None,
            dest_operands: vec![],
            src_operands: vec![],
            control_codes: vec![],
            stall_count: None,
            wait_barrier: None,
            read_barrier: None,
            write_barrier: None,
        };

        self.line_mappings.push(SassLineMapping {
            sass_address: sass_addr,
            sass_offset: format!("0x{:04x}", sass_addr),
            sass_instruction: sass_inst,
            ptx_file: ptx_file.to_string(),
            ptx_line,
            ptx_column: 0,
            function_name: None,
        });
    }

    /// Get all PTX lines that have SASS mappings
    pub fn get_mapped_ptx_lines(&self) -> Vec<(String, u32)> {
        self.ptx_to_sass
            .keys()
            .filter_map(|key| {
                let parts: Vec<&str> = key.rsplitn(2, ':').collect();
                if parts.len() == 2 {
                    let line = parts[0].parse::<u32>().ok()?;
                    let file = parts[1].to_string();
                    Some((file, line))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all SASS addresses in order
    pub fn get_sass_addresses_ordered(&self) -> Vec<u64> {
        let mut addrs: Vec<u64> = self.sass_to_ptx.keys().copied().collect();
        addrs.sort();
        addrs
    }

    /// Set PTX source for context retrieval
    pub fn set_ptx_source(&mut self, source: String) {
        self.ptx_source = Some(source);
    }

    /// Get PTX source reference
    pub fn get_ptx_source(&self) -> Option<&String> {
        self.ptx_source.as_ref()
    }

    /// Dump mapping table for debugging
    pub fn dump_mapping_table(&self) -> String {
        let mut output = String::new();
        output.push_str("=== SASS ↔ PTX Mapping Table ===\n\n");

        output.push_str("SASS Address → PTX Location:\n");
        let mut addrs: Vec<_> = self.sass_to_ptx.iter().collect();
        addrs.sort_by_key(|(addr, _)| *addr);
        for (addr, loc) in addrs {
            output.push_str(&format!("  0x{:08x} → {}:{}\n", addr, loc.file, loc.line));
        }

        output.push_str("\nPTX Line → SASS Addresses:\n");
        for (key, addrs) in &self.ptx_to_sass {
            let addrs_str: Vec<String> = addrs.iter().map(|a| format!("0x{:08x}", a)).collect();
            output.push_str(&format!("  {} → [{}]\n", key, addrs_str.join(", ")));
        }

        output
    }

    /// Load mappings from natively parsed CUBIN data
    ///
    /// This method integrates with the native CUBIN parser in the sass module,
    /// enabling debug info extraction without relying on external tools like cuobjdump.
    pub fn load_from_parsed_cubin(&mut self, parsed: &crate::sass::ParsedCubin) {
        // Import debug line mappings from DWARF
        for (addr, debug_info) in &parsed.debug_lines {
            let ptx_loc = PtxSourceLocation {
                file: debug_info.file.clone(),
                line: debug_info.line,
                column: debug_info.column,
                instruction_offset: *addr as usize,
            };

            self.sass_to_ptx.insert(*addr, ptx_loc);

            let key = format!("{}:{}", debug_info.file, debug_info.line);
            self.ptx_to_sass
                .entry(key)
                .or_insert_with(Vec::new)
                .push(*addr);
        }

        // Build function ranges from kernels
        for kernel in &parsed.kernels {
            let end_addr = kernel.address + kernel.size as u64;
            self.function_ranges
                .insert(kernel.name.clone(), (kernel.address, end_addr));
        }
    }

    /// Load mappings from enhanced SASS instructions
    ///
    /// This method takes disassembled and enhanced SASS instructions and builds
    /// the bidirectional mapping table.
    pub fn load_from_enhanced_instructions(
        &mut self,
        kernel_name: &str,
        instructions: &[crate::sass::EnhancedSassInstruction],
    ) {
        for inst in instructions {
            if let (Some(file), Some(line)) = (&inst.ptx_file, inst.ptx_line) {
                let ptx_loc = PtxSourceLocation {
                    file: file.clone(),
                    line,
                    column: inst.ptx_column.unwrap_or(0),
                    instruction_offset: inst.address as usize,
                };

                self.sass_to_ptx.insert(inst.address, ptx_loc);

                let key = format!("{}:{}", file, line);
                self.ptx_to_sass
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(inst.address);

                // Create detailed line mapping
                let sass_inst = SassInstruction {
                    opcode: inst.opcode.clone(),
                    instruction: inst.instruction_text.clone(),
                    address: inst.address,
                    predicate: inst.predicate.as_ref().map(|p| format!("{:?}", p)),
                    dest_operands: inst
                        .dest_operands
                        .iter()
                        .map(|r| format!("{:?}", r))
                        .collect(),
                    src_operands: inst
                        .src_operands
                        .iter()
                        .map(|r| format!("{:?}", r))
                        .collect(),
                    control_codes: inst.modifiers.clone(),
                    stall_count: Some(inst.control_codes.stall_count),
                    wait_barrier: Some(inst.control_codes.wait_barrier_mask),
                    read_barrier: Some(inst.control_codes.read_barrier),
                    write_barrier: Some(inst.control_codes.write_barrier),
                };

                self.line_mappings.push(SassLineMapping {
                    sass_address: inst.address,
                    sass_offset: format!("0x{:04x}", inst.address),
                    sass_instruction: sass_inst,
                    ptx_file: file.clone(),
                    ptx_line: line,
                    ptx_column: inst.ptx_column.unwrap_or(0),
                    function_name: Some(kernel_name.to_string()),
                });
            }
        }

        // Update function range
        if let (Some(first), Some(last)) = (instructions.first(), instructions.last()) {
            let start = first.address;
            let end = last.address + last.size as u64;
            self.function_ranges
                .insert(kernel_name.to_string(), (start, end));
        }
    }

    /// Get instruction details at a SASS address
    pub fn get_instruction_at(&self, sass_addr: u64) -> Option<&SassLineMapping> {
        self.line_mappings
            .iter()
            .find(|m| m.sass_address == sass_addr)
    }

    /// Find all SASS addresses within a function
    pub fn get_function_addresses(&self, function_name: &str) -> Vec<u64> {
        self.line_mappings
            .iter()
            .filter(|m| m.function_name.as_deref() == Some(function_name))
            .map(|m| m.sass_address)
            .collect()
    }
}

impl Default for SassPtxMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CUBIN/ELF Line Info Parser
// ============================================================================

/// CUBIN (ELF) debug info section parser
#[derive(Debug)]
pub struct CubinDebugInfo {
    /// Parsed SASS-PTX mappings
    pub mapper: SassPtxMapper,
    /// DWARF debug_line entries
    pub debug_line_entries: Vec<DebugLineEntry>,
    /// Function symbols
    pub symbols: Vec<CubinSymbol>,
}

/// DWARF debug_line entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugLineEntry {
    pub address: u64,
    pub file_index: u32,
    pub line: u32,
    pub column: u32,
    pub is_stmt: bool,
    pub basic_block: bool,
    pub end_sequence: bool,
    pub prologue_end: bool,
    pub epilogue_begin: bool,
}

/// CUBIN symbol (function or variable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubinSymbol {
    pub name: String,
    pub address: u64,
    pub size: u64,
    pub symbol_type: CubinSymbolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CubinSymbolType {
    Function,
    Variable,
    Section,
    Unknown,
}

impl CubinDebugInfo {
    /// Create new CUBIN debug info parser
    pub fn new() -> Self {
        Self {
            mapper: SassPtxMapper::new(),
            debug_line_entries: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Parse CUBIN file using system tools (cuobjdump, readelf)
    #[cfg(unix)]
    pub fn parse_cubin_file(
        &mut self,
        cubin_path: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        use std::process::Command;

        if let Some(source) = ptx_source {
            self.mapper.ptx_source = Some(source);
        }
        self.mapper.cubin_path = Some(cubin_path.to_string());

        // Use cuobjdump to get SASS with line info
        let output = Command::new("cuobjdump")
            .args(&["-sass", "-lineinfo", cubin_path])
            .output()
            .map_err(|e| format!("Failed to run cuobjdump: {}", e))?;

        if output.status.success() {
            let sass_output = String::from_utf8_lossy(&output.stdout);
            self.mapper.parse_cuobjdump_output(&sass_output)?;
        }

        // Use cuobjdump to get symbols
        let sym_output = Command::new("cuobjdump")
            .args(&["-symbols", cubin_path])
            .output()
            .map_err(|e| format!("Failed to run cuobjdump for symbols: {}", e))?;

        if sym_output.status.success() {
            let symbols_text = String::from_utf8_lossy(&sym_output.stdout);
            self.parse_symbols_output(&symbols_text)?;
        }

        Ok(())
    }

    #[cfg(not(unix))]
    pub fn parse_cubin_file(
        &mut self,
        _cubin_path: &str,
        _ptx_source: Option<String>,
    ) -> Result<(), String> {
        Err("CUBIN parsing only supported on Unix systems".to_string())
    }

    /// Parse cuobjdump -symbols output
    fn parse_symbols_output(&mut self, output: &str) -> Result<(), String> {
        // Parse symbol table entries
        // Format varies, but typically: "Index  Name  Size  Type  Flags  ..."
        for line in output.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                // Try to extract symbol info
                if let Some(name) = parts.get(1) {
                    if name.starts_with("_")
                        || name
                            .chars()
                            .next()
                            .map(|c| c.is_alphabetic())
                            .unwrap_or(false)
                    {
                        let addr = parts
                            .get(0)
                            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                            .unwrap_or(0);
                        let size = parts
                            .get(2)
                            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                            .unwrap_or(0);

                        let sym_type = if line.contains("FUNC") {
                            CubinSymbolType::Function
                        } else if line.contains("OBJECT") {
                            CubinSymbolType::Variable
                        } else {
                            CubinSymbolType::Unknown
                        };

                        self.symbols.push(CubinSymbol {
                            name: name.to_string(),
                            address: addr,
                            size,
                            symbol_type: sym_type,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse CUBIN file using native parser (no external tools required)
    ///
    /// This method uses the built-in CUBIN/ELF parser and DWARF extractor
    /// to parse debug information directly from the binary.
    pub fn parse_cubin_native(
        &mut self,
        cubin_data: &[u8],
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        if let Some(source) = ptx_source {
            self.mapper.ptx_source = Some(source);
        }

        // Parse CUBIN structure
        let parsed = crate::sass::CubinParser::new(cubin_data.to_vec())
            .parse()
            .map_err(|e| format!("Native CUBIN parse error: {:?}", e))?;

        // Load debug line mappings
        self.mapper.load_from_parsed_cubin(&parsed);

        // Build symbol table from parsed kernels
        for kernel in &parsed.kernels {
            self.symbols.push(CubinSymbol {
                name: kernel.name.clone(),
                address: kernel.address,
                size: kernel.size as u64,
                symbol_type: CubinSymbolType::Function,
            });
        }

        // Disassemble each kernel and load enhanced instructions
        for kernel in &parsed.kernels {
            if let Ok(disasm) = crate::sass::SassDisassembler::new(kernel.sm_version) {
                let mut instructions = disasm.disassemble(&kernel.code, kernel.address);

                // Apply debug line info
                for inst in &mut instructions {
                    if let Some(debug_info) = parsed.debug_lines.get(&inst.address) {
                        inst.ptx_file = Some(debug_info.file.clone());
                        inst.ptx_line = Some(debug_info.line);
                        inst.ptx_column = Some(debug_info.column);
                    }
                    inst.function_name = Some(kernel.name.clone());
                }

                // Analyze control flow
                crate::sass::ControlFlowAnalyzer::find_basic_blocks(&mut instructions);
                crate::sass::ControlFlowAnalyzer::analyze_data_flow(&mut instructions);

                // Load into mapper
                self.mapper
                    .load_from_enhanced_instructions(&kernel.name, &instructions);
            }
        }

        Ok(())
    }

    /// Parse CUBIN from file path using native parser
    pub fn parse_cubin_file_native(
        &mut self,
        cubin_path: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        let cubin_data =
            std::fs::read(cubin_path).map_err(|e| format!("Failed to read CUBIN file: {}", e))?;

        self.mapper.cubin_path = Some(cubin_path.to_string());
        self.parse_cubin_native(&cubin_data, ptx_source)
    }

    /// Parse cuobjdump text output and enhance with semantic analysis
    pub fn parse_cuobjdump_enhanced(
        &mut self,
        output: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        if let Some(source) = ptx_source {
            self.mapper.ptx_source = Some(source);
        }

        // Parse cuobjdump output to enhanced instructions
        let instructions = crate::sass::TextDisassemblyParser::parse_cuobjdump_output(output);

        // Group by function
        let mut by_function: std::collections::HashMap<
            String,
            Vec<crate::sass::EnhancedSassInstruction>,
        > = std::collections::HashMap::new();

        for inst in instructions {
            let func = inst
                .function_name
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            by_function.entry(func).or_default().push(inst);
        }

        // Load each function's instructions
        for (func_name, mut insts) in by_function {
            crate::sass::ControlFlowAnalyzer::find_basic_blocks(&mut insts);
            crate::sass::ControlFlowAnalyzer::analyze_data_flow(&mut insts);
            self.mapper
                .load_from_enhanced_instructions(&func_name, &insts);
        }

        Ok(())
    }

    /// Get the SASS-PTX mapper
    pub fn get_mapper(&self) -> &SassPtxMapper {
        &self.mapper
    }

    /// Get mutable reference to mapper
    pub fn get_mapper_mut(&mut self) -> &mut SassPtxMapper {
        &mut self.mapper
    }

    /// Get parsed symbols
    pub fn get_symbols(&self) -> &[CubinSymbol] {
        &self.symbols
    }

    /// Get functions (symbols with Function type)
    pub fn get_functions(&self) -> Vec<&CubinSymbol> {
        self.symbols
            .iter()
            .filter(|s| matches!(s.symbol_type, CubinSymbolType::Function))
            .collect()
    }
}

impl Default for CubinDebugInfo {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Runtime Debug Query Interface for hetGPU
// ============================================================================

/// Runtime debug interface for hetGPU integration
#[derive(Debug)]
pub struct HetGpuDebugInterface {
    /// SASS-PTX mapper
    mapper: SassPtxMapper,
    /// Active breakpoints (SASS address → breakpoint info)
    breakpoints: HashMap<u64, RuntimeBreakpoint>,
    /// Watchpoints for memory access
    watchpoints: Vec<Watchpoint>,
    /// Step mode state
    step_mode: StepMode,
    /// Current execution context
    current_context: Option<ExecutionContext>,
}

/// Runtime breakpoint
#[derive(Debug, Clone)]
pub struct RuntimeBreakpoint {
    pub id: u32,
    pub sass_address: u64,
    pub ptx_location: PtxSourceLocation,
    pub enabled: bool,
    pub hit_count: u32,
    pub condition: Option<String>,
    /// Original instruction bytes for restoration
    pub original_instruction: Vec<u8>,
}

/// Memory watchpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watchpoint {
    pub id: u32,
    pub address: u64,
    pub size: u32,
    pub watch_type: WatchType,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WatchType {
    Read,
    Write,
    ReadWrite,
}

/// Step mode for debugging
#[derive(Debug, Clone, PartialEq)]
pub enum StepMode {
    /// Continue execution
    Continue,
    /// Step to next SASS instruction
    StepInstruction,
    /// Step to next PTX line
    StepLine,
    /// Step into function call
    StepInto,
    /// Step over function call
    StepOver,
    /// Step out of current function
    StepOut,
}

/// Execution context at breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Current SASS address
    pub sass_address: u64,
    /// Current PTX location
    pub ptx_location: PtxSourceLocation,
    /// Register state (name → value)
    pub registers: HashMap<String, u64>,
    /// Predicate register state
    pub predicates: HashMap<String, bool>,
    /// Thread ID (x, y, z)
    pub thread_id: (u32, u32, u32),
    /// Block ID (x, y, z)
    pub block_id: (u32, u32, u32),
    /// Warp ID
    pub warp_id: u32,
    /// Lane ID within warp
    pub lane_id: u32,
    /// Active thread mask
    pub active_mask: u64,
    /// Stack trace
    pub call_stack: Vec<StackFrame>,
}

/// Stack frame for call stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub function_name: String,
    pub sass_address: u64,
    pub ptx_location: PtxSourceLocation,
}

impl HetGpuDebugInterface {
    /// Create new debug interface
    pub fn new() -> Self {
        Self {
            mapper: SassPtxMapper::new(),
            breakpoints: HashMap::new(),
            watchpoints: Vec::new(),
            step_mode: StepMode::Continue,
            current_context: None,
        }
    }

    /// Create with existing mapper
    pub fn with_mapper(mapper: SassPtxMapper) -> Self {
        Self {
            mapper,
            breakpoints: HashMap::new(),
            watchpoints: Vec::new(),
            step_mode: StepMode::Continue,
            current_context: None,
        }
    }

    /// Load mappings from cuobjdump output
    pub fn load_from_cuobjdump(&mut self, output: &str) -> Result<(), String> {
        self.mapper.parse_cuobjdump_output(output)
    }

    /// Load mappings from CUBIN file
    #[cfg(unix)]
    pub fn load_from_cubin(
        &mut self,
        cubin_path: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        let mut debug_info = CubinDebugInfo::new();
        debug_info.parse_cubin_file(cubin_path, ptx_source)?;
        self.mapper = debug_info.mapper;
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn load_from_cubin(
        &mut self,
        _cubin_path: &str,
        _ptx_source: Option<String>,
    ) -> Result<(), String> {
        Err("CUBIN loading only supported on Unix systems".to_string())
    }

    /// Load mappings from CUBIN file using native parser (no cuobjdump required)
    ///
    /// This method uses the built-in CUBIN/ELF parser, DWARF extractor, and SASS
    /// disassembler to build mappings without external tool dependencies.
    pub fn load_from_cubin_native(
        &mut self,
        cubin_path: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        let mut debug_info = CubinDebugInfo::new();
        debug_info.parse_cubin_file_native(cubin_path, ptx_source)?;
        self.mapper = debug_info.mapper;
        Ok(())
    }

    /// Load mappings from CUBIN data bytes using native parser
    pub fn load_from_cubin_data(
        &mut self,
        cubin_data: &[u8],
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        let mut debug_info = CubinDebugInfo::new();
        debug_info.parse_cubin_native(cubin_data, ptx_source)?;
        self.mapper = debug_info.mapper;
        Ok(())
    }

    /// Load mappings from cuobjdump output with enhanced semantic analysis
    pub fn load_from_cuobjdump_enhanced(
        &mut self,
        output: &str,
        ptx_source: Option<String>,
    ) -> Result<(), String> {
        let mut debug_info = CubinDebugInfo::new();
        debug_info.parse_cuobjdump_enhanced(output, ptx_source)?;
        self.mapper = debug_info.mapper;
        Ok(())
    }

    /// Set breakpoint at PTX line
    pub fn set_breakpoint_at_ptx_line(&mut self, file: &str, line: u32) -> Result<u32, String> {
        let sass_addr = self
            .mapper
            .ptx_to_sass_address(file, line)
            .ok_or_else(|| format!("No SASS address found for {}:{}", file, line))?;

        self.set_breakpoint_at_sass_address(sass_addr, file, line)
    }

    /// Set breakpoint at SASS address
    pub fn set_breakpoint_at_sass_address(
        &mut self,
        sass_addr: u64,
        file: &str,
        line: u32,
    ) -> Result<u32, String> {
        let id = self.breakpoints.len() as u32;

        let bp = RuntimeBreakpoint {
            id,
            sass_address: sass_addr,
            ptx_location: PtxSourceLocation {
                file: file.to_string(),
                line,
                column: 0,
                instruction_offset: sass_addr as usize,
            },
            enabled: true,
            hit_count: 0,
            condition: None,
            original_instruction: Vec::new(),
        };

        self.breakpoints.insert(sass_addr, bp);
        Ok(id)
    }

    /// Remove breakpoint
    pub fn remove_breakpoint(&mut self, id: u32) -> bool {
        let addr = self
            .breakpoints
            .iter()
            .find(|(_, bp)| bp.id == id)
            .map(|(addr, _)| *addr);

        if let Some(addr) = addr {
            self.breakpoints.remove(&addr);
            true
        } else {
            false
        }
    }

    /// Check if SASS address is a breakpoint
    pub fn is_breakpoint(&self, sass_addr: u64) -> Option<&RuntimeBreakpoint> {
        self.breakpoints.get(&sass_addr).filter(|bp| bp.enabled)
    }

    /// Handle breakpoint hit - returns PTX location and context
    pub fn handle_breakpoint_hit(
        &mut self,
        sass_addr: u64,
    ) -> Option<(&PtxSourceLocation, &RuntimeBreakpoint)> {
        if let Some(bp) = self.breakpoints.get_mut(&sass_addr) {
            if bp.enabled {
                bp.hit_count += 1;
                return Some((&bp.ptx_location, bp));
            }
        }
        None
    }

    /// Query PTX location for SASS address (main runtime query)
    pub fn query_ptx_location(&self, sass_addr: u64) -> Option<&PtxSourceLocation> {
        self.mapper.sass_to_ptx_location(sass_addr)
    }

    /// Query PTX location with tolerance for address matching
    pub fn query_ptx_location_nearest(
        &self,
        sass_addr: u64,
        tolerance: u64,
    ) -> Option<&PtxSourceLocation> {
        self.mapper
            .sass_to_ptx_location_nearest(sass_addr, tolerance)
    }

    /// Get PTX source line content
    pub fn get_ptx_source_line(&self, line: u32) -> Option<&str> {
        self.mapper.get_ptx_source_line(line)
    }

    /// Get PTX source context (surrounding lines)
    pub fn get_ptx_context(&self, line: u32, context: u32) -> Vec<(u32, String)> {
        self.mapper.get_ptx_context(line, context)
    }

    /// Set step mode
    pub fn set_step_mode(&mut self, mode: StepMode) {
        self.step_mode = mode;
    }

    /// Get current step mode
    pub fn get_step_mode(&self) -> &StepMode {
        &self.step_mode
    }

    /// Update execution context
    pub fn set_execution_context(&mut self, context: ExecutionContext) {
        self.current_context = Some(context);
    }

    /// Get current execution context
    pub fn get_execution_context(&self) -> Option<&ExecutionContext> {
        self.current_context.as_ref()
    }

    /// Get next expected SASS address for step line
    pub fn get_next_ptx_line_address(&self, current_addr: u64) -> Option<u64> {
        if let Some(current_loc) = self.mapper.sass_to_ptx_location(current_addr) {
            let current_line = current_loc.line;

            // Find next address that maps to a different PTX line
            let mut addrs = self.mapper.get_sass_addresses_ordered();
            if let Some(pos) = addrs.iter().position(|&a| a == current_addr) {
                for &addr in &addrs[pos + 1..] {
                    if let Some(loc) = self.mapper.sass_to_ptx_location(addr) {
                        if loc.line > current_line {
                            return Some(addr);
                        }
                    }
                }
            }
        }
        None
    }

    /// Add watchpoint
    pub fn add_watchpoint(&mut self, address: u64, size: u32, watch_type: WatchType) -> u32 {
        let id = self.watchpoints.len() as u32;
        self.watchpoints.push(Watchpoint {
            id,
            address,
            size,
            watch_type,
            enabled: true,
        });
        id
    }

    /// Remove watchpoint
    pub fn remove_watchpoint(&mut self, id: u32) -> bool {
        if let Some(pos) = self.watchpoints.iter().position(|w| w.id == id) {
            self.watchpoints.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if memory access triggers watchpoint
    pub fn check_watchpoint(&self, address: u64, size: u32, is_write: bool) -> Option<&Watchpoint> {
        for wp in &self.watchpoints {
            if !wp.enabled {
                continue;
            }

            let wp_end = wp.address + wp.size as u64;
            let access_end = address + size as u64;

            // Check for overlap
            if address < wp_end && access_end > wp.address {
                match (&wp.watch_type, is_write) {
                    (WatchType::Read, false) => return Some(wp),
                    (WatchType::Write, true) => return Some(wp),
                    (WatchType::ReadWrite, _) => return Some(wp),
                    _ => {}
                }
            }
        }
        None
    }

    /// Get mapper reference
    pub fn get_mapper(&self) -> &SassPtxMapper {
        &self.mapper
    }

    /// Get mutable mapper reference
    pub fn get_mapper_mut(&mut self) -> &mut SassPtxMapper {
        &mut self.mapper
    }

    /// Generate debug report at current location
    pub fn generate_debug_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== hetGPU Debug Report ===\n\n");

        if let Some(ctx) = &self.current_context {
            report.push_str(&format!("Current Location:\n"));
            report.push_str(&format!("  SASS Address: 0x{:08x}\n", ctx.sass_address));
            report.push_str(&format!("  PTX File: {}\n", ctx.ptx_location.file));
            report.push_str(&format!("  PTX Line: {}\n", ctx.ptx_location.line));

            if let Some(line_content) = self.mapper.get_ptx_source_line(ctx.ptx_location.line) {
                report.push_str(&format!("  Source: {}\n", line_content.trim()));
            }

            report.push_str(&format!(
                "\nThread: ({}, {}, {})\n",
                ctx.thread_id.0, ctx.thread_id.1, ctx.thread_id.2
            ));
            report.push_str(&format!(
                "Block: ({}, {}, {})\n",
                ctx.block_id.0, ctx.block_id.1, ctx.block_id.2
            ));
            report.push_str(&format!("Warp: {}, Lane: {}\n", ctx.warp_id, ctx.lane_id));
            report.push_str(&format!("Active Mask: 0x{:08x}\n", ctx.active_mask));

            report.push_str("\nRegisters:\n");
            for (name, value) in &ctx.registers {
                report.push_str(&format!("  {} = 0x{:016x}\n", name, value));
            }

            report.push_str("\nPredicates:\n");
            for (name, value) in &ctx.predicates {
                report.push_str(&format!("  {} = {}\n", name, value));
            }

            report.push_str("\nCall Stack:\n");
            for (i, frame) in ctx.call_stack.iter().enumerate() {
                report.push_str(&format!(
                    "  #{} {} at {}:{} (0x{:08x})\n",
                    i,
                    frame.function_name,
                    frame.ptx_location.file,
                    frame.ptx_location.line,
                    frame.sass_address
                ));
            }
        } else {
            report.push_str("No execution context available\n");
        }

        report.push_str(&format!("\nBreakpoints ({}):\n", self.breakpoints.len()));
        for bp in self.breakpoints.values() {
            let status = if bp.enabled { "enabled" } else { "disabled" };
            report.push_str(&format!(
                "  #{} 0x{:08x} → {}:{} ({}, {} hits)\n",
                bp.id,
                bp.sass_address,
                bp.ptx_location.file,
                bp.ptx_location.line,
                status,
                bp.hit_count
            ));
        }

        report.push_str(&format!("\nWatchpoints ({}):\n", self.watchpoints.len()));
        for wp in &self.watchpoints {
            let status = if wp.enabled { "enabled" } else { "disabled" };
            let wtype = match wp.watch_type {
                WatchType::Read => "read",
                WatchType::Write => "write",
                WatchType::ReadWrite => "rw",
            };
            report.push_str(&format!(
                "  #{} 0x{:08x} size {} {} ({})\n",
                wp.id, wp.address, wp.size, wtype, status
            ));
        }

        report
    }
}

impl Default for HetGpuDebugInterface {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PTX Source Reconstruction Helper
// ============================================================================

/// Helper for reconstructing PTX context from SASS state
#[derive(Debug)]
pub struct PtxReconstructor {
    /// Original PTX source
    ptx_source: String,
    /// Parsed PTX lines
    ptx_lines: Vec<String>,
    /// SASS-PTX mapper
    mapper: SassPtxMapper,
}

impl PtxReconstructor {
    /// Create new reconstructor from PTX source
    pub fn new(ptx_source: String) -> Self {
        let ptx_lines: Vec<String> = ptx_source.lines().map(|s| s.to_string()).collect();
        Self {
            ptx_source,
            ptx_lines,
            mapper: SassPtxMapper::new(),
        }
    }

    /// Create with existing mapper
    pub fn with_mapper(ptx_source: String, mapper: SassPtxMapper) -> Self {
        let ptx_lines: Vec<String> = ptx_source.lines().map(|s| s.to_string()).collect();
        Self {
            ptx_source,
            ptx_lines,
            mapper,
        }
    }

    /// Get PTX instruction at line
    pub fn get_ptx_instruction(&self, line: u32) -> Option<&str> {
        self.ptx_lines
            .get((line.saturating_sub(1)) as usize)
            .map(|s| s.as_str())
    }

    /// Get PTX context window
    pub fn get_context_window(&self, line: u32, before: u32, after: u32) -> String {
        let mut result = String::new();
        let start = line.saturating_sub(before).saturating_sub(1) as usize;
        let end = std::cmp::min((line + after) as usize, self.ptx_lines.len());

        for (i, ptx_line) in self.ptx_lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            let marker = if line_num == line as usize {
                ">>>"
            } else {
                "   "
            };
            result.push_str(&format!("{} {:4}: {}\n", marker, line_num, ptx_line));
        }
        result
    }

    /// Reconstruct PTX state from SASS execution point
    pub fn reconstruct_from_sass(&self, sass_addr: u64) -> Option<PtxExecutionState> {
        let ptx_loc = self.mapper.sass_to_ptx_location(sass_addr)?;
        let ptx_line = self.get_ptx_instruction(ptx_loc.line)?;

        Some(PtxExecutionState {
            sass_address: sass_addr,
            ptx_file: ptx_loc.file.clone(),
            ptx_line: ptx_loc.line,
            ptx_instruction: ptx_line.to_string(),
            context: self.get_context_window(ptx_loc.line, 3, 3),
        })
    }

    /// Get all SASS addresses for a PTX instruction pattern
    pub fn find_sass_for_ptx_pattern(&self, pattern: &str) -> Vec<(u32, u64)> {
        let mut results = Vec::new();

        for (i, line) in self.ptx_lines.iter().enumerate() {
            if line.contains(pattern) {
                let line_num = (i + 1) as u32;
                if let Some(addrs) = self.mapper.ptx_to_sass_addresses("kernel.ptx", line_num) {
                    for &addr in addrs {
                        results.push((line_num, addr));
                    }
                }
            }
        }
        results
    }

    /// Get mapper reference
    pub fn get_mapper(&self) -> &SassPtxMapper {
        &self.mapper
    }

    /// Get mapper mutable reference
    pub fn get_mapper_mut(&mut self) -> &mut SassPtxMapper {
        &mut self.mapper
    }
}

/// PTX execution state reconstructed from SASS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxExecutionState {
    /// Current SASS address
    pub sass_address: u64,
    /// PTX source file
    pub ptx_file: String,
    /// PTX line number
    pub ptx_line: u32,
    /// PTX instruction text
    pub ptx_instruction: String,
    /// Context around instruction
    pub context: String,
}

// ============================================================================
// DWARF Line Program Integration
// ============================================================================

/// Build SASS-PTX mapping from LLVM DWARF debug info
impl PtxDwarfBuilder {
    /// Create SASS mapping entry for an instruction
    pub fn create_sass_mapping(
        &mut self,
        sass_address: u64,
        sass_instruction: &str,
        predicate: Option<String>,
        ptx_line: u32,
    ) {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line: ptx_line,
            column: 0,
            instruction_offset: sass_address as usize,
        };

        let target_inst = TargetInstruction::Sass {
            instruction: sass_instruction.to_string(),
            address: sass_address,
            predicate,
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_inst],
            variable_mappings: HashMap::new(),
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        self.variable_counter += 1;
    }

    /// Build SassPtxMapper from current mappings
    pub fn build_sass_mapper(&self, ptx_source: Option<String>) -> SassPtxMapper {
        let mut mapper = match ptx_source {
            Some(src) => SassPtxMapper::with_ptx_source(src),
            None => SassPtxMapper::new(),
        };

        for mapping in &self.source_mappings {
            for target_inst in &mapping.target_instructions {
                if let TargetInstruction::Sass {
                    instruction,
                    address,
                    ..
                } = target_inst
                {
                    mapper.add_mapping(
                        *address,
                        &mapping.ptx_location.file,
                        mapping.ptx_location.line,
                        instruction,
                    );
                }
            }
        }

        mapper
    }

    /// Export all mappings to JSON for runtime loading
    pub fn export_mappings_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.source_mappings)
            .map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Get all SASS mappings
    pub fn get_sass_mappings(&self) -> Vec<&DwarfMappingEntry> {
        self.source_mappings
            .iter()
            .filter(|m| {
                m.target_instructions
                    .iter()
                    .any(|t| matches!(t, TargetInstruction::Sass { .. }))
            })
            .collect()
    }
}

// ============================================================================
// GPU Trap Handler and State Checkpoint/Resume
// ============================================================================

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Global flag for trap signal
static TRAP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Global trap handler ID counter
static TRAP_HANDLER_ID: AtomicU64 = AtomicU64::new(0);

/// GPU trap reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrapReason {
    /// User requested trap (Ctrl+C / SIGINT)
    UserInterrupt,
    /// Breakpoint hit
    Breakpoint { id: u32, sass_address: u64 },
    /// Watchpoint triggered
    Watchpoint {
        id: u32,
        address: u64,
        access_type: String,
    },
    /// Exception (e.g., illegal instruction, memory fault)
    Exception { code: u32, message: String },
    /// Checkpoint requested programmatically
    CheckpointRequest { label: String },
    /// Single-step completed
    SingleStep,
}

/// Complete GPU execution state for checkpoint/resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCheckpointState {
    /// Version for compatibility checking
    pub version: u32,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Reason for the trap
    pub trap_reason: TrapReason,
    /// Current execution context
    pub execution_context: ExecutionContext,
    /// SASS-PTX mapping data
    pub sass_ptx_mappings: SassPtxMapper,
    /// Original PTX source code
    pub ptx_source: Option<String>,
    /// Active breakpoints
    pub breakpoints: Vec<CheckpointBreakpoint>,
    /// Active watchpoints
    pub watchpoints: Vec<Watchpoint>,
    /// All thread states (for warp/block-level checkpointing)
    pub thread_states: Vec<ThreadCheckpointState>,
    /// Global memory regions to checkpoint
    pub memory_regions: Vec<MemoryRegion>,
    /// Shared memory state per block (key is "block_x_y_z")
    pub shared_memory: HashMap<String, Vec<u8>>,
    /// Constant memory
    pub constant_memory: Vec<u8>,
    /// Kernel arguments
    pub kernel_args: Vec<KernelArgument>,
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Kernel name
    pub kernel_name: String,
}

/// Breakpoint info for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBreakpoint {
    pub id: u32,
    pub sass_address: u64,
    pub ptx_file: String,
    pub ptx_line: u32,
    pub enabled: bool,
    pub hit_count: u32,
    pub condition: Option<String>,
}

/// Per-thread checkpoint state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadCheckpointState {
    pub thread_id: (u32, u32, u32),
    pub block_id: (u32, u32, u32),
    pub warp_id: u32,
    pub lane_id: u32,
    pub sass_address: u64,
    pub registers: HashMap<String, u64>,
    pub predicates: HashMap<String, bool>,
    pub local_memory: Vec<u8>,
    pub active: bool,
}

/// Memory region for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub name: String,
    pub base_address: u64,
    pub size: usize,
    pub data: Vec<u8>,
    pub memory_space: MemorySpace,
}

/// Memory space types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
    Texture,
    Surface,
}

/// Kernel argument for restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArgument {
    pub index: u32,
    pub name: Option<String>,
    pub size: usize,
    pub data: Vec<u8>,
    pub is_pointer: bool,
}

impl GpuCheckpointState {
    /// Current checkpoint format version
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint state
    pub fn new(
        trap_reason: TrapReason,
        execution_context: ExecutionContext,
        kernel_name: String,
    ) -> Self {
        Self {
            version: Self::VERSION,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            trap_reason,
            execution_context,
            sass_ptx_mappings: SassPtxMapper::new(),
            ptx_source: None,
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
            thread_states: Vec::new(),
            memory_regions: Vec::new(),
            shared_memory: HashMap::new(),
            constant_memory: Vec::new(),
            kernel_args: Vec::new(),
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            kernel_name,
        }
    }

    /// Save checkpoint to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| format!("Failed to serialize checkpoint: {}", e))
    }

    /// Save checkpoint to binary file (more compact)
    pub fn save_to_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Write magic header
        writer
            .write_all(b"HETGPU_CKPT")
            .map_err(|e| format!("Write error: {}", e))?;
        writer
            .write_all(&self.version.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;

        // Serialize as JSON for now (could use bincode for production)
        let json = serde_json::to_vec(self).map_err(|e| format!("Serialize error: {}", e))?;
        let len = json.len() as u64;
        writer
            .write_all(&len.to_le_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
        writer
            .write_all(&json)
            .map_err(|e| format!("Write error: {}", e))?;

        writer.flush().map_err(|e| format!("Flush error: {}", e))?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);
        let state: Self = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to deserialize checkpoint: {}", e))?;

        if state.version > Self::VERSION {
            return Err(format!(
                "Checkpoint version {} is newer than supported version {}",
                state.version,
                Self::VERSION
            ));
        }

        Ok(state)
    }

    /// Load checkpoint from binary file
    pub fn load_from_binary<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic header
        let mut magic = [0u8; 11];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Read error: {}", e))?;
        if &magic != b"HETGPU_CKPT" {
            return Err("Invalid checkpoint file format".to_string());
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| format!("Read error: {}", e))?;
        let version = u32::from_le_bytes(version_bytes);
        if version > Self::VERSION {
            return Err(format!(
                "Checkpoint version {} is newer than supported version {}",
                version,
                Self::VERSION
            ));
        }

        // Read JSON length and data
        let mut len_bytes = [0u8; 8];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| format!("Read error: {}", e))?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut json_data = vec![0u8; len];
        reader
            .read_exact(&mut json_data)
            .map_err(|e| format!("Read error: {}", e))?;

        serde_json::from_slice(&json_data).map_err(|e| format!("Deserialize error: {}", e))
    }

    /// Get PTX context at current execution point
    pub fn get_ptx_context(&self, lines_before: u32, lines_after: u32) -> String {
        self.sass_ptx_mappings
            .get_ptx_context(
                self.execution_context.ptx_location.line,
                lines_before.max(lines_after),
            )
            .iter()
            .map(|(num, line)| format!("{:4}: {}", num, line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== GPU Checkpoint Summary ===\n\n");
        s.push_str(&format!("Kernel: {}\n", self.kernel_name));
        s.push_str(&format!("Timestamp: {}\n", self.timestamp));
        s.push_str(&format!("Trap Reason: {:?}\n", self.trap_reason));
        s.push_str(&format!(
            "Grid: {:?}, Block: {:?}\n",
            self.grid_dim, self.block_dim
        ));
        s.push_str(&format!("\nCurrent PTX Location:\n"));
        s.push_str(&format!(
            "  File: {}\n",
            self.execution_context.ptx_location.file
        ));
        s.push_str(&format!(
            "  Line: {}\n",
            self.execution_context.ptx_location.line
        ));
        s.push_str(&format!(
            "  SASS Address: 0x{:08x}\n",
            self.execution_context.sass_address
        ));
        s.push_str(&format!(
            "\nThread: ({}, {}, {}), Block: ({}, {}, {})\n",
            self.execution_context.thread_id.0,
            self.execution_context.thread_id.1,
            self.execution_context.thread_id.2,
            self.execution_context.block_id.0,
            self.execution_context.block_id.1,
            self.execution_context.block_id.2,
        ));
        s.push_str(&format!(
            "Warp: {}, Lane: {}\n",
            self.execution_context.warp_id, self.execution_context.lane_id
        ));
        s.push_str(&format!("\nThread States: {}\n", self.thread_states.len()));
        s.push_str(&format!("Memory Regions: {}\n", self.memory_regions.len()));
        s.push_str(&format!("Breakpoints: {}\n", self.breakpoints.len()));

        if let Some(ref source) = self.ptx_source {
            s.push_str(&format!("\nPTX Context:\n{}\n", self.get_ptx_context(3, 3)));
        }

        s
    }
}

/// GPU Trap Handler for runtime debugging
pub struct GpuTrapHandler {
    /// Debug interface for SASS-PTX mapping
    debug_interface: Arc<RwLock<HetGpuDebugInterface>>,
    /// Current checkpoint state
    current_checkpoint: Arc<Mutex<Option<GpuCheckpointState>>>,
    /// Handler ID
    handler_id: u64,
    /// Checkpoint directory
    checkpoint_dir: String,
    /// Auto-checkpoint on trap
    auto_checkpoint: bool,
}

impl GpuTrapHandler {
    /// Create a new trap handler
    pub fn new(checkpoint_dir: &str) -> Self {
        let handler_id = TRAP_HANDLER_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            debug_interface: Arc::new(RwLock::new(HetGpuDebugInterface::new())),
            current_checkpoint: Arc::new(Mutex::new(None)),
            handler_id,
            checkpoint_dir: checkpoint_dir.to_string(),
            auto_checkpoint: true,
        }
    }

    /// Install signal handler for Ctrl+C (SIGINT)
    #[cfg(unix)]
    pub fn install_signal_handler() -> Result<(), String> {
        use std::os::raw::c_int;

        extern "C" fn sigint_handler(_: c_int) {
            TRAP_REQUESTED.store(true, Ordering::SeqCst);
            eprintln!("\n[hetGPU] Trap requested - GPU execution will pause at next safe point");
        }

        unsafe {
            let mut handler: libc::sigaction = std::mem::zeroed();
            handler.sa_sigaction = sigint_handler as usize;
            handler.sa_flags = 0;
            libc::sigemptyset(&mut handler.sa_mask);

            if libc::sigaction(libc::SIGINT, &handler, std::ptr::null_mut()) != 0 {
                return Err("Failed to install SIGINT handler".to_string());
            }
        }

        Ok(())
    }

    #[cfg(not(unix))]
    pub fn install_signal_handler() -> Result<(), String> {
        // Windows implementation using SetConsoleCtrlHandler would go here
        Err("Signal handler not supported on this platform".to_string())
    }

    /// Check if trap was requested
    pub fn is_trap_requested() -> bool {
        TRAP_REQUESTED.load(Ordering::SeqCst)
    }

    /// Clear trap request flag
    pub fn clear_trap_request() {
        TRAP_REQUESTED.store(false, Ordering::SeqCst);
    }

    /// Request a trap programmatically
    pub fn request_trap() {
        TRAP_REQUESTED.store(true, Ordering::SeqCst);
    }

    /// Handle trap at current execution point
    pub fn handle_trap(
        &self,
        reason: TrapReason,
        sass_address: u64,
        registers: HashMap<String, u64>,
        predicates: HashMap<String, bool>,
        thread_id: (u32, u32, u32),
        block_id: (u32, u32, u32),
        warp_id: u32,
        lane_id: u32,
        kernel_name: &str,
    ) -> Result<GpuCheckpointState, String> {
        Self::clear_trap_request();

        // Get PTX location from SASS address
        let debug_iface = self
            .debug_interface
            .read()
            .map_err(|e| format!("Lock error: {}", e))?;

        let ptx_location = debug_iface
            .query_ptx_location(sass_address)
            .cloned()
            .unwrap_or_else(|| PtxSourceLocation {
                file: "unknown.ptx".to_string(),
                line: 0,
                column: 0,
                instruction_offset: sass_address as usize,
            });

        // Create execution context
        let execution_context = ExecutionContext {
            sass_address,
            ptx_location,
            registers,
            predicates,
            thread_id,
            block_id,
            warp_id,
            lane_id,
            active_mask: 0xFFFFFFFF,
            call_stack: Vec::new(),
        };

        // Create checkpoint state
        let mut checkpoint =
            GpuCheckpointState::new(reason, execution_context, kernel_name.to_string());

        // Copy mappings from debug interface
        checkpoint.sass_ptx_mappings = debug_iface.get_mapper().clone();

        drop(debug_iface);

        // Auto-save checkpoint if enabled
        if self.auto_checkpoint {
            let filename = format!(
                "{}/checkpoint_{}_{}.json",
                self.checkpoint_dir, self.handler_id, checkpoint.timestamp
            );
            if let Err(e) = checkpoint.save_to_file(&filename) {
                eprintln!("[hetGPU] Warning: Failed to save checkpoint: {}", e);
            } else {
                eprintln!("[hetGPU] Checkpoint saved to: {}", filename);
            }
        }

        // Store current checkpoint
        if let Ok(mut current) = self.current_checkpoint.lock() {
            *current = Some(checkpoint.clone());
        }

        // Print trap info
        self.print_trap_info(&checkpoint);

        Ok(checkpoint)
    }

    /// Print trap information
    fn print_trap_info(&self, checkpoint: &GpuCheckpointState) {
        eprintln!(
            "\n=== GPU Trap at SASS 0x{:08x} ===",
            checkpoint.execution_context.sass_address
        );
        eprintln!(
            "PTX: {}:{}",
            checkpoint.execution_context.ptx_location.file,
            checkpoint.execution_context.ptx_location.line
        );
        eprintln!("Reason: {:?}", checkpoint.trap_reason);
        eprintln!(
            "Thread: ({},{},{}), Block: ({},{},{})",
            checkpoint.execution_context.thread_id.0,
            checkpoint.execution_context.thread_id.1,
            checkpoint.execution_context.thread_id.2,
            checkpoint.execution_context.block_id.0,
            checkpoint.execution_context.block_id.1,
            checkpoint.execution_context.block_id.2
        );

        // Print PTX source context if available
        if checkpoint.sass_ptx_mappings.ptx_source.is_some() {
            eprintln!("\nPTX Context:");
            eprintln!("{}", checkpoint.get_ptx_context(2, 2));
        }
    }

    /// Add thread state for bulk checkpointing
    pub fn add_thread_state(&self, state: ThreadCheckpointState) {
        if let Ok(mut checkpoint) = self.current_checkpoint.lock() {
            if let Some(ref mut ckpt) = *checkpoint {
                ckpt.thread_states.push(state);
            }
        }
    }

    /// Add memory region to checkpoint
    pub fn add_memory_region(&self, region: MemoryRegion) {
        if let Ok(mut checkpoint) = self.current_checkpoint.lock() {
            if let Some(ref mut ckpt) = *checkpoint {
                ckpt.memory_regions.push(region);
            }
        }
    }

    /// Get current checkpoint
    pub fn get_current_checkpoint(&self) -> Option<GpuCheckpointState> {
        self.current_checkpoint.lock().ok()?.clone()
    }

    /// Resume execution from checkpoint
    pub fn prepare_resume(&self, checkpoint: &GpuCheckpointState) -> ResumeInfo {
        ResumeInfo {
            sass_address: checkpoint.execution_context.sass_address,
            ptx_line: checkpoint.execution_context.ptx_location.line,
            registers: checkpoint.execution_context.registers.clone(),
            predicates: checkpoint.execution_context.predicates.clone(),
            thread_states: checkpoint.thread_states.clone(),
            memory_regions: checkpoint.memory_regions.clone(),
            kernel_name: checkpoint.kernel_name.clone(),
            grid_dim: checkpoint.grid_dim,
            block_dim: checkpoint.block_dim,
        }
    }

    /// Get debug interface for configuration
    pub fn get_debug_interface(&self) -> Arc<RwLock<HetGpuDebugInterface>> {
        Arc::clone(&self.debug_interface)
    }
}

/// Information needed to resume GPU execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeInfo {
    pub sass_address: u64,
    pub ptx_line: u32,
    pub registers: HashMap<String, u64>,
    pub predicates: HashMap<String, bool>,
    pub thread_states: Vec<ThreadCheckpointState>,
    pub memory_regions: Vec<MemoryRegion>,
    pub kernel_name: String,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}

impl ResumeInfo {
    /// Load resume info from checkpoint file
    pub fn from_checkpoint_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let checkpoint = GpuCheckpointState::load_from_file(path)?;
        Ok(Self {
            sass_address: checkpoint.execution_context.sass_address,
            ptx_line: checkpoint.execution_context.ptx_location.line,
            registers: checkpoint.execution_context.registers,
            predicates: checkpoint.execution_context.predicates,
            thread_states: checkpoint.thread_states,
            memory_regions: checkpoint.memory_regions,
            kernel_name: checkpoint.kernel_name,
            grid_dim: checkpoint.grid_dim,
            block_dim: checkpoint.block_dim,
        })
    }

    /// Get restoration commands for registers
    pub fn get_register_restoration_commands(&self) -> Vec<String> {
        self.registers
            .iter()
            .map(|(name, value)| format!("mov.u64 {}, 0x{:016x};", name, value))
            .collect()
    }
}

/// Checkpoint manager for multiple kernels
pub struct CheckpointManager {
    checkpoints: HashMap<String, Vec<GpuCheckpointState>>,
    checkpoint_dir: String,
    max_checkpoints_per_kernel: usize,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: &str) -> Self {
        Self {
            checkpoints: HashMap::new(),
            checkpoint_dir: checkpoint_dir.to_string(),
            max_checkpoints_per_kernel: 10,
        }
    }

    /// Add checkpoint
    pub fn add_checkpoint(&mut self, checkpoint: GpuCheckpointState) {
        let kernel_name = checkpoint.kernel_name.clone();
        let checkpoints = self.checkpoints.entry(kernel_name).or_insert_with(Vec::new);

        // Remove old checkpoints if over limit
        while checkpoints.len() >= self.max_checkpoints_per_kernel {
            checkpoints.remove(0);
        }

        checkpoints.push(checkpoint);
    }

    /// Get latest checkpoint for kernel
    pub fn get_latest_checkpoint(&self, kernel_name: &str) -> Option<&GpuCheckpointState> {
        self.checkpoints.get(kernel_name)?.last()
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Vec<(String, u64, usize)> {
        self.checkpoints
            .iter()
            .flat_map(|(kernel, ckpts)| {
                ckpts
                    .iter()
                    .map(move |c| (kernel.clone(), c.timestamp, c.thread_states.len()))
            })
            .collect()
    }

    /// Save all checkpoints to disk
    pub fn save_all(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.checkpoint_dir)
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;

        for (kernel, checkpoints) in &self.checkpoints {
            for (i, checkpoint) in checkpoints.iter().enumerate() {
                let filename = format!(
                    "{}/{}_{}_{}.json",
                    self.checkpoint_dir, kernel, checkpoint.timestamp, i
                );
                checkpoint.save_to_file(&filename)?;
            }
        }
        Ok(())
    }

    /// Load checkpoints from directory
    pub fn load_from_directory(&mut self) -> Result<usize, String> {
        let mut count = 0;
        let dir = std::fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in dir {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(checkpoint) = GpuCheckpointState::load_from_file(&path) {
                    self.add_checkpoint(checkpoint);
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

// ============================================================================
// Unit Tests for SASS ↔ PTX Mapping
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample cuobjdump -sass -lineinfo output for testing
    const SAMPLE_CUOBJDUMP_OUTPUT: &str = r#"
        code for sm_86
        Function : vector_add
        .headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
## File "kernel.ptx", line 16
/*0000*/ @!PT MOV R1, RZ ;
## File "kernel.ptx", line 18
/*0010*/ S2R R0, SR_TID.X ;
## Line 20
/*0020*/ LDG.E.SYS R2, [R4] ;
## Line 23
/*0030*/ @P0 FADD R5, R2, R3 ;
## Line 28
/*0040*/ STG.E.SYS [R6], R5 ;
/*0050*/ EXIT ;
"#;

    #[test]
    fn test_sass_ptx_mapper_parse_cuobjdump() {
        let mut mapper = SassPtxMapper::new();
        mapper
            .parse_cuobjdump_output(SAMPLE_CUOBJDUMP_OUTPUT)
            .unwrap();

        // Test SASS → PTX lookup
        let loc = mapper.sass_to_ptx_location(0x0000).unwrap();
        assert_eq!(loc.line, 16);
        assert_eq!(loc.file, "kernel.ptx");

        let loc = mapper.sass_to_ptx_location(0x0010).unwrap();
        assert_eq!(loc.line, 18);

        let loc = mapper.sass_to_ptx_location(0x0020).unwrap();
        assert_eq!(loc.line, 20);

        let loc = mapper.sass_to_ptx_location(0x0030).unwrap();
        assert_eq!(loc.line, 23);

        let loc = mapper.sass_to_ptx_location(0x0040).unwrap();
        assert_eq!(loc.line, 28);
    }

    #[test]
    fn test_sass_ptx_mapper_reverse_lookup() {
        let mut mapper = SassPtxMapper::new();
        mapper
            .parse_cuobjdump_output(SAMPLE_CUOBJDUMP_OUTPUT)
            .unwrap();

        // Test PTX → SASS lookup
        let addrs = mapper.ptx_to_sass_addresses("kernel.ptx", 16).unwrap();
        assert!(addrs.contains(&0x0000));

        let addrs = mapper.ptx_to_sass_addresses("kernel.ptx", 23).unwrap();
        assert!(addrs.contains(&0x0030));

        // First address for a line
        let addr = mapper.ptx_to_sass_address("kernel.ptx", 18).unwrap();
        assert_eq!(addr, 0x0010);
    }

    #[test]
    fn test_sass_ptx_mapper_nearest_match() {
        let mut mapper = SassPtxMapper::new();
        mapper
            .parse_cuobjdump_output(SAMPLE_CUOBJDUMP_OUTPUT)
            .unwrap();

        // Test nearest match with tolerance
        // Address 0x0015 should match 0x0010 (line 18) with tolerance 8
        let loc = mapper.sass_to_ptx_location_nearest(0x0015, 8).unwrap();
        assert_eq!(loc.line, 18);

        // Address 0x0025 should match 0x0020 (line 20) with tolerance 8
        let loc = mapper.sass_to_ptx_location_nearest(0x0025, 8).unwrap();
        assert_eq!(loc.line, 20);

        // No match with tolerance 0
        assert!(mapper.sass_to_ptx_location_nearest(0x0015, 0).is_none());
    }

    #[test]
    fn test_sass_instruction_parsing() {
        let mapper = SassPtxMapper::new();

        // Test basic instruction parsing
        let inst = mapper.parse_sass_instruction_line("/*0030*/ @P0 FADD R5, R2, R3 ;");
        assert!(inst.is_some());
        let inst = inst.unwrap();
        assert_eq!(inst.address, 0x0030);
        assert_eq!(inst.opcode, "FADD");
        assert_eq!(inst.predicate, Some("@P0".to_string()));

        // Test instruction without predicate
        let inst = mapper.parse_sass_instruction_line("/*0010*/ S2R R0, SR_TID.X ;");
        assert!(inst.is_some());
        let inst = inst.unwrap();
        assert_eq!(inst.address, 0x0010);
        assert_eq!(inst.opcode, "S2R");
        assert!(inst.predicate.is_none());

        // Test instruction with modifiers
        let inst = mapper.parse_sass_instruction_line("/*0020*/ LDG.E.SYS R2, [R4] ;");
        assert!(inst.is_some());
        let inst = inst.unwrap();
        assert_eq!(inst.opcode, "LDG");
        assert!(inst.control_codes.contains(&".E".to_string()));
        assert!(inst.control_codes.contains(&".SYS".to_string()));
    }

    #[test]
    fn test_sass_ptx_mapper_add_mapping() {
        let mut mapper = SassPtxMapper::new();

        mapper.add_mapping(0x100, "test.ptx", 10, "MOV R0, R1");
        mapper.add_mapping(0x110, "test.ptx", 11, "ADD R2, R0, R1");
        mapper.add_mapping(0x120, "test.ptx", 12, "MUL R3, R2, c[0x0][0x0]");

        // Verify lookups
        let loc = mapper.sass_to_ptx_location(0x100).unwrap();
        assert_eq!(loc.line, 10);

        let loc = mapper.sass_to_ptx_location(0x120).unwrap();
        assert_eq!(loc.line, 12);

        let addrs = mapper.ptx_to_sass_addresses("test.ptx", 11).unwrap();
        assert!(addrs.contains(&0x110));
    }

    #[test]
    fn test_sass_ptx_mapper_with_source() {
        let ptx_source = r#"
.version 8.0
.target sm_86
.entry test()
{
    mov.u32 %r0, %tid.x;
    add.u32 %r1, %r0, 1;
    st.global.u32 [%rd0], %r1;
    ret;
}
"#;

        let mut mapper = SassPtxMapper::with_ptx_source(ptx_source.to_string());
        mapper.add_mapping(0x0, "kernel.ptx", 6, "MOV R0, SR_TID.X");
        mapper.add_mapping(0x10, "kernel.ptx", 7, "ADD R1, R0, 0x1");

        // Test source line retrieval
        let line = mapper.get_ptx_source_line(6).unwrap();
        assert!(line.contains("mov.u32"));

        let line = mapper.get_ptx_source_line(7).unwrap();
        assert!(line.contains("add.u32"));

        // Test context retrieval
        let context = mapper.get_ptx_context(6, 2);
        assert!(!context.is_empty());
        assert!(context.iter().any(|(num, _)| *num == 6));
    }

    #[test]
    fn test_sass_ptx_mapper_json_roundtrip() {
        let mut mapper = SassPtxMapper::new();
        mapper.add_mapping(0x100, "test.ptx", 10, "MOV R0, R1");
        mapper.add_mapping(0x110, "test.ptx", 11, "ADD R2, R0, R1");

        // Export to JSON
        let json = mapper.export_json().unwrap();
        assert!(json.contains("0x100") || json.contains("256")); // Address in hex or decimal

        // Import from JSON
        let imported = SassPtxMapper::import_json(&json).unwrap();
        assert_eq!(imported.sass_to_ptx_location(0x100).unwrap().line, 10);
        assert_eq!(imported.sass_to_ptx_location(0x110).unwrap().line, 11);
    }

    #[test]
    fn test_hetgpu_debug_interface() {
        let mut iface = HetGpuDebugInterface::new();

        // Add mappings manually
        iface
            .get_mapper_mut()
            .add_mapping(0x0, "kernel.ptx", 10, "MOV");
        iface
            .get_mapper_mut()
            .add_mapping(0x10, "kernel.ptx", 11, "ADD");
        iface
            .get_mapper_mut()
            .add_mapping(0x20, "kernel.ptx", 12, "MUL");

        // Set breakpoint
        let bp_id = iface
            .set_breakpoint_at_sass_address(0x10, "kernel.ptx", 11)
            .unwrap();
        assert_eq!(bp_id, 0);

        // Check breakpoint
        assert!(iface.is_breakpoint(0x10).is_some());
        assert!(iface.is_breakpoint(0x0).is_none());

        // Handle breakpoint hit
        let result = iface.handle_breakpoint_hit(0x10);
        assert!(result.is_some());
        let (loc, bp) = result.unwrap();
        assert_eq!(loc.line, 11);
        assert_eq!(bp.hit_count, 1);

        // Query PTX location
        let loc = iface.query_ptx_location(0x20).unwrap();
        assert_eq!(loc.line, 12);
    }

    #[test]
    fn test_hetgpu_debug_interface_step_mode() {
        let mut iface = HetGpuDebugInterface::new();

        assert_eq!(iface.get_step_mode(), &StepMode::Continue);

        iface.set_step_mode(StepMode::StepLine);
        assert_eq!(iface.get_step_mode(), &StepMode::StepLine);

        iface.set_step_mode(StepMode::StepInstruction);
        assert_eq!(iface.get_step_mode(), &StepMode::StepInstruction);
    }

    #[test]
    fn test_hetgpu_debug_interface_watchpoints() {
        let mut iface = HetGpuDebugInterface::new();

        // Add watchpoint
        let wp_id = iface.add_watchpoint(0x1000, 4, WatchType::Write);
        assert_eq!(wp_id, 0);

        // Check write access
        let wp = iface.check_watchpoint(0x1000, 4, true);
        assert!(wp.is_some());
        assert_eq!(wp.unwrap().id, 0);

        // Check read access (should not trigger write-only watchpoint)
        let wp = iface.check_watchpoint(0x1000, 4, false);
        assert!(wp.is_none());

        // Add read-write watchpoint
        iface.add_watchpoint(0x2000, 8, WatchType::ReadWrite);

        // Both read and write should trigger
        assert!(iface.check_watchpoint(0x2000, 4, true).is_some());
        assert!(iface.check_watchpoint(0x2000, 4, false).is_some());

        // Check overlapping access
        assert!(iface.check_watchpoint(0x2004, 4, true).is_some());
    }

    #[test]
    fn test_hetgpu_debug_interface_execution_context() {
        let mut iface = HetGpuDebugInterface::new();

        assert!(iface.get_execution_context().is_none());

        let ctx = ExecutionContext {
            sass_address: 0x100,
            ptx_location: PtxSourceLocation {
                file: "test.ptx".to_string(),
                line: 42,
                column: 0,
                instruction_offset: 0x100,
            },
            registers: HashMap::from([("R0".to_string(), 123), ("R1".to_string(), 456)]),
            predicates: HashMap::from([("P0".to_string(), true), ("P1".to_string(), false)]),
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            warp_id: 0,
            lane_id: 0,
            active_mask: 0xFFFFFFFF,
            call_stack: vec![],
        };

        iface.set_execution_context(ctx);

        let ctx = iface.get_execution_context().unwrap();
        assert_eq!(ctx.sass_address, 0x100);
        assert_eq!(ctx.ptx_location.line, 42);
        assert_eq!(*ctx.registers.get("R0").unwrap(), 123);
    }

    #[test]
    fn test_ptx_reconstructor() {
        let ptx_source = r#".version 8.0
.target sm_86
.entry kernel()
{
    mov.u32 %r0, %tid.x;
    add.u32 %r1, %r0, 1;
    mul.lo.u32 %r2, %r1, 4;
    st.global.u32 [%rd0], %r2;
    ret;
}"#;

        let mut reconstructor = PtxReconstructor::new(ptx_source.to_string());
        reconstructor
            .get_mapper_mut()
            .add_mapping(0x0, "kernel.ptx", 5, "MOV");
        reconstructor
            .get_mapper_mut()
            .add_mapping(0x10, "kernel.ptx", 6, "ADD");
        reconstructor
            .get_mapper_mut()
            .add_mapping(0x20, "kernel.ptx", 7, "MUL");

        // Test instruction retrieval
        let inst = reconstructor.get_ptx_instruction(5).unwrap();
        assert!(inst.contains("mov.u32"));

        // Test context window
        let ctx = reconstructor.get_context_window(6, 1, 1);
        assert!(ctx.contains("mov.u32"));
        assert!(ctx.contains("add.u32"));
        assert!(ctx.contains("mul.lo.u32"));

        // Test SASS reconstruction
        let state = reconstructor.reconstruct_from_sass(0x10).unwrap();
        assert_eq!(state.ptx_line, 6);
        assert!(state.ptx_instruction.contains("add.u32"));
    }

    #[test]
    fn test_sass_ptx_mapper_function_ranges() {
        let mut mapper = SassPtxMapper::new();

        let cuobjdump_output = r#"
Function : kernel_a
## File "kernel.ptx", line 10
/*0000*/ MOV R0, RZ ;
/*0010*/ ADD R1, R0, R2 ;

Function : kernel_b
## File "kernel.ptx", line 20
/*0100*/ MOV R0, RZ ;
/*0110*/ MUL R1, R0, R2 ;
"#;

        mapper.parse_cuobjdump_output(cuobjdump_output).unwrap();

        // Check function mappings
        let kernel_a_mappings = mapper.get_function_mappings("kernel_a");
        assert_eq!(kernel_a_mappings.len(), 2);
        assert!(kernel_a_mappings.iter().all(|m| m.ptx_line == 10));

        let kernel_b_mappings = mapper.get_function_mappings("kernel_b");
        assert_eq!(kernel_b_mappings.len(), 2);
        assert!(kernel_b_mappings.iter().all(|m| m.ptx_line == 20));
    }

    #[test]
    fn test_sass_ptx_mapper_ordered_addresses() {
        let mut mapper = SassPtxMapper::new();
        mapper.add_mapping(0x30, "test.ptx", 3, "INST3");
        mapper.add_mapping(0x10, "test.ptx", 1, "INST1");
        mapper.add_mapping(0x20, "test.ptx", 2, "INST2");

        let ordered = mapper.get_sass_addresses_ordered();
        assert_eq!(ordered, vec![0x10, 0x20, 0x30]);
    }

    #[test]
    fn test_hetgpu_debug_next_ptx_line() {
        let mut iface = HetGpuDebugInterface::new();

        iface
            .get_mapper_mut()
            .add_mapping(0x0, "kernel.ptx", 10, "MOV");
        iface
            .get_mapper_mut()
            .add_mapping(0x10, "kernel.ptx", 10, "ADD");
        iface
            .get_mapper_mut()
            .add_mapping(0x20, "kernel.ptx", 11, "MUL");
        iface
            .get_mapper_mut()
            .add_mapping(0x30, "kernel.ptx", 12, "STG");

        // From address 0x0 (line 10), next PTX line should be at 0x20 (line 11)
        let next = iface.get_next_ptx_line_address(0x0);
        assert_eq!(next, Some(0x20));

        // From address 0x10 (still line 10), next PTX line should still be 0x20 (line 11)
        let next = iface.get_next_ptx_line_address(0x10);
        assert_eq!(next, Some(0x20));

        // From address 0x20 (line 11), next PTX line should be 0x30 (line 12)
        let next = iface.get_next_ptx_line_address(0x20);
        assert_eq!(next, Some(0x30));
    }

    #[test]
    fn test_sass_ptx_mapper_dump() {
        let mut mapper = SassPtxMapper::new();
        mapper.add_mapping(0x0, "test.ptx", 1, "MOV");
        mapper.add_mapping(0x10, "test.ptx", 2, "ADD");

        let dump = mapper.dump_mapping_table();
        assert!(dump.contains("SASS ↔ PTX Mapping Table"));
        assert!(dump.contains("0x00000000"));
        assert!(dump.contains("0x00000010"));
        assert!(dump.contains("test.ptx:1"));
        assert!(dump.contains("test.ptx:2"));
    }

    #[test]
    fn test_cubin_debug_info() {
        let debug_info = CubinDebugInfo::new();
        assert!(debug_info.symbols.is_empty());
        assert!(debug_info.debug_line_entries.is_empty());
    }
}
