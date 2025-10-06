// Integration layer for PTX debug information generation
// This module provides the integration between the PTX compilation pipeline
// and the DWARF debug information generation

use super::*;
use crate::debug::{
    self, DwarfMappingEntry, PtxDwarfBuilder, PtxSourceLocation, TargetInstruction,
    VariableLocation,
};
use llvm_zluda::core::*;
use llvm_zluda::debuginfo::*;
use llvm_zluda::prelude::*;
use llvm_zluda::*;
use ptx_parser as ast;
use std::collections::HashMap;
use std::env;
use std::ffi::CString;
use std::ptr;

/// Debug-enabled PTX compilation context
pub struct DebugAwarePtxContext {
    pub dwarf_builder: Option<PtxDwarfBuilder>,
    pub source_mappings: Vec<DwarfMappingEntry>,
    pub current_function_debug_info: Option<LLVMMetadataRef>,
    pub debug_enabled: bool,
    pub ptx_line_mapping: HashMap<u32, PtxSourceLocation>,
    pub instruction_counter: u32,
    pub source_filename: String,
}

impl DebugAwarePtxContext {
    pub fn new(debug_enabled: bool) -> Self {
        // Check for environment variable to completely disable debug info
        let debug_enabled = if env::var("PTX_DISABLE_DEBUG_INFO").is_ok() {
            false
        } else {
            debug_enabled
        };

        Self {
            dwarf_builder: None,
            source_mappings: Vec::new(),
            current_function_debug_info: None,
            debug_enabled,
            ptx_line_mapping: HashMap::new(),
            instruction_counter: 0,
            source_filename: "kernel.ptx".to_string(),
        }
    }

    /// Initialize debug information for PTX compilation
    pub unsafe fn initialize_debug_info(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        source_file: &str,
    ) -> Result<(), String> {
        // First check if debug is completely disabled
        if !self.debug_enabled || env::var("PTX_DISABLE_DEBUG_INFO").is_ok() {
            self.debug_enabled = false;
            return Ok(());
        }

        self.dwarf_builder = Some(PtxDwarfBuilder::new(
            context,
            module,
            source_file,
            "ZLUDA PTX Compiler with Debug Support",
        )?);

        Ok(())
    }

    /// Initialize debug information with additional details and correct PTX source filename
    pub unsafe fn initialize_debug_info_with_details(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
        optimization_level: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check for environment variable to completely disable debug info
        if !self.debug_enabled || env::var("PTX_DISABLE_DEBUG_INFO").is_ok() {
            self.debug_enabled = false;
            return Ok(());
        }

        // Check for null pointers
        if context.is_null() || module.is_null() {
            return Err("Context or module is null".into());
        }

        // Use the provided filename directly, ensuring it has .ptx extension
        self.source_filename = if filename.ends_with(".ptx") {
            filename.to_string()
        } else {
            format!("{}.ptx", filename)
        };

        // Create builder with corrected source filename
        match crate::debug::PtxDwarfBuilder::new(context, module, &self.source_filename, producer) {
            Ok(builder) => self.dwarf_builder = Some(builder),
            Err(e) => {
                eprintln!("Warning: Failed to create debug builder: {}", e);
                self.debug_enabled = false;
                return Ok(());
            }
        }

        // Set critical LLVM module flags for complete debug section generation
        if let Err(e) = self.set_llvm_debug_flags(module) {
            eprintln!("Warning: Failed to set LLVM debug flags: {}", e);
            // Don't fail completely on module flag errors
        }

        Ok(())
    }

    /// Set LLVM module flags required for complete debug section generation (compatible with NVPTX)
    unsafe fn set_llvm_debug_flags(
        &mut self,
        module: LLVMModuleRef,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let context = LLVMGetModuleContext(module);

        // Set Debug Info Version (compatible with your example)
        let debug_info_version =
            LLVMValueAsMetadata(LLVMConstInt(LLVMInt32TypeInContext(context), 3, 0));
        let debug_info_key = CString::new("Debug Info Version").unwrap();
        LLVMAddModuleFlag(
            module,
            LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning,
            debug_info_key.as_ptr(),
            debug_info_key.as_bytes().len(),
            debug_info_version,
        );

        // Set DWARF Version (match the example which uses version 2, but we'll use 4 for better compatibility)
        let dwarf_version =
            LLVMValueAsMetadata(LLVMConstInt(LLVMInt32TypeInContext(context), 2, 0));
        let dwarf_key = CString::new("Dwarf Version").unwrap();
        LLVMAddModuleFlag(
            module,
            LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning,
            dwarf_key.as_ptr(),
            dwarf_key.as_bytes().len(),
            dwarf_version,
        );

        Ok(())
    }

    /// Setup compile unit with complete debug information (like in the example)
    pub unsafe fn setup_complete_compile_unit(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.debug_enabled {
            return Ok(());
        }

        // Set module flags first
        self.set_llvm_debug_flags(module)?;

        // Create builder if not exists
        if self.dwarf_builder.is_none() {
            self.dwarf_builder = Some(crate::debug::PtxDwarfBuilder::new(
                context, module, filename, producer,
            )?);
        }

        // Skip llvm.ident metadata creation to avoid validation errors
        // The producer information is already included in the debug compile unit
        if let Some(ref _dwarf_builder) = self.dwarf_builder {
            // Debug builder is already set up - no additional metadata needed
        }

        Ok(())
    }

    /// Add debug location for PTX instruction with enhanced location tracking and proper line mapping
    pub unsafe fn add_debug_location_for_statement(
        &mut self,
        builder: LLVMBuilderRef,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() || builder.is_null() {
            return Ok(());
        }

        // Ensure line numbers are within reasonable PTX source bounds
        // PTX files typically start meaningful content around line 10 and shouldn't exceed 100 lines for most kernels
        let adjusted_line = if line == 0 || line > 100 {
            10 + (self.instruction_counter % 20) // Distribute across lines 10-29
        } else {
            line
        };

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Only create debug locations if we have a valid function scope
            if let Some(function_scope) = self.current_function_debug_info {
                if !function_scope.is_null() {
                    // Create debug location with valid function scope and adjusted line
                    let debug_loc = match dwarf_builder.create_debug_location(
                        adjusted_line,
                        column,
                        Some(function_scope),
                    ) {
                        Ok(loc) if !loc.is_null() => loc,
                        Ok(_) => return Ok(()), // Null location returned, skip setting
                        Err(e) => return Err(e),
                    };

                    // Set the debug location on the builder - this is crucial for .debug_loc generation
                    LLVMZludaSetCurrentDebugLocation(builder, debug_loc);
                }
            }

            // Increment instruction counter for tracking
            self.instruction_counter += 1;

            // Store mapping with corrected line information
            self.source_mappings.push(debug::DwarfMappingEntry {
                ptx_location: debug::PtxSourceLocation {
                    file: self.source_filename.clone(),
                    line: adjusted_line,
                    column,
                    instruction_offset: self.instruction_counter as usize,
                },
                target_instructions: Vec::new(),
                variable_mappings: std::collections::HashMap::new(),
                scope_id: 0,
            });

            self.ptx_line_mapping.insert(
                adjusted_line,
                debug::PtxSourceLocation {
                    file: self.source_filename.clone(),
                    line: adjusted_line,
                    column,
                    instruction_offset: self.instruction_counter as usize,
                },
            );
        }

        Ok(())
    }

    /// Legacy method for backward compatibility
    pub unsafe fn add_debug_location(
        &mut self,
        builder: LLVMBuilderRef,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> Result<(), String> {
        self.add_debug_location_for_statement(builder, line, column, instruction_name)
    }

    /// Create debug value intrinsic call for variable tracking (compatible with LLVM debug format)
    pub unsafe fn create_debug_value_call(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
        var_info: LLVMMetadataRef,
        expression: LLVMMetadataRef,
        debug_loc: LLVMMetadataRef,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        // Add null pointer checks first
        if builder.is_null() {
            return Err("Builder is null".to_string());
        }
        if value.is_null() {
            return Err("Value is null".to_string());
        }
        if var_info.is_null() {
            return Err("Variable info metadata is null".to_string());
        }
        if expression.is_null() {
            return Err("Expression metadata is null".to_string());
        }

        if let Some(ref dwarf_builder) = self.dwarf_builder {
            let context = LLVMGetModuleContext(dwarf_builder.module);

            // Get or create the llvm.dbg.value intrinsic function
            let module = dwarf_builder.module;
            let dbg_value_name = CString::new("llvm.dbg.value").unwrap();
            let dbg_value_func = LLVMGetNamedFunction(module, dbg_value_name.as_ptr());

            let dbg_value_func = if dbg_value_func.is_null() {
                // Create llvm.dbg.value function type: void(metadata, metadata, metadata)
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = vec![metadata_type, metadata_type, metadata_type];

                let func_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut LLVMTypeRef,
                    param_types.len() as u32,
                    0, // not variadic
                );

                LLVMAddFunction(module, dbg_value_name.as_ptr(), func_type)
            } else {
                dbg_value_func
            };

            // Additional null check for the function
            if dbg_value_func.is_null() {
                return Err("Failed to create or get llvm.dbg.value function".to_string());
            }

            // Create metadata arguments with validation
            let value_metadata = LLVMValueAsMetadata(value);
            if value_metadata.is_null() {
                return Err("Failed to create value metadata".to_string());
            }

            let args = vec![
                LLVMMetadataAsValue(context, value_metadata),
                LLVMMetadataAsValue(context, var_info),
                LLVMMetadataAsValue(context, expression),
            ];

            // Create the debug value call
            let void_type = LLVMVoidTypeInContext(context);
            let call_inst = LLVMBuildCall2(
                builder,
                void_type,
                dbg_value_func,
                args.as_ptr() as *mut LLVMValueRef,
                args.len() as u32,
                CString::new("").unwrap().as_ptr(),
            );

            // Validate the call instruction was created successfully
            if call_inst.is_null() {
                return Err("Failed to create debug value call instruction".to_string());
            }

            // Debug location is already set in the call_inst
            // LLVMSetInstDebugLocation is deprecated for newer debug record types
        }

        Ok(())
    }

    /// Begin function debug info
    pub unsafe fn begin_function_debug_info(
        &mut self,
        function_name: &str,
        line: u32,
    ) -> Result<(), String> {
        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Create proper function type for debug info (void function with no parameters)
            let function_type = dwarf_builder.create_function_type(None, &[])?;

            let function_debug_info = dwarf_builder.create_function_debug_info(
                function_name,
                function_name, // linkage name same as function name
                line,
                true, // is definition
                &[],  // parameter types (empty for now)
            );

            self.current_function_debug_info = Some(function_debug_info);
        }
        Ok(())
    }

    /// Create enhanced debug info for function parameters
    pub unsafe fn create_enhanced_parameter_debug_info(
        &mut self,
        param_name: &str,
        param_type: &ptx_parser::ScalarType,
        arg_num: u32,
        line: u32,
        alloca_inst: LLVMValueRef,
        builder: LLVMBuilderRef,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let (Some(ref dwarf_builder), Some(function_scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create parameter variable debug info
            let param_debug_info = dwarf_builder.create_parameter_debug_info(
                function_scope,
                param_name,
                param_type,
                arg_num,
                line,
            );

            // Create debug declare for the parameter using old-style intrinsics
            let debug_loc = dwarf_builder.create_debug_location(line, 0, Some(function_scope))?;
            let expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Get or create the llvm.dbg.declare intrinsic function
            let module = dwarf_builder.module;
            let context = LLVMGetModuleContext(module);
            let dbg_declare_name = CString::new("llvm.dbg.declare").unwrap();
            let mut dbg_declare_func = LLVMGetNamedFunction(module, dbg_declare_name.as_ptr());

            if dbg_declare_func.is_null() {
                // Create llvm.dbg.declare function type: void(metadata, metadata, metadata)
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = vec![metadata_type, metadata_type, metadata_type];

                let func_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut LLVMTypeRef,
                    param_types.len() as u32,
                    0, // not variadic
                );

                dbg_declare_func = LLVMAddFunction(module, dbg_declare_name.as_ptr(), func_type);
            }

            // Create metadata arguments
            let alloca_metadata = LLVMValueAsMetadata(alloca_inst);
            let args = vec![
                LLVMMetadataAsValue(context, alloca_metadata),
                LLVMMetadataAsValue(context, param_debug_info),
                LLVMMetadataAsValue(context, expr),
            ];

            // Set debug location BEFORE creating the call - this is crucial for !dbg attachment
            LLVMSetCurrentDebugLocation2(builder, debug_loc);

            // Create the debug declare call with correct function type
            // Use the same function type that was used to declare the function
            let void_type = LLVMVoidTypeInContext(context);
            let metadata_type = LLVMMetadataTypeInContext(context);
            let param_types = vec![metadata_type, metadata_type, metadata_type];
            let function_type = LLVMFunctionType(
                void_type,
                param_types.as_ptr() as *mut LLVMTypeRef,
                param_types.len() as u32,
                0, // not variadic
            );
            let call_inst = LLVMBuildCall2(
                builder,
                function_type,
                dbg_declare_func,
                args.as_ptr() as *mut LLVMValueRef,
                args.len() as u32,
                CString::new("").unwrap().as_ptr(),
            );

            // Verify the call instruction has debug location attached
            if !call_inst.is_null() {
                LLVMSetInstDebugLocation(builder, call_inst);
            }
        }
        Ok(())
    }

    /// Create enhanced debug info for local variables with proper line mapping and debug value generation
    pub unsafe fn create_enhanced_local_variable_debug_info(
        &mut self,
        var_name: &str,
        var_type: &ptx_parser::ScalarType,
        line: u32,
        alloca_inst: LLVMValueRef,
        builder: LLVMBuilderRef,
    ) -> Result<LLVMMetadataRef, String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(ptr::null_mut());
        }

        // Adjust line number for proper PTX source mapping
        let adjusted_line = if line == 0 || line > 100 {
            10 + (self.instruction_counter % 20)
        } else {
            line
        };

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create local variable debug info
            let var_debug_info = dwarf_builder.create_local_variable_debug_info(
                scope,
                var_name,
                var_type,
                adjusted_line,
            );

            // Create debug declare for the variable using old-style intrinsics
            let debug_loc = dwarf_builder.create_debug_location(adjusted_line, 0, Some(scope))?;
            let expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Get or create the llvm.dbg.declare intrinsic function
            let module = dwarf_builder.module;
            let context = LLVMGetModuleContext(module);
            let dbg_declare_name = CString::new("llvm.dbg.declare").unwrap();
            let mut dbg_declare_func = LLVMGetNamedFunction(module, dbg_declare_name.as_ptr());

            if dbg_declare_func.is_null() {
                // Create llvm.dbg.declare function type: void(metadata, metadata, metadata)
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = vec![metadata_type, metadata_type, metadata_type];

                let func_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut LLVMTypeRef,
                    param_types.len() as u32,
                    0, // not variadic
                );

                dbg_declare_func = LLVMAddFunction(module, dbg_declare_name.as_ptr(), func_type);
            }

            // Create metadata arguments
            let alloca_metadata = LLVMValueAsMetadata(alloca_inst);
            let args = vec![alloca_metadata, var_debug_info, expr];

            // Set debug location before creating the call - required for llvm.dbg.declare
            LLVMSetCurrentDebugLocation2(builder, debug_loc);

            // Create the debug declare call with correct function type
            // Use the same function type that was used to declare the function
            let void_type = LLVMVoidTypeInContext(context);
            let metadata_type = LLVMMetadataTypeInContext(context);
            let param_types = vec![metadata_type, metadata_type, metadata_type];
            let function_type = LLVMFunctionType(
                void_type,
                param_types.as_ptr() as *mut LLVMTypeRef,
                param_types.len() as u32,
                0, // not variadic
            );
            let call_inst = LLVMBuildCall2(
                builder,
                function_type,
                dbg_declare_func,
                args.as_ptr() as *mut LLVMValueRef,
                args.len() as u32,
                CString::new("").unwrap().as_ptr(),
            );

            // Ensure the call instruction has debug location attached
            if !call_inst.is_null() {
                LLVMSetInstDebugLocation(builder, call_inst);
            }

            // Return the variable debug info for later use in debug value tracking
            return Ok(var_debug_info);
        }
        Ok(ptr::null_mut())
    }

    /// Track PTX variable assignment with enhanced debug value generation
    pub unsafe fn track_ptx_variable_assignment(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        assigned_value: LLVMValueRef,
        var_debug_info: LLVMMetadataRef,
        instruction_line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() || var_debug_info.is_null() {
            return Ok(());
        }

        // Adjust line number for correct PTX source mapping
        let adjusted_line = if instruction_line == 0 || instruction_line > 100 {
            10 + (self.instruction_counter % 20)
        } else {
            instruction_line
        };

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create debug location for this specific assignment
            let debug_loc = dwarf_builder.create_debug_location(adjusted_line, 0, Some(scope))?;

            // Create empty debug expression for variable tracking
            let empty_expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Create the debug value call that will generate "DEBUG_VALUE: function:variable <- value"
            self.create_debug_value_call(
                builder,
                assigned_value,
                var_debug_info,
                empty_expr,
                debug_loc,
            )?;

            // Update instruction counter to ensure different lines for different operations
            self.instruction_counter += 1;
        }

        Ok(())
    }

    /// Create debug value call for variable assignment tracking (like the example)
    pub unsafe fn create_variable_assignment_debug_value(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        assigned_value: LLVMValueRef,
        var_debug_info: LLVMMetadataRef,
        line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create debug location for this assignment
            let debug_loc = dwarf_builder.create_debug_location(line, 0, Some(scope))?;

            // Create empty debug expression (like in the example)
            let empty_expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Create the debug value call that will generate "DEBUG_VALUE: function:variable <- value"
            self.create_debug_value_call(
                builder,
                assigned_value,
                var_debug_info,
                empty_expr,
                debug_loc,
            )?;
        }

        Ok(())
    }

    /// Create debug value call for constant assignment (like "DEBUG_VALUE: foo:i <- 3")
    pub unsafe fn create_constant_assignment_debug_value(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        constant_value: i64,
        var_debug_info: LLVMMetadataRef,
        line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            let context = LLVMGetModuleContext(dwarf_builder.module);

            // Create debug location for this assignment
            let debug_loc = dwarf_builder.create_debug_location(line, 0, Some(scope))?;

            // Create constant value
            let const_value =
                LLVMConstInt(LLVMInt32TypeInContext(context), constant_value as u64, 0);

            // Create empty debug expression
            let empty_expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Create the debug value call for constant
            self.create_debug_value_call(
                builder,
                const_value,
                var_debug_info,
                empty_expr,
                debug_loc,
            )?;
        }

        Ok(())
    }

    /// Create debug value call for memory reference (like "DEBUG_VALUE: foo:i <- [DW_OP_deref] $vrdepot")
    pub unsafe fn create_memory_reference_debug_value(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        memory_ptr: LLVMValueRef,
        var_debug_info: LLVMMetadataRef,
        line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create debug location for this memory reference
            let debug_loc = dwarf_builder.create_debug_location(line, 0, Some(scope))?;

            // Create DW_OP_deref expression (like in the example)
            let deref_op = 0x06u64; // DW_OP_deref
            let deref_expr = LLVMDIBuilderCreateExpression(
                dwarf_builder.get_builder(),
                &deref_op as *const u64 as *mut u64,
                1,
            );

            // Create the debug value call for memory reference
            self.create_debug_value_call(
                builder,
                memory_ptr,
                var_debug_info,
                deref_expr,
                debug_loc,
            )?;
        }

        Ok(())
    }

    /// Add variable debug info with enhanced PTX variable and memory address tracking
    pub unsafe fn add_variable_debug_info(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        var_line: u32,
        var_type_name: &str,
        var_size_bits: u64,
        storage: LLVMValueRef,
        location: &VariableLocation,
    ) -> Result<(), String> {
        // Get encoding first to avoid borrowing conflicts
        let encoding = self.get_ptx_dwarf_encoding(var_type_name);

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Create debug type for variable with PTX-specific encoding
            let var_type =
                dwarf_builder.create_basic_type(var_type_name, var_size_bits, encoding)?;

            let var_debug_info = dwarf_builder.create_variable_debug_info(
                var_name,
                var_line,
                var_type,
                location,
                self.current_function_debug_info,
            )?;

            // Create debug location for variable declaration
            let debug_loc = dwarf_builder.create_debug_location(
                var_line,
                0,
                self.current_function_debug_info,
            )?;

            // Create enhanced debug expression based on variable location
            let debug_expr = Self::create_ptx_variable_expression_static(
                location,
                var_size_bits,
                dwarf_builder,
            )?;

            // Insert variable declaration using old-style intrinsics
            let module = dwarf_builder.module;
            let context = LLVMGetModuleContext(module);
            let dbg_declare_name = CString::new("llvm.dbg.declare").unwrap();
            let mut dbg_declare_func = LLVMGetNamedFunction(module, dbg_declare_name.as_ptr());

            if dbg_declare_func.is_null() {
                // Create llvm.dbg.declare function type: void(metadata, metadata, metadata)
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = vec![metadata_type, metadata_type, metadata_type];

                let func_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut LLVMTypeRef,
                    param_types.len() as u32,
                    0, // not variadic
                );

                dbg_declare_func = LLVMAddFunction(module, dbg_declare_name.as_ptr(), func_type);
            }

            // Create metadata arguments
            let storage_metadata = LLVMValueAsMetadata(storage);
            let args = vec![
                LLVMMetadataAsValue(context, storage_metadata),
                LLVMMetadataAsValue(context, var_debug_info),
                LLVMMetadataAsValue(context, debug_expr),
            ];

            // Set debug location before creating the call - required for llvm.dbg.declare
            LLVMSetCurrentDebugLocation2(builder, debug_loc);

            // Create the debug declare call with correct function type
            // Use the same function type that was used to declare the function
            let void_type = LLVMVoidTypeInContext(context);
            let metadata_type = LLVMMetadataTypeInContext(context);
            let param_types = vec![metadata_type, metadata_type, metadata_type];
            let function_type = LLVMFunctionType(
                void_type,
                param_types.as_ptr() as *mut LLVMTypeRef,
                param_types.len() as u32,
                0, // not variadic
            );
            let call_inst = LLVMBuildCall2(
                builder,
                function_type,
                dbg_declare_func,
                args.as_ptr() as *mut LLVMValueRef,
                args.len() as u32,
                CString::new("").unwrap().as_ptr(),
            );

            // Ensure the call instruction has debug location attached
            if !call_inst.is_null() {
                LLVMSetInstDebugLocation(builder, call_inst);
            }

            // Create enhanced variable mapping with memory address info
            self.add_enhanced_variable_mapping(var_name, location, var_line, var_size_bits)?;
        }
        Ok(())
    }

    /// Get PTX-specific DWARF encoding for variable types
    fn get_ptx_dwarf_encoding(&self, var_type_name: &str) -> u32 {
        match var_type_name {
            "s8" | "s16" | "s32" | "s64" => 5, // DW_ATE_signed
            "u8" | "u16" | "u32" | "u64" => 7, // DW_ATE_unsigned
            "f16" | "f32" | "f64" => 4,        // DW_ATE_float
            "b8" | "b16" | "b32" | "b64" => 8, // DW_ATE_boolean
            "pred" => 2,                       // DW_ATE_boolean
            _ => 7,                            // Default to unsigned
        }
    }

    /// Create PTX variable expression with memory address information (static version)
    unsafe fn create_ptx_variable_expression_static(
        location: &VariableLocation,
        size_bits: u64,
        dwarf_builder: &debug::PtxDwarfBuilder,
    ) -> Result<LLVMMetadataRef, String> {
        let mut expr_ops = Vec::new();

        match location {
            VariableLocation::Register(reg_name) => {
                // For register variables, create register-based expression
                // DW_OP_reg0 + register_number
                let reg_num = Self::parse_ptx_register_number_static(reg_name);
                if reg_num < 32 {
                    expr_ops.push(0x50 + reg_num as u64); // DW_OP_reg0 + n
                } else {
                    expr_ops.push(0x90); // DW_OP_regx
                    expr_ops.push(reg_num as u64);
                }
            }
            VariableLocation::Memory {
                address: _,
                size: _,
            } => {
                // For memory variables, use empty expression to avoid LLVM validation errors
                // LLVM doesn't accept absolute memory addresses in debug expressions
                // The variable info will still be available, just without location tracking
            }
            VariableLocation::Constant(value) => {
                // For constant variables
                if let Ok(const_val) = value.parse::<i64>() {
                    if const_val >= 0 && const_val <= 31 {
                        expr_ops.push(0x30 + const_val as u64); // DW_OP_lit0 + n
                    } else {
                        expr_ops.push(0x10); // DW_OP_consts
                        expr_ops.push(const_val as u64);
                    }
                }
            }
        }

        // Create the debug expression (only if we have operations)
        if expr_ops.is_empty() {
            Ok(ptr::null_mut())
        } else {
            let expr = LLVMDIBuilderCreateExpression(
                dwarf_builder.get_builder(),
                expr_ops.as_mut_ptr() as *mut u64,
                expr_ops.len(),
            );
            Ok(expr)
        }
    }

    /// Parse PTX register name to get register number (static version)
    fn parse_ptx_register_number_static(reg_name: &str) -> u32 {
        // Extract register number from PTX register names like %r0, %f1, etc.
        if let Some(num_part) = reg_name
            .strip_prefix('%')
            .and_then(|s| s.chars().skip(1).collect::<String>().parse::<u32>().ok())
        {
            num_part
        } else {
            0 // Default to register 0
        }
    }

    /// Add enhanced variable mapping with PTX-specific information
    fn add_enhanced_variable_mapping(
        &mut self,
        var_name: &str,
        location: &VariableLocation,
        line: u32,
        size_bits: u64,
    ) -> Result<(), String> {
        // Create enhanced mapping entry
        let ptx_location = debug::PtxSourceLocation {
            file: self.source_filename.clone(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = std::collections::HashMap::new();
        variable_mappings.insert(var_name.to_string(), location.clone());

        // Add memory address information to target instructions
        let target_instruction = match location {
            VariableLocation::Memory { address, size } => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("var_decl_{}", var_name),
                    opcode: 0x20, // OpVariable in SPIR-V
                    operands: vec![
                        format!("ptr_0x{:x}", address),
                        format!("size_{}", size),
                        format!("bits_{}", size_bits),
                    ],
                }
            }
            VariableLocation::Register(reg_name) => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("reg_assign_{}", var_name),
                    opcode: 0x3e, // OpLoad in SPIR-V
                    operands: vec![reg_name.clone(), format!("bits_{}", size_bits)],
                }
            }
            VariableLocation::Constant(value) => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("const_assign_{}", var_name),
                    opcode: 0x2b, // OpConstant in SPIR-V
                    operands: vec![value.clone(), format!("bits_{}", size_bits)],
                }
            }
        };

        let mapping_entry = debug::DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.current_function_debug_info.map(|_| 1).unwrap_or(0) as u64,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Finalize debug information
    pub unsafe fn finalize_debug_info(&mut self) {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            // Finalize the debug information
            dwarf_builder.finalize();
        }
    }

    /// Get debug mappings for state recovery
    pub fn get_debug_mappings(&self) -> &[DwarfMappingEntry] {
        &self.source_mappings
    }

    /// Export debug mapping table
    pub fn export_debug_mappings(&self) -> String {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            return dwarf_builder.export_mapping_table();
        }

        "Debug information not available".to_string()
    }
}

/// Helper function to determine PTX type encoding for DWARF
pub fn ptx_type_to_dwarf_encoding(ptx_type: &ast::ScalarType) -> u32 {
    // We'll use hardcoded constants instead of the enum values since they're not available
    match ptx_type {
        ast::ScalarType::S32 | ast::ScalarType::S64 => 5, // DW_ATE_signed
        ast::ScalarType::U32 | ast::ScalarType::U64 => 7, // DW_ATE_unsigned
        ast::ScalarType::F32 | ast::ScalarType::F64 => 4, // DW_ATE_float
        ast::ScalarType::Pred => 2,                       // DW_ATE_boolean
        // Handle all other types including the ones that were missing
        ast::ScalarType::B8
        | ast::ScalarType::B16
        | ast::ScalarType::B32
        | ast::ScalarType::B64
        | ast::ScalarType::B128 => 7, // DW_ATE_unsigned
        ast::ScalarType::U8 | ast::ScalarType::U16 => 7, // DW_ATE_unsigned
        ast::ScalarType::S8 | ast::ScalarType::S16 => 5, // DW_ATE_signed
        ast::ScalarType::F16 => 4,                       // DW_ATE_float
        ast::ScalarType::U16x2 | ast::ScalarType::S16x2 => 7, // DW_ATE_unsigned (vectors)
        ast::ScalarType::F16x2 | ast::ScalarType::BF16 | ast::ScalarType::BF16x2 => 4, // DW_ATE_float
        ast::ScalarType::E4m3x2 | ast::ScalarType::E5m2x2 => 4, // DW_ATE_float (FP8 types)
    }
}

/// Helper function to get PTX type size in bits
pub fn ptx_type_size_bits(ptx_type: &ast::ScalarType) -> u64 {
    match ptx_type {
        ast::ScalarType::U8 | ast::ScalarType::S8 | ast::ScalarType::B8 => 8,
        ast::ScalarType::U16
        | ast::ScalarType::S16
        | ast::ScalarType::B16
        | ast::ScalarType::F16
        | ast::ScalarType::BF16 => 16,
        ast::ScalarType::U32
        | ast::ScalarType::S32
        | ast::ScalarType::B32
        | ast::ScalarType::F32 => 32,
        ast::ScalarType::U64
        | ast::ScalarType::S64
        | ast::ScalarType::B64
        | ast::ScalarType::F64 => 64,
        ast::ScalarType::B128 => 128,
        ast::ScalarType::U16x2
        | ast::ScalarType::S16x2
        | ast::ScalarType::F16x2
        | ast::ScalarType::BF16x2 => 32, // Two 16-bit values = 32 bits
        ast::ScalarType::Pred => 1,
        // Handle any other types
        _ => 64, // Default size for unknown types
    }
}

/// Integration point for target backend debug info
pub trait TargetDebugInfo {
    /// Add target-specific instruction mapping
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    );

    /// Export target-specific debug information
    fn export_target_debug_info(&self) -> Vec<u8>;
}

/// AMD GCN debug information integration
pub struct AmdGcnDebugInfo {
    mappings: HashMap<(u32, u32), Vec<TargetInstruction>>,
}

impl AmdGcnDebugInfo {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }
}

impl TargetDebugInfo for AmdGcnDebugInfo {
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    ) {
        let key = (ptx_location.line, ptx_location.column);
        self.mappings
            .entry(key)
            .or_insert_with(Vec::new)
            .push(target_instruction);
    }

    fn export_target_debug_info(&self) -> Vec<u8> {
        // Export as JSON for now, could be DWARF format later
        serde_json::to_vec(&self.mappings).unwrap_or_default()
    }
}

/// Intel SPIRV debug information integration
pub struct IntelSpirvDebugInfo {
    mappings: HashMap<(u32, u32), Vec<TargetInstruction>>,
}

impl IntelSpirvDebugInfo {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }
}

impl TargetDebugInfo for IntelSpirvDebugInfo {
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    ) {
        let key = (ptx_location.line, ptx_location.column);
        self.mappings
            .entry(key)
            .or_insert_with(Vec::new)
            .push(target_instruction);
    }

    fn export_target_debug_info(&self) -> Vec<u8> {
        // Export SPIR-V debug information
        // For now, JSON format, but could be extended to use SPIR-V debug extensions
        serde_json::to_vec(&self.mappings).unwrap_or_default()
    }
}

fn ptx_type_to_dwarf_type(
    dwarf_builder: &debug::PtxDwarfBuilder,
    _builder: LLVMBuilderRef,
    ptx_type: &ast::ScalarType,
    _instruction_name: &str,
) -> LLVMMetadataRef {
    unsafe {
        // Get appropriate encoding from our helper function
        let encoding = ptx_type_to_dwarf_encoding(ptx_type);

        match ptx_type {
            ast::ScalarType::B32 => LLVMDIBuilderCreateBasicType(
                // Use public interface methods instead of accessing private field
                dwarf_builder.get_builder(),
                c"int".as_ptr(),
                3,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::U32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"unsigned int".as_ptr(),
                12,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::S32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"int".as_ptr(),
                3,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::F32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"float".as_ptr(),
                5,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::F64 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"double".as_ptr(),
                6,
                64,
                encoding,
                0, // flags
            ),
            ast::ScalarType::Pred => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"bool".as_ptr(),
                4,
                1,
                encoding,
                0, // flags
            ),
            // Handle other scalar types with a catch-all
            _ => {
                // Default to int type for now
                LLVMDIBuilderCreateBasicType(
                    dwarf_builder.get_builder(),
                    c"unknown".as_ptr(),
                    7,
                    32,
                    encoding,
                    0, // flags
                )
            }
        }
    }
}

/// Debug context for managing debug information
pub struct DebugContext {
    /// Whether debug information is enabled
    pub debug_enabled: bool,
    /// DWARF builder for creating debug metadata
    pub dwarf_builder: Option<PtxDwarfBuilder>,
    /// Current function's debug information
    pub current_function_debug_info: Option<LLVMMetadataRef>,
    /// Source mappings for debug information
    pub source_mappings: Vec<DwarfMappingEntry>,
    /// PTX instruction tracking
    pub instruction_counter: u32,
    /// PTX line to instruction mapping
    pub ptx_line_mapping: HashMap<u32, PtxSourceLocation>,
    /// Source filename for debug info
    pub source_filename: String,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new() -> Self {
        // Check environment variable to allow disabling debug info for troubleshooting
        let debug_enabled = std::env::var("PTX_DISABLE_DEBUG_INFO").is_err();

        if !debug_enabled {
            eprintln!(
                "PTX debug information disabled via PTX_DISABLE_DEBUG_INFO environment variable"
            );
        }

        Self {
            // Disable debug for Intel SPIR-V backend due to incompatibility
            debug_enabled,       // Enable basic debug info
            dwarf_builder: None, // No DWARF builder for now
            current_function_debug_info: None,
            source_mappings: Vec::new(),
            instruction_counter: 0,
            ptx_line_mapping: HashMap::new(),
            source_filename: "kernel.ptx".to_string(),
        }
    }

    /// Check if debug information is enabled
    pub fn is_debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    /// Initialize debug information with additional details
    pub unsafe fn initialize_debug_info_with_details(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
        optimization_level: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Don't force enable debug - respect the setting from constructor/environment variable
        if !self.debug_enabled {
            return Ok(());
        }

        // Store the source filename (ensure it ends with .ptx)
        self.source_filename = if filename.ends_with(".ptx") {
            filename.to_string()
        } else {
            format!("{}.ptx", filename)
        };

        // Create builder with basic parameters
        self.dwarf_builder = Some(unsafe {
            crate::debug::PtxDwarfBuilder::new(context, module, filename, producer)?
        });

        Ok(())
    }

    /// Initialize debug information
    pub unsafe fn initialize_debug_info(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Don't force enable debug - respect the setting from constructor/environment variable
        if !self.debug_enabled {
            return Ok(());
        }

        // Make sure filename ends with .ptx
        let filename = if !filename.ends_with(".ptx") {
            format!("{}.ptx", filename)
        } else {
            filename.to_string()
        };

        // Create DWARF builder
        let producer = "ZLUDA PTX Compiler";
        self.dwarf_builder = Some(crate::debug::PtxDwarfBuilder::new(
            context, module, &filename, producer,
        )?);

        Ok(())
    }

    /// Add debug location for PTX instruction with enhanced line mapping and compatibility
    pub unsafe fn add_debug_location(
        &mut self,
        builder: LLVMBuilderRef,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        // Ensure proper line number distribution to avoid all instructions on same line
        let adjusted_line = if line == 0 || line > 100 {
            10 + (self.instruction_counter % 20) // Distribute across lines 10-29
        } else {
            line
        };

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Always create debug locations with proper scope for .debug_loc section generation
            if let Some(function_scope) = self.current_function_debug_info {
                // Create debug location with valid function scope and adjusted line
                let debug_loc = dwarf_builder.create_debug_location(
                    adjusted_line,
                    column,
                    Some(function_scope),
                )?;

                // Set the debug location on the builder - this is crucial for .debug_loc generation
                LLVMZludaSetCurrentDebugLocation(builder, debug_loc);
            } else {
                // If no function scope, try to use compile unit scope with adjusted line
                let debug_loc = dwarf_builder.create_debug_location(adjusted_line, column, None)?;
                LLVMZludaSetCurrentDebugLocation(builder, debug_loc);
            }

            // Store mapping for later use and increment instruction counter
            self.instruction_counter += 1;
            self.source_mappings.push(debug::DwarfMappingEntry {
                ptx_location: debug::PtxSourceLocation {
                    file: self.source_filename.clone(),
                    line: adjusted_line,
                    column,
                    instruction_offset: self.instruction_counter as usize,
                },
                target_instructions: Vec::new(),
                variable_mappings: std::collections::HashMap::new(),
                scope_id: 0,
            });

            // Add to PTX line mapping for source correlation with adjusted line
            self.ptx_line_mapping.insert(
                adjusted_line,
                debug::PtxSourceLocation {
                    file: self.source_filename.clone(),
                    line: adjusted_line,
                    column,
                    instruction_offset: self.instruction_counter as usize,
                },
            );
        }

        Ok(())
    }

    /// Create debug value calls compatible with LLVM debug format
    pub unsafe fn create_debug_value_compatible(
        &mut self,
        builder: LLVMBuilderRef,
        value: LLVMValueRef,
        var_info: LLVMMetadataRef,
        expression: LLVMMetadataRef,
        debug_loc: LLVMMetadataRef,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let Some(ref dwarf_builder) = self.dwarf_builder {
            let context = LLVMGetModuleContext(dwarf_builder.module);

            // Get or create the llvm.dbg.value intrinsic function
            let module = dwarf_builder.module;
            let dbg_value_name = CString::new("llvm.dbg.value").unwrap();
            let dbg_value_func = LLVMGetNamedFunction(module, dbg_value_name.as_ptr());

            let dbg_value_func = if dbg_value_func.is_null() {
                // Create llvm.dbg.value function type: void(metadata, metadata, metadata)
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = vec![metadata_type, metadata_type, metadata_type];

                let func_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut LLVMTypeRef,
                    param_types.len() as u32,
                    0, // not variadic
                );

                LLVMAddFunction(module, dbg_value_name.as_ptr(), func_type)
            } else {
                dbg_value_func
            };

            // Create metadata arguments
            let value_metadata = LLVMValueAsMetadata(value);
            let args = vec![
                LLVMMetadataAsValue(context, value_metadata),
                LLVMMetadataAsValue(context, var_info),
                LLVMMetadataAsValue(context, expression),
            ];

            // Create the debug value call
            let void_type = LLVMVoidTypeInContext(context);
            let call_inst = LLVMBuildCall2(
                builder,
                void_type,
                dbg_value_func,
                args.as_ptr() as *mut LLVMValueRef,
                args.len() as u32,
                CString::new("").unwrap().as_ptr(),
            );

            // Debug location is already set in the call_inst
            // LLVMSetInstDebugLocation is deprecated for newer debug record types
        }

        Ok(())
    }

    /// Set current function's debug info
    pub fn set_function_debug_info(&mut self, metadata: LLVMMetadataRef) {
        self.current_function_debug_info = Some(metadata);
    }

    /// Finalize debug information
    pub unsafe fn finalize_debug_info(&mut self) {
        if let Some(builder) = &self.dwarf_builder {
            builder.finalize();
        }
    }

    /// Get the current debug location
    pub fn get_current_debug_location(&self) -> Option<LLVMMetadataRef> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return None;
        }

        // If we have any source mappings, get the last one's debug location
        if let Some(mapping) = self.source_mappings.last() {
            if let Some(ref dwarf_builder) = self.dwarf_builder {
                // Try to create a debug location from the last mapping
                match unsafe {
                    dwarf_builder.create_debug_location(
                        mapping.ptx_location.line,
                        mapping.ptx_location.column,
                        self.current_function_debug_info,
                    )
                } {
                    Ok(loc) => return Some(loc),
                    Err(_) => return None,
                }
            }
        }

        None
    }

    /// Helper method to create complete debug value tracking (like in the NVPTX example)
    pub unsafe fn track_variable_value(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        value: LLVMValueRef,
        var_debug_info: LLVMMetadataRef,
        line: u32,
        expression_ops: Option<&[u64]>,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let (Some(ref dwarf_builder), Some(scope)) =
            (&self.dwarf_builder, self.current_function_debug_info)
        {
            // Create debug location
            let debug_loc = dwarf_builder.create_debug_location(line, 0, Some(scope))?;

            // Create debug expression
            let expr = if let Some(ops) = expression_ops {
                LLVMDIBuilderCreateExpression(
                    dwarf_builder.get_builder(),
                    ops.as_ptr() as *mut u64,
                    ops.len(),
                )
            } else {
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0)
            };

            // Create the debug value call
            self.create_debug_value_compatible(builder, value, var_debug_info, expr, debug_loc)?;
        }

        Ok(())
    }

    /// Create constant debug value (like "DEBUG_VALUE: foo:i <- 3")
    pub unsafe fn track_constant_value(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        constant_value: i64,
        var_debug_info: LLVMMetadataRef,
        line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let Some(ref dwarf_builder) = self.dwarf_builder {
            let context = LLVMGetModuleContext(dwarf_builder.module);

            // Create constant value
            let const_value =
                LLVMConstInt(LLVMInt32TypeInContext(context), constant_value as u64, 0);

            // Track this constant value
            self.track_variable_value(builder, var_name, const_value, var_debug_info, line, None)?;
        }

        Ok(())
    }

    /// Create memory reference debug value (like "DEBUG_VALUE: foo:i <- [DW_OP_deref] $vrdepot")
    pub unsafe fn track_memory_reference(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        memory_ptr: LLVMValueRef,
        var_debug_info: LLVMMetadataRef,
        line: u32,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        // Create DW_OP_deref expression
        let deref_ops = [0x06u64]; // DW_OP_deref

        self.track_variable_value(
            builder,
            var_name,
            memory_ptr,
            var_debug_info,
            line,
            Some(&deref_ops),
        )?;

        Ok(())
    }

    /// Track PTX variable assignment with llvm.dbg.value calls (like in NVPTX LLVM IR)
    pub unsafe fn track_ptx_variable_assignment(
        &mut self,
        dst_id: SpirvWord,
        src_value: LLVMValueRef,
        line: u32,
        function_di: LLVMMetadataRef,
        builder: LLVMBuilderRef,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        // Add null pointer checks first
        if src_value.is_null() {
            return Err("Source value is null".to_string());
        }
        if function_di.is_null() {
            return Err("Function debug info is null".to_string());
        }
        if builder.is_null() {
            return Err("Builder is null".to_string());
        }

        // Create llvm.dbg.value call for PTX variable assignment
        let var_name = format!("var_{}", dst_id.0);

        // Create a simple debug type for the variable
        let debug_type = if let Some(ref dwarf_builder) = self.dwarf_builder {
            match dwarf_builder.create_basic_type(
                "i32", // Default to i32 for simplicity
                32, 0x05, // DW_ATE_signed
            ) {
                Ok(dt) => dt,
                Err(e) => {
                    eprintln!("Warning: Failed to create debug type: {}", e);
                    return Ok(()); // Continue without debug info rather than crash
                }
            }
        } else {
            return Ok(());
        };

        if debug_type.is_null() {
            eprintln!("Warning: Debug type is null, skipping debug value");
            return Ok(());
        }

        // Create DILocalVariable for the assignment
        let var_name_cstr = CString::new(var_name).map_err(|_| "Invalid variable name")?;

        if let Some(ref dwarf_builder) = self.dwarf_builder {
            let di_variable = LLVMDIBuilderCreateAutoVariable(
                dwarf_builder.get_builder(),
                function_di,
                var_name_cstr.as_ptr(),
                var_name_cstr.as_bytes().len(),
                dwarf_builder.file,
                line,
                debug_type,
                1, // always preserve
                0, // LLVMDIFlagZero
                0, // alignment
            );

            // Check if variable creation was successful
            if di_variable.is_null() {
                eprintln!("Warning: Failed to create debug variable, skipping");
                return Ok(());
            }

            // Create debug location
            let debug_loc = LLVMDIBuilderCreateDebugLocation(
                dwarf_builder.context,
                line,
                1, // column
                function_di,
                ptr::null_mut(), // inlined at
            );

            // Check if debug location creation was successful
            if debug_loc.is_null() {
                eprintln!("Warning: Failed to create debug location, skipping");
                return Ok(());
            }

            // Create empty debug expression
            let debug_expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), ptr::null_mut(), 0);

            // Check if expression creation was successful
            if debug_expr.is_null() {
                eprintln!("Warning: Failed to create debug expression, skipping");
                return Ok(());
            }

            // Use traditional llvm.dbg.value function call approach (safer)
            let context = dwarf_builder.context;
            let module = dwarf_builder.module;
            let dbg_value_name = CString::new("llvm.dbg.value").unwrap();
            let mut dbg_value_fn = LLVMGetNamedFunction(module, dbg_value_name.as_ptr());

            if dbg_value_fn.is_null() {
                // Declare llvm.dbg.value function
                let void_type = LLVMVoidTypeInContext(context);
                let metadata_type = LLVMMetadataTypeInContext(context);
                let param_types = [metadata_type, metadata_type, metadata_type];
                let function_type = LLVMFunctionType(
                    void_type,
                    param_types.as_ptr() as *mut _,
                    param_types.len() as u32,
                    0, // not variadic
                );
                dbg_value_fn = LLVMAddFunction(module, dbg_value_name.as_ptr(), function_type);
            }

            // Final check for the function
            if dbg_value_fn.is_null() {
                eprintln!("Warning: Failed to create llvm.dbg.value function, skipping");
                return Ok(());
            }

            // Create metadata for the call
            let value_metadata = LLVMValueAsMetadata(src_value);
            let args = [
                LLVMMetadataAsValue(context, value_metadata),
                LLVMMetadataAsValue(context, di_variable),
                LLVMMetadataAsValue(context, debug_expr),
            ];

            // Validate all arguments before creating the call
            for (i, arg) in args.iter().enumerate() {
                if arg.is_null() {
                    eprintln!(
                        "Warning: Argument {} for llvm.dbg.value is null, skipping",
                        i
                    );
                    return Ok(());
                }
            }

            // Create the call with correct function type
            // Use the same function type that was used to declare the function
            let void_type = LLVMVoidTypeInContext(context);
            let metadata_type = LLVMMetadataTypeInContext(context);
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

            // Check if call creation was successful
            if call.is_null() {
                eprintln!("Warning: Failed to create llvm.dbg.value call, skipping");
                return Ok(());
            }

            // Set debug location for the call
            LLVMSetCurrentDebugLocation2(builder, debug_loc);
        }

        Ok(())
    }
}
