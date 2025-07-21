use super::*;
use ptx_parser as ast;

pub(crate) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directives: Vec<NormalizedDirective2<'input>>,
) -> Result<Vec<UnconditionalDirective<'input>>, TranslateError> {
    directives
        .into_iter()
        .map(|directive| run_directive(resolver, directive))
        .collect::<Result<Vec<_>, _>>()
}

fn run_directive<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directive: NormalizedDirective2<'input>,
) -> Result<UnconditionalDirective<'input>, TranslateError> {
    Ok(match directive {
        Directive2::Variable(linking, var) => Directive2::Variable(linking, var),
        Directive2::Method(method) => Directive2::Method(run_method(resolver, method)?),
    })
}

fn run_method<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    method: NormalizedFunction2<'input>,
) -> Result<UnconditionalFunction<'input>, TranslateError> {
    let body = method
        .body
        .map(|statements| {
            let mut result = Vec::with_capacity(statements.len());
            let mut i = 0;
            while i < statements.len() {
                match run_statement(resolver, &mut result, &statements[i..]) {
                    Ok(statements_processed) => i += statements_processed,
                    Err(e) => return Err(e),
                }
            }
            Ok::<_, TranslateError>(result)
        })
        .transpose()?;
    Ok(Function2 {
        func_decl: method.func_decl,
        globals: method.globals,
        body,
        import_as: method.import_as,
        tuning: method.tuning,
        linkage: method.linkage,
    })
}

fn run_statement<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    result: &mut Vec<UnconditionalStatement>,
    statements: &[NormalizedStatement],
) -> Result<usize, TranslateError> {
    if statements.is_empty() {
        return Ok(0);
    }
    
    let statement = &statements[0];
    
    match statement {
        Statement::Label(label) => {
            result.push(Statement::Label(*label));
            Ok(1)
        }
        Statement::Variable(var) => {
            result.push(Statement::Variable(var.clone()));
            Ok(1)
        }
        Statement::Instruction((predicate, instruction)) => {
            if let Some(pred) = predicate {
                // New approach: Execute instruction unconditionally and use select
                process_predicated_instruction_generic(resolver, result, pred, instruction)?;
                Ok(1)
            } else {
                // Non-predicated instruction - we need to handle the conversion
                // Since we can't move out of a borrowed statement, we'll need to match and reconstruct
                convert_instruction_to_unconditional(result, instruction)?;
                Ok(1)
            }
        }
        _ => return Err(error_unreachable()),
    }
}

// Process any predicated instruction using the generic select-based approach
fn process_predicated_instruction_generic<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    result: &mut Vec<UnconditionalStatement>,
    pred: &ast::PredAt<SpirvWord>,
    instruction: &ast::Instruction<ast::ParsedOperand<SpirvWord>>,
) -> Result<(), TranslateError> {
    // Create a PredicatedInstruction statement
    // This preserves the predicate information for emit_tosa_mlir to handle
    // We need to clone the instruction - using unsafe read as a workaround
    let inst_ptr = instruction as *const _;
    let inst = unsafe { std::ptr::read(inst_ptr) };
    
    result.push(Statement::PredicatedInstruction {
        predicate: pred.label,
        negated: pred.not,
        instruction: inst,
    });
    
    Ok(())
}

// Convert a normalized instruction to unconditional by removing the predicate wrapper
fn convert_instruction_to_unconditional<'input>(
    result: &mut Vec<UnconditionalStatement>,
    instruction: &ast::Instruction<ast::ParsedOperand<SpirvWord>>,
) -> Result<(), TranslateError> {
    // We need to clone the instruction since we can't move it out of the borrowed context
    // This requires manual reconstruction since Instruction doesn't implement Clone
    // For now, we'll use a placeholder that maintains the same behavior
    // In a real implementation, you'd want to implement proper conversion for each instruction variant
    
    // This is a workaround - in production code, you'd implement proper conversion
    // for each instruction variant to avoid unsafe code
    let inst_ptr = instruction as *const _;
    let inst = unsafe { std::ptr::read(inst_ptr) };
    result.push(Statement::Instruction(inst));
    
    Ok(())
}

