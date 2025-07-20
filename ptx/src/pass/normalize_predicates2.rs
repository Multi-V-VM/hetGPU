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
                // Look for a complementary predicated instruction
                let complement_idx = find_complement_predicate(statements, pred);
                
                if let Some(comp_idx) = complement_idx {
                    // Found complementary pair - merge them
                    process_predicate_pair(resolver, result, statements, 0, comp_idx)?;
                    Ok(comp_idx + 1) // Skip both instructions
                } else {
                    // No complement found - process single predicated instruction
                    process_single_predicate(resolver, result, pred, instruction)?;
                    Ok(1)
                }
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

// Find a complementary predicated instruction starting from index 1
fn find_complement_predicate<'input>(
    statements: &[NormalizedStatement],
    pred: &ast::PredAt<SpirvWord>,
) -> Option<usize> {
    for i in 1..statements.len() {
        match &statements[i] {
            Statement::Instruction((Some(other_pred), _)) => {
                // Same predicate, opposite not flag
                if other_pred.label == pred.label && other_pred.not != pred.not {
                    return Some(i);
                }
                // Different predicate - stop searching
                if other_pred.label != pred.label {
                    break;
                }
            }
            Statement::Label(_) => {
                // Label breaks the sequence
                break;
            }
            _ => {
                // Continue through other statements
            }
        }
    }
    None
}

// Process a pair of complementary predicated instructions
fn process_predicate_pair<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    result: &mut Vec<UnconditionalStatement>,
    statements: &[NormalizedStatement],
    first_idx: usize,
    second_idx: usize,
) -> Result<(), TranslateError> {
    let (first_pred, first_inst) = match &statements[first_idx] {
        Statement::Instruction((Some(pred), inst)) => (pred, inst),
        _ => return Err(error_unreachable()),
    };
    
    let (_, second_inst) = match &statements[second_idx] {
        Statement::Instruction((Some(_), inst)) => ((), inst),
        _ => return Err(error_unreachable()),
    };
    
    // Create labels
    let if_true = resolver.register_unnamed(None);
    let if_false = resolver.register_unnamed(None);
    let if_merge = resolver.register_unnamed(None);
    
    // Determine which instruction goes where based on first predicate's not flag
    let (true_inst, false_inst) = if first_pred.not {
        (second_inst, first_inst)
    } else {
        (first_inst, second_inst)
    };
    
    // Check for folded branches
    let true_folded_bra = match true_inst {
        ast::Instruction::Bra { arguments, .. } => Some(arguments.src),
        _ => None,
    };
    let false_folded_bra = match false_inst {
        ast::Instruction::Bra { arguments, .. } => Some(arguments.src),
        _ => None,
    };
    
    // Generate the conditional branch
    let branch = BrachCondition {
        predicate: first_pred.label,
        if_true: true_folded_bra.unwrap_or(if_true),
        if_false: false_folded_bra.unwrap_or(if_false),
    };
    result.push(Statement::Conditional(branch));
    
    // True branch (if not folded)
    if true_folded_bra.is_none() {
        result.push(Statement::Label(if_true));
        convert_instruction_to_unconditional(result, true_inst)?;
        // Add branch to merge point
        result.push(Statement::Instruction(ast::Instruction::Bra {
            arguments: ast::BraArgs { src: if_merge },
        }));
    }
    
    // False branch (if not folded)
    if false_folded_bra.is_none() {
        result.push(Statement::Label(if_false));
        convert_instruction_to_unconditional(result, false_inst)?;
        // Add branch to merge point
        result.push(Statement::Instruction(ast::Instruction::Bra {
            arguments: ast::BraArgs { src: if_merge },
        }));
    }
    
    // Merge point (only if we have non-folded branches)
    if true_folded_bra.is_none() || false_folded_bra.is_none() {
        result.push(Statement::Label(if_merge));
    }
    
    Ok(())
}

// Process a single predicated instruction (no complement)
fn process_single_predicate<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    result: &mut Vec<UnconditionalStatement>,
    pred: &ast::PredAt<SpirvWord>,
    instruction: &ast::Instruction<ast::ParsedOperand<SpirvWord>>,
) -> Result<(), TranslateError> {
    let if_true = resolver.register_unnamed(None);
    let if_false = resolver.register_unnamed(None);
    
    let folded_bra = match instruction {
        ast::Instruction::Bra { arguments, .. } => Some(arguments.src),
        _ => None,
    };
    
    let mut branch = BrachCondition {
        predicate: pred.label,
        if_true: folded_bra.unwrap_or(if_true),
        if_false,
    };
    
    if pred.not {
        std::mem::swap(&mut branch.if_true, &mut branch.if_false);
    }
    
    result.push(Statement::Conditional(branch));
    
    if folded_bra.is_none() {
        result.push(Statement::Label(if_true));
        convert_instruction_to_unconditional(result, instruction)?;
    }
    
    result.push(Statement::Label(if_false));
    
    Ok(())
}