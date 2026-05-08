use super::*;
use ptx_parser as ast;

pub(crate) fn run<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    directives: Vec<ast::Directive<'input, ast::ParsedOperand<&'input str>>>,
) -> Result<Vec<NormalizedDirective2>, TranslateError> {
    resolver.start_scope();
    let result = directives
        .into_iter()
        .map(|directive| run_directive(resolver, directive))
        .collect::<Result<Vec<_>, _>>()?;
    resolver.end_scope();
    Ok(result)
}

fn run_directive<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    directive: ast::Directive<'input, ast::ParsedOperand<&'input str>>,
) -> Result<NormalizedDirective2, TranslateError> {
    Ok(match directive {
        ast::Directive::Variable(linking, var) => {
            NormalizedDirective2::Variable(linking, run_variable(resolver, var)?)
        }
        ast::Directive::Method(linking, directive) => {
            NormalizedDirective2::Method(run_method(resolver, linking, directive)?)
        }
    })
}

fn run_method<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    linkage: ast::LinkingDirective,
    method: ast::Function<'input, &'input str, ast::Statement<ast::ParsedOperand<&'input str>>>,
) -> Result<NormalizedFunction2, TranslateError> {
    let name = match method.func_directive.name {
        ast::MethodName::Kernel(name) => ast::MethodName::Kernel(name),
        ast::MethodName::Func(text) => {
            ast::MethodName::Func(resolver.add_or_get_in_current_scope_untyped(text)?)
        }
    };
    resolver.start_scope();
    let func_decl = run_function_decl(resolver, method.func_directive, name)?;
    let body = method
        .body
        .map(|statements| {
            let mut result = Vec::with_capacity(statements.len());
            run_statements(resolver, &mut result, statements)?;
            Ok::<_, TranslateError>(result)
        })
        .transpose()?;
    resolver.end_scope();
    let is_kernel = matches!(func_decl.name, ast::MethodName::Kernel(_));
    let name = match func_decl.name {
        ast::MethodName::Kernel(n) => resolver.add_or_get_in_current_scope_untyped(n)?,
        ast::MethodName::Func(n) => n,
    };
    Ok(Function2 {
        return_arguments: func_decl.return_arguments,
        name,
        input_arguments: func_decl.input_arguments,
        body,
        is_kernel,
        import_as: None,
        tuning: method.tuning,
        linkage,
        flush_to_zero_f32: false,
        flush_to_zero_f16f64: false,
        rounding_mode_f32: ast::RoundingMode::NearestEven,
        rounding_mode_f16f64: ast::RoundingMode::NearestEven,
    })
}

fn run_function_decl<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    func_directive: ast::MethodDeclaration<'input, &'input str>,
    name: ast::MethodName<'input, SpirvWord>,
) -> Result<ast::MethodDeclaration<'input, SpirvWord>, TranslateError> {
    assert!(func_directive.shared_mem.is_none());
    let return_arguments = func_directive
        .return_arguments
        .into_iter()
        .map(|var| run_variable(resolver, var))
        .collect::<Result<Vec<_>, _>>()?;
    let input_arguments = func_directive
        .input_arguments
        .into_iter()
        .map(|var| run_variable(resolver, var))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ast::MethodDeclaration {
        return_arguments,
        name,
        input_arguments,
        shared_mem: None,
    })
}

fn run_variable<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    variable: ast::Variable<&'input str>,
) -> Result<ast::Variable<SpirvWord>, TranslateError> {
    Ok(ast::Variable {
        name: resolver.add(
            Cow::Borrowed(variable.name),
            Some((variable.info.v_type.clone(), variable.info.state_space)),
        )?,
        info: ast::VariableInfo {
            align: variable.info.align,
            v_type: variable.info.v_type,
            state_space: variable.info.state_space,
            array_init: variable
                .info
                .array_init
                .into_iter()
                .map(|reg_or_imm| match reg_or_imm {
                    ast::RegOrImmediate::Reg(name) => resolver
                        .add_or_get_in_current_scope_untyped(name)
                        .map(ast::RegOrImmediate::Reg),
                    ast::RegOrImmediate::Imm(imm) => Ok(ast::RegOrImmediate::Imm(imm)),
                    ast::RegOrImmediate::Discard => Ok(ast::RegOrImmediate::Discard),
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
    })
}

fn run_statements<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    result: &mut Vec<NormalizedStatement>,
    statements: Vec<ast::Statement<ast::ParsedOperand<&'input str>>>,
) -> Result<(), TranslateError> {
    for statement in statements.iter() {
        match statement {
            ast::Statement::Label(label) => {
                resolver.add(Cow::Borrowed(*label), None)?;
            }
            _ => {}
        }
    }
    for statement in statements {
        match statement {
            ast::Statement::Label(label) => {
                result.push(Statement::Label(resolver.get_in_current_scope(label)?))
            }
            ast::Statement::Variable(variable) => run_multivariable(resolver, result, variable)?,
            ast::Statement::Instruction(predicate, instruction) => {
                result.push(Statement::Instruction((
                    predicate
                        .map(|pred| {
                            Ok::<_, TranslateError>(ast::PredAt {
                                not: pred.not,
                                label: resolver.get(pred.label)?,
                            })
                        })
                        .transpose()?,
                    run_instruction(resolver, instruction)?,
                )))
            }
            ast::Statement::Block(block) => {
                resolver.start_scope();
                run_statements(resolver, result, block)?;
                resolver.end_scope();
            }
        }
    }
    Ok(())
}

fn run_instruction<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    instruction: ast::Instruction<ast::ParsedOperand<&'input str>>,
) -> Result<ast::Instruction<ast::ParsedOperand<SpirvWord>>, TranslateError> {
    ast::visit_map(instruction, &mut |name: &'input str,
                                      _: Option<(
        &ast::Type,
        ast::StateSpace,
    )>,
                                      _,
                                      _| {
        resolver.get(&name)
    })
}

fn convert_array_init<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    array_init: &[ast::RegOrImmediate<&'input str>],
) -> Result<Vec<ast::RegOrImmediate<SpirvWord>>, TranslateError> {
    array_init
        .iter()
        .map(|reg_or_imm| match reg_or_imm {
            ast::RegOrImmediate::Reg(name) => resolver
                .add_or_get_in_current_scope_untyped(name)
                .map(ast::RegOrImmediate::Reg),
            ast::RegOrImmediate::Imm(imm) => Ok(ast::RegOrImmediate::Imm(*imm)),
            ast::RegOrImmediate::Discard => Ok(ast::RegOrImmediate::Discard),
        })
        .collect::<Result<Vec<_>, _>>()
}

fn run_multivariable<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    result: &mut Vec<NormalizedStatement>,
    variable: ast::MultiVariable<&'input str>,
) -> Result<(), TranslateError> {
    match variable {
        ast::MultiVariable::Parameterized { info, name, count } => {
            let converted_array_init = convert_array_init(resolver, &info.array_init)?;
            for i in 0..count {
                let var_name = Cow::Owned(format!("{}{}", name, i));
                let ident =
                    resolver.add(var_name, Some((info.v_type.clone(), info.state_space)))?;
                result.push(Statement::Variable(ast::Variable {
                    name: ident,
                    info: ast::VariableInfo {
                        align: info.align,
                        v_type: info.v_type.clone(),
                        state_space: info.state_space,
                        array_init: converted_array_init.clone(),
                    },
                }));
            }
        }
        ast::MultiVariable::Names { info, names } => {
            let converted_array_init = convert_array_init(resolver, &info.array_init)?;
            for name in names {
                let var_name = Cow::Borrowed(name);
                let ident =
                    resolver.add(var_name, Some((info.v_type.clone(), info.state_space)))?;
                result.push(Statement::Variable(ast::Variable {
                    name: ident,
                    info: ast::VariableInfo {
                        align: info.align,
                        v_type: info.v_type.clone(),
                        state_space: info.state_space,
                        array_init: converted_array_init.clone(),
                    },
                }));
            }
        }
    }
    Ok(())
}
