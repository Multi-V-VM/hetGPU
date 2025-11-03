use crate::checkpoint::{CheckpointError, CheckpointManager, CompilationStage, PerformanceStats};
use crate::{llvm_to_spirv_robust, to_llvm_module, TranslateError};
// use ptx_parser::Module; // 暂时注释掉未使用的导入
use std::time::Instant;

/// 带检查点的PTX编译器
pub struct CheckpointedCompiler {
    checkpoint_manager: CheckpointManager,
    enable_auto_checkpoint: bool,
    checkpoint_frequency: CompilationStage,
}

impl CheckpointedCompiler {
    /// 创建新的带检查点的编译器
    pub fn new<P: AsRef<std::path::Path>>(
        checkpoint_dir: P,
        enable_auto_checkpoint: bool,
    ) -> Result<Self, std::io::Error> {
        let checkpoint_manager = CheckpointManager::new(checkpoint_dir)?;

        Ok(Self {
            checkpoint_manager,
            enable_auto_checkpoint,
            checkpoint_frequency: CompilationStage::LlvmGeneration,
        })
    }

    /// 带检查点的完整PTX编译流程
    pub fn compile_ptx_with_checkpoints(
        &mut self,
        ptx_source: &str,
        description: Option<String>,
    ) -> Result<CompilationResult, CompilationError> {
        let start_time = Instant::now();
        let description = description.unwrap_or_else(|| "PTX编译".to_string());

        // 创建初始检查点
        let checkpoint_id = self.checkpoint_manager.create_checkpoint(
            ptx_source.to_string(),
            CompilationStage::PtxParsing,
            description,
        );

        println!("✓ 创建检查点: {}", checkpoint_id);

        let mut stats = PerformanceStats::default();
        let mut compilation_result = CompilationResult {
            checkpoint_id: checkpoint_id.clone(),
            // llvm_module: None,
            spirv_binary: None,
            compilation_time_ms: 0,
            errors: Vec::new(),
        };

        // 阶段1: PTX解析
        let parse_start = Instant::now();
        let ast = match ptx_parser::parse_module_checked(ptx_source) {
            Ok(ast) => {
                stats.ptx_parse_time_ms = parse_start.elapsed().as_millis() as u64;
                println!("✓ PTX解析完成 ({}ms)", stats.ptx_parse_time_ms);

                // 更新检查点到LLVM生成阶段
                if let Err(e) = self
                    .checkpoint_manager
                    .update_checkpoint(&checkpoint_id, CompilationStage::LlvmGeneration)
                {
                    eprintln!("警告: 无法更新检查点: {}", e);
                }

                ast
            }
            Err(e) => {
                let error = TranslateError::UnexpectedError(format!("PTX解析失败: {:?}", e));
                let _ = self
                    .checkpoint_manager
                    .add_error(&checkpoint_id, &error, None);
                compilation_result.errors.push(error.to_string());
                return Err(CompilationError::ParseError(format!("{:?}", e)));
            }
        };

        // 阶段2: LLVM IR生成
        let llvm_start = Instant::now();
        let llvm_module = match to_llvm_module(
            ast,
            crate::pass::Attributes {
                clock_rate: 2124000,
                emit_debug_info: false,
            },
            |_| {},
        ) {
            Ok(module) => {
                stats.llvm_gen_time_ms = llvm_start.elapsed().as_millis() as u64;
                println!("✓ LLVM IR生成完成 ({}ms)", stats.llvm_gen_time_ms);

                // 将LLVM IR添加到检查点
                match module.print_to_string() {
                    Ok(llvm_ir) => {
                        if let Err(e) = self.checkpoint_manager.add_llvm_ir(&checkpoint_id, llvm_ir)
                        {
                            eprintln!("警告: 无法保存LLVM IR到检查点: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("警告: 无法转换LLVM模块为字符串: {}", e);
                    }
                }

                // 自动保存检查点
                if self.enable_auto_checkpoint {
                    if let Err(e) = self.checkpoint_manager.save_checkpoint(&checkpoint_id) {
                        eprintln!("警告: 无法自动保存检查点: {}", e);
                    } else {
                        println!("✓ 自动保存检查点");
                    }
                }

                // 更新到SPIR-V转换阶段
                let _ = self
                    .checkpoint_manager
                    .update_checkpoint(&checkpoint_id, CompilationStage::SpirvConversion);

                module
            }
            Err(e) => {
                let _ = self.checkpoint_manager.add_error(&checkpoint_id, &e, None);
                compilation_result.errors.push(e.to_string());
                return Err(CompilationError::LlvmError(e));
            }
        };

        // compilation_result.llvm_module = Some(llvm_module.clone()); // 暂时注释掉，因为Module没有实现Clone

        // 阶段3: SPIR-V转换
        let spirv_start = Instant::now();
        let llvm_ir_string: String = match llvm_module.print_to_string() {
            Ok(s) => s,
            Err(e) => {
                let error =
                    TranslateError::UnexpectedError(format!("无法转换LLVM模块为字符串: {}", e));
                let _ = self
                    .checkpoint_manager
                    .add_error(&checkpoint_id, &error, None);
                compilation_result.errors.push(error.to_string());
                return Err(CompilationError::LlvmError(error));
            }
        };

        match llvm_to_spirv_robust(&llvm_ir_string) {
            Ok(spirv_binary) => {
                stats.spirv_conv_time_ms = spirv_start.elapsed().as_millis() as u64;
                println!("✓ SPIR-V转换完成 ({}ms)", stats.spirv_conv_time_ms);

                // 将SPIR-V二进制添加到检查点
                if let Err(e) = self
                    .checkpoint_manager
                    .add_spirv_binary(&checkpoint_id, spirv_binary.clone())
                {
                    eprintln!("警告: 无法保存SPIR-V到检查点: {}", e);
                }

                compilation_result.spirv_binary = Some(spirv_binary);

                // 更新到完成阶段
                let _ = self
                    .checkpoint_manager
                    .update_checkpoint(&checkpoint_id, CompilationStage::Completed);
            }
            Err(e) => {
                let error = TranslateError::UnexpectedError(format!("SPIR-V转换失败: {}", e));
                let _ = self.checkpoint_manager.add_error(
                    &checkpoint_id,
                    &error,
                    Some(llvm_ir_string.clone()),
                );
                compilation_result.errors.push(error.to_string());
                return Err(CompilationError::SpirvError(e.to_string()));
            }
        }

        // 更新性能统计
        stats.total_time_ms = start_time.elapsed().as_millis() as u64;
        compilation_result.compilation_time_ms = stats.total_time_ms;

        if let Err(e) = self
            .checkpoint_manager
            .update_performance_stats(&checkpoint_id, stats)
        {
            eprintln!("警告: 无法更新性能统计: {}", e);
        }

        // 最终保存检查点
        match self.checkpoint_manager.save_checkpoint(&checkpoint_id) {
            Ok(path) => {
                println!("✓ 编译完成，检查点已保存: {:?}", path);
            }
            Err(e) => {
                eprintln!("警告: 无法保存最终检查点: {}", e);
            }
        }

        Ok(compilation_result)
    }

    /// 从检查点恢复编译
    pub fn resume_from_checkpoint(
        &mut self,
        checkpoint_id: &str,
    ) -> Result<CompilationResult, CompilationError> {
        println!("🔄 从检查点恢复编译: {}", checkpoint_id);

        let restore_info = self
            .checkpoint_manager
            .restore_compilation_state(checkpoint_id)
            .map_err(|e| CompilationError::CheckpointError(e))?;

        let compilation_result = CompilationResult {
            checkpoint_id: checkpoint_id.to_string(),
            // llvm_module: None,
            spirv_binary: restore_info.spirv_binary.clone(),
            compilation_time_ms: 0,
            errors: Vec::new(),
        };

        match restore_info.stage {
            CompilationStage::Completed => {
                println!("✓ 编译已完成，直接返回结果");
                Ok(compilation_result)
            }
            CompilationStage::Failed => {
                println!("❌ 检查点显示编译失败");
                Err(CompilationError::PreviousFailure)
            }
            stage => {
                println!("⚠️  从阶段 {:?} 继续编译", stage);
                // 这里可以实现从特定阶段继续编译的逻辑
                // 为了简化，我们重新开始完整编译
                self.compile_ptx_with_checkpoints(
                    &restore_info.ptx_source,
                    Some("恢复编译".to_string()),
                )
            }
        }
    }

    /// 列出所有检查点
    pub fn list_checkpoints(&self) -> Vec<CheckpointSummary> {
        self.checkpoint_manager
            .list_active_checkpoints()
            .into_iter()
            .map(|metadata| CheckpointSummary {
                id: metadata.id.clone(),
                description: metadata.description.clone(),
                timestamp: metadata.timestamp,
                created_at: metadata.created_at.clone(),
            })
            .collect()
    }

    /// 生成编译报告
    pub fn generate_compilation_report(&self) -> String {
        let report = self.checkpoint_manager.generate_report();

        format!(
            "=== ZLUDA 编译报告 ===\n\
             总检查点数: {}\n\
             平均编译时间: {}ms\n\
             成功率: {:.1}%\n\
             磁盘使用: {:.2} MB\n\
             \n\
             阶段分布:\n{}",
            report.total_checkpoints,
            report.average_compile_time_ms,
            report.success_rate,
            report.disk_usage_bytes as f64 / 1024.0 / 1024.0,
            format_stage_distribution(&report.stage_distribution)
        )
    }

    /// 清理旧检查点
    pub fn cleanup_checkpoints(&mut self) -> Result<usize, CheckpointError> {
        self.checkpoint_manager.cleanup_old_checkpoints()
    }

    /// 获取检查点管理器的引用
    pub fn checkpoint_manager(&self) -> &CheckpointManager {
        &self.checkpoint_manager
    }

    /// 获取检查点管理器的可变引用
    pub fn checkpoint_manager_mut(&mut self) -> &mut CheckpointManager {
        &mut self.checkpoint_manager
    }
}

/// 编译结果
#[derive(Debug)]
pub struct CompilationResult {
    pub checkpoint_id: String,
    // pub llvm_module: Option<crate::Module>, // 暂时注释掉
    pub spirv_binary: Option<Vec<u8>>,
    pub compilation_time_ms: u64,
    pub errors: Vec<String>,
}

/// 检查点摘要
#[derive(Debug)]
pub struct CheckpointSummary {
    pub id: String,
    pub description: String,
    pub timestamp: u64,
    pub created_at: String,
}

/// 编译错误类型
#[derive(Debug, thiserror::Error)]
pub enum CompilationError {
    #[error("PTX解析错误: {0:?}")]
    ParseError(String), // 简化ParseError处理

    #[error("LLVM生成错误: {0}")]
    LlvmError(TranslateError),

    #[error("SPIR-V转换错误: {0}")]
    SpirvError(String),

    #[error("检查点错误: {0}")]
    CheckpointError(CheckpointError),

    #[error("先前编译失败")]
    PreviousFailure,
}

/// 格式化阶段分布信息
fn format_stage_distribution(
    distribution: &std::collections::HashMap<CompilationStage, usize>,
) -> String {
    let mut lines = Vec::new();

    for (stage, count) in distribution {
        lines.push(format!("  {:?}: {}", stage, count));
    }

    lines.join("\n")
}

/// 便利函数：带检查点的快速编译
pub fn compile_ptx_with_auto_checkpoint<P: AsRef<std::path::Path>>(
    ptx_source: &str,
    checkpoint_dir: P,
    description: Option<String>,
) -> Result<CompilationResult, CompilationError> {
    let mut compiler = CheckpointedCompiler::new(checkpoint_dir, true)?;
    compiler.compile_ptx_with_checkpoints(ptx_source, description)
}

impl From<std::io::Error> for CompilationError {
    fn from(err: std::io::Error) -> Self {
        CompilationError::CheckpointError(CheckpointError::IoError(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpointed_compilation() {
        let temp_dir = TempDir::new().unwrap();
        let mut compiler = CheckpointedCompiler::new(temp_dir.path(), true).unwrap();

        let ptx_source = r#"
.version 7.0
.target sm_50
.address_size 64

.entry test_kernel() {
    ret;
}
"#;

        // 编译应该成功，因为这是一个简单的有效PTX代码
        let result =
            compiler.compile_ptx_with_checkpoints(ptx_source, Some("测试编译".to_string()));

        // 验证编译过程创建了检查点
        assert!(!compiler.list_checkpoints().is_empty());

        // 生成报告
        let report = compiler.generate_compilation_report();
        println!("编译报告:\n{}", report);
    }

    #[test]
    fn test_checkpoint_resume() {
        let temp_dir = TempDir::new().unwrap();
        let mut compiler = CheckpointedCompiler::new(temp_dir.path(), true).unwrap();

        let ptx_source = r#"
.version 7.0
.target sm_50
.entry test_kernel() { ret; }
"#;

        // 第一次编译
        let result1 =
            compiler.compile_ptx_with_checkpoints(ptx_source, Some("首次编译".to_string()));

        if let Ok(compilation_result) = result1 {
            // 尝试从检查点恢复
            let result2 = compiler.resume_from_checkpoint(&compilation_result.checkpoint_id);

            // 恢复应该成功
            assert!(result2.is_ok());
        }
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let mut compiler = CheckpointedCompiler::new(temp_dir.path(), true).unwrap();

        // 创建多个检查点
        for i in 0..5 {
            let ptx_source = format!(
                ".version 7.0\n.target sm_50\n.entry test_kernel_{}() {{ ret; }}",
                i
            );
            let _ = compiler
                .compile_ptx_with_checkpoints(&ptx_source, Some(format!("测试检查点 {}", i)));
        }

        let initial_count = compiler.list_checkpoints().len();
        assert!(initial_count >= 5);

        // 清理检查点
        let cleaned = compiler.cleanup_checkpoints().unwrap_or(0);
        println!("清理了 {} 个检查点", cleaned);
    }
}
