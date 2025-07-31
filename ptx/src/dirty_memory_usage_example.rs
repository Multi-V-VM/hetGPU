//! 脏内存跟踪使用示例
//! 
//! 这个模块展示了如何使用动态增量检查点系统中的脏内存跟踪功能，
//! 实现只复制修改过的内存页面，大幅减少检查点存储开销。

use crate::dynamic_delta_analyzer::{
    DynamicDeltaAnalyzer, DirtyMemoryConfig, MemoryUsageReport, DirtyMemoryDelta
};
use crate::delta_checkpoint::{CompilationState, DynamicAnalysisDeltas};
use crate::TranslateError;

/// 脏内存跟踪使用示例
pub struct DirtyMemoryExample {
    analyzer: DynamicDeltaAnalyzer,
}

impl DirtyMemoryExample {
    /// 创建新的示例实例
    pub fn new() -> Self {
        // 配置脏内存跟踪
        let dirty_config = DirtyMemoryConfig {
            page_size: 4096,           // 4KB 页面大小
            enable_cow: true,          // 启用写时复制
            max_dirty_pages: 50000,    // 最大脏页数量
            hash_algorithm: crate::dynamic_delta_analyzer::HashAlgorithm::Xxhash,
            enable_compression: true,   // 启用压缩
        };

        let mut analyzer = DynamicDeltaAnalyzer::new();
        analyzer.dirty_memory_manager.config = dirty_config;

        Self { analyzer }
    }

    /// 演示基本的脏内存跟踪流程
    pub fn demonstrate_basic_dirty_tracking(&mut self) -> Result<(), TranslateError> {
        println!("=== 基本脏内存跟踪演示 ===");

        // 1. 安装内存访问钩子
        println!("1. 安装内存访问钩子...");
        self.analyzer.install_dirty_memory_hooks()?;

        // 2. 模拟内存写入操作
        println!("2. 模拟内存写入操作...");
        let data = vec![0x42u8; 1024]; // 写入 1KB 数据
        self.analyzer.on_memory_write_access(0x10000000, data.len() as u32, &data)?;
        
        let data2 = vec![0x24u8; 2048]; // 写入 2KB 数据到不同地址
        self.analyzer.on_memory_write_access(0x20000000, data2.len() as u32, &data2)?;

        // 3. 获取脏内存统计
        println!("3. 获取脏内存统计...");
        let report = self.analyzer.get_memory_usage_report();
        self.print_memory_report(&report);

        // 4. 获取只包含脏页的增量数据
        println!("4. 获取脏内存增量数据...");
        let dirty_deltas = self.analyzer.get_dirty_memory_for_checkpoint()?;
        println!("   脏页数量: {}", dirty_deltas.len());
        
        for (i, delta) in dirty_deltas.iter().enumerate().take(3) {
            println!("   脏页 {}: 地址 0x{:x}, 大小 {} bytes, 操作: {:?}", 
                     i, delta.address, delta.size, delta.operation);
        }

        Ok(())
    }

    /// 演示写时复制（Copy-on-Write）优化
    pub fn demonstrate_copy_on_write(&mut self) -> Result<(), TranslateError> {
        println!("\n=== 写时复制优化演示 ===");

        // 1. 启用写时复制
        println!("1. 启用写时复制优化...");
        self.analyzer.enable_copy_on_write()?;

        // 2. 创建共享内存区域
        println!("2. 创建共享内存区域...");
        let shared_data = vec![0x55u8; 8192]; // 8KB 共享数据
        let base_address = 0x30000000;
        
        // 模拟多个检查点共享同一内存区域
        for i in 0..3 {
            let offset = i * 0x1000; // 每个检查点偏移 4KB
            self.analyzer.on_memory_write_access(
                base_address + offset, 
                shared_data.len() as u32, 
                &shared_data
            )?;
        }

        // 3. 模拟只修改其中一个页面
        println!("3. 修改其中一个页面...");
        let modified_data = vec![0xAAu8; 512]; // 只修改 512 字节
        self.analyzer.on_memory_write_access(
            base_address + 0x1000 + 100, 
            modified_data.len() as u32, 
            &modified_data
        )?;

        // 4. 检查COW效果
        let report = self.analyzer.get_memory_usage_report();
        println!("4. COW优化效果:");
        if report.cow_enabled {
            println!("   COW已启用，共享页面数量: {}", report.cow_page_count);
            if let Some(savings) = report.compression_savings {
                println!("   压缩节省空间: {:.2}%", savings);
            }
        }

        Ok(())
    }

    /// 演示增量恢复过程
    pub fn demonstrate_incremental_recovery(&mut self) -> Result<(), TranslateError> {
        println!("\n=== 增量恢复演示 ===");

        // 1. 获取当前脏内存快照
        println!("1. 获取脏内存快照...");
        let dirty_deltas = self.analyzer.get_dirty_memory_for_checkpoint()?;
        
        println!("   快照包含 {} 个脏页", dirty_deltas.len());
        let total_dirty_size: u64 = dirty_deltas.iter()
            .map(|d| d.size as u64)
            .sum();
        println!("   总脏数据大小: {:.2} KB", total_dirty_size as f64 / 1024.0);

        // 2. 模拟从快照恢复内存状态
        println!("2. 从脏内存快照恢复状态...");
        
        // 创建新的分析器实例来模拟恢复过程
        let mut recovery_analyzer = DynamicDeltaAnalyzer::new();
        
        // 恢复脏内存状态
        recovery_analyzer.restore_memory_from_dirty_deltas(&dirty_deltas)?;
        
        // 3. 验证恢复效果
        println!("3. 验证恢复效果...");
        let recovery_report = recovery_analyzer.get_memory_usage_report();
        println!("   恢复后内存使用: {:.2} MB", recovery_report.total_memory_mb);
        println!("   恢复的页面数: {}", recovery_report.page_count);

        Ok(())
    }

    /// 演示性能优化分析
    pub fn demonstrate_performance_analysis(&mut self) -> Result<(), TranslateError> {
        println!("\n=== 性能优化分析演示 ===");

        // 1. 模拟大量小写入操作（性能反模式）
        println!("1. 模拟大量小写入操作...");
        for i in 0..1000 {
            let small_data = vec![i as u8; 64]; // 64字节小写入
            let address = 0x40000000 + (i * 64) as u64;
            self.analyzer.on_memory_write_access(address, 64, &small_data)?;
        }

        // 2. 分析内存访问模式
        println!("2. 分析内存访问模式...");
        let report = self.analyzer.get_memory_usage_report();
        
        // 计算碎片化程度
        let fragmentation_ratio = report.dirty_page_count as f64 / 
            (report.dirty_memory_mb * 1024.0 * 1024.0 / 4096.0);
        
        println!("   内存碎片化比率: {:.2}", fragmentation_ratio);
        
        if fragmentation_ratio > 0.5 {
            println!("   ⚠️  检测到高内存碎片化！");
            println!("   建议: 考虑批量写入或内存池化");
        }

        // 3. 提供优化建议
        println!("3. 优化建议:");
        if report.dirty_percentage > 80.0 {
            println!("   - 脏页比例过高({:.1}%)，考虑更频繁的检查点清理", report.dirty_percentage);
        }
        
        if !report.cow_enabled {
            println!("   - 启用COW可以减少内存复制开销");
        }
        
        if report.compression_savings.is_none() {
            println!("   - 启用压缩可以减少存储空间使用");
        }

        Ok(())
    }

    /// 完整的使用示例工作流
    pub fn run_complete_example(&mut self) -> Result<(), TranslateError> {
        println!("===== 脏内存跟踪完整示例 =====\n");

        // 运行所有演示
        self.demonstrate_basic_dirty_tracking()?;
        self.demonstrate_copy_on_write()?;
        self.demonstrate_incremental_recovery()?;
        self.demonstrate_performance_analysis()?;

        // 最终报告
        println!("\n=== 最终内存使用报告 ===");
        let final_report = self.analyzer.get_memory_usage_report();
        self.print_detailed_memory_report(&final_report);

        Ok(())
    }

    // 辅助方法

    fn print_memory_report(&self, report: &MemoryUsageReport) {
        println!("   总内存: {:.2} MB", report.total_memory_mb);
        println!("   脏内存: {:.2} MB ({:.1}%)", 
                 report.dirty_memory_mb, report.dirty_percentage);
        println!("   清洁内存: {:.2} MB", report.clean_memory_mb);
        println!("   页面总数: {}", report.page_count);
        println!("   脏页数量: {}", report.dirty_page_count);
    }

    fn print_detailed_memory_report(&self, report: &MemoryUsageReport) {
        println!("内存使用详细信息:");
        println!("├── 总内存使用: {:.2} MB", report.total_memory_mb);
        println!("├── 脏内存: {:.2} MB ({:.1}%)", 
                 report.dirty_memory_mb, report.dirty_percentage);
        println!("├── 清洁内存: {:.2} MB", report.clean_memory_mb);
        println!("├── 页面统计:");
        println!("│   ├── 总页面数: {}", report.page_count);
        println!("│   └── 脏页面数: {}", report.dirty_page_count);
        println!("├── 优化特性:");
        println!("│   ├── COW启用: {}", if report.cow_enabled { "是" } else { "否" });
        if report.cow_enabled {
            println!("│   ├── COW页面数: {}", report.cow_page_count);
        }
        if let Some(savings) = report.compression_savings {
            println!("│   └── 压缩节省: {:.2}%", savings);
        } else {
            println!("│   └── 压缩: 未启用");
        }
        
        // 计算效率指标
        let efficiency = (report.clean_memory_mb / report.total_memory_mb) * 100.0;
        println!("└── 内存效率: {:.1}%", efficiency);
        
        if efficiency < 50.0 {
            println!("    ⚠️  内存效率较低，建议优化");
        } else {
            println!("    ✓  内存使用效率良好");
        }
    }
}

/// 使用示例和最佳实践
pub mod usage_patterns {
    use super::*;

    /// PTX编译器集成示例
    pub fn integrate_with_ptx_compiler() -> Result<(), TranslateError> {
        let mut analyzer = DynamicDeltaAnalyzer::new();
        
        // 1. 在编译开始时安装钩子
        analyzer.install_dirty_memory_hooks()?;
        
        // 2. 在每个编译阶段后创建检查点
        let dirty_deltas = analyzer.get_dirty_memory_for_checkpoint()?;
        
        // 3. 只存储脏页，大幅减少存储空间
        println!("检查点只需要存储 {} 个脏页", dirty_deltas.len());
        
        Ok(())
    }

    /// 最佳实践建议
    pub fn best_practices() {
        println!("脏内存跟踪最佳实践:");
        println!("1. 启用COW以减少内存复制开销");
        println!("2. 启用压缩以减少存储空间");
        println!("3. 定期清理不再需要的检查点");
        println!("4. 监控内存碎片化程度");
        println!("5. 根据工作负载调整页面大小");
        println!("6. 使用内存池化减少分配开销");
    }

    /// 性能调优指南
    pub fn performance_tuning_guide() {
        println!("性能调优指南:");
        println!("页面大小选择:");
        println!("  - 4KB: 适合细粒度跟踪，内存开销低");
        println!("  - 64KB: 适合大块操作，减少元数据开销");
        println!("  - 1MB: 适合大型数据集，最小化跟踪开销");
        
        println!("COW策略:");
        println!("  - 读多写少: 启用COW，共享干净页面");
        println!("  - 写密集: 考虑禁用COW，避免复制开销");
        
        println!("压缩选择:");
        println!("  - LZ4: 快速压缩，适合实时场景");
        println!("  - ZSTD: 高压缩比，适合存储优化");
        println!("  - 无压缩: 最快速度，适合内存充足环境");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_memory_example() {
        let mut example = DirtyMemoryExample::new();
        
        // 测试基本功能
        assert!(example.demonstrate_basic_dirty_tracking().is_ok());
        
        // 验证内存报告生成
        let report = example.analyzer.get_memory_usage_report();
        assert!(report.total_memory_mb >= 0.0);
        assert!(report.dirty_percentage >= 0.0 && report.dirty_percentage <= 100.0);
    }

    #[test]
    fn test_cow_optimization() {
        let mut example = DirtyMemoryExample::new();
        
        // 测试COW功能
        assert!(example.demonstrate_copy_on_write().is_ok());
        
        let report = example.analyzer.get_memory_usage_report();
        assert_eq!(report.cow_enabled, true);
    }

    #[test]
    fn test_incremental_recovery() {
        let mut example = DirtyMemoryExample::new();
        
        // 先写入一些数据
        let data = vec![0x42u8; 1024];
        assert!(example.analyzer.on_memory_write_access(0x50000000, 1024, &data).is_ok());
        
        // 测试恢复过程
        assert!(example.demonstrate_incremental_recovery().is_ok());
    }
}