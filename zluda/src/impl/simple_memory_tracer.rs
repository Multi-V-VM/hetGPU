//! 完整版动态内存跟踪器
//! 
//! 提供全面的内存分配跟踪、访问模式分析、内存泄漏检测等功能

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::thread;
use std::fs::File;
use std::io::Write;

/// 全局内存跟踪器实例
static GLOBAL_TRACER: std::sync::LazyLock<Arc<Mutex<CompleteMemoryTracer>>> = 
    std::sync::LazyLock::new(|| {
        Arc::new(Mutex::new(CompleteMemoryTracer::new()))
    });

/// 完整版内存跟踪器
pub struct CompleteMemoryTracer {
    allocations: HashMap<u64, MemoryAllocation>,
    access_history: VecDeque<MemoryAccess>,
    stats: MemoryStats,
    config: TrackerConfig,
    dirty_pages: HashMap<u64, DirtyPage>,
    allocation_patterns: HashMap<String, AllocationPattern>,
    leak_candidates: Vec<LeakCandidate>,
    memory_pools: HashMap<String, MemoryPool>,
    enabled: bool,
}

/// 内存分配记录
#[derive(Debug, Clone)]
struct MemoryAllocation {
    size: usize,
    timestamp: u64,
    thread_id: u64,
    stack_trace: Vec<String>,
    allocation_type: AllocationType,
    access_count: u64,
    last_access: u64,
    freed: bool,
}

/// 内存访问记录
#[derive(Debug, Clone)]
struct MemoryAccess {
    address: u64,
    size: usize,
    access_type: AccessType,
    timestamp: u64,
    thread_id: u64,
}

/// 脏页面记录
#[derive(Debug, Clone)]
struct DirtyPage {
    page_addr: u64,
    size: usize,
    modification_count: u64,
    first_modified: u64,
    last_modified: u64,
}

/// 分配模式
#[derive(Debug, Clone)]
struct AllocationPattern {
    pattern_name: String,
    size_range: (usize, usize),
    frequency: u64,
    avg_lifetime: u64,
    access_pattern: Vec<AccessType>,
}

/// 泄漏候选
#[derive(Debug, Clone)]
struct LeakCandidate {
    address: u64,
    size: usize,
    age: u64,
    access_count: u64,
    suspected_leak: bool,
}

/// 内存池
#[derive(Debug, Clone)]
struct MemoryPool {
    name: String,
    total_size: usize,
    used_size: usize,
    allocations: Vec<u64>,
    fragmentation_ratio: f64,
}

/// 分配类型
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AllocationType {
    Device,
    Host,
    Unified,
    Pinned,
    Unknown,
}

/// 访问类型
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
    Copy,
    Memset,
}

/// 跟踪器配置
#[derive(Debug, Clone)]
struct TrackerConfig {
    enable_dirty_tracking: bool,
    enable_access_tracking: bool,
    enable_pattern_analysis: bool,
    enable_leak_detection: bool,
    max_history_size: usize,
    page_size: usize,
    leak_detection_threshold: u64,
    report_interval_ms: u64,
}

/// 完整的统计信息
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_frees: u64,
    pub fragmentation_count: u64,
    pub dirty_pages_count: u64,
    pub leak_count: u64,
    pub access_violations: u64,
    pub avg_allocation_size: f64,
    pub avg_allocation_lifetime: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            enable_dirty_tracking: true,
            enable_access_tracking: true,
            enable_pattern_analysis: true,
            enable_leak_detection: true,
            max_history_size: 10000,
            page_size: 4096,
            leak_detection_threshold: 60000, // 1 minute in ms
            report_interval_ms: 30000, // 30 seconds
        }
    }
}

impl CompleteMemoryTracer {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            access_history: VecDeque::new(),
            stats: MemoryStats::default(),
            config: TrackerConfig::default(),
            dirty_pages: HashMap::new(),
            allocation_patterns: HashMap::new(),
            leak_candidates: Vec::new(),
            memory_pools: HashMap::new(),
            enabled: true,
        }
    }

    /// 跟踪分配
    pub fn track_alloc(&mut self, address: u64, size: usize) {
        self.track_alloc_with_type(address, size, AllocationType::Device);
    }

    /// 带类型的分配跟踪
    pub fn track_alloc_with_type(&mut self, address: u64, size: usize, alloc_type: AllocationType) {
        if !self.enabled {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let thread_id = self.get_thread_id();
        let stack_trace = self.capture_stack_trace();

        let allocation = MemoryAllocation {
            size,
            timestamp,
            thread_id,
            stack_trace,
            allocation_type: alloc_type.clone(),
            access_count: 0,
            last_access: timestamp,
            freed: false,
        };

        self.allocations.insert(address, allocation);

        // 更新统计
        self.stats.total_allocations += 1;
        self.stats.current_allocations += 1;
        self.stats.current_bytes += size as u64;

        if self.stats.current_allocations > self.stats.peak_allocations {
            self.stats.peak_allocations = self.stats.current_allocations;
        }

        if self.stats.current_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.current_bytes;
        }

        // 更新平均分配大小
        self.stats.avg_allocation_size = 
            self.stats.current_bytes as f64 / self.stats.current_allocations as f64;

        // 分析分配模式
        if self.config.enable_pattern_analysis {
            self.analyze_allocation_pattern(address, size, &alloc_type);
        }

        eprintln!("[CompleteTracer] Alloc {} bytes at 0x{:x} (type: {:?})", size, address, alloc_type);
    }

    /// 跟踪释放
    pub fn track_free(&mut self, address: u64) {
        if let Some(mut allocation) = self.allocations.remove(&address) {
            allocation.freed = true;
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            let lifetime = current_time - allocation.timestamp;
            
            self.stats.current_allocations -= 1;
            self.stats.current_bytes -= allocation.size as u64;
            self.stats.total_frees += 1;
            
            // 更新平均生命周期
            self.stats.avg_allocation_lifetime = 
                (self.stats.avg_allocation_lifetime * (self.stats.total_frees - 1) as f64 + lifetime as f64) 
                / self.stats.total_frees as f64;

            // 从泄漏候选中移除
            self.leak_candidates.retain(|candidate| candidate.address != address);
            
            eprintln!("[CompleteTracer] Free {} bytes at 0x{:x} (lifetime: {}ms)", 
                     allocation.size, address, lifetime / 1_000_000);
        }
    }

    /// 跟踪内存访问
    pub fn track_memory_access(&mut self, address: u64, size: usize, access_type: AccessType) {
        if !self.enabled || !self.config.enable_access_tracking {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let thread_id = self.get_thread_id();

        let access = MemoryAccess {
            address,
            size,
            access_type: access_type.clone(),
            timestamp,
            thread_id,
        };

        // 限制历史记录大小
        if self.access_history.len() >= self.config.max_history_size {
            self.access_history.pop_front();
        }
        self.access_history.push_back(access);

        // 更新分配的访问计数
        if let Some(allocation) = self.allocations.get_mut(&address) {
            allocation.access_count += 1;
            allocation.last_access = timestamp;
        }

        // 跟踪脏页面
        if access_type == AccessType::Write && self.config.enable_dirty_tracking {
            self.track_dirty_page(address, size);
        }
    }

    /// 跟踪脏页面
    fn track_dirty_page(&mut self, address: u64, size: usize) {
        let page_addr = (address / self.config.page_size as u64) * self.config.page_size as u64;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        if let Some(dirty_page) = self.dirty_pages.get_mut(&page_addr) {
            dirty_page.modification_count += 1;
            dirty_page.last_modified = timestamp;
        } else {
            let dirty_page = DirtyPage {
                page_addr,
                size: self.config.page_size,
                modification_count: 1,
                first_modified: timestamp,
                last_modified: timestamp,
            };
            self.dirty_pages.insert(page_addr, dirty_page);
            self.stats.dirty_pages_count += 1;
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.stats.total_allocations,
            self.stats.current_allocations,
            self.stats.peak_allocations,
            self.stats.current_bytes,
            self.stats.peak_bytes,
        )
    }

    /// 获取完整统计信息
    pub fn get_complete_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// 分析分配模式
    fn analyze_allocation_pattern(&mut self, address: u64, size: usize, alloc_type: &AllocationType) {
        let pattern_key = format!("{:?}_{}", alloc_type, size / 1024); // 按KB分组
        
        if let Some(pattern) = self.allocation_patterns.get_mut(&pattern_key) {
            pattern.frequency += 1;
        } else {
            let pattern = AllocationPattern {
                pattern_name: pattern_key.clone(),
                size_range: (size, size),
                frequency: 1,
                avg_lifetime: 0,
                access_pattern: Vec::new(),
            };
            self.allocation_patterns.insert(pattern_key, pattern);
        }
    }

    /// 检测内存泄漏
    pub fn detect_memory_leaks(&mut self) {
        if !self.config.enable_leak_detection {
            return;
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.leak_candidates.clear();

        for (address, allocation) in &self.allocations {
            let age = current_time - allocation.timestamp;
            let age_ms = age / 1_000_000;

            if age_ms > self.config.leak_detection_threshold {
                let candidate = LeakCandidate {
                    address: *address,
                    size: allocation.size,
                    age: age_ms,
                    access_count: allocation.access_count,
                    suspected_leak: allocation.access_count == 0 && age_ms > self.config.leak_detection_threshold * 2,
                };
                self.leak_candidates.push(candidate);
            }
        }

        self.stats.leak_count = self.leak_candidates.len() as u64;
    }

    /// 打印完整报告
    pub fn print_report(&self) {
        println!("\n=== Complete Memory Tracer Report ===");
        
        // 基本统计
        println!("📊 Basic Statistics:");
        println!("  Total allocations: {}", self.stats.total_allocations);
        println!("  Current allocations: {}", self.stats.current_allocations);
        println!("  Peak allocations: {}", self.stats.peak_allocations);
        println!("  Current memory: {:.2} MB", self.stats.current_bytes as f64 / (1024.0 * 1024.0));
        println!("  Peak memory: {:.2} MB", self.stats.peak_bytes as f64 / (1024.0 * 1024.0));
        println!("  Average allocation size: {:.2} KB", self.stats.avg_allocation_size / 1024.0);
        println!("  Average allocation lifetime: {:.2} ms", self.stats.avg_allocation_lifetime / 1_000_000.0);
        
        // 脏页面统计
        println!("\n📄 Dirty Pages:");
        println!("  Dirty pages count: {}", self.stats.dirty_pages_count);
        println!("  Dirty memory: {:.2} MB", 
                (self.stats.dirty_pages_count * self.config.page_size as u64) as f64 / (1024.0 * 1024.0));
        
        // 分配模式
        if !self.allocation_patterns.is_empty() {
            println!("\n🔍 Allocation Patterns:");
            for (name, pattern) in &self.allocation_patterns {
                println!("  {}: {} allocations", name, pattern.frequency);
            }
        }
        
        // 内存泄漏
        if self.stats.current_allocations > 0 {
            println!("\n⚠️ Memory Leaks Analysis:");
            println!("  Potential leaks: {} allocations", self.stats.current_allocations);
            println!("  Suspected leaks: {}", self.leak_candidates.iter().filter(|c| c.suspected_leak).count());
            
            // 显示前5个可疑泄漏
            let mut suspects: Vec<_> = self.leak_candidates.iter().filter(|c| c.suspected_leak).collect();
            suspects.sort_by_key(|c| c.size);
            suspects.reverse();
            
            for (i, candidate) in suspects.iter().take(5).enumerate() {
                println!("  {}. Address: 0x{:x}, Size: {} KB, Age: {} ms, Accesses: {}", 
                        i + 1, candidate.address, candidate.size / 1024, 
                        candidate.age, candidate.access_count);
            }
        } else {
            println!("\n✓ No memory leaks detected");
        }
        
        // 内存池信息
        if !self.memory_pools.is_empty() {
            println!("\n🏊 Memory Pools:");
            for (name, pool) in &self.memory_pools {
                println!("  {}: {:.1}% used, {:.2}% fragmented", 
                        name, 
                        (pool.used_size as f64 / pool.total_size as f64) * 100.0,
                        pool.fragmentation_ratio * 100.0);
            }
        }
        
        println!("=======================================\n");
    }

    /// 导出详细报告到文件
    pub fn export_report(&self, filename: &str) -> Result<(), std::io::Error> {
        let mut file = File::create(filename)?;
        
        writeln!(file, "Complete Memory Tracer Detailed Report")?;
        writeln!(file, "Generated at: {:?}", SystemTime::now())?;
        writeln!(file, "=".repeat(50))?;
        
        // 写入统计信息
        writeln!(file, "\nStatistics:")?;
        writeln!(file, "Total allocations: {}", self.stats.total_allocations)?;
        writeln!(file, "Current allocations: {}", self.stats.current_allocations)?;
        writeln!(file, "Peak allocations: {}", self.stats.peak_allocations)?;
        writeln!(file, "Current bytes: {}", self.stats.current_bytes)?;
        writeln!(file, "Peak bytes: {}", self.stats.peak_bytes)?;
        
        // 写入当前分配
        writeln!(file, "\nCurrent Allocations:")?;
        for (address, allocation) in &self.allocations {
            writeln!(file, "0x{:x}: {} bytes, type: {:?}, accesses: {}", 
                    address, allocation.size, allocation.allocation_type, allocation.access_count)?;
        }
        
        Ok(())
    }

    /// 获取线程ID
    fn get_thread_id(&self) -> u64 {
        // 简化实现，使用线程名称hash作为ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        thread::current().id().hash(&mut hasher);
        hasher.finish()
    }

    /// 捕获堆栈跟踪（简化版）
    fn capture_stack_trace(&self) -> Vec<String> {
        // 简化实现，返回当前函数信息
        vec!["track_alloc_with_type".to_string()]
    }

    /// 设置配置
    pub fn set_config(&mut self, config: TrackerConfig) {
        self.config = config;
    }

    /// 获取脏页面数量
    pub fn get_dirty_pages_count(&self) -> u64 {
        self.stats.dirty_pages_count
    }

    /// 启用/禁用跟踪
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// 获取全局跟踪器
pub fn get_simple_tracer() -> Arc<Mutex<CompleteMemoryTracer>> {
    GLOBAL_TRACER.clone()
}

/// 便捷函数：跟踪内存复制
pub fn track_memory_copy(dst: u64, src: u64, size: usize) {
    if let Ok(mut tracer) = GLOBAL_TRACER.try_lock() {
        tracer.track_memory_access(src, size, AccessType::Read);
        tracer.track_memory_access(dst, size, AccessType::Write);
    }
}

/// 便捷函数：跟踪内存设置
pub fn track_memory_set(dst: u64, size: usize) {
    if let Ok(mut tracer) = GLOBAL_TRACER.try_lock() {
        tracer.track_memory_access(dst, size, AccessType::Memset);
    }
}

/// 便捷函数：执行内存泄漏检测
pub fn detect_leaks() {
    if let Ok(mut tracer) = GLOBAL_TRACER.try_lock() {
        tracer.detect_memory_leaks();
    }
}

// C API 导出函数
#[no_mangle]
pub unsafe extern "C" fn zluda_get_memory_stats(
    total_allocations: *mut u64,
    current_allocations: *mut u64,
    peak_allocations: *mut u64,
    current_bytes: *mut u64,
    peak_bytes: *mut u64,
    dirty_pages_count: *mut u64,
) -> i32 {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        let stats = tracer.get_complete_stats();

        if !total_allocations.is_null() {
            *total_allocations = stats.total_allocations;
        }
        if !current_allocations.is_null() {
            *current_allocations = stats.current_allocations;
        }
        if !peak_allocations.is_null() {
            *peak_allocations = stats.peak_allocations;
        }
        if !current_bytes.is_null() {
            *current_bytes = stats.current_bytes;
        }
        if !peak_bytes.is_null() {
            *peak_bytes = stats.peak_bytes;
        }
        if !dirty_pages_count.is_null() {
            *dirty_pages_count = stats.dirty_pages_count;
        }

        0 // Success
    } else {
        -1 // Error
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_print_memory_report() {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.print_report();
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_check_memory_leaks() -> i32 {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        let (_, _, _, current, _) = tracer.get_stats();
        if current > 0 {
            1 // Has leaks
        } else {
            0 // No leaks
        }
    } else {
        -1 // Error
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_get_dirty_pages_count() -> u64 {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.get_dirty_pages_count()
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_set_tracer_config(
    enable_dirty_tracking: i32,
    page_size: usize,
    max_history_size: usize,
    enable_leak_detection: i32,
    leak_threshold_ms: u64,
) -> i32 {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        let config = TrackerConfig {
            enable_dirty_tracking: enable_dirty_tracking != 0,
            enable_access_tracking: true,
            enable_pattern_analysis: true,
            enable_leak_detection: enable_leak_detection != 0,
            max_history_size,
            page_size,
            leak_detection_threshold: leak_threshold_ms,
            report_interval_ms: 30000,
        };
        tracer.set_config(config);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_memory_tracer_init() {
    let _tracer = get_simple_tracer();
    eprintln!("[ZLUDA] Complete Memory Tracer initialized with advanced features");
}

/// 新增的C API函数
#[no_mangle]
pub unsafe extern "C" fn zluda_track_memory_access(
    address: u64,
    size: usize,
    access_type: i32, // 0=Read, 1=Write, 2=ReadWrite, 3=Copy, 4=Memset
) -> i32 {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        let access = match access_type {
            0 => AccessType::Read,
            1 => AccessType::Write,
            2 => AccessType::ReadWrite,
            3 => AccessType::Copy,
            4 => AccessType::Memset,
            _ => AccessType::Read,
        };
        tracer.track_memory_access(address, size, access);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_detect_memory_leaks() -> i32 {
    if let Ok(mut tracer) = get_simple_tracer().try_lock() {
        tracer.detect_memory_leaks();
        tracer.get_complete_stats().leak_count as i32
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_export_memory_report(filename: *const std::os::raw::c_char) -> i32 {
    if filename.is_null() {
        return -1;
    }
    
    let filename_str = match std::ffi::CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        match tracer.export_report(filename_str) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn zluda_memory_tracer_cleanup() {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.print_report();
    }
}