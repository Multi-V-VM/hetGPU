//! 简化的动态内存跟踪器
//! 
//! 这是一个轻量级版本，专注于基本的内存分配跟踪功能

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// 全局内存跟踪器实例
static GLOBAL_TRACER: std::sync::LazyLock<Arc<Mutex<SimpleMemoryTracer>>> = 
    std::sync::LazyLock::new(|| {
        Arc::new(Mutex::new(SimpleMemoryTracer::new()))
    });

/// 简化的内存跟踪器
pub struct SimpleMemoryTracer {
    allocations: HashMap<u64, SimpleAllocation>,
    stats: SimpleStats,
    enabled: bool,
}

/// 简化的分配记录
#[derive(Debug, Clone)]
struct SimpleAllocation {
    size: usize,
    timestamp: u64,
}

/// 简化的统计信息
#[derive(Debug, Default)]
struct SimpleStats {
    total_allocations: u64,
    current_allocations: u64,
    peak_allocations: u64,
    current_bytes: u64,
    peak_bytes: u64,
}

impl SimpleMemoryTracer {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            stats: SimpleStats::default(),
            enabled: true,
        }
    }

    /// 跟踪分配
    pub fn track_alloc(&mut self, address: u64, size: usize) {
        if !self.enabled {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let allocation = SimpleAllocation { size, timestamp };
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

        eprintln!("[SimpleTracer] Alloc {} bytes at 0x{:x}", size, address);
    }

    /// 跟踪释放
    pub fn track_free(&mut self, address: u64) {
        if let Some(allocation) = self.allocations.remove(&address) {
            self.stats.current_allocations -= 1;
            self.stats.current_bytes -= allocation.size as u64;
            
            eprintln!("[SimpleTracer] Free {} bytes at 0x{:x}", allocation.size, address);
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

    /// 打印报告
    pub fn print_report(&self) {
        println!("\n=== Simple Memory Tracer Report ===");
        println!("Total allocations: {}", self.stats.total_allocations);
        println!("Current allocations: {}", self.stats.current_allocations);
        println!("Peak allocations: {}", self.stats.peak_allocations);
        println!("Current memory: {:.2} MB", self.stats.current_bytes as f64 / (1024.0 * 1024.0));
        println!("Peak memory: {:.2} MB", self.stats.peak_bytes as f64 / (1024.0 * 1024.0));
        
        if self.stats.current_allocations > 0 {
            println!("⚠️ Memory leaks detected: {} unreleased allocations", self.stats.current_allocations);
        } else {
            println!("✓ No memory leaks detected");
        }
        println!("==================================\n");
    }
}

/// 获取全局跟踪器
pub fn get_simple_tracer() -> Arc<Mutex<SimpleMemoryTracer>> {
    GLOBAL_TRACER.clone()
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
        let (total, current, peak, bytes_current, bytes_peak) = tracer.get_stats();

        if !total_allocations.is_null() {
            *total_allocations = total;
        }
        if !current_allocations.is_null() {
            *current_allocations = current;
        }
        if !peak_allocations.is_null() {
            *peak_allocations = peak;
        }
        if !current_bytes.is_null() {
            *current_bytes = bytes_current;
        }
        if !peak_bytes.is_null() {
            *peak_bytes = bytes_peak;
        }
        if !dirty_pages_count.is_null() {
            *dirty_pages_count = 0; // 简化版本不跟踪脏页面
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
    0 // 简化版本不跟踪脏页面
}

#[no_mangle]
pub unsafe extern "C" fn zluda_set_tracer_config(
    _enable_dirty_tracking: i32,
    _page_size: usize,
    _max_dirty_pages: usize,
    _enable_compression: i32,
    _enable_stats: i32,
) -> i32 {
    0 // 简化版本忽略配置
}

#[no_mangle]
pub unsafe extern "C" fn zluda_memory_tracer_init() {
    let _tracer = get_simple_tracer();
    eprintln!("[ZLUDA] Simple Memory Tracer initialized");
}

#[no_mangle]
pub unsafe extern "C" fn zluda_memory_tracer_cleanup() {
    if let Ok(tracer) = get_simple_tracer().try_lock() {
        tracer.print_report();
    }
}