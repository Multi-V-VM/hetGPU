//! LD_PRELOAD 集成的动态内存分配器
//! 
//! 这个模块提供了一个可以通过 LD_PRELOAD 预加载的动态内存分配器，
//! 专门用于跟踪和管理 PTX/CUDA 程序的内存分配，集成脏页面跟踪功能。

use crate::dynamic_delta_analyzer::{DirtyMemoryManager, DirtyMemoryConfig, DirtyMemoryDelta};
use crate::TranslateError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::ffi::{c_void, CStr, CString};
use libc::{size_t, c_int, c_char};
use std::ptr;

/// 全局内存分配器实例
static GLOBAL_ALLOCATOR: std::sync::LazyLock<Arc<Mutex<PreloadAllocator>>> = 
    std::sync::LazyLock::new(|| {
        Arc::new(Mutex::new(PreloadAllocator::new()))
    });

/// LD_PRELOAD 兼容的内存分配器
pub struct PreloadAllocator {
    /// 脏内存管理器
    dirty_manager: DirtyMemoryManager,
    /// 分配记录
    allocations: HashMap<*mut c_void, AllocationInfo>,
    /// CUDA 内存分配映射
    cuda_allocations: HashMap<*mut c_void, CudaAllocationInfo>,
    /// 原始 libc 函数指针
    original_functions: OriginalFunctions,
    /// 统计信息
    stats: AllocationStats,
    /// 钩子是否已安装
    hooks_installed: bool,
}

/// 分配信息记录
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: size_t,
    pub timestamp: u64,
    pub backtrace: Vec<String>,
    pub allocation_type: AllocationType,
}

/// CUDA 分配信息
#[derive(Debug, Clone)]
pub struct CudaAllocationInfo {
    pub size: size_t,
    pub device_id: i32,
    pub memory_type: CudaMemoryType,
    pub allocation_flags: u32,
}

/// 分配类型
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationType {
    Malloc,
    Calloc,
    Realloc,
    Memalign,
    CudaMalloc,
    CudaMallocManaged,
    CudaMallocHost,
}

/// CUDA 内存类型
#[derive(Debug, Clone, PartialEq)]
pub enum CudaMemoryType {
    Device,
    Host,
    Managed,
    Pinned,
}

/// 原始函数指针
#[derive(Debug)]
pub struct OriginalFunctions {
    pub malloc: Option<unsafe extern "C" fn(size_t) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(*mut c_void)>,
    pub calloc: Option<unsafe extern "C" fn(size_t, size_t) -> *mut c_void>,
    pub realloc: Option<unsafe extern "C" fn(*mut c_void, size_t) -> *mut c_void>,
    pub memalign: Option<unsafe extern "C" fn(size_t, size_t) -> *mut c_void>,
    pub cuda_malloc: Option<unsafe extern "C" fn(*mut *mut c_void, size_t) -> c_int>,
    pub cuda_free: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
}

/// 分配统计信息
#[derive(Debug, Default)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub total_bytes_allocated: u64,
    pub total_bytes_freed: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub cuda_allocations: u64,
    pub host_allocations: u64,
}

impl PreloadAllocator {
    /// 创建新的预加载分配器
    pub fn new() -> Self {
        let config = DirtyMemoryConfig {
            page_size: 4096,
            enable_cow: true,
            enable_compression: true,
            max_dirty_pages: 100000,
            hash_algorithm: crate::dynamic_delta_analyzer::HashAlgorithm::Xxhash,
        };

        Self {
            dirty_manager: DirtyMemoryManager::new(config),
            allocations: HashMap::new(),
            cuda_allocations: HashMap::new(),
            original_functions: OriginalFunctions::new(),
            stats: AllocationStats::default(),
            hooks_installed: false,
        }
    }

    /// 初始化钩子和原始函数指针
    pub fn initialize(&mut self) -> Result<(), TranslateError> {
        if self.hooks_installed {
            return Ok(());
        }

        // 获取原始函数指针
        self.original_functions.load_original_functions()?;
        
        // 安装脏内存钩子
        self.dirty_manager.install_memory_hooks()?;
        
        self.hooks_installed = true;
        
        eprintln!("[PreloadAllocator] 初始化完成，钩子已安装");
        Ok(())
    }

    /// 跟踪内存分配
    pub fn track_allocation(
        &mut self,
        ptr: *mut c_void,
        size: size_t,
        alloc_type: AllocationType,
    ) -> Result<(), TranslateError> {
        if ptr.is_null() {
            return Ok(());
        }

        let info = AllocationInfo {
            size,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            backtrace: self.capture_backtrace(),
            allocation_type: alloc_type.clone(),
        };

        self.allocations.insert(ptr, info);
        
        // 更新统计
        self.stats.total_allocations += 1;
        self.stats.current_allocations += 1;
        self.stats.total_bytes_allocated += size as u64;
        self.stats.current_bytes += size as u64;
        
        if self.stats.current_allocations > self.stats.peak_allocations {
            self.stats.peak_allocations = self.stats.current_allocations;
        }
        
        if self.stats.current_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.current_bytes;
        }

        match alloc_type {
            AllocationType::CudaMalloc | 
            AllocationType::CudaMallocManaged | 
            AllocationType::CudaMallocHost => {
                self.stats.cuda_allocations += 1;
            }
            _ => {
                self.stats.host_allocations += 1;
            }
        }

        // 通知脏内存管理器有新的内存分配
        if size > 0 {
            let dummy_data = vec![0u8; std::cmp::min(size, 4096)]; // 最多记录4KB
            self.dirty_manager.on_memory_write(ptr as u64, dummy_data.len() as u32, &dummy_data)?;
        }

        Ok(())
    }

    /// 跟踪内存释放
    pub fn track_deallocation(&mut self, ptr: *mut c_void) -> Result<(), TranslateError> {
        if ptr.is_null() {
            return Ok(());
        }

        if let Some(info) = self.allocations.remove(&ptr) {
            // 更新统计
            self.stats.total_deallocations += 1;
            self.stats.current_allocations -= 1;
            self.stats.total_bytes_freed += info.size as u64;
            self.stats.current_bytes -= info.size as u64;
        }

        // 从CUDA分配记录中移除
        self.cuda_allocations.remove(&ptr);

        Ok(())
    }

    /// 获取分配统计信息
    pub fn get_stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// 获取脏内存增量
    pub fn get_dirty_memory_delta(&self) -> Result<Vec<DirtyMemoryDelta>, TranslateError> {
        self.dirty_manager.get_dirty_memory_delta()
    }

    /// 捕获调用栈
    fn capture_backtrace(&self) -> Vec<String> {
        // 简化的调用栈捕获 - 在实际实现中可以使用 backtrace crate
        vec!["<backtrace_placeholder>".to_string()]
    }

    /// 打印内存泄漏报告
    pub fn print_leak_report(&self) {
        if self.allocations.is_empty() {
            eprintln!("[PreloadAllocator] 没有检测到内存泄漏");
            return;
        }

        eprintln!("[PreloadAllocator] 内存泄漏报告:");
        eprintln!("  未释放分配数量: {}", self.allocations.len());
        
        let total_leaked: u64 = self.allocations.values()
            .map(|info| info.size as u64)
            .sum();
        
        eprintln!("  总泄漏字节数: {} ({:.2} MB)", total_leaked, total_leaked as f64 / (1024.0 * 1024.0));
        
        // 按分配类型分组
        let mut type_stats: HashMap<AllocationType, (u64, u64)> = HashMap::new();
        for info in self.allocations.values() {
            let (count, size) = type_stats.entry(info.allocation_type.clone()).or_insert((0, 0));
            *count += 1;
            *size += info.size as u64;
        }
        
        for (alloc_type, (count, size)) in type_stats {
            eprintln!("  {:?}: {} 次分配, {} 字节", alloc_type, count, size);
        }
    }
}

impl OriginalFunctions {
    pub fn new() -> Self {
        Self {
            malloc: None,
            free: None,
            calloc: None,
            realloc: None,
            memalign: None,
            cuda_malloc: None,
            cuda_free: None,
        }
    }

    pub fn load_original_functions(&mut self) -> Result<(), TranslateError> {
        unsafe {
            // 获取原始 libc 函数
            self.malloc = Self::get_original_function("malloc");
            self.free = Self::get_original_function("free");
            self.calloc = Self::get_original_function("calloc");
            self.realloc = Self::get_original_function("realloc");
            self.memalign = Self::get_original_function("memalign");
            
            // 尝试获取 CUDA 函数（可能不存在）
            self.cuda_malloc = Self::get_original_function("cudaMalloc");
            self.cuda_free = Self::get_original_function("cudaFree");
        }
        
        Ok(())
    }

    unsafe fn get_original_function<T>(name: &str) -> Option<T> {
        let name_cstr = CString::new(name).ok()?;
        let handle = libc::dlsym(libc::RTLD_NEXT, name_cstr.as_ptr());
        if handle.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&handle))
        }
    }
}

// LD_PRELOAD 导出的 C 函数接口
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// malloc 钩子
#[no_mangle]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    // 初始化分配器（如果尚未初始化）
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        let _ = allocator.initialize();
        
        // 调用原始 malloc
        if let Some(original_malloc) = allocator.original_functions.malloc {
            let ptr = original_malloc(size);
            
            // 跟踪分配
            let _ = allocator.track_allocation(ptr, size, AllocationType::Malloc);
            
            return ptr;
        }
    }
    
    // 回退到系统 malloc
    libc::malloc(size)
}

/// free 钩子
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        // 跟踪释放
        let _ = allocator.track_deallocation(ptr);
        
        // 调用原始 free
        if let Some(original_free) = allocator.original_functions.free {
            original_free(ptr);
            return;
        }
    }
    
    // 回退到系统 free
    libc::free(ptr);
}

/// calloc 钩子
#[no_mangle]
pub unsafe extern "C" fn calloc(nmemb: size_t, size: size_t) -> *mut c_void {
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        let _ = allocator.initialize();
        
        if let Some(original_calloc) = allocator.original_functions.calloc {
            let ptr = original_calloc(nmemb, size);
            let total_size = nmemb * size;
            
            let _ = allocator.track_allocation(ptr, total_size, AllocationType::Calloc);
            
            return ptr;
        }
    }
    
    libc::calloc(nmemb, size)
}

/// realloc 钩子
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        let _ = allocator.initialize();
        
        // 先跟踪旧指针的释放
        if !ptr.is_null() {
            let _ = allocator.track_deallocation(ptr);
        }
        
        if let Some(original_realloc) = allocator.original_functions.realloc {
            let new_ptr = original_realloc(ptr, size);
            
            // 跟踪新分配
            if !new_ptr.is_null() {
                let _ = allocator.track_allocation(new_ptr, size, AllocationType::Realloc);
            }
            
            return new_ptr;
        }
    }
    
    libc::realloc(ptr, size)
}

/// CUDA malloc 钩子 (如果链接了 CUDA)
#[no_mangle]
pub unsafe extern "C" fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> c_int {
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        let _ = allocator.initialize();
        
        if let Some(original_cuda_malloc) = allocator.original_functions.cuda_malloc {
            let result = original_cuda_malloc(devPtr, size);
            
            if result == 0 && !devPtr.is_null() { // cudaSuccess
                let ptr = *devPtr;
                let _ = allocator.track_allocation(ptr, size, AllocationType::CudaMalloc);
                
                // 记录 CUDA 特定信息
                let cuda_info = CudaAllocationInfo {
                    size,
                    device_id: 0, // 可以通过 cudaGetDevice 获取
                    memory_type: CudaMemoryType::Device,
                    allocation_flags: 0,
                };
                allocator.cuda_allocations.insert(ptr, cuda_info);
            }
            
            return result;
        }
    }
    
    // 如果没有原始 CUDA 函数，返回错误
    1 // cudaErrorMemoryAllocation
}

/// CUDA free 钩子
#[no_mangle]
pub unsafe extern "C" fn cudaFree(devPtr: *mut c_void) -> c_int {
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        let _ = allocator.track_deallocation(devPtr);
        
        if let Some(original_cuda_free) = allocator.original_functions.cuda_free {
            return original_cuda_free(devPtr);
        }
    }
    
    1 // cudaErrorInvalidValue
}

/// 获取分配器统计信息的 C 接口
#[no_mangle]
pub unsafe extern "C" fn get_allocator_stats() -> *const AllocationStats {
    if let Ok(allocator) = GLOBAL_ALLOCATOR.try_lock() {
        allocator.get_stats() as *const AllocationStats
    } else {
        ptr::null()
    }
}

/// 打印内存泄漏报告的 C 接口
#[no_mangle]
pub unsafe extern "C" fn print_memory_leak_report() {
    if let Ok(allocator) = GLOBAL_ALLOCATOR.try_lock() {
        allocator.print_leak_report();
    }
}

/// 获取脏内存增量的 C 接口
#[no_mangle]
pub unsafe extern "C" fn get_dirty_memory_delta_count() -> size_t {
    if let Ok(allocator) = GLOBAL_ALLOCATOR.try_lock() {
        if let Ok(deltas) = allocator.get_dirty_memory_delta() {
            return deltas.len();
        }
    }
    0
}

/// 程序退出时的清理函数
#[no_mangle]
pub unsafe extern "C" fn __attribute__((destructor)) cleanup_allocator() {
    if let Ok(allocator) = GLOBAL_ALLOCATOR.try_lock() {
        eprintln!("[PreloadAllocator] 程序退出清理:");
        eprintln!("  总分配: {}", allocator.stats.total_allocations);
        eprintln!("  总释放: {}", allocator.stats.total_deallocations);
        eprintln!("  当前分配: {}", allocator.stats.current_allocations);
        eprintln!("  峰值内存: {:.2} MB", allocator.stats.peak_bytes as f64 / (1024.0 * 1024.0));
        
        allocator.print_leak_report();
    }
}

// 构造函数，程序启动时自动调用
#[no_mangle]
pub unsafe extern "C" fn __attribute__((constructor)) init_allocator() {
    eprintln!("[PreloadAllocator] 正在初始化内存分配器钩子...");
    
    if let Ok(mut allocator) = GLOBAL_ALLOCATOR.try_lock() {
        if let Err(e) = allocator.initialize() {
            eprintln!("[PreloadAllocator] 初始化失败: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_tracking() {
        let mut allocator = PreloadAllocator::new();
        let _ = allocator.initialize();
        
        // 模拟分配
        let ptr = 0x1000 as *mut c_void;
        allocator.track_allocation(ptr, 1024, AllocationType::Malloc).unwrap();
        
        // 检查统计
        assert_eq!(allocator.stats.total_allocations, 1);
        assert_eq!(allocator.stats.current_bytes, 1024);
        
        // 模拟释放
        allocator.track_deallocation(ptr).unwrap();
        assert_eq!(allocator.stats.current_allocations, 0);
    }

    #[test]
    fn test_cuda_allocation_tracking() {
        let mut allocator = PreloadAllocator::new();
        let _ = allocator.initialize();
        
        let ptr = 0x2000 as *mut c_void;
        allocator.track_allocation(ptr, 2048, AllocationType::CudaMalloc).unwrap();
        
        assert_eq!(allocator.stats.cuda_allocations, 1);
        assert_eq!(allocator.stats.host_allocations, 0);
    }
}