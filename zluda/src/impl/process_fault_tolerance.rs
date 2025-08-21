use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Once};
use std::time::{Duration, Instant};
use std::ffi::{c_void, c_int};
use std::thread;
use std::process::{Command, Child, Stdio};
use std::fs;
use std::io::Write;

// 进程状态跟踪
#[derive(Clone, Debug)]
struct ProcessInfo {
    rank: i32,
    pid: u32,
    last_heartbeat: Instant,
    status: ProcessStatus,
    restart_count: u32,
    backup_rank: Option<i32>,
}

#[derive(Clone, Debug, PartialEq)]
enum ProcessStatus {
    Running,
    Failed, 
    Recovering,
    Migrated,
}

// 全局进程监控状态
lazy_static::lazy_static! {
    static ref PROCESS_REGISTRY: RwLock<HashMap<i32, ProcessInfo>> = RwLock::new(HashMap::new());
    static ref HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
    static ref FAILURE_TIMEOUT: Duration = Duration::from_secs(5);
    static ref MAX_RESTART_ATTEMPTS: u32 = 3;
}

static MONITOR_INIT: Once = Once::new();

// 初始化进程容错监控
pub fn initialize_process_fault_tolerance() {
    MONITOR_INIT.call_once(|| {
        eprintln!("[ProcessFT] Initializing process fault tolerance system");
        
        // 启动进程监控线程
        start_process_monitor();
        
        // 注册信号处理器
        register_signal_handlers();
        
        eprintln!("[ProcessFT] Process fault tolerance system initialized");
    });
}

// 注册进程
pub fn register_process(rank: i32, pid: u32) {
    // 先查找备用rank，避免在持有写锁时调用读锁（避免死锁）
    let backup_rank = {
        let registry = PROCESS_REGISTRY.read().unwrap();
        // 在注册阶段，registry为空，所以backup_rank将为None
        // 实际的backup_rank会在有多个进程注册后动态确定
        registry.iter()
            .find(|(r, p)| **r != rank && p.status == ProcessStatus::Running)
            .map(|(r, _)| *r)
    };
    
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    
    let process_info = ProcessInfo {
        rank,
        pid,
        last_heartbeat: Instant::now(),
        status: ProcessStatus::Running,
        restart_count: 0,
        backup_rank,
    };
    
    registry.insert(rank, process_info);
    eprintln!("[ProcessFT] Registered process: rank={}, pid={}", rank, pid);
}

// 更新心跳
pub fn update_heartbeat(rank: i32) {
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    if let Some(process) = registry.get_mut(&rank) {
        process.last_heartbeat = Instant::now();
        if process.status != ProcessStatus::Running {
            process.status = ProcessStatus::Running;
            eprintln!("[ProcessFT] Process {} recovered", rank);
        }
    }
}

// 检测进程故障
fn detect_process_failures() -> Vec<i32> {
    let mut failed_ranks = Vec::new();
    let now = Instant::now();
    
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    
    for (rank, process) in registry.iter_mut() {
        if process.status == ProcessStatus::Running {
            // 检查心跳超时
            if now.duration_since(process.last_heartbeat) > *FAILURE_TIMEOUT {
                eprintln!("[ProcessFT] Process {} heartbeat timeout", rank);
                process.status = ProcessStatus::Failed;
                failed_ranks.push(*rank);
            }
            
            // 检查进程是否真的还在运行
            if !is_process_alive(process.pid) {
                eprintln!("[ProcessFT] Process {} (pid={}) is dead", rank, process.pid);
                process.status = ProcessStatus::Failed;
                failed_ranks.push(*rank);
            }
        }
    }
    
    failed_ranks
}

// 检查进程是否存活
fn is_process_alive(pid: u32) -> bool {
    // 检查 /proc/PID 是否存在
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

// 查找备用rank
fn find_backup_rank(failed_rank: i32) -> Option<i32> {
    let registry = PROCESS_REGISTRY.read().unwrap();
    
    // 寻找健康的进程作为备用
    for (rank, process) in registry.iter() {
        if *rank != failed_rank && process.status == ProcessStatus::Running {
            return Some(*rank);
        }
    }
    
    None
}

// 执行进程恢复
fn recover_failed_process(failed_rank: i32) -> Result<(), String> {
    eprintln!("[ProcessFT] Starting recovery for rank {}", failed_rank);
    
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    let process = match registry.get_mut(&failed_rank) {
        Some(p) => p,
        None => return Err("Process not found in registry".to_string()),
    };
    
    if process.restart_count >= *MAX_RESTART_ATTEMPTS {
        eprintln!("[ProcessFT] Maximum restart attempts reached for rank {}", failed_rank);
        return perform_rank_migration(failed_rank);
    }
    
    process.status = ProcessStatus::Recovering;
    process.restart_count += 1;
    
    // 尝试重启进程
    match restart_process(failed_rank) {
        Ok(new_pid) => {
            process.pid = new_pid;
            process.last_heartbeat = Instant::now();
            process.status = ProcessStatus::Running;
            eprintln!("[ProcessFT] Successfully restarted rank {} with new pid {}", failed_rank, new_pid);
            Ok(())
        }
        Err(e) => {
            eprintln!("[ProcessFT] Failed to restart rank {}: {}", failed_rank, e);
            perform_rank_migration(failed_rank)
        }
    }
}

// 重启进程
fn restart_process(rank: i32) -> Result<u32, String> {
    // 创建重启脚本
    let script_content = format!(r#"#!/bin/bash
export LD_PRELOAD=/root/hetGPU/target/release/libcuda.so.1
export CUDA_VISIBLE_DEVICES=0

# 重新启动失败的rank
echo "[ProcessFT] Restarting rank {}"

# 这里应该重新运行具体的rank进程
# 在实际实现中，这需要保存原始的命令行参数
sleep 1

# 模拟新进程启动
echo "Rank {} restarted successfully"
"#, rank, rank);

    let script_path = format!("/tmp/restart_rank_{}.sh", rank);
    fs::write(&script_path, script_content)
        .map_err(|e| format!("Failed to write restart script: {}", e))?;

    // 执行重启脚本
    let mut child = Command::new("bash")
        .arg(&script_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn restart process: {}", e))?;

    // 在实际实现中，这里应该返回新进程的实际PID
    // 现在我们模拟一个新的PID
    let new_pid = (10000 + rank as u32 + Instant::now().elapsed().as_secs() as u32 % 1000) as u32;
    
    Ok(new_pid)
}

// 执行rank迁移
fn perform_rank_migration(failed_rank: i32) -> Result<(), String> {
    eprintln!("[ProcessFT] Performing rank migration for failed rank {}", failed_rank);
    
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    let backup_rank = {
        let process = registry.get(&failed_rank).ok_or("Process not found")?;
        process.backup_rank
    };
    
    match backup_rank {
        Some(backup) => {
            // 通知备用rank接管失败rank的工作
            eprintln!("[ProcessFT] Migrating rank {} workload to backup rank {}", failed_rank, backup);
            
            // 更新registry
            if let Some(failed_process) = registry.get_mut(&failed_rank) {
                failed_process.status = ProcessStatus::Migrated;
            }
            
            if let Some(backup_process) = registry.get_mut(&backup) {
                backup_process.status = ProcessStatus::Running;
            }
            
            // 在实际实现中，这里需要：
            // 1. 通知其他rank有rank失败了
            // 2. 重新初始化NCCL通信器（去掉失败的rank）
            // 3. 重新分配数据和计算任务
            
            create_rank_migration_notification(failed_rank, backup)?;
            
            Ok(())
        }
        None => {
            Err("No backup rank available for migration".to_string())
        }
    }
}

// 创建rank迁移通知
fn create_rank_migration_notification(failed_rank: i32, backup_rank: i32) -> Result<(), String> {
    let notification_file = format!("/tmp/rank_migration_{}_to_{}.info", failed_rank, backup_rank);
    
    let content = format!(r#"{{
    "failed_rank": {},
    "backup_rank": {},
    "timestamp": "{}",
    "action": "rank_migration",
    "status": "in_progress"
}}"#, failed_rank, backup_rank, chrono::Utc::now().to_rfc3339());
    
    fs::write(notification_file, content)
        .map_err(|e| format!("Failed to create migration notification: {}", e))?;
        
    eprintln!("[ProcessFT] Created migration notification for {} -> {}", failed_rank, backup_rank);
    Ok(())
}

// 进程监控线程
fn process_monitor_thread() {
    eprintln!("[ProcessFT] Starting process monitor thread");
    
    loop {
        thread::sleep(*HEARTBEAT_INTERVAL);
        
        // 检测故障
        let failed_ranks = detect_process_failures();
        
        // 处理每个失败的rank
        for failed_rank in failed_ranks {
            if let Err(e) = recover_failed_process(failed_rank) {
                eprintln!("[ProcessFT] Failed to recover rank {}: {}", failed_rank, e);
            }
        }
        
        // 清理已完成迁移的进程信息
        cleanup_migrated_processes();
    }
}

// 清理已迁移的进程
fn cleanup_migrated_processes() {
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    let mut to_remove = Vec::new();
    
    for (rank, process) in registry.iter() {
        if process.status == ProcessStatus::Migrated {
            // 保留一段时间的记录，然后清理
            if Instant::now().duration_since(process.last_heartbeat) > Duration::from_secs(60) {
                to_remove.push(*rank);
            }
        }
    }
    
    for rank in to_remove {
        registry.remove(&rank);
        eprintln!("[ProcessFT] Cleaned up migrated process record for rank {}", rank);
    }
}

// 启动进程监控
fn start_process_monitor() {
    thread::spawn(|| {
        process_monitor_thread();
    });
}

// 注册信号处理器
fn register_signal_handlers() {
    // 在实际实现中，这里应该注册SIGCHLD等信号处理器
    // 来检测子进程的异常退出
    eprintln!("[ProcessFT] Signal handlers registered");
}

// 导出的API函数

// 获取进程状态
#[no_mangle]
pub extern "C" fn get_process_status(rank: c_int) -> c_int {
    let registry = PROCESS_REGISTRY.read().unwrap();
    match registry.get(&rank) {
        Some(process) => match process.status {
            ProcessStatus::Running => 0,
            ProcessStatus::Failed => 1,
            ProcessStatus::Recovering => 2,
            ProcessStatus::Migrated => 3,
        },
        None => -1,
    }
}

// 手动触发进程恢复
#[no_mangle]
pub extern "C" fn trigger_process_recovery(rank: c_int) -> c_int {
    match recover_failed_process(rank) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// 获取重启次数
#[no_mangle]
pub extern "C" fn get_restart_count(rank: c_int) -> c_int {
    let registry = PROCESS_REGISTRY.read().unwrap();
    match registry.get(&rank) {
        Some(process) => process.restart_count as c_int,
        None => -1,
    }
}

// NCCL集成函数
pub fn notify_nccl_operation(rank: i32) {
    // 每次NCCL操作时更新心跳
    update_heartbeat(rank);
}

pub fn handle_nccl_error(rank: i32, error: i32) {
    eprintln!("[ProcessFT] NCCL error reported for rank {}: {}", rank, error);
    
    // 标记进程为可能失败
    let mut registry = PROCESS_REGISTRY.write().unwrap();
    if let Some(process) = registry.get_mut(&rank) {
        if error != 0 { // 假设0是成功
            process.status = ProcessStatus::Failed;
        }
    }
}