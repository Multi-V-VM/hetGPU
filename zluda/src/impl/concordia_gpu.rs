//! Concordia GPU-Side Persistent Kernel
//!
//! Generates PTX for a persistent worker kernel, loads it via the CUDA driver API,
//! launches once, and provides a managed-memory ring buffer for task submission.
//! The GPU kernel polls the ring buffer and dispatches through a function pointer table.
//!
//! This is the GPUOS pattern implemented entirely in Rust via CUDA driver API calls.

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

/// The PTX source for the persistent worker kernel.
/// This is a minimal but complete implementation matching GPUOS's persistent_worker.
/// It defines:
///   - Task struct (op, numel, 3 pointers for in0/in1/out0)
///   - WorkQueue struct (tasks ring, head, tail, quit, sync counters)
///   - persistent_worker: polls ring, dispatches through indirect call
///   - Built-in op_add for testing
// We generate PTX at runtime via NVRTC instead of handwriting it,
// since handwritten PTX is error-prone. This CUDA C source gets compiled
// to PTX by nvrtcCompileProgram, then loaded via cuModuleLoadData.
const PERSISTENT_KERNEL_CUDA: &str = r#"
extern "C" {

struct Task {
    int op;
    int flags;
    long long numel;
    void* in0;
    void* in1;
    void* out0;
    int num_params;
    int _pad[5];  // pad to 64 bytes
};

struct WorkQueue {
    Task* tasks;
    int capacity;
    int* head;
    int* tail;
    int* quit;
    unsigned long long* processed;
};

__device__ void do_add(const Task& t) {
    long long N = t.numel;
    float* a = (float*)t.in0;
    float* b = (float*)t.in1;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x; i < N; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void persistent_worker(
    Task* tasks, int capacity,
    int* head, int* tail, int* quit,
    unsigned long long* processed
) {
    __shared__ Task s_task;
    __shared__ int s_has_work;

    // Signal that kernel is alive
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU] persistent_worker started: tasks=%p cap=%d head=%p tail=%p quit=%p proc=%p\\n",
               tasks, capacity, head, tail, quit, processed);
    }
    __syncthreads();

    while (atomicAdd(quit, 0) == 0) {
        if (threadIdx.x == 0) {
            s_has_work = 0;
            int h = atomicAdd(head, 0);
            int t = atomicAdd(tail, 0);
            if (h < t) {
                if (atomicCAS(head, h, h + 1) == h) {
                    s_task = tasks[h % capacity];
                    s_has_work = 1;
                }
            }
        }
        __syncthreads();

        if (!s_has_work) {
            if (threadIdx.x == 0) {
                // Brief sleep to avoid burning SM
                #if __CUDA_ARCH__ >= 700
                __nanosleep(1000);
                #endif
            }
            __syncthreads();
            continue;
        }

        // Dispatch: currently inline add
        do_add(s_task);
        __syncthreads();

        if (threadIdx.x == 0) {
            __threadfence_system();
            // System-scope atomic: ensures host sees the update immediately
            atomicAdd_system(processed, 1ULL);
        }
        __syncthreads();
    }
}

} // extern "C"
"#;

const PERSISTENT_KERNEL_PTX: &str = r#"
UNUSED - we use NVRTC now

// Task descriptor: 64 bytes
// offset 0:  i32 op
// offset 4:  i32 flags
// offset 8:  i64 numel
// offset 16: ptr in0
// offset 24: ptr in1
// offset 32: ptr out0
// offset 40: i32 num_params
// offset 44: padding

// WorkQueue passed as kernel param struct:
// offset 0:  ptr tasks       (ring buffer of Task[capacity])
// offset 8:  i32 capacity
// offset 12: padding
// offset 16: ptr head         (device-side pop index, managed memory)
// offset 24: ptr tail         (host-side push index, managed memory)
// offset 32: ptr quit         (stop flag, managed memory)
// offset 40: ptr processed    (host-mapped completion counter)

// Built-in add operator: out0[i] = in0[i] + in1[i] for i in 0..numel
.visible .func op_add_builtin(
    .param .b64 task_ptr
)
{
    .reg .b64 %task, %in0, %in1, %out0;
    .reg .b64 %numel, %i, %stride;
    .reg .f32 %a, %b, %c;
    .reg .pred %p;

    ld.param.b64 %task, [task_ptr];

    // Load numel (offset 8)
    ld.global.u64 %numel, [%task + 8];

    // Load pointers
    ld.global.u64 %in0, [%task + 16];
    ld.global.u64 %in1, [%task + 24];
    ld.global.u64 %out0, [%task + 32];

    // i = threadIdx.x
    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %i, %r0;

    // stride = blockDim.x
    mov.u32 %r1, %ntid.x;
    cvt.u64.u32 %stride, %r1;

LOOP:
    setp.ge.u64 %p, %i, %numel;
    @%p bra DONE;

    // a = in0[i]
    shl.b64 %r2, %i, 2;  // *4 for float
    add.u64 %r3, %in0, %r2;
    ld.global.f32 %a, [%r3];

    // b = in1[i]
    add.u64 %r4, %in1, %r2;
    ld.global.f32 %b, [%r4];

    // c = a + b
    add.f32 %c, %a, %b;

    // out0[i] = c
    add.u64 %r5, %out0, %r2;
    st.global.f32 [%r5], %c;

    add.u64 %i, %i, %stride;
    bra LOOP;
DONE:
    ret;
}

// Persistent worker: polls WorkQueue, dispatches tasks
.visible .entry persistent_worker(
    .param .b64 .ptr .global tasks_ptr,
    .param .b32 capacity,
    .param .b64 .ptr .global head_ptr,
    .param .b64 .ptr .global tail_ptr,
    .param .b64 .ptr .global quit_ptr,
    .param .b64 .ptr .global processed_ptr
)
{
    .reg .b64 %tasks, %head_p, %tail_p, %quit_p, %proc_p;
    .reg .b32 %cap, %h, %t, %q, %has_work, %old, %slot;
    .reg .b64 %task_addr, %op_val, %one64;
    .reg .b32 %op, %zero32, %one32;
    .reg .pred %p_quit, %p_work, %p_leader;

    // Load params
    ld.param.b64 %tasks,  [tasks_ptr];
    ld.param.b32 %cap,    [capacity];
    ld.param.b64 %head_p, [head_ptr];
    ld.param.b64 %tail_p, [tail_ptr];
    ld.param.b64 %quit_p, [quit_ptr];
    ld.param.b64 %proc_p, [processed_ptr];

    mov.b32 %zero32, 0;
    mov.b32 %one32, 1;
    mov.b64 %one64, 1;

    // Check if thread 0
    mov.u32 %r0, %tid.x;
    setp.eq.u32 %p_leader, %r0, 0;

POLL_LOOP:
    // Check quit flag
    ld.global.b32 %q, [%quit_p];
    setp.ne.b32 %p_quit, %q, 0;
    @%p_quit bra EXIT;

    // Only leader thread polls
    mov.b32 %has_work, 0;
    @!%p_leader bra WAIT_SYNC;

    // Read head and tail
    ld.global.b32 %h, [%head_p];
    ld.global.b32 %t, [%tail_p];

    // If head >= tail, no work
    setp.ge.s32 %p_work, %h, %t;
    @%p_work bra WAIT_SYNC;

    // CAS to claim slot: head = head + 1
    atom.global.cas.b32 %old, [%head_p], %h, %h;
    // Simplified: just increment (single-block version)
    // For multi-block, use proper CAS loop
    add.s32 %slot, %h, 0;
    // Compute task address: tasks + (slot % capacity) * 64
    rem.s32 %r1, %slot, %cap;
    mul.lo.s32 %r2, %r1, 64;  // sizeof(Task) = 64
    cvt.u64.u32 %r3, %r2;
    add.u64 %task_addr, %tasks, %r3;

    // Advance head
    add.s32 %r4, %h, 1;
    st.global.b32 [%head_p], %r4;
    mov.b32 %has_work, 1;

WAIT_SYNC:
    bar.sync 0;

    // If no work, sleep and retry
    setp.eq.b32 %p_work, %has_work, 0;
    @%p_work bra SLEEP;

    // TODO: dispatch through function pointer table based on task.op
    // For now, execute inline add: out0[i] = in0[i] + in1[i]
    // This proves the persistent kernel works; JIT operators replace this.
    {
        .reg .b64 %numel, %in0, %in1, %out0;
        .reg .b64 %i, %stride, %off;
        .reg .f32 %a, %b, %c;
        .reg .pred %p_done;

        ld.global.u64 %numel, [%task_addr + 8];
        ld.global.u64 %in0,   [%task_addr + 16];
        ld.global.u64 %in1,   [%task_addr + 24];
        ld.global.u64 %out0,  [%task_addr + 32];

        mov.u32 %r10, %tid.x;
        cvt.u64.u32 %i, %r10;
        mov.u32 %r11, %ntid.x;
        cvt.u64.u32 %stride, %r11;

    EW_LOOP:
        setp.ge.u64 %p_done, %i, %numel;
        @%p_done bra EW_DONE;

        shl.b64 %off, %i, 2;
        add.u64 %r20, %in0, %off;
        ld.global.f32 %a, [%r20];
        add.u64 %r21, %in1, %off;
        ld.global.f32 %b, [%r21];
        add.f32 %c, %a, %b;
        add.u64 %r22, %out0, %off;
        st.global.f32 [%r22], %c;
        add.u64 %i, %i, %stride;
        bra EW_LOOP;
    EW_DONE:
    }

    bar.sync 0;

    // Leader: signal completion
    @!%p_leader bra POLL_LOOP;
    membar.sys;
    atom.global.add.u64 %r30, [%proc_p], %one64;
    bra POLL_LOOP;

SLEEP:
    // nanosleep not available in PTX directly, use empty loop
    // In sm_70+, we'd use nanosleep instruction via inline asm
    bra POLL_LOOP;

EXIT:
    ret;
}
"#;

/// GPU-side persistent kernel state.
pub struct GpuPersistentKernel {
    /// CUmodule handle
    pub module: cuda_types::cuda::CUmodule,
    /// CUfunction for persistent_worker
    pub worker_func: cuda_types::cuda::CUfunction,
    /// Managed memory: task ring buffer
    pub tasks_ptr: u64,     // device ptr to Task[capacity]
    pub capacity: u32,
    /// Control pointers — device-side (passed to kernel)
    pub head_ptr: u64,
    pub tail_ptr: u64,
    pub quit_ptr: u64,
    pub processed_ptr: u64,
    /// Control pointers — host-side (for CPU read/write)
    pub head_host: u64,
    pub tail_host: u64,
    pub quit_host: u64,
    pub processed_host: u64,
    /// Host-side shadow of processed counter
    pub submitted: AtomicU64,
    /// Stream the kernel runs on
    pub stream: u64,
    /// Whether kernel is launched
    pub running: bool,
    /// Number of blocks (SMs)
    pub num_blocks: u32,
    pub threads_per_block: u32,
}

/// Size of our Task struct in bytes (must match PTX layout)
const TASK_SIZE: usize = 64;

impl GpuPersistentKernel {
    /// Initialize: compile PTX, allocate managed memory, launch persistent kernel.
    pub fn init(device_id: i32, capacity: u32) -> Result<Self, String> {
        let _ = nvidia_runtime_sys::init();

        // Ensure CUDA is initialized and we have a context
        let r = nvidia_runtime_sys::cuInit(0);
        if r != 0 {
            return Err(format!("cuInit failed: {}", r));
        }

        // Retain primary context for this device
        let mut ctx = cuda_types::cuda::CUcontext(ptr::null_mut());
        let r = nvidia_runtime_sys::cuDevicePrimaryCtxRetain(&mut ctx, device_id);
        if r != 0 {
            return Err(format!("cuDevicePrimaryCtxRetain failed: {}", r));
        }
        let r = nvidia_runtime_sys::cuCtxSetCurrent(ctx);
        if r != 0 {
            return Err(format!("cuCtxSetCurrent failed: {}", r));
        }

        eprintln!("[Concordia:GPU] Initializing persistent kernel on device {}", device_id);

        // 1. Compile CUDA C source to PTX via nvcc, then load
        let ptx = Self::compile_cuda_to_ptx(PERSISTENT_KERNEL_CUDA)?;

        let ptx_cstr = CString::new(ptx.as_bytes())
            .map_err(|e| format!("CString error: {}", e))?;

        // Use REAL CUDA functions (not our shim) to avoid interception
        let funcs = nvidia_runtime_sys::get_cuda_funcs()
            .ok_or("CUDA functions not loaded")?;
        let real_module_load = funcs.cuModuleLoadData
            .ok_or("cuModuleLoadData not loaded")?;
        let real_get_func = funcs.cuModuleGetFunction
            .ok_or("cuModuleGetFunction not loaded")?;

        let mut module = cuda_types::cuda::CUmodule(ptr::null_mut());
        let r = unsafe { real_module_load(&mut module, ptx_cstr.as_ptr() as *const c_void) };
        match r {
            Ok(()) => {}
            Err(e) => return Err(format!("cuModuleLoadData failed: {:?} (PTX {} bytes)", e, ptx.len())),
        }
        eprintln!("[Concordia:GPU] Module loaded: {:?} ({} bytes PTX)", module.0, ptx.len());

        // 2. Get persistent_worker function
        let func_name = CString::new("persistent_worker").unwrap();
        let mut worker_func = cuda_types::cuda::CUfunction(ptr::null_mut());
        let r = unsafe { real_get_func(&mut worker_func, module, func_name.as_ptr()) };
        match r {
            Ok(()) => {}
            Err(e) => return Err(format!("cuModuleGetFunction failed: {:?}", e)),
        }
        eprintln!("[Concordia:GPU] persistent_worker function: {:?}", worker_func.0);

        // 3. Allocate HOST-MAPPED memory for task ring buffer too
        // Must be host-mapped (not managed) to avoid page migration deadlock
        // with the persistent kernel.
        let tasks_bytes = (capacity as usize) * TASK_SIZE;
        let mut tasks_host: *mut c_void = ptr::null_mut();
        let r = nvidia_runtime_sys::cuMemAllocHost_v2(&mut tasks_host, tasks_bytes);
        if r != 0 {
            return Err(format!("cuMemAllocHost for tasks failed: {}", r));
        }
        unsafe { std::ptr::write_bytes(tasks_host as *mut u8, 0, tasks_bytes); }
        let tasks_ptr = tasks_host as u64; // UVA: same address for host and device

        // 4. Allocate HOST-MAPPED (pinned) memory for control variables
        // Host-mapped = lives in host DRAM, GPU accesses via PCIe.
        // No page migration needed — critical for persistent kernel that holds GPU.
        let mut ctrl_host: *mut c_void = ptr::null_mut();
        // Need 4 ints (head, tail, quit) + 1 u64 (processed) = 24 bytes
        let ctrl_size = 32usize; // padded
        let r = nvidia_runtime_sys::cuMemAllocHost_v2(&mut ctrl_host, ctrl_size);
        if r != 0 {
            return Err(format!("cuMemAllocHost for ctrl failed: {}", r));
        }
        unsafe { std::ptr::write_bytes(ctrl_host as *mut u8, 0, ctrl_size); }

        // Get DEVICE-SIDE alias of host-mapped memory
        // The GPU kernel needs device pointers to access host memory via PCIe
        let funcs = nvidia_runtime_sys::get_cuda_funcs()
            .ok_or("CUDA not initialized")?;

        // Use cuMemHostGetDevicePointer via dlsym
        type CuMemHostGetDevicePointerFn = unsafe extern "C" fn(
            *mut cuda_types::cuda::CUdeviceptr, *mut c_void, u32,
        ) -> cuda_types::cuda::CUresult;
        let get_dev_ptr: CuMemHostGetDevicePointerFn = unsafe {
            let sym = libc::dlsym(
                funcs.lib_handle.0,
                b"cuMemHostGetDevicePointer_v2\0".as_ptr() as *const _,
            );
            if sym.is_null() {
                return Err("cuMemHostGetDevicePointer not found".to_string());
            }
            std::mem::transmute(sym)
        };

        let mut ctrl_dev = cuda_types::cuda::CUdeviceptr_v2(ptr::null_mut());
        let r = unsafe { get_dev_ptr(&mut ctrl_dev, ctrl_host, 0) };
        match r {
            Ok(()) => {}
            Err(e) => return Err(format!("cuMemHostGetDevicePointer failed: {:?}", e)),
        }
        let ctrl_dev_base = ctrl_dev.0 as u64;
        eprintln!("[Concordia:GPU] Host ctrl={:p}, device alias=0x{:x}", ctrl_host, ctrl_dev_base);

        // Layout: [head:i32, tail:i32, quit:i32, pad:i32, processed:u64]
        // Host pointers (for CPU read/write)
        let head_host = ctrl_host as u64;
        let tail_host = (ctrl_host as u64) + 4;
        let quit_host = (ctrl_host as u64) + 8;
        let processed_host = (ctrl_host as u64) + 16;
        // Device pointers (for kernel params)
        let head_ptr = ctrl_dev_base;
        let tail_ptr = ctrl_dev_base + 4;
        let quit_ptr = ctrl_dev_base + 8;
        let processed_ptr = ctrl_dev_base + 16;

        // 5. Get SM count for block count
        let mut sm_count: i32 = 0;
        // Use a reasonable default
        sm_count = 1; // Start with 1 block for testing
        eprintln!("[Concordia:GPU] Using {} block(s), 128 threads", sm_count);

        // 6. Create dedicated stream
        let mut stream: u64 = 0;
        let r = nvidia_runtime_sys::cuStreamCreate_ckpt(
            &mut stream as *mut u64 as *mut cuda_types::cuda::CUstream,
            1, // CU_STREAM_NON_BLOCKING
        );
        if r != 0 {
            return Err(format!("cuStreamCreate failed: {}", r));
        }

        let module_handle = module;
        let worker_func_handle = worker_func;

        let mut kernel = GpuPersistentKernel {
            module: module_handle,
            worker_func: worker_func_handle,
            tasks_ptr,
            capacity,
            head_ptr,       // device-side
            tail_ptr,       // device-side
            quit_ptr,       // device-side
            processed_ptr,  // device-side
            head_host,      // host-side
            tail_host,      // host-side
            quit_host,      // host-side
            processed_host, // host-side
            submitted: AtomicU64::new(0),
            stream,
            running: false,
            num_blocks: sm_count as u32,
            threads_per_block: 128,
        };

        // 7. Launch persistent kernel
        kernel.launch()?;

        Ok(kernel)
    }

    /// Compile CUDA C source to PTX using nvcc.
    fn compile_cuda_to_ptx(cuda_src: &str) -> Result<String, String> {
        use std::process::Command;

        let src_path = std::env::temp_dir().join("concordia_persistent.cu");
        let ptx_path = std::env::temp_dir().join("concordia_persistent.ptx");

        std::fs::write(&src_path, cuda_src)
            .map_err(|e| format!("write source: {}", e))?;

        // Detect GPU arch
        let arch = std::env::var("CONCORDIA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

        let output = Command::new("nvcc")
            .arg("--ptx")
            .arg("-o").arg(ptx_path.to_str().unwrap())
            .arg(src_path.to_str().unwrap())
            .arg(format!("-arch={}", arch))
            .arg("--std=c++17")
            .arg("-ccbin").arg("g++-13")
            .output()
            .map_err(|e| format!("nvcc: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("nvcc failed:\n{}", stderr));
        }

        std::fs::read_to_string(&ptx_path)
            .map_err(|e| format!("read PTX: {}", e))
    }

    fn alloc_managed(size: usize) -> Result<u64, String> {
        let mut dptr = cuda_types::cuda::CUdeviceptr_v2(ptr::null_mut());
        // CU_MEM_ATTACH_GLOBAL = 1
        let r = nvidia_runtime_sys::cuMemAllocManaged(
            &mut dptr as *mut cuda_types::cuda::CUdeviceptr,
            size, 1,
        );
        if r != 0 {
            // Fallback to regular device memory
            eprintln!("[Concordia:GPU] cuMemAllocManaged failed ({}), trying cuMemAlloc", r);
            let r2 = nvidia_runtime_sys::cuMemAlloc_v2(
                &mut dptr as *mut cuda_types::cuda::CUdeviceptr,
                size,
            );
            if r2 != 0 {
                return Err(format!("cuMemAlloc also failed: {} (size={})", r2, size));
            }
        }
        Ok(dptr.0 as u64)
    }

    fn memset(ptr: u64, val: u8, size: usize) {
        let dptr = cuda_types::cuda::CUdeviceptr_v2(ptr as *mut c_void);
        nvidia_runtime_sys::cuMemsetD8_v2(dptr, val, size);
    }

    /// Launch the persistent worker kernel.
    /// Uses the REAL cuLaunchKernel (not our shim) to avoid deadlock.
    fn launch(&mut self) -> Result<(), String> {
        // Params must match PTX: (u64, u32, u64, u64, u64, u64)
        let mut p0: u64 = self.tasks_ptr;
        let mut p1: u32 = self.capacity;
        let mut p2: u64 = self.head_ptr;
        let mut p3: u64 = self.tail_ptr;
        let mut p4: u64 = self.quit_ptr;
        let mut p5: u64 = self.processed_ptr;

        let mut param_ptrs: [*mut c_void; 6] = [
            &mut p0 as *mut u64 as *mut c_void,
            &mut p1 as *mut u32 as *mut c_void,
            &mut p2 as *mut u64 as *mut c_void,
            &mut p3 as *mut u64 as *mut c_void,
            &mut p4 as *mut u64 as *mut c_void,
            &mut p5 as *mut u64 as *mut c_void,
        ];

        // Get real cuLaunchKernel from nvidia_runtime_sys to bypass our shim
        let funcs = nvidia_runtime_sys::get_cuda_funcs()
            .ok_or("CUDA not initialized")?;
        let real_launch = funcs.cuLaunchKernel
            .ok_or("cuLaunchKernel not loaded")?;

        let func = self.worker_func;
        let stream = cuda_types::cuda::CUstream(self.stream as *mut _);

        let r = unsafe {
            real_launch(
                func,
                self.num_blocks, 1, 1,
                self.threads_per_block, 1, 1,
                0,
                stream,
                param_ptrs.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        match r {
            Ok(()) => {}
            Err(e) => return Err(format!("cuLaunchKernel failed: {:?}", e)),
        }

        self.running = true;
        eprintln!("[Concordia:GPU] Persistent kernel launched: {}x{} threads",
                  self.num_blocks, self.threads_per_block);
        Ok(())
    }

    /// Enqueue a task to the GPU ring buffer.
    /// The persistent kernel will pick it up and execute.
    pub fn enqueue_task(&self, op: i32, numel: i64, in0: u64, in1: u64, out0: u64) -> Result<u64, String> {
        if !self.running {
            return Err("Persistent kernel not running".to_string());
        }
        eprintln!("[Concordia:GPU] enqueue_task: op={} numel={} in0=0x{:x} in1=0x{:x} out0=0x{:x}",
                  op, numel, in0, in1, out0);

        // Host-mapped memory: read via HOST pointers (CPU-side)
        let tail = unsafe { std::ptr::read_volatile(self.tail_host as *const i32) };
        let head = unsafe { std::ptr::read_volatile(self.head_host as *const i32) };

        // Check if full
        if (tail - head) >= self.capacity as i32 {
            return Err("Ring buffer full".to_string());
        }

        // Write task to ring buffer slot via cuMemcpyHtoD
        let slot = (tail as u32) % self.capacity;
        let task_dev_addr = self.tasks_ptr + (slot as u64) * (TASK_SIZE as u64);

        // Build task in host memory first
        let mut task_buf = [0u8; TASK_SIZE];
        unsafe {
            *(task_buf.as_mut_ptr().add(0) as *mut i32) = op;
            *(task_buf.as_mut_ptr().add(4) as *mut i32) = 0; // flags
            *(task_buf.as_mut_ptr().add(8) as *mut i64) = numel;
            *(task_buf.as_mut_ptr().add(16) as *mut u64) = in0;
            *(task_buf.as_mut_ptr().add(24) as *mut u64) = in1;
            *(task_buf.as_mut_ptr().add(32) as *mut u64) = out0;
        }

        // Managed memory: direct host write (visible to GPU via coherence)
        unsafe {
            std::ptr::copy_nonoverlapping(
                task_buf.as_ptr(), task_dev_addr as *mut u8, TASK_SIZE);
            // Write tail AFTER task data (store-release ordering)
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(self.tail_host as *mut i32, tail + 1);
        }

        let seq = self.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Wait for all submitted tasks to complete (poll processed counter).
    pub fn sync(&self) -> u64 {
        let target = self.submitted.load(Ordering::Acquire);
        let mut spins = 0u64;
        loop {
            // processed_ptr is host-mapped (pinned) memory, safe to read directly
            let processed = unsafe { std::ptr::read_volatile(self.processed_host as *const u64) };
            if processed >= target {
                return processed;
            }
            spins += 1;
            if spins > 100_000_000 {
                eprintln!("[Concordia:GPU] sync timeout: target={} processed={}", target, processed);
                return processed;
            }
            std::hint::spin_loop();
        }
    }

    /// Signal the persistent kernel to stop and wait for it.
    pub fn shutdown(&mut self) {
        if !self.running { return; }

        // Set quit via HOST pointer (host-mapped memory, GPU sees via PCIe)
        unsafe { std::ptr::write_volatile(self.quit_host as *mut i32, 1); }

        // Sync stream to wait for kernel to exit
        nvidia_runtime_sys::cuStreamSynchronize_ckpt(
            cuda_types::cuda::CUstream(self.stream as *mut _)
        );

        self.running = false;
        let processed = unsafe { std::ptr::read_volatile(self.processed_host as *const u64) };
        eprintln!("[Concordia:GPU] Persistent kernel stopped. Processed: {}", processed);
    }
}

impl Drop for GpuPersistentKernel {
    fn drop(&mut self) {
        self.shutdown();
        // Free managed memory (in production; skip for now to avoid double-free)
    }
}

/// C-callable: initialize the GPU persistent kernel.
#[no_mangle]
pub extern "C" fn concordia_gpu_init(device_id: i32, capacity: u32) -> i64 {
    match GpuPersistentKernel::init(device_id, capacity) {
        Ok(kernel) => {
            // Store in global state
            let handle = Box::into_raw(Box::new(kernel)) as i64;
            eprintln!("[Concordia:GPU] Init OK, handle=0x{:x}", handle);
            handle
        }
        Err(e) => {
            eprintln!("[Concordia:GPU] Init FAILED: {}", e);
            -1
        }
    }
}

/// C-callable: enqueue a task (add: out = in0 + in1).
#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_enqueue(
    handle: i64, op: i32, numel: i64, in0: u64, in1: u64, out0: u64,
) -> i64 {
    let kernel = &*(handle as *const GpuPersistentKernel);
    match kernel.enqueue_task(op, numel, in0, in1, out0) {
        Ok(seq) => seq as i64,
        Err(_) => -1,
    }
}

/// C-callable: wait for all enqueued tasks to finish.
#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_sync(handle: i64) -> u64 {
    let kernel = &*(handle as *const GpuPersistentKernel);
    kernel.sync()
}

/// C-callable: shutdown the persistent kernel.
#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_shutdown(handle: i64) {
    let kernel = &mut *(handle as *mut GpuPersistentKernel);
    kernel.shutdown();
}
