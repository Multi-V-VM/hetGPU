"""
Site customize hook to help PyTorch run with hetGPU's virtual CUDA driver.

When LD_PRELOAD points to hetGPU's libcuda, PyTorch's CUDA lazy init may skip
the native _cuda_init() checks (by virtue of a patch) which means
torch.cuda._get_device_properties may never be defined. That causes a
NameError during initialization. This hook defines a minimal fallback for
_get_device_properties when hetGPU is detected.

To activate automatically, ensure this directory is on PYTHONPATH. For example:

  LD_PRELOAD=/root/hetGPU/target/release/libcuda.so.1 \
  PYTHONPATH=/root/hetGPU:$PYTHONPATH \
  python your_script.py
"""
from __future__ import annotations

import os
import types
import warnings


def _is_hetgpu_active() -> bool:
    preload = os.getenv("LD_PRELOAD", "")
    libpath = os.getenv("LD_LIBRARY_PATH", "")
    # Look for our libcuda and/or project marker in env paths
    hints = ("hetGPU", "libcuda.so", "libnvcuda.so")
    return any(h in preload for h in hints) or any(h in libpath for h in hints)


def _install_torch_cuda_fallbacks() -> None:
    try:
        import torch  # type: ignore
        import torch.cuda as cuda  # type: ignore
    except Exception:
        return

    # Only apply on hetGPU. Even if PyTorch was built without CUDA, we still
    # provide minimal fallbacks so higher-level helpers don't crash.
    if not _is_hetgpu_active():
        return

    # Define a simple device properties object with common fields
    class HetGPUDeviceProps:
        def __init__(self, index: int = 0):
            # Conservative, GPU-like defaults
            self.name = "Virtual GPU (TMatmul)"
            self.major = 8
            self.minor = 0
            # 8 GB by default
            self.total_memory = 8 * 1024**3
            self.multi_processor_count = 64
            self.max_threads_per_multiprocessor = 2048
            self.max_threads_per_block = 1024
            self.warp_size = 32
            self.clock_rate = 1000000  # kHz
            self.memory_clock_rate = 1000000  # kHz
            self.memory_bus_width = 256
            self.l2_cache_size = 4 * 1024**2
            self.shared_memory_per_block = 48 * 1024
            self.shared_memory_per_multiprocessor = 100 * 1024
            self.multiGpuBoard = False
            self.is_multi_gpu_board = False
            self.uuid = "00000000-0000-0000-0000-000000000000"
            self.pci_device_id = 0
            self.pci_domain_id = 0
            self.pci_bus_id = 0
            self.pci_device = 0
            self.pci_function = 0
            self.index = index

        # Some callers may inspect dict-like view
        def __repr__(self) -> str:
            return f"<HetGPUDeviceProps name={self.name!r} cc={self.major}.{self.minor} mem={self.total_memory}>"

    # Fallback getter used by torch.cuda.get_device_properties
    def _hetgpu_get_device_properties(device: int) -> HetGPUDeviceProps:  # type: ignore[override]
        # Normalize device index
        try:
            # Try to reuse torch helper for normalization if available
            from torch.cuda import _get_device_index  # type: ignore

            device = _get_device_index(device, optional=True)  # type: ignore
            if device is None:
                device = 0
        except Exception:
            device = 0
        return HetGPUDeviceProps(device)

    # Install our device properties provider unconditionally to avoid sm_0 misreads
    cuda._get_device_properties = _hetgpu_get_device_properties  # type: ignore[attr-defined]
    # Warn once
    if not hasattr(cuda, "_hetgpu_warned_props"):
        warnings.warn("[hetGPU] Overriding torch.cuda._get_device_properties for virtual device")
        setattr(cuda, "_hetgpu_warned_props", True)

    # Also ensure public helpers work even if bindings are partial
    def _hetgpu_get_device_name(device=None) -> str:
        return cuda._get_device_properties(device).name  # type: ignore[attr-defined]

    def _hetgpu_get_device_capability(device=None):
        # Force 8.0 capability for virtual backend to avoid sm_0
        return (8, 0)

    if getattr(cuda.get_device_name, "__module__", "").startswith("torch.cuda"):
        # Wrap only if original exists; keep signature intact
        cuda.get_device_name = _hetgpu_get_device_name  # type: ignore[assignment]
    cuda.get_device_capability = _hetgpu_get_device_capability  # type: ignore[assignment]

    # Reduce CUDA JIT/NVRTC activity that can depend on arch strings
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "80")
    os.environ.setdefault("TORCH_JIT_DISABLE_NVFUSER", "1")
    os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("CUDA_DISABLE_PTX_JIT", "1")

    # Safe cumsum shim: route CUDA cumsum to CPU, then copy back.
    # This avoids CUB kernel launches on virtual devices.
    try:
        import torch

        _orig_tensor_cumsum = torch.Tensor.cumsum
        _orig_cumsum = getattr(torch, "cumsum", None)

        def _hetgpu_tensor_cumsum(self, dim=0, dtype=None):  # type: ignore[override]
            if getattr(self, "is_cuda", False):
                cpu = self.detach().cpu()
                if dtype is not None:
                    out = _orig_tensor_cumsum(cpu, dim, dtype=dtype)  # keyword-only dtype
                else:
                    out = _orig_tensor_cumsum(cpu, dim)
                return out.to(self.device)
            if dtype is not None:
                return _orig_tensor_cumsum(self, dim, dtype=dtype)
            return _orig_tensor_cumsum(self, dim)

        def _hetgpu_cumsum(input, dim=0, dtype=None):  # type: ignore[override]
            if getattr(input, "is_cuda", False):
                return _hetgpu_tensor_cumsum(input, dim, dtype)
            if _orig_cumsum is not None:
                if dtype is not None:
                    return _orig_cumsum(input, dim, dtype=dtype)
                return _orig_cumsum(input, dim)
            # Fallback to method form
            return _hetgpu_tensor_cumsum(input, dim, dtype)

        # Install shims idempotently
        torch.Tensor.cumsum = _hetgpu_tensor_cumsum  # type: ignore[assignment]
        if _orig_cumsum is not None:
            torch.cumsum = _hetgpu_cumsum  # type: ignore[assignment]
    except Exception:
        pass

    # Patch Triton device properties to avoid sm_0 and NVRTC/NVVM paths on virtual backend
    try:
        import types
        from triton.runtime import driver as _triton_driver  # type: ignore
        import triton.compiler as _triton_compiler  # type: ignore
        # Optionally swallow Triton launch errors on virtual backend
        try:
            import triton.backends.nvidia.driver as _nvd  # type: ignore

            def _hetgpu_wrap_launch(fn):
                def _wrapped(self, *args, **kwargs):
                    # Extract launch parameters
                    if len(args) >= 4:
                        gridX, gridY, gridZ, stream = args[0], args[1], args[2], args[3]
                        function = args[4] if len(args) > 4 else None

                        # Try to process through tmatmul if enabled
                        backend = _get_tmatmul_backend()
                        if backend and hasattr(self, '_kernel_name') and hasattr(self, '_ptx_code'):
                            try:
                                # Extract kernel arguments from remaining args
                                kernel_args = args[7:] if len(args) > 7 else []

                                result = backend.process_kernel_launch(
                                    self._ptx_code,
                                    self._kernel_name,
                                    (gridX, gridY, gridZ),
                                    (1, 1, 1),  # Block size not available here
                                    kernel_args
                                )

                                if result and result.get('simulated'):
                                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                                        print(f"[hetGPU] Kernel executed via cocotb simulation")
                                    # Return early - simulation handled execution
                                    return None
                            except Exception as e:
                                if os.getenv("HETGPU_DEBUG", "0") == "1":
                                    print(f"[hetGPU] TMatmul processing failed: {e}")

                    # Normal CUDA launch path
                    try:
                        return fn(self, *args, **kwargs)
                    except RuntimeError as e:
                        err_msg = str(e)
                        if "invalid resource handle" in err_msg:
                            # Check if we're in autotuning context (stack inspection)
                            import traceback
                            stack = traceback.format_stack()
                            in_autotuner = any("autotuner.py" in frame for frame in stack)

                            if in_autotuner:
                                # During autotuning, skip invalid kernels and let Triton try other configs
                                if os.getenv("HETGPU_DEBUG", "0") == "1":
                                    print(f"[hetGPU] Autotuner kernel failed with invalid handle, trying next config")
                                return None
                            else:
                                # Not in autotuning - this is a real error, propagate it
                                if os.getenv("HETGPU_DEBUG", "0") == "1":
                                    print(f"[hetGPU] Launch failed outside autotuning: {err_msg}")
                                raise
                        raise
                return _wrapped

            # Patch CudaLauncher.__call__ to catch invalid stream/resource errors
            for _cls_name in ("CudaLauncher", "Function", "CuFunction"):
                if hasattr(_nvd, _cls_name):
                    _cls = getattr(_nvd, _cls_name)
                    if hasattr(_cls, "__call__"):
                        _cls.__call__ = _hetgpu_wrap_launch(_cls.__call__)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Also wrap the higher-level JIT run() to catch any remaining driver exceptions
        try:
            import triton.runtime.jit as _triton_jit  # type: ignore

            def _hetgpu_wrap_jit_run(run_fn):
                def _wrapped(self, *args, **kwargs):
                    try:
                        return run_fn(self, *args, **kwargs)
                    except RuntimeError as e:
                        err_msg = str(e)
                        if "invalid resource handle" in err_msg:
                            # Check if we're in autotuning context
                            import traceback
                            stack = traceback.format_stack()
                            in_autotuner = any("autotuner.py" in frame for frame in stack)

                            if in_autotuner:
                                # During autotuning, skip and try next config
                                if os.getenv("HETGPU_DEBUG", "0") == "1":
                                    print(f"[hetGPU] JIT run failed in autotuner, trying next config")
                                return None
                            else:
                                # Real error outside autotuning
                                if os.getenv("HETGPU_DEBUG", "0") == "1":
                                    print(f"[hetGPU] JIT run failed outside autotuning: {err_msg}")
                                raise
                        raise
                return _wrapped

            if hasattr(_triton_jit, "JITFunction") and hasattr(_triton_jit.JITFunction, "run"):
                _triton_jit.JITFunction.run = _hetgpu_wrap_jit_run(_triton_jit.JITFunction.run)  # type: ignore[assignment]
        except Exception:
            pass

        def _hetgpu_triton_get_device_properties(device=None):
            # Conservative, GPU-like values
            return {
                "name": "Virtual GPU (hetGPU sm_80)",
                "sm": 80,
                "capability": (8, 0),
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_threads_per_sm": 2048,
                "multiprocessor_count": 64,
                "max_shared_mem": 100 * 1024,           # per SM
                "max_shared_mem_per_block": 48 * 1024,  # per block
            }

        # Ensure utils exists and patch get_device_properties
        utils = getattr(_triton_driver.active, "utils", None)
        if utils is None:
            utils = types.SimpleNamespace()
            setattr(_triton_driver.active, "utils", utils)
        utils.get_device_properties = _hetgpu_triton_get_device_properties  # type: ignore[attr-defined]
        # Caches for PTX by kernel name and last loaded module for symbol resolution
        if not hasattr(utils, "_hetgpu_ptx_by_name"):
            utils._hetgpu_ptx_by_name = {}
        if not hasattr(utils, "_hetgpu_last_module"):
            utils._hetgpu_last_module = None
        if not hasattr(utils, "_hetgpu_last_module"):
            utils._hetgpu_last_module = None

        # Stub load/load_binary to avoid decoding/compiling CUBIN/PTX on virtual backend
        import ctypes as _ct
        import gzip as _gzip
        import io as _io
        import tempfile as _tempfile
        import subprocess as _subprocess
        import shutil as _shutil

        # TMatmul + Cocotb Integration
        _tmatmul_backend = None
        _tmatmul_enabled = os.getenv("HETGPU_TMATMUL_COCOTB", "0") == "1"

        def _get_tmatmul_backend():
            """Lazy-load TMatmul backend"""
            global _tmatmul_backend
            if _tmatmul_backend is None and _tmatmul_enabled:
                try:
                    import tmatmul_cocotb_integration
                    _tmatmul_backend = tmatmul_cocotb_integration.get_backend()
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print("[hetGPU] TMatmul cocotb backend loaded")
                except Exception as e:
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] Could not load TMatmul backend: {e}")
            return _tmatmul_backend

        # Global CUDA initialization state
        _cuda_ctx_initialized = [False]

        def _ensure_cuda_context():
            """Ensure CUDA driver is initialized and has a valid context"""
            if _cuda_ctx_initialized[0]:
                return True
            try:
                # Check if PyTorch already initialized CUDA
                import torch
                if torch.cuda.is_available():
                    # Force PyTorch to initialize CUDA
                    _ = torch.zeros(1, device='cuda')
                    _cuda_ctx_initialized[0] = True
                    return True
            except Exception as e:
                if os.getenv("HETGPU_DEBUG", "0") == "1":
                    print(f"[hetGPU] Failed to ensure CUDA context: {e}")
            return False

        # JIT-compile/load via CUDA driver entry (cuModuleLoadData), which our
        # libcuda routes to the TMatmul pipeline for PTX on the virtual backend.
        def _hetgpu_triton_load_binary(*args, **kwargs):
            # Ensure CUDA context is initialized before loading modules
            _ensure_cuda_context()

            if os.getenv("HETGPU_DEBUG", "0") == "1":
                print(f"[hetGPU] load_binary called with name={kwargs.get('name')}, args len={len(args)}")

            # Accept flexible Triton signatures; try to locate the actual binary buffer
            kernel_bytes = kwargs.get("binary") or kwargs.get("kernel")
            name = kwargs.get("name")
            # Heuristics over positional args: prefer a bytes-like arg that looks like PTX or ELF
            def _looks_like_elf(b: bytes) -> bool:
                return isinstance(b, (bytes, bytearray)) and len(b) >= 4 and b[0] == 0x7F and b[1:4] == b'ELF'
            def _looks_like_ptx(b: bytes) -> bool:
                if not isinstance(b, (bytes, bytearray)):
                    return False
                head_txt = b[:128].decode("latin1", errors="ignore")
                return (".version" in head_txt) or (".entry" in head_txt) or (".target" in head_txt)
            # Try to discover kernel name from positional args
            if name is None and args and isinstance(args[0], str):
                name = args[0]
            if kernel_bytes is None and args:
                # Try to find a bytes-like arg that looks like ELF/PTX
                candidates = [a for a in args if isinstance(a, (bytes, bytearray))]
                pick = None
                for c in candidates:
                    if _looks_like_elf(c) or _looks_like_ptx(c):
                        pick = c
                        break
                # Fallback: if two args, assume (name, binary)
                if pick is None and len(args) >= 2 and isinstance(args[1], (bytes, bytearray)):
                    pick = args[1]
                # Last resort: if first arg is bytes, take it
                if pick is None and isinstance(args[0], (bytes, bytearray)):
                    pick = args[0]
                kernel_bytes = pick
            if kernel_bytes is None:
                return (0, 0, 0, 0, 1024)
            try:
                # Try to load hetGPU's libcuda (may be preloaded via LD_PRELOAD)
                lib = _ct.CDLL(None)  # Load from current process (picks up LD_PRELOAD)
                # Verify it has CUDA functions
                if not hasattr(lib, 'cuModuleLoadData'):
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] Preloaded lib missing cuModuleLoadData, trying libcuda.so.1")
                    lib = _ct.CDLL("libcuda.so.1")
                else:
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] Using preloaded libcuda with cuModuleLoadData")
            except OSError as e:
                if os.getenv("HETGPU_DEBUG", "0") == "1":
                    print(f"[hetGPU] Failed to load preloaded lib: {e}, trying libcuda.so.1")
                try:
                    lib = _ct.CDLL("libcuda.so.1")
                except OSError:
                    # No driver available; return placeholders
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] ERROR: Could not load libcuda.so.1")
                    return (0, 0, 0, 0, 1024)

            CUmodule = _ct.c_void_p
            try:
                cuModuleLoadData = lib.cuModuleLoadData
                cuModuleLoadData.argtypes = [_ct.POINTER(CUmodule), _ct.c_void_p]
                cuModuleLoadData.restype = _ct.c_int
                # For symbol resolution when only a name is provided
                CUfunction = _ct.c_void_p
                cuModuleGetFunction = lib.cuModuleGetFunction
                cuModuleGetFunction.argtypes = [_ct.POINTER(CUfunction), CUmodule, _ct.c_char_p]
                cuModuleGetFunction.restype = _ct.c_int
            except AttributeError:
                return (0, 0, 0, 0, 1024)

            # If we didn't get a binary buffer, try resolving a symbol from the last module
            if kernel_bytes is None:
                sym = name
                if sym is None:
                    for a in args:
                        if isinstance(a, str):
                            sym = a
                            break
                mod = getattr(utils, "_hetgpu_last_module", None)
                if sym and mod:
                    func = CUfunction()
                    try:
                        res = cuModuleGetFunction(_ct.byref(func), _ct.c_void_p(mod), sym.encode("utf-8"))
                        if res == 0:
                            return (int(mod or 0), int(func.value or 0), 0, 0, 1024)
                    except Exception:
                        pass
                return (0, 0, 0, 0, 1024)

            data = kernel_bytes
            # If gzipped PTX, decompress
            if isinstance(data, (bytes, bytearray)) and len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
                try:
                    data = _gzip.decompress(data)
                except Exception:
                    pass

            # Detect PTX text vs ELF/CUBIN; if ELF, try to extract PTX via cuobjdump
            is_ptx = False
            if isinstance(data, (bytes, bytearray)):
                head = data[:8]
                if _looks_like_elf(data):
                    # If we previously recorded PTX for this kernel name, prefer it
                    if name and isinstance(name, str):
                        cached_ptx = getattr(utils, "_hetgpu_ptx_by_name", {}).get(name)
                        if cached_ptx:
                            data = cached_ptx
                            is_ptx = True
                    # Attempt PTX extraction using cuobjdump if available
                    _cuobjdump = os.getenv("HETGPU_CUOBJDUMP") or _shutil.which("cuobjdump") or _shutil.which("/usr/local/cuda/bin/cuobjdump")
                    if _cuobjdump is not None:
                        try:
                            with _tempfile.NamedTemporaryFile(delete=True, suffix=".cubin") as tf:
                                tf.write(data)
                                tf.flush()
                                # Prefer '--dump-ptx' but fall back to '-ptx'
                                cmd = [_cuobjdump, "--dump-ptx", tf.name]
                                proc = _subprocess.run(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE)
                                if proc.returncode != 0:
                                    cmd = [_cuobjdump, "-ptx", tf.name]
                                    proc = _subprocess.run(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE)
                                if proc.returncode == 0 and proc.stdout:
                                    out = proc.stdout.decode("latin1", errors="ignore")
                                    # Heuristic: take from first '.version' to end
                                    start = out.find(".version")
                                    if start != -1:
                                        ptx_text = out[start:]
                                        data = ptx_text.encode("utf-8", errors="ignore")
                                        if name and isinstance(name, str):
                                            getattr(utils, "_hetgpu_ptx_by_name", {})[name] = data
                                        is_ptx = True
                        except Exception:
                            pass
                    # If cuobjdump failed, keep as ELF (will likely not load on virtual device)
                else:
                    head_txt = data[:128].decode("latin1", errors="ignore")
                    is_ptx = (".version" in head_txt) or (".entry" in head_txt) or (".target" in head_txt)
            elif isinstance(data, str):
                is_ptx = True
                data = data.encode("utf-8")
                if name and isinstance(name, str):
                    getattr(utils, "_hetgpu_ptx_by_name", {})[name] = data

            # If neither ELF nor PTX, don't pass to driver (e.g., got a symbol name)
            if not is_ptx and not _looks_like_elf(data):
                return (None, None, 0, 0, 1024)

            # Prepare buffer; PTX should be NUL-terminated
            if is_ptx:
                buf = _ct.create_string_buffer(data + (b"\0" if not data.endswith(b"\0") else b""))
                image_ptr = _ct.cast(buf, _ct.c_void_p)
            else:
                # Treat as raw binary (ELF/CUBIN)
                buf = ( _ct.c_ubyte * len(data) )(*data)
                image_ptr = _ct.cast(buf, _ct.c_void_p)

            mod = CUmodule()
            try:
                res = cuModuleLoadData(_ct.byref(mod), image_ptr)
                if os.getenv("HETGPU_DEBUG", "0") == "1":
                    print(f"[hetGPU] cuModuleLoadData result: {res}, module: {mod.value}, name: {name}, is_ptx: {is_ptx}")
                if res == 0 and mod.value:
                    utils._hetgpu_last_module = int(mod.value or 0)
                    # Try to resolve a function for the kernel name if provided
                    func_handle = _ct.c_void_p()
                    if name and isinstance(name, str):
                        try:
                            # Resolve to a valid CUfunction; if it fails, leave placeholder
                            func_res = cuModuleGetFunction(_ct.byref(func_handle), mod, name.encode("utf-8"))
                            if os.getenv("HETGPU_DEBUG", "0") == "1":
                                print(f"[hetGPU] cuModuleGetFunction({name}) result: {func_res}, function: {func_handle.value}")
                        except Exception as e:
                            if os.getenv("HETGPU_DEBUG", "0") == "1":
                                print(f"[hetGPU] cuModuleGetFunction({name}) exception: {e}")
                            pass
                    result = (int(mod.value or 0), int(func_handle.value or 0), 0, 0, 1024)
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] Returning handles: module={result[0]}, function={result[1]}, name={name}")
                    # Warn if handles are invalid
                    if result[0] == 0 or result[1] == 0:
                        if os.getenv("HETGPU_DEBUG", "0") == "1":
                            print(f"[hetGPU] WARNING: Invalid handles returned! This will cause launch failures")
                    return result
                else:
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] cuModuleLoadData failed with result {res}")
                return (0, 0, 0, 0, 1024)
            except Exception as e:
                if os.getenv("HETGPU_DEBUG", "0") == "1":
                    print(f"[hetGPU] cuModuleLoadData exception: {e}")
                return (0, 0, 0, 0, 1024)

        def _hetgpu_triton_load(src, name=None):
            # Load from PTX source string
            if isinstance(src, str):
                return _hetgpu_triton_load_binary(src.encode("utf-8"), name)
            if isinstance(src, (bytes, bytearray)):
                return _hetgpu_triton_load_binary(src, name)
            return (0, 0, 0, 0, 1024)

        utils.load_binary = _hetgpu_triton_load_binary  # type: ignore[attr-defined]
        utils.load = _hetgpu_triton_load  # type: ignore[attr-defined]

        # Note: We don't stub launch here - let the compiled launcher actually invoke
        # cuLaunchKernel, which will be routed through hetGPU's libcuda to the backend

        # Hook Triton's compile to capture PTX and map it by kernel name.
        try:
            _orig_triton_compile = getattr(_triton_compiler, "compile")

            def _extract_ptx_from_res(res):
                # Try common attributes/containers to locate PTX text
                # Returns bytes or None
                try:
                    asm = getattr(res, "asm", None)
                    # asm may be a dict mapping formats/archs
                    if isinstance(asm, dict):
                        # Look for any string value containing '.version'
                        for v in asm.values():
                            if isinstance(v, str) and ".version" in v:
                                return v.encode("utf-8", errors="ignore")
                            if isinstance(v, (list, tuple)):
                                for s in v:
                                    if isinstance(s, str) and ".version" in s:
                                        return s.encode("utf-8", errors="ignore")
                            if isinstance(v, dict):
                                for vv in v.values():
                                    if isinstance(vv, str) and ".version" in vv:
                                        return vv.encode("utf-8", errors="ignore")
                    if isinstance(asm, str) and ".version" in asm:
                        return asm.encode("utf-8", errors="ignore")
                    # Try direct PTX attributes
                    for attr in ("ptx", "ptx_ir", "ir_ptx"):
                        val = getattr(res, attr, None)
                        if isinstance(val, str) and ".version" in val:
                            return val.encode("utf-8", errors="ignore")
                        if isinstance(val, (bytes, bytearray)) and b".version" in val[:256]:
                            return bytes(val)
                except Exception:
                    return None
                return None

            def _extract_name_from_args_res(cargs, ckwargs, res):
                kname = ckwargs.get("name")
                if kname:
                    return kname
                # Try from first arg object (module/function)
                if cargs:
                    obj0 = cargs[0]
                    kname = getattr(obj0, "name", None)
                    if kname:
                        return kname
                # Try from result object
                kname = getattr(res, "name", None)
                if kname:
                    return kname
                return None

            def _hetgpu_triton_compile(*cargs, **ckwargs):
                res = _orig_triton_compile(*cargs, **ckwargs)
                try:
                    kname = _extract_name_from_args_res(cargs, ckwargs, res)
                    ptx_bytes = _extract_ptx_from_res(res)
                    if not ptx_bytes and hasattr(res, "_ptx"):
                        raw = getattr(res, "_ptx")
                        if isinstance(raw, str) and ".version" in raw:
                            ptx_bytes = raw.encode("utf-8", errors="ignore")
                    # If still no PTX, try pulling from any field that looks like PTX text
                    if not ptx_bytes:
                        for attr in dir(res):
                            try:
                                val = getattr(res, attr)
                                if isinstance(val, str) and ".version" in val:
                                    ptx_bytes = val.encode("utf-8", errors="ignore")
                                    break
                            except Exception:
                                continue
                    if kname and ptx_bytes:
                        try:
                            utils._hetgpu_ptx_by_name[kname] = ptx_bytes  # type: ignore[index]

                            # Also process through TMatmul if enabled
                            backend = _get_tmatmul_backend()
                            if backend:
                                try:
                                    asm_file = backend.compile_ptx_to_tmatmul(ptx_bytes, kname)
                                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                                        print(f"[hetGPU] Compiled {kname} → {asm_file}")
                                except Exception as e:
                                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                                        print(f"[hetGPU] TMatmul compilation failed for {kname}: {e}")
                        except Exception:
                            pass
                except Exception:
                    pass
                return res

            if callable(_orig_triton_compile):
                _triton_compiler.compile = _hetgpu_triton_compile  # type: ignore[assignment]
        except Exception:
            pass

        # Provide minimal stubs often queried
        if not hasattr(_triton_driver.active, "num_devices"):
            _triton_driver.active.num_devices = lambda: 1  # type: ignore[attr-defined]
        if not hasattr(_triton_driver.active, "get_current_device"):
            _triton_driver.active.get_current_device = lambda: 0  # type: ignore[attr-defined]

        # Get stream from PyTorch's CUDA context instead of creating our own
        if not hasattr(_triton_driver.active, "get_current_stream"):
            def _hetgpu_get_pytorch_stream(device=None):
                """Get the current CUDA stream from PyTorch"""
                try:
                    import torch
                    if device is None:
                        device = torch.cuda.current_device()
                    # Get PyTorch's current stream handle for this device
                    stream = torch.cuda.current_stream(device)
                    # Extract the raw CUDA stream pointer
                    # cuda_stream is a property that returns the integer stream handle
                    return stream.cuda_stream
                except Exception as e:
                    # Fall back to default/null stream (0) if PyTorch stream unavailable
                    if os.getenv("HETGPU_DEBUG", "0") == "1":
                        print(f"[hetGPU] Could not get PyTorch stream: {e}, using default stream")
                    return 0
            _triton_driver.active.get_current_stream = _hetgpu_get_pytorch_stream  # type: ignore[attr-defined]
    except Exception:
        pass

    # Patch mmfreellm swiglu to CPU fallback to avoid CUDA Jiterator usage on virtual backend
    if os.getenv("HETGPU_TMATMUL_COCOTB", "0").lower() in ("1", "true"):
        try:
            import importlib
            import torch
            import torch.nn.functional as F
            act_mod = importlib.import_module("mmfreellm.modules.activations")

            def _hetgpu_swiglu_fwd(x, y):
                # CPU fallback: x * silu(y)
                return x * F.silu(y)

            # Only override if the symbol exists
            if hasattr(act_mod, "swiglu_fwd"):
                setattr(act_mod, "swiglu_fwd", _hetgpu_swiglu_fwd)
        except Exception:
            pass

        # Intercept Torch CUDA Jiterator kernel launch: force-CUDA path for virtual backend,
        # with optional CPU fallback via HETGPU_JIT_CPU_FALLBACK=1
        try:
            import torch
            import torch.nn.functional as F

            if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_jiterator_compile_and_launch_kernel"):
                _orig_cuda_jit = torch._C._cuda_jiterator_compile_and_launch_kernel  # type: ignore[attr-defined]

                def _hetgpu_cuda_jiterator_compile_and_launch_kernel(kernel_src, kernel_name, num_outputs, tensors, scalars, *rest):  # type: ignore[override]
                    # CPU fallback (opt-in)
                    if os.getenv("HETGPU_JIT_CPU_FALLBACK", "0").lower() in ("1", "true"):
                        try:
                            outs = list(tensors)[: int(num_outputs) if isinstance(num_outputs, int) else int(num_outputs)]
                            ins = list(tensors)[int(num_outputs) if isinstance(num_outputs, int) else int(num_outputs) :]
                            if outs:
                                if len(ins) >= 2:
                                    x, y = ins[0], ins[1]
                                    out = outs[0]
                                    out.copy_((x.to(out.device)) * F.silu(y.to(out.device)))
                                    return None
                                elif len(ins) >= 1:
                                    outs[0].copy_(ins[0].to(outs[0].device))
                                    return None
                        except Exception:
                            pass
                    # Force tensors to CUDA for virtual backend to bypass CUDAGuard on CPU
                    if os.getenv("HETGPU_TMATMUL_COCOTB", "0").lower() in ("1", "true") and os.getenv("HETGPU_JIT_FORCE_CUDA", "1").lower() in ("1", "true"):
                        try:
                            nout = int(num_outputs) if isinstance(num_outputs, int) else int(num_outputs)
                            outs_orig = list(tensors)[:nout]
                            ins_orig = list(tensors)[nout:]
                            outs_cuda = [t.to("cuda") if hasattr(t, "to") else t for t in outs_orig]
                            ins_cuda = [t.to("cuda") if hasattr(t, "to") else t for t in ins_orig]
                            res = _orig_cuda_jit(kernel_src, kernel_name, num_outputs, tuple(outs_cuda + ins_cuda), scalars, *rest)
                            for i, oc in enumerate(outs_cuda):
                                try:
                                    outs_orig[i].copy_(oc.to(outs_orig[i].device))
                                except Exception:
                                    pass
                            return res
                        except Exception:
                            pass
                    return _orig_cuda_jit(kernel_src, kernel_name, num_outputs, tensors, scalars, *rest)

                torch._C._cuda_jiterator_compile_and_launch_kernel = _hetgpu_cuda_jiterator_compile_and_launch_kernel  # type: ignore[assignment]
        except Exception:
            pass

        # Also patch torch.cuda.jiterator.Jiterator.__call__ to force CUDA on virtual backend
        try:
            import importlib
            ji = importlib.import_module("torch.cuda.jiterator")
            if hasattr(ji, "Jiterator") and hasattr(ji.Jiterator, "__call__"):
                _orig_jit_call = ji.Jiterator.__call__

                def _hetgpu_jit_call(self, *args, **kwargs):
                    try:
                        nout = getattr(self, "num_outputs", None)
                        if nout is None:
                            nout = getattr(self, "_num_outputs", 1)
                        nout = int(nout)
                    except Exception:
                        nout = 1
                    # CPU fallback (opt-in)
                    if os.getenv("HETGPU_JIT_CPU_FALLBACK", "0").lower() in ("1", "true"):
                        try:
                            outs = list(args)[:nout]
                            ins = list(args)[nout:]
                            if outs and len(ins) >= 2:
                                x, y = ins[0], ins[1]
                                outs[0].copy_(x.to(outs[0].device) * F.silu(y.to(outs[0].device)))
                                return None
                            elif outs and len(ins) >= 1:
                                outs[0].copy_(ins[0].to(outs[0].device))
                                return None
                        except Exception:
                            pass
                    # Force tensors to CUDA on virtual backend
                    if os.getenv("HETGPU_TMATMUL_COCOTB", "0").lower() in ("1", "true") and os.getenv("HETGPU_JIT_FORCE_CUDA", "1").lower() in ("1", "true"):
                        try:
                            args = list(args)
                            outs = args[:nout]
                            ins = args[nout:]
                            outs_cuda = [t.to("cuda") if hasattr(t, "to") else t for t in outs]
                            ins_cuda = [t.to("cuda") if hasattr(t, "to") else t for t in ins]
                            res = _orig_jit_call(self, *(outs_cuda + ins_cuda), **kwargs)
                            for i, oc in enumerate(outs_cuda):
                                try:
                                    outs[i].copy_(oc.to(outs[i].device))
                                except Exception:
                                    pass
                            return res
                        except Exception:
                            pass
                    return _orig_jit_call(self, *args, **kwargs)

                ji.Jiterator.__call__ = _hetgpu_jit_call  # type: ignore[assignment]
        except Exception:
            pass

    # Optionally force CPU execution by making .cuda() and .to('cuda') no-ops
    # This helps avoid unallocated device memory with virtual backends.
    force_cpu = os.getenv("HETGPU_FORCE_CPU", "0").lower() in ("1", "true")
    if force_cpu:
        try:
            import torch
            import torch.nn as nn

            def _module_cuda(self, *args, **kwargs):
                return self  # no-op

            def _tensor_cuda(self, *args, **kwargs):
                return self  # no-op

            nn.Module.cuda = _module_cuda  # type: ignore[assignment]
            torch.Tensor.cuda = _tensor_cuda  # type: ignore[assignment]

            # Make .to('cuda') a no-op
            _orig_to = torch.Tensor.to

            def _tensor_to(self, *args, **kwargs):
                if len(args) > 0 and isinstance(args[0], (str,)) and str(args[0]).startswith("cuda"):
                    args = list(args)
                    args[0] = "cpu"
                    return _orig_to(self, *args, **kwargs)
                if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
                    kwargs = dict(kwargs)
                    kwargs["device"] = "cpu"
                    return _orig_to(self, *args, **kwargs)
                return _orig_to(self, *args, **kwargs)

            torch.Tensor.to = _tensor_to  # type: ignore[assignment]
        except Exception:
            pass

    # Predefine missing torchvision operators (e.g., torchvision::nms) to avoid
    # import-time failures when compiled C++ ops are unavailable.
    try:
        import torch
        from torch.library import Library

        # Define nms op if it's not already defined
        # Use torch internals to check existence idempotently
        def _has_schema(op: str) -> bool:
            try:
                return torch._C._dispatch_find_schema(op) is not None  # type: ignore[attr-defined]
            except Exception:
                return False

        if not _has_schema("torchvision::nms"):
            lib = Library("torchvision", "DEF")
            try:
                lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
            except Exception:
                pass  # may race with torchvision

        # CPU implementation in pure Python as a fallback
        def _nms_cpu(dets, scores, iou_threshold: float):
            import torch
            if dets.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=dets.device)
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            order = scores.argsort(descending=True)
            keep = []
            while int(order.numel()) > 0:
                i = int(order[0])
                keep.append(i)
                if int(order.numel()) == 1:
                    break
                rest = order[1:]
                xx1 = torch.maximum(x1[rest], x1[i])
                yy1 = torch.maximum(y1[rest], y1[i])
                xx2 = torch.minimum(x2[rest], x2[i])
                yy2 = torch.minimum(y2[rest], y2[i])
                w = (xx2 - xx1).clamp(min=0)
                h = (yy2 - yy1).clamp(min=0)
                inter = w * h
                union = areas[rest] + areas[i] - inter + 1e-8
                ovr = inter / union
                mask = ovr <= iou_threshold
                order = rest[mask]
            return torch.tensor(keep, dtype=torch.long, device=dets.device)

        # Only register CPU impl if none is present
        try:
            has_cpu = torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "CPU")  # type: ignore[attr-defined]
        except Exception:
            has_cpu = False
        if not has_cpu:
            try:
                from torch.library import impl as _impl
                _impl("torchvision::nms", "CPU")(_nms_cpu)
            except Exception:
                try:
                    from torch.library import register_impl
                    register_impl("torchvision::nms", "CPU")(_nms_cpu)  # type: ignore
                except Exception:
                    pass

        # Register a fake implementation for Meta tensors to satisfy import-time checks
        # Do NOT register a Fake (Meta) impl here to avoid duplicate registration
        # with torchvision's own meta registrations. We only ensure the schema
        # exists and provide a CPU fallback; torchvision will register its fake.
    except Exception:
        # If torch.library is not available, skip silently
        pass


# Eagerly install fallbacks so they are present before torch.cuda lazy init callbacks run
try:
    _install_torch_cuda_fallbacks()
except Exception as e:
    warnings.warn(f"[hetGPU] sitecustomize failed to install fallbacks: {e}")

# Install Triton compatibility patches for hetGPU
if _is_hetgpu_active():
    try:
        import sys

        # Use both the import-hook approach and direct patching
        # Import hook for modules loaded later
        import triton_hetgpu_patch

        # Direct hot-patch for modules that may already be loaded
        try:
            import triton_hotpatch

            # Try to apply patches immediately if triton is already loaded
            triton_hotpatch.apply_patches()

            # Also install deferred patcher in case triton loads later
            triton_hotpatch.install_deferred_patcher()

        except Exception as e:
            # If hotpatch fails, that's ok - the import hook will catch it
            pass

        print("[hetGPU sitecustomize] Triton patches installed")
    except ImportError as e:
        warnings.warn(f"[hetGPU] Could not import triton patches: {e}")
