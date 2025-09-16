[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/SxbqA3z2)

# hetGPU

hetGPU is a drop-in replacement for CUDA on non-NVIDIA GPU. hetGPU allows to run unmodified CUDA applications using non-NVIDIA GPUs with near-native performance.

hetGPU is work in progress. Follow development here and say hi on [Discord](https://discord.gg/SxbqA3z2).


## Usage
**Warning**: hetGPU is under heavy development (see news [here](https://zett.asplod.dev/)). Instructions below might not work.

### Windows
You should have the most recent ROCm  installed.
Run your application like this:
```
<hetGPU_DIRECTORY>\hetGPU_with.exe -- <APPLICATION> <APPLICATIONS_ARGUMENTS>
```

### Linux

Run your application like this:
```
LD_LIBRARY_PATH=<hetGPU_DIRECTORY> <APPLICATION> <APPLICATIONS_ARGUMENTS>
```

### MacOS

Not supported

## Building
**Warning**: hetGPU is under heavy development (see news [here](https://vosen.github.io/hetGPU/blog/hetGPUs-third-life/)). Instructions below might not work.

_Note_: This repo has submodules. Make sure to recurse submodules when cloning this repo, e.g.: `git clone --recursive https://github.com/vosen/hetGPU.git`

Dependency: 

```bash
# oneapi
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
apt install level-zero level-zero-dev

# cuda
apt install nvidia-cuda-toolkit
```

You should have a relatively recent version of Rust installed, then you just do:

```
cargo build --release
```
in the main directory of the project.  

### Caveats: LLVMTarget not built by default

```
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-Wl,--version-script=/tmp/rustctsJxAq/list" "-Wl,--no-undefined-version" "-m64" "/tmp/rustctsJxAq/symbols.o" "<89 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/xx/hetGPU/target/debug/deps/{libptx-7bdaff80a2317f98.rlib,libwhich-f30bb1abebdc81fc.rlib,libhome-7e8a5bcbbae9393a.rlib,libeither-64643123df8b89ff.rlib,librustix-af06f22e44d470fe.rlib,liblinux_raw_sys-3594dde066d7bd9d.rlib,libtempfile-95c2d4d89752ec6d.rlib,libgetrandom-b0234ab360848404.rlib,libfastrand-e6b7bf4f5223db08.rlib,libonce_cell-3762ade5b54492ce.rlib,librustix-7214c2022c62c874.rlib,libbitflags-df5e13556f379dca.rlib,liblinux_raw_sys-93ad30317744a679.rlib,libserde_json-96cad77e399e48a3.rlib,libitoa-e93891d541c4924d.rlib,libryu-89b8058e107f9d11.rlib,libregex-cc941c181af0db7a.rlib,libregex_automata-bda824cb2424702e.rlib,libaho_corasick-084dd9c39f7ac0ae.rlib,libmemchr-098566f0fddadd5e.rlib,libregex_syntax-c89bdbd6e66e0c64.rlib,libstrum-b548b1a93b339f7b.rlib,libquick_error-20ff717f463892e5.rlib,libptx_parser-1e5967146d4cc314.rlib,libthiserror-182054611c23cf3e.rlib,libbitflags-fdc24f4688603348.rlib,libwinnow-42c14bebbe4544ba.rlib,librustc_hash-50d45f4cee64a708.rlib,liblogos-b3726b5a3afef72d.rlib,libderive_more-9910237a94e5634b.rlib,libllvm_hetGPU-be4f1e966c7c00b7.rlib,libllvm_sys-dec2a59b25406277.rlib,libserde-50cedc28066d9a15.rlib,librand-7f4d8b7fe7531938.rlib,librand_chacha-376b75692958a1cc.rlib,libppv_lite86-a6f6214d46654bf2.rlib,libzerocopy-e3604abd9a947746.rlib,librand_core-82ab10e7dd7b6353.rlib,libgetrandom-cc67d104b46bae27.rlib,liblibc-83ba1c66a8baec75.rlib,libcfg_if-e3a2a86c3f7f7605.rlib,libbase64-abde09cee03ffba5.rlib,librustc_hash-bb091058ca505f29.rlib,libcuda_types-a88f0ad52e7d6047.rlib,libze_runtime_sys-56ce8c843283f7fa.rlib}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,libcfg_if-*,liblibc-*,liballoc-*,librustc_std_workspace_core-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lLLVMBitWriter" "-lLLVMAnalysis" "-lLLVMProfileData" "-lLLVMSymbolize" "-lLLVMDebugInfoBTF" "-lLLVMDebugInfoPDB" "-lLLVMDebugInfoMSF" "-lLLVMDebugInfoCodeView" "-lLLVMDebugInfoDWARF" "-lLLVMObject" "-lLLVMTextAPI" "-lLLVMMCParser" "-lLLVMIRReader" "-lLLVMAsmParser" "-lLLVMMC" "-lLLVMBitReader" "-lLLVMCore" "-lLLVMRemarks" "-lLLVMBitstreamReader" "-lLLVMBinaryFormat" "-lLLVMTargetParser" "-lLLVMSupport" "-lLLVMDemangle" "-lLLVMTarget" "-lstdc++" "-lze_loader" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustctsJxAq/raw-dylibs" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/xx/hetGPU/target/debug/build/ze_runtime_sys-1222225a9a9570c5/out" "-L" "/xx/hetGPU/target/debug/build/lz4-sys-d3a5e3b6c2386e54/out" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/lib" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/lib/../../../../../../../ext/llvm-project/build/lib" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out" "-L" "/xx/hetGPU/target/debug/build/tt_runtime_sys-2582b52c9d1e3544/out" "-L" "/opt/rocm/lib/" "-L" "/opt/rocm/lib/" "-L" "/usr/lib/x86_64-linux-gnu/" "-L" "lib" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/xx/hetGPU/target/debug/deps/libnvcuda.so" "-Wl,--gc-sections" "-shared" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: /usr/bin/ld: cannot find -lLLVMTarget
```
fix:
```
pushd /xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/
ninja LLVMTarget
popd
```
### Linux

If you are building on Linux you must also symlink (or rename) the hetGPU output binaries after hetGPU build finishes:
```
ln -s libnvcuda.so target/release/libcuda.so
ln -s libnvcuda.so target/release/libcuda.so.1
ln -s libnvml.so target/release/libnvidia-ml.so
```

## Contributing

If you want to develop hetGPU itself, read [CONTRIBUTING.md](CONTRIBUTING.md), it contains instructions how to set up dependencies and run tests


## License

This software is dual-licensed under either the Apache 2.0 license or the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) or [LICENSE-MIT](LICENSE-MIT) for details
