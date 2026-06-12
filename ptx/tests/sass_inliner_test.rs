//! Integration tests for SASS to LLVM IR inlining
//!
//! These tests verify the complete flow of parsing SASS and generating LLVM IR.

use std::io::Write;
use std::process::{Command, Stdio};

// Sample cuobjdump -sass output for testing
const SAMPLE_CUOBJDUMP_OUTPUT: &str = r#"
Fatbin elf code:
================
arch = sm_86
code version = [1,7]
host = linux
compile_size = 64bit

        code for sm_86
                Function : _Z9vectorAddPKfS0_Pfi
        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM86 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM86)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   S2R R2, SR_CTAID.X ;
        /*0030*/                   IMAD R2, R2, c[0x0][0x0], R0 ;
        /*0040*/                   ISETP.GE.AND P0, PT, R2, c[0x0][0x170], PT ;
        /*0050*/              @P0  EXIT ;
        /*0060*/                   IMAD.WIDE R4, R2, 0x4, c[0x0][0x160] ;
        /*0070*/                   IMAD.WIDE R6, R2, 0x4, c[0x0][0x168] ;
        /*0080*/                   LDG.E.SYS R4, [R4] ;
        /*0090*/                   LDG.E.SYS R6, [R6] ;
        /*00a0*/                   IMAD.WIDE R2, R2, 0x4, c[0x0][0x178] ;
        /*00b0*/                   FADD R4, R4, R6 ;
        /*00c0*/                   STG.E.SYS [R2], R4 ;
        /*00d0*/                   EXIT ;
"#;

const SIMPLE_KERNEL_OUTPUT: &str = r#"
Function : simple_add
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   IADD R0, R1, R2 ;
        /*0020*/                   FADD R3, R4, R5 ;
        /*0030*/                   EXIT ;
"#;

fn get_sass_inliner_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("target");
    path.push("debug");
    path.push("sass_inliner");
    path
}

#[test]
fn test_sass_inliner_help() {
    let output = Command::new(get_sass_inliner_path())
        .arg("--help")
        .output()
        .expect("Failed to run sass_inliner");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Check for key help text components
    assert!(
        stdout.contains("SASS") || stdout.contains("sass"),
        "Expected SASS in help text"
    );
    assert!(
        stdout.contains("--strategy") || stdout.contains("strategy"),
        "Expected strategy option"
    );
    assert!(
        stdout.contains("--stdin") || stdout.contains("stdin"),
        "Expected stdin option"
    );
}

#[test]
fn test_sass_inliner_stdin_metadata_only() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("metadata-only")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check that LLVM IR was generated
    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
    assert!(
        stdout.contains("_sass_simple_add"),
        "Expected kernel function name"
    );
}

#[test]
fn test_sass_inliner_stdin_ptx_reconstruction() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("ptx-reconstruction")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_stdin_inline_assembly() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("inline-asm")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_stdin_hybrid() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("hybrid")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_vector_add_kernel() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("metadata-only")
        .arg("--verbose")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SAMPLE_CUOBJDUMP_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);

    // Check LLVM IR structure
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
    assert!(stdout.contains("_sass_"), "Expected SASS kernel prefix");

    // Check verbose output
    assert!(
        stderr.contains("Parsed") || stderr.contains("instruction"),
        "Expected verbose output"
    );
}

#[test]
fn test_sass_inliner_with_dump_sass() {
    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--dump-sass")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);

    // Check that SASS instructions were dumped to stdout
    assert!(
        stdout.contains("MOV")
            || stdout.contains("IADD")
            || stdout.contains("FADD")
            || stdout.contains("SASS"),
        "Expected SASS instruction dump in stdout, got: {}",
        stdout
    );
}

#[test]
fn test_sass_inliner_memory_operations() {
    let memory_kernel = r#"
Function : memory_test
        /*0000*/                   LDG.E.SYS R0, [R2] ;
        /*0010*/                   LDS.U R1, [R3] ;
        /*0020*/                   LDC R4, c[0x0][0x10] ;
        /*0030*/                   STG.E.SYS [R5], R0 ;
        /*0040*/                   STS [R6], R1 ;
        /*0050*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("hybrid")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(memory_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_arithmetic_operations() {
    let arith_kernel = r#"
Function : arith_test
        /*0000*/                   IADD R0, R1, R2 ;
        /*0010*/                   IMAD R3, R4, R5, R6 ;
        /*0020*/                   FADD R7, R8, R9 ;
        /*0030*/                   FFMA R10, R11, R12, R13 ;
        /*0040*/                   FMUL R14, R15, R16 ;
        /*0050*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("ptx-reconstruction")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(arith_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);

    // PTX reconstruction should generate actual LLVM operations
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
    // Check for various IR operations that could be generated
    assert!(
        stdout.contains("add")
            || stdout.contains("fadd")
            || stdout.contains("mul")
            || stdout.contains("sitofp")
            || stdout.contains("call")
            || stdout.contains("ret"),
        "Expected LLVM operations in output, got: {}",
        &stdout[..std::cmp::min(500, stdout.len())]
    );
}

#[test]
fn test_sass_inliner_control_flow() {
    let control_kernel = r#"
Function : control_test
        /*0000*/                   MOV R0, 0x0 ;
        /*0010*/                   ISETP.LT.AND P0, PT, R0, R1, PT ;
        /*0020*/              @P0  BRA 0x50 ;
        /*0030*/                   IADD R2, R2, 0x1 ;
        /*0040*/                   BRA 0x10 ;
        /*0050*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("metadata-only")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(control_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_synchronization() {
    let sync_kernel = r#"
Function : sync_test
        /*0000*/                   BAR.SYNC 0x0 ;
        /*0010*/                   MEMBAR.GL ;
        /*0020*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("inline-asm")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(sync_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_special_registers() {
    let special_kernel = r#"
Function : special_test
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   S2R R1, SR_TID.Y ;
        /*0020*/                   S2R R2, SR_CTAID.X ;
        /*0030*/                   S2R R3, SR_LANEID ;
        /*0040*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("metadata-only")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(special_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(
        stdout.contains("define"),
        "Expected LLVM IR function definition"
    );
}

#[test]
fn test_sass_inliner_output_file() {
    use std::fs;
    use std::path::Path;

    let output_path = "/tmp/sass_inliner_test_output.ll";

    // Clean up if exists
    let _ = fs::remove_file(output_path);

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--strategy")
        .arg("metadata-only")
        .arg("--output")
        .arg(output_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(SIMPLE_KERNEL_OUTPUT.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("stderr: {}", stderr);
    }
    assert!(output.status.success(), "sass_inliner failed");

    // Check output file was created
    assert!(Path::new(output_path).exists(), "Output file should exist");

    let content = fs::read_to_string(output_path).expect("Failed to read output");
    assert!(
        content.contains("define"),
        "Output file should contain LLVM IR"
    );

    // Clean up
    let _ = fs::remove_file(output_path);
}
