//! Integration tests for gpu_rr (GPU Record and Replay)
//!
//! Tests the record, replay, analyze, and info commands.

use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Sample SASS text for testing (vectorAdd kernel)
const SAMPLE_SASS: &str = r#"
Function : _Z9vectorAddPKfS0_Pfi
	/*0000*/                   MOV R1, c[0x0][0x28] ;
	/*0010*/                   S2R R0, SR_TID.X ;
	/*0020*/                   MOV R2, c[0x0][0x168] ;
	/*0030*/                   IMAD.WIDE R2, R0, 0x4, c[0x0][0x160] ;
	/*0040*/                   LDG.E R2, [R2.64] ;
	/*0050*/                   IMAD.WIDE R4, R0, 0x4, c[0x0][0x168] ;
	/*0060*/                   LDG.E R3, [R4.64] ;
	/*0070*/                   FADD R2, R2, R3 ;
	/*0080*/                   IMAD.WIDE R4, R0, 0x4, c[0x0][0x170] ;
	/*0090*/                   STG.E [R4.64], R2 ;
	/*00a0*/                   EXIT ;
"#;

/// Sample SASS with branching for testing divergence
const SAMPLE_SASS_WITH_BRANCH: &str = r#"
Function : _Z10kernelTestPfPi
	/*0000*/                   MOV R1, c[0x0][0x28] ;
	/*0010*/                   S2R R0, SR_TID.X ;
	/*0020*/                   ISETP.LT.AND P0, PT, R0, 0x10, PT ;
	/*0030*/                   @P0 BRA 0x50 ;
	/*0040*/                   MOV R2, 0x0 ;
	/*0050*/                   MOV R3, 0x1 ;
	/*0060*/                   BAR.SYNC 0x0 ;
	/*0070*/                   EXIT ;
"#;

fn get_gpu_rr_path() -> String {
    // Find the target directory
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test executable name
    path.pop(); // Remove deps
    path.push("gpu_rr");
    path.to_string_lossy().to_string()
}

fn run_gpu_rr(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(get_gpu_rr_path())
        .args(args)
        .output()
        .expect("Failed to execute gpu_rr");

    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (success, stdout, stderr)
}

#[test]
fn test_gpu_rr_help() {
    let (success, stdout, _) = run_gpu_rr(&["--help"]);

    assert!(success);
    assert!(stdout.contains("GPU Record and Replay"));
    assert!(stdout.contains("record"));
    assert!(stdout.contains("replay"));
    assert!(stdout.contains("analyze"));
    assert!(stdout.contains("info"));
}

#[test]
fn test_gpu_rr_record() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    fs::write(&input_path, SAMPLE_SASS).unwrap();

    let (success, stdout, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);

    assert!(success, "Record command should succeed");
    assert!(output_path.exists(), "Recording file should be created");
    assert!(stdout.contains("Recording saved"));
    assert!(stdout.contains("Kernels: 1"));
    assert!(stdout.contains("Total instructions: 11"));
}

#[test]
fn test_gpu_rr_record_verbose() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    fs::write(&input_path, SAMPLE_SASS).unwrap();

    let (success, stdout, stderr) = run_gpu_rr(&[
        "-v",
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);

    assert!(success);
    // Verbose output goes to stderr
    assert!(stderr.contains("Recording from:") || stdout.contains("Recording from:"));
}

#[test]
fn test_gpu_rr_info() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // First, record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Then get info
    let (success, stdout, _) = run_gpu_rr(&["info", output_path.to_str().unwrap()]);

    assert!(success, "Info command should succeed");
    assert!(stdout.contains("GPU Recording Info"));
    assert!(stdout.contains("_Z9vectorAddPKfS0_Pfi"));
    assert!(stdout.contains("Kernels: 1"));
    assert!(stdout.contains("SM Version: sm_"));
}

#[test]
fn test_gpu_rr_analyze() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // First, record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Then analyze
    let (success, stdout, _) = run_gpu_rr(&["analyze", output_path.to_str().unwrap()]);

    assert!(success, "Analyze command should succeed");
    assert!(stdout.contains("GPU Recording Analysis Report"));
    assert!(stdout.contains("Summary:"));
    assert!(stdout.contains("Hotspots:"));
    assert!(stdout.contains("Memory Patterns:"));
    assert!(stdout.contains("Divergence Analysis:"));
    assert!(stdout.contains("Recommendations:"));
}

#[test]
fn test_gpu_rr_replay() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // First, record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Then replay (non-interactive)
    let (success, stdout, _) = run_gpu_rr(&["replay", output_path.to_str().unwrap()]);

    assert!(success, "Replay command should succeed");
    assert!(stdout.contains("End of recording reached"));
}

#[test]
fn test_gpu_rr_replay_verbose() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // First, record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Replay in verbose mode
    let (success, stdout, _) = run_gpu_rr(&["-v", "replay", output_path.to_str().unwrap()]);

    assert!(success);
    // Should show address and sequence for each step
    assert!(stdout.contains("0x00000010: seq="));
    assert!(stdout.contains("End of recording reached"));
}

#[test]
fn test_gpu_rr_replay_with_breakpoint() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // First, record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Replay with breakpoint at FADD instruction (0x70)
    let (success, stdout, _) =
        run_gpu_rr(&["replay", "--break", "0x70", output_path.to_str().unwrap()]);

    assert!(success);
    assert!(stdout.contains("Breakpoint") && stdout.contains("hit"));
}

#[test]
fn test_gpu_rr_with_branching_kernel() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("branch.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // Record kernel with branches
    fs::write(&input_path, SAMPLE_SASS_WITH_BRANCH).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Analyze
    let (success, stdout, _) = run_gpu_rr(&["analyze", output_path.to_str().unwrap()]);

    assert!(success);
    // Should detect the branch
    assert!(stdout.contains("branches:") || stdout.contains("Total branches:"));
}

#[test]
fn test_gpu_rr_recording_format() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.sass");
    let output_path = temp_dir.path().join("trace.gpur");

    // Record
    fs::write(&input_path, SAMPLE_SASS).unwrap();
    let (success, _, _) = run_gpu_rr(&[
        "record",
        input_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);
    assert!(success);

    // Read recording file and verify it's valid JSON
    let content = fs::read_to_string(&output_path).unwrap();
    let recording: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Verify structure
    assert!(recording.get("header").is_some());
    assert!(recording.get("kernels").is_some());
    assert!(recording.get("timeline").is_some());

    // Verify header
    let header = recording.get("header").unwrap();
    assert_eq!(header.get("version").unwrap().as_u64(), Some(1));
    assert!(header.get("id").is_some());
    assert!(header.get("device_info").is_some());

    // Verify kernels
    let kernels = recording.get("kernels").unwrap().as_array().unwrap();
    assert_eq!(kernels.len(), 1);

    let kernel = &kernels[0];
    assert!(kernel
        .get("name")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("vectorAdd"));
    assert!(kernel.get("instructions").is_some());
    assert!(kernel.get("execution_records").is_some());

    // Verify instructions
    let instructions = kernel.get("instructions").unwrap().as_array().unwrap();
    assert_eq!(instructions.len(), 11);
}

#[test]
fn test_gpu_rr_error_no_input() {
    let (success, _, _) = run_gpu_rr(&["record"]);
    assert!(!success, "Should fail without input file");
}

#[test]
fn test_gpu_rr_error_invalid_file() {
    let (success, _, stderr) = run_gpu_rr(&["info", "/nonexistent/file.gpur"]);
    assert!(!success, "Should fail with non-existent file");
    assert!(stderr.contains("Error") || stderr.contains("Failed"));
}
