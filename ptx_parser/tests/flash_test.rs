#[test]
fn test_parse_flash_kernel() {
    let ptx = std::fs::read_to_string("/tmp/hetgpu_small.ptx").unwrap_or_default();
    if ptx.is_empty() {
        eprintln!("Skipping test: /tmp/hetgpu_small.ptx not found");
        return;
    }
    match ptx_parser::parse_module_checked(&ptx) {
        Ok(m) => {
            eprintln!("OK: {} directives", m.directives.len());
        }
        Err(errs) => {
            eprintln!("FAIL: {} errors", errs.len());
            for (i, e) in errs.iter().enumerate().take(20) {
                eprintln!("  error {}: {:?}", i, e);
            }
            panic!("Parse failed with {} errors", errs.len());
        }
    }
}
