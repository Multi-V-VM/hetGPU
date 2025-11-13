# TMatmul k*D Matrix Multiplication Tests

This test suite implements two matrix multiplication patterns for the TMatmul backend, handling non-power-of-2 values of k with zero padding.

## Matrix Multiplication Patterns

### Pattern 1: [1 k*D][k*D D]

**Dimensions:**
- Input: 1 × k*D vector
- Weight: k*D × D matrix
- Output: 1 × D vector

**Strategy:**
1. Split the k*D input into k separate D-sized vectors
2. Multiply each D-sized vector by the corresponding D×D weight block
3. Sum all k partial results to produce the final D-sized output

**Implementation:** `test_matmul_1_kd_by_kd_d()`

**Example (k=3, D=128):**
```assembly
; Load and process k=3 blocks
ldv    v1,X          ; Load block 0 (elements 0:D)
matmul v1,W0 -> v2   ; v2 = input[0:D] * W0

ldv    v3,X          ; Load block 1 (elements D:2D)
matmul v3,W1 -> v4   ; v4 = input[D:2D] * W1

ldv    v5,X          ; Load block 2 (elements 2D:3D)
matmul v5,W2 -> v6   ; v6 = input[2D:3D] * W2

; Sum partial results
add    v7,v2,v4      ; v7 = v2 + v4
add    v1,v7,v6      ; v1 = v7 + v6 (final result)
sv     v1,O          ; Store output
```

### Pattern 2: [1 D][D k*D]

**Dimensions:**
- Input: 1 × D vector
- Weight: D × k*D matrix
- Output: 1 × k*D vector

**Strategy:**
1. Perform k separate D×D matmuls, one for each D-column block of the weight matrix
2. Accumulate results to build the k*D output vector
3. Pad with zeros if k is not a power of 2

**Implementation:** `test_matmul_1_d_by_d_kd()`

**Example (k=5, D=128):**
```assembly
; Input: 1xD, produces k*D outputs via k matmuls
ldv    v0,X          ; Load input (1 × D)
ldv    v1,zeros      ; Initialize accumulator

; Process k=5 blocks
matmul v0,W -> v2    ; Block 0: W[:,0:D]
add    v3,v1,v2      ; Accumulate
add    v1,v3,v3      ; Update accumulator

matmul v0,W -> v2    ; Block 1: W[:,D:2D]
add    v3,v1,v2      ; Accumulate
; ... repeat for blocks 2,3,4 ...

sv     v3,O          ; Store k*D result
```

## Handling Non-Power-of-2 k Values

When k is not a power of 2, we pad with zeros to reach the next power of 2:

**Example:** k=7 → k_padded=8

```assembly
; Process 7 real blocks
ldv v1,X
; ... process blocks 0-6 ...

; Add 1 padding block (zeros)
ldv v2,zeros
add v3,v1,v2
add v1,v3,v3

sv  v1,O
```

**Padding Function:**
```rust
fn next_power_of_2(k: usize) -> usize {
    if k == 0 { return 1; }
    let mut v = k - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}
```

## Register Management

The TMatmul backend has only 8 vector registers (v0-v7). All implementations carefully reuse registers to avoid exhaustion:

- Load data into temporary registers
- Perform operations
- Reuse registers for intermediate results
- Copy final result to output registers

## Testing

### Run Tests
```bash
cd /home/yhgan913/hetGPU/ptx
cargo test --test tmatmul_kd_matmul_test -- --nocapture
```

### Test Coverage

1. **`test_matmul_1_kd_by_kd_d`**: Pattern [1 k*D][k*D D] with k=3
2. **`test_matmul_1_d_by_d_kd`**: Pattern [1 D][D k*D] with k=5
3. **`test_matmul_with_padding`**: Zero padding for k=7→8
4. **`test_ptx_to_tmatmul_kd_pattern`**: PTX compilation to TMatmul
5. **`test_generate_cocotb_assembly`**: Generate assembly for hardware simulation

## Cocotb Backend Testing

The generated assembly can be tested on the cocotb hardware simulator:

```bash
# Assembly is generated at:
/tmp/tmatmul_kd_test.S

# To run on hardware simulator:
cd /root/matmulfreellm/hardware/ternary_matmul/cocotb
make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs
```

## PTX to TMatmul Compilation

The full pipeline supports:
1. PTX source code
2. Parse PTX with `ptx_parser`
3. Compile to TMatmul IR using `ptx_to_tmatmul` pass
4. Generate TMatmul assembly with `emit_tmatmul_asm`
5. Test on cocotb hardware simulator

**Example:**
```rust
let ptx_source = r#"
    .version 7.0
    .target sm_80
    .visible .entry kernel(...) {
        // PTX code
    }
"#;

let assembly = ptx::pass::ptx_to_tmatmul_assembly(ptx_source)?;
```

## Implementation Files

- **Tests:** `/home/yhgan913/hetGPU/ptx/tests/tmatmul_kd_matmul_test.rs`
- **TMatmul ASM Backend:** `/home/yhgan913/hetGPU/ptx/src/pass/emit_tmatmul_asm.rs`
- **PTX→TMatmul Compiler:** `/home/yhgan913/hetGPU/ptx/src/pass/ptx_to_tmatmul.rs`
- **Generated Assembly:** `/tmp/tmatmul_kd_test.S`

## Key Features

✅ Supports non-power-of-2 k values with zero padding
✅ Efficient register reuse (fits in 8 registers)
✅ Two complementary matrix multiplication patterns
✅ PTX compilation to TMatmul assembly
✅ Cocotb hardware backend integration
✅ Comprehensive test coverage

## Example Output

```
Running tests...
test test_matmul_1_kd_by_kd_d ... ok
test test_matmul_1_d_by_d_kd ... ok
test test_matmul_with_padding ... ok
test test_ptx_to_tmatmul_kd_pattern ... ok
test test_generate_cocotb_assembly ... ok

Assembly written to: /tmp/tmatmul_kd_test.S

To test with cocotb:
  cd /root/matmulfreellm/hardware/ternary_matmul/cocotb
  make SIM=verilator MODULE=tb_asm TESTCASE=test_asm_all_programs
```
