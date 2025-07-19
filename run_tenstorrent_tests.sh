#!/bin/bash

# Script to run all spirv_run tests with Tenstorrent features
# Usage: ./run_tenstorrent_tests.sh

# Array of all test names from the test_ptx! macro calls
tests=(
    # "ld_st"
    # "ld_st_implicit"
    # "mov"
    # "mul_lo"
    # "mul_hi"
    "add"
    # "setp"
    # "setp_gt"
    # "setp_leu"
    # "bra"
    "not"
    "shl"
    # "cvt_sat_s_u"
    # "cvta"
    # "block"
    # "local_align"
    # "call"
    # "vector"
    # "vector4"
    # "ld_st_offset"
    # "ntid"
    # "reg_local"
    # "mov_address"
    # "b64tof64"
    # "pred_not"
    # "mad_s32"
    # "mul_wide"
    # "vector_extract"
    "shr"
    "or"
    "sub"
    "min"
    "max"
    # "global_array"
    # "extern_shared"
    # "extern_shared_call"
    # "rcp"
    # "mul_ftz"
    # "mul_non_ftz"
    # "constant_f32"
    # "constant_negative"
    "and"
    # "selp"
    # "selp_true"
    "fma"
    # "shared_variable"
    # "shared_ptr_32"
    # "atom_cas"
    # "atom_inc"
    # "atom_add"
    "div_approx"
    # "sqrt"
    # "rsqrt"
    # "neg"
    "sin"
    "cos"
    # "lg2"
    # "ex2"
    # "cvt_rni"
    # "cvt_rzi"
    # "cvt_s32_f32"
    # "clz"
    # "popc"
    # "brev"
    "xor"
    # "rem"
    # "bfe"
    # "bfi"
    # "stateful_ld_st_simple"
    # "stateful_ld_st_ntid"
    # "stateful_ld_st_ntid_chain"
    # "stateful_ld_st_ntid_sub"
    # "shared_ptr_take_address"
    # "cvt_s64_s32"
    # "add_tuning"
    # "add_non_coherent"
    # "sign_extend"
    # "atom_add_float"
    # "setp_nan"
    # "setp_num"
    # "non_scalar_ptr_offset"
    # "stateful_neg_offset"
    # "const"
    # "cvt_s16_s8"
    # "cvt_f64_f32"
    # "prmt"
    # "activemask"
    # "membar"
    # "shared_unify_extern"
    # "shared_unify_local"
    # "assertfail"
    # "func_ptr"
    # "lanemask_lt"
    # "extern_func"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
passed=0
failed=0
total=${#tests[@]}

echo "Running $total Tenstorrent tests..."
echo "================================="

# Run each test
for test in "${tests[@]}"; do
    echo -n "Running test: $test..."
    
    # Run the test and capture output
    output=$(cargo test --no-default-features --features=tenstorrent -p ptx "test::spirv_run::${test}_hip" -- --nocapture --test-threads=1 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e " ${GREEN}PASSED${NC}"
        ((passed++))
    else
        echo -e " ${RED}FAILED${NC}"
        ((failed++))
        echo -e "${YELLOW}Error output:${NC}"
        echo "$output" | tail -20
        echo "---"
    fi
done

echo "================================="
echo -e "Test Summary:"
echo -e "  Total:  $total"
echo -e "  ${GREEN}Passed: $passed${NC}"
echo -e "  ${RED}Failed: $failed${NC}"

# Exit with error if any tests failed
if [ $failed -gt 0 ]; then
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi