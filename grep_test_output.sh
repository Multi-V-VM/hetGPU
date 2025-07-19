#!/bin/bash

# Script to grep for a specific string in the output of all spirv_run tests
# Usage: ./grep_test_output.sh "search_string"

# Check if search string is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"search_string\""
    echo "Example: $0 \"error\""
    exit 1
fi

SEARCH_STRING="$1"

# Array of all test names from the test_ptx! macro calls
tests=(
    "ld_st"
    "ld_st_implicit"
    "mov"
    "mul_lo"
    "mul_hi"
    "add"
    "setp"
    "setp_gt"
    "setp_leu"
    "bra"
    "not"
    "shl"
    "cvt_sat_s_u"
    "cvta"
    "block"
    "local_align"
    "call"
    "vector"
    "vector4"
    "ld_st_offset"
    "ntid"
    "reg_local"
    "mov_address"
    "b64tof64"
    "pred_not"
    "mad_s32"
    "mul_wide"
    "vector_extract"
    "shr"
    "or"
    "sub"
    "min"
    "max"
    "global_array"
    "extern_shared"
    "extern_shared_call"
    "rcp"
    "mul_ftz"
    "mul_non_ftz"
    "constant_f32"
    "constant_negative"
    "and"
    "selp"
    "selp_true"
    "fma"
    "shared_variable"
    "shared_ptr_32"
    "atom_cas"
    "atom_inc"
    "atom_add"
    "div_approx"
    "sqrt"
    "rsqrt"
    "neg"
    "sin"
    "cos"
    "lg2"
    "ex2"
    "cvt_rni"
    "cvt_rzi"
    "cvt_s32_f32"
    "clz"
    "popc"
    "brev"
    "xor"
    "rem"
    "bfe"
    "bfi"
    "stateful_ld_st_simple"
    "stateful_ld_st_ntid"
    "stateful_ld_st_ntid_chain"
    "stateful_ld_st_ntid_sub"
    "shared_ptr_take_address"
    "cvt_s64_s32"
    "add_tuning"
    "add_non_coherent"
    "sign_extend"
    "atom_add_float"
    "setp_nan"
    "setp_num"
    "non_scalar_ptr_offset"
    "stateful_neg_offset"
    "const"
    "cvt_s16_s8"
    "cvt_f64_f32"
    "prmt"
    "activemask"
    "membar"
    "shared_unify_extern"
    "shared_unify_local"
    "assertfail"
    "func_ptr"
    "lanemask_lt"
    "extern_func"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
found_count=0
total=${#tests[@]}

echo -e "${BLUE}Searching for \"$SEARCH_STRING\" in $total tests...${NC}"
echo "================================="

# Array to store tests that contain the string
matching_tests=()

# Run each test and grep for the string
for test in "${tests[@]}"; do
    echo -n "Checking test: $test..."
    
    # Run the test and capture both stdout and stderr
    output=$(cargo test --no-default-features --features=tenstorrent -p ptx "test::spirv_run::${test}_hip" -- --nocapture --test-threads=1 2>&1)
    
    # Check if the output contains the search string
    if echo "$output" | grep -q "$SEARCH_STRING"; then
        echo -e " ${RED}FOUND${NC}"
        matching_tests+=("$test")
        ((found_count++))
        
        # Show the matching lines with context
        echo -e "${YELLOW}Matching lines in $test:${NC}"
        echo "$output" | grep -n --color=always -C 2 "$SEARCH_STRING" | sed 's/^/  /'
        echo "---"
    else
        echo -e " ${GREEN}not found${NC}"
    fi
done

echo "================================="
echo -e "${BLUE}Search Summary:${NC}"
echo -e "  Search string: \"$SEARCH_STRING\""
echo -e "  Total tests:   $total"
echo -e "  Tests with matches: ${RED}$found_count${NC}"

if [ $found_count -gt 0 ]; then
    echo -e "\n${YELLOW}Tests containing \"$SEARCH_STRING\":${NC}"
    for test in "${matching_tests[@]}"; do
        echo "  - $test"
    done
else
    echo -e "\n${GREEN}No tests contain \"$SEARCH_STRING\"${NC}"
fi

# Exit with the count of matching tests
exit $found_count