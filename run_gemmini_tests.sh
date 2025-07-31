#!/bin/bash

# Array to store commands
commands=(
    "cargo test --features=gemmini -p ptx test::spirv_run::and_hip"
    "cargo test --features=gemmini -p ptx test::spirv_run::xor_hip"
    # Add more commands here as needed
)

# Counters
total=0
passed=0
failed=0

# Results array
declare -a results

echo "Running tests..."
echo "================"

# Run each command
for cmd in "${commands[@]}"; do
    echo
    echo "Running: $cmd"
    echo "----------------------------------------"
    
    # Execute command
    if $cmd; then
        echo "✓ PASSED"
        results+=("✓ PASSED: $cmd")
        ((passed++))
    else
        echo "✗ FAILED"
        results+=("✗ FAILED: $cmd")
        ((failed++))
    fi
    
    ((total++))
done

# Summary
echo
echo "================"
echo "TEST SUMMARY"
echo "================"
echo "Total tests: $total"
echo "Passed: $passed"
echo "Failed: $failed"
echo

# Detailed results
echo "DETAILED RESULTS:"
echo "-----------------"
for result in "${results[@]}"; do
    echo "$result"
done

# Exit with failure if any test failed
if [ $failed -gt 0 ]; then
    exit 1
fi