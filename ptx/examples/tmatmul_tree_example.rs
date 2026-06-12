// Example: TMatmul Algorithm Tree Generation
// Demonstrates generating tmatmul assembly from high-level operations

use ptx::pass::tmatmul_algorithm_tree::*;
use std::collections::HashMap;

fn main() {
    println!("=== TMatmul Algorithm Tree Example ===\n");

    // Example 1: Simple vector add
    example_vector_add();

    println!("\n{}\n", "=".repeat(60));

    // Example 2: Matrix-vector multiplication
    example_matmul();

    println!("\n{}\n", "=".repeat(60));

    // Example 3: FFN layer (Feed-Forward Network)
    example_ffn_layer();
}

/// Example 1: Simple vector add (C = A + B)
fn example_vector_add() {
    println!("Example 1: Vector Addition (C = A + B)");
    println!("{}", "-".repeat(40));

    let config = TMatmulConfig {
        d: 64,
        num_vector_registers: 8,
        ..Default::default()
    };
    let mut tree = AlgorithmTree::new(config);

    // Create abstract vectors (64 elements each)
    let a = tree.new_abstract_vector(64);
    let b = tree.new_abstract_vector(64);
    let c = tree.new_abstract_vector(64);

    // Load A
    let mut info_a = HashMap::new();
    info_a.insert(
        "address_label".to_string(),
        OperationInfo::String("A".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![a], Some(info_a));

    // Load B
    let mut info_b = HashMap::new();
    info_b.insert(
        "address_label".to_string(),
        OperationInfo::String("B".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![b], Some(info_b));

    // C = A + B
    tree.new_abstract_operation(AbstractOperation::Add, vec![a, b], vec![c], None);

    // Store C
    let mut info_c = HashMap::new();
    info_c.insert(
        "address_label".to_string(),
        OperationInfo::String("C".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Sv, vec![c], vec![], Some(info_c));

    tree.print_summary();

    println!("\nGenerated Assembly:");
    println!("{}", tree.generate_assembly());
}

/// Example 2: Matrix-vector multiplication (Y = W * X)
fn example_matmul() {
    println!("Example 2: Matrix-Vector Multiplication (Y = W * X)");
    println!("{}", "-".repeat(40));

    let config = TMatmulConfig {
        d: 32, // 32x32 blocks for smaller example
        num_vector_registers: 8,
        ..Default::default()
    };
    let mut tree = AlgorithmTree::new(config);

    // 64-element input/output (2 blocks of 32)
    let input = tree.new_abstract_vector(64);
    let output = tree.new_abstract_vector(64);

    // Load input
    let mut info_x = HashMap::new();
    info_x.insert(
        "address_label".to_string(),
        OperationInfo::String("X".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![input], Some(info_x));

    // TMatmul: 64x64 weight matrix
    let mut info_w = HashMap::new();
    info_w.insert(
        "address_label".to_string(),
        OperationInfo::String("W".to_string()),
    );
    info_w.insert("NumRows".to_string(), OperationInfo::Int(64));
    info_w.insert("NumColumns".to_string(), OperationInfo::Int(64));
    tree.new_abstract_operation(
        AbstractOperation::TMatmul,
        vec![input],
        vec![output],
        Some(info_w),
    );

    // Store output
    let mut info_y = HashMap::new();
    info_y.insert(
        "address_label".to_string(),
        OperationInfo::String("Y".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Sv, vec![output], vec![], Some(info_y));

    tree.print_summary();

    println!("\nGenerated Assembly:");
    println!("{}", tree.generate_assembly());
}

/// Example 3: FFN layer with SiLU activation
/// Implements: output = down_proj(silu(gate_proj(x)) * up_proj(x))
fn example_ffn_layer() {
    println!("Example 3: FFN Layer (SiLU-Gated FFN)");
    println!("{}", "-".repeat(40));

    let config = TMatmulConfig {
        d: 32,
        num_vector_registers: 8,
        ..Default::default()
    };
    let mut tree = AlgorithmTree::new(config);

    let hidden_dim = 64;
    let intermediate_dim = 128;

    // Input vector
    let x = tree.new_abstract_vector(hidden_dim);
    // Intermediate vectors
    let gate = tree.new_abstract_vector(intermediate_dim);
    let up = tree.new_abstract_vector(intermediate_dim);
    let gate_silu = tree.new_abstract_vector(intermediate_dim);
    let intermediate = tree.new_abstract_vector(intermediate_dim);
    // Output vector
    let output = tree.new_abstract_vector(hidden_dim);

    // Load input
    let mut info_x = HashMap::new();
    info_x.insert(
        "address_label".to_string(),
        OperationInfo::String("X".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Ldv, vec![], vec![x], Some(info_x));

    // gate = gate_proj(x)
    let mut info_wg = HashMap::new();
    info_wg.insert(
        "address_label".to_string(),
        OperationInfo::String("W_gate".to_string()),
    );
    info_wg.insert(
        "NumRows".to_string(),
        OperationInfo::Int(intermediate_dim as i64),
    );
    info_wg.insert(
        "NumColumns".to_string(),
        OperationInfo::Int(hidden_dim as i64),
    );
    tree.new_abstract_operation(
        AbstractOperation::TMatmul,
        vec![x],
        vec![gate],
        Some(info_wg),
    );

    // up = up_proj(x)
    let mut info_wu = HashMap::new();
    info_wu.insert(
        "address_label".to_string(),
        OperationInfo::String("W_up".to_string()),
    );
    info_wu.insert(
        "NumRows".to_string(),
        OperationInfo::Int(intermediate_dim as i64),
    );
    info_wu.insert(
        "NumColumns".to_string(),
        OperationInfo::Int(hidden_dim as i64),
    );
    tree.new_abstract_operation(AbstractOperation::TMatmul, vec![x], vec![up], Some(info_wu));

    // gate_silu = silu(gate)
    tree.new_abstract_operation(AbstractOperation::SiLU, vec![gate], vec![gate_silu], None);

    // intermediate = gate_silu * up (element-wise)
    tree.new_abstract_operation(
        AbstractOperation::Mul,
        vec![gate_silu, up],
        vec![intermediate],
        None,
    );

    // output = down_proj(intermediate)
    let mut info_wd = HashMap::new();
    info_wd.insert(
        "address_label".to_string(),
        OperationInfo::String("W_down".to_string()),
    );
    info_wd.insert("NumRows".to_string(), OperationInfo::Int(hidden_dim as i64));
    info_wd.insert(
        "NumColumns".to_string(),
        OperationInfo::Int(intermediate_dim as i64),
    );
    tree.new_abstract_operation(
        AbstractOperation::TMatmul,
        vec![intermediate],
        vec![output],
        Some(info_wd),
    );

    // Store output
    let mut info_out = HashMap::new();
    info_out.insert(
        "address_label".to_string(),
        OperationInfo::String("Output".to_string()),
    );
    tree.new_abstract_operation(AbstractOperation::Sv, vec![output], vec![], Some(info_out));

    tree.print_summary();

    println!("\nGenerated Assembly:");
    println!("{}", tree.generate_assembly());
}
