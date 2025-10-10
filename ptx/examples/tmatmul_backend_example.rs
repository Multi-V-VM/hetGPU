// tmatmul_backend_example.rs - Demonstrates TOSA to TMatmul backend
// This example shows how to use the TMatmul backend to generate assembly

use ptx::pass::emit_tmatmul_asm::*;

fn main() {
    println!("TMatmul Backend Example");
    println!("=======================\n");

    // Example 1: Simple operations
    example_simple_ops();

    // Example 2: MLGRU layer (from the specification)
    example_mlgru_layer();

    // Example 3: GLU (Gated Linear Unit)
    example_glu();
}

fn example_simple_ops() {
    println!("Example 1: Simple Operations");
    println!("-----------------------------\n");

    let mut codegen = TMatmulCodegen::new();

    // Map memory locations
    codegen.map_memory("input", MemoryLocation::X);
    codegen.map_memory("output", MemoryLocation::O);
    codegen.map_memory("weights", MemoryLocation::WF);

    // Generate code
    codegen.add_section("SIMPLE OPERATIONS EXAMPLE");

    // Load input
    codegen.emit_operation("tmatmul.ldv", &["input"], &["%0"]).unwrap();

    // Normalize
    codegen.emit_operation("tmatmul.norm", &["%0"], &["%1"]).unwrap();

    // Sigmoid activation
    codegen.emit_operation("tmatmul.sig", &["%1"], &["%2"]).unwrap();

    // Store output
    codegen.emit_operation("tmatmul.sv", &["%2", "output"], &[]).unwrap();

    let assembly = codegen.get_assembly();
    println!("{}\n", assembly);
}

fn example_mlgru_layer() {
    println!("\nExample 2: MLGRU Layer");
    println!("----------------------\n");

    let mut codegen = TMatmulCodegen::new();

    // Map memory locations
    codegen.map_memory("X", MemoryLocation::X);
    codegen.map_memory("oH", MemoryLocation::OH);
    codegen.map_memory("WF", MemoryLocation::WF);
    codegen.map_memory("WC", MemoryLocation::WC);
    codegen.map_memory("WG", MemoryLocation::WG);
    codegen.map_memory("WO", MemoryLocation::WO);
    codegen.map_memory("TEMP_VEC", MemoryLocation::TempVec);

    codegen.add_section("SETUP/INITIALIZATION");

    // Load and normalize input
    codegen.emit_operation("tmatmul.ldv", &["X"], &["%0"]).unwrap();
    codegen.emit_operation("tmatmul.sv", &["%0", "TEMP_VEC"], &[]).unwrap();
    codegen.emit_operation("tmatmul.norm", &["%0"], &["%0"]).unwrap();

    // Load hidden state
    codegen.emit_operation("tmatmul.ldv", &["oH"], &["%1"]).unwrap();

    codegen.add_section("MLGRU_Linear: Compute f_t, c_t, g_t");

    // Compute F_t = X * WF
    codegen.add_comment("v2 = X * WF (forget gate)");
    codegen.emit_operation("tmatmul.matmul", &["%0", "WF"], &["%2"]).unwrap();

    // Compute C_t = X * WC
    codegen.add_comment("v3 = X * WC (candidate)");
    codegen.emit_operation("tmatmul.matmul", &["%0", "WC"], &["%3"]).unwrap();

    // Start G_t = X * WG (will export later)
    codegen.add_comment("Start G_t = X * WG (gate)");
    codegen.emit_operation("tmatmul.tmatmul_import", &["%0"], &[]).unwrap();
    codegen.emit_operation("tmatmul.tmatmul_go", &["WG"], &[]).unwrap();

    // Apply activations
    codegen.add_comment("v7 = sig(F_t)");
    codegen.emit_operation("tmatmul.sig", &["%2"], &["%7"]).unwrap();
    codegen.add_comment("v3 = silu(C_t)");
    codegen.emit_operation("tmatmul.silu", &["%3"], &["%3"]).unwrap();

    codegen.add_section("MLGRU Hidden State Update");
    codegen.add_comment("H_new = σ(F_t) * H_old + (1-σ(F_t)) * silu(C_t)");

    // v5 = sig(F_t) * H_old
    codegen.emit_operation("tmatmul.mul", &["%7", "%1"], &["%5"]).unwrap();

    // v6 = csig(F_t) = 1 - sig(F_t)
    codegen.emit_operation("tmatmul.csig", &["%2"], &["%6"]).unwrap();

    // v6 = (1-sig(F_t)) * silu(C_t)
    codegen.emit_operation("tmatmul.mul", &["%6", "%3"], &["%6"]).unwrap();

    // v1 = H_t = v5 + v6
    codegen.emit_operation("tmatmul.add", &["%5", "%6"], &["%1"]).unwrap();

    // Save hidden state
    codegen.emit_operation("tmatmul.sv", &["%1", "oH"], &[]).unwrap();

    codegen.add_section("MLGRU Output");
    codegen.add_comment("x_o = norm(G_t) * H_new * σ(H_new)");

    // Export G_t
    codegen.emit_operation("tmatmul.tmatmul_export", &[], &["%4"]).unwrap();

    // Normalize G_t
    codegen.emit_operation("tmatmul.norm", &["%4"], &["%4"]).unwrap();

    // v2 = sig(H_t)
    codegen.emit_operation("tmatmul.sig", &["%1"], &["%2"]).unwrap();

    // v5 = H_t * sig(H_t)
    codegen.emit_operation("tmatmul.mul", &["%1", "%2"], &["%5"]).unwrap();

    // v5 = norm(G_t) * H_t * sig(H_t)
    codegen.emit_operation("tmatmul.mul", &["%4", "%5"], &["%5"]).unwrap();

    // Normalize and project through WO
    codegen.emit_operation("tmatmul.norm", &["%5"], &["%5"]).unwrap();
    codegen.emit_operation("tmatmul.matmul", &["%5", "WO"], &["%5"]).unwrap();

    // Residual connection
    codegen.add_comment("Residual Connection #1");
    codegen.emit_operation("tmatmul.ldv", &["TEMP_VEC"], &["%0"]).unwrap();
    codegen.emit_operation("tmatmul.add", &["%0", "%5"], &["%0"]).unwrap();

    let assembly = codegen.get_assembly();
    println!("{}\n", assembly);
}

fn example_glu() {
    println!("\nExample 3: GLU (Gated Linear Unit)");
    println!("-----------------------------------\n");

    let mut codegen = TMatmulCodegen::new();

    // Map memory locations
    codegen.map_memory("X", MemoryLocation::X);
    codegen.map_memory("WU1", MemoryLocation::WU1);
    codegen.map_memory("WU2", MemoryLocation::WU2);
    codegen.map_memory("WN", MemoryLocation::WN);
    codegen.map_memory("TEMP_VEC", MemoryLocation::TempVec);

    codegen.add_section("GLU - Gated Linear Unit");

    // Double normalization
    codegen.add_comment("Double norm for GLU");
    codegen.emit_operation("tmatmul.ldv", &["X"], &["%0"]).unwrap();
    codegen.emit_operation("tmatmul.norm", &["%0"], &["%0"]).unwrap();
    codegen.emit_operation("tmatmul.norm", &["%0"], &["%0"]).unwrap();

    codegen.add_comment("Up-projection 1: up1 = x * WU1");
    codegen.emit_operation("tmatmul.matmul", &["%0", "WU1"], &["%2"]).unwrap();

    codegen.add_comment("Up-projection 2: up2 = x * WU2");
    codegen.emit_operation("tmatmul.matmul", &["%0", "WU2"], &["%3"]).unwrap();

    codegen.add_comment("Gating: gated = silu(up1) * up2");
    codegen.emit_operation("tmatmul.silu", &["%2"], &["%2"]).unwrap();
    codegen.emit_operation("tmatmul.mul", &["%2", "%3"], &["%2"]).unwrap();

    codegen.add_comment("Down-projection: output = gated * WN");
    codegen.emit_operation("tmatmul.norm", &["%2"], &["%2"]).unwrap();
    codegen.emit_operation("tmatmul.matmul", &["%2", "WN"], &["%2"]).unwrap();

    codegen.add_comment("Residual Connection");
    codegen.emit_operation("tmatmul.ldv", &["TEMP_VEC"], &["%0"]).unwrap();
    codegen.emit_operation("tmatmul.add", &["%0", "%2"], &["%0"]).unwrap();

    let assembly = codegen.get_assembly();
    println!("{}\n", assembly);
}
