use std::ffi::CString;
use std::ptr;
use tt_comgr_sys::*;

fn main() {
    println!("Testing TTMLIR integration in tt_comgr_sys");

    // Test basic API functionality
    test_version_info();
    test_mlir_processing();
}

fn test_version_info() {
    println!("\n=== Version Info Test ===");

    let mut major: u32 = 0;
    let mut minor: u32 = 0;

    let result = tt_comgr_get_version(&mut major, &mut minor);
    if result.is_ok() {
        println!("Version: {}.{}", major, minor);
    } else {
        println!("Failed to get version");
    }
}

fn test_mlir_processing() {
    println!("\n=== MLIR Processing Test ===");

    unsafe {
        // Create input data set
        let mut input_set = tt_comgr_data_set_t { handle: 0 };
        if tt_comgr_create_data_set(&mut input_set).is_err() {
            println!("Failed to create input data set");
            return;
        }

        // Create output data set
        let mut output_set = tt_comgr_data_set_t { handle: 0 };
        if tt_comgr_create_data_set(&mut output_set).is_err() {
            println!("Failed to create output data set");
            tt_comgr_release_data_set(input_set).ok();
            return;
        }

        // Create source data for MLIR file
        let mut mlir_data = tt_comgr_data_t { handle: 0 };
        if tt_comgr_create_data(
            tt_comgr_data_kind_s::TT_COMGR_DATA_KIND_SOURCE,
            &mut mlir_data,
        )
        .is_err()
        {
            println!("Failed to create MLIR data");
            tt_comgr_release_data_set(input_set).ok();
            tt_comgr_release_data_set(output_set).ok();
            return;
        }

        // Set MLIR file name
        let mlir_filename = CString::new("test_example.mlir").unwrap();
        if tt_comgr_data_set_name(mlir_data, mlir_filename.as_ptr()).is_err() {
            println!("Failed to set MLIR data name");
            tt_comgr_release_data(mlir_data).ok();
            tt_comgr_release_data_set(input_set).ok();
            tt_comgr_release_data_set(output_set).ok();
            return;
        }

        // Read the test MLIR file
        let mlir_content = std::fs::read_to_string("test_example.mlir").unwrap_or_else(|_| {
            println!("Warning: Could not read test_example.mlir, using mock content");
            "module { func.func @test() { return } }".to_string()
        });

        // Set MLIR content
        if tt_comgr_data_set_bytes(
            mlir_data,
            mlir_content.as_ptr() as *const std::os::raw::c_void,
            mlir_content.len(),
        )
        .is_err()
        {
            println!("Failed to set MLIR data content");
            tt_comgr_release_data(mlir_data).ok();
            tt_comgr_release_data_set(input_set).ok();
            tt_comgr_release_data_set(output_set).ok();
            return;
        }

        // Add MLIR data to input set
        if tt_comgr_data_set_add(input_set, mlir_data).is_err() {
            println!("Failed to add MLIR data to input set");
            tt_comgr_release_data(mlir_data).ok();
            tt_comgr_release_data_set(input_set).ok();
            tt_comgr_release_data_set(output_set).ok();
            return;
        }

        // Create action info
        let mut action_info = tt_comgr_action_info_t { handle: 0 };
        if tt_comgr_create_action_info(&mut action_info).is_err() {
            println!("Failed to create action info");
            tt_comgr_release_data(mlir_data).ok();
            tt_comgr_release_data_set(input_set).ok();
            tt_comgr_release_data_set(output_set).ok();
            return;
        }

        // Set language (not specifically used for MLIR, but required)
        if tt_comgr_action_info_set_language(
            action_info,
            tt_comgr_language_s::TT_COMGR_LANGUAGE_SYCL,
        )
        .is_err()
        {
            println!("Failed to set language");
        }

        // Perform compilation action (this will detect .mlir and use TTMLIR tools)
        println!("Attempting to process MLIR file...");
        let result = tt_comgr_do_action(
            tt_comgr_action_kind_s::TT_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
            action_info,
            input_set,
            output_set,
        );

        if result.is_ok() {
            println!("✓ MLIR processing completed successfully!");

            // Check output data count
            let mut output_count = 0;
            if tt_comgr_get_data_count(output_set, &mut output_count).is_ok() {
                println!("Generated {} output files", output_count);

                // List output files
                for i in 0..output_count {
                    let mut output_data = tt_comgr_data_t { handle: 0 };
                    if tt_comgr_get_data(output_set, i, &mut output_data).is_ok() {
                        let mut name_size = 0;
                        if tt_comgr_get_data_name(output_data, &mut name_size, ptr::null_mut())
                            .is_ok()
                            && name_size > 1
                        {
                            let mut name_buffer = vec![0i8; name_size];
                            if tt_comgr_get_data_name(
                                output_data,
                                &mut name_size,
                                name_buffer.as_mut_ptr(),
                            )
                            .is_ok()
                            {
                                let name = CString::from_vec_unchecked(
                                    name_buffer[0..name_size - 1]
                                        .iter()
                                        .map(|&c| c as u8)
                                        .collect(),
                                );
                                println!("  - Output file {}: {}", i, name.to_string_lossy());
                            }
                        }
                    }
                }
            }
        } else {
            println!(
                "✗ MLIR processing failed, but this is expected if ttmlir tools are not available"
            );
            println!("The system should have fallen back to mock processing");
        }

        // Clean up
        tt_comgr_release_action_info(action_info).ok();
        tt_comgr_release_data(mlir_data).ok();
        tt_comgr_release_data_set(input_set).ok();
        tt_comgr_release_data_set(output_set).ok();
    }

    println!("Test completed!");
}
