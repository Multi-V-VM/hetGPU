#include <stdint.h>
#include <stddef.h>  // for size_t
// Simple malloc implementation for bare metal
static char heap[16384];  // 16KB heap
static char* heap_ptr = heap;
void* malloc(size_t size) {
    // Align to 8 bytes
    size = (size + 7) & ~7;
    if (heap_ptr + size > heap + sizeof(heap)) {
        return 0;  // Out of memory
    }
    void* result = heap_ptr;
    heap_ptr += size;
    return result;
}
// RISC-V 64-bit multiplication helper (for systems without M extension)
long long __muldi3(long long a, long long b) {
    // Simple multiplication - this assumes the processor has mul instruction
    // For processors without M extension, this would need a software implementation
    return a * b;
}
// Minimal system calls for bare metal RISC-V
static int strlen__(const char* str) {
    int len = 0;
    while (str[len]) len++;
    return len;
}
int write(int fd, const void* buf, int count) {
    // For Spike, we can use a simple system call
    // This is a minimal implementation that outputs to stdout
    register int a0 asm("a0") = fd;
    register const void* a1 asm("a1") = buf;
    register int a2 asm("a2") = count;
    register int a7 asm("a7") = 64; // SYS_write
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}
void _exit(int status) {
    register int a0 asm("a0") = status;
    register int a7 asm("a7") = 93; // SYS_exit
    asm volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
    while(1); // Should never reach here
}
// Forward declaration
int main(void);
// Entry point for bare metal
void _start() {
    main();
    _exit(0);
}
// MLIR memref descriptor
typedef struct {
    void* data;
    void* aligned_data;
    long offset;
    long sizes[2];
    long strides[2];
} memref_2d_f32_t;
// External kernel function from the object file
// Use weak symbols so missing functions don't cause link errors
extern void _mlir_ciface_matmul_kernel(memref_2d_f32_t* input1, memref_2d_f32_t* input2, memref_2d_f32_t* output) __attribute__((weak));
extern void _mlir_ciface_atom_add_float(memref_2d_f32_t* input1, memref_2d_f32_t* input2, memref_2d_f32_t* output) __attribute__((weak));
extern void _mlir_ciface_xor(memref_2d_f32_t* result, memref_2d_f32_t* input1, memref_2d_f32_t* input2) __attribute__((weak));
extern void _mlir_ciface_and(memref_2d_f32_t* result, memref_2d_f32_t* input1, memref_2d_f32_t* input2) __attribute__((weak));
extern void _mlir_ciface_add(memref_2d_f32_t* input1, memref_2d_f32_t* input2, memref_2d_f32_t* output) __attribute__((weak));
// External symbols for input data embedded in the executable
extern uint8_t __input_data_start[] __attribute__((weak));
extern uint8_t __input_data_end[] __attribute__((weak));
extern uint64_t __input_data_size __attribute__((weak));
// Calculate buffer sizes based on matrix dimensions
// For two input matrices of MATRIX_DIM_X x MATRIX_DIM_Y int32 elements
#define INPUT_BUFFER_SIZE (2 * MATRIX_DIM_X * MATRIX_DIM_Y * sizeof(int32_t))
// For one output matrix of MATRIX_DIM_X x MATRIX_DIM_Y int32 elements  
#define OUTPUT_BUFFER_SIZE (MATRIX_DIM_X * MATRIX_DIM_Y * sizeof(int32_t))

// Global buffers for input/output data (as bytes for arbitrary types)
static uint8_t input_buffer[INPUT_BUFFER_SIZE];
static uint8_t output_buffer[OUTPUT_BUFFER_SIZE];
void print_uint(uint32_t value) {
    char buffer[32];
    int len = 0;
    
    if (value == 0) {
        buffer[0] = '0';
        len = 1;
    } else {
        // Convert to string
        char temp[16];
        int i = 0;
        while (value > 0) {
            temp[i++] = '0' + (value % 10);
            value /= 10;
        }
        // Reverse the string
        while (i > 0) {
            buffer[len++] = temp[--i];
        }
    }
    
    write(1, buffer, len);
}
void print_byte_hex(uint8_t byte) {
    char buffer[3];
    const char* hex_digits = "0123456789abcdef";
    buffer[0] = hex_digits[(byte >> 4) & 0xF];
    buffer[1] = hex_digits[byte & 0xF];
    buffer[2] = ' ';
    write(1, buffer, 3);
}
void print_bytes_hex(const uint8_t* bytes, int count) {
    for (int i = 0; i < count; i++) {
        print_byte_hex(bytes[i]);
    }
}
void print_float(float value) {
    int int_part = (int)value;
    int frac_part = (int)((value - int_part) * 100);
    if (frac_part < 0) frac_part = -frac_part;
    
    char buffer[32];
    char* p = buffer;
    
    if (value < 0) {
        *p++ = '-';
        int_part = -int_part;
    }
    
    if (int_part == 0) {
        *p++ = '0';
    } else {
        char temp[16];
        int i = 0;
        while (int_part > 0) {
            temp[i++] = '0' + (int_part % 10);
            int_part /= 10;
        }
        while (i > 0) {
            *p++ = temp[--i];
        }
    }
    
    *p++ = '.';
    *p++ = '0' + (frac_part / 10);
    *p++ = '0' + (frac_part % 10);
    *p++ = ' ';
    *p = '\0';
    
    write(1, buffer, p - buffer);
}
void write_output_data(uint8_t* buffer, int size) {
    const char* output_prefix = "GEMMINI_OUTPUT: ";
    write(1, output_prefix, strlen__(output_prefix));
    
    int bytes_to_print = size;  // Print all bytes, not just first 16
    print_bytes_hex(buffer, bytes_to_print);
    
    const char* newline = "\n";
    write(1, newline, 1);
}
int main() {
    const char* start_msg = "GEMMINI_START: Executing kernel\n";
    write(1, start_msg, strlen__(start_msg));
    
    if (__input_data_start && __input_data_end) {
        uint64_t data_size = __input_data_end - __input_data_start;
        uint64_t copy_size = data_size < sizeof(input_buffer) ? data_size : sizeof(input_buffer);
        
        for (uint64_t i = 0; i < copy_size; i++) {
            input_buffer[i] = __input_data_start[i];
        }
        
        const char* load_msg = "GEMMINI_DEBUG: Loaded ";
        write(1, load_msg, strlen__(load_msg));
        print_uint((uint32_t)copy_size);
        write(1, " bytes of embedded input data\n", 31);
    } else {
        const char* no_data_msg = "GEMMINI_DEBUG: No embedded input data found\n";
        write(1, no_data_msg, strlen__(no_data_msg));
    }
    
    // Use dimensions from preprocessor or default to 1x1
    #ifndef MATRIX_DIM_X
    #define MATRIX_DIM_X 1
    #endif
    #ifndef MATRIX_DIM_Y
    #define MATRIX_DIM_Y 1
    #endif
    
    memref_2d_f32_t input1_desc, input2_desc, output_desc;
    
    int dim1 = MATRIX_DIM_X, dim2 = MATRIX_DIM_Y;
    
    int element_size = sizeof(float);
    
    input1_desc = (memref_2d_f32_t){
        .data = (void*)input_buffer,
        .aligned_data = (void*)input_buffer, 
        .offset = 0,
        .sizes = {dim1, dim2},
        .strides = {dim2, 1}
    };
    
    input2_desc = (memref_2d_f32_t){
        .data = (void*)(input_buffer + (dim1 * dim2 * element_size)),  // Second buffer after first
        .aligned_data = (void*)(input_buffer + (dim1 * dim2 * element_size)),
        .offset = 0, 
        .sizes = {dim1, dim2},
        .strides = {dim2, 1}
    };
    
    output_desc = (memref_2d_f32_t){
        .data = (void*)output_buffer,
        .aligned_data = (void*)output_buffer,
        .offset = 0,
        .sizes = {dim1, dim2}, 
        .strides = {dim2, 1}
    };
    
    const char* kernel_msg = "GEMMINI_KERNEL: Calling compiled kernel\n";
    write(1, kernel_msg, strlen__(kernel_msg));
    
    if (_mlir_ciface_matmul_kernel) {
        const char* matmul_msg = "GEMMINI_KERNEL: Found matmul_kernel\n";
        write(1, matmul_msg, strlen__(matmul_msg));
        _mlir_ciface_matmul_kernel(&input1_desc, &input2_desc, &output_desc);
    } else if (_mlir_ciface_atom_add_float) {
        const char* atom_msg = "GEMMINI_KERNEL: Found atom_add_float\n";
        write(1, atom_msg, strlen__(atom_msg));
        _mlir_ciface_atom_add_float(&input1_desc, &input2_desc, &output_desc);
    } else if (_mlir_ciface_xor) {
        const char* xor_msg = "GEMMINI_KERNEL: Found xor\n";
        write(1, xor_msg, strlen__(xor_msg));
        
        memref_2d_f32_t result_desc;
        _mlir_ciface_xor(&result_desc, &input1_desc, &input2_desc);
        
        uint8_t* result_data = (uint8_t*)result_desc.aligned_data;
        int copy_size = dim1 * dim2 * sizeof(float);
        for (int i = 0; i < copy_size; i++) {
            output_buffer[i] = result_data[i];
        }
    } else if (_mlir_ciface_and) {
        const char* and_msg = "GEMMINI_KERNEL: Found and\n";
        write(1, and_msg, strlen__(and_msg));
        
        // Debug: Print first few values of input matrices
        const char* input1_msg = "GEMMINI_DEBUG: Input1 first 10 values (as i32): ";
        write(1, input1_msg, strlen__(input1_msg));
        int32_t* input1_data = (int32_t*)input1_desc.aligned_data;
        for (int i = 0; i < 10 && i < dim1 * dim2; i++) {
            print_uint((uint32_t)input1_data[i]);
            write(1, " ", 1);
        }
        write(1, "\n", 1);
        
        const char* input2_msg = "GEMMINI_DEBUG: Input2 first 10 values (as i32): ";
        write(1, input2_msg, strlen__(input2_msg));
        int32_t* input2_data = (int32_t*)input2_desc.aligned_data;
        for (int i = 0; i < 10 && i < dim1 * dim2; i++) {
            print_uint((uint32_t)input2_data[i]);
            write(1, " ", 1);
        }
        write(1, "\n", 1);
        
        memref_2d_f32_t result_desc;
        _mlir_ciface_and(&result_desc, &input1_desc, &input2_desc);
        
        // Debug: Print first few values of result
        const char* result_msg = "GEMMINI_DEBUG: Result first 10 values (as i32): ";
        write(1, result_msg, strlen__(result_msg));
        int32_t* result_data_i32 = (int32_t*)result_desc.aligned_data;
        for (int i = 0; i < 10 && i < dim1 * dim2; i++) {
            print_uint((uint32_t)result_data_i32[i]);
            write(1, " ", 1);
        }
        write(1, "\n", 1);
        
        uint8_t* result_data = (uint8_t*)result_desc.aligned_data;
        int copy_size = dim1 * dim2 * sizeof(float);
        for (int i = 0; i < copy_size; i++) {
            output_buffer[i] = result_data[i];
        }
    } else if (_mlir_ciface_add) {
        const char* add_msg = "GEMMINI_KERNEL: Found add\n";
        write(1, add_msg, strlen__(add_msg));
        _mlir_ciface_add(&input1_desc, &input2_desc, &output_desc);
    }
    
    write_output_data(output_buffer, OUTPUT_BUFFER_SIZE);
    
    const char* end_msg = "GEMMINI_END: Kernel execution completed\n";
    write(1, end_msg, strlen__(end_msg));
    
    return 0;
}