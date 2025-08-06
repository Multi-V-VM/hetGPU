OUTPUT_ARCH("riscv")
ENTRY(_start)
MEMORY
{
    ram (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000
}
SECTIONS
{
    . = 0x80000000;
    
    .text ALIGN(4): {
        *(.text.init)
        *(.text)
        *(.text.*)
    } > ram
    
    .rodata ALIGN(4): {
        *(.rodata)
        *(.rodata.*)
        *(.srodata)
        *(.srodata.*)
    } > ram
    
    .data ALIGN(4): {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > ram
    
    .bss ALIGN(8): {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        __bss_end = .;
    } > ram
    
    . = ALIGN(8);
    _end = .;
}