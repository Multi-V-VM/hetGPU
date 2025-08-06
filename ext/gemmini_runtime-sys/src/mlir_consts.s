    .section .rodata,"a",@progbits
    .align 4
    .global .L__constant_1x1xi32
    .type .L__constant_1x1xi32, @object
    .size .L__constant_1x1xi32, 4
.L__constant_1x1xi32:
    .word 1

    .global .L__constant_1x1xf32
    .type .L__constant_1x1xf32, @object
    .size .L__constant_1x1xf32, 4
.L__constant_1x1xf32:
    .float 1.0

    .global .L__constant_2x2xi32
    .type .L__constant_2x2xi32, @object  
    .size .L__constant_2x2xi32, 16
.L__constant_2x2xi32:
    .word 1
    .word 1
    .word 1
    .word 1

    .global .L__constant_1x1xi64
    .type .L__constant_1x1xi64, @object
    .size .L__constant_1x1xi64, 8
.L__constant_1x1xi64:
    .quad 1

    .global .L__constant_1x1xf64
    .type .L__constant_1x1xf64, @object
    .size .L__constant_1x1xf64, 8
.L__constant_1x1xf64:
    .double 1.0