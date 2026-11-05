#include "Vtb_check.h"
#include <cstdio>
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* top = new Vtb_check;
    for (int i = 0; i < 10; i++) { top->clk = i & 1; top->eval(); }
    printf("cnt=%u\n", top->cnt);
    return top->cnt == 5 ? 0 : 1;
}
