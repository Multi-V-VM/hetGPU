// Verilator testbench for spatial_core: kernel execution with oracle
// checking and cross-instance live migration.
//
// Modes:
//   run      : execute to completion, report cycles/events + final memory
//   migrate  : run until the checkpoint safe point (PC == --at-pc), dump
//              full architectural state + memory to a file, exit
//   resume   : restore state into a fresh core instance, run to completion
//
// The migration is a real process boundary: run and resume are separate
// invocations, so the resumed core is a freshly elaborated Verilator model
// (fresh "fabric instance"), not the same object.

#include "Vspatial_core.h"
#include "Vspatial_core___024root.h"
#include "verilated.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint32_t MEM_WORDS = 1 << 16;
static constexpr int LANES = 32, NREGS = 32, NPREDS = 8;

static uint32_t mem[MEM_WORDS];

struct Args {
    std::string mode = "run";
    std::string rom = "rom_mm.hex";
    std::string state = "state.bin";
    std::string out = "spatial_run.json";
    uint32_t at_pc = 0;
    uint32_t tiles = 8, tailw = 20, base_a = 0, base_b = 4096, base_c = 8192;
    uint64_t max_cycles = 200000;
    uint32_t seed = 20261105;
    std::string mem_init;                 // raw 64K x int32 preload (bridge)
    bool params_set[5] = {false, false, false, false, false};
    uint32_t params_v[5] = {0, 0, 0, 0, 0};
};

static Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (s == "--mode") a.mode = nx();
        else if (s == "--rom") a.rom = nx();
        else if (s == "--state") a.state = nx();
        else if (s == "--out") a.out = nx();
        else if (s == "--at-pc") a.at_pc = std::stoul(nx());
        else if (s == "--tiles") a.tiles = std::stoul(nx());
        else if (s == "--tailw") a.tailw = std::stoul(nx());
        else if (s == "--seed") a.seed = std::stoul(nx());
        else if (s == "--max-cycles") a.max_cycles = std::stoull(nx());
        else if (s == "--mem-init") a.mem_init = nx();
        else if (s == "--params") {
            for (int k = 0; k < 5; k++) {
                a.params_v[k] = std::stoul(nx());
                a.params_set[k] = true;
            }
        }
    }
    return a;
}

// deterministic int32 LCG for kernel inputs
static uint32_t lcg(uint64_t &s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(s >> 33);
}

struct Clockwork {
    Vspatial_core *top;
    VerilatedContext *ctx;
    uint64_t cycles = 0;
    double wall_s = 0;

    explicit Clockwork(const Args &a) {
        ctx = new VerilatedContext;
        top = new Vspatial_core{ctx};
        top->clk = 0;
        top->rst = 1;
        top->mem_rdata = 0;
        top->mem_ack = 1;
        // parameter buffer, in .entry declaration order:
        // [pA, pB, pC, pTILES, pTAILW]
        // parameter buffer, in .entry declaration order:
        // mm_reduce: [pA, pB, pC, pTILES, pTAILW]
        top->params[0] = a.params_set[0] ? a.params_v[0] : a.base_a;
        top->params[1] = a.params_set[1] ? a.params_v[1] : a.base_b;
        top->params[2] = a.params_set[2] ? a.params_v[2] : a.base_c;
        top->params[3] = a.params_set[3] ? a.params_v[3] : a.tiles;
        top->params[4] = a.params_set[4] ? a.params_v[4] : a.tailw;
        eval();
        top->rst = 0;
    }

    void eval() {
        top->eval();
    }

    void tick() {
        auto t0 = std::chrono::steady_clock::now();
        top->clk = 0;
        eval();
        top->clk = 1;
        eval();
        cycles++;
        wall_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }

    // behavioral memory: combinational response to the core's request
    void service_mem() {
        if (top->mem_req) {
            uint32_t addr = top->mem_addr;
            if (top->mem_we) mem[addr & (MEM_WORDS - 1)] = top->mem_wdata;
            top->mem_rdata = mem[addr & (MEM_WORDS - 1)];
        }
    }
};

static void load_rom(Clockwork &cw, const std::string &path, int max_instr) {
    std::ifstream f(path);
    std::string line;
    int i = 0;
    while (std::getline(f, line) && i < max_instr) {
        if (line.empty()) continue;
        uint32_t hi = std::stoul(line.substr(0, 8), nullptr, 16);
        uint32_t lo = std::stoul(line.substr(9, 8), nullptr, 16);
        cw.top->rootp->spatial_core__DOT__rom[2 * i] = hi;
        cw.top->rootp->spatial_core__DOT__rom[2 * i + 1] = lo;
        i++;
    }
}

static void load_mem_init(const Args &a) {
    if (a.mem_init.empty()) return;
    std::ifstream f(a.mem_init, std::ios::binary);
    f.read((char *)mem, sizeof(mem));
    printf("memory preloaded from %s\n", a.mem_init.c_str());
}

static void fill_inputs(const Args &a) {
    // A: tiles*32 words at base_a; B: 32 words at base_b; C zeroed
    uint64_t s = a.seed;
    for (uint32_t i = 0; i < a.tiles * 32; i++)
        mem[(a.base_a + i) & (MEM_WORDS - 1)] = (int32_t)(lcg(s) % 2000 - 1000);
    for (uint32_t i = 0; i < 32; i++)
        mem[(a.base_b + i) & (MEM_WORDS - 1)] = (int32_t)(lcg(s) % 2000 - 1000);
    for (uint32_t i = 0; i < a.tiles + 1; i++)
        mem[(a.base_c + i) & (MEM_WORDS - 1)] = 0;
}

static void dump_state(Clockwork &cw, const Args &a, uint32_t cycles_before) {
    auto *r = cw.top->rootp;
    std::ofstream f(a.state, std::ios::binary);
    auto put = [&](const void *p, size_t n) { f.write((const char *)p, n); };
    uint32_t magic = 0xC0DEC0DE;
    put(&magic, 4);
    put(&cycles_before, 4);
    put(&r->spatial_core__DOT__pc_r, 2);
    put(&r->spatial_core__DOT__events_r, 4);
    put(&r->spatial_core__DOT__waits_r, 4);
    for (int l = 0; l < LANES; l++)
        for (int reg = 0; reg < NREGS; reg++)
            put(&r->spatial_core__DOT__regs[l][reg], 4);
    for (int l = 0; l < LANES; l++)
        for (int pr = 0; pr < NPREDS; pr++) {
            uint8_t pv = r->spatial_core__DOT__preds[l][pr];
            put(&pv, 1);
        }
    put(mem, sizeof(mem));
    f.close();
    printf("state dumped at pc=%u cycles_before=%u events=%u -> %s\n",
           r->spatial_core__DOT__pc_r, cycles_before,
           r->spatial_core__DOT__events_r, a.state.c_str());
}

static void load_state(Clockwork &cw, const Args &a, uint32_t &cycles_before) {
    auto *r = cw.top->rootp;
    std::ifstream f(a.state, std::ios::binary);
    auto get = [&](void *p, size_t n) { f.read((char *)p, n); };
    uint32_t magic;
    get(&magic, 4);
    if (magic != 0xC0DEC0DE) { fprintf(stderr, "bad state file\n"); exit(2); }
    get(&cycles_before, 4);
    get(&r->spatial_core__DOT__pc_r, 2);
    get(&r->spatial_core__DOT__events_r, 4);
    get(&r->spatial_core__DOT__waits_r, 4);
    for (int l = 0; l < LANES; l++)
        for (int reg = 0; reg < NREGS; reg++)
            get(&r->spatial_core__DOT__regs[l][reg], 4);
    for (int l = 0; l < LANES; l++)
        for (int pr = 0; pr < NPREDS; pr++) {
            uint8_t pv;
            get(&pv, 1);
            r->spatial_core__DOT__preds[l][pr] = pv;
        }
    get(mem, sizeof(mem));
    cw.top->eval();
}

static void write_json(const Args &a, Clockwork &cw, uint32_t cycles_before,
                       bool migrated, const char *extra) {
    auto *r = cw.top->rootp;
    FILE *f = fopen(a.out.c_str(), "w");
    fprintf(f, "{\n  \"mode\": \"%s\",\n  \"rom\": \"%s\",\n"
               "  \"tiles\": %u,\n  \"tailw\": %u,\n"
               "  \"cycles_before\": %u,\n  \"cycles_after\": %llu,\n"
               "  \"cycles_total\": %llu,\n"
               "  \"mem_events\": %u,\n  \"token_waits\": %u,\n"
               "  \"wall_s\": %.4f,\n  \"migrated\": %s,\n"
               "  \"checkpoint_pc\": %u,\n  \"final_c0\": %d,\n  \"sentinel\": %d\n%s}",
            a.mode.c_str(), a.rom.c_str(), a.tiles, a.tailw,
            cycles_before, (unsigned long long)cw.cycles,
            (unsigned long long)(cycles_before + cw.cycles),
            r->spatial_core__DOT__events_r, r->spatial_core__DOT__waits_r,
            cw.wall_s, migrated ? "true" : "false", a.at_pc,
            (int32_t)mem[a.base_c & (MEM_WORDS - 1)],
            (int32_t)mem[(a.base_c + a.tiles) & (MEM_WORDS - 1)],
            extra ? extra : "");
    fclose(f);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Args a = parse(argc, argv);

    Clockwork cw(a);
    load_rom(cw, a.rom, 256);
    fill_inputs(a);
    load_mem_init(a);

    uint32_t cycles_before = 0;

    if (a.mode == "migrate") {
        // run until the core's PC reaches the safe point, then dump
        while (!cw.top->done && cw.cycles < a.max_cycles) {
            cw.service_mem();
            cw.tick();
            cw.service_mem();
            if (cw.top->pc == a.at_pc && cw.top->mem_events > 0) break;
        }
        bool hit = (cw.top->pc == a.at_pc);
        if (!hit) { fprintf(stderr, "checkpoint PC never reached\n"); return 3; }
        dump_state(cw, a, cw.cycles);
        write_json(a, cw, 0, false, ",\n  \"note\": \"source phase\"");
        return 0;
    }

    if (a.mode == "resume") {
        load_state(cw, a, cycles_before);
    }

    const bool trace = getenv("TRACE") != nullptr;
    while (!cw.top->done && cw.cycles < a.max_cycles) {
        cw.service_mem();
        if (trace && cw.cycles < 60)
            printf("c%03llu st=%d pc=%u op=%u ev=%u\n",
                   (unsigned long long)cw.cycles, (int)cw.top->rootp->spatial_core__DOT__state,
                   cw.top->pc, cw.top->rootp->spatial_core__DOT__opcode,
                   cw.top->mem_events);
        cw.tick();
        cw.service_mem();
    }

    auto *r = cw.top->rootp;
    if (cw.top->done) {
        printf("done: pc=%u events=%u waits=%u cycles=%llu\n",
               r->spatial_core__DOT__pc_r, r->spatial_core__DOT__events_r,
               r->spatial_core__DOT__waits_r,
               (unsigned long long)cw.cycles);
    } else {
        fprintf(stderr, "TIMEOUT at pc=%u\n", cw.top->pc);
        return 4;
    }
    write_json(a, cw, cycles_before, a.mode == "resume",
               a.mode == "resume" ? ",\n  \"note\": \"target phase\"" : nullptr);
    // final memory dump for oracle comparison
    std::ofstream mf(a.out + ".mem", std::ios::binary);
    mf.write((const char *)mem, sizeof(mem));
    return 0;
}
