#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum {
    PACC_CLUST_NUM = 4,
    PACC_CORE_NUM = 4,
};

static const uint64_t pacc_base[PACC_CLUST_NUM] = {
    0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};

static const uint64_t PACC_TOP_REG_OFF = 0x201000ULL;
static const uint64_t CORE_RST_OFF[PACC_CORE_NUM] = {0x14, 0x18, 0x1c, 0x20};
static const uint64_t SYSTEM_RST_OFF = 0x24;
static const uint64_t FORCE_RESETPC_RELOAD_OFF = 0x28;
static const uint64_t RESET_VEC_LO_OFF = 0x6c;

struct opts {
    bool status;
    bool core_reset;
    bool sys_cold_reset;
    bool release;
    bool dry_run;
    bool cores_are_wfi;
    uint32_t pacc_mask;
    uint32_t core_mask;
    uint32_t reset_vec;
};

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s --status [--pacc-mask 0xf]\n"
        "  %s --core-reset --cores-are-wfi [--pacc-mask 0xf] [--core-mask 0xf] [--reset-vec 0x30080000]\n"
        "  %s --sys-cold-reset --cores-are-wfi [--pacc-mask 0xf] [--reset-vec 0x30080000]\n"
        "  %s --release [--pacc-mask 0xf] [--core-mask 0xf] [--reset-vec 0x30080000]\n"
        "\n"
        "Safety:\n"
        "  Reset operations require --cores-are-wfi. Do not use them while any PACC core is running.\n"
        "  --dry-run prints the register writes without touching /dev/mem.\n",
        argv0, argv0, argv0, argv0);
}

static uint32_t parse_u32(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 0);
    if (errno || !end || *end != '\0' || v > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)v;
}

static uint64_t top_addr(unsigned pacc_id, uint64_t off) {
    return pacc_base[pacc_id] + PACC_TOP_REG_OFF + off;
}

static uint32_t mmio_read32(int memfd, uint64_t addr) {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~page_mask);
    off_t off = (off_t)(addr & page_mask);
    void *map = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap read 0x%016" PRIx64 " failed: %s\n", addr, strerror(errno));
        exit(1);
    }
    uint32_t val = *(volatile uint32_t *)((char *)map + off);
    munmap(map, (size_t)page_size);
    return val;
}

static void mmio_write32(int memfd, uint64_t addr, uint32_t val, bool dry_run) {
    if (dry_run) {
        printf("WRITE32 0x%016" PRIx64 " = 0x%08" PRIx32 "\n", addr, val);
        return;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~page_mask);
    off_t off = (off_t)(addr & page_mask);
    void *map = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap write 0x%016" PRIx64 " failed: %s\n", addr, strerror(errno));
        exit(1);
    }
    *(volatile uint32_t *)((char *)map + off) = val;
    __sync_synchronize();
    munmap(map, (size_t)page_size);
}

static int open_devmem(bool dry_run) {
    if (dry_run) return -1;
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s; run with sudo or use --dry-run\n", strerror(errno));
        exit(1);
    }
    return fd;
}

static void print_status(int memfd, const struct opts *o) {
    if (o->dry_run) {
        printf("status requires /dev/mem reads; --dry-run only validates arguments\n");
        return;
    }
    for (unsigned p = 0; p < PACC_CLUST_NUM; ++p) {
        if ((o->pacc_mask & (1u << p)) == 0) continue;
        printf("PACC%u top=0x%016" PRIx64 "\n", p, top_addr(p, 0));
        printf("  SYSTEM_RST[0x24]          = 0x%08" PRIx32 "\n", mmio_read32(memfd, top_addr(p, SYSTEM_RST_OFF)));
        printf("  FORCE_RESETPC_RELOAD[0x28]= 0x%08" PRIx32 "\n", mmio_read32(memfd, top_addr(p, FORCE_RESETPC_RELOAD_OFF)));
        for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
            printf("  CORE%u_RST[0x%02" PRIx64 "]        = 0x%08" PRIx32 "\n",
                c, CORE_RST_OFF[c], mmio_read32(memfd, top_addr(p, CORE_RST_OFF[c])));
        }
        for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
            printf("  CORE%u_RESET_VEC_LO       = 0x%08" PRIx32 "\n",
                c, mmio_read32(memfd, top_addr(p, RESET_VEC_LO_OFF + 8ULL * c)));
        }
    }
}

static void program_reset_vectors(int memfd, const struct opts *o, unsigned p) {
    for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
        if ((o->core_mask & (1u << c)) == 0) continue;
        mmio_write32(memfd, top_addr(p, RESET_VEC_LO_OFF + 8ULL * c), o->reset_vec, o->dry_run);
    }
    mmio_write32(memfd, top_addr(p, FORCE_RESETPC_RELOAD_OFF), o->core_mask & 0xfu, o->dry_run);
    mmio_write32(memfd, top_addr(p, FORCE_RESETPC_RELOAD_OFF), 0, o->dry_run);
}

static void run_reset(int memfd, const struct opts *o) {
    if ((o->core_reset || o->sys_cold_reset) && !o->cores_are_wfi) {
        fprintf(stderr, "refusing reset: pass --cores-are-wfi only after confirming all selected PACC cores are in WFI\n");
        exit(2);
    }

    for (unsigned p = 0; p < PACC_CLUST_NUM; ++p) {
        if ((o->pacc_mask & (1u << p)) == 0) continue;
        printf("PACC%u\n", p);

        if (o->core_reset) {
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, top_addr(p, CORE_RST_OFF[c]), 0x3, o->dry_run);
            }
            program_reset_vectors(memfd, o, p);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, top_addr(p, CORE_RST_OFF[c]), 0x0, o->dry_run);
            }
        }

        if (o->sys_cold_reset) {
            mmio_write32(memfd, top_addr(p, SYSTEM_RST_OFF), 0x1, o->dry_run);
            program_reset_vectors(memfd, o, p);
            mmio_write32(memfd, top_addr(p, SYSTEM_RST_OFF), 0x0, o->dry_run);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, top_addr(p, CORE_RST_OFF[c]), 0x0, o->dry_run);
            }
        }

        if (o->release) {
            program_reset_vectors(memfd, o, p);
            mmio_write32(memfd, top_addr(p, SYSTEM_RST_OFF), 0x0, o->dry_run);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, top_addr(p, CORE_RST_OFF[c]), 0x0, o->dry_run);
            }
        }
    }
}

int main(int argc, char **argv) {
    struct opts o = {
        .pacc_mask = 0xf,
        .core_mask = 0xf,
        .reset_vec = 0x30080000,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--status") == 0) o.status = true;
        else if (strcmp(argv[i], "--core-reset") == 0) o.core_reset = true;
        else if (strcmp(argv[i], "--sys-cold-reset") == 0) o.sys_cold_reset = true;
        else if (strcmp(argv[i], "--release") == 0) o.release = true;
        else if (strcmp(argv[i], "--dry-run") == 0) o.dry_run = true;
        else if (strcmp(argv[i], "--cores-are-wfi") == 0) o.cores_are_wfi = true;
        else if (strcmp(argv[i], "--pacc-mask") == 0 && i + 1 < argc) o.pacc_mask = parse_u32(argv[++i], "pacc-mask");
        else if (strcmp(argv[i], "--core-mask") == 0 && i + 1 < argc) o.core_mask = parse_u32(argv[++i], "core-mask");
        else if (strcmp(argv[i], "--reset-vec") == 0 && i + 1 < argc) o.reset_vec = parse_u32(argv[++i], "reset-vec");
        else {
            usage(argv[0]);
            return 2;
        }
    }

    int ops = (int)o.status + (int)o.core_reset + (int)o.sys_cold_reset + (int)o.release;
    if (ops != 1 || (o.pacc_mask & ~0xfu) || (o.core_mask & ~0xfu)) {
        usage(argv[0]);
        return 2;
    }

    int memfd = open_devmem(o.dry_run);
    if (o.status) print_status(memfd, &o);
    else run_reset(memfd, &o);
    if (memfd >= 0) close(memfd);
    return 0;
}
