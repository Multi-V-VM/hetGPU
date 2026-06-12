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
#include <sys/stat.h>
#include <unistd.h>

enum {
    PACC_CLUST_NUM = 4,
    PACC_CORE_NUM = 4,
};

static const uint64_t pacc_base[PACC_CLUST_NUM] = {
    0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};

static const uint64_t LX500_PACC_TOP_ABP_CRG_BASE = 0x200000ULL;
static const uint64_t LX500_PACC_TOP_ABP_CFG_BASE = 0x201000ULL;
static const uint64_t LX500_PACC_TOP_ABP_MBOX_RAM_BASE = 0x210000ULL;
static const uint64_t LX500_PACC_TOP_ABP_MBOX_DB_BASE = 0x214000ULL;
static const uint64_t LX500_PACC_CFG_CORE_RESET[PACC_CORE_NUM] = {0x14, 0x18, 0x1c, 0x20};
static const uint64_t LX500_PACC_CFG_SYS_RESET = 0x24;
static const uint64_t LX500_PACC_CFG_FORCE_RESETPC_RELOAD = 0x28;
static const uint64_t LX500_PACC_CFG_RST_VECTOR_0_LOW = 0x6c;
static const uint64_t LX500_PACC_CFG_RST_VECTOR_0_HIGH = 0x70;
static const uint64_t LX500_PACC_CFG_SECURE_TIEOFF = 0xbc;
static const uint64_t LX500_PACC_CFG_MAM_CRM = 0xc4;
static const uint64_t LX500_PACC_MBOX_RAM_SIZE = 0x4000ULL;

struct pacc_base_cmd {
    uint64_t reserved0;
    uint64_t reserved1;
    uint64_t base;
    uint64_t size;
    uint64_t reserved2;
} __attribute__((packed));

struct opts {
    const char *image;
    uint64_t load_addr;
    uint64_t entry;
    uint32_t pacc_mask;
    uint32_t core_mask;
    bool status;
    bool load_only;
    bool release;
    bool core_reset;
    bool sys_cold_reset;
    bool cores_are_wfi;
    bool dry_run;
    bool init_crg;
    bool send_base_cmd;
    bool have_secure_tieoff;
    bool have_mam_crm;
    uint32_t secure_tieoff;
    uint32_t mam_crm;
    uint64_t reserved_size;
};

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s --status [--pacc-mask 0xf]\n"
        "  %s --image Image --load-addr 0xc0000000 [--entry 0xc0000000] --load-only\n"
        "  %s --image Image --load-addr 0xc0000000 [--entry 0xc0000000] --release --cores-are-wfi\n"
        "  %s --image Image --load-addr 0xc0000000 [--entry 0xc0000000] --core-reset --cores-are-wfi\n"
        "  %s --image Image --load-addr 0xc0000000 [--entry 0xc0000000] --sys-cold-reset --cores-are-wfi\n"
        "\n"
        "options:\n"
        "  --pacc-mask 0xf        select PACC clusters\n"
        "  --core-mask 0xf        select cores inside each PACC\n"
        "  --init-crg             run the vendor 1100MHz ref-to-PLL CRG init sequence\n"
        "  --send-base-cmd        send the loader base/size command through PACC mailbox RAM\n"
        "  --reserved-size SIZE   reserved-memory size used by --send-base-cmd (default 0x8000000)\n"
        "  --secure-tieoff VAL    write CFG+0xbc before release\n"
        "  --mam-crm VAL          write CFG+0xc4 before release\n"
        "  --dry-run              print image load and MMIO writes only\n"
        "\n"
        "Reverse-engineered boot sequence:\n"
        "  1. copy raw lx500_pacc.bin firmware image to PACC-visible DDR at --load-addr\n"
        "  2. program COREn reset vector low/high at CFG+0x6c/0x70 + n*8\n"
        "  3. optionally run CRG init and send the boot base command\n"
        "  4. optionally program SECURE_TIEOFF CFG+0xbc and MAM_CRM CFG+0xc4\n"
        "  5. pulse FORCE_RESETPC_RELOAD CFG+0x28, then release SYS/CORE reset\n"
        "\n"
        "Safety:\n"
        "  Reset operations require --cores-are-wfi. Do not reset running PACC cores.\n",
        argv0, argv0, argv0, argv0, argv0);
}

static uint64_t parse_u64(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 0);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint64_t)v;
}

static uint32_t parse_u32(const char *s, const char *name) {
    uint64_t v = parse_u64(s, name);
    if (v > UINT32_MAX) {
        fprintf(stderr, "%s too large: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)v;
}

static uint64_t cfg_addr(unsigned pacc_id, uint64_t off) {
    return pacc_base[pacc_id] + LX500_PACC_TOP_ABP_CFG_BASE + off;
}

static uint64_t crg_addr(unsigned pacc_id, uint64_t off) {
    return pacc_base[pacc_id] + LX500_PACC_TOP_ABP_CRG_BASE + off;
}

static uint64_t mbox_ram_addr(unsigned pacc_id, uint64_t off) {
    return pacc_base[pacc_id] + LX500_PACC_TOP_ABP_MBOX_RAM_BASE + off;
}

static uint64_t mbox_db_addr(unsigned pacc_id, uint64_t off) {
    return pacc_base[pacc_id] + LX500_PACC_TOP_ABP_MBOX_DB_BASE + off;
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

static uint32_t mmio_read32(int memfd, uint64_t addr) {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~mask);
    off_t off = (off_t)(addr & mask);
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
    uint64_t mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~mask);
    off_t off = (off_t)(addr & mask);
    void *map = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap write 0x%016" PRIx64 " failed: %s\n", addr, strerror(errno));
        exit(1);
    }
    *(volatile uint32_t *)((char *)map + off) = val;
    __sync_synchronize();
    munmap(map, (size_t)page_size);
}

static void mmio_write64(int memfd, uint64_t addr, uint64_t val, bool dry_run) {
    if (dry_run) {
        printf("WRITE64 0x%016" PRIx64 " = 0x%016" PRIx64 "\n", addr, val);
        return;
    }
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~mask);
    off_t off = (off_t)(addr & mask);
    void *map = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap write 0x%016" PRIx64 " failed: %s\n", addr, strerror(errno));
        exit(1);
    }
    *(volatile uint64_t *)((char *)map + off) = val;
    __sync_synchronize();
    munmap(map, (size_t)page_size);
}

static void zero_phys(int memfd, uint64_t addr, size_t len, bool dry_run) {
    if (dry_run) {
        printf("ZERO   0x%016" PRIx64 " len=0x%zx\n", addr, len);
        return;
    }
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t mask = (uint64_t)page_size - 1;
    off_t page = (off_t)(addr & ~mask);
    off_t off = (off_t)(addr & mask);
    size_t map_len = (off + len + (size_t)page_size - 1) & ~((size_t)page_size - 1);
    void *map = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap zero 0x%016" PRIx64 " failed: %s\n", addr, strerror(errno));
        exit(1);
    }
    memset((char *)map + off, 0, len);
    __sync_synchronize();
    munmap(map, map_len);
}

static void copy_image_to_phys(int memfd, const struct opts *o) {
    if (!o->image) return;

    int imgfd = open(o->image, O_RDONLY);
    if (imgfd < 0) {
        fprintf(stderr, "open image %s failed: %s\n", o->image, strerror(errno));
        exit(1);
    }
    struct stat st;
    if (fstat(imgfd, &st) != 0) {
        fprintf(stderr, "stat image %s failed: %s\n", o->image, strerror(errno));
        exit(1);
    }
    if (o->dry_run) {
        printf("LOAD image %s size=%" PRIuMAX " to phys 0x%016" PRIx64 "\n",
               o->image, (uintmax_t)st.st_size, o->load_addr);
        close(imgfd);
        return;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t written = 0;
    while (written < (uint64_t)st.st_size) {
        uint64_t phys = o->load_addr + written;
        uint64_t page_mask = (uint64_t)page_size - 1;
        off_t page = (off_t)(phys & ~page_mask);
        size_t off = (size_t)(phys & page_mask);
        size_t chunk = (size_t)page_size - off;
        uint64_t remaining = (uint64_t)st.st_size - written;
        if (chunk > remaining) chunk = (size_t)remaining;

        void *map = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, page);
        if (map == MAP_FAILED) {
            fprintf(stderr, "mmap image dst 0x%016" PRIx64 " failed: %s\n", phys, strerror(errno));
            exit(1);
        }
        ssize_t n = pread(imgfd, (char *)map + off, chunk, (off_t)written);
        if (n < 0 || (size_t)n != chunk) {
            fprintf(stderr, "read image failed at 0x%016" PRIx64 ": %s\n", written, strerror(errno));
            exit(1);
        }
        msync(map, (size_t)page_size, MS_SYNC);
        munmap(map, (size_t)page_size);
        written += chunk;
    }
    close(imgfd);
    printf("loaded %s (%" PRIu64 " bytes) to 0x%016" PRIx64 "\n", o->image, written, o->load_addr);
}

static void print_status(int memfd, const struct opts *o) {
    if (o->dry_run) {
        printf("status requires /dev/mem reads; --dry-run only validates arguments\n");
        return;
    }
    for (unsigned p = 0; p < PACC_CLUST_NUM; ++p) {
        if ((o->pacc_mask & (1u << p)) == 0) continue;
        printf("PACC%u cfg=0x%016" PRIx64 " crg=0x%016" PRIx64 "\n",
               p,
               pacc_base[p] + LX500_PACC_TOP_ABP_CFG_BASE,
               pacc_base[p] + LX500_PACC_TOP_ABP_CRG_BASE);
        printf("  SYS_RESET[0x24]       = 0x%08" PRIx32 "\n", mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_SYS_RESET)));
        printf("  SECURE_TIEOFF[0xbc]   = 0x%08" PRIx32 "\n", mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_SECURE_TIEOFF)));
        printf("  MAM_CRM[0xc4]         = 0x%08" PRIx32 "\n", mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_MAM_CRM)));
        for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
            printf("  CORE%u_RESET[0x%02" PRIx64 "]  = 0x%08" PRIx32 "\n",
                   c, LX500_PACC_CFG_CORE_RESET[c], mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_CORE_RESET[c])));
            printf("  CORE%u_RST_VEC        = 0x%08" PRIx32 "%08" PRIx32 "\n",
                   c,
                   mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_RST_VECTOR_0_HIGH + 8ULL * c)),
                   mmio_read32(memfd, cfg_addr(p, LX500_PACC_CFG_RST_VECTOR_0_LOW + 8ULL * c)));
        }
    }
}

static void program_cfg(int memfd, const struct opts *o, unsigned p) {
    uint32_t lo = (uint32_t)(o->entry & 0xffffffffu);
    uint32_t hi = (uint32_t)(o->entry >> 32);
    for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
        if ((o->core_mask & (1u << c)) == 0) continue;
        mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_RST_VECTOR_0_LOW + 8ULL * c), lo, o->dry_run);
        mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_RST_VECTOR_0_HIGH + 8ULL * c), hi, o->dry_run);
    }
    if (o->have_secure_tieoff) {
        mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_SECURE_TIEOFF), o->secure_tieoff, o->dry_run);
    }
    if (o->have_mam_crm) {
        mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_MAM_CRM), o->mam_crm, o->dry_run);
    }
    mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_FORCE_RESETPC_RELOAD), o->core_mask & 0xfu, o->dry_run);
    mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_FORCE_RESETPC_RELOAD), 0, o->dry_run);
}

static void init_crg_1100_ref2pll(int memfd, unsigned p, bool dry_run) {
    uint32_t value;
    unsigned i;

    mmio_write32(memfd, crg_addr(p, 0x8), 0x00dc0a12, dry_run);
    mmio_write32(memfd, crg_addr(p, 0x0), 0x00000110, dry_run);
    mmio_write32(memfd, crg_addr(p, 0x0), 0x00000111, dry_run);
    if (!dry_run) {
        for (i = 0; i < 200; ++i) {
            value = mmio_read32(memfd, crg_addr(p, 0x0));
            if (value & (1u << 24))
                break;
            usleep(1000);
        }
    }
    value = dry_run ? 0 : mmio_read32(memfd, crg_addr(p, 0x0));
    mmio_write32(memfd, crg_addr(p, 0x0), value & ~(1u << 4), dry_run);
    value = dry_run ? 0 : mmio_read32(memfd, crg_addr(p, 0x0));
    mmio_write32(memfd, crg_addr(p, 0x0), value | (1u << 12), dry_run);
    value = (1u << 21) - 1u;
    mmio_write32(memfd, crg_addr(p, 0x130), value, dry_run);
    mmio_write32(memfd, crg_addr(p, 0x134), value, dry_run);
    if (!dry_run)
        printf("PACC%u CRG init done pll_status=0x%08" PRIx32 "\n",
               p, mmio_read32(memfd, crg_addr(p, 0x0)));
}

static void send_base_cmd(int memfd, const struct opts *o, unsigned p) {
    struct pacc_base_cmd cmd = {
        .base = o->load_addr,
        .size = o->reserved_size,
    };

    zero_phys(memfd, mbox_ram_addr(p, 0), (size_t)LX500_PACC_MBOX_RAM_SIZE, o->dry_run);
    mmio_write32(memfd, mbox_db_addr(p, 0x10), 0xffffffffu, o->dry_run);
    mmio_write64(memfd, mbox_ram_addr(p, 0), cmd.reserved0, o->dry_run);
    mmio_write64(memfd, mbox_ram_addr(p, 8), cmd.reserved1, o->dry_run);
    mmio_write64(memfd, mbox_ram_addr(p, 16), cmd.base, o->dry_run);
    mmio_write64(memfd, mbox_ram_addr(p, 24), cmd.size, o->dry_run);
    mmio_write64(memfd, mbox_ram_addr(p, 32), cmd.reserved2, o->dry_run);
    mmio_write32(memfd, mbox_db_addr(p, 0x0), 1u << p, o->dry_run);
    printf("PACC%u base command base=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
           p, cmd.base, cmd.size);
}

static void boot_selected(int memfd, const struct opts *o) {
    if ((o->core_reset || o->sys_cold_reset || o->release) && !o->cores_are_wfi) {
        fprintf(stderr, "refusing reset/release: pass --cores-are-wfi only after all selected cores are in WFI\n");
        exit(2);
    }
    for (unsigned p = 0; p < PACC_CLUST_NUM; ++p) {
        if ((o->pacc_mask & (1u << p)) == 0) continue;
        printf("PACC%u entry=0x%016" PRIx64 "\n", p, o->entry);

        if (o->core_reset) {
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_CORE_RESET[c]), 0x3, o->dry_run);
            }
            if (o->init_crg)
                init_crg_1100_ref2pll(memfd, p, o->dry_run);
            program_cfg(memfd, o, p);
            if (o->send_base_cmd)
                send_base_cmd(memfd, o, p);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_CORE_RESET[c]), 0x0, o->dry_run);
            }
        } else if (o->sys_cold_reset) {
            mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_SYS_RESET), 0x1, o->dry_run);
            if (o->init_crg)
                init_crg_1100_ref2pll(memfd, p, o->dry_run);
            program_cfg(memfd, o, p);
            mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_SYS_RESET), 0x0, o->dry_run);
            if (o->send_base_cmd)
                send_base_cmd(memfd, o, p);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_CORE_RESET[c]), 0x0, o->dry_run);
            }
        } else if (o->release) {
            if (o->init_crg)
                init_crg_1100_ref2pll(memfd, p, o->dry_run);
            program_cfg(memfd, o, p);
            mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_SYS_RESET), 0x0, o->dry_run);
            if (o->send_base_cmd)
                send_base_cmd(memfd, o, p);
            for (unsigned c = 0; c < PACC_CORE_NUM; ++c) {
                if ((o->core_mask & (1u << c)) == 0) continue;
                mmio_write32(memfd, cfg_addr(p, LX500_PACC_CFG_CORE_RESET[c]), 0x0, o->dry_run);
            }
        }
    }
}

int main(int argc, char **argv) {
    struct opts o = {
        .pacc_mask = 0xf,
        .core_mask = 0xf,
        .reserved_size = 0x08000000ULL,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--status") == 0) o.status = true;
        else if (strcmp(argv[i], "--load-only") == 0) o.load_only = true;
        else if (strcmp(argv[i], "--release") == 0) o.release = true;
        else if (strcmp(argv[i], "--core-reset") == 0) o.core_reset = true;
        else if (strcmp(argv[i], "--sys-cold-reset") == 0) o.sys_cold_reset = true;
        else if (strcmp(argv[i], "--cores-are-wfi") == 0) o.cores_are_wfi = true;
        else if (strcmp(argv[i], "--dry-run") == 0) o.dry_run = true;
        else if (strcmp(argv[i], "--init-crg") == 0) o.init_crg = true;
        else if (strcmp(argv[i], "--send-base-cmd") == 0) o.send_base_cmd = true;
        else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) o.image = argv[++i];
        else if (strcmp(argv[i], "--load-addr") == 0 && i + 1 < argc) o.load_addr = parse_u64(argv[++i], "load-addr");
        else if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) o.entry = parse_u64(argv[++i], "entry");
        else if (strcmp(argv[i], "--pacc-mask") == 0 && i + 1 < argc) o.pacc_mask = parse_u32(argv[++i], "pacc-mask");
        else if (strcmp(argv[i], "--core-mask") == 0 && i + 1 < argc) o.core_mask = parse_u32(argv[++i], "core-mask");
        else if (strcmp(argv[i], "--reserved-size") == 0 && i + 1 < argc) o.reserved_size = parse_u64(argv[++i], "reserved-size");
        else if (strcmp(argv[i], "--secure-tieoff") == 0 && i + 1 < argc) {
            o.secure_tieoff = parse_u32(argv[++i], "secure-tieoff");
            o.have_secure_tieoff = true;
        } else if (strcmp(argv[i], "--mam-crm") == 0 && i + 1 < argc) {
            o.mam_crm = parse_u32(argv[++i], "mam-crm");
            o.have_mam_crm = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    int ops = (int)o.status + (int)o.load_only + (int)o.release + (int)o.core_reset + (int)o.sys_cold_reset;
    if (ops != 1 || (o.pacc_mask & ~0xfu) || (o.core_mask & ~0xfu)) {
        usage(argv[0]);
        return 2;
    }
    if (!o.status && (!o.image || o.load_addr == 0)) {
        usage(argv[0]);
        return 2;
    }
    if (!o.entry) o.entry = o.load_addr;

    int memfd = open_devmem(o.dry_run);
    if (o.status) {
        print_status(memfd, &o);
    } else {
        copy_image_to_phys(memfd, &o);
        if (!o.load_only) {
            boot_selected(memfd, &o);
        }
    }
    if (memfd >= 0) close(memfd);
    return 0;
}
