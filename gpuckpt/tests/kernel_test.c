/* Runs the GPU SHA-256 kernel's per-chunk code (compiled as C) against the
 * library's SHA-256 over lengths that hit every padding branch, unaligned
 * starts, and a multi-chunk image with a short tail. */
#include "internal.h"
#include "kernels/sha256_chunks.cu"

static int check(const uint8_t *p, uint64_t len, const char *what)
{
    uint8_t a[32], b[32];
    gc_k_sha256(p, len, a);
    gc_sha256(p, (size_t)len, b);
    if (memcmp(a, b, 32)) { printf("FAIL kernel sha256 %s len=%llu\n", what, (unsigned long long)len); return 1; }
    return 0;
}

int main(void)
{
    size_t cap = (2u << 20) + 64;
    uint8_t *buf = malloc(cap);
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < cap; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; buf[i] = (uint8_t)x; }
    int bad = 0, n = 0;
    uint64_t lens[] = {0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1000, 4095, 4096,
                       65537, (1u << 20) - 1, 1u << 20, (1u << 20) + 1};
    for (size_t i = 0; i < sizeof lens / sizeof lens[0]; i++)
        for (int mis = 0; mis < 4; mis++) { bad += check(buf + mis, lens[i], mis ? "unaligned" : "aligned"); n++; }
    /* FIPS 180-2 vector */
    uint8_t d[32], want[32];
    gc_k_sha256((const uint8_t *)"abc", 3, d);
    gc_unhex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", want, 32);
    if (memcmp(d, want, 32)) { printf("FAIL kernel sha256(abc)\n"); bad++; }
    n++;
    /* whole-image entry point: 2 MiB + 17 bytes in 64 KiB chunks */
    uint64_t size = (2u << 20) + 17, cs = 65536, nch = (size + cs - 1) / cs;
    uint8_t *out = calloc(nch, 32);
    for (uint64_t i = 0; i < nch + 3; i++) gc_sha256_chunk_at(i, buf, size, cs, nch, out); /* extra threads must no-op */
    for (uint64_t i = 0; i < nch; i++) {
        uint8_t h[32];
        uint64_t len = size - i * cs < cs ? size - i * cs : cs;
        gc_sha256(buf + i * cs, (size_t)len, h);
        if (memcmp(h, out + 32 * i, 32)) { printf("FAIL chunk %llu\n", (unsigned long long)i); bad++; }
        n++;
    }
    printf("kernel_test: %d cases, %d failures\n", n, bad);
    free(out); free(buf);
    return bad != 0;
}
