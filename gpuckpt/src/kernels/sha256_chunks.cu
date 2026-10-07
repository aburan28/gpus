/* SHA-256 of every fixed-size chunk of a device-resident image, one thread
 * per chunk.
 *
 * This file is compiled two ways:
 *   - as CUDA C++ at runtime by NVRTC (src/cuda_backend.c embeds it), where
 *     __CUDACC__ is defined and gc_sha256_chunks is the kernel entry point;
 *   - as plain C by the test suite (tests/kernel_test.c and the mock driver),
 *     where gc_sha256_chunk_at() is called directly for each thread index.
 * The per-chunk code is identical in both, so the CPU tests exercise the
 * exact arithmetic the GPU runs.
 *
 * The digest is plain SHA-256 of the chunk's bytes, bit-for-bit what the CPU
 * path computes, so GPU- and CPU-hashed snapshots share one store and every
 * stored chunk stays verifiable on the CPU.
 *
 * Throughput note: SHA-256 is sequential within a message, so parallelism is
 * across chunks. An image of N chunks launches N threads; images with fewer
 * chunks than the GPU has lanes leave it underused, which matters little
 * because such images are small.
 */
#ifdef __CUDACC__
typedef unsigned char      gc_u8;
typedef unsigned int       gc_u32;
typedef unsigned long long gc_u64;
#define GC_DEV   __device__ __forceinline__
#define GC_CONST __constant__
#else
#include <stdint.h>
typedef uint8_t  gc_u8;
typedef uint32_t gc_u32;
typedef uint64_t gc_u64;
#define GC_DEV   static inline
#define GC_CONST static const
#endif

GC_CONST gc_u32 gc_k256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

#define GC_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

/* One compression round over 16 big-endian words already in w[0..15]. */
GC_DEV void gc_k_compress(gc_u32 st[8], gc_u32 w[16])
{
    gc_u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        gc_u32 wi;
        if (i < 16) {
            wi = w[i];
        } else {
            gc_u32 w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            gc_u32 s0 = GC_ROR(w15, 7) ^ GC_ROR(w15, 18) ^ (w15 >> 3);
            gc_u32 s1 = GC_ROR(w2, 17) ^ GC_ROR(w2, 19) ^ (w2 >> 10);
            wi = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
            w[i & 15] = wi;
        }
        gc_u32 S1 = GC_ROR(e, 6) ^ GC_ROR(e, 11) ^ GC_ROR(e, 25);
        gc_u32 ch = (e & f) ^ (~e & g);
        gc_u32 t1 = h + S1 + ch + gc_k256[i] + wi;
        gc_u32 S0 = GC_ROR(a, 2) ^ GC_ROR(a, 13) ^ GC_ROR(a, 22);
        gc_u32 mj = (a & b) ^ (a & c) ^ (b & c);
        gc_u32 t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

GC_DEV gc_u32 gc_k_be32(const gc_u8 *p)
{
    return (gc_u32)p[0] << 24 | (gc_u32)p[1] << 16 | (gc_u32)p[2] << 8 | (gc_u32)p[3];
}

GC_DEV gc_u32 gc_k_bswap(gc_u32 x)
{
    return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

/* SHA-256 of len bytes at p, written to out[32]. */
GC_DEV void gc_k_sha256(const gc_u8 *p, gc_u64 len, gc_u8 *out)
{
    gc_u32 st[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    gc_u32 w[16];
    gc_u64 full = len / 64;
    /* Word loads when the chunk start is 4-byte aligned (always, for device
     * allocations and power-of-two chunk sizes); byte loads otherwise. */
    int aligned = (((gc_u64)p) & 3) == 0;
    for (gc_u64 blk = 0; blk < full; blk++) {
        const gc_u8 *q = p + blk * 64;
        if (aligned) {
            const gc_u32 *q32 = (const gc_u32 *)q;
            for (int i = 0; i < 16; i++) w[i] = gc_k_bswap(q32[i]);   /* little-endian host/GPU */
        } else {
            for (int i = 0; i < 16; i++) w[i] = gc_k_be32(q + 4 * i);
        }
        gc_k_compress(st, w);
    }
    /* tail: remaining bytes, 0x80, zero pad, 64-bit big-endian bit length */
    gc_u8 tail[128];
    gc_u64 rem = len - full * 64;
    const gc_u8 *q = p + full * 64;
    for (gc_u64 i = 0; i < 128; i++) tail[i] = 0;
    for (gc_u64 i = 0; i < rem; i++) tail[i] = q[i];
    tail[rem] = 0x80;
    gc_u64 tl = rem < 56 ? 64 : 128;
    gc_u64 bits = len * 8;
    for (int i = 0; i < 8; i++) tail[tl - 1 - i] = (gc_u8)(bits >> (8 * i));
    for (gc_u64 off = 0; off < tl; off += 64) {
        for (int i = 0; i < 16; i++) w[i] = gc_k_be32(tail + off + 4 * i);
        gc_k_compress(st, w);
    }
    for (int i = 0; i < 8; i++) {
        out[4 * i]     = (gc_u8)(st[i] >> 24);
        out[4 * i + 1] = (gc_u8)(st[i] >> 16);
        out[4 * i + 2] = (gc_u8)(st[i] >> 8);
        out[4 * i + 3] = (gc_u8)st[i];
    }
}

/* Hash chunk i of an image of `size` bytes cut into `chunk_size` pieces. */
GC_DEV void gc_sha256_chunk_at(gc_u64 i, const gc_u8 *base, gc_u64 size, gc_u64 chunk_size,
                               gc_u64 nchunks, gc_u8 *out)
{
    if (i >= nchunks) return;
    gc_u64 off = i * chunk_size;
    gc_u64 len = size - off < chunk_size ? size - off : chunk_size;
    gc_k_sha256(base + off, len, out + 32 * i);
}

#ifdef __CUDACC__
extern "C" __global__ void gc_sha256_chunks(const gc_u8 *base, gc_u64 size, gc_u64 chunk_size,
                                            gc_u64 nchunks, gc_u8 *out)
{
    gc_u64 i = (gc_u64)blockIdx.x * blockDim.x + threadIdx.x;
    gc_sha256_chunk_at(i, base, size, chunk_size, nchunks, out);
}
#endif
