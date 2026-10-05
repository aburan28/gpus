/* SHA-256. OpenSSL EVP when built with -DGC_USE_OPENSSL (hardware SHA-NI on
 * most hosts), otherwise a compact portable implementation. Both produce
 * identical digests; the test suite checks a known vector. */
#include "internal.h"

#ifdef GC_USE_OPENSSL
#include <openssl/evp.h>

typedef struct { EVP_MD_CTX *ctx; } ossl_ctx;

const char *gc_sha256_impl(void) { return "openssl-evp"; }

void gc_sha256_init(gc_sha256_ctx *c)
{
    ossl_ctx *o = (ossl_ctx *)c->opaque;
    o->ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(o->ctx, EVP_sha256(), NULL);
}
void gc_sha256_update(gc_sha256_ctx *c, const void *data, size_t len)
{
    ossl_ctx *o = (ossl_ctx *)c->opaque;
    EVP_DigestUpdate(o->ctx, data, len);
}
void gc_sha256_final(gc_sha256_ctx *c, uint8_t out[GC_HASH_LEN])
{
    ossl_ctx *o = (ossl_ctx *)c->opaque;
    unsigned n = 0;
    EVP_DigestFinal_ex(o->ctx, out, &n);
    EVP_MD_CTX_free(o->ctx);
    o->ctx = NULL;
}
#else
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t buflen; } sw_ctx;

const char *gc_sha256_impl(void) { return "builtin"; }

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void block(sw_ctx *s, const uint8_t *p)
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4*i]<<24 | (uint32_t)p[4*i+1]<<16 | (uint32_t)p[4*i+2]<<8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i-15],7) ^ ROR(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = ROR(w[i-2],17) ^ ROR(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR(e,6)^ROR(e,11)^ROR(e,25);
        uint32_t ch = (e&f)^(~e&g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = ROR(a,2)^ROR(a,13)^ROR(a,22);
        uint32_t mj = (a&b)^(a&c)^(b&c);
        uint32_t t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d; s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

void gc_sha256_init(gc_sha256_ctx *c)
{
    sw_ctx *s = (sw_ctx *)c->opaque;
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->len = 0; s->buflen = 0;
}
void gc_sha256_update(gc_sha256_ctx *c, const void *data, size_t len)
{
    sw_ctx *s = (sw_ctx *)c->opaque;
    const uint8_t *p = data;
    s->len += len;
    if (s->buflen) {
        size_t take = 64 - s->buflen; if (take > len) take = len;
        memcpy(s->buf + s->buflen, p, take);
        s->buflen += take; p += take; len -= take;
        if (s->buflen == 64) { block(s, s->buf); s->buflen = 0; }
    }
    while (len >= 64) { block(s, p); p += 64; len -= 64; }
    if (len) { memcpy(s->buf, p, len); s->buflen = len; }
}
void gc_sha256_final(gc_sha256_ctx *c, uint8_t out[GC_HASH_LEN])
{
    sw_ctx *s = (sw_ctx *)c->opaque;
    uint64_t bits = s->len * 8;
    uint8_t pad = 0x80;
    gc_sha256_update(c, &pad, 1);
    uint8_t z = 0;
    while (s->buflen != 56) gc_sha256_update(c, &z, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8*i));
    gc_sha256_update(c, lb, 8);
    for (int i = 0; i < 8; i++) {
        out[4*i] = (uint8_t)(s->h[i]>>24); out[4*i+1] = (uint8_t)(s->h[i]>>16);
        out[4*i+2] = (uint8_t)(s->h[i]>>8); out[4*i+3] = (uint8_t)s->h[i];
    }
}
#endif

void gc_sha256(const void *data, size_t len, uint8_t out[GC_HASH_LEN])
{
    gc_sha256_ctx c;
    gc_sha256_init(&c);
    gc_sha256_update(&c, data, len);
    gc_sha256_final(&c, out);
}
