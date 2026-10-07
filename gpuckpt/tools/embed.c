/* Build helper: emit a C source defining `const char NAME[]` with the bytes
 * of FILE plus a NUL terminator. Used to embed the CUDA kernel source for
 * NVRTC so the binary carries exactly the file the tests compile. */
#include <stdio.h>
int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: embed NAME FILE\n"); return 1; }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    printf("/* generated from %s by tools/embed.c; do not edit */\n", argv[2]);
    printf("const char %s[] = {\n", argv[1]);
    int c, n = 0;
    while ((c = fgetc(f)) != EOF) printf("%d,%s", c, (++n % 24) ? "" : "\n");
    printf("0};\n");
    fclose(f);
    return 0;
}
