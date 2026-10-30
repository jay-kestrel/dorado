/* End-to-end comparator: byte-compares the driver's walk_data.json against
 * the committed golden produced by benchmarks/run_reference.py (see the
 * golden README). Byte equality is the strongest check: the C JSON writer
 * reproduces json.dumps formatting exactly (int-typed values, repr floats,
 * ', '/': ' separators). */
#include <stdio.h>
#include <stdlib.h>

static unsigned char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("FAIL: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        printf("FAIL: short read on %s\n", path);
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    *len = (size_t)n;
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("usage: %s <golden.json> <produced.json>\n", argv[0]);
        return 2;
    }
    size_t la, lb;
    unsigned char *a = slurp(argv[1], &la);
    unsigned char *b = slurp(argv[2], &lb);
    if (!a || !b)
        return 1;
    if (la != lb) {
        printf("FAIL: golden is %zu bytes, produced %zu bytes\n", la, lb);
        return 1;
    }
    for (size_t i = 0; i < la; i++) {
        if (a[i] != b[i]) {
            printf("FAIL: first difference at byte %zu: golden '%.20s', "
                   "produced '%.20s'\n", i, (const char *)&a[i],
                   (const char *)&b[i]);
            return 1;
        }
    }
    printf("walk_data.json matches the reference golden byte-for-byte "
           "(%zu bytes)\n", la);
    free(a);
    free(b);
    return 0;
}
