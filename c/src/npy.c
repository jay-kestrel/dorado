#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* npy 1.0 layout: 6-byte magic, 2-byte version, 2-byte header length,
 * header dict (padded to a 64-byte total boundary), raw data. */

static const unsigned char NPY_MAGIC[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "npy: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static uint16_t rd_u16(const unsigned char *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void wr_u16(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)(v >> 8);
}

/* Pull "key": value out of the header dict. Returns a malloc'd copy of the
 * value text without quotes/whitespace, or NULL when absent. */
static char *header_value(const char *header, const char *key)
{
    const char *p = strstr(header, key);
    if (!p)
        return NULL;
    p += strlen(key);
    p = strchr(p, ':');
    if (!p)
        return NULL;
    p++;
    while (*p == ' ')
        p++;
    const char *end = p;
    if (*p == '(') {
        /* tuple value like (2, 3): scan to the closing paren, not the
         * first comma */
        end = strchr(p, ')');
        if (!end)
            return NULL;
        end++;
    } else {
        while (*end && *end != ',' && *end != '}' && *end != '\n')
            end++;
        while (end > p && (end[-1] == ' ' || end[-1] == '\n'))
            end--;
    }
    size_t n = (size_t)(end - p);
    char *out = xmalloc(n + 1);
    memcpy(out, p, n);
    out[n] = '\0';
    if (out[0] == '\'' || out[0] == '"') {
        /* strip surrounding quotes */
        memmove(out, out + 1, n - 2);
        out[n - 2] = '\0';
    }
    return out;
}

NpyArray *npy_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "npy: cannot open %s\n", path);
        return NULL;
    }
    unsigned char pre[8];
    if (fread(pre, 1, 8, f) != 8 || memcmp(pre, NPY_MAGIC, 6) != 0) {
        fprintf(stderr, "npy: %s is not an npy file\n", path);
        fclose(f);
        return NULL;
    }
    unsigned char major = pre[6];
    size_t header_len;
    if (major == 1) {
        /* bytes: [0..5] magic, [6] major, [7] minor, then u16 len */
        unsigned char len2[2];
        if (fread(len2, 1, 2, f) != 2) {
            fprintf(stderr, "npy: truncated header in %s\n", path);
            fclose(f);
            return NULL;
        }
        header_len = rd_u16(len2);
    } else {
        unsigned char len4[4];
        if (fread(len4, 1, 4, f) != 4) {
            fprintf(stderr, "npy: truncated header in %s\n", path);
            fclose(f);
            return NULL;
        }
        header_len = rd_u32(len4);
    }
    char *header = xmalloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) {
        fprintf(stderr, "npy: truncated header dict in %s\n", path);
        free(header);
        fclose(f);
        return NULL;
    }
    header[header_len] = '\0';

    char *descr = header_value(header, "'descr'");
    char *fortran = header_value(header, "'fortran_order'");
    char *shape = header_value(header, "'shape'");
    if (!descr || !fortran || !shape) {
        fprintf(stderr, "npy: missing header keys in %s\n", path);
        free(descr);
        free(fortran);
        free(shape);
        free(header);
        fclose(f);
        return NULL;
    }

    NpyArray *a = xmalloc(sizeof(NpyArray));
    memset(a, 0, sizeof(*a));
    int ok = 1;
    if (strcmp(descr, "<f8") == 0 || strcmp(descr, "|f8") == 0)
        a->dtype = NPY_F8;
    else if (strcmp(descr, "<i8") == 0 || strcmp(descr, "|i8") == 0)
        a->dtype = NPY_I8;
    else {
        fprintf(stderr, "npy: %s has unsupported dtype '%s' (want <f8 or <i8)\n",
                path, descr);
        ok = 0;
    }
    if (ok && strcmp(fortran, "False") != 0) {
        fprintf(stderr, "npy: %s is fortran-order; port contract is C order\n", path);
        ok = 0;
    }

    if (ok) {
        /* shape like (120, 240) or (5,): count the integer entries */
        a->ndim = 0;
        const char *c = shape;
        while (*c) {
            if (*c >= '0' && *c <= '9') {
                a->shape[a->ndim++] = (size_t)strtoull(c, (char **)&c, 10);
            } else {
                c++;
            }
        }
        if (a->ndim == 0 || a->ndim > 2) {
            fprintf(stderr, "npy: %s has unsupported ndim (%d)\n", path, a->ndim);
            ok = 0;
        }
    }

    if (ok) {
        size_t count = a->shape[0] * (a->ndim == 2 ? a->shape[1] : 1);
        if (a->dtype == NPY_F8) {
            a->f64 = xmalloc(count * sizeof(double));
            if (fread(a->f64, sizeof(double), count, f) != count) {
                fprintf(stderr, "npy: truncated data in %s\n", path);
                ok = 0;
            }
        } else {
            a->i64 = xmalloc(count * sizeof(int64_t));
            if (fread(a->i64, sizeof(int64_t), count, f) != count) {
                fprintf(stderr, "npy: truncated data in %s\n", path);
                ok = 0;
            }
        }
    }

    free(descr);
    free(fortran);
    free(shape);
    free(header);
    fclose(f);
    if (!ok) {
        npy_free(a);
        return NULL;
    }
    return a;
}

static int write_header(FILE *f, const char *descr, int ndim, const size_t *shape)
{
    char dict[256];
    if (ndim == 2)
        snprintf(dict, sizeof(dict),
                 "{'descr': '%s', 'fortran_order': False, 'shape': (%zu, %zu), }",
                 descr, shape[0], shape[1]);
    else
        snprintf(dict, sizeof(dict),
                 "{'descr': '%s', 'fortran_order': False, 'shape': (%zu,), }",
                 descr, shape[0]);
    size_t dlen = strlen(dict);
    /* total header = 10 bytes (magic+version+len) + dict; pad to 64 with
     * spaces and terminate with a newline, matching numpy.save */
    size_t total = 10 + dlen + 1;
    size_t pad = 64 - (total % 64);
    uint16_t header_len = (uint16_t)(dlen + pad + 1);
    unsigned char pre[10];
    memcpy(pre, NPY_MAGIC, 6);
    pre[6] = 1;
    pre[7] = 0;
    wr_u16(pre + 8, header_len);
    if (fwrite(pre, 1, 10, f) != 10)
        return -1;
    if (fwrite(dict, 1, dlen, f) != dlen)
        return -1;
    for (size_t i = 0; i < pad; i++)
        if (fputc(' ', f) == EOF)
            return -1;
    if (fputc('\n', f) == EOF)
        return -1;
    return 0;
}

static int save_values(const char *path, const char *descr, size_t elem_size,
                       int ndim, const size_t *shape, const void *values)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "npy: cannot write %s\n", path);
        return -1;
    }
    int rc = write_header(f, descr, ndim, shape);
    size_t count = shape[0] * (ndim == 2 ? shape[1] : 1);
    if (rc == 0 && fwrite(values, elem_size, count, f) != count)
        rc = -1;
    if (fclose(f) != 0)
        rc = -1;
    if (rc != 0)
        fprintf(stderr, "npy: failed to write %s\n", path);
    return rc;
}

int npy_save_f64(const char *path, const double *values, int ndim,
                 const size_t *shape)
{
    return save_values(path, "<f8", sizeof(double), ndim, shape, values);
}

int npy_save_i64(const char *path, const int64_t *values, int ndim,
                 const size_t *shape)
{
    return save_values(path, "<i8", sizeof(int64_t), ndim, shape, values);
}

void npy_free(NpyArray *a)
{
    if (!a)
        return;
    free(a->f64);
    free(a->i64);
    free(a);
}
