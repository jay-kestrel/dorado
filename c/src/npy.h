/* Minimal NumPy .npy reader/writer (uncompressed, version 1.0 format).
 *
 * The interface contract (spec, locked decision 5) gives the ports
 * harness-normalized .npy arrays: a short header plus raw values, so no
 * zip/npz support is needed. Only the dtypes the fixtures use are
 * accepted: '<f8' (float64) and '<i8' (int64), C order, 1 or 2 dimensions.
 * All fixtures are little-endian; big-endian files are rejected loudly.
 */
#ifndef DORADO_C_NPY_H
#define DORADO_C_NPY_H

#include <stddef.h>
#include <stdint.h>

typedef enum { NPY_F8 = 0, NPY_I8 = 1 } NpyDtype;

typedef struct {
    NpyDtype dtype;
    int ndim;          /* 1 or 2 */
    size_t shape[2];
    /* Raw data, converted to the natural host type. Exactly one of
     * f64/i64 points at a malloc'd array of shape[0]*shape[1] values. */
    double *f64;
    int64_t *i64;
} NpyArray;

/* Load "<path>"; returns NULL (and prints a diagnostic to stderr) on any
 * format/validation problem. */
NpyArray *npy_load(const char *path);

/* Save values as '<f8' (dtype == NPY_F8) or '<i8' C-order npy. Returns 0 on
 * success, nonzero on failure. */
int npy_save_f64(const char *path, const double *values, int ndim,
                 const size_t *shape);
int npy_save_i64(const char *path, const int64_t *values, int ndim,
                 const size_t *shape);

void npy_free(NpyArray *a);

#endif /* DORADO_C_NPY_H */
