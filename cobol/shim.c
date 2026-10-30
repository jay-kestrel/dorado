/* Dorado COBOL kernel shim: owns the one-time GnuCOBOL runtime startup
 * (cob_init) and exposes the two batched walk kernels plus the IEEE
 * helper ops the kernels use for float arithmetic.
 *
 * Why the helpers exist: GnuCOBOL 3.2 lowers COMP-2 COMPUTE to the GMP
 * decimal runtime, which rounds intermediates below binary64 precision
 * (~50% of multiplies differ from NumPy by 1 ULP) and decimalizes
 * non-finite operands. Each helper below is exactly one C floating
 * point op, bit-identical to the stepwise ops the Python reference
 * performs. Probe-validated against NumPy on 200k-element vectors and
 * all IEEE special values (NaN, +-inf, signed zeros).
 */
#include "libcob.h"
#include <math.h>
#include <stdint.h>

/* GnuCOBOL mangles PROGRAM-ID hyphens to double underscores in the
 * generated C symbols (PARTICLE-PICK -> PARTICLE__PICK). */
extern int PARTICLE__PICK(int32_t *k, int32_t *nw, int32_t *n,
                          double *rows, double *us, int32_t *out);
extern int PARTICLE__TRAV(int32_t *k, double *dist, double *ov,
                          double *nv, double *an, double *ao, double *du,
                          double *dm, double *vm, int32_t *mv,
                          double *dc, double *out);

static int cobol_ready = 0;

static void ensure_cobol(void) {
    if (!cobol_ready) {
        cob_init(0, NULL);
        cobol_ready = 1;
    }
}

/* Phase A: batched D8 direction pick. rows is k*nw doubles (nw is the
 * per-particle weight window width, 9 for the D8 walk), us is k
 * pre-drawn uniforms, out is k picked slot indices (0-based). */
int dorado_pick_cobol(int32_t *k, int32_t *nw, int32_t *n, double *rows,
                      double *us, int32_t *out) {
    ensure_cobol();
    return PARTICLE__PICK(k, nw, n, rows, us, out);
}

/* Phase B: batched travel times for the walk core. */
int dorado_travel_cobol(int32_t *k, double *dist, double *ov, double *nv,
                        double *an, double *ao, double *du, double *dm,
                        double *vm, int32_t *mv, double *dc,
                        double *out) {
    ensure_cobol();
    return PARTICLE__TRAV(k, dist, ov, nv, an, ao, du, dm, vm, mv, dc,
                          out);
}

/* libm cos exposed to COBOL via CALL "cobol_cos": GnuCOBOL's intrinsic
 * COS routes through the GMP decimal runtime and is not bit-compatible
 * with the libm cos the Python reference uses (math.cos). */
void cobol_cos(double *x, double *out) {
    *out = cos(*x);
}

/* IEEE-754 primitive helpers: each is exactly one C float op. */
void cobol_fadd(double *a, double *b, double *out) { *out = *a + *b; }
void cobol_fsub(double *a, double *b, double *out) { *out = *a - *b; }
void cobol_fmul(double *a, double *b, double *out) { *out = *a * *b; }
void cobol_fdiv(double *a, double *b, double *out) { *out = *a / *b; }

/* Exact three-way compare written to *out: -1, 0, +1 (2 = unordered,
 * at least one NaN). C comparisons are exact on the double
 * representation; the COBOL side uses this where near-tie correctness
 * is load-bearing (searchsorted-style scans, diff_coeff > 0 tests). */
void cobol_fcmp(double *a, double *b, int32_t *out) {
    if (*a < *b) *out = -1;
    else if (*a > *b) *out = 1;
    else if (*a == *b) *out = 0;
    else *out = 2;
}
