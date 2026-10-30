#include "field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "npy.h"

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "field: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static double *alloc_grid(int L, int W)
{
    return xmalloc((size_t)L * (size_t)W * sizeof(double));
}

/* D8 constant tables (Particles.__init__ defaults). angles follow
 * math.pi like the reference: 3*pi/4 means (3*pi)/4. */
static const int IWALK[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
static const int JWALK[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
static const double ANGLES[9] = {
    3.0 * M_PI / 4.0, M_PI / 2.0, M_PI / 4.0,
    M_PI, 0.0, 0.0,
    5.0 * M_PI / 4.0, 3.0 * M_PI / 2.0, 7.0 * M_PI / 4.0
};

static void set_d8_defaults(Field *f)
{
    double sqrt2 = sqrt(2.0);
    double sqrt05 = sqrt(0.5);
    double d[9] = { sqrt2, 1, sqrt2, 1, 1, 1, sqrt2, 1, sqrt2 };
    double iv[9] = { -sqrt05, 0, sqrt05, -1, 0, 1, -sqrt05, 0, sqrt05 };
    double jv[9] = { -sqrt05, -1, -sqrt05, 0, 0, 0, sqrt05, 1, sqrt05 };
    memcpy(f->distances, d, sizeof(d));
    memcpy(f->ivec, iv, sizeof(iv));
    memcpy(f->jvec, jv, sizeof(jv));
    f->iwalk = IWALK;
    f->jwalk = JWALK;
    f->angles = ANGLES;
}

void field_set_nan_distances(Field *f)
{
    for (int k = 0; k < 9; k++)
        f->distances[k] = NAN;
}

Field *field_from_arrays(double *depth, double *stage, double *qx, double *qy,
                         int64_t *cell_type, int L, int W, double dx,
                         double theta, double gamma, double diff_coeff,
                         double dry_depth, bool steepest_descent,
                         bool delta_rcm)
{
    Field *f = xmalloc(sizeof(Field));
    memset(f, 0, sizeof(*f));
    f->L = L;
    f->W = W;
    f->dx = dx;
    f->theta = theta;
    f->gamma = gamma;
    f->diff_coeff = diff_coeff;
    f->dry_depth = dry_depth;
    f->steepest_descent = steepest_descent;
    f->delta_rcm = delta_rcm;
    set_d8_defaults(f);

    size_t n = (size_t)L * (size_t)W;
    f->depth = depth;
    f->stage = stage;
    f->qx = qx;
    f->qy = qy;
    f->cell_type = cell_type;
    f->u = alloc_grid(L, W);
    f->v = alloc_grid(L, W);
    f->velocity = alloc_grid(L, W);
    f->velocity_angle = alloc_grid(L, W);
    f->weight = xmalloc(n * 9 * sizeof(double));

    /* --- Particles.__init__ derivations, same op order as the reference --- */
    for (size_t i = 0; i < n; i++)
        if (isnan(f->depth[i]))
            f->depth[i] = 0.0;
    for (size_t i = 0; i < n; i++)
        if (f->depth[i] == 0.0)
            f->stage[i] = NAN;

    /* DeltaRCM: u = qx*depth/(depth**2 + 1e-8); non-DeltaRCM: the axis swap
     * qx = -1*qy, qy = qx happens first. NaN discharges were zeroed before
     * these expressions in the reference (isnan checks on the copies). */
    if (!f->delta_rcm) {
        for (size_t i = 0; i < n; i++) {
            double qxr = -1.0 * f->qy[i];
            double qyr = f->qx[i];
            f->qx[i] = isnan(qxr) ? 0.0 : qxr;
            f->qy[i] = isnan(qyr) ? 0.0 : qyr;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            if (isnan(f->qx[i]))
                f->qx[i] = 0.0;
            if (isnan(f->qy[i]))
                f->qy[i] = 0.0;
        }
    }

    for (size_t i = 0; i < n; i++) {
        double d2 = f->depth[i] * f->depth[i] + 1e-8;
        f->u[i] = f->qx[i] * f->depth[i] / d2;
        f->v[i] = f->qy[i] * f->depth[i] / d2;
        if (isnan(f->u[i]))
            f->u[i] = 0.0;
        if (isnan(f->v[i]))
            f->v[i] = 0.0;
    }

    /* velocity = sqrt(u**2 + v**2), then floored, then u/v tiny-floored,
     * then the angle (which sees the floored u/v) */
    for (size_t i = 0; i < n; i++)
        f->velocity[i] = sqrt(f->u[i] * f->u[i] + f->v[i] * f->v[i]);
    for (size_t i = 0; i < n; i++)
        if (f->velocity[i] < 1e-8)
            f->velocity[i] = 1e-8;
    for (size_t i = 0; i < n; i++) {
        /* (np.sign(u) + (u == 0.0)) * 1e-8 : exact zeros keep a positive
         * floor, tiny negatives keep their sign (parity with PR #5) */
        if (fabs(f->u[i]) < 1e-8) {
            double s = f->u[i] > 0.0 ? 1.0 : (f->u[i] < 0.0 ? -1.0 : 0.0);
            f->u[i] = (s + (f->u[i] == 0.0 ? 1.0 : 0.0)) * 1e-8;
        }
        if (fabs(f->v[i]) < 1e-8) {
            double s = f->v[i] > 0.0 ? 1.0 : (f->v[i] < 0.0 ? -1.0 : 0.0);
            f->v[i] = (s + (f->v[i] == 0.0 ? 1.0 : 0.0)) * 1e-8;
        }
    }
    for (size_t i = 0; i < n; i++)
        f->velocity_angle[i] = atan2(-1.0 * f->u[i], f->v[i]);

    field_make_weight(f);
    return f;
}

Field *field_load(const char *fixture_dir)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/params.json", fixture_dir);
    JsonValue *params = json_parse_file(path);
    if (!params) {
        fprintf(stderr, "field: cannot parse %s\n", path);
        return NULL;
    }
    const JsonValue *arrays = json_get(params, "arrays");
    if (!arrays || arrays->kind != JSON_OBJECT) {
        fprintf(stderr, "field: params.json missing 'arrays' map\n");
        json_free(params);
        return NULL;
    }
    const JsonValue *model = json_get(params, "model");
    if (!model || model->kind != JSON_STRING) {
        fprintf(stderr, "field: params.json missing 'model'\n");
        json_free(params);
        return NULL;
    }
    bool delta_rcm = strcmp(model->str_val, "DeltaRCM") == 0;

    const char *names[5] = { "depth", "stage", "qx", "qy", "cell_type" };
    NpyArray *arrs[5] = { NULL, NULL, NULL, NULL, NULL };
    for (int a = 0; a < 5; a++) {
        const JsonValue *file = json_get(arrays, names[a]);
        if (!file || file->kind != JSON_STRING) {
            fprintf(stderr, "field: params.json arrays missing '%s'\n", names[a]);
            goto fail;
        }
        snprintf(path, sizeof(path), "%s/%s", fixture_dir, file->str_val);
        arrs[a] = npy_load(path);
        if (!arrs[a])
            goto fail;
        if ((a < 4 && arrs[a]->dtype != NPY_F8) ||
            (a == 4 && arrs[a]->dtype != NPY_I8) || arrs[a]->ndim != 2) {
            fprintf(stderr, "field: %s has wrong dtype/ndim\n", names[a]);
            goto fail;
        }
    }
    for (int a = 1; a < 5; a++) {
        if (arrs[a]->shape[0] != arrs[0]->shape[0] ||
            arrs[a]->shape[1] != arrs[0]->shape[1]) {
            fprintf(stderr, "field: array shape mismatch on %s\n", names[a]);
            goto fail;
        }
    }

    const JsonValue *v;
    v = json_get(params, "dx");
    double dx = v ? v->num_val : 0.0;
    v = json_get(params, "theta");
    double theta = v ? v->num_val : 1.0;
    v = json_get(params, "gamma");
    double gamma = v ? v->num_val : 0.05;
    v = json_get(params, "diff_coeff");
    double diff_coeff = v ? v->num_val : 0.2;
    v = json_get(params, "dry_depth");
    double dry_depth = v ? v->num_val : 0.1;
    v = json_get(params, "steepest_descent");
    bool steepest = v ? v->bool_val : false;

    Field *f = field_from_arrays(
        arrs[0]->f64, arrs[1]->f64, arrs[2]->f64, arrs[3]->f64, arrs[4]->i64,
        (int)arrs[0]->shape[0], (int)arrs[0]->shape[1], dx, theta, gamma,
        diff_coeff, dry_depth, steepest, delta_rcm);

    /* the Field took over the data pointers; free the shells */
    for (int a = 0; a < 5; a++) {
        arrs[a]->f64 = NULL;
        arrs[a]->i64 = NULL;
        npy_free(arrs[a]);
    }
    json_free(params);
    return f;

fail:
    for (int a = 0; a < 5; a++)
        npy_free(arrs[a]);
    json_free(params);
    return NULL;
}

void field_free(Field *f)
{
    if (!f)
        return;
    free(f->depth);
    free(f->stage);
    free(f->qx);
    free(f->qy);
    free(f->u);
    free(f->v);
    free(f->velocity);
    free(f->velocity_angle);
    free(f->cell_type);
    free(f->weight);
    free(f);
}

void field_make_weight(Field *f)
{
    int L = f->L, W = f->W;
    /* neighbor offsets: k=0..8 -> (di, dj), matching raster[i-1:i+2,
     * j-1:j+2].ravel() = [NW, N, NE, W, 0, E, SW, S, SE] */
    static const int DI[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    static const int DJ[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

    double *w_sfc = xmalloc((size_t)L * W * 9 * sizeof(double));
    double *w_int = xmalloc((size_t)L * W * 9 * sizeof(double));
    double *depth_ind = xmalloc((size_t)L * W * 9 * sizeof(double));
    double *ct_ind = xmalloc((size_t)L * W * 9 * sizeof(double));

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < W; j++) {
            size_t cell = (size_t)i * W + j;
            bool interior = (i >= 1 && i <= L - 2 && j >= 1 && j <= W - 2);
            for (int k = 0; k < 9; k++) {
                size_t idx = cell * 9 + k;
                /* big_sliding_window: 0 on the border rows/cols */
                double sfc_win = 0.0, depth_win = 0.0, ct_win = 0.0;
                double stage_here = f->stage[cell];
                double qx_here = f->qx[cell];
                double qy_here = f->qy[cell];
                if (interior) {
                    size_t nb = (size_t)(i + DI[k]) * W + (size_t)(j + DJ[k]);
                    sfc_win = f->stage[nb];
                    depth_win = f->depth[nb];
                    ct_win = (double)f->cell_type[nb];
                }
                depth_ind[idx] = depth_win;
                ct_ind[idx] = ct_win;

                /* weight_sfc = tile(stage) - window(stage); /= distances */
                double sfc = stage_here - sfc_win;
                sfc /= f->distances[k];
                /* weight_int = tile(qx)*jvec + tile(qy)*ivec; /= distances */
                double inter = qx_here * f->jvec[k] + qy_here * f->ivec[k];
                inter /= f->distances[k];
                w_sfc[idx] = sfc;
                w_int[idx] = inter;
            }
        }
    }

    /* [w <= 0] = 0, then clear_borders (border cells were computed against
     * zero windows, then zeroed — replicate the net effect) */
    size_t n9 = (size_t)L * W * 9;
    /* weight_sfc[weight_sfc <= 0] = 0 : NaN compares false in numpy, so a
     * NaN surface slope survives this mask (it is excluded from the nansum
     * norm below) — same C semantics, no special case */
    for (size_t idx = 0; idx < n9; idx++) {
        if (w_sfc[idx] <= 0.0)
            w_sfc[idx] = 0.0;
        if (w_int[idx] <= 0.0)
            w_int[idx] = 0.0;
    }

    for (int k = 0; k < 9; k++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < W; j++) {
                if (i == 0 || i == L - 1 || j == 0 || j == W - 1) {
                    size_t idx = ((size_t)i * W + j) * 9 + k;
                    w_sfc[idx] = 0.0;
                    w_int[idx] = 0.0;
                }
            }
        }
    }

    /* dry / land mask zeroes both components (before normalization) */
    for (size_t idx = 0; idx < n9; idx++) {
        if (depth_ind[idx] <= f->dry_depth || ct_ind[idx] == 2.0) {
            w_sfc[idx] = 0.0;
            w_int[idx] = 0.0;
        }
    }

    /* normalize by the nansum over k where the sum is positive */
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < W; j++) {
            size_t cell = (size_t)i * W + j;
            double norm_sfc = 0.0, norm_int = 0.0;
            for (int k = 0; k < 9; k++) {
                double s = w_sfc[cell * 9 + k];
                double t = w_int[cell * 9 + k];
                if (!isnan(s))
                    norm_sfc += s;
                if (!isnan(t))
                    norm_int += t;
            }
            if (norm_sfc > 0.0)
                for (int k = 0; k < 9; k++)
                    w_sfc[cell * 9 + k] /= norm_sfc;
            if (norm_int > 0.0)
                for (int k = 0; k < 9; k++)
                    w_int[cell * 9 + k] /= norm_int;
        }
    }

    /* blend and depth^theta weighting */
    for (size_t idx = 0; idx < n9; idx++) {
        double w = f->gamma * w_sfc[idx] + (1.0 - f->gamma) * w_int[idx];
        f->weight[idx] = pow(depth_ind[idx], f->theta) * w;
    }

    /* re-apply the dry/land mask, then the dead-end deepest-cell fill */
    for (size_t idx = 0; idx < n9; idx++) {
        if (depth_ind[idx] <= f->dry_depth || ct_ind[idx] == 2.0)
            f->weight[idx] = 0.0;
    }

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < W; j++) {
            size_t cell = (size_t)i * W + j;
            double sum = 0.0, max_depth = -INFINITY;
            double masked_max = -INFINITY;
            bool any_not_land = false;
            for (int k = 0; k < 9; k++) {
                double w = f->weight[cell * 9 + k];
                if (!isnan(w))
                    sum += w;
                /* depth is NaN-free after init cleanup, so plain max
                 * matches np.max; NaN depth would propagate in numpy the
                 * same way all-false comparisons do here */
                double d = depth_ind[cell * 9 + k];
                if (d > max_depth)
                    max_depth = d;
                bool not_land = ct_ind[cell * 9 + k] != 2.0;
                if (not_land) {
                    any_not_land = true;
                    if (d > masked_max)
                        masked_max = d;
                }
            }
            bool zero_weights = sum <= 0.0;
            for (int k = 0; k < 9; k++) {
                size_t idx = cell * 9 + k;
                double d = depth_ind[idx];
                bool not_land = ct_ind[idx] != 2.0;
                bool deepest = (d == max_depth);
                bool deepest_dry = (d == masked_max) && not_land;
                bool deepest_cells = (deepest_dry) |
                                     ((deepest) & (!any_not_land));
                if (zero_weights && deepest_cells)
                    f->weight[idx] = 1.0;
            }
        }
    }

    /* final clear_borders */
    for (int k = 0; k < 9; k++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < W; j++) {
                if (i == 0 || i == L - 1 || j == 0 || j == W - 1)
                    f->weight[((size_t)i * W + j) * 9 + k] = 0.0;
            }
        }
    }

    free(w_sfc);
    free(w_int);
    free(depth_ind);
    free(ct_ind);
}
