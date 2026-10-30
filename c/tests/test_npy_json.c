/* Minimal npy reader/writer and JSON formatter tests. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npy.h"
#include "json.h"

static int failures = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            printf("FAIL %s: ", __func__);                        \
            printf(__VA_ARGS__);                                  \
            printf("\n");                                         \
            failures++;                                           \
        }                                                         \
    } while (0)

static void test_npy_roundtrip(void)
{
    const char *path = "/tmp/dorado_c_test_rt.npy";
    double data[6] = { 1.0, -2.5, 0.0, 1e300, NAN, 3.141592653589793 };
    const size_t shape[2] = { 2, 3 };
    if (npy_save_f64(path, data, 2, shape) != 0) {
        CHECK(0, "npy_save_f64 failed");
        return;
    }

    NpyArray *back = npy_load(path);
    CHECK(back != NULL, "npy_load");
    if (!back)
        return;
    CHECK(back->dtype == NPY_F8, "dtype");
    CHECK(back->ndim == 2, "ndim %d", back->ndim);
    CHECK(back->shape[0] == 2 && back->shape[1] == 3, "shape");
    for (int i = 0; i < 6; i++) {
        if (isnan(data[i]))
            CHECK(isnan(back->f64[i]), "nan at %d", i);
        else
            CHECK(back->f64[i] == data[i], "value %d", i);
    }
    npy_free(back);
}

static void test_json_formatting(void)
{
    char buf[64];
    json_fmt_double(buf, 7007593448.337235);
    CHECK(strcmp(buf, "7007593448.337235") == 0, "plain: %s", buf);

    json_fmt_double(buf, 1e-6);
    CHECK(strcmp(buf, "1e-06") == 0, "small: %s", buf);

    json_fmt_double(buf, 1.5e17);
    CHECK(strcmp(buf, "1.5e+17") == 0, "big: %s", buf);

    json_fmt_double(buf, 0.5609806565385976);
    CHECK(strcmp(buf, "0.5609806565385976") == 0, "travel: %s", buf);

    json_fmt_double(buf, -1.0);
    CHECK(strcmp(buf, "-1.0") == 0, "neg one: %s", buf);

    json_fmt_double(buf, 3.0);
    CHECK(strcmp(buf, "3.0") == 0, "three: %s", buf);

    json_fmt_double(buf, 0.0);
    CHECK(strcmp(buf, "0.0") == 0, "zero: %s", buf);
}

static void test_json_parse(void)
{
    const char *text = "{\"a\": 1, \"b\": [2.5, 3], \"c\": {\"d\": null}, "
                       "\"e\": \"hi\", \"f\": true}";
    JsonValue *root = json_parse(text);
    CHECK(root != NULL, "parse");
    if (!root)
        return;
    const JsonValue *a = json_get(root, "a");
    CHECK(a && a->kind == JSON_NUMBER && a->num_val == 1.0, "a");
    const JsonValue *b = json_get(root, "b");
    CHECK(b && b->kind == JSON_ARRAY && b->n_items == 2, "b");
    CHECK(b && b->items[0]->kind == JSON_NUMBER &&
          b->items[0]->num_val == 2.5, "b[0]");
    CHECK(b && b->items[1]->kind == JSON_NUMBER &&
          b->items[1]->num_val == 3.0, "b[1] int-typed");
    const JsonValue *f = json_get(root, "f");
    CHECK(f && f->kind == JSON_BOOL, "f");
    json_free(root);
}

int main(void)
{
    test_npy_roundtrip();
    test_json_formatting();
    test_json_parse();
    if (failures) {
        printf("test_npy_json: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_npy_json: ok\n");
    return 0;
}
