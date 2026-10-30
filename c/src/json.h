/* Minimal JSON: recursive-descent parser plus a walk-data writer.
 *
 * Parser: covers the full JSON grammar well enough for params.json and the
 * golden walk-data comparisons (objects, arrays, strings, numbers,
 * true/false/null). The reference's params.json is flat and ASCII, but the
 * parser is a real parser so failures are loud, not silent misreads.
 *
 * Writer: reproduces Python json.dumps formatting for walk data — ", " and
 * ": " separators, ints as integers, and floats as the shortest decimal
 * string that round-trips (Python repr semantics), because the end-to-end
 * parity gate diffs the C output against the Python reference's JSON.
 */
#ifndef DORADO_C_JSON_H
#define DORADO_C_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,   /* double value; is_int records integer-literal syntax */
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonKind;

typedef struct JsonValue JsonValue;

struct JsonValue {
    JsonKind kind;
    bool bool_val;
    double num_val;
    bool is_int;          /* number written without '.' or exponent */
    char *str_val;
    JsonValue **items;    /* array/object values */
    char **keys;          /* object keys (parallel to items) */
    size_t n_items;
};

/* Parse text (NUL-terminated). Returns NULL on syntax error. */
JsonValue *json_parse(const char *text);

/* Convenience: parse a file; returns NULL on read or syntax error. */
JsonValue *json_parse_file(const char *path);

void json_free(JsonValue *v);

/* Object lookup (top level only). NULL when absent or not an object. */
const JsonValue *json_get(const JsonValue *obj, const char *key);

/* Growable output buffer with json.dumps-compatible formatting. */
typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} JsonBuf;

void jbuf_init(JsonBuf *b);
void jbuf_free(JsonBuf *b);
void jbuf_puts(JsonBuf *b, const char *s);

/* Append one number preserving int/float fidelity: is_int -> "0" style,
 * float -> shortest round-trip decimal ("0.0", "7007593448.337235"). */
void jbuf_number(JsonBuf *b, double v, bool is_int);

/* Shortest-round-trip representation of v (Python repr). Fails (exit) on
 * non-finite values, mirroring json.dumps(allow_nan=False). */
void json_fmt_double(char out[40], double v);

#endif /* DORADO_C_JSON_H */
