#include "json.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "json: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

/* ---------------- parser ---------------- */

typedef struct {
    const char *cur;
} Parser;

static void skip_ws(Parser *p)
{
    while (*p->cur && isspace((unsigned char)*p->cur))
        p->cur++;
}

static JsonValue *parse_value(Parser *p);

static JsonValue *new_value(JsonKind kind)
{
    JsonValue *v = xmalloc(sizeof(JsonValue));
    memset(v, 0, sizeof(*v));
    v->kind = kind;
    return v;
}

static char *parse_string_raw(Parser *p)
{
    if (*p->cur != '"') {
        fprintf(stderr, "json: expected string at ...%.16s\n", p->cur);
        return NULL;
    }
    p->cur++;
    /* unescaped scan first: params.json/golden strings have no escapes, but
     * stay correct for the general case */
    size_t cap = 32, n = 0;
    char *out = xmalloc(cap);
    while (*p->cur && *p->cur != '"') {
        char c = *p->cur;
        if (c == '\\') {
            p->cur++;
            char e = *p->cur;
            switch (e) {
            case '"': c = '"'; break;
            case '\\': c = '\\'; break;
            case '/': c = '/'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            case 'u':
                fprintf(stderr, "json: \\u escapes unsupported\n");
                free(out);
                return NULL;
            default:
                fprintf(stderr, "json: bad escape '\\%c'\n", e);
                free(out);
                return NULL;
            }
            p->cur++;
        } else {
            p->cur++;
        }
        if (n + 1 >= cap) {
            cap *= 2;
            out = realloc(out, cap);
            if (!out) {
                fprintf(stderr, "json: out of memory\n");
                exit(1);
            }
        }
        out[n++] = c;
    }
    if (*p->cur != '"') {
        fprintf(stderr, "json: unterminated string\n");
        free(out);
        return NULL;
    }
    p->cur++;
    out[n] = '\0';
    return out;
}

static JsonValue *parse_number(Parser *p)
{
    const char *start = p->cur;
    char *end;
    double d = strtod(p->cur, &end);
    if (end == start) {
        fprintf(stderr, "json: bad number at ...%.16s\n", start);
        return NULL;
    }
    bool is_int = true;
    for (const char *c = start; c < end; c++)
        if (*c == '.' || *c == 'e' || *c == 'E')
            is_int = false;
    p->cur = end;
    JsonValue *v = new_value(JSON_NUMBER);
    v->num_val = d;
    v->is_int = is_int;
    return v;
}

static JsonValue *parse_array(Parser *p)
{
    p->cur++; /* '[' */
    JsonValue *v = new_value(JSON_ARRAY);
    size_t cap = 8;
    v->items = xmalloc(cap * sizeof(JsonValue *));
    skip_ws(p);
    if (*p->cur == ']') {
        p->cur++;
        return v;
    }
    for (;;) {
        JsonValue *item = parse_value(p);
        if (!item) {
            json_free(v);
            return NULL;
        }
        if (v->n_items + 1 > cap) {
            cap *= 2;
            v->items = realloc(v->items, cap * sizeof(JsonValue *));
            if (!v->items) {
                fprintf(stderr, "json: out of memory\n");
                exit(1);
            }
        }
        v->items[v->n_items++] = item;
        skip_ws(p);
        if (*p->cur == ',') {
            p->cur++;
            skip_ws(p);
            continue;
        }
        if (*p->cur == ']') {
            p->cur++;
            return v;
        }
        fprintf(stderr, "json: expected ',' or ']' at ...%.16s\n", p->cur);
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_object(Parser *p)
{
    p->cur++; /* '{' */
    JsonValue *v = new_value(JSON_OBJECT);
    size_t cap = 8;
    v->items = xmalloc(cap * sizeof(JsonValue *));
    v->keys = xmalloc(cap * sizeof(char *));
    skip_ws(p);
    if (*p->cur == '}') {
        p->cur++;
        return v;
    }
    for (;;) {
        skip_ws(p);
        char *key = parse_string_raw(p);
        if (!key) {
            json_free(v);
            return NULL;
        }
        skip_ws(p);
        if (*p->cur != ':') {
            fprintf(stderr, "json: expected ':' after key \"%s\"\n", key);
            free(key);
            json_free(v);
            return NULL;
        }
        p->cur++;
        skip_ws(p);
        JsonValue *item = parse_value(p);
        if (!item) {
            free(key);
            json_free(v);
            return NULL;
        }
        if (v->n_items + 1 > cap) {
            cap *= 2;
            v->items = realloc(v->items, cap * sizeof(JsonValue *));
            v->keys = realloc(v->keys, cap * sizeof(char *));
            if (!v->items || !v->keys) {
                fprintf(stderr, "json: out of memory\n");
                exit(1);
            }
        }
        v->keys[v->n_items] = key;
        v->items[v->n_items] = item;
        v->n_items++;
        skip_ws(p);
        if (*p->cur == ',') {
            p->cur++;
            continue;
        }
        if (*p->cur == '}') {
            p->cur++;
            return v;
        }
        fprintf(stderr, "json: expected ',' or '}' at ...%.16s\n", p->cur);
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_value(Parser *p)
{
    skip_ws(p);
    if (*p->cur == '{')
        return parse_object(p);
    if (*p->cur == '[')
        return parse_array(p);
    if (*p->cur == '"') {
        char *s = parse_string_raw(p);
        if (!s)
            return NULL;
        JsonValue *v = new_value(JSON_STRING);
        v->str_val = s;
        return v;
    }
    if (strncmp(p->cur, "true", 4) == 0) {
        p->cur += 4;
        JsonValue *v = new_value(JSON_BOOL);
        v->bool_val = true;
        return v;
    }
    if (strncmp(p->cur, "false", 5) == 0) {
        p->cur += 5;
        JsonValue *v = new_value(JSON_BOOL);
        v->bool_val = false;
        return v;
    }
    if (strncmp(p->cur, "null", 4) == 0) {
        p->cur += 4;
        return new_value(JSON_NULL);
    }
    if (*p->cur == '-' || isdigit((unsigned char)*p->cur))
        return parse_number(p);
    fprintf(stderr, "json: unexpected character '%c'\n", *p->cur);
    return NULL;
}

JsonValue *json_parse(const char *text)
{
    Parser p = { text };
    JsonValue *v = parse_value(&p);
    if (!v)
        return NULL;
    skip_ws(&p);
    if (*p.cur != '\0') {
        fprintf(stderr, "json: trailing characters after value\n");
        json_free(v);
        return NULL;
    }
    return v;
}

JsonValue *json_parse_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "json: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    rewind(f);
    char *text = xmalloc((size_t)size + 1);
    if (fread(text, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "json: cannot read %s\n", path);
        free(text);
        fclose(f);
        return NULL;
    }
    text[size] = '\0';
    fclose(f);
    JsonValue *v = json_parse(text);
    free(text);
    return v;
}

void json_free(JsonValue *v)
{
    if (!v)
        return;
    free(v->str_val);
    for (size_t i = 0; i < v->n_items; i++) {
        json_free(v->items[i]);
        if (v->keys)
            free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    free(v);
}

const JsonValue *json_get(const JsonValue *obj, const char *key)
{
    if (!obj || obj->kind != JSON_OBJECT)
        return NULL;
    for (size_t i = 0; i < obj->n_items; i++)
        if (strcmp(obj->keys[i], key) == 0)
            return obj->items[i];
    return NULL;
}

/* ---------------- writer ---------------- */

void jbuf_init(JsonBuf *b)
{
    b->cap = 1 << 16;
    b->buf = xmalloc(b->cap);
    b->len = 0;
    b->buf[0] = '\0';
}

void jbuf_free(JsonBuf *b)
{
    free(b->buf);
    b->buf = NULL;
    b->len = b->cap = 0;
}

static void jbuf_reserve(JsonBuf *b, size_t extra)
{
    if (b->len + extra + 1 > b->cap) {
        size_t cap = b->cap;
        while (cap < b->len + extra + 1)
            cap *= 2;
        b->buf = realloc(b->buf, cap);
        if (!b->buf) {
            fprintf(stderr, "json: out of memory\n");
            exit(1);
        }
        b->cap = cap;
    }
}

void jbuf_puts(JsonBuf *b, const char *s)
{
    size_t n = strlen(s);
    jbuf_reserve(b, n);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

void json_fmt_double(char out[40], double v)
{
    if (!isfinite(v)) {
        /* the reference serializes with allow_nan=False */
        fprintf(stderr, "json: non-finite value cannot be serialized\n");
        exit(1);
    }
    if (v == 0.0) {
        snprintf(out, 40, signbit(v) ? "-0.0" : "0.0");
        return;
    }
    if (v == floor(v) && fabs(v) < 1e16) {
        /* Python repr prints integral floats positionally ("1.0"); %g in
         * this range would print "1" or switch to exponent form */
        snprintf(out, 40, "%.1f", v);
        return;
    }
    /* shortest decimal that round-trips: Python repr's choice of digits */
    for (int prec = 15; prec <= 17; prec++) {
        snprintf(out, 40, "%.*g", prec, v);
        if (strtod(out, NULL) == v)
            return;
    }
}

void jbuf_number(JsonBuf *b, double v, bool is_int)
{
    char tmp[40];
    if (is_int && v == floor(v) && fabs(v) < 9e15)
        snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    else
        json_fmt_double(tmp, v);
    jbuf_puts(b, tmp);
}
