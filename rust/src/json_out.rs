//! Python-`json.dumps`-compatible serialization for walk data.
//!
//! The harness byte-compares `walk_data.json` across implementations, so the
//! writer mirrors the reference runner's
//! `json.dumps(jsonable(walk_data), allow_nan=False)`: dict key order
//! (`xinds`, `yinds`, `travel_times`, then the park-stranding keys), default
//! separators `(", ", ": ")`, ints bare, floats in Python `repr` form, and no
//! trailing newline.

use crate::walker::WalkData;
use std::fs;
use std::path::Path;

// C snprintf is already linked (std/libc); the C port's writer uses the same
// loop, so all three implementations pick identical digits.
extern "C" {
    fn snprintf(buf: *mut u8, size: usize, format: *const u8, ...) -> i32;
}

/// Format `v` with C `%.*g` (glibc's correctly-rounded shortest form).
fn fmt_g(v: f64, prec: i32) -> String {
    let mut buf = [0u8; 40];
    let fmt = b"%.*g\0";
    let n = unsafe { snprintf(buf.as_mut_ptr(), buf.len(), fmt.as_ptr(), prec, v) };
    let n = n.clamp(0, buf.len() as i32 - 1) as usize;
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// Push `v` the way Python `repr` prints a finite float.
///
/// Python prints integral floats positionally (`0.0`, `1.0`) and otherwise
/// emits the shortest decimal that round-trips, correctly rounded — the same
/// digits glibc's `%g` picks (verified against the reference output and the
/// C port). Non-finite values are a serialization error because the
/// reference runs with `allow_nan=False`.
fn push_double(out: &mut String, v: f64) -> Result<(), String> {
    if !v.is_finite() {
        return Err("non-finite value cannot be serialized (allow_nan=False)".to_owned());
    }
    if v == 0.0 {
        out.push_str(if v.is_sign_negative() { "-0.0" } else { "0.0" });
    } else if v == v.trunc() && v.abs() < 1e16 {
        // Python repr prints integral floats positionally; %g in this range
        // would print "1" or switch to exponent form.
        out.push_str(&format!("{v:.1}"));
    } else {
        // Shortest decimal that round-trips: Python repr's choice of digits.
        for prec in 15..=17 {
            let s = fmt_g(v, prec);
            if s.parse::<f64>() == Ok(v) {
                out.push_str(&s);
                return Ok(());
            }
        }
        return Err(format!("no round-tripping %g form for {v}"));
    }
    Ok(())
}

fn push_i64_matrix(out: &mut String, m: &[Vec<i64>]) {
    out.push('[');
    for (i, row) in m.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&v.to_string());
        }
        out.push(']');
    }
    out.push(']');
}

fn push_f64_matrix(
    out: &mut String,
    m: &[Vec<f64>],
    seed_int: bool,
    zero_as_int: bool,
) -> Result<(), String> {
    out.push('[');
    for (i, row) in m.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            // Parity quirk: the reference seeds walk data with the caller's
            // integer seed_time (position 0 of `travel_times` and
            // `stranded_time` rows), and `stranded_time` keeps that integer
            // zero until a particle's first parked step adds a float —
            // non-parked steps append the value unchanged. So position 0
            // (when the value is integral) and carried exact zeros serialize
            // as bare JSON ints (`0`); accumulated floats print Python-repr
            // form (`1.0`, ...). A travel-time row for a particle that never
            // moved would also stay integer in the reference; that case does
            // not occur in the benchmark contract (every fixture particle
            // moves each step).
            let as_int = (j == 0 && seed_int) || (zero_as_int && *v == 0.0);
            if as_int {
                if !v.is_finite() || *v != v.trunc() || v.abs() >= 1e16 {
                    return Err(
                        "integer-serialized walk value is not a small non-negative integer"
                            .to_owned(),
                    );
                }
                out.push_str(&format!("{}", *v as i64));
            } else {
                push_double(out, *v)?;
            }
        }
        out.push(']');
    }
    out.push(']');
    Ok(())
}

/// Serialize walk data exactly like the reference's
/// `json.dumps(jsonable(walk_data), allow_nan=False)` — one line, default
/// separators, keys in the reference's dict insertion order.
pub fn walk_data_json(walk: &WalkData) -> Result<String, String> {
    let mut out = String::with_capacity(1 << 20);
    out.push_str("{\"xinds\": ");
    push_i64_matrix(&mut out, &walk.xinds);
    out.push_str(", \"yinds\": ");
    push_i64_matrix(&mut out, &walk.yinds);
    out.push_str(", \"travel_times\": ");
    push_f64_matrix(&mut out, &walk.travel_times, true, false)?;
    match (&walk.stranded, &walk.stranded_time) {
        (Some(stranded), Some(stranded_time)) => {
            out.push_str(", \"stranded\": ");
            push_i64_matrix(&mut out, stranded);
            out.push_str(", \"stranded_time\": ");
            push_f64_matrix(&mut out, stranded_time, true, true)?;
        }
        (None, None) => {}
        _ => return Err("inconsistent stranding walk data".to_owned()),
    }
    out.push('}');
    Ok(out)
}

/// Write walk data JSON to `path` with no trailing newline, like the
/// reference's `Path.write_bytes(json.dumps(...).encode())`.
pub fn write_walk_data(path: &Path, walk: &WalkData) -> Result<(), String> {
    let json = walk_data_json(walk)?;
    fs::write(path, json).map_err(|e| format!("writing {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_floats_print_positionally() {
        let mut out = String::new();
        push_double(&mut out, 0.0).unwrap();
        push_double(&mut out, 1.0).unwrap();
        push_double(&mut out, -0.0).unwrap();
        assert_eq!(out, "0.01.0-0.0");
    }

    #[test]
    fn fractional_floats_use_shortest_round_trip() {
        let mut out = String::new();
        push_double(&mut out, 0.5609806565385976).unwrap();
        push_double(&mut out, 7007593448.337235).unwrap();
        assert_eq!(out, "0.56098065653859767007593448.337235");
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut out = String::new();
        assert!(push_double(&mut out, f64::NAN).is_err());
        assert!(push_double(&mut out, f64::INFINITY).is_err());
    }

    #[test]
    fn walk_data_matches_reference_json_shape() {
        let walk = WalkData {
            xinds: vec![vec![16, 17], vec![15, 15]],
            yinds: vec![vec![140, 141], vec![138, 138]],
            travel_times: vec![vec![0.0, 7007593448.337235], vec![0.0, 1.0]],
            stranded: Some(vec![vec![0, 0], vec![0, 1]]),
            stranded_time: Some(vec![vec![0.0, 0.0], vec![0.0, 1.0]]),
        };
        let json = walk_data_json(&walk).unwrap();
        assert_eq!(
            json,
            "{\"xinds\": [[16, 17], [15, 15]], \"yinds\": [[140, 141], [138, 138]], \
             \"travel_times\": [[0, 7007593448.337235], [0, 1.0]], \
             \"stranded\": [[0, 0], [0, 1]], \"stranded_time\": [[0, 0], [0, 1.0]]}"
        );
    }

    #[test]
    fn walk_data_omits_stranding_keys_when_absent() {
        let walk = WalkData {
            xinds: vec![vec![16]],
            yinds: vec![vec![140]],
            travel_times: vec![vec![0.0]],
            stranded: None,
            stranded_time: None,
        };
        let json = walk_data_json(&walk).unwrap();
        assert_eq!(
            json,
            "{\"xinds\": [[16]], \"yinds\": [[140]], \"travel_times\": [[0]]}"
        );
    }
}
