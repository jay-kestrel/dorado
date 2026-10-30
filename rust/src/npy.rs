//! Minimal uncompressed-`.npy` reader (NumPy 1.0/2.0/3.0 header
//! formats) for float32/float64 and int32/int64 arrays. The harness
//! normalizes npz fixtures to plain `.npy` bytes so the port needs no
//! zip support; see rust/README.md (interface contract). Values widen
//! to f64 — f32→f64 widening is exact, matching the reference's
//! in-run casts; cell-type integers are small enough to widen exactly.

use std::fs;
use std::path::Path;

/// A parsed `.npy` array: row-major (C-order) values widened to f64.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl NpyArray {
    /// Row-count of the array (first shape dimension).
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Column-count of the array (second shape dimension).
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}

/// Read an uncompressed `.npy` file of dtype `<f4`/`<f8`/`<f2`.
pub fn read_npy(path: impl AsRef<Path>) -> Result<NpyArray, String> {
    let bytes = fs::read(path.as_ref()).map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
    parse_npy(&bytes)
}

/// Parse the header dict and payload of an uncompressed `.npy` byte
/// stream. Header sizes: 1.x = 2-byte length, 2.0 = 4-byte, 3.0 = 4-byte
/// (UTF-8, same layout as 2.0 for our fixtures).
fn parse_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not an npy file (bad magic)".to_string());
    }
    let (header_len, dict_start) = match bytes[6] {
        1u8 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2u8 | 3u8 => {
            if bytes.len() < 12 {
                return Err("npy header truncated".to_string());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        v => return Err(format!("unsupported npy version byte {v}")),
    };
    let dict_end = dict_start + header_len;
    if dict_end > bytes.len() {
        return Err("npy header overruns file".to_string());
    }
    let header = String::from_utf8_lossy(&bytes[dict_start..dict_end]);

    let descr = header_field(&header, "descr").ok_or("npy header missing descr")?;
    let dtype = descr.trim().to_owned();
    let item_size = match dtype.as_str() {
        "<f4" | "|f4" | "<i4" | "|i4" => 4usize,
        "<f8" | "|f8" | "<i8" | "|i8" => 8usize,
        other => {
            return Err(format!(
                "unsupported npy dtype {other} (need float32/64 or int32/64)"
            ))
        }
    };
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err("fortran-order npy not supported (harness writes C order)".to_string());
    }

    let shape = parse_shape(&header).ok_or("npy header missing shape")?;
    let n: usize = shape.iter().product();
    let payload = &bytes[dict_end..];
    if payload.len() < n * item_size {
        return Err(format!(
            "npy payload truncated: need {} bytes, have {}",
            n * item_size,
            payload.len()
        ));
    }

    let data: Vec<f64> = match dtype.as_str() {
        "<f4" | "|f4" => payload[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        "<i4" | "|i4" => payload[..n * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        "<i8" | "|i8" => payload[..n * 8]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
            .collect(),
        _ => payload[..n * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
    };
    Ok(NpyArray { shape, data })
}

/// Pull a single-quoted scalar field out of the header dict.
fn header_field(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}': '");
    let start = header.find(&needle)? + needle.len();
    let end = header[start..].find('\'')? + start;
    Some(header[start..end].to_string())
}

/// Parse `shape: (a, b[, ...])` (or `(a,)`) from the header dict.
fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let marker = "'shape': (";
    let start = header.find(marker)? + marker.len();
    let end = header[start..].find(')')? + start;
    let inner = &header[start..end];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if dims.is_empty() {
        None
    } else {
        Some(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x3 float64 array serializes with a 1.x header; round-trip it.
    #[test]
    fn parses_f64_npy() {
        // header built to mirror np.save output for arange(6).reshape(2, 3)
        let dict = b"{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }";
        let pad = 64 - (dict.len() + 10) % 64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(dict.len() as u16 + pad as u16).to_le_bytes());
        bytes.extend_from_slice(dict);
        bytes.extend(vec![b' '; pad - 1]);
        bytes.push(b'\n');
        for v in [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let arr = parse_npy(&bytes).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse_npy(b"garbage data here").is_err());
    }
}
