# Minimal reader for the uncompressed .npy files the Python fixture
# normalizer writes (benchmarks/normalize_fixtures.py) and the golden
# generator (julia/goldens/generate_goldens.py) re-dumps: little-endian
# float64/int64 (plus f4/i4/bool), C order. Only what those fixtures use —
# anything else is an error rather than a silent misread.

const NPY_MAGIC = UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')]

const NPY_DTYPES = Dict(
    "<f8" => Float64,
    "<i8" => Int64,
    "<f4" => Float32,
    "<i4" => Int32,
    "|b1" => Bool,
)

function _npy_header(path::AbstractString)
    bytes = read(path)
    (length(bytes) > 10 && bytes[1:6] == NPY_MAGIC) ||
        error("not a .npy file: $path")
    major = Int(bytes[7])
    if major == 1
        # v1: 2-byte little-endian header length at bytes 9..10, data after.
        hlen = UInt32(bytes[9]) | (UInt32(bytes[10]) << 8)
        hdr_off = 10
    elseif major == 2 || major == 3
        # v2/3: 4-byte little-endian header length at bytes 9..12, data after.
        hlen = UInt32(bytes[9]) | (UInt32(bytes[10]) << 8) |
               (UInt32(bytes[11]) << 16) | (UInt32(bytes[12]) << 24)
        hdr_off = 12
    else
        error("unsupported npy version $major: $path")
    end
    hdr = String(bytes[(hdr_off+1):(hdr_off+Int(hlen))])
    # 1-indexed: the first payload byte follows the consumed header.
    return hdr, bytes, hdr_off + Int(hlen) + 1
end

"""
    npread(path) -> Array

Read an uncompressed NumPy ``.npy`` file (little-endian `<f8`/`<i8`/`<f4`/
`<i4`/bool, C order, up to 3 dimensions). 1D data returns a `Vector`; 2D data
returns a `Matrix` indexed ``[row, col]`` matching NumPy's ``[x, y]``.
"""
function npread(path::AbstractString)
    hdr, bytes, data_off = _npy_header(path)

    m = match(r"'descr'\s*:\s*'([^']+)'", hdr)
    m === nothing && error("no descr in npy header: $path")
    descr = m.captures[1]
    T = get(NPY_DTYPES, descr, nothing)
    T === nothing && error("unsupported npy dtype '$descr' in $path")

    (startswith(descr, "<") || startswith(descr, "|")) ||
        error("only little-endian npy data is supported: $path")

    m = match(r"'fortran_order'\s*:\s*(True|False)", hdr)
    (m === nothing || m.captures[1] != "False") &&
        error("only C-order (fortran_order=False) npy files are supported: $path")

    m = match(r"'shape'\s*:\s*\(([^)]*)\)", hdr)
    m === nothing && error("no shape in npy header: $path")
    dims = [parse(Int, strip(s)) for s in split(m.captures[1], ',') if !isempty(strip(s))]
    (1 <= length(dims) <= 3) ||
        error("only 1D/2D/3D npy files are supported: $path")

    n = prod(dims)
    nbytes = n * sizeof(T)
    (data_off + nbytes - 1 <= length(bytes)) || error("npy data truncated: $path")
    flat = collect(reinterpret(T, view(bytes, data_off:(data_off + nbytes - 1))))

    if length(dims) == 1
        return flat
    end
    # NumPy C order is row-major: the first NumPy axis varies slowest. Julia
    # reshape is column-major, so reshape with the dims reversed puts NumPy's
    # last axis first, then permute back to NumPy axis order.
    A = reshape(flat, reverse(dims)...)
    return copy(permutedims(A, reverse(ntuple(identity, length(dims)))))
end
