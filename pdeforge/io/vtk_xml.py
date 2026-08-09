"""
A dependency-free reader for VTK XML files (.vtu / .vtp).

Enough of the format to get point coordinates and point data out of the
serial, inline-encoded files that CFD codes emit — no vtk, pyvista or meshio
required, which keeps a single interop loader from dragging a heavy
visualisation stack into PDEForge's install.

Inline data is base64. When the writer declares a compressor, VTK emits the
block header and the payload as TWO independent base64 streams, concatenated:
a (3 + nBlocks)-word header (nBlocks, uncompressed block size, last partial
block size, then one compressed size per block) followed by the zlib blocks.
That quirk is the only genuinely fiddly part of the format.

Not supported: ``format="appended"`` (raw binary after the XML, which is not
well-formed XML and needs a different parse), and parallel .pvtu/.pvtp files.
Both raise rather than returning something subtly wrong.
"""

import base64
import zlib
from pathlib import Path
from typing import Dict
from xml.etree import ElementTree

import numpy as np

_VTK_DTYPES = {
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
    "Float32": np.float32,
    "Float64": np.float64,
}


def _b64_len(n_bytes: int) -> int:
    """Length of the base64 encoding of n_bytes (with padding)."""
    return ((n_bytes + 2) // 3) * 4


def _uncompressed_candidates(b: str, head_b64: int, head_size: int):
    """Payload bytes under each inline layout, cheapest guess first."""
    # two streams: base64(header) then base64(data)
    try:
        yield base64.b64decode(b[head_b64:])
    except Exception:
        pass
    # one stream: header and data encoded together
    try:
        yield base64.b64decode(b)[head_size:]
    except Exception:
        pass


def _decode_array(text: str, dtype, header_dtype, compressed: bool) -> np.ndarray:
    b = "".join(text.split())
    if not b:
        return np.empty(0, dtype=dtype)

    head_size = np.dtype(header_dtype).itemsize
    head_b64 = _b64_len(head_size)

    if not compressed:
        # The header word is the first head_size bytes either way, so it can
        # be read before deciding the layout.
        n_bytes = int(
            np.frombuffer(base64.b64decode(b[:head_b64]), dtype=header_dtype, count=1)[
                0
            ]
        )
        # Writers differ: some emit base64(header) + base64(data) as two
        # streams (as the compressed path always does), others encode header
        # and data together as one. Take whichever yields exactly n_bytes.
        for candidate in _uncompressed_candidates(b, head_b64, head_size):
            if len(candidate) >= n_bytes:
                return np.frombuffer(candidate[:n_bytes], dtype=dtype)
        raise ValueError(
            f"uncompressed DataArray declares {n_bytes} bytes but neither "
            "inline layout provides them"
        )

    # nBlocks lives in the first header word; decode only enough b64 to read it.
    n_blocks = int(
        np.frombuffer(
            base64.b64decode(b[: _b64_len(head_size)]), dtype=header_dtype, count=1
        )[0]
    )
    header_b64 = _b64_len(head_size * (3 + n_blocks))
    header = np.frombuffer(
        base64.b64decode(b[:header_b64]), dtype=header_dtype, count=3 + n_blocks
    )

    payload = base64.b64decode(b[header_b64:])
    chunks, offset = [], 0
    for size in header[3:]:
        size = int(size)
        chunks.append(zlib.decompress(payload[offset : offset + size]))
        offset += size
    return np.frombuffer(b"".join(chunks), dtype=dtype)


def read_vtk_xml(path) -> Dict[str, np.ndarray]:
    """
    Read point coordinates and point data from a VTK XML .vtu / .vtp file.

    Returns a dict of arrays: ``"points"`` shaped (n_points, 3) plus one entry
    per PointData array, shaped (n_points,) for scalars and (n_points, c) for
    c-component vectors. Cell data and connectivity are not returned — the
    point cloud is what operator-learning interop needs.
    """
    path = Path(path)
    root = ElementTree.parse(path).getroot()

    if root.find(".//AppendedData") is not None:
        raise NotImplementedError(
            f"{path.name} uses appended raw data, which this reader does not "
            "support. Re-export with inline binary, or install meshio/pyvista."
        )

    compressed = "compressor" in root.attrib
    header_dtype = _VTK_DTYPES[root.get("header_type", "UInt32")]

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path.name}: no <Piece> found (parallel file?)")
    n_points = int(piece.get("NumberOfPoints", 0))

    out: Dict[str, np.ndarray] = {}
    for section, default_name in (("Points", "points"), ("PointData", None)):
        node = piece.find(section)
        if node is None:
            continue
        for da in node.findall("DataArray"):
            name = default_name or da.get("Name")
            if name is None:
                continue
            type_name = da.get("type")
            if type_name is None:
                raise ValueError(f"DataArray {name!r} has no 'type' attribute")
            dtype = _VTK_DTYPES[type_name]
            n_comp = int(da.get("NumberOfComponents", 1))
            fmt = da.get("format", "binary")
            if fmt == "ascii":
                values = np.fromstring(da.text or "", sep=" ", dtype=float).astype(
                    dtype
                )
            elif fmt == "binary":
                values = _decode_array(da.text or "", dtype, header_dtype, compressed)
            else:
                raise NotImplementedError(f"{path.name}: DataArray format {fmt!r}")
            out[name] = values.reshape(-1, n_comp) if n_comp > 1 else values

    out["n_points"] = n_points
    return out
