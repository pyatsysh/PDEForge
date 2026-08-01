"""
Read PyTorch ``.pt`` files without PyTorch.

The canonical operator-learning datasets are distributed as ``torch.save``
archives, which would otherwise make a ~2 GB deep-learning framework a
prerequisite for looking at a plain array of floats. It is not needed: since
PyTorch 1.6 a ``.pt`` file is an uncompressed zip holding one pickle plus the
raw little-endian storage bytes, so the tensors can be memory-mapped straight
into numpy.

Only the tensor-loading path is implemented. Anything else a pickle can carry
(modules, optimiser state, custom classes) raises rather than being silently
approximated -- see :func:`read_torch_pt`.
"""

import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["read_torch_pt"]

# torch.<X>Storage -> numpy dtype. Bfloat16 has no numpy equivalent and is
# absent from every dataset this reader targets, so it is left out.
_STORAGE_DTYPES = {
    "DoubleStorage": np.dtype("<f8"),
    "FloatStorage": np.dtype("<f4"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
    "ComplexFloatStorage": np.dtype("<c8"),
    "ComplexDoubleStorage": np.dtype("<c16"),
}

_DTYPES = {
    "float64": np.dtype("<f8"),
    "float32": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "int16": np.dtype("<i2"),
    "int8": np.dtype("i1"),
    "uint8": np.dtype("u1"),
    "bool": np.dtype("?"),
    "complex64": np.dtype("<c8"),
    "complex128": np.dtype("<c16"),
}


class _Storage:
    """A pickled storage, resolved to bytes only when a tensor needs it."""

    def __init__(self, key: str, dtype: np.dtype):
        self.key = key
        self.dtype = dtype


class _StorageType:
    """Stand-in for ``torch.FloatStorage`` and friends."""

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype


def _rebuild_tensor(reader, storage, storage_offset, size, stride, *rest):
    """numpy equivalent of ``torch._utils._rebuild_tensor_v2``."""
    itemsize = storage.dtype.itemsize
    base = reader._storage_array(storage)
    if not size:  # 0-d tensor
        return base[storage_offset : storage_offset + 1].reshape(())
    view = base[storage_offset:]
    # A tensor is a strided view of its storage; strides are in elements.
    return np.lib.stride_tricks.as_strided(
        view, shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )


class _Unpickler(pickle.Unpickler):
    """Resolves torch names to numpy, and refuses everything else."""

    def __init__(self, file, reader):
        super().__init__(file, encoding="utf-8")
        self._reader = reader

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("torch"):
            if name in _STORAGE_DTYPES:
                return _StorageType(_STORAGE_DTYPES[name])
            if name in _DTYPES:
                return _StorageType(_DTYPES[name])
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return lambda *a: _rebuild_tensor(self._reader, *a)
            if name == "TypedStorage":  # only ever reached via persistent_id
                return _StorageType
            if name == "OrderedDict":
                return dict
            raise NotImplementedError(
                f"{module}.{name} is not a plain tensor; this reader loads "
                "tensor data only (no modules, optimisers or custom classes)"
            )
        if module in ("collections", "builtins", "__builtin__"):
            return super().find_class(module, name)
        raise NotImplementedError(
            f"refusing to unpickle {module}.{name}: only tensors and plain "
            "containers are supported"
        )

    def persistent_load(self, pid) -> _Storage:
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise NotImplementedError(f"unsupported persistent id: {pid!r}")
        _, storage_type, key, _location, _numel = pid
        return _Storage(str(key), storage_type.dtype)


class _PtReader:
    def __init__(self, path: Path, mmap: bool):
        self.path = Path(path)
        self.mmap = mmap
        self.zf = zipfile.ZipFile(self.path)
        names = self.zf.namelist()
        pkl = [n for n in names if n.endswith("data.pkl")]
        if not pkl:
            raise ValueError(
                f"{self.path} has no data.pkl: legacy (pre-1.6) .pt files and "
                "raw pickles are not supported"
            )
        self.prefix = pkl[0][: -len("data.pkl")]
        self._cache: Dict[str, np.ndarray] = {}

    def _storage_array(self, storage: _Storage) -> np.ndarray:
        if storage.key in self._cache:
            return self._cache[storage.key]
        name = f"{self.prefix}data/{storage.key}"
        info = self.zf.getinfo(name)
        if self.mmap and info.compress_type == zipfile.ZIP_STORED:
            # torch writes storages uncompressed, so they can be mapped in
            # place -- the whole point of not materialising a 7 GB file.
            with self.zf.open(name) as fh:
                offset = fh._orig_compress_start  # type: ignore[attr-defined]
            arr = np.memmap(
                self.path,
                dtype=storage.dtype,
                mode="r",
                offset=offset,
                shape=(info.file_size // storage.dtype.itemsize,),
            )
        else:
            arr = np.frombuffer(self.zf.read(name), dtype=storage.dtype)
        self._cache[storage.key] = arr
        return arr

    def load(self) -> Any:
        with self.zf.open(f"{self.prefix}data.pkl") as fh:
            return _Unpickler(fh, self).load()


def read_torch_pt(path, mmap: bool = True, keys: Optional[list] = None) -> Any:
    """
    Load a ``torch.save`` archive as numpy arrays.

    Parameters
    ----------
    path : str or Path
        A ``.pt``/``.pth`` file written by PyTorch 1.6 or later.
    mmap : bool
        Memory-map the storages instead of reading them (default). The
        canonical Darcy files are ~7 GB each, so this matters; the returned
        arrays are read-only views onto the file.
    keys : list of str, optional
        For a dict payload, load only these entries.

    Returns
    -------
    The stored object with every tensor replaced by a numpy array: usually a
    dict such as ``{"x": (N, r, r) float32, "y": ...}``.

    Raises
    ------
    NotImplementedError
        If the archive holds anything but tensors and plain containers.
    """
    obj = _PtReader(Path(path), mmap).load()
    if keys is not None:
        if not isinstance(obj, dict):
            raise TypeError(f"keys= given but {path} holds a {type(obj).__name__}")
        return {k: obj[k] for k in keys}
    return obj
