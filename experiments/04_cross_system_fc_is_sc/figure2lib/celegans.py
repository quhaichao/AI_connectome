"""Small HDF5 helpers for the Uzel et al. C. elegans dataset."""

from __future__ import annotations

import h5py
import numpy as np


def get_matlab_string(handle: h5py.File, reference) -> str:
    """Resolve nested MATLAB HDF5 references and decode a character array."""
    values = np.asarray(handle[reference]).ravel()
    while values.size and isinstance(values[0], h5py.h5r.Reference):
        values = np.asarray(handle[values[0]]).ravel()
    return "".join(chr(int(value)) for value in values).strip()
