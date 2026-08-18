"""Core matrix operations for the Figure 2 analysis.

Input similarity (IS) is defined consistently as the Pearson correlation
between incoming structural-connectivity profiles (matrix columns).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, skew, spearmanr


def as_square_matrix(matrix, name: str = "matrix", *, copy: bool = True) -> np.ndarray:
    """Return a finite-valued square matrix from a DataFrame or array."""
    values = matrix.to_numpy() if hasattr(matrix, "to_numpy") else np.asarray(matrix)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be square; got shape {values.shape}.")
    return values.copy() if copy else values


def prepare_fc_sc(fc_matrix, sc_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Validate FC and SC, preserve direction, and zero their diagonals."""
    fc = as_square_matrix(fc_matrix, "FC")
    sc = as_square_matrix(sc_matrix, "SC")
    if fc.shape != sc.shape:
        raise ValueError(f"FC and SC shapes differ: {fc.shape} versus {sc.shape}.")
    fc[~np.isfinite(fc)] = 0.0
    sc[~np.isfinite(sc)] = 0.0
    np.fill_diagonal(fc, 0.0)
    np.fill_diagonal(sc, 0.0)
    return fc, sc


def calculate_pearson_similarity_matrix(input_matrix) -> np.ndarray:
    """Correlate incoming SC profiles; log-transform clearly skewed nonnegative SC."""
    matrix = as_square_matrix(input_matrix, "SC")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("SC contains NaN or Inf.")
    if matrix.shape[0] < 2:
        raise ValueError("SC must contain at least two nodes.")
    np.fill_diagonal(matrix, 0.0)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(off_diagonal))))
    nonnegative = bool(np.min(off_diagonal) >= -tolerance)
    if nonnegative:
        matrix[matrix < 0] = 0.0
    positive = off_diagonal[off_diagonal > tolerance]
    skewness = float(skew(positive, bias=False)) if positive.size >= 3 else np.nan
    scale = float(np.median(positive)) if positive.size else np.nan
    tail_ratio = float(np.percentile(positive, 95) / scale) if positive.size else np.nan
    use_log = nonnegative and positive.size >= 20 and (skewness >= 1.0 or tail_ratio >= 10.0)
    if use_log:
        matrix = np.log1p(matrix / scale)
    transform = "log1p" if use_log else "raw"
    print(f"SC-to-IS transform: {transform} | nonnegative={nonnegative} | n_positive={positive.size} | skew={skewness:.2f} | p95/median={tail_ratio:.2f} | scale={scale:.6g}")
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.corrcoef(matrix, rowvar=False)
    similarity = np.asarray(similarity, dtype=float)
    similarity[~np.isfinite(similarity)] = 0.0
    similarity = (similarity + similarity.T) / 2.0
    np.fill_diagonal(similarity, 1.0)
    return similarity

def extract_upper_triangle(matrix, k: int = 1) -> np.ndarray:
    """Return the upper-triangle values of a square matrix."""
    values = as_square_matrix(matrix, copy=False)
    return values[np.triu_indices_from(values, k=k)]


def paired_matrix_values(first, second) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite off-diagonal values, preserving both directions."""
    first = as_square_matrix(first, copy=False)
    second = as_square_matrix(second, copy=False)
    if first.shape != second.shape:
        raise ValueError(f"Matrix shapes differ: {first.shape} versus {second.shape}.")
    mask = ~np.eye(first.shape[0], dtype=bool)
    first_values, second_values = first[mask], second[mask]
    valid = np.isfinite(first_values) & np.isfinite(second_values)
    return first_values[valid], second_values[valid]


def matrix_correlation(first, second, method: str = "pearson") -> tuple[float, float]:
    """Correlate corresponding off-diagonal entries of two matrices."""
    x, y = paired_matrix_values(first, second)
    if x.size < 2:
        return np.nan, np.nan
    if method == "pearson":
        statistic, p_value = pearsonr(x, y)
    elif method == "spearman":
        statistic, p_value = spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    return float(statistic), float(p_value)


def prepare_system_matrices(
    fc_matrix,
    sc_matrix,
    is_matrix=None,
) -> dict[str, np.ndarray]:
    """Prepare FC/SC and either validate or derive Pearson-based IS."""
    fc, sc = prepare_fc_sc(fc_matrix, sc_matrix)
    if is_matrix is None:
        similarity = calculate_pearson_similarity_matrix(sc)
    else:
        similarity = as_square_matrix(is_matrix, "IS")
        if similarity.shape != fc.shape:
            raise ValueError(
                f"IS shape {similarity.shape} does not match FC/SC shape {fc.shape}."
            )
        similarity[~np.isfinite(similarity)] = 0.0
        np.fill_diagonal(similarity, 1.0)
    return {
        "FC": fc,
        "IS": similarity,
        "SC": sc,
    }
