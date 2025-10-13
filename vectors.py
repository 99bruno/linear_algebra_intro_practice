import math
from typing import Sequence

import numpy as np
from scipy import sparse


DIM_POSITIVE_INT_ERROR = ValueError("Dim must be a positive integer")
SHAPE_MISMATCH_ERROR = ValueError("Shape mismatch")


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    if dim <= 0:
        raise DIM_POSITIVE_INT_ERROR
    return np.random.randn(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    if dim <= 0:
        raise DIM_POSITIVE_INT_ERROR

    return sparse.random(
        dim,
        1,
    )


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    x = np.asarray(x).reshape(-1, 1)
    return a * x


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) == 0:
        raise ValueError("Vectors must be a non-empty Sequence")

    if len(vectors) != len(coeffs):
        raise ValueError("Vectors and Coeffs must have the same length")

    cols = [np.asarray(v).reshape(-1, 1) for v in vectors]
    first_shape = cols[0].shape

    if any(c.shape != first_shape for c in cols[1:]):
        raise SHAPE_MISMATCH_ERROR

    result = np.zeros_like(cols[0], dtype=float)
    for v, a in zip(cols, coeffs):
        result += a * v
    return result


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR

    return float(np.dot(x, y))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    x = np.asarray(x).ravel()

    if order not in (1, 2, np.inf) and not (isinstance(order, float) and math.isinf(order)):
        raise ValueError("order must be 1, 2, or inf")

    return float(np.linalg.norm(x, ord=order))


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR

    return float(np.linalg.norm(x - y))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        np.ndarray: angle in deg.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        raise ValueError("angle is undefined for zero vector(s)")

    cos_theta = float(np.dot(x, y) / (nx * ny))
    # Numerical safety: clamp into [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_rad = math.acos(cos_theta)
    return math.degrees(theta_rad)


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    dp = dot_product(x, y)
    return math.isclose(dp, 0.0, abs_tol=1e-12)


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim != 2:
        raise ValueError("a must be a 2D array (matrix)")
    if a.shape[0] != b.shape[0]:
        raise SHAPE_MISMATCH_ERROR

    # Prefer exact solve for square systems; otherwise use least squares.
    if a.shape[0] == a.shape[1]:
        x = np.linalg.solve(a, b)
    else:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)

    # Ensure column vector shape if b was 1D/column-like
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x
