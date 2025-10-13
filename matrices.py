import math

import numpy as np

from vectors import SHAPE_MISMATCH_ERROR, DIM_POSITIVE_INT_ERROR


POSITIVE_VALUES_ERROR = ValueError("Values must be positive integers")
MATRIX_DIMENSION_ERROR = ValueError("Matrix must be 2D")


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    if n <= 0 or m <= 0:
        raise POSITIVE_VALUES_ERROR

    return np.random.randn(n, m)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return np.asarray(x) * a


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    try:
        return x @ y
    except ValueError:
        raise SHAPE_MISMATCH_ERROR


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    if dim <= 0:
        raise DIM_POSITIVE_INT_ERROR

    return np.eye(dim, dtype=float)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise SHAPE_MISMATCH_ERROR

    return np.linalg.inv(x)


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return np.asarray(x).T


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise SHAPE_MISMATCH_ERROR
    return x * y


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    A = np.asarray(x, dtype=float)

    if A.ndim != 2:
        raise MATRIX_DIMENSION_ERROR

    n, m = A.shape
    if m == 0 or n == 0:
        return tuple()

    selected: list[int] = []
    if m == 1:
        return (0,) if np.linalg.norm(A[:, 0]) > 0 else tuple()

    B = np.zeros((n, 0), dtype=float)
    rank_B = 0

    for j in range(m):
        candidate = np.hstack([B, A[:, [j]]])
        new_rank = np.linalg.matrix_rank(candidate)

        if new_rank > rank_B:
            selected.append(j)
            B = candidate
            rank_B = new_rank
            if rank_B == min(n, m):
                break

    return tuple(selected)


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2:
        raise MATRIX_DIMENSION_ERROR

    if order == 'fro':
        return float(np.linalg.norm(X, ord='fro'))

    if order == 2:
        return float(np.linalg.norm(X, ord=2))

    if order == np.inf or (isinstance(order, float) and math.isinf(order)):
        return float(np.linalg.norm(X, ord=np.inf))

    raise ValueError("order must be 'fro', 2, or inf")
