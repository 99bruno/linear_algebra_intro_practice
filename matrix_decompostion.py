import numpy as np

from scipy.linalg import lu

from vectors import SHAPE_MISMATCH_ERROR


def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2:
        raise SHAPE_MISMATCH_ERROR

    P, L, U = lu(X)
    return P, L, U


def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2:
        raise SHAPE_MISMATCH_ERROR

    Q, R = np.linalg.qr(X, mode="reduced")
    return Q, R


def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise SHAPE_MISMATCH_ERROR

    return float(np.linalg.det(X))


def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise SHAPE_MISMATCH_ERROR

    w, V = np.linalg.eig(X)
    return w, V


def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    X = np.asarray(x, dtype=float)

    if X.ndim != 2:
        raise SHAPE_MISMATCH_ERROR

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T
    return U, S, V
