import math

import numpy as np

from vectors import SHAPE_MISMATCH_ERROR


def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -np.asarray(x)


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    X = np.asarray(x)
    if X.ndim != 2:
        raise SHAPE_MISMATCH_ERROR
    return np.flip(X)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    X = np.asarray(x, dtype=float)

    # Normalize input to (N, 2) points for processing, remember original layout.
    if X.ndim == 1 and X.shape == (2,):
        pts = X.reshape(1, 2)
        layout = "row"
    elif X.ndim == 2 and X.shape[1:] == (1,) and X.shape[0] == 2:
        pts = X.reshape(1, 2)
        layout = "colvec"
    elif X.ndim == 2 and X.shape[1] == 2:
        pts = X
        layout = "rows"
    elif X.ndim == 2 and X.shape[0] == 2:
        pts = X.T
        layout = "cols"
    else:
        raise SHAPE_MISMATCH_ERROR

    sx, sy = scale
    shx, shy = shear
    tx, ty = translate
    theta = math.radians(alpha_deg)

    # Homogeneous transform matrices
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]], dtype=float)

    # Shear matrix: x' = x + shx*y ; y' = shy*x + y
    Sh = np.array([[1, shx, 0],
                   [shy, 1, 0],
                   [0, 0, 1]], dtype=float)

    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta), math.cos(theta), 0],
                  [0, 0, 1]], dtype=float)

    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=float)

    # Total transform: T * R * Sh * S
    M = T @ R @ Sh @ S

    # Convert points to homogeneous coords: (N, 3)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.hstack([pts, ones])

    # Apply transform
    pts_t = (M @ pts_h.T).T[:, :2]

    # Restore original layout
    if layout == "rows":
        return pts_t
    if layout == "cols":
        return pts_t.T
    if layout == "row":
        return pts_t.reshape(2, )
    if layout == "colvec":
        return pts_t.reshape(2, 1)

    return pts_t
