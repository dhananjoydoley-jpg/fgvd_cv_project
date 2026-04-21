"""
src/graph/features.py
Extract per-pixel node features from a 64×64 crop.

Three feature types (paper §III-B-1):
  - RGB   : 3-channel raw colour
  - Gabor : texture via Gabor filter bank  (eq. 1)
  - Sobel : gradient magnitude              (eq. 2)

All outputs are numpy arrays shaped (H, W, C) with values in [0, 1].
"""

import numpy as np
import cv2
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# RGB
# ---------------------------------------------------------------------------

def extract_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Return H×W×3 float32 array in [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


# ---------------------------------------------------------------------------
# Gabor texture  (paper §III-B-1, eq. 1)
# ---------------------------------------------------------------------------

def extract_gabor(
    img_bgr: np.ndarray,
    wavelength: float = 6.0,   # λ — paper §IV
    frequency: float = 1.0,    # 1/λ in skimage convention
    gamma: float = 1.0,        # aspect ratio
    orientations: tuple = (0, 45, 90, 135),
) -> np.ndarray:
    """
    Apply a bank of Gabor filters at multiple orientations and return
    the mean response magnitude as H×W×len(orientations) float32.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    responses = []
    for theta_deg in orientations:
        theta = np.deg2rad(theta_deg)
        kernel = gabor_kernel(
            frequency=frequency,
            theta=theta,
            sigma_x=wavelength,
            sigma_y=wavelength / gamma,
        )
        real = convolve(gray, np.real(kernel))
        imag = convolve(gray, np.imag(kernel))
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        responses.append(magnitude)

    gabor_feat = np.stack(responses, axis=-1).astype(np.float32)
    # Normalise to [0,1]
    gabor_feat = (gabor_feat - gabor_feat.min()) / (gabor_feat.max() - gabor_feat.min() + 1e-8)
    return gabor_feat


# ---------------------------------------------------------------------------
# Sobel gradient  (paper §III-B-1, eq. 2)
# ---------------------------------------------------------------------------

def extract_sobel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Compute Sobel magnitude |G| = sqrt(Gx² + Gy²) on grayscale image.
    Returns H×W×1 float32 normalised to [0,1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # hx kernel
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # hy kernel
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude[..., np.newaxis]                  # H×W×1


# ---------------------------------------------------------------------------
# Combined extractor
# ---------------------------------------------------------------------------

FEATURE_EXTRACTORS = {
    "rgb":   extract_rgb,
    "gabor": extract_gabor,
    "sobel": extract_sobel,
}


def extract_features(img_bgr: np.ndarray, feature_types: list[str]) -> np.ndarray:
    """
    Concatenate selected features along the channel axis.

    Args:
        img_bgr: uint8 BGR image, already resized to 64×64.
        feature_types: subset of ['rgb', 'gabor', 'sobel'].

    Returns:
        np.ndarray of shape (H, W, C_total), float32, values in [0,1].
    """
    parts = []
    for ft in feature_types:
        if ft not in FEATURE_EXTRACTORS:
            raise ValueError(f"Unknown feature type '{ft}'. Choose from {list(FEATURE_EXTRACTORS)}")
        parts.append(FEATURE_EXTRACTORS[ft](img_bgr))
    return np.concatenate(parts, axis=-1)


def feature_dim(feature_types: list[str],
                n_gabor_orientations: int = 4) -> int:
    """Return the total number of feature channels for a given config."""
    dims = {"rgb": 3, "gabor": n_gabor_orientations, "sobel": 1}
    return sum(dims[ft] for ft in feature_types)
