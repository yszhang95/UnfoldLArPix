"""Track-fit + projection dQ/dx, following pick_pgun2.compute_dqdx.

Pipeline per event and estimator:
  1. pixel charges -> points (y, z) in cm (drift x is fixed by design)
  2. tube selection: keep points within TUBE_CM of the line fitted to
     high-Q points (iterated once)
  3. PCA direction from high-Q points in the tube, project ALL tube points
  4. histogram the projection, BIN_CM wide, charge weighted
  5. drop the first and last NON-EMPTY bins (partial track coverage)
  6. dQ/dx = content / BIN_CM
"""
import numpy as np
from sklearn.decomposition import PCA

PITCH = 0.4434
LL = np.array([-62.076, 2.462])       # (y, z) lower-left of TPC0 pixel plane
BIN_CM = 3.0
TUBE_CM = 2.0
QTHRES = 10.0
NVALID = 5


def px_to_cm(pa, pb):
    return LL[0] + (pa + 0.5) * PITCH, LL[1] + (pb + 0.5) * PITCH


def fit_direction(yz, q):
    hi = q > QTHRES
    if hi.sum() < NVALID:
        return None, None
    c = yz[hi].mean(axis=0)
    p = PCA(n_components=1).fit(yz[hi] - c)
    d = p.components_[0]
    if d[-1] < 0:
        d = -d
    return c, d


def segment_dqdx(pa, pb, q):
    """Returns the per-segment dQ/dx array for one event."""
    y, z = px_to_cm(np.asarray(pa, float), np.asarray(pb, float))
    yz = np.stack([y, z], axis=1)
    q = np.asarray(q, float)
    c, d = fit_direction(yz, q)
    if c is None:
        return np.array([])
    # tube: perpendicular distance to the fitted line
    rel = yz - c
    proj = rel @ d
    perp = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    keep = perp < TUBE_CM
    if keep.sum() < NVALID:
        return np.array([])
    # refit on tube points (one iteration, delta rays rejected)
    c, d = fit_direction(yz[keep], q[keep])
    if c is None:
        return np.array([])
    rel = yz[keep] - c
    proj = rel @ d
    qk = q[keep]
    edges = np.arange(proj.min(), proj.max() + BIN_CM, BIN_CM)
    if len(edges) < 4:
        return np.array([])
    h, _ = np.histogram(proj, bins=edges, weights=qk)
    ne = np.nonzero(h > 0)[0]
    if len(ne) < 3:
        return np.array([])
    h = h[ne[0] + 1: ne[-1]]          # drop first & last non-empty bins
    return h[h > 0] / BIN_CM
