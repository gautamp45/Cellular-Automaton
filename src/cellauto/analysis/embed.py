"""Embedding and clustering in plain numpy: standardise, PCA, k-means++, silhouette.

Deliberately small and dependency-free (no scikit-learn) so every step is inspectable and
deterministic: PCA components have a fixed sign, k-means takes an explicit seed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.intp]


def standardize(x: npt.ArrayLike) -> FloatMatrix:
    """Zero mean, unit variance per column (constant columns are left at zero)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array (samples x features)")
    std = a.std(axis=0)
    std[std == 0] = 1.0
    return np.asarray((a - a.mean(axis=0)) / std, dtype=np.float64)


def pca(x: npt.ArrayLike, k: int = 2) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """PCA via SVD of the centred data.

    Returns ``(scores (n, k), components (k, d), explained_variance_ratio (k,))``. Each component's
    sign is fixed so that its largest-magnitude loading is positive.
    """
    a = np.asarray(x, dtype=np.float64)
    if not 1 <= k <= min(a.shape):
        raise ValueError(f"k must be in 1..{min(a.shape)}")
    centred = a - a.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:k].copy()
    for i in range(k):
        if comps[i, np.argmax(np.abs(comps[i]))] < 0:
            comps[i] = -comps[i]
    var = s**2
    total = var.sum()
    ratio = var[:k] / total if total > 0 else np.zeros(k)
    return centred @ comps.T, comps, np.asarray(ratio, dtype=np.float64)


def _kmeans_pp(x: FloatMatrix, k: int, rng: np.random.Generator) -> FloatMatrix:
    n = x.shape[0]
    centres = [x[rng.integers(n)]]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centres.append(x[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(axis=1))
    return np.asarray(centres, dtype=np.float64)


def kmeans(
    x: npt.ArrayLike, k: int, n_init: int = 20, seed: int = 0, max_iter: int = 100
) -> tuple[IntVector, FloatMatrix, float]:
    """Lloyd's algorithm with k-means++ seeding; returns the best of ``n_init`` runs.

    Returns ``(labels, centroids, inertia)``. Labels are renumbered by decreasing cluster size
    (ties by first appearance) so results do not depend on the seeding order.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not 1 <= k <= a.shape[0]:
        raise ValueError("need a 2-D array with at least k samples")
    rng = np.random.default_rng(seed)
    best: tuple[float, IntVector, FloatMatrix] | None = None
    for _run in range(n_init):
        centres = _kmeans_pp(a, k, rng)
        labels = np.zeros(a.shape[0], dtype=np.intp)
        for it in range(max_iter):
            dist = ((a[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1).astype(np.intp)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = a[labels == j]
                if len(members):
                    centres[j] = members.mean(axis=0)
        inertia = float(((a - centres[labels]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, labels.copy(), centres.copy())
    assert best is not None
    inertia, best_labels, best_centres = best
    sizes = np.bincount(best_labels, minlength=k)
    first = np.array(
        [np.flatnonzero(best_labels == j)[0] if sizes[j] else a.shape[0] for j in range(k)]
    )
    order = np.lexsort((first, -sizes))  # decreasing size, then first appearance
    remap = np.empty(k, dtype=np.intp)
    remap[order] = np.arange(k)
    return remap[best_labels], best_centres[order], inertia


def silhouette(x: npt.ArrayLike, labels: npt.ArrayLike, chunk: int = 1024) -> float:
    """Mean silhouette coefficient (Euclidean); singleton clusters score 0."""
    a = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    classes = np.unique(lab)
    if classes.size < 2:
        raise ValueError("silhouette needs at least two clusters")
    onehot = (lab[:, None] == classes[None, :]).astype(np.float64)
    sizes = onehot.sum(axis=0)
    own = np.searchsorted(classes, lab)
    scores = np.zeros(a.shape[0])
    sq = (a**2).sum(axis=1)
    for start in range(0, a.shape[0], chunk):
        sl = slice(start, start + chunk)
        d2 = sq[sl, None] + sq[None, :] - 2.0 * (a[sl] @ a.T)
        dist = np.sqrt(np.maximum(d2, 0.0))
        sums = dist @ onehot  # (chunk, n_classes): summed distance to each cluster
        rows = np.arange(sums.shape[0])
        own_c = own[sl]
        a_i = np.where(sizes[own_c] > 1, sums[rows, own_c] / np.maximum(sizes[own_c] - 1, 1), 0.0)
        means = sums / sizes[None, :]
        means[rows, own_c] = np.inf
        b_i = means.min(axis=1)
        denom = np.maximum(a_i, b_i)
        s = np.where((sizes[own_c] > 1) & (denom > 0), (b_i - a_i) / denom, 0.0)
        scores[sl] = s
    return float(scores.mean())
