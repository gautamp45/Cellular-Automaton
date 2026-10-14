import numpy as np
import pytest

from cellauto.analysis.embed import kmeans, pca, silhouette, standardize


def test_standardize() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    z = standardize(x)
    assert np.allclose(z.mean(axis=0), 0) and np.allclose(z[:, 0].std(), 1)
    assert np.all(z[:, 1] == 0)  # constant column stays finite


def test_pca_recovers_anisotropic_axis() -> None:
    rng = np.random.default_rng(0)
    axis = np.array([3.0, 4.0]) / 5.0
    ortho = np.array([-axis[1], axis[0]])
    x = rng.normal(0, 5.0, (2000, 1)) * axis + rng.normal(0, 0.5, (2000, 1)) * ortho
    scores, comps, ratio = pca(x, 2)
    assert abs(abs(comps[0] @ axis) - 1) < 1e-2
    assert ratio[0] > 0.98 and scores.shape == (2000, 2)
    assert comps[0, np.argmax(np.abs(comps[0]))] > 0  # deterministic sign
    assert np.allclose(comps @ comps.T, np.eye(2), atol=1e-9)


def _blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]])
    truth = np.repeat(np.arange(3), 60)
    return centres[truth] + rng.normal(0, 0.6, (180, 2)), truth


def test_kmeans_finds_three_blobs_and_is_deterministic() -> None:
    x, truth = _blobs()
    labels, centres, inertia = kmeans(x, 3, n_init=10, seed=1)
    # same partition up to renaming
    assert len({(a, b) for a, b in zip(truth, labels, strict=True)}) == 3
    again = kmeans(x, 3, n_init=10, seed=1)
    assert np.array_equal(labels, again[0]) and inertia == again[2]
    assert centres.shape == (3, 2)
    assert silhouette(x, labels) > 0.7


def test_silhouette_orders_clusterings() -> None:
    x, truth = _blobs()
    bad = (np.arange(180) % 3).astype(int)
    assert silhouette(x, truth) > 0.7 > silhouette(x, bad)
    assert silhouette(x, truth, chunk=17) == pytest.approx(silhouette(x, truth))
    with pytest.raises(ValueError):
        silhouette(x, np.zeros(180, dtype=int))


def test_kmeans_validation() -> None:
    with pytest.raises(ValueError):
        kmeans(np.zeros((2, 2)), 3)
