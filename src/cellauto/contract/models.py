"""Pydantic v2 models: the single source of truth for the exported data.

Rules (see ``docs/DATA_CONTRACT.md``):

* every model is ``extra="forbid"`` and frozen; field names are snake_case and the JSON keeps them
  (the only alias is ``lambda_`` <-> ``"lambda"``);
* ``format_version`` is an integer per dataset. A breaking change bumps it and needs a frontend
  change in the same PR; after P07 only *additive* changes are allowed (new datasets, new optional
  fields);
* every field a later phase fills in is declared here as optional with a ``None`` default.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Tensor(_Model):
    """A dense float32 array, little-endian, row-major, base64-encoded."""

    shape: list[int]
    dtype: Literal["float32-le"]
    b64: str


# --- discovery -------------------------------------------------------------------------------


class ParamSpec(_Model):
    name: str
    label: str
    description: str
    kind: Literal["int", "float", "enum", "bool", "rulestring", "bitstring", "float_list", "ref"]
    default: int | float | str | bool | list[float] | None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None  # kind="enum"
    ref: str | None = None  # dataset id for kind="ref", e.g. "lifelike_rules", "nca_models"
    unit: str | None = None


class EngineSpec(_Model):
    id: Literal["elementary", "lifelike", "lenia", "nca"]
    name: str
    summary: str
    state: Literal["binary", "continuous", "multichannel"]
    dimensions: Literal[1, 2]
    params: list[ParamSpec]
    algorithm_anchor: str  # anchor in docs/ALGORITHMS.md
    fixtures: str  # path of the fixture file, relative to the data root
    content_slug: str


class DatasetRef(_Model):
    id: str
    path: str  # relative to the data root
    schema_path: str | None  # JSON Schema for JSON datasets; None for assets (png, gif, md, bin)
    format_version: int
    sha256: str
    bytes: int
    title: str
    description: str
    produced_by: str  # e.g. "cellauto export-web", "cellauto analyze eca-atlas"


class Index(_Model):
    format_version: Literal[1]
    package_version: str
    datasets: list[DatasetRef]


# --- elementary ------------------------------------------------------------------------------


class ElementaryRule(_Model):
    rule: int
    name: str | None
    description: str
    wolfram_class: int | None = None  # only for the reference rules in the catalog
    table: list[int]  # 8 entries, index = 4*left + 2*centre + right
    equivalence_class: list[int] | None = None  # P09: rules equivalent under mirror/complement
    lambda_: float | None = Field(None, alias="lambda")  # P09
    surjective: bool | None = None  # P09
    reversible: bool | None = None  # P09


# --- life-like -------------------------------------------------------------------------------


class FixedPoint(_Model):
    rho: float
    derivative: float
    stability: Literal["stable", "unstable", "marginal"]


class MeanField(_Model):  # P10
    fixed_points: list[FixedPoint]
    simulated_density: float
    correlation_gap: float


class LifeLikeRule(_Model):
    id: str
    name: str
    rulestring: str
    code: int  # 18-bit: bit n = birth on n, bit 9+n = survival on n
    table: list[list[int]]  # (2, 9): table[state][count]
    description: str
    tags: list[str]
    dual: str | None = None  # P10: rulestring of the dual rule
    self_dual: bool | None = None  # P10
    lambda_: float | None = Field(None, alias="lambda")  # P10
    mean_field: MeanField | None = None  # P10


class Pattern(_Model):
    id: str
    name: str
    category: Literal["still_life", "oscillator", "spaceship", "methuselah", "gun"]
    rule: str
    rle: str
    width: int
    height: int
    cells: list[tuple[int, int]]  # live cells as (row, col)
    period: int | None = None
    velocity: str | None = None
    description: str


# --- lenia -----------------------------------------------------------------------------------


class LeniaPreset(_Model):
    id: str
    name: str
    R: int
    T: float
    mu: float
    sigma: float
    peaks: list[float]
    growth: Literal["gaussian", "life_step"]
    kernel_core: Literal["exp_bump", "life_box"]
    cells_rle: str | None
    width: int | None
    height: int | None
    source: str


# --- neural CA -------------------------------------------------------------------------------


class Interval(_Model):
    mean: float
    ci_low: float
    ci_high: float


class NcaMetrics(_Model):  # P12 fills
    mse_200: Interval
    mse_1000: Interval
    regen_steps: Interval | None
    n_seeds: int


class NcaModel(_Model):
    id: str
    target: str
    loss: str
    filter: str
    trained_with_damage: bool
    isotropic: bool
    n_channels: int
    hidden_channels: int
    fire_rate: float
    grid_size: int
    padding: int
    weights_path: str
    target_path: str
    animation_path: str | None
    metrics: NcaMetrics | None = None  # P12


class NcaWeights(_Model):
    format_version: Literal[1]
    id: str
    n_channels: int
    hidden_channels: int
    fire_rate: float
    alive_threshold: float
    alpha_channel: int
    grid_size: int
    padding: int
    filter: str
    kernels: list[list[list[float]]]  # (3, 3, 3), already scaled: identity, dx, dy
    perception_layout: str
    w1: Tensor  # (hidden, 3 * n_channels)
    b1: Tensor  # (hidden,)
    w2: Tensor  # (n_channels, hidden)


# --- binary tables, content, fixtures --------------------------------------------------------


class ColumnSpec(_Model):
    name: str
    dtype: Literal["uint8", "uint16", "float32"]
    offset: int  # byte offset into the .bin file (8-byte aligned)
    length: int  # byte length of the column
    scale: float | None = None  # uint16 quantised: value = zero + scale * q
    zero: float | None = None
    unit: str | None = None
    description: str


class BinaryTableHeader(_Model):
    format: Literal["cabin"]
    version: Literal[1]
    rows: int
    row_key: str  # how rows map to entities, e.g. "row = code >> 1"
    bin_path: str  # relative to the header file
    bin_sha256: str
    columns: list[ColumnSpec]


class ContentPage(_Model):
    slug: str
    title: str
    page: str
    order: int
    path: str


class Fixture(_Model):
    """Generic golden-fixture container; ``cases`` items are engine-specific dicts."""

    engine: str
    description: str
    tolerance: float | None
    cases: list[dict[str, Any]]


# --- ECA behaviour atlas (P09) -----------------------------------------------------------------


class EcaProtocol(_Model):
    width: int
    steps: int
    window_start: int
    trials: int
    seed: int
    max_period: int


class EcaFeature(_Model):
    name: str
    description: str
    unit: str | None
    symmetry_invariant: bool  # unchanged by reflection and complement


class EcaEmbedding(_Model):
    features: list[str]  # standardised features used for PCA and clustering
    components: list[list[float]]  # (2, len(features)); sign-fixed
    explained_variance: list[float]


class EcaAtlasRow(_Model):
    rule: int
    damage: float
    compression: float
    density: float
    activity: float
    block_entropy: float
    period: int  # 0 = aperiodic within the protocol
    pca_x: float
    pca_y: float
    cluster: int  # clustering with `chosen_k`
    cluster_k4: int  # clustering with k = 4


class EcaClusterSummary(_Model):
    cluster: int
    size: int
    centroid: dict[str, float]  # mean of each feature over the cluster's rules
    example_rules: list[int]


class EcaAssignment(_Model):
    rule: int
    wolfram_class: int
    cluster: int


class EcaContingency(_Model):
    classes: list[int]  # Wolfram classes present among the labelled reference rules
    clusters: list[int]
    counts: list[list[int]]  # counts[i][j]: rules of class classes[i] in cluster clusters[j]
    assignments: list[EcaAssignment]


class EcaKScore(_Model):
    k: int
    silhouette: float


class EcaClustering(_Model):
    k: int
    silhouette: float
    summary: list[EcaClusterSummary]
    contingency: EcaContingency


class EcaAtlas(_Model):
    format_version: Literal[1]
    protocol: EcaProtocol
    features: list[EcaFeature]
    embedding: EcaEmbedding
    k_selection: list[EcaKScore]
    chosen_k: int
    clusterings: list[EcaClustering]  # chosen k, plus k = 4 when different
    rows: list[EcaAtlasRow]  # 256 rows, ordered by rule


# --- life-like rule-space atlas (P10) ----------------------------------------------------------


class LifeAtlasProtocol(_Model):
    grid: int
    steps: int
    window_start: int
    density: float
    seed: int
    n_rows: int  # 131072 for the complete atlas (all B0-free rules)
    row_key: str


class LifeAtlasCluster(_Model):
    cluster: int
    size: int
    centroid: dict[str, float]  # mean of each simulated feature over the cluster


class LifeAtlasKScore(_Model):
    k: int
    silhouette: float  # on a 10,000-row subsample


class LifeAtlasBin(_Model):
    """One quantile bin of lambda in the edge-of-chaos analysis."""

    lambda_lo: float
    lambda_hi: float
    n: int
    activity_median: float
    activity_q1: float
    activity_q3: float
    damage_median: float
    damage_q1: float
    damage_q3: float
    compression_median: float
    compression_q1: float
    compression_q3: float


class LifeAtlasCorrelation(_Model):
    feature: str
    spearman_with_lambda: float


class LifeAtlasCatalogRow(_Model):
    id: str
    code: int
    row: int | None  # None for B0 rules, which are not in the atlas


class LifelikeAtlasMeta(_Model):
    format_version: Literal[1]
    protocol: LifeAtlasProtocol
    header_path: str  # the BinaryTableHeader describing atlas.bin
    embedding_features: list[str]
    embedding_components: list[list[float]]
    explained_variance: list[float]
    k_selection: list[LifeAtlasKScore]
    chosen_k: int
    clusters: list[LifeAtlasCluster]
    edge_of_chaos: list[LifeAtlasBin]
    correlations: list[LifeAtlasCorrelation]
    catalog: list[LifeAtlasCatalogRow]


# --- rule inference (P11) ----------------------------------------------------------------------


class ReluNetModel(_Model):
    """``out_bias + sum_i out_weights[i] * ReLU(v + biases[i])``, ``v = neighbours + centre/2``."""

    centre_weight: float
    biases: list[float]
    out_weights: list[float]
    out_bias: float


class ReluExample(_Model):
    id: str
    rulestring: str
    code: int
    hidden_units: int
    net: ReluNetModel


class ReluExamples(_Model):
    format_version: Literal[1]
    construction: str
    examples: list[ReluExample]
    histogram: list[int]  # histogram[u] = number of the 2^18 rules needing u hidden units
    mean_hidden_units: float
