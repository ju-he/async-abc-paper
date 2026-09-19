#!/usr/bin/env python3
"""Cellular Potts protocol and parameter screening.

Why this exists
---------------
The stored asynchronous CPM campaign (``rerun_20260707/cellular_potts``) has a
signal-to-noise ratio of about 0.30: the Monte-Carlo scatter of the discrepancy
at *fixed* theta is roughly three times the entire systematic variation of the
discrepancy across the prior box.  No parameter choice survives that, so the
benchmark cannot carry an inference claim as configured.  This script measures
the two quantities that decide the matter, for any protocol and any candidate
parameter:

    identifiability = (between-theta spread) / (within-theta spread)

A parameter scoring below 1 will not be identified by a campaign, however long
it runs.

Two modes
---------
``--mode simulate`` (MPI, one node is plenty)
    Runs a design of CPM simulations under one *protocol* -- the two knobs that
    genuinely change the simulation, domain size and snapshot cadence -- and
    stores, for every written snapshot and every radial bin count, the RAW
    feature arrays the ABC distance is built from.  Nothing is reduced to a
    scalar here.  The remaining protocol knobs (which snapshots to average,
    how many radial bins, how many replicate seeds per evaluation) do not
    change the simulator, so they are explored post hoc from one corpus
    instead of costing one campaign each.

``--mode analyze`` (serial, runs anywhere)
    Reads that corpus, refits the feature-space model at each bin count, and
    reports the within-theta noise floor, the between-theta spread, and the
    resulting identifiability -- per feature block, for the total discrepancy,
    and per candidate parameter, together with each parameter's saturation
    window in physical units.

Typical use::

    srun --ntasks=48 python experiments/scripts/diag_cpm_screening.py \
        --mode simulate --out /p/scratch/.../cpm_screen_50 --blocksize 50

    python experiments/scripts/diag_cpm_screening.py \
        --mode analyze --out experiments/data/cpm_screening/blocksize50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

REPO_ROOT = EXPERIMENTS_DIR.parent
ASSETS = REPO_ROOT / "experiments" / "assets" / "cellular_potts"

# Default lives in the repo, not in a session scratch directory: a committed
# diagnostic must be runnable as-is by whoever reads the paper.
DEFAULT_OUT = REPO_ROOT / "experiments" / "data" / "cpm_screening"

# --------------------------------------------------------------------------
# Candidate parameters.
#
# ``path`` is a nastjapy config path (see simulation_config.set_by_path).
# ``lo``/``hi`` are PHYSICAL screening-prior bounds; ``scale`` says whether the
# prior is uniform in the value or in its logarithm.  ``centre`` is the value
# held fixed when another parameter is swept, and equals the shipped template.
#
# division_rate is reparameterised relative to the shipped prior: the shipped
# prior [6e-5, 0.6] puts the entire measured transition in its bottom 2%.  With
# T = 500 timesteps the expected number of division attempts per cell is
# lambda = p * T, and the measured transition is lambda ~ 6 -> 60, i.e.
# p ~ 0.012 -> 0.12.  A log-uniform prior on [0.002, 0.2] puts that mid-prior.
#
# Temperature is deliberately absent: only Delta-H / T enters the Boltzmann
# acceptance, so adhesion and temperature are exactly degenerate for static
# structure.  T is held at the template value of 50 and adhesion is inferred.
# --------------------------------------------------------------------------
CANDIDATES: Dict[str, Dict[str, Any]] = {
    "division_rate": dict(
        path="define_functions.division_cond_cancer[1]",
        lo=0.002, hi=0.2, scale="log", centre=0.03, integer=False,
        drives="population size (log_n), cluster radius (log_r95)",
    ),
    "motility": dict(
        path="CellsInSilico.orientation.motilityamount[9]",
        lo=100, hi=4000, scale="log", centre=2000, integer=True,
        drives="dispersal; gas-like fraction, density profile",
    ),
    # Surface tension of the cluster against the medium is
    # gamma = J(cancer, medium) - J(cancer, cancer) / 2.  At the template values
    # that is 151 - 103/2 = 99.5, i.e. strongly cohesive, and it stays positive
    # for every J_cc below 302.  A prior that does not cross gamma = 0 cannot
    # produce a structural change, which is why the [40, 200] screen moved the
    # summary by only 4 sigma.  Both sides of the tension are therefore
    # screened, over ranges that do cross it.
    "adhesion_cc": dict(
        path="CellsInSilico.adhesion.matrix[9][9]",
        lo=40, hi=400, scale="lin", centre=103, integer=True,
        drives="cohesion; g(r), radial density steepness, gas-like fraction",
    ),
    "adhesion_cl": dict(
        # Symmetric: the engine reads both halves of the matrix.
        path=["CellsInSilico.adhesion.matrix[9][1]",
              "CellsInSilico.adhesion.matrix[1][9]"],
        lo=20, hi=200, scale="lin", centre=151, integer=True,
        drives="surface tension against the medium; wetting vs compaction",
    ),
    "surface_lambda": dict(
        path="CellsInSilico.surface.lambda_[9]",
        lo=0.25, hi=8.0, scale="log", centre=1.0, integer=False,
        drives="cell shape; radial FA and S2, which nothing currently drives",
    ),
    "persistence": dict(
        path="CellsInSilico.orientation.persistenceMagnitude",
        lo=0.0, hi=0.99, scale="lin", centre=0.834, integer=False,
        drives="directional correlation; S2, FA",
    ),
    # Cell target volume. Sets how much space a cell occupies, so it moves the
    # cluster radius AT FIXED cell count -- the one direction in (log_n, log_r95)
    # that division and motility both leave alone. A standard CPM parameter and
    # biologically meaningful (cell size), not a tuning knob.
    "cell_volume": dict(
        path="CellsInSilico.volume.default.value",
        lo=200, hi=1200, scale="log", centre=500, integer=True,
        drives="cluster radius at fixed cell count; packing",
    ),
    # Fluctuation amplitude. Excluded from the earlier screens because only
    # Delta-H / T enters the Boltzmann acceptance, making it degenerate with
    # adhesion -- but that degeneracy only binds when adhesion is also inferred.
    # With adhesion fixed, T is a free knob acting on membrane roughness.
    "temperature": dict(
        path="CellsInSilico.temperature",
        lo=15, hi=150, scale="log", centre=50, integer=True,
        drives="membrane fluctuation; cell shape, surface roughness",
    ),
    "recalc_time": dict(
        path="CellsInSilico.orientation.recalculationtime",
        lo=2, hi=60, scale="log", centre=15, integer=True,
        drives="direction persistence time; S2, FA",
    ),
}

SEED_PATH = "Settings.randomseed"

# Parameters applied to every simulation at a constant value but not screened.
# A benchmark configuration is not defined by its inferred parameters alone: the
# shipped template sets motilityamount[9] = 50, so "do not infer motility" and
# "hold motility at 1400" are different experiments, and only the second is
# reproducible. Populated by --fix and recorded in protocol.json.
FIXED: Dict[str, float] = {}


def apply_fixed(specs: Sequence[str]) -> None:
    """Hold a candidate at a constant value instead of screening it."""
    for spec in specs or ():
        name, _, raw = spec.partition("=")
        if name not in CANDIDATES:
            raise KeyError(f"unknown parameter '{name}'; known: {sorted(CANDIDATES)}")
        entry = CANDIDATES.pop(name)
        FIXED[name] = int(round(float(raw))) if entry["integer"] else float(raw)


_PATHS: Dict[str, Any] = {name: spec["path"] for name, spec in CANDIDATES.items()}


def apply_centre_overrides(specs: Sequence[str]) -> None:
    """Move the reference point -- the theta that plays the role of observed data.

    Where the truth sits inside the prior is a design choice, not a detail: a truth at
    the edge of a parameter's responsive window produces a posterior that runs to the
    prior edge no matter how identifiable the parameter is, which is how the shipped
    configuration fails.  Format is ``name=value`` in physical units.
    """
    for spec in specs or ():
        name, _, raw = spec.partition("=")
        if name not in CANDIDATES:
            raise KeyError(f"unknown parameter '{name}'; known: {sorted(CANDIDATES)}")
        entry = CANDIDATES[name]
        value = int(round(float(raw))) if entry["integer"] else float(raw)
        lo, hi = entry["lo"], entry["hi"]
        if not (min(lo, hi) <= value <= max(lo, hi)):
            raise ValueError(
                f"--centre {name}={value} is outside its prior [{lo}, {hi}]; every "
                "one-at-a-time sweep of another parameter would hold it at a value "
                "the prior excludes"
            )
        entry["centre"] = value


def apply_prior_overrides(specs: Sequence[str], only: Sequence[str] | None) -> None:
    """Re-range or subset the candidates from the command line.

    A refinement round should not need a code edit: the first screen tells you
    which prior was in the wrong place, and this is how you move it.  Format is
    ``name=lo:hi[:scale]``, e.g. ``persistence=0.002:0.3:log``.  The resulting
    definition is written into ``protocol.json`` so the analysis re-applies
    exactly the priors the corpus was generated under.
    """
    for spec in specs or ():
        name, _, rest = spec.partition("=")
        if name not in CANDIDATES:
            raise KeyError(f"unknown parameter '{name}'; known: {sorted(CANDIDATES)}")
        parts = rest.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"--prior expects name=lo:hi[:scale], got '{spec}'")
        entry = CANDIDATES[name]
        lo, hi = float(parts[0]), float(parts[1])
        if entry["integer"]:
            lo, hi = int(round(lo)), int(round(hi))
        entry["lo"], entry["hi"] = lo, hi
        if len(parts) == 3:
            if parts[2] not in ("lin", "log"):
                raise ValueError(f"scale must be 'lin' or 'log', got '{parts[2]}'")
            entry["scale"] = parts[2]
        centre = entry["centre"]
        if not (min(lo, hi) <= centre <= max(lo, hi)):
            # The centre is the value every OTHER sweep holds fixed; leaving it
            # outside the prior would sweep around a point the prior excludes.
            entry["centre"] = _to_physical(name, 0.5)
    if only:
        unknown = set(only) - set(CANDIDATES)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}; known: {sorted(CANDIDATES)}")
        for name in list(CANDIDATES):
            if name not in only:
                del CANDIDATES[name]


def candidate_snapshot() -> Dict[str, Dict[str, Any]]:
    """The active candidate definition, for the run record."""
    return {name: {k: v for k, v in spec.items() if k != "drives"}
            for name, spec in CANDIDATES.items()}

# Blocks computed here rather than by nastjapy, from columns every CellInfo CSV
# already carries and no shipped block reads.  All are INTENSIVE by construction --
# proportions, coefficients of variation, or dimensionless ratios -- which is the
# property the whole shipped feature set lacks: measured over eleven configurations,
# every one of its blocks moves with population size.
#
#   cell_shape_index         Surface / Volume^(2/3) per cell, mean and CV. The only
#                            statistic here that sees a CELL rather than an arrangement,
#                            and the direct observable of surface.lambda and temperature.
#   cell_volume_dispersion   CV of cell volume: how tightly volume is held.
#   motility_order           polar and nematic order of the per-cell motility directions.
#                            persistence and recalculationtime control exactly this and
#                            nothing in the shipped set looks at direction at all.
#   aggregation_omega        sum n_i^2 / sum n_i over contact components, over N. The
#                            sibling campaign's aggregation index: 1/N is a haze of
#                            singletons, 1 is one blob.
#   outer_fraction           cells outside the largest contact component.
#   radial_variance_fraction mean over outer components of the fraction of a component's
#                            own spatial variance lying along its outward direction. The
#                            campaign's estimator, whose null median is free of component
#                            size (0.296 at n=3, 0.332 at n=62) -- size-invariant by
#                            construction rather than by normalisation.
CUSTOM_BLOCKS = ("cell_shape_index", "cell_volume_dispersion", "motility_order",
                 "aggregation_omega", "outer_fraction", "radial_variance_fraction")
# Contact radius as a multiple of the cloud's own median nearest-neighbour distance,
# so the graph is scale-free in the same way the features are.
CONTACT_SCALE = 1.5


def custom_features(frame: Any) -> Dict[str, List[float]]:
    """Intensive per-cell and structural features for one snapshot."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    position = frame[["CenterX", "CenterY", "CenterZ"]].to_numpy(dtype=float)
    n = len(position)
    out: Dict[str, List[float]] = {}
    if n < 4:
        return out  # below this nothing here means anything; the row is simply absent

    volume = frame["Volume"].to_numpy(dtype=float)
    surface = frame["Surface"].to_numpy(dtype=float)
    good = (volume > 0) & np.isfinite(surface)
    if good.sum() >= 3:
        shape = surface[good] / np.cbrt(volume[good]) ** 2
        mean_shape = float(np.mean(shape))
        out["cell_shape_index"] = [mean_shape,
                                   float(np.std(shape) / mean_shape) if mean_shape > 0 else 0.0]
        mean_volume = float(np.mean(volume[good]))
        out["cell_volume_dispersion"] = [float(np.std(volume[good]) / mean_volume)
                                         if mean_volume > 0 else 0.0]

    direction = frame[["MotilityDirX", "MotilityDirY", "MotilityDirZ"]].to_numpy(dtype=float)
    norm = np.linalg.norm(direction, axis=1)
    moving = norm > 1e-9
    if moving.sum() >= 3:
        unit = direction[moving] / norm[moving, None]
        polar = float(np.linalg.norm(unit.mean(axis=0)))
        # Nematic order: largest eigenvalue of the Q tensor, 0 isotropic, 1 aligned.
        q = (3.0 * (unit[:, :, None] * unit[:, None, :]).mean(axis=0) - np.eye(3)) / 2.0
        out["motility_order"] = [polar, float(np.max(np.linalg.eigvalsh(q)))]

    tree = cKDTree(position)
    nearest = tree.query(position, k=2)[0][:, 1]
    radius = CONTACT_SCALE * float(np.median(nearest))
    if not np.isfinite(radius) or radius <= 0:
        return out
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs):
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    _count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    out["aggregation_omega"] = [float(np.sum(sizes ** 2) / np.sum(sizes) / n)]
    core = int(np.argmax(sizes))
    out["outer_fraction"] = [float((n - sizes[core]) / n)]

    centre = position[labels == core].mean(axis=0) if sizes[core] else position.mean(axis=0)
    fractions = []
    for label in range(len(sizes)):
        if label == core or sizes[label] < 3:
            continue
        points = position[labels == label]
        outward = points.mean(axis=0) - centre
        length = np.linalg.norm(outward)
        if length <= 0:
            continue
        centred = points - points.mean(axis=0)
        total = float(np.sum(centred ** 2))
        if total <= 0:
            continue
        along = float(np.sum((centred @ (outward / length)) ** 2))
        fractions.append(along / total)
    # 1/3 is the isotropic expectation, so an empty periphery reports "isotropic"
    # rather than dropping the block and desynchronising the corpus.
    out["radial_variance_fraction"] = [float(np.mean(fractions)) if fractions else 1.0 / 3.0]
    return out


# Radial bin counts to extract for every snapshot.  32 is the shipped value.
BIN_COUNTS = (8, 16, 32)
# Features whose metadata carries an ``n_bins`` entry.
BINNED_FEATURES = (
    "radial_fa_equal_volume",
    "radial_s2_equal_volume",
    "radial_density_profile_equal_volume",
    "pair_correlation_gofr",
)

# The shape extractor writes these alongside radial_fa_equal_volume, so the
# corpus carries them at zero extra cost even though the shipped feature set
# ignores them.  ``--extra-blocks`` screens them as candidate replacements for
# dbscan_gaslike_fraction, which is dead in the shipped model (its scaler is the
# RobustScaler IQR-zero fallback and its block norm is 0).
# Features the sibling nastjapy inference campaign added for exactly this failure
# mode.  Its finding F10 is that the size-normalised radial profiles are
# motility-blind -- pcorr(feature, motility | division_rate) ~ 0 -- and F71 that
# `surface_roughness` and `invasion_ratio` carry most of the signal its own
# seven-feature gate was missing.  All three are registered in nastjapy's
# FEATURE_FUNCTIONS and need nothing but the cell positions, so adding them to
# the benchmark is a `distance_metric_params.json` edit.
CAMPAIGN_BLOCKS = {
    "invasion_ratio": {},
    "surface_roughness": {},
    "shape_anisotropy": {},
}

# Growth-curve features need the WHOLE trajectory, not one snapshot, so they are
# extracted from a multi-frame handler.  The campaign's nano test put
# growth_model_r at pcorr 0.64 for division given motility, where the
# single-snapshot features sat at ~0.
TRAJECTORY_BLOCKS = {
    "growth_model_r": {},
    "growth_model_K": {},
    "log_n_trajectory": {},
    # Mean squared displacement. The one observable that measures motion directly
    # rather than inferring it from a static arrangement, and it is free: it needs
    # only cell positions matched by CellID across the snapshots already written.
    # Population size does not enter it at all, which is exactly what every block
    # in the shipped summary fails to avoid.
    "msd": {},
    "non_gaussian_parameter": {},
}

# Two of the optional blocks misbehave badly enough to distort any analysis that
# includes them, so neither is on by default. Measured on the 50^3 corpus:
# `shape_anisotropy` is PC1/PC3, and a near-degenerate smallest axis sends it to
# a block norm of 6e8, after which its scaled distances swamp every other block;
# `log_n_trajectory` starts at log(4) on the four seeded spheroids, so its early
# entries are coarsely discretised and heavy-tailed, and it captures 84-96% of
# the whitened response direction of parameters whose identifiability is 0.
# Ask for them by name if you want them.
DEFAULT_EXTRA_BLOCKS = ("radial_linearity_equal_volume", "radial_planarity_equal_volume",
                        "radial_sphericity_equal_volume", "invasion_ratio",
                        "surface_roughness", "growth_model_r", "growth_model_K",
                        "msd", "non_gaussian_parameter") + CUSTOM_BLOCKS

EXTRA_BLOCKS = {
    "radial_linearity_equal_volume": 0.88,
    "radial_planarity_equal_volume": 0.88,
    "radial_sphericity_equal_volume": 0.88,
}


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def _to_physical(name: str, u: float) -> float:
    """Map a unit-interval coordinate to the candidate's physical value."""
    spec = CANDIDATES[name]
    lo, hi = float(spec["lo"]), float(spec["hi"])
    if spec["scale"] == "log":
        if lo <= 0:
            raise ValueError(f"log-scaled parameter '{name}' needs lo > 0, got {lo}")
        value = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        value = lo + u * (hi - lo)
    if spec["integer"]:
        # NAStJA reads these fields as integers in the shipped template; emitting
        # a float there is a config-schema change, not a parameter change.
        return int(round(value))
    return float(value)


def _to_unit(name: str, value: float) -> float:
    """Inverse of :func:`_to_physical` (used to place the centre on the prior)."""
    spec = CANDIDATES[name]
    lo, hi = float(spec["lo"]), float(spec["hi"])
    if spec["scale"] == "log":
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Plain LHS on the unit cube; no scipy.stats.qmc dependency."""
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        cuts = (np.arange(n) + rng.random(n)) / n
        out[:, j] = cuts[rng.permutation(n)]
    return out


def make_design(
    *,
    n_anchors: int,
    anchor_seeds: int,
    oat_levels: int,
    oat_seeds: int,
    n_lhs: int,
    lhs_seeds: int,
    ref_seeds: int,
    design_seed: int,
) -> List[Dict[str, Any]]:
    """Build the full (theta, seed) work list.

    Four strata, each answering a different question:

    ``reference``  one theta at the shipped template values, many seeds.  Its
                   replicate-averaged feature vector is the pseudo-observed
                   data for the discrepancy statistics, and it is excluded from
                   every spread statistic.
    ``anchor``     a handful of thetas spread over the box, many seeds each.
                   These measure the WITHIN-theta noise floor.
    ``oat``        one parameter swept at a time, the rest held at centre.
                   These give each parameter's response curve and its
                   saturation window.
    ``lhs``        a space-filling design with everything varying.  This
                   measures the BETWEEN-theta spread the sampler actually sees.
    """
    names = list(CANDIDATES)
    rng = np.random.default_rng(design_seed)
    points: List[Dict[str, Any]] = []

    centre = {n: CANDIDATES[n]["centre"] for n in names}
    points.append(dict(stratum="reference", label="reference", swept=None,
                       params=dict(centre), n_seeds=ref_seeds))

    anchors = _latin_hypercube(n_anchors, len(names), rng)
    for i, row in enumerate(anchors):
        params = {n: _to_physical(n, float(u)) for n, u in zip(names, row)}
        points.append(dict(stratum="anchor", label=f"anchor{i:02d}", swept=None,
                           params=params, n_seeds=anchor_seeds))

    for name in names:
        for level in range(oat_levels):
            u = (level + 0.5) / oat_levels
            params = dict(centre)
            params[name] = _to_physical(name, u)
            points.append(dict(stratum="oat", label=f"oat_{name}_{level:02d}",
                               swept=name, params=params, n_seeds=oat_seeds))

    lhs = _latin_hypercube(n_lhs, len(names), rng)
    for i, row in enumerate(lhs):
        params = {n: _to_physical(n, float(u)) for n, u in zip(names, row)}
        points.append(dict(stratum="lhs", label=f"lhs{i:04d}", swept=None,
                           params=params, n_seeds=lhs_seeds))

    work: List[Dict[str, Any]] = []
    for idx, point in enumerate(points):
        for rep in range(int(point["n_seeds"])):
            work.append(dict(
                point_index=idx,
                stratum=point["stratum"],
                label=point["label"],
                swept=point["swept"],
                params=point["params"],
                rep=rep,
                # Seeds are distinct across points so that no two thetas share a
                # Monte-Carlo realisation, which would correlate their noise.
                seed=1_000_000 + idx * 1000 + rep,
            ))
    return work


# --------------------------------------------------------------------------
# Protocol assets
# --------------------------------------------------------------------------
def build_variant_assets(out_dir: Path, *, blocksize: int, timesteps: int,
                         write_every: int) -> Dict[str, Path]:
    """Write the sim-config and config-builder assets for one protocol.

    Derived from the committed assets so the variant is reproducible from the
    repository alone.  The spheroid filling is re-centred with the domain;
    everything else is inherited.
    """
    with open(ASSETS / "sim_config.json") as f:
        sim_config = json.load(f)
    with open(ASSETS / "config_builder_params.json") as f:
        cb_params = json.load(f)

    sim_config["Geometry"]["blocksize"] = [blocksize, blocksize, blocksize]
    sim_config["Settings"]["timesteps"] = timesteps
    sim_config["Writers"]["CellInfo"]["steps"] = write_every
    cb_params["filling_generator_params"]["center"] = blocksize // 2

    variant_dir = out_dir / "assets"
    variant_dir.mkdir(parents=True, exist_ok=True)
    sim_config_path = variant_dir / "sim_config.json"
    cb_params_path = variant_dir / "config_builder_params.json"
    with open(sim_config_path, "w") as f:
        json.dump(sim_config, f, indent=2)
    cb_params["config_template"] = str(sim_config_path)
    with open(cb_params_path, "w") as f:
        json.dump(cb_params, f, indent=2)
    return dict(sim_config=sim_config_path, config_builder=cb_params_path)


def bin_metadata(n_bins: int, *, extras: Any = False) -> Dict[str, Dict[str, Any]]:
    """Shipped feature metadata with every binned feature set to ``n_bins``.

    With ``extras``, the free shape blocks are appended.  Only the analysis fit
    uses that form: extraction always runs the shipped seven, and the extras
    arrive as side-products of the same extractor.
    """
    with open(ASSETS / "distance_metric_params.json") as f:
        metadata = json.load(f)["feature_metadata"]
    for name in BINNED_FEATURES:
        if name not in metadata:
            raise KeyError(
                f"binned feature '{name}' missing from distance_metric_params.json"
            )
        metadata[name] = dict(metadata[name], n_bins=int(n_bins))
    if extras:
        template = metadata["radial_fa_equal_volume"]
        wanted = set(extras) if extras is not True else set(DEFAULT_EXTRA_BLOCKS)
        unknown = (wanted - set(EXTRA_BLOCKS) - set(CAMPAIGN_BLOCKS)
                   - set(TRAJECTORY_BLOCKS) - set(CUSTOM_BLOCKS))
        if unknown:
            raise KeyError(f"unknown extra blocks {sorted(unknown)}")
        for name in wanted:
            if name in EXTRA_BLOCKS:
                metadata[name] = dict(template, explained_var=EXTRA_BLOCKS[name])
            elif name in CAMPAIGN_BLOCKS:
                metadata[name] = dict(CAMPAIGN_BLOCKS[name])
            elif name in CUSTOM_BLOCKS:
                metadata[name] = {}
            else:
                metadata[name] = dict(TRAJECTORY_BLOCKS[name])
    return metadata


def extraction_metadata(n_bins: int) -> Dict[str, Dict[str, Any]]:
    """What the simulate step computes: the shipped seven plus the campaign's three."""
    metadata = bin_metadata(n_bins)
    metadata.update({k: dict(v) for k, v in CAMPAIGN_BLOCKS.items()})
    return metadata


# --------------------------------------------------------------------------
# Simulate mode
# --------------------------------------------------------------------------
def _frame_timesteps(n_frames: int, write_every: int, timesteps: int) -> List[int]:
    """Simulation time of each written frame that is worth keeping.

    NAStJA's CellInfo writer emits at multiples of ``steps`` but not at t = 0,
    so frame ``i`` is at ``(i + 1) * steps``.  It then appends one terminal dump
    at the very end of the run, which is a near-duplicate of the last cadence
    write (verified on a 501-step run: the last two frames differ by a single
    timestep, cell ages 295 vs 296).  Counting that as an independent snapshot
    would make snapshot averaging look better than it is, so it is dropped.

    Both facts are asserted rather than assumed: a writer change must fail here,
    not silently relabel every snapshot in the corpus.
    """
    times = [(i + 1) * write_every for i in range(n_frames)]
    kept = [t for t in times if t <= timesteps - 1]
    if not kept:
        raise ValueError(
            f"no usable frames: {n_frames} frames at cadence {write_every} "
            f"over {timesteps} timesteps"
        )
    if len(times) - len(kept) > 1:
        raise ValueError(
            f"expected at most one terminal dump beyond t={timesteps - 1}, got "
            f"{len(times) - len(kept)} ({n_frames} frames, cadence {write_every})"
        )
    return kept


def extract_corpus_row(sim_dir: Path, *, bins: Sequence[int], extract_from: int,
                       write_every: int, timesteps: int) -> List[Dict[str, Any]]:
    """Extract raw feature arrays for every (frame, bin count) of one sim."""
    from data.DataHandler import build_datahandler_for_dir
    from inference.feature_space import _feature_value_to_array

    rows: List[Dict[str, Any]] = []
    probe = build_datahandler_for_dir(
        sim_dir, feature_metadata=extraction_metadata(bins[0]),
        base_kwargs={"scale_factor": 1}, default_timestep_range=-1,
    )
    n_frames = int(probe.sim_dir.frames)
    times = _frame_timesteps(n_frames, write_every, timesteps)
    del probe

    for frame, tstep in enumerate(times):
        if tstep < extract_from:
            continue
        for n_bins in bins:
            handler = build_datahandler_for_dir(
                sim_dir, feature_metadata=extraction_metadata(n_bins),
                base_kwargs={"scale_factor": 1, "timestep_range": frame},
                default_timestep_range=None,
            )
            handler.extract_all_features(show_progress=False, quiet=True)
            features: Dict[str, List[float]] = {}
            for name, payload in handler.features.items():
                values = list(payload.values()) if isinstance(payload, dict) else [payload]
                arrays = [a for a in (_feature_value_to_array(v) for v in values) if a is not None]
                if not arrays:
                    continue
                array = np.asarray(arrays[-1], dtype=float).ravel()
                if array.size and np.isfinite(array).all():
                    features[name] = [float(x) for x in array]
            # NOT named `frame`: that is the loop's integer frame index, and shadowing
            # it fed a DataFrame back in as `timestep_range` on the next bin count.
            snapshot = handler.data[next(iter(handler.data))] if handler.data else None
            n_cells = int(len(snapshot)) if snapshot is not None else 0
            # Recomputed per bin count although they do not depend on it: a KD-tree
            # over ~50 points is far cheaper than desynchronising the corpus by
            # attaching them to only one of the three bin variants.
            for name, values in (custom_features(snapshot) if snapshot is not None else {}).items():
                if np.isfinite(values).all():
                    features[name] = [float(v) for v in values]
            rows.append(dict(frame=frame, tstep=int(tstep), bins=int(n_bins),
                             n_cells=n_cells, features=features))
            del handler

    # Growth curve over the whole trajectory, attached to every row of this
    # simulation so the analysis can treat it as one more block.
    trajectory: Dict[str, List[float]] = {}
    try:
        whole = build_datahandler_for_dir(
            sim_dir, feature_metadata={k: dict(v) for k, v in TRAJECTORY_BLOCKS.items()},
            base_kwargs={"scale_factor": 1, "timestep_range": (0, len(times), 1)},
            default_timestep_range=None,
        )
        whole.extract_all_features(show_progress=False, quiet=True)
        for name in TRAJECTORY_BLOCKS:
            payload = whole.features.get(name)
            values = list(payload.values()) if isinstance(payload, dict) else [payload]
            arrays = [a for a in (_feature_value_to_array(v) for v in values) if a is not None]
            if not arrays:
                continue
            array = np.asarray(arrays[-1], dtype=float).ravel()
            if array.size and np.isfinite(array).all():
                trajectory[name] = [float(x) for x in array]
        del whole
    except Exception as exc:  # noqa: BLE001 - a missing growth fit must not lose the snapshot rows
        print(f"[screen] growth-curve extraction failed for {sim_dir}: "
              f"{type(exc).__name__}: {exc}", flush=True)
    for row in rows:
        row["trajectory"] = trajectory
    return rows


def run_simulate(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    from async_abc.benchmarks.cellular_potts import (
        _ensure_nastjapy_on_path,
        _rewrite_generated_config_paths,
    )

    _ensure_nastjapy_on_path()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    out_dir = Path(args.out).resolve()
    corpus_dir = out_dir / "corpus"
    if rank == 0:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        paths = build_variant_assets(
            out_dir, blocksize=args.blocksize, timesteps=args.timesteps,
            write_every=args.write_every,
        )
        with open(out_dir / "protocol.json", "w") as f:
            json.dump(dict(blocksize=args.blocksize, timesteps=args.timesteps,
                           write_every=args.write_every,
                           extract_from=args.extract_from,
                           bins=list(BIN_COUNTS), n_ranks=size,
                           candidates=candidate_snapshot(),
                           fixed=dict(FIXED)), f, indent=2)
    comm.Barrier()
    paths = dict(
        sim_config=out_dir / "assets" / "sim_config.json",
        config_builder=out_dir / "assets" / "config_builder_params.json",
    )

    work = make_design(
        n_anchors=args.n_anchors, anchor_seeds=args.anchor_seeds,
        oat_levels=args.oat_levels, oat_seeds=args.oat_seeds,
        n_lhs=args.n_lhs, lhs_seeds=args.lhs_seeds,
        ref_seeds=args.ref_seeds, design_seed=args.design_seed,
    )
    if args.limit:
        work = work[: args.limit]
    mine = work[rank::size]
    if rank == 0:
        print(f"[screen] {len(work)} evaluations over {size} ranks "
              f"({len(mine)} on rank 0)", flush=True)

    from nastja.parameter_space_config import ParameterSpace
    from simulation.manager import SimulationManager
    from simulation.simulation_config import Parameter, ParameterList
    from simulation.simulation_config_builder import SimulationConfigBuilderParams

    space = ParameterSpace.model_validate(dict(parameters={
        name: dict(path=spec["path"], range=[spec["lo"], spec["hi"]])
        for name, spec in CANDIDATES.items()
    }))
    if FIXED:
        print(f"[screen] holding fixed: {FIXED}", flush=True)
    with open(paths["config_builder"]) as f:
        cb_raw = json.load(f)
    sim_root = Path(args.sim_dir or (out_dir / "sims")) / f"rank{rank:04d}"
    cb_raw["out_dir"] = str(sim_root)
    manager = SimulationManager(
        SimulationConfigBuilderParams.model_validate(cb_raw), space, None
    )

    written = 0
    failures = 0
    out_path = corpus_dir / f"rank_{rank:04d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as sink:
        for item in mine:
            entries = [Parameter(name=n, value=v, path=CANDIDATES[n]["path"])
                       for n, v in item["params"].items()]
            entries += [Parameter(name=n, value=v, path=_PATHS[n]) for n, v in FIXED.items()]
            entries.append(Parameter(name="random_seed", value=item["seed"], path=SEED_PATH))
            sim_dir: Path | None = None
            t0 = time.time()
            try:
                config_path = manager.build_simulation_config(
                    ParameterList(parameters=entries),
                    out_dir_name=f"eval_{item['point_index']:05d}_{item['rep']:03d}",
                )
                _rewrite_generated_config_paths(config_path)
                sim_dir = Path(config_path).parent
                manager.run_simulation(config_path)
                sim_s = time.time() - t0
                rows = extract_corpus_row(
                    sim_dir, bins=BIN_COUNTS, extract_from=args.extract_from,
                    write_every=args.write_every, timesteps=args.timesteps,
                )
            except Exception as exc:  # noqa: BLE001 - one bad theta must not kill the sweep
                failures += 1
                print(f"[screen] rank {rank} FAILED {item['label']} rep {item['rep']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                rows = []
                sim_s = time.time() - t0
            finally:
                # Unconditional: the scratch inode budget is the binding
                # constraint on this campaign, not disk space.
                if sim_dir is not None:
                    shutil.rmtree(sim_dir, ignore_errors=True)
            for row in rows:
                record = dict(item, sim_s=sim_s, extract_s=time.time() - t0 - sim_s, **row)
                sink.write(json.dumps(record))
                sink.write("\n")
            written += len(rows)
            if written and written % 200 == 0:
                sink.flush()

    shutil.rmtree(sim_root, ignore_errors=True)
    totals = comm.gather((len(mine), written, failures), root=0)
    if rank == 0:
        n_eval = sum(t[0] for t in totals)
        n_rows = sum(t[1] for t in totals)
        n_fail = sum(t[2] for t in totals)
        print(f"[screen] done: {n_eval} evaluations, {n_rows} corpus rows, "
              f"{n_fail} failures", flush=True)
        if n_fail > n_eval * 0.1:
            raise RuntimeError(
                f"{n_fail}/{n_eval} evaluations failed; the protocol or the "
                "parameter ranges are broken, refusing to report a screen on it"
            )


# --------------------------------------------------------------------------
# Analyze mode
#
# Every spread below is measured with a ROBUST scale (1.4826 x MAD), not a
# standard deviation.  The discrepancy distribution has a heavy right tail: a
# handful of degenerate simulations -- too few cells for a meaningful g(r), so
# the PCA z-scores land far out -- otherwise decide the answer.  Measured on the
# 50^3 corpus, moving from one snapshot to three raised the RMS within-theta
# scatter of pair_correlation_gofr from 4.8 to 22.4 purely through such
# outliers, while every other block stayed put.  The outlier RATE is reported
# separately rather than swept up in the scale estimate.
# --------------------------------------------------------------------------
class _Item:
    """Minimal DataHandler-like carrier so the shipped transforms can be reused."""

    __slots__ = ("features", "feature_metadata")

    def __init__(self, features: Dict[str, np.ndarray], metadata: Any) -> None:
        self.features = features
        self.feature_metadata = metadata


def load_corpus(corpus_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = sorted(corpus_dir.glob("rank_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no corpus shards under {corpus_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _robust_sd(values: Sequence[float]) -> float:
    """1.4826 x median absolute deviation, i.e. an outlier-proof sigma."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    scale = 1.4826 * float(np.median(np.abs(array - np.median(array))))
    if scale > 0:
        return scale
    # A zero MAD means over half the sample is identical -- a dead feature, or a
    # sample too small for the MAD to resolve.  Say so with the plain sd rather
    # than reporting a noise floor of exactly zero.
    return float(np.std(array, ddof=1))


def _vector_sd(residuals: Sequence[np.ndarray]) -> float:
    """Robust Euclidean scale of a set of centred vectors.

    Per coordinate, so one blown-up principal component cannot masquerade as
    scatter in all of them.
    """
    if not residuals:
        return float("nan")
    matrix = np.vstack(residuals)
    return float(np.sqrt(np.sum([_robust_sd(matrix[:, j]) ** 2
                                 for j in range(matrix.shape[1])])))


def _units(rows: Sequence[Dict[str, Any]], *, bins: int, tsteps: Sequence[int],
           group: int) -> Dict[int, Dict[str, Any]]:
    """Collapse the corpus into evaluation units for one protocol.

    A *unit* is what one ABC evaluation would deliver under the protocol: the
    feature arrays averaged over the chosen snapshots and over ``group``
    replicate seeds.  Averaging raw arrays (not distances) is the shipped
    behaviour of ``DistanceMetric.calculate_distance_replicates``.
    """
    wanted = set(int(t) for t in tsteps)
    per_rep: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        if int(row["bins"]) != int(bins) or int(row["tstep"]) not in wanted:
            continue
        key = (int(row["point_index"]), int(row["rep"]))
        slot = per_rep.setdefault(key, dict(meta=row, frames={}))
        # Trajectory features describe the whole simulation, so they ride along
        # on every frame; averaging a constant over frames is a no-op.
        slot["frames"][int(row["tstep"])] = dict(row["features"], **row.get("trajectory", {}))

    points: Dict[int, Dict[str, Any]] = {}
    for (point_index, _rep), slot in sorted(per_rep.items()):
        if set(slot["frames"]) != wanted:
            continue  # an incomplete evaluation is not an evaluation
        stacked: Dict[str, np.ndarray] = {}
        names = set.intersection(*(set(f) for f in slot["frames"].values()))
        for name in names:
            arrays = [np.asarray(slot["frames"][t][name], dtype=float) for t in sorted(wanted)]
            length = min(a.size for a in arrays)
            stacked[name] = np.mean(np.vstack([a[:length] for a in arrays]), axis=0)
        entry = points.setdefault(point_index, dict(meta=slot["meta"], reps=[]))
        entry["reps"].append(stacked)

    for entry in points.values():
        reps = entry["reps"]
        units: List[Dict[str, np.ndarray]] = []
        for start in range(0, len(reps) - group + 1, group):
            chunk = reps[start:start + group]
            names = set.intersection(*(set(r) for r in chunk))
            unit = {}
            for name in names:
                arrays = [r[name] for r in chunk]
                length = min(a.size for a in arrays)
                unit[name] = np.mean(np.vstack([a[:length] for a in arrays]), axis=0)
            units.append(unit)
        entry["units"] = units
    return points


def shipped_model() -> Any:
    """The feature-space model the paper's campaign actually ran with."""
    from data.feature_scaling import FeatureSpaceModel

    return FeatureSpaceModel.load_json(ASSETS / "sims_feature_space_model.json")


def shipped_reference_features(bins: int) -> Dict[str, np.ndarray]:
    """Raw features of the bundled reference simulation -- the observed data."""
    from data.DataHandler import build_datahandler_for_dir
    from inference.feature_space import _feature_value_to_array

    handler = build_datahandler_for_dir(
        ASSETS / "reference_data", feature_metadata=bin_metadata(bins),
        base_kwargs={"scale_factor": 1}, default_timestep_range=-1,
    )
    handler.extract_all_features(show_progress=False, quiet=True)
    out: Dict[str, np.ndarray] = {}
    for name, payload in handler.features.items():
        values = list(payload.values()) if isinstance(payload, dict) else [payload]
        arrays = [a for a in (_feature_value_to_array(v) for v in values) if a is not None]
        if arrays:
            out[name] = np.asarray(arrays[-1], dtype=float).ravel()
    return out


def _fit_model(points: Dict[int, Dict[str, Any]], *, bins: int, seed: int,
               extras: Any) -> Any:
    """Fit the feature-space model on the space-filling stratum.

    Fitting on the LHS stratum mirrors how the shipped model was built (over a
    corpus of training simulations) and keeps the metric independent of the
    anchor replicates whose scatter it is used to measure.
    """
    from data.feature_scaling import FeatureSpaceModel

    metadata = bin_metadata(bins, extras=extras)
    dataset: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in metadata}
    count = 0
    for entry in points.values():
        if entry["meta"]["stratum"] != "lhs":
            continue
        for unit in entry["units"]:
            for name, array in unit.items():
                if name in dataset:
                    dataset[name][f"s{count}"] = array
            count += 1
    if count < 20:
        raise ValueError(f"only {count} LHS units at bins={bins}; too few to fit a metric")
    model = FeatureSpaceModel.fit_from_dataset(dataset, metadata, ensure_extracted=False,
                                               norm_random_state=seed)
    # The shipped model leaves feature_weights empty, i.e. every block gets
    # 1/n_blocks.  Keep that so the reported discrepancy is the paper's.
    model.feature_weights = {}
    return model


def _z_of(model: Any, unit: Dict[str, np.ndarray],
          blocks: Sequence[str]) -> Dict[str, np.ndarray] | None:
    item = _Item({name: np.asarray(v, dtype=float) for name, v in unit.items()},
                 model.feature_metadata)
    out: Dict[str, np.ndarray] = {}
    for name in blocks:
        z = model._item_feature_z(item, name)
        if z is None:
            return None
        out[name] = np.asarray(z, dtype=float).ravel()
    return out


# Block weightings compared for every protocol.  ``equal`` is the shipped
# metric (feature_weights empty -> 1/n per block); the others are the
# candidate repairs this screen exists to evaluate.
WEIGHTINGS = ("equal", "drop_dead", "scalars_only", "snr_pruned")


def _weights(scheme: str, blocks: Sequence[str], snr: Dict[str, float],
             within: Dict[str, float]) -> Dict[str, float]:
    if scheme == "equal":
        chosen = list(blocks)
    elif scheme == "drop_dead":
        chosen = [b for b in blocks if within[b] > 1e-9]
    elif scheme == "scalars_only":
        chosen = [b for b in blocks if b in ("log_n", "log_r95")]
    elif scheme == "snr_pruned":
        chosen = [b for b in blocks
                  if within[b] > 1e-9 and (snr[b] == snr[b]) and snr[b] >= 1.0]
    else:
        raise ValueError(f"unknown weighting scheme '{scheme}'")
    if not chosen:
        return {}
    return {b: 1.0 / len(chosen) for b in chosen}


def _rho(z: Dict[str, np.ndarray], z_ref: Dict[str, np.ndarray],
         block_norms: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for name, weight in weights.items():
        diff = z[name] - z_ref[name]
        total += weight * float(np.dot(diff, diff)) / block_norms[name]
    return total


def _spread(per_point: Dict[int, List[float]], anchors: Sequence[int],
            others: Sequence[int]) -> Dict[str, float]:
    """Within-theta noise floor, between-theta spread and their ratio."""
    residuals: List[float] = []
    for index in anchors:
        values = per_point[index]
        if len(values) < 2:
            continue
        centre = float(np.median(values))
        residuals.extend(float(v - centre) for v in values)
    within = _robust_sd(residuals)
    units_per_point = float(np.mean([len(per_point[i]) for i in others])) if others else 1.0
    medians = [float(np.median(per_point[i])) for i in others]
    total = _robust_sd(medians)
    between = float(np.sqrt(max(0.0, total ** 2 - within ** 2 / units_per_point)))
    outliers = 0
    counted = 0
    for index in list(anchors) + list(others):
        for value in per_point[index]:
            counted += 1
            if within > 0 and abs(value - float(np.median(per_point[index]))) > 10 * within:
                outliers += 1
    return dict(within=within, between=between,
                snr=(between / within) if within > 0 else float("nan"),
                total=total, units_per_point=units_per_point,
                outlier_rate=(outliers / counted) if counted else float("nan"))


def screen(points: Dict[int, Dict[str, Any]], *, bins: int, seed: int,
           extras: Any = False, shipped: bool = False) -> Dict[str, Any]:
    """Within/between/identifiability for one protocol.

    With ``shipped``, the metric is not refitted: the paper's own feature-space
    model and reference simulation are used, so the discrepancy is the one the
    stored campaign computed rather than one tuned to this design.
    """
    if shipped:
        if bins != 32:
            raise ValueError("the shipped feature-space model is fitted for 32 bins")
        model = shipped_model()
    else:
        model = _fit_model(points, bins=bins, seed=seed, extras=extras)
    blocks = list(model.scalers)
    block_norms = {name: (float(model.block_norms.get(name, 1.0)) or 1.0) for name in blocks}

    zs: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for index, entry in points.items():
        vectors = [z for z in (_z_of(model, u, blocks) for u in entry["units"]) if z is not None]
        if vectors:
            zs[index] = vectors

    ref_index = next((i for i, e in points.items() if e["meta"]["stratum"] == "reference"), None)
    if ref_index is None or ref_index not in zs:
        raise ValueError("no usable reference stratum in the corpus")
    if shipped:
        z_ref = _z_of(model, shipped_reference_features(bins), blocks)
        if z_ref is None:
            raise ValueError("the bundled reference simulation does not transform "
                             "under the shipped feature-space model")
    else:
        z_ref = {name: np.median(np.vstack([z[name] for z in zs[ref_index]]), axis=0)
                 for name in blocks}

    def stratum(name: str) -> List[int]:
        return [i for i, e in points.items() if e["meta"]["stratum"] == name and i in zs]

    anchors, lhs = stratum("anchor"), stratum("lhs")
    if not anchors or not lhs:
        raise ValueError("corpus is missing the anchor or the LHS stratum")

    # ---- per block, in the z-space the distance actually uses ----
    block_stats: Dict[str, Dict[str, float]] = {}
    for name in blocks:
        residuals: List[np.ndarray] = []
        for index in anchors:
            stack = np.vstack([z[name] for z in zs[index]])
            residuals.extend(list(stack - np.median(stack, axis=0)))
        within = _vector_sd(residuals)
        medians = [np.median(np.vstack([z[name] for z in zs[i]]), axis=0) for i in lhs]
        grand = np.median(np.vstack(medians), axis=0)
        total = _vector_sd([m - grand for m in medians])
        units_per_point = float(np.mean([len(zs[i]) for i in lhs]))
        between = float(np.sqrt(max(0.0, total ** 2 - within ** 2 / units_per_point)))
        block_stats[name] = dict(
            within=within, between=between,
            snr=(between / within) if within > 1e-12 else float("nan"),
            dims=int(len(z_ref[name])), block_norm=block_norms[name],
            dead=bool(within <= 1e-12),
        )

    snr_by_block = {n: block_stats[n]["snr"] for n in blocks}
    within_by_block = {n: block_stats[n]["within"] for n in blocks}

    # ---- the scalar discrepancy, under each candidate block weighting ----
    weightings: Dict[str, Dict[str, Any]] = {}
    for scheme in WEIGHTINGS:
        weights = _weights(scheme, blocks, snr_by_block, within_by_block)
        if not weights:
            continue
        per_point = {i: [_rho(z, z_ref, block_norms, weights) for z in v]
                     for i, v in zs.items()}
        stats = _spread(per_point, anchors, lhs)
        stats["blocks"] = sorted(weights)
        weightings[scheme] = stats
        if scheme == "equal":
            equal_rho = per_point

    return dict(
        bins=bins, blocks=blocks, block_stats=block_stats, block_norms=block_norms,
        strata={i: points[i]["meta"]["stratum"] for i in zs},
        weightings=weightings,
        within_rho=weightings["equal"]["within"],
        between_rho=weightings["equal"]["between"],
        snr_rho=weightings["equal"]["snr"],
        outlier_rate=weightings["equal"]["outlier_rate"],
        n_anchor_units=int(sum(len(zs[i]) for i in anchors)),
        n_lhs_points=len(lhs),
        model=model, zs=zs, rho=equal_rho, z_ref=z_ref,
        within_by_block=within_by_block,
    )


def _whiten(result: Dict[str, Any]) -> Dict[str, Any]:
    """Put every unit in a noise-whitened summary space.

    Concatenate the per-block z-vectors, then divide each coordinate by its
    within-theta robust sd.  One unit of distance is then one unit of Monte
    Carlo noise, so a response of length d means "this parameter moves the
    summary by d noise sigmas" -- directly comparable across parameters and
    across blocks of different dimension.  Coordinates whose noise is exactly
    zero are dead and are dropped rather than divided by.

    The whitening is by marginal sd, not by the full noise covariance: residual
    correlation between coordinates is not removed, so these are discriminability
    estimates, not exact Mahalanobis distances.
    """
    zs, blocks = result["zs"], result["blocks"]
    anchors = [i for i, v in zs.items() if result["strata"][i] == "anchor"]

    def flat(z: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([z[b] for b in blocks])

    residuals: List[np.ndarray] = []
    for index in anchors:
        stack = np.vstack([flat(z) for z in zs[index]])
        residuals.extend(list(stack - np.median(stack, axis=0)))
    matrix = np.vstack(residuals)
    noise = np.asarray([_robust_sd(matrix[:, j]) for j in range(matrix.shape[1])])
    live = np.nonzero(noise > 1e-12)[0]
    if live.size == 0:
        raise ValueError("every summary coordinate has zero noise; corpus is degenerate")

    labels: List[str] = []
    for block in blocks:
        labels.extend([block] * len(zs[anchors[0]][0][block]))
    return dict(
        vectors={i: [flat(z)[live] / noise[live] for z in v] for i, v in zs.items()},
        labels=[labels[j] for j in live], noise=noise, live=live,
    )


def _median_filter3(values: np.ndarray) -> np.ndarray:
    """Three-point median filter with edge replication.

    A sweep occasionally contains one level whose median is thrown far out by a
    degenerate simulation -- too few cells for g(r) or the radial profiles to
    mean anything, so the PCA z-scores explode.  Measured on the round-2 50^3
    corpus, both adhesion sweeps were flat at every level except a single spike
    (adhesion_cl: -3 everywhere, +41 at one level), and an unfiltered monotone
    fit reads that spike as a 25-sigma step change.  A median filter removes an
    isolated level without touching a genuine ramp or step, which by definition
    spans more than one level.
    """
    padded = np.concatenate([values[:1], values, values[-1:]])
    return np.asarray([float(np.median(padded[i:i + 3])) for i in range(len(values))])


def _isotonic(values: np.ndarray) -> np.ndarray:
    """Best monotone fit of a sweep, in whichever direction fits better.

    Pool-adjacent-violators: the response of a physical parameter is monotone
    or saturating, so a monotone fit removes level-to-level Monte Carlo noise
    without assuming a functional form.
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.arange(len(values), dtype=float)
    best, best_error = None, float("inf")
    for increasing in (True, False):
        fitted = IsotonicRegression(increasing=increasing).fit_transform(x, values)
        error = float(np.sum((fitted - values) ** 2))
        if error < best_error:
            best, best_error = fitted, error
    return np.asarray(best, dtype=float)


# Fraction of a sweep's monotone excursion that the "responsive window" must
# carry.  Not 95%: the last few percent are usually a slow drift across the
# saturated plateau, and demanding them stretches the window over the whole
# prior even when the entire real response is one step.
RESPONSIVE_FRACTION = 0.80

# Below this many cells a 32-bin radial profile or g(r) has fewer than one cell
# per bin, so the curve blocks carry noise rather than structure.  The benchmark
# has no guard against landing there: CellularPotts.simulate returns a finite
# distance for such a run exactly as it does for a good one.
DEGENERATE_CELLS = 20


def _responsive_window(fit: np.ndarray) -> tuple[int, int]:
    """Shortest contiguous level range carrying most of a monotone response.

    Everything outside it is prior mass on which the summary barely moves, i.e.
    prior the sampler cannot resolve.
    """
    excursion = abs(float(fit[-1] - fit[0]))
    n = len(fit)
    if excursion <= 0:
        return 0, n - 1
    target = RESPONSIVE_FRACTION * excursion
    best = (0, n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(fit[j] - fit[i])) >= target:
                if (j - i) < (best[1] - best[0]):
                    best = (i, j)
                break
    return best


def _direction(level_medians: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Leading direction of a sweep, and each level's coordinate along it."""
    matrix = np.vstack(level_medians)
    centred = matrix - matrix.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    projection = centred @ direction
    if projection[-1] < projection[0]:
        direction, projection = -direction, -projection
    return direction, projection


def parameter_report(points: Dict[int, Dict[str, Any]],
                     result: Dict[str, Any]) -> Dict[str, Any]:
    """Per-parameter identifiability, saturation window, and confounding.

    The OAT sweep holds every other parameter at the template value, so the
    response measured here is that parameter's alone.  Two numbers matter and
    they are not the same:

    ``snr``          how far the sweep moves the summary, in units of the Monte
                     Carlo noise along the SAME direction.  Below ~1 the
                     parameter is invisible however long a campaign runs.
    ``confounding``  the angle between two parameters' response directions.  Two
                     parameters that move the summary the same way cannot be
                     told apart no matter how large either SNR is, so a high
                     marginal SNR is necessary but not sufficient.
    """
    whitened = _whiten(result)
    vectors = whitened["vectors"]
    zs = result["zs"]
    anchors = [i for i in vectors if result["strata"][i] == "anchor"]

    anchor_residuals: List[np.ndarray] = []
    for index in anchors:
        stack = np.vstack(vectors[index])
        anchor_residuals.extend(list(stack - np.median(stack, axis=0)))
    residual_matrix = np.vstack(anchor_residuals)

    rows: List[Dict[str, Any]] = []
    directions: Dict[str, np.ndarray] = {}
    for name in CANDIDATES:
        levels = sorted(
            (i for i, e in points.items()
             if e["meta"]["stratum"] == "oat" and e["meta"]["swept"] == name and i in vectors),
            key=lambda i: points[i]["meta"]["params"][name],
        )
        if len(levels) < 3:
            continue
        values = np.asarray([points[i]["meta"]["params"][name] for i in levels], dtype=float)
        medians = [np.median(np.vstack(vectors[i]), axis=0) for i in levels]
        direction, projection = _direction(medians)
        directions[name] = direction

        units_per_level = float(np.mean([len(vectors[i]) for i in levels]))
        within = _robust_sd(residual_matrix @ direction)
        total = _robust_sd(_median_filter3(projection))
        between = float(np.sqrt(max(0.0, total ** 2 - within ** 2 / units_per_level)))
        snr = (between / within) if within > 0 else float("nan")

        # Which blocks carry the response: squared weight of the direction.
        weight_by_block: Dict[str, float] = {}
        for label, component in zip(whitened["labels"], direction):
            weight_by_block[label] = weight_by_block.get(label, 0.0) + float(component) ** 2
        carried = sorted(weight_by_block.items(), key=lambda kv: -kv[1])

        # Saturation is read off a MONOTONE fit of the sweep, not off the raw
        # level medians.  Raw medians carry level-to-level noise that a plain
        # 5%-95% rule mistakes for response: a parameter whose whole effect is a
        # step at the bottom edge of its prior (persistence, measured) otherwise
        # reports a "responsive window" covering the entire prior.
        fit = _isotonic(_median_filter3(projection))
        span = float(abs(fit[-1] - fit[0]))
        lo_i, hi_i = _responsive_window(fit)
        window = (float(values[lo_i]), float(values[hi_i]))
        prior_fraction = abs(_to_unit(name, window[1]) - _to_unit(name, window[0]))
        sigmas_moved = (span / within) if within > 0 else float("nan")

        rows.append(dict(
            parameter=name, prior=[CANDIDATES[name]["lo"], CANDIDATES[name]["hi"]],
            scale=CANDIDATES[name]["scale"], drives=CANDIDATES[name]["drives"],
            within=within, between=between, snr=snr,
            sigmas_moved=float(sigmas_moved),
            carried_by=[[block, round(weight, 3)] for block, weight in carried[:3]],
            responsive_window=list(window),
            responsive_prior_fraction=float(prior_fraction),
            levels=[dict(value=float(v), response=float(r))
                    for v, r in zip(values, projection)],
        ))
    rows.sort(key=lambda r: -(r["sigmas_moved"] if r["sigmas_moved"] == r["sigmas_moved"] else -1.0))

    names = list(directions)
    confounding = {a: {b: float(abs(np.dot(directions[a], directions[b])))
                       for b in names} for a in names}
    return dict(parameters=rows, confounding=confounding,
                n_coordinates=int(len(whitened["labels"])),
                dead_coordinates=int(len(whitened["noise"]) - len(whitened["live"])))


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def run_analyze(args: argparse.Namespace) -> None:
    from async_abc.benchmarks.cellular_potts import _ensure_nastjapy_on_path

    _ensure_nastjapy_on_path()
    out_dir = Path(args.out).resolve()
    rows = load_corpus(out_dir / "corpus")
    with open(out_dir / "protocol.json") as f:
        protocol = json.load(f)
    # Re-apply the priors the corpus was generated under, so "share of prior" is
    # measured against the right prior rather than against this file's defaults.
    recorded = protocol.get("candidates")
    if not recorded:
        raise KeyError(
            f"{out_dir / 'protocol.json'} has no 'candidates' record, so the priors "
            "this corpus was generated under are unknown. Every 'share of prior' "
            "below would be measured against whatever this script's defaults "
            "happen to be now. Re-run the simulate step, or add the record by "
            "hand from the run's own arguments."
        )
    for name in list(CANDIDATES):
        if name not in recorded:
            del CANDIDATES[name]
    for name, spec in recorded.items():
        CANDIDATES.setdefault(name, {}).update(spec)

    if args.restrict:
        bounds: Dict[str, tuple] = {}
        for spec in args.restrict:
            name, _, rest = spec.partition("=")
            lo, _, hi = rest.partition(":")
            if name not in CANDIDATES:
                raise KeyError(f"--restrict names unknown parameter '{name}'")
            bounds[name] = (float(lo), float(hi))
        before = len(rows)
        # The reference stratum is the pseudo-observed data, not a design point:
        # restricting the design must not move the thing the design is compared to.
        rows = [r for r in rows
                if r["stratum"] == "reference"
                or all(lo <= float(r["params"][n]) <= hi for n, (lo, hi) in bounds.items())]
        print("Restricted to "
              + ", ".join(f"{n} in [{lo:g}, {hi:g}]" for n, (lo, hi) in bounds.items())
              + f": {len(rows)}/{before} corpus rows retained.\n")
        if not rows:
            raise ValueError("--restrict kept nothing")

    tsteps = sorted({int(r["tstep"]) for r in rows})
    available_bins = sorted({int(r["bins"]) for r in rows})
    max_reps = max(int(r["rep"]) for r in rows) + 1
    cells = [int(r["n_cells"]) for r in rows if int(r["tstep"]) == tsteps[-1]
             and int(r["bins"]) == available_bins[-1]]
    print(f"# CPM screening -- blocksize {protocol['blocksize']}, "
          f"{len(rows)} corpus rows, {len({(r['point_index'], r['rep']) for r in rows})} evaluations")
    print(f"snapshots {tsteps}   bins {available_bins}   replicate seeds up to {max_reps}")
    degenerate = float(np.mean([c < DEGENERATE_CELLS for c in cells]))
    print(f"cells at t={tsteps[-1]}: median {int(np.median(cells))}, "
          f"10-90% {int(np.quantile(cells, 0.1))}-{int(np.quantile(cells, 0.9))}; "
          f"{degenerate:.1%} of evaluations below {DEGENERATE_CELLS} cells, where the "
          f"curve summaries stop meaning anything\n")

    variants: List[Dict[str, Any]] = []
    for bins in available_bins:
        for n_snap in args.snapshot_counts:
            if n_snap > len(tsteps):
                continue
            for group in args.seed_groups:
                variants.append(dict(bins=bins, tsteps=tsteps[-n_snap:], group=group))

    summary: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}
    print("## Protocol screen -- signal-to-noise of the reported discrepancy\n")
    print("| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) "
          "| SNR (best weighting) | outliers |")
    print("|---|---|---|---|---|---|---|---|")
    for variant in variants:
        points = _units(rows, bins=variant["bins"], tsteps=variant["tsteps"],
                        group=variant["group"])
        try:
            result = screen(points, bins=variant["bins"], seed=args.fit_seed,
                            extras=(True if args.extra_blocks == [] else args.extra_blocks),
                            shipped=args.shipped_model)
        except ValueError as exc:
            print(f"| {variant['bins']} | {len(variant['tsteps'])} | {variant['group']} "
                  f"| skipped: {exc} | | | | |")
            continue
        key = f"b{variant['bins']}_s{len(variant['tsteps'])}_k{variant['group']}"
        best_scheme = max(result["weightings"],
                          key=lambda s: (result["weightings"][s]["snr"]
                                         if result["weightings"][s]["snr"] == result["weightings"][s]["snr"]
                                         else -1.0))
        best_snr = result["weightings"][best_scheme]["snr"]
        print(f"| {variant['bins']} | {len(variant['tsteps'])} | {variant['group']} "
              f"| {_fmt(result['within_rho'], 4)} | {_fmt(result['between_rho'], 4)} "
              f"| **{_fmt(result['snr_rho'], 2)}** "
              f"| {_fmt(best_snr, 2)} ({best_scheme}) "
              f"| {result['outlier_rate']:.1%} |")
        summary.append(dict(key=key, bins=variant["bins"], snapshots=variant["tsteps"],
                            seeds=variant["group"], within_rho=result["within_rho"],
                            between_rho=result["between_rho"], snr_rho=result["snr_rho"],
                            outlier_rate=result["outlier_rate"],
                            weightings={s: {k: v for k, v in st.items()}
                                        for s, st in result["weightings"].items()},
                            block_stats=result["block_stats"],
                            n_anchor_units=result["n_anchor_units"],
                            n_lhs_points=result["n_lhs_points"]))
        detail[key] = (points, result)

    if not summary:
        raise RuntimeError("no protocol variant could be screened")
    best = max(summary, key=lambda s: (s["snr_rho"] if s["snr_rho"] == s["snr_rho"] else -1.0))
    print(f"\nBest protocol on the shipped (equal-weight) metric: **{best['key']}** -- "
          f"{best['bins']} bins, {len(best['snapshots'])} snapshot(s) at {best['snapshots']}, "
          f"{best['seeds']} seed(s) per evaluation; SNR {_fmt(best['snr_rho'], 2)}\n")

    print("## Per-block behaviour under that protocol\n")
    print("| block | dims | within-theta | between-theta | SNR | block norm |")
    print("|---|---|---|---|---|---|")
    stats = best["block_stats"]
    for block in sorted(stats, key=lambda b: -(stats[b]["snr"] if stats[b]["snr"] == stats[b]["snr"] else -1.0)):
        note = "  (dead)" if stats[block]["dead"] else ""
        print(f"| {block}{note} | {stats[block]['dims']} | {_fmt(stats[block]['within'], 4)} "
              f"| {_fmt(stats[block]['between'], 4)} | {_fmt(stats[block]['snr'], 2)} "
              f"| {_fmt(stats[block]['block_norm'], 3)} |")

    print("\n## Block weighting\n")
    print("| weighting | blocks kept | within-theta | between-theta | SNR |")
    print("|---|---|---|---|---|")
    for scheme, stat in best["weightings"].items():
        print(f"| {scheme} | {len(stat['blocks'])} | {_fmt(stat['within'], 4)} "
              f"| {_fmt(stat['between'], 4)} | **{_fmt(stat['snr'], 2)}** |")

    points, result = detail[best["key"]]
    screen_out = parameter_report(points, result)
    parameters = screen_out["parameters"]
    print(f"\n## Parameter screen under that protocol\n")
    print(f"Summary space: {screen_out['n_coordinates']} live coordinates "
          f"({screen_out['dead_coordinates']} dead). Distances are in units of the "
          f"Monte Carlo noise along each parameter's own response direction.\n")
    print("| parameter | prior | identifiability | sigmas moved | carried by "
          "| responsive window | share of prior |")
    print("|---|---|---|---|---|---|---|")
    for row in parameters:
        lo, hi = row["prior"]
        wlo, whi = row["responsive_window"]
        carried = ", ".join(f"{b} {w:.0%}" for b, w in row["carried_by"][:2])
        print(f"| {row['parameter']} | [{lo:g}, {hi:g}] {row['scale']} "
              f"| **{_fmt(row['snr'], 2)}** | {row['sigmas_moved']:.0f} "
              f"| {carried} | [{wlo:.4g}, {whi:.4g}] "
              f"| {row['responsive_prior_fraction']:.0%} |")

    names = [r["parameter"] for r in parameters]
    print("\n## Confounding -- |cos| between response directions\n")
    print("A pair near 1 moves the summary the same way and cannot be separated, "
          "however identifiable each is on its own.\n")
    print("| |" + "|".join(f" {n} " for n in names) + "|")
    print("|---|" + "|".join("---" for _ in names) + "|")
    for a in names:
        cells = "|".join(f" {screen_out['confounding'][a][b]:.2f} " for b in names)
        print(f"| **{a}** |{cells}|")

    report = dict(protocol=protocol, protocol_screen=summary,
                  best_protocol=best["key"], parameters=parameters,
                  confounding=screen_out["confounding"],
                  n_coordinates=screen_out["n_coordinates"],
                  dead_coordinates=screen_out["dead_coordinates"])
    report_path = out_dir / args.report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nWrote {report_path}")


# --------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["simulate", "analyze"], required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="run directory (corpus, assets, report)")
    parser.add_argument("--sim-dir", default=None,
                        help="scratch root for per-evaluation simulation dirs "
                             "(default: <out>/sims). Always removed afterwards.")
    # protocol
    parser.add_argument("--blocksize", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=501)
    parser.add_argument("--write-every", type=int, default=50,
                        help="CellInfo writer cadence; extra snapshots are free")
    parser.add_argument("--extract-from", type=int, default=250,
                        help="ignore snapshots before this simulation time")
    # design
    parser.add_argument("--n-anchors", type=int, default=10)
    parser.add_argument("--anchor-seeds", type=int, default=24)
    parser.add_argument("--oat-levels", type=int, default=14)
    parser.add_argument("--oat-seeds", type=int, default=6)
    parser.add_argument("--n-lhs", type=int, default=400)
    parser.add_argument("--lhs-seeds", type=int, default=4)
    parser.add_argument("--ref-seeds", type=int, default=48)
    parser.add_argument("--design-seed", type=int, default=20260918)
    parser.add_argument("--prior", action="append", default=[], metavar="NAME=LO:HI[:SCALE]",
                        help="re-range a candidate, e.g. persistence=0.002:0.3:log "
                             "(repeatable; recorded in protocol.json)")
    parser.add_argument("--parameters", nargs="+", default=None,
                        help="screen only these candidates (default: all)")
    parser.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                        help="apply a candidate at a constant value without screening "
                             "it (repeatable); recorded in protocol.json")
    parser.add_argument("--centre", action="append", default=[], metavar="NAME=VALUE",
                        help="move the reference point -- the theta standing in for "
                             "observed data -- in physical units (repeatable)")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N evaluations (smoke tests)")
    # analysis
    parser.add_argument("--snapshot-counts", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--seed-groups", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--report", default="screening_report.json",
                        help="report filename inside --out; give variants their "
                             "own name so a control run cannot overwrite the main one")
    parser.add_argument("--shipped-model", action="store_true",
                        help="score with the paper's own feature-space model and "
                             "reference simulation instead of refitting the metric "
                             "on this design (32 bins only)")
    parser.add_argument("--restrict", action="append", default=[], metavar="NAME=LO:HI",
                        help="analyse only the part of the corpus inside these "
                             "bounds (repeatable). Use it to ask what the signal "
                             "looks like in the sub-region a converged sampler "
                             "occupies, rather than across the whole prior.")
    parser.add_argument("--extra-blocks", nargs="*", default=None, metavar="NAME",
                        help="also score optional blocks the shipped feature set "
                             "ignores. Bare flag uses the vetted default set; "
                             f"names available: {', '.join(sorted(set(EXTRA_BLOCKS) | set(CAMPAIGN_BLOCKS) | set(TRAJECTORY_BLOCKS)))}")
    args = parser.parse_args(argv)
    apply_prior_overrides(args.prior, args.parameters)
    apply_centre_overrides(args.centre)
    apply_fixed(args.fix)

    if args.mode == "simulate":
        run_simulate(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
