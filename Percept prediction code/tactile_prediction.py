"""Public, end-to-end LOPO tactile prediction from the released raw CSV files.

Place this script beside ``p1.csv`` through ``p24.csv`` and run it directly.
It creates the two final tactile-prediction figures without inferential
statistics and without using any file outside this directory.

Methods
-------
1. Only finally accepted ``ET`` responses are used. Points are used directly;
   every drawn polygon is represented by its polygon centroid. Rejected redraws
   and ``no_sensation`` trials do not contribute tactile components.
2. Each outer leave-one-participant-out (LOPO) fold fits a population tactile
   map from the other 23 participants. The manuscript-defined one-/two-
   component pair structure is fixed before LOPO, while all component centers
   and assignments are re-estimated within the fold.
3. Five calibration pairs are chosen greedily using nested participant-level
   LOPO inside the 23-person outer training set. Candidate selection minimizes
   the same prediction error used for final evaluation. The outer participant
   is never used to choose calibration pairs or fit the population map.
4. The personalized prediction is the fold population tactile map translated
   by the mean pair-level calibration residual. The mean offset is shrunk by
   ``K_valid / (K_valid + n0)``, with the prespecified ``n0 = 4``. A missing
   calibration response counts as an attempted pair, is not replaced, and does
   not contribute a residual vector.
5. Calibration pairs are excluded from evaluation. The three compared methods
   are the physical electrode-pair midpoint, the training-only population
   tactile map, and that map plus the nested M0 global offset.

Prediction error is the mean distance from each observed tactile component to
its nearest predicted component.

Outputs
-------
``figure_outputs/tactile_prediction/`` contains two SVG figures and their two
plain CSV data files. No PNG, PDF, or statistical-test output is created.

Dependencies: Python 3.10+, NumPy, pandas, and Matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figure_outputs" / "tactile_prediction"

PARTICIPANTS = tuple(range(1, 25))
PAIR_IDS = tuple(range(1, 29))
EXPECTED_RAW_COLUMNS = (
    "Trial",
    "PhaseNum",
    "Condition",
    "PairID",
    "Response",
    "ConfirmationStatus",
    "ConfirmationAttempt",
    "ConfirmationRedraws",
    "NumResponseItems",
    "ResponseItems",
    "NumZones",
    "ZonePoints_mm",
    "Centroid_mm_x",
    "Centroid_mm_y",
)

PAIR_LABELS = (
    "LV1",
    "LV3",
    "LV5",
    "MV1",
    "MV2",
    "MV3",
    "MV4",
    "MV5",
    "RV1",
    "RV3",
    "RV5",
    "TH1",
    "TH3",
    "TH5",
    "MH1",
    "MH2",
    "MH3",
    "MH4",
    "MH5",
    "BH1",
    "BH3",
    "BH5",
    "DA1",
    "DA3",
    "DA5",
    "DB1",
    "DB3",
    "DB5",
)
PAIR_LABEL_BY_ID = dict(zip(PAIR_IDS, PAIR_LABELS))

# (anode electrode, cathode electrode) for PairID 1-28. Embedding the layout
# here keeps this public script independent of the private analysis directory.
PAIR_ELECTRODES = (
    (3, 4),
    (2, 5),
    (1, 6),
    (15, 16),
    (14, 16),
    (14, 17),
    (13, 17),
    (13, 18),
    (34, 33),
    (35, 32),
    (36, 31),
    (13, 19),
    (7, 25),
    (1, 36),
    (15, 21),
    (9, 21),
    (9, 27),
    (3, 27),
    (3, 34),
    (18, 24),
    (12, 30),
    (6, 31),
    (15, 22),
    (8, 29),
    (1, 31),
    (21, 16),
    (26, 11),
    (36, 6),
)

# Fixed before all folds, as defined in the manuscript analysis.
TWO_COMPONENT_PAIR_IDS = frozenset({13, 14, 17, 18, 19, 21, 22, 24, 25, 27, 28})
CALIBRATION_BUDGET = 5
N0 = 4.0
TWO_PD_THRESHOLD_MM = 17.5

METHOD_ORDER = ("midpoint", "population", "personalized")
METHOD_LABELS = {
    "midpoint": "Midpoint",
    "population": "Data-driven mean tactile map",
    "personalized": "Data-driven mean + nested M0 global offset",
}
METHOD_COLORS = {
    "midpoint": "#6B7280",
    "population": "#1B9E77",
    "personalized": "#F58518",
}

# Millimetre coordinates used by the experimental interface (electrodes 1-36).
ELECTRODE_POSITIONS_MM = np.asarray(
    [
        (19.90, 4.90),
        (16.90, 24.90),
        (13.90, 44.90),
        (10.90, 64.90),
        (7.90, 84.90),
        (4.90, 104.90),
        (41.90, 4.90),
        (40.10, 24.90),
        (38.30, 44.90),
        (36.50, 64.90),
        (34.70, 84.90),
        (32.90, 104.90),
        (63.90, 4.90),
        (63.30, 24.90),
        (62.70, 44.90),
        (62.10, 64.90),
        (61.50, 84.90),
        (60.90, 104.90),
        (85.90, 4.90),
        (86.50, 24.90),
        (87.10, 44.90),
        (87.70, 64.90),
        (88.30, 84.90),
        (88.90, 104.90),
        (107.90, 4.90),
        (109.70, 24.90),
        (111.50, 44.90),
        (113.30, 64.90),
        (115.10, 84.90),
        (116.90, 104.90),
        (144.90, 104.90),
        (141.90, 84.90),
        (138.90, 64.90),
        (135.90, 44.90),
        (132.90, 24.90),
        (129.90, 4.90),
    ],
    dtype=float,
)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "svg.fonttype": "none",
    }
)


def as_component_set(points: object) -> np.ndarray:
    """Return finite two-dimensional points with shape (n_components, 2)."""
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim == 1 and array.shape == (2,):
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Invalid component-set shape: {array.shape}")
    return array[np.all(np.isfinite(array), axis=1)]


def directed_prediction_error(observed: object, predicted: object) -> float:
    """Mean observed-component distance to its nearest predicted component."""
    observed = as_component_set(observed)
    predicted = as_component_set(predicted)
    if len(observed) == 0 or len(predicted) == 0:
        return float("nan")
    distances = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def parse_json_list(value: object, source: str) -> list:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{source} must contain a JSON list.")
    return parsed


def polygon_centroid(zone: object) -> tuple[float, float] | None:
    points = np.asarray(zone, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    twice_area = float(np.sum(cross))
    if abs(twice_area) < 1e-9:
        return float(np.mean(x)), float(np.mean(y))
    centroid_x = float(np.sum((x + x_next) * cross) / (3.0 * twice_area))
    centroid_y = float(np.sum((y + y_next) * cross) / (3.0 * twice_area))
    return centroid_x, centroid_y


def extract_components(row: pd.Series, source: str) -> np.ndarray:
    """Apply the point/polygon-centroid representation to one accepted row."""
    zones = parse_json_list(row["ZonePoints_mm"], f"{source} ZonePoints_mm")
    items = parse_json_list(row["ResponseItems"], f"{source} ResponseItems")
    if int(row["NumZones"]) != len(zones):
        raise ValueError(f"{source}: NumZones does not match ZonePoints_mm.")
    if int(row["NumResponseItems"]) != len(items):
        raise ValueError(f"{source}: NumResponseItems does not match ResponseItems.")

    components: list[tuple[float, float]] = []
    for zone in zones:
        centroid = polygon_centroid(zone)
        if centroid is not None:
            components.append(centroid)

    fallback_areas: list[tuple[float, float]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: each ResponseItems entry must be an object.")
        item_type = item.get("type")
        if item_type == "point":
            point = item.get("point_mm", item.get("strongest_point_mm"))
            if not isinstance(point, list) or len(point) < 2:
                raise ValueError(f"{source}: point response lacks point_mm.")
            components.append((float(point[0]), float(point[1])))
        elif item_type == "area" and not components:
            for zone in item.get("zones_mm") or []:
                centroid = polygon_centroid(zone)
                if centroid is not None:
                    fallback_areas.append(centroid)
        elif item_type != "area":
            raise ValueError(f"{source}: unsupported response-item type {item_type!r}.")
    if not components:
        components.extend(fallback_areas)
    return as_component_set(components)


class ComponentDataset:
    """Accepted tactile component sets indexed by participant and pair."""

    def __init__(self, sets: dict[tuple[int, int], np.ndarray]) -> None:
        self._sets = {
            (int(participant), int(pair_id)): as_component_set(points)
            for (participant, pair_id), points in sets.items()
            if len(as_component_set(points))
        }

    def get_set(self, participant: int, pair_id: int) -> np.ndarray:
        points = self._sets.get((int(participant), int(pair_id)))
        return np.empty((0, 2), dtype=float) if points is None else points.copy()

    def pair_ids(self, participant: int) -> set[int]:
        participant = int(participant)
        return {
            pair_id
            for candidate, pair_id in self._sets
            if candidate == participant
        }


def participant_number(path: Path) -> int | None:
    match = re.fullmatch(r"p(\d+)\.csv", path.name.lower())
    return int(match.group(1)) if match else None


def load_tactile_data(input_dir: Path) -> ComponentDataset:
    """Read the 24 public raw files and retain accepted tactile responses."""
    input_dir = Path(input_dir)
    files = sorted(
        [path for path in input_dir.glob("p*.csv") if participant_number(path) is not None],
        key=lambda path: participant_number(path),
    )
    found = tuple(participant_number(path) for path in files)
    if found != PARTICIPANTS:
        raise ValueError(
            f"Expected p1.csv through p24.csv in {input_dir}; found participant IDs {found}."
        )

    tactile_sets: dict[tuple[int, int], np.ndarray] = {}
    for path in files:
        participant = int(participant_number(path))
        rows = pd.read_csv(path, keep_default_na=False)
        if tuple(rows.columns) != EXPECTED_RAW_COLUMNS:
            raise ValueError(f"{path.name} does not have the expected 14-column schema.")
        if rows.empty:
            raise ValueError(f"{path.name} is empty.")
        phase = pd.to_numeric(rows["PhaseNum"], errors="raise")
        pair_ids = pd.to_numeric(rows["PairID"], errors="raise")
        if (phase >= 57).any():
            raise ValueError(f"{path.name} still contains PhaseNum >= 57.")
        if not pair_ids.between(1, 28).all():
            raise ValueError(f"{path.name} contains PairID outside 1-28.")

        accepted_tactile = rows.loc[
            rows["ConfirmationStatus"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("accepted_yes")
            & rows["Condition"].astype(str).str.strip().eq("ET")
        ]
        for row_index, row in accepted_tactile.iterrows():
            pair_id = int(row["PairID"])
            source = f"{path.name} row {row_index + 2}"
            components = extract_components(row, source)
            if len(components) == 0:
                raise ValueError(f"{source}: accepted ET response has no components.")
            key = (participant, pair_id)
            if key in tactile_sets:
                raise ValueError(
                    f"{path.name}: duplicate accepted ET response for PairID {pair_id}."
                )
            tactile_sets[key] = components

    data = ComponentDataset(tactile_sets)
    for participant in PARTICIPANTS:
        if not data.pair_ids(participant):
            raise ValueError(f"Participant {participant} has no accepted tactile responses.")
    return data


def deterministic_kmeans(points: object, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    points = as_component_set(points)
    if len(points) < n_clusters:
        raise ValueError("Fewer training points than fixed percept components.")
    if n_clusters == 1:
        return np.mean(points, axis=0, keepdims=True)
    if n_clusters != 2:
        raise ValueError("Only the fixed one-/two-component structure is supported.")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.asarray([points[first], points[second]], dtype=float)
    for _iteration in range(max_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        updated = centers.copy()
        for cluster_id in range(n_clusters):
            cluster_points = points[assignments == cluster_id]
            if len(cluster_points):
                updated[cluster_id] = np.mean(cluster_points, axis=0)
            else:
                nearest = np.min(distances, axis=1)
                updated[cluster_id] = points[int(np.argmax(nearest))]
        if np.allclose(updated, centers, atol=1e-9, rtol=0.0):
            centers = updated
            break
        centers = updated
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def build_population_map(
    tactile_data: ComponentDataset,
    training_participants: tuple[int, ...],
    excluded_participants: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Fit fold-specific assignments and equal-participant population centers."""
    training_participants = tuple(sorted(int(value) for value in training_participants))
    excluded = {int(value) for value in excluded_participants}
    if excluded & set(training_participants):
        raise AssertionError("An excluded participant entered a population-map fit.")

    population_map: dict[int, np.ndarray] = {}
    for pair_id in PAIR_IDS:
        n_components = 2 if pair_id in TWO_COMPONENT_PAIR_IDS else 1
        participant_sets = {
            participant: tactile_data.get_set(participant, pair_id)
            for participant in training_participants
        }
        blocks = [points for points in participant_sets.values() if len(points)]
        if not blocks:
            raise ValueError(f"No training tactile responses for PairID {pair_id}.")
        assignment_centers = deterministic_kmeans(np.vstack(blocks), n_components)
        participant_cluster_means: dict[int, list[np.ndarray]] = {
            cluster_id: [] for cluster_id in range(n_components)
        }
        for participant in training_participants:
            participant_set = participant_sets[participant]
            if len(participant_set) == 0:
                continue
            distances = np.linalg.norm(
                participant_set[:, None, :] - assignment_centers[None, :, :], axis=2
            )
            assignments = np.argmin(distances, axis=1)
            for cluster_id in range(n_components):
                cluster_points = participant_set[assignments == cluster_id]
                if len(cluster_points):
                    participant_cluster_means[cluster_id].append(
                        np.mean(cluster_points, axis=0)
                    )
        predicted = []
        for cluster_id in range(n_components):
            values = participant_cluster_means[cluster_id]
            if not values:
                raise ValueError(
                    f"No training participant represents component {cluster_id + 1} "
                    f"of PairID {pair_id}."
                )
            predicted.append(np.mean(np.asarray(values, dtype=float), axis=0))
        population_map[pair_id] = as_component_set(predicted)
    return population_map


def deterministic_component_matching(
    observed: object, population: object
) -> np.ndarray:
    """Match components without using target-pair evaluation information."""
    observed = as_component_set(observed)
    population = as_component_set(population)
    if len(observed) == 0 or len(population) == 0:
        return np.empty(0, dtype=int)
    distances = np.linalg.norm(observed[:, None, :] - population[None, :, :], axis=2)
    if len(observed) == 2 and len(population) == 2:
        direct = float(distances[0, 0] + distances[1, 1])
        swapped = float(distances[0, 1] + distances[1, 0])
        return np.asarray([0, 1] if direct <= swapped else [1, 0], dtype=int)
    return np.argmin(distances, axis=1)


def pair_residual(
    tactile_data: ComponentDataset,
    participant: int,
    pair_id: int,
    population_map: dict[int, np.ndarray],
) -> np.ndarray | None:
    observed = tactile_data.get_set(participant, pair_id)
    population = population_map[pair_id]
    if len(observed) == 0 or len(population) == 0:
        return None
    matched = deterministic_component_matching(observed, population)
    return np.mean(observed - population[matched], axis=0)


def fit_global_offset(
    tactile_data: ComponentDataset,
    participant: int,
    population_map: dict[int, np.ndarray],
    calibration_pairs: tuple[int, ...],
) -> dict[str, object]:
    """Fit the M0 equal-pair-weight global offset with fixed n0=4."""
    residuals = []
    valid_pairs = []
    missing_pairs = []
    for pair_id in calibration_pairs:
        residual = pair_residual(tactile_data, participant, pair_id, population_map)
        if residual is None:
            missing_pairs.append(int(pair_id))
        else:
            valid_pairs.append(int(pair_id))
            residuals.append(residual)
    k_valid = len(valid_pairs)
    if k_valid:
        center = np.mean(np.asarray(residuals, dtype=float), axis=0)
        shrink_factor = k_valid / (k_valid + N0)
        translation = center * shrink_factor
    else:
        center = np.zeros(2, dtype=float)
        shrink_factor = 0.0
        translation = np.zeros(2, dtype=float)
    return {
        "translation": translation,
        "unshrunk_center": center,
        "shrink_factor": float(shrink_factor),
        "k_attempted": len(calibration_pairs),
        "k_valid": k_valid,
        "valid_pairs": tuple(valid_pairs),
        "missing_pairs": tuple(missing_pairs),
    }


def evaluate_participant(
    tactile_data: ComponentDataset,
    participant: int,
    population_map: dict[int, np.ndarray],
    calibration_pairs: tuple[int, ...],
    midpoints: dict[int, np.ndarray] | None = None,
    include_rows: bool = False,
) -> dict[str, object]:
    """Evaluate a calibration set on all observed non-calibration pairs."""
    fit = fit_global_offset(
        tactile_data, participant, population_map, calibration_pairs
    )
    attempted = set(int(value) for value in calibration_pairs)
    personalized_errors = []
    population_errors = []
    midpoint_errors = []
    rows = []
    for pair_id in PAIR_IDS:
        if pair_id in attempted:
            continue
        observed = tactile_data.get_set(participant, pair_id)
        if len(observed) == 0:
            continue
        population = population_map[pair_id]
        personalized = population + np.asarray(fit["translation"], dtype=float)
        population_error = directed_prediction_error(observed, population)
        personalized_error = directed_prediction_error(observed, personalized)
        population_errors.append(population_error)
        personalized_errors.append(personalized_error)
        midpoint_error = float("nan")
        if midpoints is not None:
            midpoint_error = directed_prediction_error(observed, midpoints[pair_id])
            midpoint_errors.append(midpoint_error)
        if include_rows:
            rows.append(
                {
                    "participant": int(participant),
                    "pair_number": int(pair_id),
                    "pair_id": PAIR_LABEL_BY_ID[pair_id],
                    "calibration_pair_ids": " ".join(
                        PAIR_LABEL_BY_ID[value] for value in calibration_pairs
                    ),
                    "k_valid": int(fit["k_valid"]),
                    "midpoint": midpoint_error,
                    "population": population_error,
                    "personalized": personalized_error,
                }
            )
    if not personalized_errors:
        raise ValueError(f"Participant {participant} has no evaluation targets.")
    return {
        "fit": fit,
        "mean_personalized_error_mm": float(np.mean(personalized_errors)),
        "mean_population_error_mm": float(np.mean(population_errors)),
        "mean_midpoint_error_mm": (
            float(np.mean(midpoint_errors)) if midpoint_errors else float("nan")
        ),
        "target_count": len(personalized_errors),
        "rows": rows,
    }


def score_calibration_set(
    tactile_data: ComponentDataset,
    inner_contexts: dict[int, dict[int, np.ndarray]],
    calibration_pairs: tuple[int, ...],
) -> float:
    """Participant-balanced nested-LOPO selection objective."""
    participant_errors = []
    for inner_validation, population_map in inner_contexts.items():
        result = evaluate_participant(
            tactile_data,
            inner_validation,
            population_map,
            calibration_pairs,
        )
        participant_errors.append(result["mean_personalized_error_mm"])
    return float(np.mean(participant_errors))


def greedy_select_calibration_pairs(
    tactile_data: ComponentDataset,
    inner_contexts: dict[int, dict[int, np.ndarray]],
) -> tuple[int, ...]:
    """Choose five pairs using training-only nested LOPO and fixed n0=4."""
    selected: list[int] = []
    while len(selected) < CALIBRATION_BUDGET:
        scored = []
        for candidate in PAIR_IDS:
            if candidate in selected:
                continue
            trial_pairs = tuple(selected + [candidate])
            score = score_calibration_set(tactile_data, inner_contexts, trial_pairs)
            scored.append((score, candidate))
        _score, best_pair = min(scored, key=lambda value: (value[0], value[1]))
        selected.append(int(best_pair))
    return tuple(selected)


def pair_midpoints() -> dict[int, np.ndarray]:
    if len(PAIR_ELECTRODES) != len(PAIR_IDS):
        raise AssertionError("PAIR_ELECTRODES must define PairID 1-28.")
    predictions = {}
    for pair_id, (anode_number, cathode_number) in zip(PAIR_IDS, PAIR_ELECTRODES):
        anode = ELECTRODE_POSITIONS_MM[anode_number - 1]
        cathode = ELECTRODE_POSITIONS_MM[cathode_number - 1]
        predictions[pair_id] = ((anode + cathode) / 2.0).reshape(1, 2)
    return predictions


def run_nested_lopo(tactile_data: ComponentDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 24 strict outer folds and return participant and pair raw data."""
    midpoint_map = pair_midpoints()
    participant_rows = []
    target_rows = []
    for outer_test in PARTICIPANTS:
        outer_training = tuple(
            participant for participant in PARTICIPANTS if participant != outer_test
        )
        if len(outer_training) != 23 or outer_test in outer_training:
            raise AssertionError("Invalid outer-fold training boundary.")
        outer_population = build_population_map(
            tactile_data, outer_training, (outer_test,)
        )

        inner_contexts: dict[int, dict[int, np.ndarray]] = {}
        for inner_validation in outer_training:
            inner_training = tuple(
                participant
                for participant in outer_training
                if participant != inner_validation
            )
            if (
                len(inner_training) != 22
                or outer_test in inner_training
                or inner_validation in inner_training
            ):
                raise AssertionError("Invalid inner-fold training boundary.")
            inner_contexts[inner_validation] = build_population_map(
                tactile_data,
                inner_training,
                (outer_test, inner_validation),
            )

        selected_pairs = greedy_select_calibration_pairs(
            tactile_data, inner_contexts
        )
        evaluation = evaluate_participant(
            tactile_data,
            outer_test,
            outer_population,
            selected_pairs,
            midpoints=midpoint_map,
            include_rows=True,
        )
        fit = evaluation["fit"]
        if len(selected_pairs) != CALIBRATION_BUDGET:
            raise AssertionError("A fold did not select exactly five calibration pairs.")
        participant_rows.append(
            {
                "participant": outer_test,
                "midpoint_mean_error_mm": evaluation["mean_midpoint_error_mm"],
                "data_driven_mean_tactile_map_mean_error_mm": evaluation[
                    "mean_population_error_mm"
                ],
                "nested_m0_global_offset_mean_error_mm": evaluation[
                    "mean_personalized_error_mm"
                ],
            }
        )
        for row in evaluation["rows"]:
            for method in METHOD_ORDER:
                target_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "participant": row["participant"],
                        "method": method,
                        "prediction_error_mm": row[method],
                    }
                )
        labels = " ".join(PAIR_LABEL_BY_ID[pair_id] for pair_id in selected_pairs)
        missing = " ".join(
            PAIR_LABEL_BY_ID[pair_id] for pair_id in fit["missing_pairs"]
        ) or "none"
        print(
            f"Participant {outer_test:02d}: calibration={labels}; "
            f"K_valid={fit['k_valid']}; missing={missing}",
            flush=True,
        )

    participant_data = pd.DataFrame(participant_rows)
    pair_data = pd.DataFrame(target_rows)
    if len(participant_data) != len(PARTICIPANTS):
        raise AssertionError("Participant plot data are incomplete.")
    expected_methods = set(METHOD_ORDER)
    for (_participant, _pair_id), group in pair_data.groupby(
        ["participant", "pair_id"], sort=False
    ):
        if set(group["method"]) != expected_methods:
            raise AssertionError("Methods do not share the same target observations.")
    return participant_data, pair_data


def sem(values: pd.Series) -> float:
    array = np.asarray(pd.Series(values).dropna(), dtype=float)
    return float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0


def save_svg(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_participant_boxplot(data: pd.DataFrame, output_dir: Path) -> None:
    columns = {
        "midpoint": "midpoint_mean_error_mm",
        "population": "data_driven_mean_tactile_map_mean_error_mm",
        "personalized": "nested_m0_global_offset_mean_error_mm",
    }
    values = [data[columns[method]].to_numpy(dtype=float) for method in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    box = ax.boxplot(
        values,
        positions=x,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        showmeans=True,
        medianprops={"color": "#111827", "linewidth": 1.0},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111827",
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
            "markersize": 3.7,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#374151",
            "markeredgewidth": 0.7,
            "markersize": 3.0,
        },
        whiskerprops={"color": "#4B5563", "linewidth": 0.9},
        capprops={"color": "#4B5563", "linewidth": 0.9},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.32)
        patch.set_edgecolor(METHOD_COLORS[method])
        patch.set_linewidth(1.1)
    rng = np.random.default_rng(41)
    for index, method_values in enumerate(values):
        jitter = rng.normal(0.0, 0.045, size=len(method_values))
        ax.scatter(
            np.full(len(method_values), x[index]) + jitter,
            method_values,
            s=22,
            color=METHOD_COLORS[METHOD_ORDER[index]],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.axhline(
        TWO_PD_THRESHOLD_MM,
        color="#8B1E3F",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )
    ax.text(
        -0.45,
        TWO_PD_THRESHOLD_MM + 0.55,
        "2PD = 17.5 mm",
        color="#8B1E3F",
        fontsize=8,
        ha="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Midpoint", "Data-driven\nmean tactile", "Nested M0\nglobal offset"]
    )
    ax.set_ylabel("Prediction error (mm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.18)
    save_svg(
        fig,
        output_dir / "final_tactile_global_offset_participant_boxplot.svg",
    )


def plot_pair_error(pair_data: pd.DataFrame, output_dir: Path) -> None:
    pair_order = {label: index for index, label in enumerate(PAIR_LABELS, start=1)}
    summary = (
        pair_data.groupby(["pair_id", "method"], as_index=False)
        .agg(
            mean_error_mm=("prediction_error_mm", "mean"),
            sem_error_mm=("prediction_error_mm", sem),
        )
    )
    summary["pair_number"] = summary["pair_id"].map(pair_order)
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    for method in METHOD_ORDER:
        rows = summary.loc[summary["method"] == method].sort_values("pair_number")
        ax.errorbar(
            rows["pair_number"],
            rows["mean_error_mm"],
            yerr=rows["sem_error_mm"],
            marker="o",
            markersize=4.2 if method != "midpoint" else 3.8,
            linewidth=2.1 if method != "midpoint" else 1.5,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            color=METHOD_COLORS[method],
            alpha=0.95 if method != "midpoint" else 0.72,
            label=METHOD_LABELS[method],
            zorder=4 if method != "midpoint" else 3,
        )
    ax.axhline(
        TWO_PD_THRESHOLD_MM,
        color="#8B1E3F",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )
    ax.text(0.55, TWO_PD_THRESHOLD_MM + 0.7, "2PD = 17.5 mm", color="#8B1E3F", fontsize=8)
    ax.set_xlim(0.5, 28.5)
    ax.set_xticks(PAIR_IDS)
    ax.set_xticklabels(PAIR_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Electrode pair")
    ax.set_ylabel("Prediction error (mm)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.24)
    save_svg(
        fig,
        output_dir / "final_tactile_global_offset_pair_error_by_pair.svg",
    )


def write_outputs(
    participant_data: pd.DataFrame,
    pair_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_path = (
        output_dir / "final_tactile_global_offset_participant_boxplot_raw_data.csv"
    )
    pair_path = output_dir / "final_tactile_global_offset_pair_error_by_pair_raw_data.csv"
    participant_data.to_csv(participant_path, index=False, float_format="%.9f")
    pair_data.to_csv(pair_path, index=False, float_format="%.9f")
    plot_participant_boxplot(participant_data, output_dir)
    plot_pair_error(pair_data, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing p1.csv through p24.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the two SVG figures and their raw CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tactile_data = load_tactile_data(args.input_dir)
    participant_data, pair_data = run_nested_lopo(tactile_data)
    write_outputs(participant_data, pair_data, args.output_dir)
    print("\nParticipant-level mean prediction error (mm):")
    print(
        participant_data.drop(columns="participant").mean().round(6).to_string()
    )
    print(f"\nWrote tactile-prediction outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
