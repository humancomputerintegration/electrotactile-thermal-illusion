"""Public, end-to-end LOPO thermal prediction from the released raw CSV files.

Place this script beside ``p1.csv`` through ``p24.csv`` and run it directly.
It creates the two Analysis figures without running inferential statistics.

Methods
-------
1. Accepted ``ET`` responses form each participant's tactile component set.
   Accepted ``ET+Thermal`` responses form the corresponding thermal set.
   Points are used directly; every drawn polygon is represented by its polygon
   centroid.  ``no_sensation`` and rejected attempts contain no accepted
   component set and are therefore treated as unavailable observations.
2. Each outer leave-one-participant-out (LOPO) fold refits all population
   thermal component centers and all alignment quantities from the other 23
   participants.  The one-/two-component pair structure is fixed before LOPO.
3. Five valid thermal calibration pairs are revealed sequentially.  If the user
   reports no thermal sensation during calibration, that pair is skipped and
   selection continues until five valid thermal responses have been obtained.
   At each step, the pair with the largest posterior predictive variance is
   selected using only the training library and held-out thermal-response
   availability.  Each valid revealed thermal response updates participant-
   similarity weights through a Gaussian likelihood.
4. A training-only pair-bias adjustment is applied to the five revealed
   thermal-to-tactile distances.  Fold-specific three-group thresholds classify
   the held-out participant as strong, intermediate, or weak alignment.
5. Unrevealed thermal responses are predicted by three methods named exactly:
   ``midpoint``, ``thermal Bayesian``, and ``tactile-guided Bayesian``.
   Strong alignment uses the personal tactile set, weak alignment uses the
   thermal Bayesian set, and intermediate alignment blends the two using a
   training-only reliability weight.

The prediction error is the mean distance from every
observed thermal component to its nearest predicted component.  Revealed pairs
never enter evaluation.

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figure_outputs" / "thermal_prediction"

EXPECTED_PARTICIPANTS = tuple(range(1, 25))
EXPECTED_PAIR_IDS = tuple(range(1, 29))
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
PAIR_LABEL_BY_ID = dict(zip(EXPECTED_PAIR_IDS, PAIR_LABELS))

# (anode electrode, cathode electrode) for PairID 1-28.  Keeping this small
# experimental-layout table here makes the released script independent of
# analysis files outside Prediction_code.
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

# Fixed before the outer folds; this is not re-selected from a held-out person.
TWO_COMPONENT_PAIR_IDS = frozenset({13, 14, 17, 18, 19, 21, 22, 24, 25, 27, 28})
CALIBRATION_BUDGET = 5
THERMAL_SIGMA_MM = 25.0
TACTILE_SIGMA_MM = 25.0
MAX_TACTILE_WEIGHT = 0.50
GROUP_ORDER = ("strong", "intermediate", "weak")
METHOD_ORDER = ("midpoint", "thermal_bayesian", "tactile_guided_bayesian")
METHOD_LABELS = {
    "midpoint": "midpoint",
    "thermal_bayesian": "thermal Bayesian",
    "tactile_guided_bayesian": "tactile-guided Bayesian",
}
METHOD_COLORS = {
    "midpoint": "#6B7280",
    "thermal_bayesian": "#1B9E77",
    "tactile_guided_bayesian": "#F58518",
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


def symmetric_set_distance(first: object, second: object) -> float:
    """Symmetric mean nearest-neighbour distance between two component sets."""
    first = as_component_set(first)
    second = as_component_set(second)
    if len(first) == 0 or len(second) == 0:
        return float("nan")
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        0.5
        * (
            np.mean(np.min(distances, axis=1))
            + np.mean(np.min(distances, axis=0))
        )
    )


def directed_coverage_error(observed: object, predicted: object) -> float:
    """Mean observed-component distance to its nearest predicted component."""
    observed = as_component_set(observed)
    predicted = as_component_set(predicted)
    if len(observed) == 0 or len(predicted) == 0:
        return float("nan")
    distances = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def normalized_weights(weights: object) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be a nonempty finite vector.")
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Weights cannot be normalized.")
    return weights / total


def parse_json_cell(value: object, name: str) -> list:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must contain a JSON list.")
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
    """Apply the published point/polygon-centroid representation to one row."""
    zones = parse_json_cell(row["ZonePoints_mm"], f"{source} ZonePoints_mm")
    response_items = parse_json_cell(row["ResponseItems"], f"{source} ResponseItems")
    if int(row["NumZones"]) != len(zones):
        raise ValueError(f"{source}: NumZones does not match ZonePoints_mm.")
    if int(row["NumResponseItems"]) != len(response_items):
        raise ValueError(f"{source}: NumResponseItems does not match ResponseItems.")

    components: list[tuple[float, float]] = []
    for zone in zones:
        centroid = polygon_centroid(zone)
        if centroid is not None:
            components.append(centroid)

    fallback_areas: list[tuple[float, float]] = []
    for item in response_items:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: every ResponseItems entry must be an object.")
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
            raise ValueError(f"{source}: unsupported ResponseItems type {item_type!r}.")

    if not components:
        components.extend(fallback_areas)
    return as_component_set(components)


class ComponentDataset:
    """Participant-by-pair measured component sets."""

    def __init__(
        self,
        sets: dict[tuple[int, int], np.ndarray],
        participants: tuple[int, ...],
    ) -> None:
        self._sets = {
            (int(participant), int(pair_id)): as_component_set(points)
            for (participant, pair_id), points in sets.items()
            if len(as_component_set(points))
        }
        self._participants = tuple(int(value) for value in participants)

    @property
    def participants(self) -> list[int]:
        return list(self._participants)

    def pair_ids(self, participant: int) -> set[int]:
        participant = int(participant)
        return {
            pair_id
            for candidate, pair_id in self._sets
            if candidate == participant
        }

    def get_set(self, participant: int, pair_id: int) -> np.ndarray:
        points = self._sets.get((int(participant), int(pair_id)))
        if points is None:
            return np.empty((0, 2), dtype=float)
        return points.copy()


def participant_number(path: Path) -> int | None:
    match = re.fullmatch(r"p(\d+)\.csv", path.name.lower())
    return int(match.group(1)) if match else None


def load_raw_component_datasets(
    input_dir: Path,
) -> tuple[ComponentDataset, ComponentDataset, pd.DataFrame]:
    input_dir = Path(input_dir)
    files = sorted(
        [path for path in input_dir.glob("p*.csv") if participant_number(path) is not None],
        key=lambda path: participant_number(path),
    )
    found = tuple(participant_number(path) for path in files)
    if found != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected p1.csv through p24.csv in {input_dir}; found participant IDs {found}."
        )

    tactile_sets: dict[tuple[int, int], np.ndarray] = {}
    thermal_sets: dict[tuple[int, int], np.ndarray] = {}
    audit_rows = []
    for path in files:
        participant = int(participant_number(path))
        rows = pd.read_csv(path, keep_default_na=False)
        if tuple(rows.columns) != EXPECTED_RAW_COLUMNS:
            raise ValueError(
                f"{path.name} does not have the expected 14-column public schema."
            )
        if rows.empty:
            raise ValueError(f"{path.name} is empty.")
        numeric_phase = pd.to_numeric(rows["PhaseNum"], errors="raise")
        numeric_pair = pd.to_numeric(rows["PairID"], errors="raise")
        if (numeric_phase >= 57).any():
            raise ValueError(f"{path.name} still contains PhaseNum >= 57.")
        if not numeric_pair.between(1, 28).all():
            raise ValueError(f"{path.name} contains a PairID outside 1-28.")

        accepted = rows.loc[
            rows["ConfirmationStatus"].astype(str).str.strip().str.lower().eq("accepted_yes")
        ]
        for row_index, row in accepted.iterrows():
            condition = str(row["Condition"]).strip()
            if condition == "ET":
                modality = "tactile"
                destination = tactile_sets
            elif condition in {"ET+Thermal", "ET+Heat"}:
                modality = "thermal"
                destination = thermal_sets
            else:
                raise ValueError(f"{path.name}: unsupported Condition {condition!r}.")
            pair_id = int(row["PairID"])
            source = f"{path.name} row {row_index + 2}"
            components = extract_components(row, source)
            if len(components) == 0:
                raise ValueError(f"{source}: accepted response contains no components.")
            key = (participant, pair_id)
            if key in destination:
                raise ValueError(
                    f"{path.name}: duplicate accepted {modality} response for PairID {pair_id}."
                )
            destination[key] = components
            audit_rows.append(
                {
                    "participant": participant,
                    "pair_id": pair_id,
                    "pair_label": PAIR_LABEL_BY_ID[pair_id],
                    "modality": modality,
                    "component_count": len(components),
                }
            )

    tactile = ComponentDataset(tactile_sets, EXPECTED_PARTICIPANTS)
    thermal = ComponentDataset(thermal_sets, EXPECTED_PARTICIPANTS)
    return tactile, thermal, pd.DataFrame(audit_rows)


def load_pair_metadata() -> pd.DataFrame:
    if len(PAIR_ELECTRODES) != len(EXPECTED_PAIR_IDS):
        raise AssertionError("PAIR_ELECTRODES must define PairID 1-28 exactly once.")
    rows = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "anode": electrodes[0],
                "cathode": electrodes[1],
                "pair_label": PAIR_LABEL_BY_ID[pair_id],
            }
            for pair_id, electrodes in zip(EXPECTED_PAIR_IDS, PAIR_ELECTRODES)
        ]
    )
    if not rows["anode"].between(1, 36).all() or not rows["cathode"].between(1, 36).all():
        raise ValueError("Electrode numbers must be between 1 and 36.")
    return rows


def pair_midpoints(pair_metadata: pd.DataFrame) -> dict[int, np.ndarray]:
    midpoints = {}
    for row in pair_metadata.itertuples(index=False):
        anode = ELECTRODE_POSITIONS_MM[int(row.anode) - 1]
        cathode = ELECTRODE_POSITIONS_MM[int(row.cathode) - 1]
        midpoints[int(row.pair_id)] = ((anode + cathode) / 2.0).reshape(1, 2)
    return midpoints


def deterministic_kmeans(points: object, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    points = as_component_set(points)
    if len(points) < n_clusters:
        raise ValueError("Fewer training points than requested clusters.")
    if n_clusters == 1:
        return np.mean(points, axis=0, keepdims=True)
    if n_clusters != 2:
        raise ValueError("Only one or two fixed components are supported.")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.asarray([points[first], points[second]], dtype=float)
    for _ in range(max_iter):
        center_distances = np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2
        )
        assignments = np.argmin(center_distances, axis=1)
        updated = centers.copy()
        for cluster_id in range(n_clusters):
            cluster_points = points[assignments == cluster_id]
            if len(cluster_points):
                updated[cluster_id] = np.mean(cluster_points, axis=0)
            else:
                nearest = np.min(center_distances, axis=1)
                updated[cluster_id] = points[int(np.argmax(nearest))]
        if np.allclose(updated, centers, atol=1e-9, rtol=0.0):
            centers = updated
            break
        centers = updated
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


class FoldThermalLibrary:
    """Training-only thermal component library for one outer fold."""

    def __init__(
        self,
        thermal_data: ComponentDataset,
        train_participants: list[int],
        pair_ids: tuple[int, ...],
        test_participant: int,
    ) -> None:
        self.train_participants = tuple(int(value) for value in train_participants)
        self.pair_ids = tuple(int(value) for value in pair_ids)
        if int(test_participant) in self.train_participants:
            raise AssertionError("The held-out participant entered the training library.")
        self.cluster_maps: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        for pair_id in self.pair_ids:
            n_clusters = 2 if pair_id in TWO_COMPONENT_PAIR_IDS else 1
            blocks = [
                thermal_data.get_set(participant, pair_id)
                for participant in self.train_participants
                if len(thermal_data.get_set(participant, pair_id))
            ]
            if not blocks:
                raise ValueError(f"No training thermal components for PairID {pair_id}.")
            centers = deterministic_kmeans(np.vstack(blocks), n_clusters)
            for participant in self.train_participants:
                participant_set = thermal_data.get_set(participant, pair_id)
                if len(participant_set) == 0:
                    self.cluster_maps[(participant, pair_id)] = {}
                    continue
                distances = np.linalg.norm(
                    participant_set[:, None, :] - centers[None, :, :], axis=2
                )
                assignments = np.argmin(distances, axis=1)
                cluster_map = {}
                for cluster_id in range(n_clusters):
                    assigned = participant_set[assignments == cluster_id]
                    if len(assigned):
                        cluster_map[cluster_id] = np.mean(assigned, axis=0)
                self.cluster_maps[(participant, pair_id)] = cluster_map

    def get_participant_set(self, participant: int, pair_id: int) -> np.ndarray:
        cluster_map = self.cluster_maps.get((int(participant), int(pair_id)), {})
        if not cluster_map:
            return np.empty((0, 2), dtype=float)
        return as_component_set(
            np.asarray([cluster_map[key] for key in sorted(cluster_map)], dtype=float)
        )

    def predict(self, weights: object, pair_id: int) -> np.ndarray:
        weights = normalized_weights(weights)
        pair_id = int(pair_id)
        n_clusters = 2 if pair_id in TWO_COMPONENT_PAIR_IDS else 1
        predicted = []
        for cluster_id in range(n_clusters):
            numerator = np.zeros(2, dtype=float)
            denominator = 0.0
            for index, participant in enumerate(self.train_participants):
                point = self.cluster_maps.get((participant, pair_id), {}).get(cluster_id)
                if point is None:
                    continue
                numerator += float(weights[index]) * point
                denominator += float(weights[index])
            if denominator <= 0.0:
                raise ValueError(
                    f"No posterior-weighted component for PairID {pair_id}, cluster {cluster_id + 1}."
                )
            predicted.append(numerator / denominator)
        return as_component_set(predicted)

    def predictive_variance(self, weights: object, pair_id: int) -> float:
        weights = normalized_weights(weights)
        prediction = self.predict(weights, pair_id)
        numerator = 0.0
        denominator = 0.0
        for index, participant in enumerate(self.train_participants):
            participant_set = self.get_participant_set(participant, pair_id)
            distance = symmetric_set_distance(participant_set, prediction)
            if not np.isfinite(distance):
                continue
            numerator += float(weights[index]) * distance**2
            denominator += float(weights[index])
        return float(numerator / denominator) if denominator > 0.0 else float("-inf")


def interval_sse(
    prefix_sum: np.ndarray,
    prefix_square_sum: np.ndarray,
    start: int,
    stop: int,
) -> float:
    count = stop - start
    total = prefix_sum[stop] - prefix_sum[start]
    square_total = prefix_square_sum[stop] - prefix_square_sum[start]
    return float(square_total - total * total / count)


def exact_one_dimensional_kmeans(values: object, n_clusters: int = 3) -> np.ndarray:
    """Return globally optimal ordered 1-D k-means centers by dynamic programming."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Alignment values must be a finite vector.")
    if not 1 <= n_clusters <= len(values):
        raise ValueError("Invalid one-dimensional cluster count.")
    sorted_values = np.sort(values, kind="stable")
    n_values = len(sorted_values)
    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_values)))
    prefix_square_sum = np.concatenate(([0.0], np.cumsum(sorted_values**2)))
    objective = np.full((n_clusters + 1, n_values + 1), np.inf)
    previous = np.full((n_clusters + 1, n_values + 1), -1, dtype=int)
    objective[0, 0] = 0.0
    for cluster_count in range(1, n_clusters + 1):
        for stop in range(cluster_count, n_values + 1):
            for start in range(cluster_count - 1, stop):
                candidate = objective[cluster_count - 1, start] + interval_sse(
                    prefix_sum, prefix_square_sum, start, stop
                )
                if candidate < objective[cluster_count, stop]:
                    objective[cluster_count, stop] = candidate
                    previous[cluster_count, stop] = start
    segments = []
    stop = n_values
    for cluster_count in range(n_clusters, 0, -1):
        start = int(previous[cluster_count, stop])
        if start < 0:
            raise RuntimeError("One-dimensional k-means failed.")
        segments.append((start, stop))
        stop = start
    segments.reverse()
    return np.asarray(
        [float(np.mean(sorted_values[start:stop])) for start, stop in segments],
        dtype=float,
    )


def alignment_group(distance: float, d1: float, d2: float) -> str:
    if not np.isfinite(distance):
        raise ValueError("Alignment classification requires a finite distance.")
    if distance < d1:
        return "strong"
    if distance < d2:
        return "intermediate"
    return "weak"


def fit_fold_alignment_groups(
    test_participant: int,
    train_participants: list[int],
    pair_ids: tuple[int, ...],
    thermal_data: ComponentDataset,
    tactile_data: ComponentDataset,
) -> tuple[float, float, dict[int, str], dict[int, float]]:
    full_alignment = {}
    for participant in train_participants:
        distances = [
            symmetric_set_distance(
                thermal_data.get_set(participant, pair_id),
                tactile_data.get_set(participant, pair_id),
            )
            for pair_id in pair_ids
        ]
        distances = [value for value in distances if np.isfinite(value)]
        if not distances:
            raise ValueError(f"Training participant {participant} has no alignment pairs.")
        full_alignment[int(participant)] = float(np.mean(distances))
    centers = exact_one_dimensional_kmeans(list(full_alignment.values()), n_clusters=3)
    d1 = float(0.5 * (centers[0] + centers[1]))
    d2 = float(0.5 * (centers[1] + centers[2]))
    if not d1 < d2:
        raise AssertionError(f"Fold {test_participant} has invalid alignment thresholds.")
    groups = {
        participant: alignment_group(distance, d1, d2)
        for participant, distance in full_alignment.items()
    }
    return d1, d2, groups, full_alignment


def estimate_pair_biases(
    test_participant: int,
    train_participants: list[int],
    pair_ids: tuple[int, ...],
    thermal_data: ComponentDataset,
    tactile_data: ComponentDataset,
    full_alignment: dict[int, float],
) -> dict[int, float]:
    if int(test_participant) in full_alignment:
        raise AssertionError("Held-out participant entered pair-bias estimation.")
    biases = {}
    for pair_id in pair_ids:
        residuals = []
        for participant in train_participants:
            distance = symmetric_set_distance(
                thermal_data.get_set(participant, pair_id),
                tactile_data.get_set(participant, pair_id),
            )
            if np.isfinite(distance):
                residuals.append(float(distance - full_alignment[participant]))
        if not residuals:
            raise ValueError(f"Fold {test_participant}: no pair-bias data for PairID {pair_id}.")
        biases[pair_id] = float(np.mean(residuals))
    return biases


def compute_intermediate_reliability(
    test_participant: int,
    train_participants: list[int],
    training_groups: dict[int, str],
    pair_ids: tuple[int, ...],
    thermal_data: ComponentDataset,
    tactile_data: ComponentDataset,
) -> dict[int, tuple[float, float]]:
    intermediate = [
        participant
        for participant in train_participants
        if training_groups[participant] == "intermediate"
    ]
    if int(test_participant) in intermediate:
        raise AssertionError("Held-out participant entered reliability estimation.")
    result = {}
    for pair_id in pair_ids:
        distances = [
            symmetric_set_distance(
                thermal_data.get_set(participant, pair_id),
                tactile_data.get_set(participant, pair_id),
            )
            for participant in intermediate
        ]
        distances = [value for value in distances if np.isfinite(value)]
        reliability = float(np.mean(distances)) if distances else float("nan")
        tactile_weight = (
            float(
                MAX_TACTILE_WEIGHT
                * np.exp(-0.5 * (reliability / TACTILE_SIGMA_MM) ** 2)
            )
            if np.isfinite(reliability)
            else float("nan")
        )
        result[pair_id] = reliability, tactile_weight
    return result


def select_thermal_calibration_pairs(
    test_participant: int,
    fold_library: FoldThermalLibrary,
    thermal_data: ComponentDataset,
    pair_ids: tuple[int, ...],
) -> tuple[list[int], np.ndarray, dict[int, np.ndarray], list[dict]]:
    train_participants = fold_library.train_participants
    weights = np.full(len(train_participants), 1.0 / len(train_participants), dtype=float)
    available = sorted(set(pair_ids) & thermal_data.pair_ids(test_participant))
    if len(available) < CALIBRATION_BUDGET:
        raise ValueError(f"Participant {test_participant} has fewer than five thermal responses.")
    selected = []
    observed = {}
    audit_rows = []
    for step in range(1, CALIBRATION_BUDGET + 1):
        candidates = [pair_id for pair_id in available if pair_id not in selected]
        scored = [
            (fold_library.predictive_variance(weights, pair_id), pair_id)
            for pair_id in candidates
        ]
        scored = [(variance, pair_id) for variance, pair_id in scored if np.isfinite(variance)]
        if not scored:
            raise ValueError(f"Participant {test_participant}: no valid calibration candidate.")
        best_variance, selected_pair = max(scored, key=lambda item: (item[0], -item[1]))
        selected.append(int(selected_pair))
        observed_set = thermal_data.get_set(test_participant, selected_pair)
        if len(observed_set) == 0:
            raise AssertionError("A selected pair has no revealable thermal response.")
        observed[selected_pair] = observed_set

        prior = normalized_weights(weights.copy())
        likelihoods = np.zeros(len(train_participants), dtype=float)
        for index, training_participant in enumerate(train_participants):
            training_set = fold_library.get_participant_set(
                training_participant, selected_pair
            )
            distance = symmetric_set_distance(observed_set, training_set)
            if np.isfinite(distance):
                likelihoods[index] = float(
                    np.exp(-0.5 * (distance / THERMAL_SIGMA_MM) ** 2)
                )
        unnormalized = prior * likelihoods
        normalizer = float(np.sum(unnormalized))
        if normalizer <= np.finfo(float).tiny or not np.isfinite(normalizer):
            raise FloatingPointError(
                f"Participant {test_participant}, step {step}: Bayesian normalization failed."
            )
        weights = unnormalized / normalizer
        if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=0.0):
            raise AssertionError("Posterior weights do not sum to one.")
        audit_rows.append(
            {
                "participant": test_participant,
                "step": step,
                "pair_id": selected_pair,
                "pair_label": PAIR_LABEL_BY_ID[selected_pair],
                "posterior_predictive_variance_mm2": float(best_variance),
            }
        )
    if len(selected) != CALIBRATION_BUDGET or len(set(selected)) != CALIBRATION_BUDGET:
        raise AssertionError("Calibration did not produce five unique pairs.")
    return selected, weights, observed, audit_rows


def deterministic_tactile_blend(
    bayesian_set: object,
    tactile_set: object,
    tactile_weight: float,
) -> np.ndarray:
    bayesian = as_component_set(bayesian_set)
    tactile = as_component_set(tactile_set)
    if len(bayesian) == 0 or len(tactile) == 0:
        return np.empty((0, 2), dtype=float)
    if not 0.0 <= tactile_weight <= 1.0:
        raise ValueError("Tactile blend weight must lie in [0, 1].")
    if len(bayesian) == 2 and len(tactile) == 2:
        identity = float(np.linalg.norm(bayesian[0] - tactile[0]) + np.linalg.norm(bayesian[1] - tactile[1]))
        swapped = float(np.linalg.norm(bayesian[0] - tactile[1]) + np.linalg.norm(bayesian[1] - tactile[0]))
        mapping = np.asarray([0, 1] if identity <= swapped else [1, 0], dtype=int)
    elif len(bayesian) == 1 and len(tactile) == 1:
        mapping = np.asarray([0], dtype=int)
    else:
        distances = np.linalg.norm(bayesian[:, None, :] - tactile[None, :, :], axis=2)
        mapping = np.argmin(distances, axis=1)
    matched = tactile[mapping]
    return as_component_set((1.0 - tactile_weight) * bayesian + tactile_weight * matched)


def tactile_guided_prediction(
    online_group: str,
    bayesian_set: object,
    tactile_set: object,
    tactile_weight: float,
) -> np.ndarray:
    if online_group == "strong":
        return as_component_set(tactile_set)
    if online_group == "weak":
        return as_component_set(bayesian_set)
    if online_group != "intermediate":
        raise ValueError(f"Unexpected online group {online_group!r}.")
    if not np.isfinite(tactile_weight):
        raise ValueError("Intermediate routing requires a finite tactile weight.")
    return deterministic_tactile_blend(bayesian_set, tactile_set, tactile_weight)


def run_lopo(
    tactile_data: ComponentDataset,
    thermal_data: ComponentDataset,
    pair_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = tactile_data.participants
    if participants != list(EXPECTED_PARTICIPANTS) or thermal_data.participants != participants:
        raise ValueError("LOPO requires participants 1-24.")
    pair_ids = EXPECTED_PAIR_IDS
    midpoints = pair_midpoints(pair_metadata)
    error_rows = []
    group_rows = []
    selection_rows = []

    for test_participant in participants:
        train_participants = [
            participant for participant in participants if participant != test_participant
        ]
        if len(train_participants) != 23:
            raise AssertionError("Every outer fold must contain 23 training participants.")
        d1, d2, training_groups, full_alignment = fit_fold_alignment_groups(
            test_participant,
            train_participants,
            pair_ids,
            thermal_data,
            tactile_data,
        )
        pair_bias = estimate_pair_biases(
            test_participant,
            train_participants,
            pair_ids,
            thermal_data,
            tactile_data,
            full_alignment,
        )
        reliability = compute_intermediate_reliability(
            test_participant,
            train_participants,
            training_groups,
            pair_ids,
            thermal_data,
            tactile_data,
        )
        fold_library = FoldThermalLibrary(
            thermal_data, train_participants, pair_ids, test_participant
        )
        selected, posterior, observed_selected, fold_selection = (
            select_thermal_calibration_pairs(
                test_participant,
                fold_library,
                thermal_data,
                pair_ids,
            )
        )
        selection_rows.extend(fold_selection)

        raw_distances = []
        adjusted_distances = []
        for pair_id in selected:
            tactile_set = tactile_data.get_set(test_participant, pair_id)
            distance = symmetric_set_distance(observed_selected[pair_id], tactile_set)
            if np.isfinite(distance):
                raw_distances.append(float(distance))
                adjusted_distances.append(float(distance - pair_bias[pair_id]))
        if not raw_distances:
            raise ValueError(
                f"Participant {test_participant} has no valid tactile alignment among selected pairs."
            )
        raw_group = alignment_group(float(np.mean(raw_distances)), d1, d2)
        adjusted_distance = float(np.mean(adjusted_distances))
        online_group = alignment_group(adjusted_distance, d1, d2)
        group_rows.append(
            {
                "participant": test_participant,
                "online_group": online_group,
                "raw_online_group": raw_group,
                "adjusted_alignment_distance_mm": adjusted_distance,
                "fold_D1_mm": d1,
                "fold_D2_mm": d2,
                "n_valid_alignment_pairs": len(adjusted_distances),
                "selected_pairs": " ".join(PAIR_LABEL_BY_ID[value] for value in selected),
            }
        )

        selected_set = set(selected)
        for pair_id in pair_ids:
            if pair_id in selected_set:
                continue
            midpoint_prediction = midpoints[pair_id]
            bayesian_prediction = fold_library.predict(posterior, pair_id)
            tactile_set = tactile_data.get_set(test_participant, pair_id)
            _reliability_mm, tactile_weight = reliability[pair_id]
            guided_prediction = tactile_guided_prediction(
                online_group, bayesian_prediction, tactile_set, tactile_weight
            )
            either_route_requires_tactile = (
                raw_group in {"strong", "intermediate"}
                or online_group in {"strong", "intermediate"}
            )

            # Target thermal truth is deliberately accessed only after all three
            # predictions and routing decisions above are complete.
            observed_thermal = thermal_data.get_set(test_participant, pair_id)
            included = bool(
                len(observed_thermal)
                and (len(tactile_set) or not either_route_requires_tactile)
                and len(guided_prediction)
            )
            if not included:
                continue
            predictions = {
                "midpoint": midpoint_prediction,
                "thermal_bayesian": bayesian_prediction,
                "tactile_guided_bayesian": guided_prediction,
            }
            for method in METHOD_ORDER:
                error_rows.append(
                    {
                        "participant": test_participant,
                        "pair_number": pair_id,
                        "pair_id": PAIR_LABEL_BY_ID[pair_id],
                        "online_group": online_group,
                        "method": METHOD_LABELS[method],
                        "directed_coverage_error_mm": directed_coverage_error(
                            observed_thermal, predictions[method]
                        ),
                    }
                )

    errors = pd.DataFrame(error_rows)
    groups = pd.DataFrame(group_rows).sort_values("participant").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        ["participant", "step"]
    ).reset_index(drop=True)
    if len(selections) != len(participants) * CALIBRATION_BUDGET:
        raise AssertionError("Expected 120 calibration selections.")
    if errors.empty:
        raise ValueError("No unrevealed predictions were eligible for evaluation.")
    counts = errors.groupby(["participant", "method"])["pair_number"].nunique().unstack()
    if counts.isna().any().any() or not counts.nunique(axis=1).eq(1).all():
        raise AssertionError("Methods were not evaluated on identical target pairs.")
    selected_lookup = selections.groupby("participant")["pair_id"].apply(set).to_dict()
    if any(
        row.pair_number in selected_lookup[int(row.participant)]
        for row in errors.itertuples(index=False)
    ):
        raise AssertionError("A revealed calibration pair entered evaluation.")
    return errors, groups, selections


def sem(values: object) -> float:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1) / math.sqrt(len(values)))


def build_pair_plot_data(errors: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        errors.groupby(["pair_number", "pair_id", "method"], as_index=False)
        .agg(
            participant_count=("participant", "nunique"),
            mean_error_mm=("directed_coverage_error_mm", "mean"),
            sem_error_mm=("directed_coverage_error_mm", sem),
        )
    )
    rows = []
    for pair_number, pair_label in PAIR_LABEL_BY_ID.items():
        pair_rows = grouped.loc[grouped["pair_number"] == pair_number].set_index("method")
        output = {"pair_id": pair_label, "participant_count": 0}
        if not pair_rows.empty:
            counts = pair_rows["participant_count"].astype(int)
            if counts.nunique() != 1 or set(pair_rows.index) != set(METHOD_LABELS.values()):
                raise AssertionError(f"Pair {pair_label} has unequal method coverage.")
            output["participant_count"] = int(counts.iloc[0])
        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            prefix = method
            output[f"{prefix}_mean_error_mm"] = (
                float(pair_rows.loc[label, "mean_error_mm"])
                if label in pair_rows.index
                else float("nan")
            )
            output[f"{prefix}_sem_error_mm"] = (
                float(pair_rows.loc[label, "sem_error_mm"])
                if label in pair_rows.index
                else float("nan")
            )
        rows.append(output)
    return pd.DataFrame(rows)


def build_participant_plot_data(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = (
        errors.groupby(["participant", "online_group", "method"], as_index=False)
        .agg(
            n_unrevealed_evaluated_pairs=("pair_number", "nunique"),
            participant_mean_error_mm=("directed_coverage_error_mm", "mean"),
        )
    )
    counts = long.groupby("participant")["n_unrevealed_evaluated_pairs"].nunique()
    if not counts.eq(1).all():
        raise AssertionError("Participant method coverage is unequal.")
    wide = long.pivot(
        index=["participant", "online_group", "n_unrevealed_evaluated_pairs"],
        columns="method",
        values="participant_mean_error_mm",
    ).reset_index()
    expected_labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    if any(label not in wide.columns for label in expected_labels):
        raise AssertionError("Participant plot data are missing a method.")
    wide = wide.rename(
        columns={
            METHOD_LABELS["midpoint"]: "midpoint_mean_error_mm",
            METHOD_LABELS["thermal_bayesian"]: "thermal_bayesian_mean_error_mm",
            METHOD_LABELS["tactile_guided_bayesian"]: "tactile_guided_bayesian_mean_error_mm",
        }
    )
    wide = wide[
        [
            "participant",
            "online_group",
            "n_unrevealed_evaluated_pairs",
            "midpoint_mean_error_mm",
            "thermal_bayesian_mean_error_mm",
            "tactile_guided_bayesian_mean_error_mm",
        ]
    ]
    return wide.sort_values("participant").reset_index(drop=True), long


def save_figure(figure: plt.Figure, base_path: Path) -> None:
    figure.savefig(
        base_path.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_pairwise_error(pair_data: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.2, 5.1))
    x = np.arange(1, 29, dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    for offset, method in zip(offsets, METHOD_ORDER):
        mean_values = pair_data[f"{method}_mean_error_mm"].to_numpy(dtype=float)
        sem_values = pair_data[f"{method}_sem_error_mm"].to_numpy(dtype=float)
        finite = np.isfinite(mean_values) & np.isfinite(sem_values)
        axis.errorbar(
            x[finite] + offset,
            mean_values[finite],
            yerr=sem_values[finite],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker="o",
            markersize=3.8,
            linewidth=1.0,
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.94,
        )
    axis.set_xticks(x, PAIR_LABELS, rotation=45, ha="right")
    axis.set_xlim(0.4, 28.6)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Electrode pair")
    axis.set_ylabel("Prediction error (mm)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.65, alpha=0.8)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    save_figure(
        figure,
        output_dir / "fig_pairwise_unrevealed_prediction_error",
    )


def plot_by_online_group(participant_long: pd.DataFrame, output_dir: Path) -> None:
    panels = ("all", *GROUP_ORDER)
    figure, axes = plt.subplots(1, len(panels), figsize=(12.4, 4.2), sharey=True)
    rng = np.random.default_rng(47003)
    for axis, group_name in zip(axes, panels):
        subset = participant_long.copy()
        if group_name != "all":
            subset = subset.loc[subset["online_group"] == group_name]
        participant_count = int(subset["participant"].nunique())
        data = [
            subset.loc[
                subset["method"] == METHOD_LABELS[method],
                "participant_mean_error_mm",
            ].to_numpy(dtype=float)
            for method in METHOD_ORDER
        ]
        if participant_count == 0 or any(len(values) == 0 for values in data):
            axis.set_visible(False)
            continue
        box = axis.boxplot(
            data,
            positions=np.arange(1, 4),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.2},
            whiskerprops={"color": "#4B5563", "linewidth": 0.9},
            capprops={"color": "#4B5563", "linewidth": 0.9},
        )
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_edgecolor(METHOD_COLORS[method])
            patch.set_alpha(0.35)
        for position, method, values in zip(range(1, 4), METHOD_ORDER, data):
            jitter = rng.uniform(-0.10, 0.10, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=16,
                color=METHOD_COLORS[method],
                alpha=0.72,
                edgecolors="none",
                zorder=3,
            )
        axis.set_xticks(
            range(1, 4),
            ("midpoint", "thermal\nBayesian", "tactile-guided\nBayesian"),
        )
        title = "All" if group_name == "all" else group_name.capitalize()
        axis.set_title(f"{title} (n={participant_count})")
        axis.set_ylim(bottom=0.0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.65, alpha=0.8)
    axes[0].set_ylabel("Prediction error (mm)")
    figure.tight_layout()
    save_figure(
        figure,
        output_dir / "fig_midpoint_bayesian_tactile_guided_by_online_group",
    )


def run_analysis(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tactile_data, thermal_data, component_audit = load_raw_component_datasets(input_dir)
    pair_metadata = load_pair_metadata()
    errors, groups, selections = run_lopo(tactile_data, thermal_data, pair_metadata)
    pair_data = build_pair_plot_data(errors)
    participant_data, participant_long = build_participant_plot_data(errors)

    pair_data.to_csv(
        output_dir / "fig_pairwise_unrevealed_prediction_error_raw_data.csv",
        index=False,
        float_format="%.9f",
    )
    participant_data.to_csv(
        output_dir
        / "fig_midpoint_bayesian_tactile_guided_by_online_group_participant_values.csv",
        index=False,
        float_format="%.9f",
    )
    plot_pairwise_error(pair_data, output_dir)
    plot_by_online_group(participant_long, output_dir)

    return {
        "output_dir": output_dir,
        "component_audit": component_audit,
        "errors": errors,
        "groups": groups,
        "selections": selections,
        "pair_data": pair_data,
        "participant_data": participant_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    results = run_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    group_counts = results["groups"]["online_group"].value_counts().reindex(
        GROUP_ORDER, fill_value=0
    )
    print(f"Read 24 participant CSVs from: {args.input_dir}")
    print(f"Wrote two figures and their plot data to: {results['output_dir']}")
    print(
        "Online groups: "
        + ", ".join(f"{group}={int(group_counts[group])}" for group in GROUP_ORDER)
    )
    print(
        f"Evaluated {results['errors']['participant'].nunique()} participants and "
        f"{len(results['errors']) // len(METHOD_ORDER)} participant-pair targets."
    )


if __name__ == "__main__":
    main()
