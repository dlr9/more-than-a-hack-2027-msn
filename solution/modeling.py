"""Train and select the two probabilistic classifiers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .decoder import emission_probabilities, viterbi_decode
from .evidence import labels_from_batch, period_reference, score_sequence


def _imports():
    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import balanced_accuracy_score, log_loss
    except ImportError as error:
        raise RuntimeError("training needs requirements-solution.txt") from error
    return joblib, pd, HistGradientBoostingClassifier, balanced_accuracy_score, log_loss


def load_prepared(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        batch = {name: data[name] for name in data.files}
    x = batch["X"]
    n = len(x)
    if x.ndim != 2 or x.shape[1] != len(batch["feature_names"]) or np.isinf(x).any():
        raise ValueError("invalid prepared feature matrix")
    if len(set(batch["feature_names"].tolist())) != x.shape[1]:
        raise ValueError("duplicate feature names")
    for name in ("frame", "period", "y_alive", "y_home", "nearest_codes"):
        if batch[name].shape != (n,):
            raise ValueError(f"invalid prepared column {name}")
    if not np.array_equal(batch["frame"], np.arange(n)):
        raise ValueError("prepared frames must be contiguous zero-based")
    for name in ("y_alive", "y_home"):
        if not np.isin(batch[name], [-1, 0, 1]).all():
            raise ValueError("invalid prepared labels")
    if not np.isfinite(batch["fps"]) or float(batch["fps"]) <= 0:
        raise ValueError("invalid prepared fps")
    return batch


def _weights(y: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(y, return_counts=True)
    scale = len(y) / (len(labels) * counts.astype(float))
    mapping = dict(zip(labels.tolist(), scale.tolist()))
    return np.asarray([mapping[int(label)] for label in y])


def _combine(files: list[Path], target: str, require_alive: bool = False):
    batches = [load_prepared(path) for path in files]
    names = batches[0]["feature_names"].astype(str).tolist()
    if any(batch["feature_names"].astype(str).tolist() != names for batch in batches[1:]):
        raise ValueError("prepared matches use different feature schemas")
    protocols = {(int(b.get("schema_version", 1)), str(b.get("observation_view", "legacy_all"))) for b in batches}
    if len(protocols) != 1:
        raise ValueError("prepared schema/view mismatch")
    matrices, labels = [], []
    for batch in batches:
        y = batch[target]
        mask = y >= 0
        if require_alive:
            mask &= batch["y_alive"] == 1
        matrices.append(batch["X"][mask])
        labels.append(y[mask])
    return np.vstack(matrices), np.concatenate(labels), names


def select_classifier(train_files: list[Path], validation_files: list[Path], target: str,
                      grid: dict, seed: int, require_alive: bool = False):
    _, pd, Classifier, balanced_accuracy, log_loss = _imports()
    train_x, train_y, names = _combine(train_files, target, require_alive)
    validation_x, validation_y, validation_names = _combine(validation_files, target, require_alive)
    if names != validation_names:
        raise ValueError("training and validation feature schemas differ")
    if set(np.unique(train_y)) != {0, 1}:
        raise ValueError(f"{target} requires both classes; unknown labels cannot become negatives")
    train_frame = pd.DataFrame(train_x, columns=names)
    validation_frame = pd.DataFrame(validation_x, columns=names)
    candidates = []
    for leaves, rate, l2 in itertools.product(
        grid["max_leaf_nodes"], grid["learning_rate"], grid["l2_regularization"]
    ):
        model = Classifier(
            max_leaf_nodes=int(leaves), learning_rate=float(rate),
            l2_regularization=float(l2), max_iter=int(grid["max_iter"]),
            early_stopping=False, random_state=seed,
        )
        model.fit(train_frame, train_y, sample_weight=_weights(train_y))
        probability = model.predict_proba(validation_frame)[:, 1]
        score = float(balanced_accuracy(validation_y, probability >= 0.5))
        loss = float(log_loss(validation_y, probability, labels=[0, 1]))
        candidates.append((score, -loss, -int(leaves), -float(rate), -float(l2), model,
                           {"balanced_accuracy": score, "log_loss": loss,
                            "max_leaf_nodes": leaves, "learning_rate": rate,
                            "l2_regularization": l2}))
    winner = max(candidates, key=lambda item: item[:5])
    return winner[5], winner[6]


def _truth_and_blocks(batch: dict) -> tuple[np.ndarray, list[tuple[int, int]]]:
    from .quality import period_blocks
    return labels_from_batch(batch), period_blocks(batch["period"])


def _probabilities(alive, home, batch: dict, pd) -> np.ndarray:
    names = batch["feature_names"].astype(str).tolist()
    frame = pd.DataFrame(batch["X"], columns=names)
    p_alive = alive.predict_proba(frame[list(alive.feature_names_in_)])[:, 1]
    p_home = home.predict_proba(frame[list(home.feature_names_in_)])[:, 1]
    return emission_probabilities(p_alive, p_home)


def _evaluate_files(files: list[Path], alive, home, selected: dict, reference_window: int,
                    pd, pm) -> dict:
    matches = []
    for path in files:
        batch = load_prepared(path)
        truth, blocks = _truth_and_blocks(batch)
        predicted = viterbi_decode(
            _probabilities(alive, home, batch, pd), blocks,
            float(selected["team_switch_penalty"]),
            float(selected["dead_switch_penalty"]),
        )
        baseline = period_reference(batch, reference_window / 25.)
        model_result = score_sequence(truth, predicted, batch["period"], float(batch["fps"]))
        baseline_result = score_sequence(truth, baseline, batch["period"], float(batch["fps"]))
        matches.append({
            "match_id": str(batch["match_id"]),
            "model_spell_f1": None, "baseline_spell_f1": None,
            "model_macro_f1": model_result["partial_macro_f1"],
            "baseline_macro_f1": baseline_result["partial_macro_f1"],
            "model_diagnostics": model_result, "baseline_diagnostics": baseline_result,
        })
    return {"matches": matches, "mean_model_spell_f1": None,
            "mean_baseline_spell_f1": None, "non_decrease_fraction": None,
            "status": "incomplete_or_unverified_truth"}


def _binary_macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _metrica_gate(path: str | Path, home, reference_window: int, pd, alive=None, selected=None) -> dict:
    batch = load_prepared(path)
    mask = (batch["y_alive"] == 1) & (batch["y_home"] >= 0)
    if not np.any(mask):
        raise ValueError("Metrica prepared file has no high-confidence H/A frames")
    names = batch["feature_names"].astype(str).tolist()
    frame = pd.DataFrame(batch["X"], columns=names)
    model_prediction = (home.predict_proba(frame[list(home.feature_names_in_)])[:, 1] >= 0.5).astype(int)
    baseline_labels = period_reference(batch, reference_window / 25.)
    baseline_prediction = np.full(len(baseline_labels), -1, dtype=int)
    baseline_prediction[baseline_labels == "H"] = 1
    baseline_prediction[baseline_labels == "A"] = 0
    model_score = _binary_macro_f1(batch["y_home"][mask], model_prediction[mask])
    baseline_score = _binary_macro_f1(batch["y_home"][mask], baseline_prediction[mask])
    had = {"status": "not_tested"}
    if alive is not None:
        from sklearn.metrics import f1_score
        truth, blocks = _truth_and_blocks(batch)
        params = selected or {"team_switch_penalty": 3., "dead_switch_penalty": 1.}
        prediction = viterbi_decode(_probabilities(alive, home, batch, pd), blocks,
                                    params["team_switch_penalty"], params["dead_switch_penalty"])
        valid = (truth != "X") & (batch["period"] > 0)
        had = {"status": "partial_labels_only", "n": int(valid.sum()),
               "per_class_f1": dict(zip("HAD", f1_score(truth[valid], prediction[valid], labels=list("HAD"), average=None, zero_division=0).tolist()))}
    return {
        "partial_had": had,
        "n_frames": int(np.sum(mask)),
        "model_ha_macro_f1": model_score,
        "baseline_ha_macro_f1": baseline_score,
        "delta": model_score - baseline_score,
        "ha_noninferiority": model_score - baseline_score >= -0.02,
        "full_had_sequence_status": "not_tested",
        "passed": False,
    }


def train_from_split(prepared_dir: str | Path, split_path: str | Path,
                     output_dir: str | Path, config: dict,
                     metrica_path: str | Path | None = None) -> dict:
    joblib, pd, _, _, _ = _imports()
    folder = Path(prepared_dir)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    groups = [set(map(str, split[name])) for name in ("train", "validation", "test")]
    if any(len(split[name]) != len(groups[i]) for i, name in enumerate(("train", "validation", "test"))):
        raise ValueError("duplicate match ID within split")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("match overlap between train/validation/test")
    sealed = {"1996435", "2011166", "2015213"}
    if (groups[0] | groups[1]) & sealed:
        raise ValueError("previously consumed test matches cannot be used for development")
    paths = {path.stem: path for path in folder.glob("*.npz")}
    missing = [match for group in ("train", "validation", "test")
               for match in split[group] if match not in paths]
    if missing:
        raise FileNotFoundError(f"prepared matches missing: {sorted(set(missing))}")
    development = {m: load_prepared(paths[m]) for m in split["train"] + split["validation"]}
    if any(str(b["match_id"]) != m for m, b in development.items()):
        raise ValueError("prepared match identity does not match split")
    protocols = {(int(b.get("schema_version", 1)), str(b.get("observation_view", "legacy_all")))
                 for b in development.values()}
    if len(protocols) != 1:
        raise ValueError("training/validation schema or observation view mismatch")
    train_files = [paths[match] for match in split["train"]]
    validation_files = [paths[match] for match in split["validation"]]
    alive, alive_report = select_classifier(train_files, validation_files, "y_alive",
                                             config["model_grid"], config["random_seed"])
    home, home_report = select_classifier(train_files, validation_files, "y_home",
                                           config["model_grid"], config["random_seed"], True)

    # Weak/partial labels cannot rank full-spell decoders. Keep the predeclared baseline.
    selected = {"team_switch_penalty": 3., "dead_switch_penalty": 1.,
                "mean_spell_f1": None,
                "status": "fixed_not_selected_incomplete_truth"}
    reference_window = int(config.get("reference_window_frames", 50))
    validation = _evaluate_files(validation_files, alive, home, selected,
                                 reference_window, pd, None)
    selected["mean_macro_f1"] = float(np.mean([
        row["model_macro_f1"] for row in validation["matches"]]))
    viterbi_rows = [selected.copy()]
    validation["mean_spell_f1_delta"] = None
    validation["passed"] = False
    metrica = None if metrica_path is None else _metrica_gate(
        metrica_path, home, reference_window, pd, alive, selected
    )
    promoted = False  # Full H/A/D external sequence gate and fresh holdout are not established.
    test_report = None
    schema, view = next(iter(protocols))
    metadata = {**selected, "seed": config["random_seed"], "promoted": promoted,
                "schema_version": schema, "observation_view": view,
                "calibration_status": "not_calibrated_legacy_training_path"}
    alive.mtah_metadata = metadata
    home.mtah_metadata = metadata
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(alive, output / "alive.joblib")
    joblib.dump(home, output / "home.joblib")
    report = {"alive_model": alive_report, "home_model": home_report,
              "viterbi": viterbi_rows, "selected_viterbi": selected,
              "validation_gate": validation, "metrica_gate": metrica,
              "promoted": promoted, "test": test_report, "split": split}
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

