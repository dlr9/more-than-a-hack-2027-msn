"""Train and select the two probabilistic classifiers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .decoder import emission_probabilities, viterbi_decode
from .heuristic import majority_smooth


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
        return {name: data[name] for name in data.files}


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
    train_frame = pd.DataFrame(train_x, columns=names)
    validation_frame = pd.DataFrame(validation_x, columns=names)
    candidates = []
    for leaves, rate, l2 in itertools.product(
        grid["max_leaf_nodes"], grid["learning_rate"], grid["l2_regularization"]
    ):
        model = Classifier(
            max_leaf_nodes=int(leaves), learning_rate=float(rate),
            l2_regularization=float(l2), max_iter=int(grid["max_iter"]),
            early_stopping=True, random_state=seed,
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
    # Strict labels train the classifiers; the fuller phase-based surrogate is needed to assess
    # whole spells without turning every high-confidence island into a separate possession.
    y_alive = batch.get("eval_y_alive", batch["y_alive"])
    y_home = batch.get("eval_y_home", batch["y_home"])
    known = (y_alive >= 0) & ((y_alive == 0) | (y_home >= 0))
    indices = np.flatnonzero(known)
    blocks: list[tuple[int, int]] = []
    if len(indices):
        start = previous = int(indices[0])
        for raw_index in indices[1:]:
            index = int(raw_index)
            if index != previous + 1 or batch["period"][index] != batch["period"][previous]:
                blocks.append((start, previous))
                start = index
            previous = index
        blocks.append((start, previous))
    truth = np.full(len(known), "X", dtype="<U1")
    truth[y_alive == 0] = "D"
    truth[(y_alive == 1) & (y_home == 1)] = "H"
    truth[(y_alive == 1) & (y_home == 0)] = "A"
    return truth, blocks


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
        baseline = majority_smooth(batch["nearest_codes"], reference_window)
        model_result = pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted), bootstrap_n=0)
        baseline_result = pm.evaluate(pm.label_codes(truth), pm.label_codes(baseline), bootstrap_n=0)
        matches.append({
            "match_id": str(batch["match_id"]),
            "model_spell_f1": float(model_result["spell_f1_mean"]),
            "baseline_spell_f1": float(baseline_result["spell_f1_mean"]),
            "model_macro_f1": float(model_result["macro_f1"]),
            "baseline_macro_f1": float(baseline_result["macro_f1"]),
        })
    return {
        "matches": matches,
        "mean_model_spell_f1": float(np.mean([row["model_spell_f1"] for row in matches])),
        "mean_baseline_spell_f1": float(np.mean([row["baseline_spell_f1"] for row in matches])),
        "non_decrease_fraction": float(np.mean([
            row["model_spell_f1"] >= row["baseline_spell_f1"] for row in matches
        ])),
    }


def _binary_macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _metrica_gate(path: str | Path, home, reference_window: int, pd) -> dict:
    batch = load_prepared(path)
    mask = (batch["y_alive"] == 1) & (batch["y_home"] >= 0)
    if not np.any(mask):
        raise ValueError("Metrica prepared file has no high-confidence H/A frames")
    names = batch["feature_names"].astype(str).tolist()
    frame = pd.DataFrame(batch["X"], columns=names)
    model_prediction = (home.predict_proba(frame[list(home.feature_names_in_)])[:, 1] >= 0.5).astype(int)
    baseline_labels = majority_smooth(batch["nearest_codes"], reference_window)
    baseline_prediction = np.full(len(baseline_labels), -1, dtype=int)
    baseline_prediction[baseline_labels == "H"] = 1
    baseline_prediction[baseline_labels == "A"] = 0
    model_score = _binary_macro_f1(batch["y_home"][mask], model_prediction[mask])
    baseline_score = _binary_macro_f1(batch["y_home"][mask], baseline_prediction[mask])
    return {
        "n_frames": int(np.sum(mask)),
        "model_ha_macro_f1": model_score,
        "baseline_ha_macro_f1": baseline_score,
        "delta": model_score - baseline_score,
        "passed": model_score - baseline_score >= -0.02,
    }


def train_from_split(prepared_dir: str | Path, split_path: str | Path,
                     output_dir: str | Path, config: dict,
                     metrica_path: str | Path | None = None) -> dict:
    joblib, pd, _, _, _ = _imports()
    folder = Path(prepared_dir)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    paths = {path.stem: path for path in folder.glob("*.npz")}
    missing = [match for group in ("train", "validation", "test")
               for match in split[group] if match not in paths]
    if missing:
        raise FileNotFoundError(f"prepared matches missing: {sorted(set(missing))}")
    train_files = [paths[match] for match in split["train"]]
    validation_files = [paths[match] for match in split["validation"]]
    alive, alive_report = select_classifier(train_files, validation_files, "y_alive",
                                             config["model_grid"], config["random_seed"])
    home, home_report = select_classifier(train_files, validation_files, "y_home",
                                           config["model_grid"], config["random_seed"], True)

    # Select transition penalties on full validation timelines using the official spell metric.
    from evaluation import possession_metrics as pm
    viterbi_rows = []
    for team_penalty, dead_penalty in itertools.product(
        config["viterbi_grid"]["team_switch_penalty"],
        config["viterbi_grid"]["dead_switch_penalty"],
    ):
        spell_scores, macro_scores, segment_counts = [], [], []
        for path in validation_files:
            batch = load_prepared(path)
            probabilities = _probabilities(alive, home, batch, pd)
            truth, blocks = _truth_and_blocks(batch)
            predicted = viterbi_decode(probabilities, blocks, float(team_penalty), float(dead_penalty))
            result = pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted), bootstrap_n=0)
            spell_scores.append(result["spell_f1_mean"])
            macro_scores.append(result["macro_f1"])
            segment_counts.append(result["n_pred_segments"])
        viterbi_rows.append({"team_switch_penalty": team_penalty,
                             "dead_switch_penalty": dead_penalty,
                             "mean_spell_f1": float(np.mean(spell_scores)),
                             "mean_macro_f1": float(np.mean(macro_scores)),
                             "mean_pred_segments": float(np.mean(segment_counts))})
    selected = max(viterbi_rows, key=lambda row: (row["mean_spell_f1"],
                                                   row["mean_macro_f1"],
                                                   -row["mean_pred_segments"],
                                                   -row["team_switch_penalty"],
                                                   -row["dead_switch_penalty"]))
    reference_window = int(config.get("reference_window_frames", 50))
    validation = _evaluate_files(
        validation_files, alive, home, selected, reference_window, pd, pm
    )
    validation["mean_spell_f1_delta"] = (
        validation["mean_model_spell_f1"] - validation["mean_baseline_spell_f1"]
    )
    validation["passed"] = (
        validation["mean_spell_f1_delta"] >= 0.02
        and validation["non_decrease_fraction"] >= 0.70
    )
    metrica = None if metrica_path is None else _metrica_gate(
        metrica_path, home, reference_window, pd
    )
    promoted = bool(validation["passed"] and metrica is not None and metrica["passed"])
    test_report = None
    if promoted:
        test_files = [paths[match] for match in split["test"]]
        test_report = _evaluate_files(
            test_files, alive, home, selected, reference_window, pd, pm
        )
    metadata = {**selected, "seed": config["random_seed"], "promoted": promoted}
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

