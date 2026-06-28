from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURES = [
    "Inflammation score",
    "Alveolar area (%)",
    "Number of alveoli/mm2",
    "Alveolar area (mm2/mm2)",
    "Circumference (mm/mm2)",
    "IL-1b (ng/L)",
    "IL-18 (ng/L)",
    "Bacterial load (lgCFU/g)",
    "TUNEL+ cells",
    "Apoptosis rate (%)",
]

GROUPS = (
    ["Control"] * 5
    + ["P.multocida"] * 5
    + ["Isorhy 10mg/kg"] * 5
    + ["Isorhy 20mg/kg"] * 5
    + ["Isorhy 40mg/kg"] * 5
)

DATA = {
    "Inflammation score": [
        0.0, 0.0, 0.5, 0.6, 0.7,
        4.5, 5.0, 4.8, 5.3, 4.7,
        4.0, 4.1, 3.2, 3.6, 3.9,
        3.5, 3.8, 2.9, 3.4, 3.1,
        1.0, 1.5, 0.9, 1.5, 1.0,
    ],
    "Alveolar area (%)": [
        75.0, 77.2, 74.8, 76.5, 75.9,
        46.3, 47.9, 49.2, 52.0, 41.8,
        72.5, 66.2, 74.0, 65.8, 64.9,
        66.0, 68.5, 70.2, 65.0, 67.8,
        68.1, 72.5, 70.3, 69.8, 71.0,
    ],
    "Number of alveoli/mm2": [
        730, 725, 740, 720, 735,
        355, 360, 350, 345, 368,
        450, 465, 480, 470, 460,
        510, 520, 505, 530, 515,
        650, 665, 680, 660, 670,
    ],
    "Alveolar area (mm2/mm2)": [
        4.8, 5.0, 4.9, 5.1, 4.7,
        1.2, 1.1, 1.3, 1.0, 1.2,
        2.5, 2.3, 2.8, 2.4, 2.6,
        3.0, 2.9, 3.2, 2.8, 3.1,
        4.5, 4.6, 4.4, 4.7, 4.5,
    ],
    "Circumference (mm/mm2)": [
        45, 46, 44, 47, 43,
        19, 20, 18, 21, 19,
        35, 33, 36, 34, 37,
        38, 37, 39, 36, 40,
        42, 43, 41, 44, 42,
    ],
    "IL-1b (ng/L)": [
        4.0, 4.2, 3.8, 4.1, 3.9,
        13.5, 12.8, 13.9, 12.5, 13.2,
        11.8, 10.9, 12.3, 11.2, 11.5,
        9.5, 8.8, 9.2, 8.5, 9.0,
        6.2, 5.8, 6.5, 5.9, 6.0,
    ],
    "IL-18 (ng/L)": [
        14.2, 13.9, 15.0, 14.6, 13.9,
        29.8, 33.5, 29.6, 24.9, 26.0,
        19.2, 17.2, 12.5, 13.9, 15.9,
        22.0, 21.5, 20.8, 22.5, 21.0,
        16.6, 13.0, 9.5, 14.5, 15.0,
    ],
    "Bacterial load (lgCFU/g)": [
        5.0, 5.0, 5.0, 5.0, 5.0,
        6.85, 7.30, 8.70, 6.90, 6.60,
        6.60, 6.78, 6.78, 6.00, 6.48,
        6.40, 6.30, 6.50, 6.20, 6.35,
        5.87, 5.74, 5.99, 6.00, 5.85,
    ],
    "TUNEL+ cells": [
        26, 33, 42, 30, 35,
        137, 166, 214, 172, 160,
        100, 110, 95, 105, 98,
        90, 85, 92, 88, 95,
        75, 80, 70, 78, 73,
    ],
    "Apoptosis rate (%)": [
        7.7, 7.7, 7.7, 7.9, 7.6,
        33.8, 33.8, 33.8, 35.0, 32.5,
        25.0, 25.0, 25.0, 24.0, 26.0,
        20.0, 20.0, 20.0, 19.5, 20.5,
        15.0, 15.0, 15.0, 14.5, 15.5,
    ],
}


@dataclass
class Node:
    prob: float
    feature_idx: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx is None


class SimpleDecisionTree:
    def __init__(self, max_depth: int, min_samples_leaf: int, max_features: int, seed: int):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random = random.Random(seed)
        self.importances: np.ndarray | None = None
        self.root: Node | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 1.0 - (p * p + (1.0 - p) * (1.0 - p))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)
        if parent_impurity == 0.0:
            return None, None, 0.0

        feature_candidates = self.random.sample(
            list(range(n_features)),
            k=min(self.max_features, n_features),
        )
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature_idx in feature_candidates:
            values = np.unique(X[:, feature_idx])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                weighted = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_feature = feature_idx
                    best_threshold = float(threshold)
                    best_gain = float(gain)
        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        prob = float(np.mean(y)) if len(y) else 0.0
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) <= self.min_samples_leaf * 2
        ):
            return Node(prob=prob)

        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or threshold is None or gain <= 0:
            return Node(prob=prob)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Weight gain by the fraction of samples reaching the node to emulate MDI.
        self.importances[feature_idx] += gain * len(y)

        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)
        return Node(
            prob=prob,
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTree":
        self.importances = np.zeros(X.shape[1], dtype=float)
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, row: np.ndarray, node: Node) -> float:
        if node.is_leaf:
            return node.prob
        if row[node.feature_idx] <= node.threshold:
            return self._predict_one(row, node.left)
        return self._predict_one(row, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        return np.array([self._predict_one(row, self.root) for row in X], dtype=float)


class SimpleRandomForest:
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees: list[SimpleDecisionTree] = []
        self.importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRandomForest":
        n_samples, n_features = X.shape
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        rng = np.random.default_rng(self.seed)
        tree_importances = []
        self.trees = []
        for tree_idx in range(self.n_estimators):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            X_boot = X[sample_idx]
            y_boot = y[sample_idx]
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                seed=self.seed + tree_idx,
            ).fit(X_boot, y_boot)
            self.trees.append(tree)
            tree_importances.append(tree.importances)

        stacked = np.vstack(tree_importances)
        imp = stacked.mean(axis=0)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        self.importances_ = imp
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.vstack([tree.predict_proba(X) for tree in self.trees])
        return probs.mean(axis=0)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    concordant = 0.0
    ties = 0.0
    for p in positives:
        concordant += np.sum(p > negatives)
        ties += np.sum(p == negatives)
    total = len(positives) * len(negatives)
    return float((concordant + 0.5 * ties) / total)


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> list[tuple[float, float]]:
    thresholds = sorted(set(float(v) for v in y_score), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]
    points: list[tuple[float, float]] = []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        point = (fpr, tpr)
        if not points or point != points[-1]:
            points.append(point)
    return points


def build_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_all = np.array([[DATA[feature][i] for feature in FEATURES] for i in range(len(GROUPS))], dtype=float)
    y_all = np.array([1 if g == "Control" else 0 for g in GROUPS], dtype=int)
    train_mask = np.array([g in {"Control", "P.multocida"} for g in GROUPS], dtype=bool)
    return X_all, y_all, train_mask


def save_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    out_dir = Path("output")
    X_all, y_all, train_mask = build_matrix()
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    # LOO-CV on training split only: Control vs P.multocida.
    cv_probs = []
    cv_truth = []
    for idx in range(len(X_train)):
        mask = np.ones(len(X_train), dtype=bool)
        mask[idx] = False
        forest = SimpleRandomForest(seed=42).fit(X_train[mask], y_train[mask])
        prob = forest.predict_proba(X_train[idx : idx + 1])[0]
        cv_probs.append(prob)
        cv_truth.append(int(y_train[idx]))
    cv_probs_arr = np.array(cv_probs, dtype=float)
    cv_truth_arr = np.array(cv_truth, dtype=int)

    final_forest = SimpleRandomForest(seed=42).fit(X_train, y_train)
    all_probs = final_forest.predict_proba(X_all)
    group_means = []
    for group in ["Control", "P.multocida", "Isorhy 10mg/kg", "Isorhy 20mg/kg", "Isorhy 40mg/kg"]:
        idx = np.array([g == group for g in GROUPS], dtype=bool)
        group_means.append([group, float(all_probs[idx].mean()), float(all_probs[idx].std(ddof=1))])

    importance_rows = [
        [feature, float(importance)]
        for feature, importance in sorted(
            zip(FEATURES, final_forest.importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    prob_rows = [[GROUPS[i], i + 1, float(all_probs[i])] for i in range(len(GROUPS))]
    roc_rows = [[float(fpr), float(tpr)] for fpr, tpr in roc_points(cv_truth_arr, cv_probs_arr)]
    summary_rows = [
        ["loo_auc", auc_score(cv_truth_arr, cv_probs_arr)],
        ["loo_accuracy", float(np.mean((cv_probs_arr >= 0.5).astype(int) == cv_truth_arr))],
    ]

    save_csv(out_dir / "part2_recovery_rf_importance.csv", ["feature", "importance"], importance_rows)
    save_csv(out_dir / "part2_recovery_rf_probabilities.csv", ["group", "sample_index", "p_control"], prob_rows)
    save_csv(out_dir / "part2_recovery_rf_group_means.csv", ["group", "mean_p_control", "sd_p_control"], group_means)
    save_csv(out_dir / "part2_recovery_rf_roc.csv", ["fpr", "tpr"], roc_rows)
    save_csv(out_dir / "part2_recovery_rf_summary.csv", ["metric", "value"], summary_rows)

    print("Saved:")
    print(out_dir / "part2_recovery_rf_importance.csv")
    print(out_dir / "part2_recovery_rf_probabilities.csv")
    print(out_dir / "part2_recovery_rf_group_means.csv")
    print(out_dir / "part2_recovery_rf_roc.csv")
    print(out_dir / "part2_recovery_rf_summary.csv")


if __name__ == "__main__":
    main()
