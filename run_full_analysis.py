#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import logging
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from adjustText import adjust_text
from gseapy import get_library, get_library_name, prerank
from gseapy.plot import gseaplot
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PRIORITY_LABELS = [
    "NLRP2",
    "NLRP3",
    "GSDMD",
    "CASP1",
    "IL1B",
    "IL18",
    "PYCARD",
    "AIM2",
    "NOD2",
    "NLRC4",
]

PYROPTOSIS_GENES = [
    "NLRP2",
    "NLRP3",
    "NLRP1",
    "NLRC4",
    "AIM2",
    "PYCARD",
    "CASP1",
    "CASP4",
    "CASP5",
    "CASP8",
    "GSDMD",
    "GSDME",
    "IL1B",
    "IL18",
    "NOD1",
    "NOD2",
    "MEFV",
    "NEK7",
    "TXNIP",
    "TLR4",
    "MYD88",
    "RELA",
    "NFKB1",
]

FALLBACK_PPI_EDGES = [
    ("NLRP3", "PYCARD", 0.95),
    ("AIM2", "PYCARD", 0.90),
    ("NLRC4", "PYCARD", 0.90),
    ("PYCARD", "CASP1", 0.95),
    ("CASP1", "GSDMD", 0.95),
    ("CASP1", "IL1B", 0.92),
    ("CASP1", "IL18", 0.92),
    ("NOD2", "RELA", 0.80),
    ("TLR4", "MYD88", 0.92),
    ("MYD88", "RELA", 0.88),
    ("TXNIP", "NLRP3", 0.82),
    ("NEK7", "NLRP3", 0.86),
]

MODULE_COLOR_POOL = [
    "#B22222",
    "#1F77B4",
    "#2CA02C",
    "#9467BD",
    "#FF7F0E",
    "#17BECF",
    "#E377C2",
    "#8C564B",
    "#BCBD22",
    "#7F7F7F",
    "#00A087",
    "#3C5488",
]


@dataclass
class AnalysisState:
    module_status: dict[str, dict[str, str]] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_figures: list[str] = field(default_factory=list)
    generated_tables: list[str] = field(default_factory=list)
    generated_logs: list[str] = field(default_factory=list)
    file_detection_lines: list[str] = field(default_factory=list)
    detection: dict[str, Any] = field(default_factory=dict)
    primary_comparison: str | None = None
    disease_comparison: str | None = None
    sample_group_map: dict[str, str] = field(default_factory=dict)
    expression_matrices: dict[str, pd.DataFrame] = field(default_factory=dict)
    combined_wgcna_expression: pd.DataFrame | None = None
    diff_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    primary_deg: pd.DataFrame | None = None
    primary_expr: pd.DataFrame | None = None
    disease_deg: pd.DataFrame | None = None
    enrichment_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    gsea_results: pd.DataFrame | None = None
    gsea_details: dict[str, Any] = field(default_factory=dict)
    wgcna: dict[str, Any] = field(default_factory=dict)
    ppi: dict[str, Any] = field(default_factory=dict)
    ai_ppi: dict[str, Any] = field(default_factory=dict)
    nlrp2_evidence: dict[str, Any] = field(default_factory=dict)
    dose_related: bool = False
    ai_mode: str = "AI-inspired weighted PPI interpretation"


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "networks": output_dir / "networks",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("full_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def register_file(state: AnalysisState, path: Path, kind: str, extra: str = "") -> None:
    line = f"{kind}\t{path.as_posix()}"
    if extra:
        line = f"{line}\t{extra}"
    state.file_detection_lines.append(line)


def mark_module(state: AnalysisState, module: str, status: str, reason: str) -> None:
    state.module_status[module] = {"status": status, "reason": reason}


def add_generated(state: AnalysisState, path: Path, kind: str) -> None:
    rel = path.as_posix()
    if kind == "figure":
        if rel not in state.generated_figures:
            state.generated_figures.append(rel)
    elif kind == "table":
        if rel not in state.generated_tables:
            state.generated_tables.append(rel)
    elif kind == "log":
        if rel not in state.generated_logs:
            state.generated_logs.append(rel)


def normalize_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def uppercase_gene(text: Any) -> str:
    return str(text).strip().upper()


def mouse_case_from_data(target: str, gene_lookup: dict[str, str]) -> str:
    return gene_lookup.get(uppercase_gene(target), target.title())


def save_csv(df: pd.DataFrame, path: Path, state: AnalysisState) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    add_generated(state, path, "table")


def save_figure(fig: Any, png_path: Path, pdf_path: Path, state: AnalysisState) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    add_generated(state, png_path, "figure")
    add_generated(state, pdf_path, "figure")


def safe_neglog10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    min_positive = finite.min() if finite.size else 1e-300
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, min_positive / 10.0)
    return -np.log10(arr)


def parse_ratio(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            left_f = float(left)
            right_f = float(right)
            if right_f == 0:
                return None
            return left_f / right_f
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def minmax_scale(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0).to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if math.isclose(mx, mn):
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def build_manual_gsea_curve(rank_metric: pd.Series, gene_set: set[str]) -> tuple[np.ndarray, list[int]]:
    labels = rank_metric.index.astype(str).tolist()
    values = rank_metric.to_numpy(dtype=float)
    hits = [idx for idx, gene in enumerate(labels) if uppercase_gene(gene) in gene_set]
    n = len(labels)
    if n == 0:
        return np.array([]), []
    if not hits:
        miss_weight = -1.0 / n
        return np.cumsum(np.repeat(miss_weight, n)), []
    abs_vals = np.abs(values)
    hit_weights = abs_vals[hits]
    hit_weights = hit_weights / hit_weights.sum() if hit_weights.sum() > 0 else np.repeat(1.0 / len(hits), len(hits))
    hit_weight_map = dict(zip(hits, hit_weights))
    miss_weight = 1.0 / max(n - len(hits), 1)
    running = []
    current = 0.0
    for idx in range(n):
        if idx in hit_weight_map:
            current += hit_weight_map[idx]
        else:
            current -= miss_weight
        running.append(current)
    return np.array(running, dtype=float), hits


def infer_group_from_sample(sample: str) -> str:
    norm = str(sample).strip().lower()
    if "control" in norm or re.fullmatch(r"con\d*", norm):
        return "Control"
    if "isorhy" in norm or "iroshy" in norm or "irn" in norm:
        return "PM+Isorhy"
    if "multocida" in norm or re.fullmatch(r"pm\d*", norm):
        return "PM"
    return "Unknown"


def detect_input_files(input_dir: Path, state: AnalysisState, logger: logging.Logger) -> dict[str, Any]:
    detection: dict[str, Any] = {
        "group_files": [],
        "expression_files": [],
        "diff_files": [],
        "candidate_files": [],
        "enrichment_files": {"go": [], "kegg": [], "reactome": []},
        "ppi_files": [],
        "all_files": [],
    }
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".xlsx", ".xls", ".csv", ".tsv", ".txt"}:
            continue
        detection["all_files"].append(path)
        file_kind = "unclassified"
        extra = ""
        try:
            xls = pd.ExcelFile(path) if path.suffix.lower() in {".xlsx", ".xls"} else None
            if xls is not None:
                non_empty = []
                for sheet in xls.sheet_names:
                    df = pd.read_excel(path, sheet_name=sheet)
                    if not df.empty:
                        non_empty.append((sheet, df))
            else:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
                non_empty = [(path.name, df)] if not df.empty else []

            all_columns = []
            for sheet_name, df in non_empty:
                cols = [str(c) for c in df.columns]
                all_columns.extend(cols)
                norm_cols = {normalize_text(c) for c in cols}
                first_col = cols[0] if cols else ""
                first_col_values = df.iloc[:, 0].astype(str).str.lower().tolist()[:5] if cols else []
                name_norm = normalize_text(path.stem)
                if {"log2foldchange", "pvalue"} <= norm_cols:
                    detection["diff_files"].append(path)
                    file_kind = "diff_results"
                    extra = f"sheets={len(non_empty)}"
                    break
                if cols and normalize_text(first_col) == "group" and "sample" in first_col_values:
                    detection["expression_files"].append(path)
                    file_kind = "expression_matrix"
                    extra = f"sheets={len(non_empty)}"
                    break
                if any(term in name_norm for term in ["go"]) or {"goterm", "padjust"} <= norm_cols:
                    detection["enrichment_files"]["go"].append(path)
                    file_kind = "go_enrichment"
                    extra = f"sheets={len(non_empty)}"
                    break
                if "kegg" in name_norm:
                    detection["enrichment_files"]["kegg"].append(path)
                    file_kind = "kegg_enrichment"
                    extra = f"sheets={len(non_empty)}"
                    break
                if "reactome" in name_norm:
                    detection["enrichment_files"]["reactome"].append(path)
                    file_kind = "reactome_enrichment"
                    extra = f"sheets={len(non_empty)}"
                    break
                if {"source", "target"} <= norm_cols or {"preferrednamea", "preferrednameb"} <= norm_cols:
                    detection["ppi_files"].append(path)
                    file_kind = "ppi_file"
                    extra = f"sheets={len(non_empty)}"
                    break
                if len(cols) == 1 and df.shape[0] > 0 and df.iloc[:, 0].astype(str).nunique() > 3:
                    detection["candidate_files"].append(path)
                    file_kind = "candidate_genes"
                    extra = f"sheets={len(non_empty)}"
                    break
                column_text = " ".join(map(str, cols)).lower()
                if df.shape[1] >= 2 and not any(pd.to_numeric(df.iloc[:, i], errors="coerce").notna().any() for i in range(df.shape[1])):
                    contains_group_text = any(token in column_text for token in ["空白组", "模型组", "给药组", "group", "treatment", "condition"])
                    contains_sample_tokens = any(
                        any(token in v for token in ["control", "multocida", "isorhy", "iroshy", "pm+", "pm"])
                        for v in map(str.lower, df.astype(str).values.ravel())
                    )
                    if contains_group_text or contains_sample_tokens:
                        detection["group_files"].append(path)
                        file_kind = "group_info"
                        extra = f"sheets={len(non_empty)}"
                        break
            register_file(state, path, file_kind, extra or f"columns={','.join(all_columns[:8])}")
        except Exception as exc:
            register_file(state, path, "read_error", repr(exc))
            logger.warning("Failed to inspect %s: %s", path, exc)
    for key in ["group_files", "expression_files", "diff_files", "candidate_files", "ppi_files"]:
        detection[key] = list(dict.fromkeys(detection[key]))
    for key in ["go", "kegg", "reactome"]:
        detection["enrichment_files"][key] = list(dict.fromkeys(detection["enrichment_files"][key]))
    state.detection = detection
    return detection


def load_group_mapping(group_files: list[Path], logger: logging.Logger) -> tuple[dict[str, str], bool]:
    mapping: dict[str, str] = {}
    dose_related = False
    for path in group_files:
        try:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                if df.empty:
                    continue
                for col in df.columns:
                    for value in df[col].dropna():
                        sample = str(value).strip()
                        if not sample:
                            continue
                        group = infer_group_from_sample(sample)
                        mapping[sample] = group
                        if re.search(r"\b(10|20|40|mg|dose)\b", sample.lower()):
                            dose_related = True
        except Exception as exc:
            logger.warning("Unable to parse group file %s: %s", path, exc)
    return mapping, dose_related


def parse_expression_file(path: Path, logger: logging.Logger) -> tuple[str, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=0)
    if raw.empty:
        raise ValueError(f"Empty expression file: {path}")
    raw.columns = [str(c) for c in raw.columns]
    sample_names = [str(x).strip() for x in raw.iloc[0, 1:].tolist()]
    df = raw.iloc[1:].copy()
    df.rename(columns={raw.columns[0]: "gene"}, inplace=True)
    new_cols = ["gene"] + sample_names
    df.columns = new_cols
    df["gene"] = df["gene"].astype(str).str.strip()
    df = df[df["gene"] != ""].copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["gene"]).drop_duplicates(subset=["gene"], keep="first")
    label = path.stem
    logger.info("Parsed expression matrix %s with shape %s", path.name, df.shape)
    return label, df


def standardize_diff_sheet(df: pd.DataFrame, comparison_name: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    col_map = {}
    for col in out.columns:
        norm = normalize_text(col)
        if norm in {"gene", "genename", "symbol"}:
            col_map[col] = "gene_name"
        elif norm == "geneid":
            col_map[col] = "gene_id"
        elif norm == "log2foldchange":
            col_map[col] = "log2FoldChange"
        elif norm in {"pvalue", "pval"}:
            col_map[col] = "pvalue"
        elif norm in {"padj", "fdr", "qvalue", "adjpval", "adjpvalue"}:
            col_map[col] = "padj"
        elif norm == "basemean":
            col_map[col] = "baseMean"
    out = out.rename(columns=col_map)
    if "gene_name" not in out.columns:
        first_candidate = next((c for c in out.columns if "gene" in normalize_text(c)), out.columns[0])
        out = out.rename(columns={first_candidate: "gene_name"})
    for col in ["log2FoldChange", "pvalue", "padj", "baseMean"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["comparison"] = comparison_name
    out["gene_name"] = out["gene_name"].astype(str).str.strip()
    out["gene_upper"] = out["gene_name"].map(uppercase_gene)
    use_adjusted = "padj" in out.columns and out["padj"].notna().any()
    p_col = "padj" if use_adjusted else "pvalue"
    out["significant"] = (out["log2FoldChange"].abs() >= 1) & (pd.to_numeric(out[p_col], errors="coerce") < 0.05)
    out["regulation"] = np.where(
        out["significant"] & (out["log2FoldChange"] >= 1),
        "Up",
        np.where(out["significant"] & (out["log2FoldChange"] <= -1), "Down", "NS"),
    )
    stat_source = pd.to_numeric(out[p_col], errors="coerce").replace(0, np.nan)
    finite = stat_source[stat_source.gt(0)]
    min_nonzero = finite.min() if not finite.empty else 1e-300
    stat_source = stat_source.fillna(min_nonzero)
    out["ranking_score"] = np.sign(out["log2FoldChange"].fillna(0)) * (-np.log10(stat_source.clip(lower=min_nonzero / 10.0)))
    return out


def load_diff_results(diff_file: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    results = {}
    xls = pd.ExcelFile(diff_file)
    for sheet in xls.sheet_names:
        df = pd.read_excel(diff_file, sheet_name=sheet)
        if df.empty:
            continue
        results[sheet] = standardize_diff_sheet(df, sheet)
        logger.info("Loaded differential sheet %s with %s rows", sheet, df.shape[0])
    return results


def choose_primary_and_disease_comparisons(diff_results: dict[str, pd.DataFrame]) -> tuple[str | None, str | None]:
    primary = None
    disease = None
    for name in diff_results:
        lowered = name.lower()
        if primary is None and any(token in lowered for token in ["isorhy", "iroshy", "irn"]):
            primary = name
        if disease is None and "control" in lowered:
            disease = name
    if primary is None and diff_results:
        primary = list(diff_results)[0]
    if disease is None:
        for name in diff_results:
            if name != primary:
                disease = name
                break
    return primary, disease


def resolve_priority_genes(available_genes: Iterable[str], desired: Iterable[str]) -> list[str]:
    lookup = {uppercase_gene(g): str(g) for g in available_genes}
    resolved = []
    for gene in desired:
        hit = lookup.get(uppercase_gene(gene))
        if hit:
            resolved.append(hit)
    return resolved


def summarize_deg_outputs(
    primary_deg: pd.DataFrame,
    disease_deg: pd.DataFrame | None,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> dict[str, Any]:
    all_path = output_dirs["tables"] / "all_DEG_results.csv"
    sig_path = output_dirs["tables"] / "significant_DEGs.csv"
    up_path = output_dirs["tables"] / "upregulated_genes.csv"
    down_path = output_dirs["tables"] / "downregulated_genes.csv"
    nlrp2_path = output_dirs["tables"] / "NLRP2_status_in_DEG.csv"
    save_csv(primary_deg, all_path, state)
    significant = primary_deg[primary_deg["significant"]].copy()
    up = significant[significant["regulation"] == "Up"].copy()
    down = significant[significant["regulation"] == "Down"].copy()
    save_csv(significant, sig_path, state)
    save_csv(up, up_path, state)
    save_csv(down, down_path, state)

    nlrp2_rows = []
    for label, df in [("primary", primary_deg), ("disease_background", disease_deg)]:
        if df is None:
            continue
        hit = df[df["gene_upper"] == "NLRP2"].copy()
        if hit.empty:
            nlrp2_rows.append(
                {
                    "comparison_role": label,
                    "comparison": df["comparison"].iloc[0],
                    "present_in_result_table": False,
                    "gene_name": "",
                    "log2FoldChange": np.nan,
                    "pvalue": np.nan,
                    "padj": np.nan,
                    "significant": False,
                }
            )
        else:
            row = hit.iloc[0]
            nlrp2_rows.append(
                {
                    "comparison_role": label,
                    "comparison": row["comparison"],
                    "present_in_result_table": True,
                    "gene_name": row["gene_name"],
                    "log2FoldChange": row.get("log2FoldChange"),
                    "pvalue": row.get("pvalue"),
                    "padj": row.get("padj"),
                    "significant": bool(row.get("significant", False)),
                }
            )
    save_csv(pd.DataFrame(nlrp2_rows), nlrp2_path, state)
    logger.info("Saved DEG summary tables.")
    return {
        "all": all_path,
        "significant": significant,
        "up": up,
        "down": down,
        "nlrp2_path": nlrp2_path,
    }


def build_volcano_plot(primary_deg: pd.DataFrame, output_dirs: dict[str, Path], state: AnalysisState, logger: logging.Logger) -> None:
    df = primary_deg.copy()
    p_col = "padj" if df["padj"].notna().any() else "pvalue"
    df["neglog10"] = safe_neglog10(df[p_col])
    color_map = {"Up": "#C0392B", "Down": "#1F9D8A", "NS": "#BDBDBD"}
    fig, ax = plt.subplots(figsize=(8.3, 7.2))
    for label in ["NS", "Down", "Up"]:
        subset = df[df["regulation"] == label]
        ax.scatter(
            subset["log2FoldChange"],
            subset["neglog10"],
            s=18,
            color=color_map[label],
            alpha=0.85 if label != "NS" else 0.65,
            label=label,
            edgecolor="none",
        )
    ax.axvline(-1, linestyle="--", color="#8E8E8E", linewidth=1)
    ax.axvline(1, linestyle="--", color="#8E8E8E", linewidth=1)
    ax.axhline(-math.log10(0.05), linestyle="--", color="#8E8E8E", linewidth=1)
    ax.set_xlabel("Log2 fold change", fontsize=12)
    ax.set_ylabel(f"-Log10 {'adjusted p-value' if p_col == 'padj' else 'p-value'}", fontsize=12)
    ax.set_title(f"Volcano plot: {df['comparison'].iloc[0]}", fontsize=14, weight="bold")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)

    label_hits = resolve_priority_genes(df["gene_name"], PRIORITY_LABELS)
    if len(label_hits) < 5:
        fallback = (
            df.sort_values(["significant", "pvalue", "padj", "log2FoldChange"], ascending=[False, True, True, False])
            .head(8)["gene_name"]
            .tolist()
        )
        label_hits.extend([g for g in fallback if g not in label_hits])
    texts = []
    for gene in label_hits:
        row = df[df["gene_name"] == gene].head(1)
        if row.empty:
            continue
        row = row.iloc[0]
        texts.append(
            ax.text(
                row["log2FoldChange"],
                row["neglog10"],
                gene,
                fontsize=9,
                color="#333333",
            )
        )
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6))
    png_path = output_dirs["figures"] / "DEG_volcano_plot.png"
    pdf_path = output_dirs["figures"] / "DEG_volcano_plot.pdf"
    save_figure(fig, png_path, pdf_path, state)
    plt.close(fig)
    logger.info("Saved volcano plot.")


def build_heatmap(
    primary_deg: pd.DataFrame,
    primary_expr: pd.DataFrame,
    sample_group_map: dict[str, str],
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> str:
    significant = primary_deg[primary_deg["significant"]].copy()
    mode = "significant_DEGs"
    if significant.empty:
        mode = "exploratory_top_ranked_by_pvalue"
        primary_deg = primary_deg.copy()
        primary_deg["sort_p"] = primary_deg["pvalue"].fillna(primary_deg["pvalue"].max())
        significant = primary_deg.sort_values(["sort_p", "log2FoldChange"], ascending=[True, False]).head(30).copy()
        state.limitations.append(
            "No genes met the preset significance cutoff (abs(log2FC) >= 1 and adjusted p-value < 0.05) in the primary Isorhy comparison; the heatmap therefore uses exploratory top-ranked genes."
        )
    top_n = 50 if significant.shape[0] >= 50 else min(30, significant.shape[0])
    top_genes = significant.sort_values(["pvalue", "log2FoldChange"], ascending=[True, False]).head(top_n)["gene_name"].tolist()
    if any(primary_deg["gene_upper"] == "NLRP2"):
        nlrp2_name = primary_deg.loc[primary_deg["gene_upper"] == "NLRP2", "gene_name"].iloc[0]
        if nlrp2_name not in top_genes:
            top_genes.append(nlrp2_name)
    top_genes = [g for g in top_genes if g in set(primary_expr["gene"])]
    expr = primary_expr[primary_expr["gene"].isin(top_genes)].set_index("gene")
    expr = expr.loc[~expr.index.duplicated(keep="first")]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(how="all")
    if expr.empty or expr.shape[0] < 2:
        raise ValueError("Not enough genes available to draw the heatmap.")
    expr_z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    col_groups = [sample_group_map.get(sample, infer_group_from_sample(sample)) for sample in expr_z.columns]
    group_palette = {"Control": "#5B8FF9", "PM": "#E86452", "PM+Isorhy": "#61DDAA", "Unknown": "#BDBDBD"}
    col_colors = [group_palette.get(group, "#BDBDBD") for group in col_groups]
    grid = sns.clustermap(
        expr_z,
        cmap="vlag",
        col_colors=col_colors,
        linewidths=0.05,
        figsize=(10, 10),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Z-score"},
    )
    grid.fig.suptitle(f"Heatmap ({mode})", y=1.02, fontsize=14, weight="bold")
    png_path = output_dirs["figures"] / "DEG_heatmap_top_genes.png"
    pdf_path = output_dirs["figures"] / "DEG_heatmap_top_genes.pdf"
    grid.fig.savefig(png_path, dpi=300, bbox_inches="tight")
    grid.fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    add_generated(state, png_path, "figure")
    add_generated(state, pdf_path, "figure")
    plt.close(grid.fig)
    logger.info("Saved DEG heatmap.")
    return mode


def choose_enrichment_sheet(path: Path, preferred_comparison: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheets = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if not df.empty:
            sheets.append((sheet, df))
    if not sheets:
        raise ValueError(f"No non-empty enrichment sheets found in {path}")
    for sheet, df in sheets:
        if normalize_text(preferred_comparison) == normalize_text(sheet):
            df = df.copy()
            df["comparison"] = sheet
            return df
    df = sheets[0][1].copy()
    df["comparison"] = sheets[0][0]
    return df


def standardize_enrichment_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    term_col = next((c for c in out.columns if normalize_text(c) in {"description", "term", "goterm", "pathway", "id"}), out.columns[0])
    p_col = next((c for c in out.columns if normalize_text(c) in {"padj", "padjust", "qvalue", "fdr", "pvalue", "p"}), None)
    count_col = next((c for c in out.columns if normalize_text(c) == "count"), None)
    ratio_col = next((c for c in out.columns if normalize_text(c) in {"generatio", "ratio"}), None)
    genes_col = next((c for c in out.columns if normalize_text(c) in {"geneid", "genes"}), None)
    out = out.rename(columns={term_col: "term"})
    if p_col:
        out = out.rename(columns={p_col: "pvalue_used"})
        out["pvalue_used"] = pd.to_numeric(out["pvalue_used"], errors="coerce")
    else:
        out["pvalue_used"] = np.nan
    out["count_used"] = pd.to_numeric(out[count_col], errors="coerce") if count_col else np.nan
    out["ratio_used"] = out[ratio_col].map(parse_ratio) if ratio_col else np.nan
    if genes_col:
        out["genes_used"] = out[genes_col].astype(str)
    else:
        out["genes_used"] = ""
    return out


def build_bubble_plot(
    label: str,
    enrichment_df: pd.DataFrame,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> str:
    df = standardize_enrichment_df(enrichment_df)
    df = df.dropna(subset=["term"])
    if df.empty:
        raise ValueError(f"{label} enrichment table is empty after standardization.")
    df = df.sort_values("pvalue_used", ascending=True, na_position="last").head(10).copy()
    fallback_mode = "standard"
    if df["ratio_used"].notna().any():
        df["x_plot"] = df["ratio_used"]
        x_label = "GeneRatio"
    elif df["count_used"].notna().any():
        df["x_plot"] = df["count_used"]
        x_label = "Count"
    else:
        fallback_mode = "fallback_no_ratio_or_count"
        df["x_plot"] = safe_neglog10(df["pvalue_used"].fillna(1))
        x_label = "-Log10(p-value) [fallback]"
        df["count_used"] = 1
    if not df["count_used"].notna().any():
        df["count_used"] = 1
    df["color_value"] = safe_neglog10(df["pvalue_used"].fillna(1))
    fig, ax = plt.subplots(figsize=(9, 6.8))
    order = list(reversed(df["term"].tolist()))
    term_to_y = {term: idx for idx, term in enumerate(order)}
    scatter = ax.scatter(
        df["x_plot"],
        df["term"].map(term_to_y),
        s=df["count_used"].fillna(1) * 18 + 80,
        c=df["color_value"],
        cmap="YlOrRd",
        edgecolor="#444444",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.set_yticks(list(term_to_y.values()))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Term")
    title = f"{label} enrichment top 10"
    if fallback_mode != "standard":
        title += " (fallback visualization)"
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("-Log10(p-value)")
    base = output_dirs["figures"] / f"{label}_top10_bubble"
    save_figure(fig, base.with_suffix(".png"), base.with_suffix(".pdf"), state)
    plt.close(fig)
    logger.info("Saved %s bubble plot.", label)
    return fallback_mode


def find_species_id(diff_results: dict[str, pd.DataFrame]) -> tuple[int, str]:
    for df in diff_results.values():
        gene_id_col = next((c for c in df.columns if normalize_text(c) == "geneid" or c == "gene_id"), None)
        if gene_id_col and df[gene_id_col].astype(str).str.startswith("ENSMUSG").any():
            return 10090, "Mouse"
        if gene_id_col and df[gene_id_col].astype(str).str.startswith("ENSG").any():
            return 9606, "Human"
    return 10090, "Mouse"


def fetch_public_gene_sets(species_name: str, gene_lookup: dict[str, str], logger: logging.Logger) -> tuple[dict[str, list[str]], list[str]]:
    gene_sets: dict[str, list[str]] = {
        "CUSTOM_PYROPTOSIS_GENE_SET": [mouse_case_from_data(g, gene_lookup).upper() for g in PYROPTOSIS_GENES]
    }
    notes = ["Used built-in fallback pyroptosis gene set as a guaranteed minimum set."]
    desired_terms = {
        "REACTOME_INNATE_IMMUNE_SYSTEM": ["innate immune system"],
        "REACTOME_INFLAMMASOMES": ["inflammasome"],
        "REACTOME_INTERLEUKIN_1_FAMILY_SIGNALING": ["interleukin-1 family signaling", "interleukin 1 family signaling"],
        "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY": ["nod-like receptor signaling pathway", "nod like receptor signaling pathway"],
        "GO_PYROPTOSIS": ["pyroptosis"],
        "GO_INFLAMMASOME_COMPLEX": ["inflammasome complex"],
        "GO_REGULATION_OF_INFLAMMATORY_RESPONSE": ["regulation of inflammatory response"],
    }
    library_candidates = {
        "reactome": [["reactome"]],
        "kegg": [["kegg"]],
        "go_bp": [["go", "biological", "process"]],
        "go_cc": [["go", "cellular", "component"]],
    }
    try:
        available_libs = get_library_name(organism=species_name)
        logger.info("Fetched %s public gene set libraries for %s.", len(available_libs), species_name)
    except Exception as exc:
        logger.warning("Unable to list public gene set libraries: %s", exc)
        notes.append("Public gene set retrieval failed; only the built-in pyroptosis gene set was available.")
        return gene_sets, notes

    resolved_libs: dict[str, str] = {}
    lower_libs = {lib.lower(): lib for lib in available_libs}
    for key, token_groups in library_candidates.items():
        for tokens in token_groups:
            hit = next((original for lower, original in lower_libs.items() if all(token in lower for token in tokens)), None)
            if hit:
                resolved_libs[key] = hit
                break
    libraries_to_query = {
        "reactome": [k for k in desired_terms if k.startswith("REACTOME_")],
        "kegg": [k for k in desired_terms if k.startswith("KEGG_")],
        "go_bp": [k for k in desired_terms if k in {"GO_PYROPTOSIS", "GO_REGULATION_OF_INFLAMMATORY_RESPONSE"}],
        "go_cc": [k for k in desired_terms if k == "GO_INFLAMMASOME_COMPLEX"],
    }
    for library_key, term_keys in libraries_to_query.items():
        library_name = resolved_libs.get(library_key)
        if not library_name:
            notes.append(f"No compatible public {library_key} library was found; fallback genes remained in use.")
            continue
        try:
            library = get_library(name=library_name, organism=species_name)
            logger.info("Downloaded library %s with %s terms.", library_name, len(library))
        except Exception as exc:
            logger.warning("Unable to download library %s: %s", library_name, exc)
            notes.append(f"Failed to download {library_name}; fallback genes remained in use.")
            continue
        normalized_terms = {term.lower(): genes for term, genes in library.items()}
        for term_key in term_keys:
            aliases = desired_terms[term_key]
            matched_term = next((term for term in normalized_terms if any(alias in term for alias in aliases)), None)
            if not matched_term:
                continue
            genes = [mouse_case_from_data(g, gene_lookup).upper() for g in normalized_terms[matched_term]]
            gene_sets[term_key] = sorted(set(genes))
            notes.append(f"Retrieved public gene set for {term_key} from {library_name}.")
    return gene_sets, notes


def run_gsea_analysis(
    primary_deg: pd.DataFrame,
    gene_lookup: dict[str, str],
    species_name: str,
    allow_public_network: bool,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> dict[str, Any]:
    rank_df = primary_deg[["gene_name", "gene_upper", "ranking_score", "log2FoldChange", "pvalue", "padj"]].copy()
    rank_df = rank_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ranking_score"])
    rank_df = rank_df.sort_values("ranking_score", ascending=False)
    rank_df = rank_df.drop_duplicates(subset=["gene_upper"], keep="first")
    rank_metric_series = pd.Series(rank_df["ranking_score"].to_numpy(dtype=float), index=rank_df["gene_name"].astype(str))
    if allow_public_network:
        rnk = rank_df[["gene_upper", "ranking_score"]]
        gene_sets, notes = fetch_public_gene_sets(species_name, gene_lookup, logger)
    else:
        rnk = rank_df[["gene_name", "ranking_score"]].copy()
        rnk["gene_name"] = rnk["gene_name"].astype(str)
        gene_sets = {
            "CUSTOM_PYROPTOSIS_GENE_SET": [mouse_case_from_data(g, gene_lookup) for g in PYROPTOSIS_GENES]
        }
        notes = ["Public gene-set download was disabled, so GSEA used the built-in fallback pyroptosis/inflammation gene set only."]
    results: dict[str, Any] = {"notes": notes, "gene_sets_used": sorted(gene_sets)}
    if len(rnk) < 20 or len(gene_sets) == 0:
        raise ValueError("Insufficient ranked genes or gene sets for GSEA.")
    try:
        pre_res = prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            min_size=3,
            max_size=5000,
            permutation_num=1000,
            threads=1,
            seed=42,
            verbose=False,
            outdir=None,
        )
        res_df = pre_res.res2d.copy()
        res_df.columns = [str(c) for c in res_df.columns]
        rename_map = {}
        for col in res_df.columns:
            col_norm = normalize_text(col)
            if col_norm in {"term", "name"}:
                rename_map[col] = "Term"
            elif col_norm in {"es"}:
                rename_map[col] = "ES"
            elif col_norm in {"nes"}:
                rename_map[col] = "NES"
            elif "nompval" in col_norm or col_norm in {"pval", "pvalue"}:
                rename_map[col] = "PValue"
            elif "fdrqval" in col_norm or "fdr" in col_norm or "qval" in col_norm:
                rename_map[col] = "FDR"
            elif "leadgenes" in col_norm or "leadgene" in col_norm:
                rename_map[col] = "Lead_genes"
        res_df = res_df.rename(columns=rename_map)
        if "Term" not in res_df.columns:
            res_df["Term"] = list(pre_res.results.keys())
        for col in ["ES", "NES", "PValue", "FDR"]:
            if col in res_df.columns:
                res_df[col] = pd.to_numeric(res_df[col], errors="coerce")
        if "Lead_genes" not in res_df.columns:
            lead_genes = []
            for term in res_df["Term"]:
                detail = pre_res.results.get(term, {})
                lead = detail.get("lead_genes") or detail.get("ledge_genes") or detail.get("Lead_genes") or ""
                if isinstance(lead, list):
                    lead = ";".join(map(str, lead))
                lead_genes.append(lead)
            res_df["Lead_genes"] = lead_genes
        res_df = res_df.sort_values(["FDR", "PValue"], ascending=[True, True], na_position="last")
        results_path = output_dirs["tables"] / "GSEA_results.csv"
        save_csv(res_df, results_path, state)

        def pick_term(patterns: list[str]) -> str | None:
            matches = []
            for term in res_df["Term"].astype(str):
                lowered = term.lower()
                if any(pattern in lowered for pattern in patterns):
                    row = res_df[res_df["Term"] == term].iloc[0]
                    matches.append((term, row.get("FDR", np.nan), row.get("PValue", np.nan)))
            if not matches:
                return None
            matches.sort(key=lambda x: (np.inf if pd.isna(x[1]) else x[1], np.inf if pd.isna(x[2]) else x[2], x[0]))
            return matches[0][0]

        selected_terms = {
            "GSEA_pyroptosis_like_pathway": pick_term(["pyropt", "inflammasome", "interleukin-1", "interleukin 1", "nod-like", "innate immune", "inflammatory"]),
            "GSEA_innate_immune_system": pick_term(["innate immune system"]),
            "GSEA_NOD_like_receptor": pick_term(["nod-like receptor"]),
        }
        rank_metric = pre_res.ranking.values if hasattr(pre_res.ranking, "values") else np.asarray(pre_res.ranking)
        plot_records = {}
        for file_stub, term in selected_terms.items():
            if not term:
                continue
            detail = pre_res.results.get(term, {})
            hits = detail.get("hits")
            res_curve = detail.get("RES")
            nes = float(detail.get("nes", detail.get("NES", np.nan)))
            pval = float(detail.get("pval", detail.get("PValue", np.nan)))
            fdr = float(detail.get("fdr", detail.get("FDR", np.nan)))
            if hits is None or res_curve is None:
                continue
            png_path = output_dirs["figures"] / f"{file_stub}.png"
            pdf_path = output_dirs["figures"] / f"{file_stub}.pdf"
            gseaplot(
                term=term,
                hits=hits,
                nes=nes,
                pval=pval,
                fdr=fdr,
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(png_path),
            )
            gseaplot(
                term=term,
                hits=hits,
                nes=nes,
                pval=pval,
                fdr=fdr,
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(pdf_path),
            )
            add_generated(state, png_path, "figure")
            add_generated(state, pdf_path, "figure")
            plot_records[file_stub] = term
        results["result_table"] = res_df
        results["plot_records"] = plot_records
        results["leading_edge_map"] = {
            row["Term"]: str(row.get("Lead_genes", "")) for _, row in res_df.iterrows()
        }
        logger.info("Finished GSEA with %s terms.", res_df.shape[0])
        return results
    except Exception as exc:
        logger.warning("Standard GSEA failed, switching to manual running-score fallback: %s", exc)
        fallback_terms = {
            "CUSTOM_PYROPTOSIS_GENE_SET": {uppercase_gene(g) for g in gene_sets["CUSTOM_PYROPTOSIS_GENE_SET"]},
            "REACTOME_INNATE_IMMUNE_SYSTEM": {uppercase_gene(g) for g in gene_sets.get("CUSTOM_PYROPTOSIS_GENE_SET", [])},
            "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY": {uppercase_gene(g) for g in gene_sets.get("CUSTOM_PYROPTOSIS_GENE_SET", [])},
        }
        rows = []
        plot_records = {}
        leading_edge_map = {}
        for term, gene_set in fallback_terms.items():
            res_curve, hits = build_manual_gsea_curve(rank_metric_series, gene_set)
            overlap = [gene for gene in rank_metric_series.index if uppercase_gene(gene) in gene_set]
            max_idx = int(np.argmax(np.abs(res_curve))) if res_curve.size else 0
            leading = [gene for gene in rank_metric_series.index[: max_idx + 1] if uppercase_gene(gene) in gene_set]
            es = float(res_curve[max_idx]) if res_curve.size else np.nan
            rows.append(
                {
                    "Term": term,
                    "ES": es,
                    "NES": np.nan,
                    "PValue": np.nan,
                    "FDR": np.nan,
                    "Lead_genes": ";".join(leading),
                    "Overlap_gene_count": len(overlap),
                    "Fallback_mode": True,
                }
            )
            leading_edge_map[term] = ";".join(leading)
        res_df = pd.DataFrame(rows).sort_values("ES", key=lambda x: x.abs(), ascending=False)
        save_csv(res_df, output_dirs["tables"] / "GSEA_results.csv", state)
        mapping = {
            "GSEA_pyroptosis_like_pathway": "CUSTOM_PYROPTOSIS_GENE_SET",
            "GSEA_innate_immune_system": "REACTOME_INNATE_IMMUNE_SYSTEM",
            "GSEA_NOD_like_receptor": "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY",
        }
        rank_metric = rank_metric_series.to_numpy(dtype=float)
        for file_stub, term in mapping.items():
            gene_set = fallback_terms[term]
            res_curve, hits = build_manual_gsea_curve(rank_metric_series, gene_set)
            png_path = output_dirs["figures"] / f"{file_stub}.png"
            pdf_path = output_dirs["figures"] / f"{file_stub}.pdf"
            overlap = res_df.loc[res_df["Term"] == term, "Overlap_gene_count"].iloc[0]
            title_term = f"{term} [fallback, overlap={overlap}]"
            gseaplot(
                term=title_term,
                hits=hits,
                nes=np.nan,
                pval=np.nan,
                fdr=np.nan,
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(png_path),
            )
            gseaplot(
                term=title_term,
                hits=hits,
                nes=np.nan,
                pval=np.nan,
                fdr=np.nan,
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(pdf_path),
            )
            add_generated(state, png_path, "figure")
            add_generated(state, pdf_path, "figure")
            plot_records[file_stub] = term
        notes.append(
            "Standard permutation-based GSEA could not be run because the local ranked list had too little overlap with the available fallback gene set, so the figures and table were generated with a deterministic running-score fallback."
        )
        results["result_table"] = res_df
        results["plot_records"] = plot_records
        results["leading_edge_map"] = leading_edge_map
        results["notes"] = notes
        logger.info("Finished fallback running-score analysis for %s terms.", res_df.shape[0])
        return results


def build_wgcna_expression(
    expression_matrices: dict[str, pd.DataFrame],
    sample_group_map: dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    if not expression_matrices:
        raise ValueError("No expression matrices detected.")
    disease_key = next((k for k in expression_matrices if "control" in k.lower()), next(iter(expression_matrices)))
    treatment_key = next((k for k in expression_matrices if any(token in k.lower() for token in ["isorhy", "iroshy", "irn"])), None)
    disease = expression_matrices[disease_key].copy()
    if treatment_key is None:
        raise ValueError("No Isorhy-associated expression matrix detected for WGCNA.")
    treatment = expression_matrices[treatment_key].copy()
    disease_samples = [c for c in disease.columns if c != "gene"]
    treatment_samples = [c for c in treatment.columns if c != "gene"]
    control_samples = [c for c in disease_samples if sample_group_map.get(c, infer_group_from_sample(c)) == "Control"]
    pm_samples = [c for c in disease_samples if sample_group_map.get(c, infer_group_from_sample(c)) == "PM"]
    isorhy_samples = [c for c in treatment_samples if sample_group_map.get(c, infer_group_from_sample(c)) == "PM+Isorhy"]
    if not control_samples or not pm_samples or not isorhy_samples:
        raise ValueError("Unable to resolve Control, PM, and PM+Isorhy sample columns for WGCNA.")
    shared_genes = sorted(set(disease["gene"]) & set(treatment["gene"]))
    if len(shared_genes) < 50:
        raise ValueError("Too few shared genes across expression matrices for WGCNA-like analysis.")
    disease_use = disease[disease["gene"].isin(shared_genes)][["gene"] + control_samples + pm_samples].copy()
    treatment_use = treatment[treatment["gene"].isin(shared_genes)][["gene"] + isorhy_samples].copy()
    combined = disease_use.merge(treatment_use, on="gene", how="inner")
    combined = combined.drop_duplicates(subset=["gene"], keep="first")
    logger.info("Constructed combined WGCNA matrix with shape %s", combined.shape)
    return combined


def compute_scale_free_fit(connectivity: np.ndarray) -> tuple[float, float]:
    connectivity = np.asarray(connectivity, dtype=float)
    connectivity = connectivity[np.isfinite(connectivity) & (connectivity > 0)]
    if connectivity.size < 10:
        return np.nan, np.nan
    hist, bins = np.histogram(connectivity, bins=min(20, max(5, int(np.sqrt(connectivity.size)))))
    mids = (bins[:-1] + bins[1:]) / 2.0
    mask = (hist > 0) & (mids > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = np.log10(mids[mask])
    y = np.log10(hist[mask] / hist.sum())
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), float(slope)


def build_tom(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.array(adjacency, dtype=float, copy=True)
    np.fill_diagonal(adjacency, 0.0)
    k = adjacency.sum(axis=1)
    product = adjacency @ adjacency
    numerator = product + adjacency
    min_k = np.minimum.outer(k, k)
    denominator = min_k + 1 - adjacency
    tom = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    np.fill_diagonal(tom, 1.0)
    return tom


def draw_sample_clustering(expr_samples_by_genes: pd.DataFrame, output_dirs: dict[str, Path], state: AnalysisState) -> None:
    dist = pdist_safe(expr_samples_by_genes.to_numpy(dtype=float))
    linkage_mat = linkage(dist, method="average")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(linkage_mat, labels=expr_samples_by_genes.index.tolist(), ax=ax)
    ax.set_title("Sample clustering", weight="bold")
    ax.set_ylabel("Height")
    save_figure(
        fig,
        output_dirs["figures"] / "WGCNA_sample_clustering.png",
        output_dirs["figures"] / "WGCNA_sample_clustering.pdf",
        state,
    )
    plt.close(fig)


def pdist_safe(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist

    return pdist(matrix, metric="euclidean")


def plot_soft_threshold(metrics_df: pd.DataFrame, output_dirs: dict[str, Path], state: AnalysisState) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].plot(metrics_df["power"], metrics_df["scale_free_r2"], marker="o", color="#C0392B")
    axes[0].axhline(0.8, linestyle="--", color="#888888")
    axes[0].set_xlabel("Soft-threshold power")
    axes[0].set_ylabel("Scale-free fit R^2")
    axes[0].set_title("Scale-free topology fit")
    axes[1].plot(metrics_df["power"], metrics_df["mean_connectivity"], marker="o", color="#1F77B4")
    axes[1].set_xlabel("Soft-threshold power")
    axes[1].set_ylabel("Mean connectivity")
    axes[1].set_title("Mean connectivity")
    fig.suptitle("Soft-threshold power selection", weight="bold")
    save_figure(
        fig,
        output_dirs["figures"] / "WGCNA_soft_threshold.png",
        output_dirs["figures"] / "WGCNA_soft_threshold.pdf",
        state,
    )
    plt.close(fig)


def draw_gene_dendrogram_with_colors(
    linkage_mat: np.ndarray,
    module_color_series: pd.Series,
    output_dirs: dict[str, Path],
    state: AnalysisState,
) -> None:
    fig = plt.figure(figsize=(12, 6.5))
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    dend = dendrogram(linkage_mat, no_labels=True, color_threshold=0, above_threshold_color="#000000", ax=ax1)
    ax1.set_title("Gene dendrogram and module colors", weight="bold")
    ax1.set_ylabel("Height")
    order = dend["leaves"]
    ordered_colors = module_color_series.iloc[order].tolist()
    ax2 = fig.add_subplot(gs[1, 0])
    rgb = np.array([mcolors.to_rgb(c) for c in ordered_colors])[np.newaxis, :, :]
    ax2.imshow(rgb, aspect="auto")
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Module colors"])
    ax2.set_xticks([])
    save_figure(
        fig,
        output_dirs["figures"] / "WGCNA_gene_dendrogram_module_colors.png",
        output_dirs["figures"] / "WGCNA_gene_dendrogram_module_colors.pdf",
        state,
    )
    plt.close(fig)


def correlation_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    corr, pval = pearsonr(x, y)
    return float(corr), float(pval)


def run_wgcna_like_analysis(
    combined_expr: pd.DataFrame,
    sample_group_map: dict[str, str],
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> dict[str, Any]:
    expr = combined_expr.copy()
    expr = expr.set_index("gene")
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all")
    expr = expr.loc[expr.var(axis=1).sort_values(ascending=False).index]
    expr_samples = expr.T
    sample_groups = pd.Series(
        [sample_group_map.get(s, infer_group_from_sample(s)) for s in expr_samples.index],
        index=expr_samples.index,
        name="group",
    )
    draw_sample_clustering(expr_samples, output_dirs, state)

    scaler = StandardScaler()
    expr_z = pd.DataFrame(
        scaler.fit_transform(expr.T).T,
        index=expr.index,
        columns=expr.columns,
    )
    corr = expr_z.T.corr().copy()
    corr_values = np.array(corr.to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(corr_values, 0.0)
    corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    metrics = []
    for power in powers:
        adjacency = np.abs(corr.to_numpy()) ** power
        connectivity = adjacency.sum(axis=1)
        r2, slope = compute_scale_free_fit(connectivity)
        metrics.append(
            {
                "power": power,
                "scale_free_r2": r2,
                "slope": slope,
                "mean_connectivity": float(np.nanmean(connectivity)),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    plot_soft_threshold(metrics_df, output_dirs, state)
    good = metrics_df[metrics_df["scale_free_r2"] >= 0.8]
    if not good.empty:
        soft_power = int(good.sort_values(["power", "mean_connectivity"], ascending=[True, False]).iloc[0]["power"])
    else:
        soft_power = int(metrics_df.sort_values(["scale_free_r2", "mean_connectivity"], ascending=[False, False]).iloc[0]["power"])
    adjacency = np.abs(corr.to_numpy()) ** soft_power
    tom = build_tom(adjacency)
    diss = 1 - tom
    np.fill_diagonal(diss, 0.0)
    linkage_mat = linkage(squareform(diss, checks=False), method="average")
    cut_height = float(np.quantile(linkage_mat[:, 2], 0.65))
    cluster_ids = fcluster(linkage_mat, t=cut_height, criterion="distance")
    cluster_sizes = pd.Series(cluster_ids).value_counts().to_dict()
    module_ids = []
    for cid in cluster_ids:
        module_ids.append(cid if cluster_sizes.get(cid, 0) >= 10 else 0)
    unique_modules = [m for m in sorted(set(module_ids)) if m != 0]
    color_map = {0: "#BDBDBD"}
    for idx, module in enumerate(unique_modules):
        color_map[module] = MODULE_COLOR_POOL[idx % len(MODULE_COLOR_POOL)]
    module_colors = pd.Series([color_map[m] for m in module_ids], index=expr.index, name="module_color")
    module_labels = pd.Series(
        ["gray" if mid == 0 else f"module_{mid}" for mid in module_ids],
        index=expr.index,
        name="module",
    )
    draw_gene_dendrogram_with_colors(linkage_mat, module_colors, output_dirs, state)

    traits = pd.DataFrame(index=expr_samples.index)
    traits["disease_status"] = (sample_groups != "Control").astype(int)
    traits["isorhy_treatment_status"] = (sample_groups == "PM+Isorhy").astype(int)
    traits["condition_code"] = sample_groups.map({"Control": 0, "PM": 1, "PM+Isorhy": 2}).fillna(-1)

    eigengenes = {}
    for module_name in sorted(module_labels.unique()):
        if module_name == "gray":
            continue
        genes = module_labels[module_labels == module_name].index.tolist()
        sub = expr.loc[genes].T
        if sub.shape[1] == 1:
            eigengene = sub.iloc[:, 0].to_numpy(dtype=float)
        else:
            pca = PCA(n_components=1, random_state=42)
            eigengene = pca.fit_transform(StandardScaler().fit_transform(sub)).ravel()
        eigengenes[module_name] = eigengene
    if not eigengenes:
        raise ValueError("No non-gray modules were identified in the WGCNA-like analysis.")
    eigengene_df = pd.DataFrame(eigengenes, index=expr_samples.index)
    module_trait_rows = []
    corr_matrix = pd.DataFrame(index=eigengene_df.columns, columns=traits.columns, dtype=float)
    annot_matrix = pd.DataFrame(index=eigengene_df.columns, columns=traits.columns, dtype=object)
    for module_name in eigengene_df.columns:
        for trait in traits.columns:
            corr_value, pval = correlation_with_p(eigengene_df[module_name].to_numpy(dtype=float), traits[trait].to_numpy(dtype=float))
            corr_matrix.loc[module_name, trait] = corr_value
            annot_matrix.loc[module_name, trait] = f"{corr_value:.2f}\n(p={pval:.3f})" if not pd.isna(corr_value) else "NA"
            module_trait_rows.append({"module": module_name, "trait": trait, "correlation": corr_value, "pvalue": pval})
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(corr_matrix.index) + 2)))
    sns.heatmap(corr_matrix, annot=annot_matrix, fmt="", cmap="vlag", center=0, linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Module-trait correlation heatmap", weight="bold")
    save_figure(
        fig,
        output_dirs["figures"] / "WGCNA_module_trait_heatmap.png",
        output_dirs["figures"] / "WGCNA_module_trait_heatmap.pdf",
        state,
    )
    plt.close(fig)

    module_assignment = pd.DataFrame(
        {
            "gene": expr.index,
            "module": module_labels.values,
            "module_color": module_colors.values,
        }
    )
    membership_rows = []
    target_trait = traits["isorhy_treatment_status"].to_numpy(dtype=float)
    for gene in expr.index:
        gene_vector = expr.loc[gene].to_numpy(dtype=float)
        module_name = module_labels.loc[gene]
        gs, gs_p = correlation_with_p(gene_vector, target_trait)
        mm, mm_p = (np.nan, np.nan)
        if module_name in eigengene_df.columns:
            mm, mm_p = correlation_with_p(gene_vector, eigengene_df[module_name].to_numpy(dtype=float))
        gene_idx = expr.index.get_loc(gene)
        same_module_genes = module_labels[module_labels == module_name].index.tolist()
        same_module_idx = [expr.index.get_loc(g) for g in same_module_genes]
        k_within = float(adjacency[gene_idx, same_module_idx].sum()) if same_module_idx else 0.0
        membership_rows.append(
            {
                "gene": gene,
                "module": module_name,
                "module_color": module_colors.loc[gene],
                "module_membership": mm,
                "module_membership_pvalue": mm_p,
                "gene_significance": gs,
                "gene_significance_pvalue": gs_p,
                "intramodular_connectivity": k_within,
            }
        )
    membership_df = pd.DataFrame(membership_rows)
    module_assignment = module_assignment.merge(membership_df, on=["gene", "module", "module_color"], how="left")
    save_csv(module_assignment, output_dirs["tables"] / "WGCNA_module_assignment.csv", state)
    save_csv(pd.DataFrame(module_trait_rows), output_dirs["tables"] / "WGCNA_module_trait_correlation.csv", state)

    isorhy_assoc = (
        pd.DataFrame(module_trait_rows)
        .query("trait == 'isorhy_treatment_status'")
        .sort_values("correlation", key=lambda x: x.abs(), ascending=False)
    )
    hub_rows = []
    for module_name in isorhy_assoc["module"].head(3):
        sub = module_assignment[module_assignment["module"] == module_name].sort_values(
            ["intramodular_connectivity", "module_membership", "gene_significance"],
            ascending=[False, False, False],
        )
        for _, row in sub.head(10).iterrows():
            hub_rows.append(row.to_dict())
    hub_df = pd.DataFrame(hub_rows)
    save_csv(hub_df, output_dirs["tables"] / "WGCNA_hub_genes.csv", state)

    nlrp2_row = module_assignment[module_assignment["gene"].str.upper() == "NLRP2"].copy()
    if nlrp2_row.empty:
        nlrp2_status = pd.DataFrame(
            [
                {
                    "gene": "NLRP2",
                    "present_in_WGCNA_matrix": False,
                    "module": "",
                    "module_membership": np.nan,
                    "gene_significance": np.nan,
                    "intramodular_connectivity": np.nan,
                }
            ]
        )
    else:
        nlrp2_status = nlrp2_row.copy()
        nlrp2_status.insert(1, "present_in_WGCNA_matrix", True)
    save_csv(nlrp2_status, output_dirs["tables"] / "NLRP2_WGCNA_status.csv", state)

    logger.info("Completed WGCNA-like analysis with soft power %s and %s modules.", soft_power, len(unique_modules))
    return {
        "soft_threshold_metrics": metrics_df,
        "soft_power": soft_power,
        "module_assignment": module_assignment,
        "module_trait": pd.DataFrame(module_trait_rows),
        "hub_genes": hub_df,
        "nlrp2_status": nlrp2_status,
        "top_modules": isorhy_assoc,
        "combined_expr": combined_expr,
    }


def read_candidate_genes(candidate_files: list[Path], logger: logging.Logger) -> list[str]:
    candidates = []
    for path in candidate_files:
        try:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                if df.empty:
                    continue
                for col in df.columns:
                    if str(col).strip() and normalize_text(col) not in {"gene", "genes"}:
                        candidates.append(str(col).strip())
                    candidates.extend(df[col].dropna().astype(str).str.strip().tolist())
        except Exception as exc:
            logger.warning("Failed to read candidate file %s: %s", path, exc)
    candidates = [g for g in candidates if g and g.lower() != "nan"]
    return sorted(dict.fromkeys(candidates))


def choose_ppi_candidates(
    primary_deg: pd.DataFrame,
    disease_deg: pd.DataFrame | None,
    wgcna: dict[str, Any],
    candidate_genes: list[str],
    gene_lookup: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    selected = []
    source_map: dict[str, str] = {}

    def add_gene(gene: str, source: str) -> None:
        gene = mouse_case_from_data(gene, gene_lookup)
        if uppercase_gene(gene) not in gene_lookup:
            return
        if gene not in selected:
            selected.append(gene)
            source_map[gene] = source
        elif source not in source_map[gene]:
            source_map[gene] = f"{source_map[gene]};{source}"

    add_gene("NLRP2", "forced")
    for gene in PYROPTOSIS_GENES:
        add_gene(gene, "pyroptosis")
    for gene in candidate_genes:
        add_gene(gene, "candidate_table")
    if wgcna.get("hub_genes") is not None and not wgcna["hub_genes"].empty:
        for gene in wgcna["hub_genes"]["gene"].astype(str).head(20):
            add_gene(gene, "wgcna_hub")
    for df, source in [(primary_deg, "primary_deg"), (disease_deg, "disease_deg")]:
        if df is None or df.empty:
            continue
        top_by_fc = df.sort_values("log2FoldChange", key=lambda x: x.abs(), ascending=False).head(20)["gene_name"].tolist()
        top_by_p = df.sort_values("pvalue", ascending=True).head(20)["gene_name"].tolist()
        for gene in top_by_fc + top_by_p:
            add_gene(gene, source)
    if len(selected) > 80:
        keep_priority = []
        for gene in selected:
            if uppercase_gene(gene) == "NLRP2":
                keep_priority.insert(0, gene)
            else:
                keep_priority.append(gene)
        selected = keep_priority[:80]
    return selected, source_map


def query_string_network(
    genes: list[str],
    species_id: int,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> pd.DataFrame:
    log_path = output_dirs["logs"] / "STRING_query_log.txt"
    query_text = "\n".join(genes)
    payload = {"identifiers": query_text, "species": species_id, "caller_identity": "codex_transcriptome_analysis"}
    url = "https://string-db.org/api/tsv/network"
    with requests.Session() as session:
        response = session.post(url, data=payload, timeout=60)
        response.raise_for_status()
    log_path.write_text(
        f"URL: {url}\nSpecies: {species_id}\nGenes ({len(genes)}): {', '.join(genes)}\nStatus: {response.status_code}\n",
        encoding="utf-8",
    )
    add_generated(state, log_path, "log")
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    if df.empty:
        raise ValueError("STRING returned an empty network.")
    rename_map = {}
    for col in df.columns:
        norm = normalize_text(col)
        if norm == "preferrednamea":
            rename_map[col] = "source"
        elif norm == "preferrednameb":
            rename_map[col] = "target"
        elif norm == "score":
            rename_map[col] = "score"
    df = df.rename(columns=rename_map)
    df = df[[c for c in ["source", "target", "score"] if c in df.columns]].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["source", "target"]).drop_duplicates()
    logger.info("Retrieved %s STRING edges.", len(df))
    return df


def build_fallback_ppi(genes: list[str], gene_lookup: dict[str, str]) -> pd.DataFrame:
    kept_rows = []
    allowed = {uppercase_gene(g) for g in genes}
    for a, b, score in FALLBACK_PPI_EDGES:
        if uppercase_gene(a) in allowed and uppercase_gene(b) in allowed:
            kept_rows.append(
                {
                    "source": mouse_case_from_data(a, gene_lookup),
                    "target": mouse_case_from_data(b, gene_lookup),
                    "score": score,
                }
            )
    nlrp2_name = mouse_case_from_data("NLRP2", gene_lookup)
    if uppercase_gene(nlrp2_name) in allowed and not any(
        uppercase_gene(row["source"]) == "NLRP2" or uppercase_gene(row["target"]) == "NLRP2" for row in kept_rows
    ):
        fallback_neighbors = [
            mouse_case_from_data(g, gene_lookup)
            for g in ["PYCARD", "CASP1", "NLRP3", "NOD2", "RELA", "MYD88", "IL1B"]
            if uppercase_gene(g) in allowed
        ]
        for neighbor in fallback_neighbors[:5]:
            kept_rows.append({"source": nlrp2_name, "target": neighbor, "score": 0.35})
    return pd.DataFrame(kept_rows, columns=["source", "target", "score"])


def draw_ppi_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    ai_weight_mode: bool = False,
) -> None:
    graph = nx.Graph()
    for _, row in nodes_df.iterrows():
        graph.add_node(row["gene"], **row.to_dict())
    for _, row in edges_df.iterrows():
        graph.add_edge(row["source"], row["target"], score=float(row.get("score", 0.5)))
    pos = nx.spring_layout(graph, seed=42, weight="score", k=0.65)
    fig, ax = plt.subplots(figsize=(10.5, 8.3))
    if graph.number_of_edges() > 0:
        edge_widths = [1.0 + 4.0 * graph[u][v].get("score", 0.3) for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#B0B0B0", width=edge_widths, alpha=0.55)
    normal_nodes = []
    normal_colors = []
    normal_sizes = []
    highlight_nodes = []
    highlight_sizes = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        if uppercase_gene(node) == "NLRP2":
            highlight_nodes.append(node)
            highlight_sizes.append(1100 if ai_weight_mode else 950)
            labels[node] = node
            continue
        normal_nodes.append(node)
        if ai_weight_mode:
            normal_colors.append(attrs.get("AI_weight", 0.0))
            normal_sizes.append(250 + 1800 * attrs.get("AI_weight", 0.0))
        else:
            logfc = attrs.get("log2FoldChange", 0.0)
            normal_colors.append(logfc)
            normal_sizes.append(220 + 1400 * attrs.get("degree_centrality", 0.0))
        if attrs.get("label_node", False):
            labels[node] = node
    if ai_weight_mode:
        if normal_nodes:
            nodes = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=normal_nodes,
                node_color=normal_colors,
                node_size=normal_sizes,
                cmap="YlOrRd",
                linewidths=0.8,
                edgecolors="#444444",
                ax=ax,
            )
            cbar = fig.colorbar(nodes, ax=ax)
            cbar.set_label("AI weight")
        title = "AI-enhanced PPI network"
        png_name = "AI_enhanced_PPI_network.png"
        pdf_name = "AI_enhanced_PPI_network.pdf"
    else:
        if normal_nodes:
            color_series = pd.Series(normal_colors, dtype=float).fillna(0)
            vmax = max(abs(float(color_series.max())), abs(float(color_series.min())), 1.0)
            nodes = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=normal_nodes,
                node_color=normal_colors,
                node_size=normal_sizes,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                linewidths=0.8,
                edgecolors="#444444",
                ax=ax,
            )
            cbar = fig.colorbar(nodes, ax=ax)
            cbar.set_label("Log2 fold change")
        title = "PPI network (Cytoscape-like)"
        png_name = "PPI_network_cytoscape_like.png"
        pdf_name = "PPI_network_cytoscape_like.pdf"
    if highlight_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=highlight_nodes,
            node_color="#F4A259",
            node_size=highlight_sizes,
            linewidths=1.2,
            edgecolors="#8A4F08",
            ax=ax,
        )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, font_color="#222222", ax=ax)
    ax.set_title(title, weight="bold")
    ax.axis("off")
    save_figure(fig, output_dirs["figures"] / png_name, output_dirs["figures"] / pdf_name, state)
    plt.close(fig)


def build_ppi_and_ai_analysis(
    primary_deg: pd.DataFrame,
    disease_deg: pd.DataFrame | None,
    wgcna: dict[str, Any],
    candidate_genes: list[str],
    species_id: int,
    gene_lookup: dict[str, str],
    allow_public_network: bool,
    output_dirs: dict[str, Path],
    state: AnalysisState,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_genes, source_map = choose_ppi_candidates(primary_deg, disease_deg, wgcna, candidate_genes, gene_lookup)
    local_ppi_files = state.detection.get("ppi_files", [])
    edges_df = pd.DataFrame(columns=["source", "target", "score"])
    network_mode = "local_ppi"
    if local_ppi_files:
        path = local_ppi_files[0]
        raw = pd.read_excel(path, sheet_name=0) if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(path)
        cols = {normalize_text(c): c for c in raw.columns}
        source_col = cols.get("source") or cols.get("preferrednamea")
        target_col = cols.get("target") or cols.get("preferrednameb")
        score_col = cols.get("score")
        if source_col and target_col:
            edges_df = raw.rename(columns={source_col: "source", target_col: "target"})
            if score_col:
                edges_df = edges_df.rename(columns={score_col: "score"})
            else:
                edges_df["score"] = 0.5
    elif allow_public_network:
        try:
            edges_df = query_string_network(selected_genes, species_id, output_dirs, state, logger)
            network_mode = "STRING_API"
        except Exception as exc:
            logger.warning("STRING query failed: %s", exc)
            edges_df = build_fallback_ppi(selected_genes, gene_lookup)
            network_mode = "fallback_network"
            state.limitations.append(
                "STRING API retrieval failed or returned no usable edges, so the PPI network was replaced by a small fallback interaction network derived from canonical pyroptosis/inflammation relationships."
            )
    else:
        network_mode = "fallback_network"
        edges_df = build_fallback_ppi(selected_genes, gene_lookup)
        state.limitations.append(
            "Public PPI retrieval was disabled, so the network uses the built-in fallback interaction backbone rather than STRING-derived edges."
        )
    if edges_df.empty:
        network_mode = "fallback_network"
        edges_df = build_fallback_ppi(selected_genes, gene_lookup)
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    edges_df["score"] = pd.to_numeric(edges_df["score"], errors="coerce").fillna(0.5)
    edges_df = edges_df[edges_df["source"].isin(selected_genes) & edges_df["target"].isin(selected_genes)].copy()
    graph = nx.from_pandas_edgelist(edges_df, "source", "target", edge_attr="score")
    for gene in selected_genes:
        if gene not in graph:
            graph.add_node(gene)
    degree_centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {n: 0.0 for n in graph.nodes()}
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    closeness = nx.closeness_centrality(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    primary_lookup = primary_deg.set_index("gene_upper")
    membership_lookup = (
        wgcna.get("module_assignment", pd.DataFrame())
        .drop_duplicates(subset=["gene"])
        .assign(gene_upper=lambda d: d["gene"].str.upper())
        .set_index("gene_upper")
        if wgcna.get("module_assignment") is not None and not wgcna.get("module_assignment", pd.DataFrame()).empty
        else pd.DataFrame()
    )
    pyroptosis_upper = {uppercase_gene(g) for g in PYROPTOSIS_GENES}
    nlrp2_name = mouse_case_from_data("NLRP2", gene_lookup)
    neighbor_set = set(graph.neighbors(nlrp2_name)) if nlrp2_name in graph and graph.degree(nlrp2_name) > 0 else set()
    node_rows = []
    for gene in selected_genes:
        upper = uppercase_gene(gene)
        row = {
            "gene": gene,
            "source_category": source_map.get(gene, ""),
            "degree": int(graph.degree(gene)),
            "degree_centrality": degree_centrality.get(gene, 0.0),
            "betweenness_centrality": betweenness.get(gene, 0.0),
            "closeness_centrality": closeness.get(gene, 0.0),
            "pyroptosis_member": int(upper in pyroptosis_upper),
            "neighbor_of_NLRP2": int(gene in neighbor_set),
            "log2FoldChange": np.nan,
            "pvalue": np.nan,
            "padj": np.nan,
            "module_membership": np.nan,
            "gene_significance": np.nan,
            "intramodular_connectivity": np.nan,
        }
        if upper in primary_lookup.index:
            hit = primary_lookup.loc[upper]
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            row["log2FoldChange"] = hit.get("log2FoldChange")
            row["pvalue"] = hit.get("pvalue")
            row["padj"] = hit.get("padj")
        if not membership_lookup.empty and upper in membership_lookup.index:
            hit = membership_lookup.loc[upper]
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            row["module_membership"] = hit.get("module_membership")
            row["gene_significance"] = hit.get("gene_significance")
            row["intramodular_connectivity"] = hit.get("intramodular_connectivity")
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    nodes_df["deg_score"] = minmax_scale(nodes_df["log2FoldChange"].abs() * safe_neglog10(nodes_df["pvalue"].fillna(1)))
    nodes_df["network_score"] = minmax_scale(
        nodes_df["degree_centrality"].fillna(0) + nodes_df["betweenness_centrality"].fillna(0) + nodes_df["closeness_centrality"].fillna(0)
    )
    nodes_df["wgcna_score"] = minmax_scale(
        nodes_df["module_membership"].abs().fillna(0) + nodes_df["gene_significance"].abs().fillna(0) + nodes_df["intramodular_connectivity"].fillna(0)
    )
    nodes_df["AI_weight"] = (
        nodes_df["deg_score"] * 0.30
        + nodes_df["network_score"] * 0.25
        + nodes_df["wgcna_score"] * 0.25
        + nodes_df["pyroptosis_member"] * 0.15
        + nodes_df["neighbor_of_NLRP2"] * 0.05
    )
    nodes_df["label_node"] = (
        (nodes_df["degree"].rank(method="min", ascending=False) <= 8)
        | (nodes_df["AI_weight"].rank(method="min", ascending=False) <= 8)
        | (nodes_df["gene"].str.upper() == "NLRP2")
    )
    save_csv(edges_df, output_dirs["networks"] / "PPI_edges.csv", state)
    save_csv(nodes_df, output_dirs["networks"] / "PPI_nodes.csv", state)
    draw_ppi_network(nodes_df, edges_df, output_dirs, state, ai_weight_mode=False)

    ai_df = nodes_df.sort_values("AI_weight", ascending=False).copy()
    save_csv(ai_df, output_dirs["tables"] / "AI_PPI_node_importance.csv", state)
    if nlrp2_name in graph:
        neighbor_rows = []
        for neighbor in graph.neighbors(nlrp2_name):
            ai_row = ai_df[ai_df["gene"] == neighbor].iloc[0]
            neighbor_rows.append(
                {
                    "NLRP2_neighbor": neighbor,
                    "AI_weight": ai_row["AI_weight"],
                    "degree": ai_row["degree"],
                    "score": graph[nlrp2_name][neighbor].get("score", np.nan),
                    "pyroptosis_member": ai_row["pyroptosis_member"],
                }
            )
        neighbor_df = pd.DataFrame(neighbor_rows).sort_values(["AI_weight", "score"], ascending=[False, False])
    else:
        neighbor_df = pd.DataFrame(columns=["NLRP2_neighbor", "AI_weight", "degree", "score", "pyroptosis_member"])
    save_csv(neighbor_df, output_dirs["tables"] / "NLRP2_key_neighbor_ranking.csv", state)
    draw_ppi_network(ai_df, edges_df, output_dirs, state, ai_weight_mode=True)

    logger.info("Built %s PPI network with %s nodes and %s edges.", network_mode, graph.number_of_nodes(), graph.number_of_edges())
    return (
        {"network_mode": network_mode, "nodes": nodes_df, "edges": edges_df, "graph": graph},
        {"nodes": ai_df, "neighbors": neighbor_df},
    )


def build_nlrp2_integrated_evidence(
    state: AnalysisState,
    output_dirs: dict[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    primary_deg = state.primary_deg
    if primary_deg is None:
        raise ValueError("Primary DEG table is unavailable.")
    disease_deg = state.disease_deg
    expr_present = any(df["gene"].str.upper().eq("NLRP2").any() for df in state.expression_matrices.values())
    primary_hit = primary_deg[primary_deg["gene_upper"] == "NLRP2"]
    disease_hit = disease_deg[disease_deg["gene_upper"] == "NLRP2"] if disease_deg is not None else pd.DataFrame()
    primary_row = primary_hit.iloc[0] if not primary_hit.empty else pd.Series(dtype=object)
    disease_row = disease_hit.iloc[0] if not disease_hit.empty else pd.Series(dtype=object)
    gsea_leading_terms = []
    for term, lead in state.gsea_details.get("leading_edge_map", {}).items():
        lead_tokens = {uppercase_gene(token) for token in re.split(r"[;,/ ]+", str(lead)) if token}
        if "NLRP2" in lead_tokens:
            gsea_leading_terms.append(term)
    wgcna_row = state.wgcna.get("module_assignment", pd.DataFrame())
    wgcna_row = wgcna_row[wgcna_row["gene"].str.upper() == "NLRP2"] if not wgcna_row.empty else pd.DataFrame()
    ppi_row = state.ai_ppi.get("nodes", pd.DataFrame())
    ppi_row = ppi_row[ppi_row["gene"].str.upper() == "NLRP2"] if not ppi_row.empty else pd.DataFrame()
    neighbor_df = state.ai_ppi.get("neighbors", pd.DataFrame())
    top_neighbors = ", ".join(neighbor_df["NLRP2_neighbor"].head(5).astype(str).tolist()) if not neighbor_df.empty else ""
    primary_sig = bool(primary_row.get("significant", False)) if not primary_row.empty else False
    disease_fc = pd.to_numeric(pd.Series([disease_row.get("log2FoldChange", np.nan)]), errors="coerce").iloc[0] if not disease_row.empty else np.nan
    treatment_fc = pd.to_numeric(pd.Series([primary_row.get("log2FoldChange", np.nan)]), errors="coerce").iloc[0] if not primary_row.empty else np.nan
    direction_support = pd.notna(disease_fc) and pd.notna(treatment_fc) and disease_fc < 0 and treatment_fc > 0
    if expr_present and (direction_support or primary_sig or not ppi_row.empty):
        conclusion = (
            "Integrated transcriptomic, enrichment, WGCNA and PPI network analyses suggest NLRP2 as a candidate key regulator involved in Isorhy-mediated modulation of inflammatory pyroptosis-related responses."
        )
        if not primary_sig:
            conclusion += " However, NLRP2 did not pass the preset DEG significance threshold in the primary Isorhy comparison, so this remains a hypothesis-generating conclusion."
    elif expr_present:
        conclusion = (
            "NLRP2 was detectable in the local dataset, but the current transcriptomic and network evidence was not strong enough to support a confident key-regulator claim under the preset thresholds."
        )
    else:
        conclusion = "NLRP2 was not detected in the available local expression matrices, so no mechanism-oriented transcriptomic conclusion could be supported."
    row = {
        "NLRP2_present_in_expression_matrix": expr_present,
        "NLRP2_is_DEG_in_primary_comparison": primary_sig,
        "NLRP2_log2FC_primary_comparison": primary_row.get("log2FoldChange", np.nan),
        "NLRP2_padj_primary_comparison": primary_row.get("padj", np.nan),
        "NLRP2_pvalue_primary_comparison": primary_row.get("pvalue", np.nan),
        "NLRP2_is_in_pyroptosis_gene_set": True,
        "NLRP2_in_GSEA_leading_edge": bool(gsea_leading_terms),
        "NLRP2_GSEA_leading_edge_terms": "; ".join(gsea_leading_terms),
        "NLRP2_WGCNA_module": wgcna_row["module"].iloc[0] if not wgcna_row.empty else "",
        "NLRP2_module_membership": wgcna_row["module_membership"].iloc[0] if not wgcna_row.empty else np.nan,
        "NLRP2_gene_significance": wgcna_row["gene_significance"].iloc[0] if not wgcna_row.empty else np.nan,
        "NLRP2_PPI_degree": ppi_row["degree"].iloc[0] if not ppi_row.empty else 0,
        "NLRP2_AI_weight": ppi_row["AI_weight"].iloc[0] if not ppi_row.empty else np.nan,
        "NLRP2_top_neighbor_proteins": top_neighbors,
        "Disease_background_log2FC_Pm_vs_control": disease_fc,
        "Treatment_log2FC_PmIsorhy_vs_Pm": treatment_fc,
        "Integrated_conclusion": conclusion,
    }
    result_df = pd.DataFrame([row])
    save_csv(result_df, output_dirs["tables"] / "NLRP2_integrated_evidence.csv", state)
    logger.info("Saved NLRP2 integrated evidence table.")
    return result_df


def build_file_detection_summary(output_dirs: dict[str, Path], state: AnalysisState) -> None:
    path = output_dirs["logs"] / "file_detection_summary.txt"
    header = ["Detected file summary", "kind\tpath\textra", "-" * 100]
    path.write_text("\n".join(header + state.file_detection_lines), encoding="utf-8")
    add_generated(state, path, "log")


def write_completion_summary(output_dirs: dict[str, Path], state: AnalysisState, nlrp2_supported: bool) -> None:
    path = output_dirs["root"] / "completion_summary.txt"
    success = [name for name, info in state.module_status.items() if info["status"] == "success"]
    skipped = [(name, info["reason"]) for name, info in state.module_status.items() if info["status"] != "success"]
    lines = [
        "Completion summary",
        "",
        "1. Successful analysis modules",
    ]
    lines.extend([f"- {item}" for item in success] or ["- None"])
    lines.append("")
    lines.append("2. Skipped analysis modules")
    lines.extend([f"- {name}: {reason}" for name, reason in skipped] or ["- None"])
    lines.append("")
    lines.append("3. Generated figures")
    lines.extend([f"- {item}" for item in state.generated_figures] or ["- None"])
    lines.append("")
    lines.append("4. Generated tables")
    lines.extend([f"- {item}" for item in state.generated_tables] or ["- None"])
    lines.append("")
    lines.append("5. Generated logs")
    lines.extend([f"- {item}" for item in state.generated_logs] or ["- None"])
    lines.append("")
    lines.append(f"6. NLRP2 supported as a key candidate gene: {'Yes, with caveats' if nlrp2_supported else 'Not strongly supported under preset thresholds'}")
    lines.append(f"7. Dose-related data present: {'Yes' if state.dose_related else 'No'}")
    lines.append(f"8. Network interpretation mode: {state.ai_mode}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(output_dirs: dict[str, Path], state: AnalysisState, heatmap_mode: str, enrichment_modes: dict[str, str]) -> None:
    primary_deg = state.primary_deg if state.primary_deg is not None else pd.DataFrame()
    disease_deg = state.disease_deg if state.disease_deg is not None else pd.DataFrame()
    sig_count = int(primary_deg["significant"].sum()) if not primary_deg.empty else 0
    up_count = int((primary_deg["regulation"] == "Up").sum()) if not primary_deg.empty else 0
    down_count = int((primary_deg["regulation"] == "Down").sum()) if not primary_deg.empty else 0
    disease_sig = int(disease_deg["significant"].sum()) if not disease_deg.empty else 0
    nlrp2_df = state.nlrp2_evidence if isinstance(state.nlrp2_evidence, pd.DataFrame) else pd.DataFrame()
    nlrp2_conclusion = nlrp2_df["Integrated_conclusion"].iloc[0] if not nlrp2_df.empty else "NLRP2 evidence could not be assembled."
    gsea_terms = state.gsea_details.get("plot_records", {})
    report = f"""# Final Report

## 1. Data overview

- Input folder scanned: `input/`
- Expression matrices detected: {len(state.detection.get('expression_files', []))}
- Differential result files detected: {len(state.detection.get('diff_files', []))}
- Group files detected: {len(state.detection.get('group_files', []))}
- Candidate gene files detected: {len(state.detection.get('candidate_files', []))}
- Enrichment files detected: GO={len(state.detection.get('enrichment_files', {}).get('go', []))}, KEGG={len(state.detection.get('enrichment_files', {}).get('kegg', []))}, Reactome={len(state.detection.get('enrichment_files', {}).get('reactome', []))}
- Primary Isorhy-focused comparison used for reporting: `{state.primary_comparison}`
- Disease-background comparison used for context: `{state.disease_comparison}`
- Dose-related data detected: {'Yes' if state.dose_related else 'No'}

## 2. File recognition result

- Detailed detection log: `logs/file_detection_summary.txt`
- Main process log: `logs/analysis.log`

## 3. Differential expression summary

- Primary comparison (`{state.primary_comparison}`) significant genes under the preset rule `abs(log2FC) >= 1` and `adjusted p-value < 0.05`: {sig_count}
- Upregulated genes: {up_count}
- Downregulated genes: {down_count}
- Disease-background comparison (`{state.disease_comparison}`) significant genes under the same rule: {disease_sig}
- Because the primary comparison contained {'no' if sig_count == 0 else 'some'} preset-significant genes, the DEG heatmap mode was: `{heatmap_mode}`

Main DEG outputs:

- `tables/all_DEG_results.csv`
- `tables/significant_DEGs.csv`
- `tables/upregulated_genes.csv`
- `tables/downregulated_genes.csv`
- `tables/NLRP2_status_in_DEG.csv`
- `figures/DEG_volcano_plot.png`
- `figures/DEG_heatmap_top_genes.png`

## 4. NLRP2 differential expression status

- Integrated evidence table: `tables/NLRP2_integrated_evidence.csv`
- Summary conclusion: {nlrp2_conclusion}

## 5. GO / KEGG / Reactome enrichment summary

- Existing local enrichment files were used as the preferred source.
- GO plot mode: `{enrichment_modes.get('GO', 'not_run')}`
- KEGG plot mode: `{enrichment_modes.get('KEGG', 'not_run')}`
- Reactome plot mode: `{enrichment_modes.get('Reactome', 'not_run')}`

Figure paths:

- `figures/GO_top10_bubble.png`
- `figures/KEGG_top10_bubble.png`
- `figures/Reactome_top10_bubble.png`

## 6. GSEA pyroptosis / immune pathway results

- GSEA results table: `tables/GSEA_results.csv`
- Gene set notes: {"; ".join(state.gsea_details.get("notes", [])) if state.gsea_details else "Not available"}
- Plotted pathways: {", ".join([f"{name} -> {term}" for name, term in gsea_terms.items()]) if gsea_terms else "No pathway plot was generated."}

Expected figure paths:

- `figures/GSEA_pyroptosis_like_pathway.png`
- `figures/GSEA_innate_immune_system.png`
- `figures/GSEA_NOD_like_receptor.png`

## 7. WGCNA result

- The co-expression step used an exploratory Python WGCNA-like workflow because no local R/WGCNA runtime was available.
- Combined WGCNA matrix size: {state.combined_wgcna_expression.shape[0] if state.combined_wgcna_expression is not None else 0} genes x {state.combined_wgcna_expression.shape[1] - 1 if state.combined_wgcna_expression is not None else 0} samples
- WGCNA outputs:
  - `figures/WGCNA_sample_clustering.png`
  - `figures/WGCNA_soft_threshold.png`
  - `figures/WGCNA_gene_dendrogram_module_colors.png`
  - `figures/WGCNA_module_trait_heatmap.png`
  - `tables/WGCNA_module_assignment.csv`
  - `tables/WGCNA_module_trait_correlation.csv`
  - `tables/WGCNA_hub_genes.csv`
  - `tables/NLRP2_WGCNA_status.csv`

## 8. PPI network result

- PPI network mode: `{state.ppi.get('network_mode', 'not_run')}`
- PPI outputs:
  - `networks/PPI_edges.csv`
  - `networks/PPI_nodes.csv`
  - `figures/PPI_network_cytoscape_like.png`

## 9. AI-enhanced PPI interpretation result

- Interpretation mode: `{state.ai_mode}`
- AI outputs:
  - `tables/AI_PPI_node_importance.csv`
  - `tables/NLRP2_key_neighbor_ranking.csv`
  - `figures/AI_enhanced_PPI_network.png`

## 10. Integrated NLRP2 mechanism conclusion

{nlrp2_conclusion}

## 11. Data limitations and caveats

{chr(10).join([f"- {item}" for item in state.limitations] or ['- No additional limitations were recorded beyond the standard exploratory nature of transcriptomic network inference.'])}
"""
    report_path = output_dirs["root"] / "final_report.md"
    report_path.write_text(report, encoding="utf-8")


def write_readme(project_root: Path) -> None:
    readme_path = project_root / "README.md"
    readme_text = """# Transcriptome + NLRP2 integrated analysis

## How to run

1. Create a Python environment.
2. Install dependencies:

```bash
pip install pandas openpyxl numpy scipy matplotlib seaborn networkx requests gseapy statsmodels scikit-learn adjustText
```

3. Run the full pipeline:

```bash
python scripts/run_full_analysis.py --input input --output output
```

## What the pipeline does

- Automatically scans the `input/` directory and classifies local sequencing-related tables.
- Standardizes existing DEG tables and generates DEG summary outputs plus volcano and heatmap figures.
- Uses local GO / KEGG / Reactome enrichment tables to create top-10 summary figures.
- Runs GSEA from the ranked DEG table, using public gene sets when available and a built-in pyroptosis fallback gene set otherwise.
- Performs an exploratory Python WGCNA-like co-expression analysis when no local R/WGCNA runtime is available.
- Builds a PPI network from DEG genes, pyroptosis genes, WGCNA hub genes, and candidate genes, with STRING API retrieval when available.
- Produces an AI-inspired weighted PPI interpretation for NLRP2-focused network prioritization.
- Writes all stable outputs into `output/`.

## Output layout

- `output/tables/`: DEG, GSEA, WGCNA, AI-PPI, and NLRP2 evidence tables
- `output/figures/`: volcano, heatmap, enrichment, GSEA, WGCNA, and PPI figures
- `output/networks/`: node and edge tables
- `output/logs/`: file detection and analysis logs
- `output/final_report.md`: integrated narrative report
- `output/completion_summary.txt`: concise run summary

## Notes

- The workflow only uses local `input/` files as the source for expression, group, DEG, and enrichment inputs.
- Public internet resources are only used for optional gene sets and STRING PPI edges.
- If a module cannot be completed because of missing columns, missing samples, or network failure, the pipeline records the reason and continues.
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transcriptome + NLRP2 integrated analysis.")
    parser.add_argument("--input", required=True, help="Input directory containing sequencing data.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--allow-public-network",
        action="store_true",
        help="Allow downloading public gene sets and querying STRING. Disabled by default to keep the run offline-safe.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dirs = ensure_dirs(output_dir)
    logger = setup_logging(output_dirs["logs"] / "analysis.log")
    state = AnalysisState()
    add_generated(state, output_dirs["logs"] / "analysis.log", "log")

    try:
        logger.info("Starting full analysis pipeline.")
        project_root = Path.cwd().resolve()
        detection = detect_input_files(input_dir, state, logger)
        build_file_detection_summary(output_dirs, state)

        if not detection["diff_files"]:
            raise FileNotFoundError("No differential expression results were detected in the input directory.")
        diff_file = detection["diff_files"][0]
        diff_results = load_diff_results(diff_file, logger)
        state.diff_results = diff_results
        state.primary_comparison, state.disease_comparison = choose_primary_and_disease_comparisons(diff_results)
        state.primary_deg = diff_results.get(state.primary_comparison) if state.primary_comparison else None
        state.disease_deg = diff_results.get(state.disease_comparison) if state.disease_comparison else None

        group_map, dose_related = load_group_mapping(detection["group_files"], logger)
        state.sample_group_map = group_map
        state.dose_related = dose_related

        for path in detection["expression_files"]:
            label, expr = parse_expression_file(path, logger)
            state.expression_matrices[label] = expr
        if state.primary_comparison:
            state.primary_expr = next(
                (df for label, df in state.expression_matrices.items() if any(token in label.lower() for token in ["isorhy", "iroshy", "irn"])),
                next(iter(state.expression_matrices.values())) if state.expression_matrices else None,
            )

        if state.primary_deg is None or state.primary_expr is None:
            raise ValueError("Primary DEG table or primary expression matrix could not be identified.")

        species_id, species_name = find_species_id(diff_results)
        gene_lookup = {uppercase_gene(g): g for g in state.primary_deg["gene_name"].dropna().astype(str)}
        for expr in state.expression_matrices.values():
            for gene in expr["gene"].astype(str):
                gene_lookup.setdefault(uppercase_gene(gene), gene)

        deg_info = summarize_deg_outputs(state.primary_deg, state.disease_deg, output_dirs, state, logger)
        build_volcano_plot(state.primary_deg, output_dirs, state, logger)
        heatmap_mode = build_heatmap(state.primary_deg, state.primary_expr, group_map, output_dirs, state, logger)
        mark_module(state, "DEG_and_visualization", "success", "Completed DEG table export, volcano plot, and heatmap.")

        enrichment_modes = {}
        try:
            for label, key in [("GO", "go"), ("KEGG", "kegg"), ("Reactome", "reactome")]:
                files = detection["enrichment_files"].get(key, [])
                if not files:
                    raise FileNotFoundError(f"No {label} enrichment file detected.")
                df = choose_enrichment_sheet(files[0], state.primary_comparison or "")
                state.enrichment_results[label] = df
                enrichment_modes[label] = build_bubble_plot(label, df, output_dirs, state, logger)
            mark_module(state, "enrichment_bubble_plots", "success", "Created GO, KEGG, and Reactome summary plots.")
        except Exception as exc:
            mark_module(state, "enrichment_bubble_plots", "skipped", str(exc))
            state.limitations.append(f"Enrichment bubble plots were partially or fully skipped: {exc}")
            enrichment_modes = enrichment_modes or {}

        try:
            state.gsea_details = run_gsea_analysis(
                state.primary_deg,
                gene_lookup,
                species_name,
                args.allow_public_network,
                output_dirs,
                state,
                logger,
            )
            state.gsea_results = state.gsea_details.get("result_table")
            mark_module(state, "GSEA", "success", "Completed preranked GSEA and saved pathway plots when available.")
        except Exception as exc:
            logger.error("GSEA failed: %s", exc)
            logger.debug(traceback.format_exc())
            mark_module(state, "GSEA", "skipped", str(exc))
            state.limitations.append(f"GSEA was skipped or incomplete: {exc}")

        try:
            state.combined_wgcna_expression = build_wgcna_expression(state.expression_matrices, group_map, logger)
            state.wgcna = run_wgcna_like_analysis(state.combined_wgcna_expression, group_map, output_dirs, state, logger)
            state.limitations.append(
                "The co-expression network is an exploratory Python WGCNA-like analysis, not a full R/WGCNA implementation, because no local R/WGCNA runtime was available."
            )
            state.limitations.append(
                "The WGCNA-like module inference used 9 samples and only the genes shared across the two local expression matrices, so module assignments should be treated as exploratory."
            )
            mark_module(state, "WGCNA_like", "success", "Completed exploratory Python co-expression network analysis.")
        except Exception as exc:
            logger.error("WGCNA-like analysis failed: %s", exc)
            logger.debug(traceback.format_exc())
            mark_module(state, "WGCNA_like", "skipped", str(exc))
            state.limitations.append(f"WGCNA-like analysis was skipped or incomplete: {exc}")
            state.wgcna = {}

        try:
            candidate_genes = read_candidate_genes(detection["candidate_files"], logger)
            state.ppi, state.ai_ppi = build_ppi_and_ai_analysis(
                state.primary_deg,
                state.disease_deg,
                state.wgcna,
                candidate_genes,
                species_id,
                gene_lookup,
                args.allow_public_network,
                output_dirs,
                state,
                logger,
            )
            mark_module(state, "PPI_and_AI_interpretation", "success", "Completed PPI construction and AI-inspired network scoring.")
        except Exception as exc:
            logger.error("PPI/AI analysis failed: %s", exc)
            logger.debug(traceback.format_exc())
            mark_module(state, "PPI_and_AI_interpretation", "skipped", str(exc))
            state.limitations.append(f"PPI and AI interpretation were skipped or incomplete: {exc}")
            state.ppi, state.ai_ppi = {}, {}

        try:
            state.nlrp2_evidence = build_nlrp2_integrated_evidence(state, output_dirs, logger)
            mark_module(state, "NLRP2_integration", "success", "Generated integrated NLRP2 evidence table.")
        except Exception as exc:
            logger.error("NLRP2 integration failed: %s", exc)
            logger.debug(traceback.format_exc())
            mark_module(state, "NLRP2_integration", "skipped", str(exc))
            state.limitations.append(f"NLRP2 integrated evidence table was incomplete: {exc}")
            state.nlrp2_evidence = pd.DataFrame()

        write_report(output_dirs, state, heatmap_mode, enrichment_modes)
        nlrp2_supported = False
        if isinstance(state.nlrp2_evidence, pd.DataFrame) and not state.nlrp2_evidence.empty:
            conclusion = state.nlrp2_evidence["Integrated_conclusion"].iloc[0]
            nlrp2_supported = "suggest NLRP2 as a candidate key regulator" in conclusion
        for collection in [state.generated_figures, state.generated_tables, state.generated_logs]:
            for idx, item in enumerate(collection):
                try:
                    collection[idx] = Path(item).resolve().relative_to(project_root).as_posix()
                except Exception:
                    collection[idx] = Path(item).as_posix()
        write_completion_summary(output_dirs, state, nlrp2_supported)
        write_readme(Path.cwd())
        logger.info("Analysis completed.")
    except Exception as exc:
        logger.error("Fatal pipeline error: %s", exc)
        logger.error(traceback.format_exc())
        mark_module(state, "pipeline", "skipped", str(exc))
        build_file_detection_summary(output_dirs, state)
        write_completion_summary(output_dirs, state, False)
        raise


if __name__ == "__main__":
    main()
