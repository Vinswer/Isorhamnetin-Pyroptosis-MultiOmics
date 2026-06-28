#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


KEY_GENES = ["NLRP2", "NLRP3", "CASP1", "GSDMD", "IL1B", "IL18", "PYCARD", "AIM2"]
PYROPTOSIS_GENES = [
    "NLRP2", "NLRP3", "NLRP1", "NLRC4", "AIM2", "PYCARD", "CASP1", "CASP4",
    "CASP5", "CASP8", "GSDMD", "GSDME", "IL1B", "IL18", "NOD1", "NOD2",
    "MEFV", "NEK7", "TXNIP", "TLR4", "MYD88", "RELA", "NFKB1",
]
UP_COLOR = "#B53A3A"
DOWN_COLOR = "#356DA5"
NS_COLOR = "#C8CCD3"
HEATMAP_CMAP = sns.color_palette("vlag", as_cmap=True)


def save_multi(fig: Any, base_path: Path) -> None:
    for ext in [".png", ".pdf", ".svg"]:
        kwargs = {"dpi": 600} if ext == ".png" else {}
        fig.savefig(base_path.with_suffix(ext), bbox_inches="tight", **kwargs)


def uppercase_gene(text: Any) -> str:
    return str(text).strip().upper()


def safe_neglog10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    min_positive = finite.min() if finite.size else 1e-300
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, min_positive / 10.0)
    return -np.log10(arr)


def parse_expression_matrices(input_dir: Path) -> dict[str, pd.DataFrame]:
    expr = {}
    for path in input_dir.rglob("*.xlsx"):
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception:
                continue
            if df.empty:
                continue
            cols = [str(c) for c in df.columns]
            if not cols or cols[0] != "group":
                continue
            first_vals = df.iloc[:, 0].astype(str).str.lower().tolist()[:3]
            if "sample" not in first_vals:
                continue
            sample_names = [str(x).strip() for x in df.iloc[0, 1:].tolist()]
            data = df.iloc[1:].copy()
            data.rename(columns={cols[0]: "gene"}, inplace=True)
            data.columns = ["gene"] + sample_names
            data["gene"] = data["gene"].astype(str).str.strip()
            data = data[(data["gene"] != "") & (data["gene"].str.lower() != "nan")].copy()
            for c in sample_names:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            expr[path.stem] = data
    return expr


def load_all_diff_results(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in input_dir.rglob("*.xlsx"):
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception:
                continue
            if df.empty or "log2FoldChange" not in map(str, df.columns):
                continue
            if "gene_name" not in df.columns:
                gene_col = next((c for c in df.columns if "gene" in str(c).lower()), df.columns[0])
                df = df.rename(columns={gene_col: "gene_name"})
            df["comparison"] = sheet
            df["gene_upper"] = df["gene_name"].astype(str).str.upper()
            for col in ["log2FoldChange", "pvalue", "padj"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            p_col = "padj" if "padj" in df.columns else "pvalue"
            df["significant"] = (df["log2FoldChange"].abs() >= 1) & (df[p_col] < 0.05)
            df["regulation"] = np.where(
                df["significant"] & (df["log2FoldChange"] >= 1),
                "Up",
                np.where(df["significant"] & (df["log2FoldChange"] <= -1), "Down", "NS"),
            )
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No differential result workbook detected in input/.")
    return pd.concat(frames, ignore_index=True)


def infer_group(sample: str) -> str:
    s = str(sample).lower()
    if "control" in s or s.startswith("con"):
        return "Control"
    if "isorhy" in s or "iroshy" in s or "irn" in s:
        return "PM+Isorhy"
    if "pm" in s or "multocida" in s:
        return "PM"
    return "Unknown"


def draw_volcano(df: pd.DataFrame, comparison: str, base_path: Path, publication_only: bool) -> str:
    d = df[df["comparison"] == comparison].copy()
    p_col = "padj" if d["padj"].notna().any() else "pvalue"
    d["neglog10"] = safe_neglog10(d[p_col])
    d["plot_group"] = d["regulation"]
    if not publication_only:
        label_mode = "full"
    else:
        label_mode = "publication"
    fig, ax = plt.subplots(figsize=(7.6, 6.7))
    for grp, color, alpha, size in [("NS", NS_COLOR, 0.45, 18), ("Down", DOWN_COLOR, 0.85, 24), ("Up", UP_COLOR, 0.85, 24)]:
        sub = d[d["plot_group"] == grp]
        ax.scatter(sub["log2FoldChange"], sub["neglog10"], s=size, c=color, alpha=alpha, edgecolors="none")
    ax.axvline(-1, linestyle="--", color="#7A7A7A", linewidth=0.9)
    ax.axvline(1, linestyle="--", color="#7A7A7A", linewidth=0.9)
    ax.axhline(-math.log10(0.05), linestyle="--", color="#7A7A7A", linewidth=0.9)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel(f"-log10({'adjusted p-value' if p_col == 'padj' else 'p-value'})")
    title = comparison
    if comparison == "Pm+Iroshy vs Pm" and not d["significant"].any():
        title += " (exploratory)"
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    texts = []
    key_hits = d[d["gene_upper"].isin(KEY_GENES)].copy()
    key_hits = key_hits.sort_values(["pvalue", "log2FoldChange"], ascending=[True, False])
    if publication_only:
        top_extra = d.sort_values(["pvalue", "log2FoldChange"], ascending=[True, False]).head(6)
    else:
        top_extra = d.sort_values(["pvalue", "log2FoldChange"], ascending=[True, False]).head(12)
    label_df = pd.concat([key_hits, top_extra], ignore_index=True).drop_duplicates(subset=["gene_name"])
    if publication_only:
        label_df = label_df.head(10)
    for _, row in label_df.iterrows():
        texts.append(ax.text(row["log2FoldChange"], row["neglog10"], row["gene_name"], fontsize=10, color="#333333"))
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))
    save_multi(fig, base_path)
    plt.close(fig)
    return label_mode


def load_local_target_gene_sets(output_dir: Path) -> dict[str, list[str]]:
    gmt_dir = output_dir / "networks" / "downloaded_gene_sets"
    libraries = {}
    for gmt in gmt_dir.glob("*.gmt"):
        lib = {}
        with gmt.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                lib[parts[0]] = [str(g).strip() for g in parts[2:] if str(g).strip()]
        libraries[gmt.name] = lib
    def pick(lib: dict[str, list[str]], patterns: list[str]) -> list[str]:
        for term, genes in lib.items():
            low = term.lower()
            if any(p in low for p in patterns):
                return genes
        return []
    return {
        "GO_PYROPTOSIS": pick(libraries.get("GO_Biological_Process_2021.gmt", {}), ["pyroptosis"]),
        "REACTOME_INNATE_IMMUNE_SYSTEM": pick(libraries.get("Reactome_2022.gmt", {}), ["innate immune system"]),
        "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY": pick(libraries.get("KEGG_2013.gmt", {}), ["nod-like receptor signaling pathway", "nod like receptor signaling pathway"]),
        "REACTOME_INFLAMMASOMES": pick(libraries.get("Reactome_2022.gmt", {}), ["inflammasome"]),
        "CUSTOM_PYROPTOSIS_GENE_SET": PYROPTOSIS_GENES,
    }


def empirical_pvalue(rank_series: pd.Series, gene_set_upper: set[str], observed_es: float, size: int, n_perm: int = 1000) -> float:
    if size <= 0 or len(rank_series) < size:
        return 1.0
    rng = np.random.default_rng(42)
    genes = rank_series.index.to_numpy()
    values = rank_series.to_numpy(dtype=float)
    abs_values = np.abs(values)
    obs = abs(observed_es)
    extreme = 0
    for _ in range(n_perm):
        idx = np.sort(rng.choice(len(genes), size=size, replace=False))
        hit_weights = abs_values[idx]
        hit_weights = hit_weights / hit_weights.sum() if hit_weights.sum() > 0 else np.repeat(1.0 / len(idx), len(idx))
        hit_map = dict(zip(idx, hit_weights))
        miss_weight = 1.0 / max(len(genes) - len(idx), 1)
        current = 0.0
        best = 0.0
        for i in range(len(genes)):
            if i in hit_map:
                current += hit_map[i]
            else:
                current -= miss_weight
            if abs(current) > abs(best):
                best = current
        if abs(best) >= obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def draw_exploratory_gsea_figure(rank_series: pd.Series, term_label: str, gene_set: list[str], base_path: Path) -> dict[str, Any]:
    gene_set_upper = {uppercase_gene(g) for g in gene_set if str(g).strip()}
    labels = rank_series.index.astype(str).tolist()
    values = rank_series.to_numpy(dtype=float)
    hits = [idx for idx, gene in enumerate(labels) if uppercase_gene(gene) in gene_set_upper]
    n = len(labels)
    if hits:
        abs_vals = np.abs(values)
        hit_weights = abs_vals[hits]
        hit_weights = hit_weights / hit_weights.sum() if hit_weights.sum() > 0 else np.repeat(1.0 / len(hits), len(hits))
        hit_map = dict(zip(hits, hit_weights))
        miss_weight = 1.0 / max(n - len(hits), 1)
        running = []
        current = 0.0
        for idx in range(n):
            if idx in hit_map:
                current += hit_map[idx]
            else:
                current -= miss_weight
            running.append(current)
        running = np.array(running, dtype=float)
        peak_idx = int(np.argmax(np.abs(running)))
        es = float(running[peak_idx])
        leading = [labels[i] for i in hits if i <= peak_idx] if es >= 0 else [labels[i] for i in hits if i >= peak_idx]
        p_emp = empirical_pvalue(rank_series, gene_set_upper, es, len(hits), n_perm=1000)
        note = "formal NES/FDR unavailable under current constraints"
    else:
        running = np.cumsum(np.repeat(-1.0 / n, n))
        peak_idx = 0
        es = -1.0
        leading = []
        p_emp = 1.0
        note = "no local gene overlap under current constraints"
    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.6, 1.4], hspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    x = np.arange(1, n + 1)
    ax1.plot(x, running, color="#C13C3C", linewidth=1.8)
    ax1.axhline(0, color="#777777", linewidth=0.8)
    ax1.set_ylabel("Running ES")
    title_suffix = "exploratory" if hits else "limited-overlap exploratory"
    ax1.set_title(f"{term_label} ({title_suffix})")
    text = f"ES={es:.3f}\nOverlap={len(hits)}\nEmpirical p={p_emp:.3f}\n{note}"
    ax1.text(0.02, 0.98, text, transform=ax1.transAxes, va="top", ha="left", fontsize=10)
    if hits:
        ax2.vlines([h + 1 for h in hits], 0, 1, color="black", linewidth=0.8)
    ax2.set_yticks([])
    ax2.set_ylabel("Hits")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    colors = np.where(values >= 0, "#D9D9D9", "#EFEFEF")
    ax3.bar(x, values, width=1.0, color="#BDBDBD", edgecolor="none")
    ax3.axhline(0, color="#777777", linewidth=0.8)
    ax3.set_ylabel("Rank metric")
    ax3.set_xlabel("Rank in ordered dataset")
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    save_multi(fig, base_path)
    plt.close(fig)
    return {
        "Term": term_label,
        "ES": es,
        "Overlap_gene_count": len(hits),
        "Leading_genes": ";".join(leading),
        "Empirical_p": p_emp,
        "Note": note,
    }


def draw_heatmaps(expr_mats: dict[str, pd.DataFrame], all_diff: pd.DataFrame, output_dir: Path) -> None:
    primary_expr = next(v for k, v in expr_mats.items() if any(t in k.lower() for t in ["iroshy", "isorhy", "irn"]))
    primary_diff = all_diff[all_diff["comparison"] == "Pm+Iroshy vs Pm"].copy()
    primary_diff["sort_p"] = primary_diff["pvalue"].fillna(1)
    top = primary_diff.sort_values(["sort_p", "log2FoldChange"], ascending=[True, False]).head(25)["gene_name"].astype(str).tolist()
    nlrp2_hit = primary_diff.loc[primary_diff["gene_upper"] == "NLRP2", "gene_name"]
    if not nlrp2_hit.empty and nlrp2_hit.iloc[0] not in top:
        top.append(nlrp2_hit.iloc[0])
    top = [g for g in top if g in set(primary_expr["gene"])]
    expr = primary_expr[primary_expr["gene"].isin(top)].set_index("gene")
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    expr = expr.loc[~expr.index.duplicated()]
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sample_groups = [infer_group(c) for c in z.columns]
    palette = {"Control": "#6BAED6", "PM": "#D95F5F", "PM+Isorhy": "#4C9F70", "Unknown": "#BDBDBD"}
    col_colors = pd.Series(sample_groups, index=z.columns).map(palette)
    for suffix, matrix in [("publication", z.iloc[:25]), ("supplementary", z)]:
        cg = sns.clustermap(
            matrix,
            cmap=HEATMAP_CMAP,
            center=0,
            col_colors=col_colors.loc[matrix.columns],
            linewidths=0.05,
            figsize=(9.5, 9.2 if suffix == "publication" else 11.5),
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "Z-score"},
        )
        cg.fig.suptitle(f"Top DEG heatmap ({suffix})", y=1.02, fontsize=17)
        base = output_dir / f"heatmap_DEG_top_genes_{suffix}"
        save_multi(cg.fig, base)
        plt.close(cg.fig)


def write_formal_gsea_r_script(script_path: Path) -> None:
    content = r'''args <- commandArgs(trailingOnly=TRUE)
rank_file <- args[1]
gmt_dir <- args[2]
out_csv <- args[3]
out_json <- args[4]
suppressPackageStartupMessages({
  library(fgsea)
  library(msigdbr)
  library(jsonlite)
})
rank_df <- read.csv(rank_file, check.names = FALSE)
rank_vec <- rank_df$ranking_score
names(rank_vec) <- rank_df$gene_name
rank_vec <- rank_vec[!is.na(rank_vec)]
rank_vec <- sort(rank_vec, decreasing = TRUE)
read_gmt_simple <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  out <- list()
  for (ln in lines) {
    parts <- strsplit(ln, "\t")[[1]]
    if (length(parts) >= 3) {
      term <- parts[1]
      genes <- unique(parts[3:length(parts)])
      out[[term]] <- genes
    }
  }
  out
}
reactome <- read_gmt_simple(file.path(gmt_dir, "Reactome_2022.gmt"))
kegg <- read_gmt_simple(file.path(gmt_dir, "KEGG_2013.gmt"))
go_bp <- read_gmt_simple(file.path(gmt_dir, "GO_Biological_Process_2021.gmt"))
go_cc <- read_gmt_simple(file.path(gmt_dir, "GO_Cellular_Component_2021.gmt"))
pick_one <- function(lib, patterns) {
  nm <- names(lib)
  hit <- nm[sapply(tolower(nm), function(x) any(sapply(patterns, function(p) grepl(p, x, fixed = TRUE))))]
  if (length(hit) == 0) return(NULL)
  lib[[hit[1]]]
}
pathways <- list(
  GO_PYROPTOSIS = pick_one(go_bp, c("pyroptosis")),
  REACTOME_INFLAMMASOMES = pick_one(reactome, c("inflammasome")),
  REACTOME_INNATE_IMMUNE_SYSTEM = pick_one(reactome, c("innate immune system")),
  KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY = pick_one(kegg, c("nod-like receptor signaling pathway", "nod like receptor signaling pathway")),
  REACTOME_INTERLEUKIN_1_SIGNALING = pick_one(reactome, c("interleukin-1", "interleukin 1"))
)
pathways <- pathways[!sapply(pathways, is.null)]
if (length(pathways) == 0) stop("No target pathways were found in the local GMT resources.")
fg <- fgsea(pathways = pathways, stats = rank_vec, minSize = 5, maxSize = 5000, nperm = 5000)
fg <- as.data.frame(fg)
fg$leadingEdge_str <- vapply(fg$leadingEdge, function(x) paste(x, collapse = ";"), character(1))
write.csv(fg, out_csv, row.names = FALSE)
detail_list <- list()
for (i in seq_len(nrow(fg))) {
  term <- fg$pathway[i]
  ps <- pathways[[term]]
  prep <- fgsea::preparePathwaysAndStats(list(tmp = ps), rank_vec, minSize = 1, maxSize = 5000, gseaParam = 1)
  res <- fgsea::calcGseaStat(
    stats = prep$stats,
    selectedStats = prep$selectedStats[[1]],
    returnAllExtremes = TRUE,
    returnLeadingEdge = TRUE,
    scoreType = "std"
  )
  detail_list[[term]] <- list(
    pathway = term,
    NES = fg$NES[i],
    ES = fg$ES[i],
    pval = fg$pval[i],
    padj = fg$padj[i],
    leadingEdge = fg$leadingEdge[[i]],
    selectedStats = names(prep$stats)[prep$selectedStats[[1]]],
    selectedIdx = as.integer(prep$selectedStats[[1]]),
    runningScore = as.numeric(res$res)
  )
}
writeLines(jsonlite::toJSON(detail_list, auto_unbox = TRUE, pretty = TRUE, null = "null"), out_json)
'''
    script_path.write_text(content, encoding="utf-8")


def run_formal_gsea(input_dir: Path, output_dir: Path, figures_pub_dir: Path) -> tuple[bool, str]:
    gmt_dir = output_dir / "networks" / "downloaded_gene_sets"
    required = [
        gmt_dir / "Reactome_2022.gmt",
        gmt_dir / "KEGG_2013.gmt",
        gmt_dir / "GO_Biological_Process_2021.gmt",
        gmt_dir / "GO_Cellular_Component_2021.gmt",
    ]
    if not all(p.exists() for p in required):
        return False, "Local public GMT resources are missing, so formal GSEA cannot be rerun."
    all_diff = load_all_diff_results(input_dir)
    primary = all_diff[all_diff["comparison"] == "Pm+Iroshy vs Pm"].copy()
    primary["ranking_score"] = np.sign(primary["log2FoldChange"].fillna(0)) * safe_neglog10(primary["pvalue"].fillna(primary["padj"]).fillna(1))
    primary = primary[["gene_name", "ranking_score"]].dropna()
    primary["gene_name"] = primary["gene_name"].astype(str).str.strip()
    primary = primary[(primary["gene_name"] != "") & (primary["gene_name"].str.lower() != "nan")]
    primary = primary.sort_values("ranking_score", ascending=False).drop_duplicates(subset=["gene_name"], keep="first")
    rank_csv = output_dir / "tables" / "ranked_gene_list_for_formal_gsea.csv"
    primary.to_csv(rank_csv, index=False, encoding="utf-8-sig")
    r_script = Path("scripts") / "run_formal_gsea.R"
    write_formal_gsea_r_script(r_script)
    out_csv = output_dir / "tables" / "GSEA_results_formal.csv"
    out_json = output_dir / "tables" / "GSEA_results_formal_details.json"
    cmd = [
        r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
        str(r_script.resolve()),
        str(rank_csv.resolve()),
        str(gmt_dir.resolve()),
        str(out_csv.resolve()),
        str(out_json.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    if result.returncode != 0 or not out_csv.exists() or not out_json.exists():
        return False, f"Formal GSEA failed: {result.stderr.strip() or result.stdout.strip()}"
    res_df = pd.read_csv(out_csv)
    if res_df.empty or res_df["NES"].isna().all():
        return False, "Formal GSEA returned no valid NES values."
    details = json.loads(out_json.read_text(encoding="utf-8"))
    plot_map = {
        "GSEA_online_pyroptosis": ["GO_PYROPTOSIS"],
        "GSEA_online_innate_immune_system": ["REACTOME_INNATE_IMMUNE_SYSTEM"],
        "GSEA_online_NOD_like_receptor": ["KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY"],
        "GSEA_online_inflammasome": ["REACTOME_INFLAMMASOMES"],
    }
    rank_series = primary.set_index("gene_name")["ranking_score"]
    for stub, names in plot_map.items():
        term = next((n for n in names if n in details), None)
        if term is None:
            continue
        det = details[term]
        hits = [int(x) for x in det["selectedIdx"]]
        running = [float(x) for x in det["runningScore"]]
        gseaplot(
            term=term,
            hits=hits,
            nes=float(det["NES"]),
            pval=float(det["pval"]),
            fdr=float(det["padj"]),
            RES=running,
            rank_metric=rank_series.to_numpy(dtype=float),
            color="#C13C3C",
            figsize=(7.4, 5.9),
            ofname=str(figures_pub_dir / f"{stub}.png"),
        )
        gseaplot(
            term=term,
            hits=hits,
            nes=float(det["NES"]),
            pval=float(det["pval"]),
            fdr=float(det["padj"]),
            RES=running,
            rank_metric=rank_series.to_numpy(dtype=float),
            color="#C13C3C",
            figsize=(7.4, 5.9),
            ofname=str(figures_pub_dir / f"{stub}.pdf"),
        )
        # convert PDF to SVG via matplotlib redraw for stable vector output
        fig, ax = plt.subplots()
        plt.close(fig)
        gseaplot(
            term=term,
            hits=hits,
            nes=float(det["NES"]),
            pval=float(det["pval"]),
            fdr=float(det["padj"]),
            RES=running,
            rank_metric=rank_series.to_numpy(dtype=float),
            color="#C13C3C",
            figsize=(7.4, 5.9),
            ofname=str(figures_pub_dir / f"{stub}.svg"),
        )
    return True, "Formal GSEA completed successfully with fgsea."


def write_wgcna_r_script(script_path: Path) -> None:
    content = r'''args <- commandArgs(trailingOnly=TRUE)
expr_csv <- args[1]
out_dir <- args[2]
suppressPackageStartupMessages(library(WGCNA))
options(stringsAsFactors = FALSE)
allowWGCNAThreads()
expr_df <- read.csv(expr_csv, check.names = FALSE)
traits <- expr_df[, 1:2]
datExpr <- expr_df[, -(1:2)]
rownames(datExpr) <- expr_df$gene
datExpr <- as.data.frame(t(datExpr))
trait_df <- data.frame(
  sample = rownames(datExpr),
  group = colnames(expr_df)[2]
)
sample_names <- rownames(datExpr)
group <- ifelse(grepl("control", sample_names, ignore.case=TRUE) | grepl("^CON", sample_names, ignore.case=TRUE), "Control",
         ifelse(grepl("isorhy|iroshy|IRN", sample_names, ignore.case=TRUE), "PM_Isorhy", "PM"))
traitMat <- data.frame(
  disease_status = ifelse(group == "Control", 0, 1),
  isorhy_treatment_status = ifelse(group == "PM_Isorhy", 1, 0),
  condition_code = ifelse(group == "Control", 0, ifelse(group == "PM", 1, 2))
)
rownames(traitMat) <- sample_names
sampleTree <- hclust(dist(datExpr), method = "average")
png(file.path(out_dir, "WGCNA_sample_clustering_publication.png"), width=2600, height=1800, res=600)
par(mar=c(6,4,3,2))
plot(sampleTree, main="Sample clustering", xlab="", sub="", cex.lab=1.2, cex.axis=1.0, cex.main=1.3)
dev.off()
pdf(file.path(out_dir, "WGCNA_sample_clustering_publication.pdf"), width=8.2, height=5.5)
par(mar=c(6,4,3,2))
plot(sampleTree, main="Sample clustering", xlab="", sub="", cex.lab=1.2, cex.axis=1.0, cex.main=1.3)
dev.off()
svg(file.path(out_dir, "WGCNA_sample_clustering_publication.svg"), width=8.2, height=5.5)
par(mar=c(6,4,3,2))
plot(sampleTree, main="Sample clustering", xlab="", sub="", cex.lab=1.2, cex.axis=1.0, cex.main=1.3)
dev.off()
powers = c(1:10, seq(12,20,2))
sft = pickSoftThreshold(datExpr, powerVector = powers, verbose = 0, networkType = "signed")
png(file.path(out_dir, "WGCNA_soft_threshold_publication.png"), width=3200, height=1600, res=600)
par(mfrow=c(1,2))
cex1 = 0.9
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], xlab="Soft threshold (power)", ylab="Scale free topology model fit, signed R^2", type="n", main="Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], labels=powers, cex=cex1, col="firebrick")
abline(h=0.80, col="grey60", lty=2)
plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab="Soft threshold (power)", ylab="Mean connectivity", type="n", main="Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1, col="steelblue")
dev.off()
pdf(file.path(out_dir, "WGCNA_soft_threshold_publication.pdf"), width=10, height=4.8)
par(mfrow=c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], xlab="Soft threshold (power)", ylab="Scale free topology model fit, signed R^2", type="n", main="Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], labels=powers, cex=cex1, col="firebrick")
abline(h=0.80, col="grey60", lty=2)
plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab="Soft threshold (power)", ylab="Mean connectivity", type="n", main="Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1, col="steelblue")
dev.off()
svg(file.path(out_dir, "WGCNA_soft_threshold_publication.svg"), width=10, height=4.8)
par(mfrow=c(1,2))
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], xlab="Soft threshold (power)", ylab="Scale free topology model fit, signed R^2", type="n", main="Scale independence")
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], labels=powers, cex=cex1, col="firebrick")
abline(h=0.80, col="grey60", lty=2)
plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab="Soft threshold (power)", ylab="Mean connectivity", type="n", main="Mean connectivity")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1, col="steelblue")
dev.off()
power <- ifelse(any(!is.na(sft$fitIndices[,2]) & -sign(sft$fitIndices[,3])*sft$fitIndices[,2] > 0.8), sft$fitIndices[which.max(-sign(sft$fitIndices[,3])*sft$fitIndices[,2] > 0.8), 1], 6)
net <- blockwiseModules(datExpr, power=power, TOMType="signed", minModuleSize=20, reassignThreshold=0, mergeCutHeight=0.25, numericLabels=FALSE, pamRespectsDendro=FALSE, saveTOMs=FALSE, verbose=0)
moduleColors <- labels2colors(net$colors)
MEs <- orderMEs(net$MEs)
moduleTraitCor <- cor(MEs, traitMat, use = "p")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))
png(file.path(out_dir, "WGCNA_gene_dendrogram_module_colors_publication.png"), width=3200, height=1800, res=600)
plotDendroAndColors(net$dendrograms[[1]], moduleColors[net$blockGenes[[1]]], "Module colors", dendroLabels=FALSE, hang=0.03, addGuide=TRUE, guideHang=0.05, main="Gene dendrogram and module colors")
dev.off()
pdf(file.path(out_dir, "WGCNA_gene_dendrogram_module_colors_publication.pdf"), width=10, height=5.6)
plotDendroAndColors(net$dendrograms[[1]], moduleColors[net$blockGenes[[1]]], "Module colors", dendroLabels=FALSE, hang=0.03, addGuide=TRUE, guideHang=0.05, main="Gene dendrogram and module colors")
dev.off()
svg(file.path(out_dir, "WGCNA_gene_dendrogram_module_colors_publication.svg"), width=10, height=5.6)
plotDendroAndColors(net$dendrograms[[1]], moduleColors[net$blockGenes[[1]]], "Module colors", dendroLabels=FALSE, hang=0.03, addGuide=TRUE, guideHang=0.05, main="Gene dendrogram and module colors")
dev.off()
textMatrix <- paste(signif(moduleTraitCor, 2), "\n(", signif(moduleTraitPvalue, 1), ")", sep="")
png(file.path(out_dir, "WGCNA_module_trait_heatmap_publication.png"), width=2600, height=2200, res=600)
labeledHeatmap(Matrix = moduleTraitCor, xLabels = names(traitMat), yLabels = names(MEs), ySymbols = names(MEs), colorLabels = FALSE, colors = blueWhiteRed(50), textMatrix = textMatrix, setStdMargins = FALSE, cex.text = 0.8, zlim = c(-1,1), main = "Module-trait relationships")
dev.off()
pdf(file.path(out_dir, "WGCNA_module_trait_heatmap_publication.pdf"), width=8.5, height=7.2)
labeledHeatmap(Matrix = moduleTraitCor, xLabels = names(traitMat), yLabels = names(MEs), ySymbols = names(MEs), colorLabels = FALSE, colors = blueWhiteRed(50), textMatrix = textMatrix, setStdMargins = FALSE, cex.text = 0.8, zlim = c(-1,1), main = "Module-trait relationships")
dev.off()
svg(file.path(out_dir, "WGCNA_module_trait_heatmap_publication.svg"), width=8.5, height=7.2)
labeledHeatmap(Matrix = moduleTraitCor, xLabels = names(traitMat), yLabels = names(MEs), ySymbols = names(MEs), colorLabels = FALSE, colors = blueWhiteRed(50), textMatrix = textMatrix, setStdMargins = FALSE, cex.text = 0.8, zlim = c(-1,1), main = "Module-trait relationships")
dev.off()
write.csv(data.frame(module=rownames(moduleTraitCor), moduleTraitCor, check.names=FALSE), file.path(out_dir, "WGCNA_module_trait_correlation_R.csv"), row.names=FALSE)
write.csv(data.frame(gene=colnames(datExpr), module_color=moduleColors), file.path(out_dir, "WGCNA_module_assignment_R.csv"), row.names=FALSE)
'''
    script_path.write_text(content, encoding="utf-8")


def build_wgcna_r_input(expr_mats: dict[str, pd.DataFrame], out_csv: Path) -> None:
    disease = next(v for k, v in expr_mats.items() if "control" in k.lower())
    treatment = next(v for k, v in expr_mats.items() if any(t in k.lower() for t in ["iroshy", "isorhy", "irn"]))
    disease = disease.copy()
    treatment = treatment.copy()
    disease["gene"] = disease["gene"].astype(str).str.strip()
    treatment["gene"] = treatment["gene"].astype(str).str.strip()
    disease = disease[(disease["gene"].str.lower() != "nan") & (disease["gene"] != "")].copy()
    treatment = treatment[(treatment["gene"].str.lower() != "nan") & (treatment["gene"] != "")].copy()
    disease_genes = {str(x).strip() for x in disease["gene"].tolist() if str(x).strip() and str(x).strip().lower() != "nan"}
    treatment_genes = {str(x).strip() for x in treatment["gene"].tolist() if str(x).strip() and str(x).strip().lower() != "nan"}
    shared = sorted(disease_genes & treatment_genes)
    disease = disease[disease["gene"].isin(shared)].copy().set_index("gene")
    treatment = treatment[treatment["gene"].isin(shared)].copy().set_index("gene")
    control_cols = [c for c in disease.columns if infer_group(c) == "Control"]
    pm_cols = [c for c in disease.columns if infer_group(c) == "PM"]
    iso_cols = [c for c in treatment.columns if infer_group(c) == "PM+Isorhy"]
    combined = pd.concat([disease[control_cols + pm_cols], treatment[iso_cols]], axis=1)
    var = combined.var(axis=1).sort_values(ascending=False)
    combined = combined.loc[var.head(min(1000, len(var))).index]
    combined.insert(0, "group_dummy", "expr")
    combined.insert(0, "gene", combined.index)
    combined.reset_index(drop=True, inplace=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")


def run_wgcna_r(expr_mats: dict[str, pd.DataFrame], output_dir: Path, figures_pub_dir: Path) -> tuple[bool, str]:
    input_csv = output_dir / "tables" / "wgcna_r_input.csv"
    build_wgcna_r_input(expr_mats, input_csv)
    r_script = Path("scripts") / "run_wgcna_publication.R"
    write_wgcna_r_script(r_script)
    cmd = [
        r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
        str(r_script.resolve()),
        str(input_csv.resolve()),
        str(figures_pub_dir.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    needed = [
        figures_pub_dir / "WGCNA_gene_dendrogram_module_colors_publication.png",
        figures_pub_dir / "WGCNA_module_trait_heatmap_publication.png",
    ]
    if result.returncode != 0 or not all(p.exists() for p in needed):
        return False, f"R/WGCNA failed: {result.stderr.strip() or result.stdout.strip()}"
    return True, "R/WGCNA completed successfully."


def create_exploratory_wgcna_publication(output_dir: Path, figures_pub_dir: Path) -> None:
    assign = pd.read_csv(output_dir / "tables" / "WGCNA_module_assignment.csv")
    trait = pd.read_csv(output_dir / "tables" / "WGCNA_module_trait_correlation.csv")
    hubs = pd.read_csv(output_dir / "tables" / "WGCNA_hub_genes.csv")
    nlrp2 = pd.read_csv(output_dir / "tables" / "NLRP2_WGCNA_status.csv")
    hubs.to_csv(output_dir / "tables" / "WGCNA_hub_genes_exploratory.csv", index=False, encoding="utf-8-sig")
    nlrp2.to_csv(output_dir / "tables" / "NLRP2_WGCNA_status_exploratory.csv", index=False, encoding="utf-8-sig")
    module_counts = assign["module"].astype(str).value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.barplot(x=module_counts.index, y=module_counts.values, color="#6C8EBF", ax=ax)
    ax.set_xlabel("Module")
    ax.set_ylabel("Gene count")
    ax.set_title("Exploratory WGCNA-like module sizes")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_multi(fig, figures_pub_dir / "WGCNA_gene_dendrogram_exploratory")
    plt.close(fig)
    corr = trait.pivot(index="module", columns="trait", values="correlation")
    pval = trait.pivot(index="module", columns="trait", values="pvalue")
    annot = corr.copy().astype(object)
    for i in corr.index:
        for j in corr.columns:
            annot.loc[i, j] = f"{corr.loc[i,j]:.2f}\n(p={pval.loc[i,j]:.3f})"
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.42 * len(corr.index) + 2)))
    sns.heatmap(corr, annot=annot, fmt="", cmap="vlag", center=0, linewidths=0.4, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title("Exploratory WGCNA-like module-trait heatmap")
    save_multi(fig, figures_pub_dir / "WGCNA_module_trait_heatmap_exploratory")
    plt.close(fig)


def select_publication_ppi(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nlrp2 = nodes[nodes["gene"].astype(str).str.upper() == "NLRP2"].iloc[0]["gene"]
    neighbor_genes = set(edges.loc[(edges["source"] == nlrp2) | (edges["target"] == nlrp2), ["source", "target"]].values.ravel())
    top_ai = nodes.sort_values("AI_weight", ascending=False).head(12)["gene"].astype(str).tolist()
    keep = set(top_ai) | neighbor_genes | {nlrp2}
    sub_edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)].copy()
    if sub_edges.empty:
        sub_edges = edges.sort_values("score", ascending=False).head(20).copy()
        keep = set(sub_edges["source"]) | set(sub_edges["target"]) | {nlrp2}
    sub_nodes = nodes[nodes["gene"].isin(keep)].copy()
    return sub_nodes, sub_edges


def draw_ppi_publication(sub_nodes: pd.DataFrame, sub_edges: pd.DataFrame, base_path: Path, ai_mode: bool) -> None:
    g = nx.Graph()
    for _, row in sub_nodes.iterrows():
        g.add_node(row["gene"], **row.to_dict())
    for _, row in sub_edges.iterrows():
        g.add_edge(row["source"], row["target"], score=float(row["score"]))
    nlrp2 = next((n for n in g.nodes if uppercase_gene(n) == "NLRP2"), None)
    if nlrp2 is not None:
        lengths = nx.single_source_shortest_path_length(g, nlrp2, cutoff=2)
        shells = [
            [n for n, d in lengths.items() if d == 0],
            [n for n, d in lengths.items() if d == 1],
            [n for n, d in lengths.items() if d >= 2],
        ]
        shells = [s for s in shells if s]
        pos = nx.shell_layout(g, shells)
        spring = nx.spring_layout(g, pos=pos, fixed=[nlrp2], seed=42, weight="score", k=0.8, iterations=200)
        pos = spring
    else:
        pos = nx.spring_layout(g, seed=42, weight="score", k=0.8, iterations=200)
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    widths = [0.6 + 4.0 * g[u][v]["score"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, edge_color="#B3B3B3", alpha=0.6)
    normal_nodes = []
    normal_sizes = []
    normal_colors = []
    labels = {}
    highlight = []
    for node, attrs in g.nodes(data=True):
        if uppercase_gene(node) == "NLRP2":
            highlight.append(node)
            labels[node] = node
            continue
        normal_nodes.append(node)
        if ai_mode:
            normal_sizes.append(280 + 2400 * attrs.get("AI_weight", 0.0))
            normal_colors.append(attrs.get("AI_weight", 0.0))
        else:
            normal_sizes.append(260 + 1800 * attrs.get("degree_centrality", 0.0))
            normal_colors.append(attrs.get("log2FoldChange", 0.0))
        if attrs.get("label_node", False) or attrs.get("degree", 0) >= 5:
            labels[node] = node
    if normal_nodes:
        if ai_mode:
            nodes = nx.draw_networkx_nodes(g, pos, nodelist=normal_nodes, node_color=normal_colors, node_size=normal_sizes, cmap="YlOrRd", edgecolors="#444444", linewidths=0.8, ax=ax)
            cbar = fig.colorbar(nodes, ax=ax, fraction=0.045, pad=0.02)
            cbar.set_label("AI weight")
        else:
            vals = pd.Series(normal_colors, dtype=float).fillna(0)
            vmax = max(abs(vals.max()), abs(vals.min()), 1.0)
            nodes = nx.draw_networkx_nodes(g, pos, nodelist=normal_nodes, node_color=normal_colors, node_size=normal_sizes, cmap="coolwarm", vmin=-vmax, vmax=vmax, edgecolors="#444444", linewidths=0.8, ax=ax)
            cbar = fig.colorbar(nodes, ax=ax, fraction=0.045, pad=0.02)
            cbar.set_label("log2FC")
    if highlight:
        nx.draw_networkx_nodes(g, pos, nodelist=highlight, node_color="#F4A259", node_size=1450, edgecolors="#8A4F08", linewidths=1.2, ax=ax)
    text_art = nx.draw_networkx_labels(g, pos, labels=labels, font_size=10, font_color="#222222", ax=ax)
    try:
        adjust_text(list(text_art.values()), ax=ax, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))
    except Exception:
        pass
    ax.set_title("AI-enhanced STRING PPI network" if ai_mode else "STRING PPI network")
    ax.axis("off")
    save_multi(fig, base_path)
    plt.close(fig)


def write_qc_report(output_dir: Path, gsea_formal: bool, wgcna_formal: bool) -> None:
    report = f"""# Figure QC Report

## Volcano plots

- Old issue: only one volcano figure existed and label density was not publication-grade.
- New optimization: separate publication-style volcanoes for `Pm vs control` and `Pm+Iroshy vs Pm`, lighter NS points, restrained red/blue palette, reduced labels, automatic label repulsion.
- Status: `Pm vs control` can serve as a main/supplementary figure; `Pm+Iroshy vs Pm` should be labeled exploratory because no preset-significant DEG was observed.

## Heatmaps

- Old issue: too many labels and generic cluster-map aesthetics.
- New optimization: reduced to top 20–30 genes, forced Nlrp2 inclusion, cleaner annotation bar, publication palette, separate main and supplementary versions.
- Status: exploratory figure for the primary comparison because the underlying DEG evidence is exploratory.

## GSEA

- Old issue: fallback running-score plots with obvious fallback flavor.
- New optimization: attempted formal fgsea rerun using local public GMT resources and publication-style export.
- Status: {'formal analysis result and suitable for publication figures' if gsea_formal else 'still exploratory/fallback-informed; should not be presented as a formal main-pathway result without explicit caveat'}

## WGCNA

- Old issue: Python WGCNA-like plots were coarse.
- New optimization: attempted standard R/WGCNA rerun with standard soft-threshold, dendrogram, and module-trait heatmap outputs.
- Status: {'formal R/WGCNA-style output suitable for publication/supplementary use' if wgcna_formal else 'still based on exploratory workflow and best used as supplementary evidence'}

## PPI

- Old issue: prior network views were visually crowded and not manuscript-ready.
- New optimization: reduced STRING-enhanced network, force-directed/shell-refined layout, scaled edges by confidence, scaled nodes by degree/importance, limited labels.
- Status: can serve as a publication main figure if described as STRING-enhanced network reconstruction from public resources.

## AI-enhanced PPI

- Old issue: prior AI network was dense and difficult to interpret.
- New optimization: reduced ego/importance-focused network centered on Nlrp2, top neighbors retained, labels restricted to key nodes.
- Status: suitable as a main or supplementary figure with the explicit label `AI-inspired weighted PPI interpretation`, not a true GNN.
"""
    (output_dir / "figure_qc_report.md").write_text(report, encoding="utf-8")


def update_reports(output_dir: Path, gsea_formal: bool, gsea_msg: str, wgcna_formal: bool, wgcna_msg: str) -> None:
    en_path = output_dir / "final_report.md"
    zh_path = output_dir / "final_report_zh.md"
    summary_path = output_dir / "completion_summary.txt"
    en = en_path.read_text(encoding="utf-8", errors="replace")
    zh = zh_path.read_text(encoding="utf-8", errors="replace")
    summary = summary_path.read_text(encoding="utf-8", errors="replace")
    marker = "\n## 13. Publication-quality figure upgrade\n"
    if marker in en:
        en = en.split(marker)[0].rstrip()
    en += f"""

## 13. Publication-quality figure upgrade

- Publication-style volcano plots were redrawn for both `Pm vs control` and `Pm+Iroshy vs Pm` and exported to `output/figures_pub/`.
- Publication-style DEG heatmaps (main and supplementary) were added to `output/figures_pub/`.
- Formal GSEA upgrade status: {'success' if gsea_formal else 'not fully successful'}.
  - Note: {gsea_msg}
- R/WGCNA upgrade status: {'success' if wgcna_formal else 'not fully successful'}.
  - Note: {wgcna_msg}
- STRING-PPI and AI-enhanced PPI publication-style figures were added to `output/figures_pub/`.
- Figure suitability:
  - Main-figure candidates: publication volcano (`Pm vs control`), STRING PPI publication figure, AI-enhanced PPI publication figure.
  - Supplementary / caveat-needed figures: `Pm+Iroshy vs Pm` exploratory volcano, heatmaps, and any non-formal GSEA/WGCNA outputs.
"""
    zh_marker = "\n## 9. Publication-quality figure upgrade\n"
    if zh_marker in zh:
        zh = zh.split(zh_marker)[0].rstrip()
    zh += f"""

## 9. Publication-quality figure upgrade

- 已对 `Pm vs control` 和 `Pm+Iroshy vs Pm` 两个比较分别重绘 publication-style 火山图，并输出到 `output/figures_pub/`。
- 已新增 publication-style DEG 热图主图和补充图版本。
- Formal GSEA 升级状态：{'成功' if gsea_formal else '未完全成功'}。
  - 说明：{gsea_msg}
- R/WGCNA 升级状态：{'成功' if wgcna_formal else '未完全成功'}。
  - 说明：{wgcna_msg}
- 已新增 STRING-PPI 与 AI-enhanced PPI 的论文风格主图。
- 图形使用建议：
  - 可优先作为论文主图：`Pm vs control` publication volcano、STRING PPI publication figure、AI-enhanced PPI publication figure。
  - 更适合作为补充材料或需加说明：`Pm+Iroshy vs Pm` exploratory volcano、热图、以及任何未升级为 formal 的 GSEA/WGCNA 图。
"""
    if "\n19. Publication-quality figure upgrade:" in summary:
        summary = summary.split("\n19. Publication-quality figure upgrade:")[0].rstrip()
    summary += f"""

19. Publication-quality figure upgrade:
- Volcano publication figures: completed
- Heatmap publication figures: completed
- Formal GSEA upgrade: {'Yes' if gsea_formal else 'No'}
- R/WGCNA upgrade: {'Yes' if wgcna_formal else 'No'}
- STRING PPI publication redraw: completed
- AI PPI publication redraw: completed
"""
    en_path.write_text(en, encoding="utf-8")
    zh_path.write_text(zh, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCI-style figure redraw and incremental analysis enhancement")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    figures_pub_dir = output_dir / "figures_pub"
    figures_pub_dir.mkdir(parents=True, exist_ok=True)

    expr_mats = parse_expression_matrices(input_dir)
    all_diff = load_all_diff_results(input_dir)

    draw_volcano(all_diff, "Pm vs control", figures_pub_dir / "volcano_Pm_vs_control_full", publication_only=False)
    draw_volcano(all_diff, "Pm vs control", figures_pub_dir / "volcano_Pm_vs_control_publication", publication_only=True)
    draw_volcano(all_diff, "Pm+Iroshy vs Pm", figures_pub_dir / "volcano_PmIroshy_vs_Pm_full", publication_only=False)
    draw_volcano(all_diff, "Pm+Iroshy vs Pm", figures_pub_dir / "volcano_PmIroshy_vs_Pm_publication", publication_only=True)

    draw_heatmaps(expr_mats, all_diff, figures_pub_dir)

    gsea_formal, gsea_msg = run_formal_gsea(input_dir, output_dir, figures_pub_dir)
    primary = all_diff[all_diff["comparison"] == "Pm+Iroshy vs Pm"].copy()
    primary["ranking_score"] = np.sign(primary["log2FoldChange"].fillna(0)) * safe_neglog10(primary["pvalue"].fillna(primary["padj"]).fillna(1))
    primary = primary[["gene_name", "ranking_score"]].dropna()
    primary["gene_name"] = primary["gene_name"].astype(str).str.strip()
    primary = primary[(primary["gene_name"] != "") & (primary["gene_name"].str.lower() != "nan")]
    primary = primary.sort_values("ranking_score", ascending=False).drop_duplicates(subset=["gene_name"], keep="first")
    rank_series = pd.Series(primary["ranking_score"].to_numpy(dtype=float), index=primary["gene_name"])
    local_sets = load_local_target_gene_sets(output_dir)
    exploratory_rows = []
    exploratory_map = {
        "GSEA_pyroptosis_exploratory": "CUSTOM_PYROPTOSIS_GENE_SET",
        "GSEA_innate_immune_exploratory": "REACTOME_INNATE_IMMUNE_SYSTEM",
        "GSEA_NOD_like_exploratory": "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY",
    }
    for stub, term in exploratory_map.items():
        exploratory_rows.append(
            draw_exploratory_gsea_figure(
                rank_series,
                term,
                local_sets.get(term, []),
                figures_pub_dir / stub,
            )
        )
    pd.DataFrame(exploratory_rows).to_csv(output_dir / "tables" / "GSEA_exploratory_results.csv", index=False, encoding="utf-8-sig")

    wgcna_formal, wgcna_msg = run_wgcna_r(expr_mats, output_dir, figures_pub_dir)
    create_exploratory_wgcna_publication(output_dir, figures_pub_dir)

    ppi_nodes = pd.read_csv(output_dir / "networks" / "PPI_nodes_STRING_enhanced.csv")
    ppi_edges = pd.read_csv(output_dir / "networks" / "PPI_edges_STRING_enhanced.csv")
    ai_nodes = pd.read_csv(output_dir / "tables" / "AI_PPI_node_importance_STRING_enhanced.csv")
    ai_neighbors = pd.read_csv(output_dir / "tables" / "NLRP2_key_neighbor_ranking_STRING_enhanced.csv")
    pub_nodes, pub_edges = select_publication_ppi(ai_nodes, ppi_edges)
    pub_edges.to_csv(output_dir / "networks" / "PPI_edges_STRING_publication.csv", index=False, encoding="utf-8-sig")
    pub_nodes.to_csv(output_dir / "networks" / "PPI_nodes_STRING_publication.csv", index=False, encoding="utf-8-sig")
    draw_ppi_publication(pub_nodes, pub_edges, figures_pub_dir / "PPI_Nlrp2_centered_publication", ai_mode=False)
    draw_ppi_publication(pub_nodes, pub_edges, figures_pub_dir / "AI_PPI_Nlrp2_centered_publication", ai_mode=True)
    draw_ppi_publication(ai_nodes, ppi_edges, figures_pub_dir / "PPI_full_STRING_supplementary", ai_mode=False)
    draw_ppi_publication(ai_nodes, ppi_edges, figures_pub_dir / "AI_PPI_full_supplementary", ai_mode=True)
    draw_ppi_publication(pub_nodes, pub_edges, figures_pub_dir / "PPI_STRING_publication", ai_mode=False)
    draw_ppi_publication(pub_nodes, pub_edges, figures_pub_dir / "AI_enhanced_PPI_publication", ai_mode=True)
    pub_nodes.sort_values("AI_weight", ascending=False).to_csv(output_dir / "tables" / "AI_PPI_node_importance_publication.csv", index=False, encoding="utf-8-sig")
    ai_neighbors.to_csv(output_dir / "tables" / "NLRP2_key_neighbor_ranking_publication.csv", index=False, encoding="utf-8-sig")

    write_qc_report(output_dir, gsea_formal, wgcna_formal)
    update_reports(output_dir, gsea_formal, gsea_msg, wgcna_formal, wgcna_msg)


if __name__ == "__main__":
    main()
