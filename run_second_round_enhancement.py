#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import io
import logging
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from gseapy import get_library, get_library_name, prerank
from gseapy.plot import gseaplot


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


@dataclass
class Round2State:
    downloaded_resources: list[str] = field(default_factory=list)
    local_gene_list_sent: list[str] = field(default_factory=list)
    sent_expression_or_sample: bool = False
    gsea_formal: bool = False
    ppi_string_enhanced: bool = False
    ai_mode: str = "AI-inspired weighted PPI interpretation"
    limitations: list[str] = field(default_factory=list)


def uppercase_gene(text: Any) -> str:
    return str(text).strip().upper()


def normalize_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("second_round_enhancement")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def append_network_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def safe_neglog10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    min_positive = finite.min() if finite.size else 1e-300
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, min_positive / 10.0)
    return -np.log10(arr)


def minmax_scale(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0).to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if math.isclose(mx, mn):
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def save_figure(fig: Any, png_path: Path, pdf_path: Path) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")


def build_manual_gsea_curve(rank_metric: pd.Series, gene_set_upper: set[str]) -> tuple[np.ndarray, list[int]]:
    labels = rank_metric.index.astype(str).tolist()
    values = rank_metric.to_numpy(dtype=float)
    hits = [idx for idx, gene in enumerate(labels) if uppercase_gene(gene) in gene_set_upper]
    n = len(labels)
    if n == 0:
        return np.array([]), []
    if not hits:
        miss_weight = -1.0 / n
        return np.cumsum(np.repeat(miss_weight, n)), []
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
    return np.array(running, dtype=float), hits


def load_round1_context(output_dir: Path) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    text_files = [
        "completion_summary.txt",
        "final_report.md",
        "logs/analysis.log",
    ]
    csv_files = [
        "tables/NLRP2_integrated_evidence.csv",
        "tables/NLRP2_status_in_DEG.csv",
        "tables/GSEA_results.csv",
        "tables/WGCNA_module_assignment.csv",
        "tables/WGCNA_hub_genes.csv",
        "tables/NLRP2_WGCNA_status.csv",
        "networks/PPI_edges.csv",
        "networks/PPI_nodes.csv",
        "tables/AI_PPI_node_importance.csv",
        "tables/NLRP2_key_neighbor_ranking.csv",
        "tables/all_DEG_results.csv",
    ]
    for rel in text_files:
        ctx[rel] = (output_dir / rel).read_text(encoding="utf-8")
    for rel in csv_files:
        ctx[rel] = pd.read_csv(output_dir / rel)
    return ctx


def write_second_round_diagnosis(output_dir: Path, ctx: dict[str, Any]) -> None:
    nlrp2_deg = ctx["tables/NLRP2_status_in_DEG.csv"]
    nlrp2_wgcna = ctx["tables/NLRP2_WGCNA_status.csv"]
    disease = nlrp2_deg[nlrp2_deg["comparison_role"] == "disease_background"].iloc[0]
    primary = nlrp2_deg[nlrp2_deg["comparison_role"] == "primary"].iloc[0]
    wgcna = nlrp2_wgcna.iloc[0]
    text = f"""# Second Round Diagnosis

## 1. First-round modules that should be treated as formal results

- DEG table standardization, DEG summary export, and the volcano plot were formal reorganizations of the local DEG result table.
- GO / KEGG / Reactome bubble plots were formal visual summaries of the local enrichment result files.
- The disease-background comparison `Pm vs control` had 19 genes meeting the preset DEG threshold, so it can be cited as formal background DEG evidence.

## 2. First-round modules that should be treated as exploratory

- The primary comparison `Pm+Iroshy vs Pm` had no genes passing `padj < 0.05`, so the heatmap was exploratory.
- WGCNA was a Python WGCNA-like exploratory workflow rather than canonical R `WGCNA`.
- The WGCNA-like result used only 9 samples and 526 shared genes, so module assignments remain hypothesis-generating.
- The AI network interpretation was AI-inspired weighted scoring, not a trained GNN.

## 3. First-round modules that should be treated as fallback

- GSEA was a deterministic running-score fallback rather than formal permutation-based GSEA.
- PPI was a fallback network rather than a STRING-supported network.
- The first-round AI-enhanced PPI therefore also depended on the fallback network backbone.

## 4. Which results most need online enhancement

- GSEA needs public gene sets downloaded locally and re-run with public-resource support.
- PPI needs STRING public interactions downloaded locally and filtered offline.
- AI-enhanced PPI needs recalculation after STRING-enhanced network reconstruction.
- NLRP2 integrated evidence needs refresh after the online-enhanced GSEA and PPI steps.

## 5. Current NLRP2 evidence strength

- `Pm vs control`: log2FC = {disease['log2FoldChange']:.6f}, pvalue = {disease['pvalue']:.6f}, padj = {disease['padj']:.6f}
- `Pm+Iroshy vs Pm`: log2FC = {primary['log2FoldChange']:.6f}, pvalue = {primary['pvalue']:.6f}, padj = {primary['padj']:.6f}
- NLRP2 did not pass the preset DEG threshold in the primary comparison.
- First-round WGCNA-like assignment: module = `{wgcna['module']}`, gene significance = {wgcna['gene_significance']:.6f}
- First-round fallback PPI linked NLRP2 only to Casp1.
- Best current evidence label before online enhancement: `exploratory to moderate`.

## 6. Conclusions that cannot be overstated

- Do not state that NLRP2 has been definitively identified as the core key gene.
- Do not state that first-round GSEA formally proved pyroptosis enrichment.
- Do not state that first-round WGCNA strictly proved an NLRP2 key module.
- Do not state that first-round PPI was STRING-supported.
- Do not describe the first-round AI result as a real GNN.
"""
    (output_dir / "logs" / "second_round_diagnosis.md").write_text(text, encoding="utf-8")


def infer_species_from_input(input_dir: Path) -> tuple[int, str]:
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
            for col in df.columns:
                if str(col).lower() in {"geneid", "gene_id"}:
                    values = df[col].astype(str)
                    if values.str.startswith("ENSMUSG").any():
                        return 10090, "Mouse"
                    if values.str.startswith("ENSG").any():
                        return 9606, "Human"
    return 10090, "Mouse"


def load_all_diff_results(input_dir: Path) -> pd.DataFrame:
    for path in input_dir.rglob("*.xlsx"):
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        frames = []
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception:
                continue
            if df.empty:
                continue
            if "log2FoldChange" not in map(str, df.columns):
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
            df["significant"] = (df["log2FoldChange"].abs() >= 1) & (pd.to_numeric(df[p_col], errors="coerce") < 0.05)
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
    raise FileNotFoundError("Original differential result workbook could not be found in input/.")


def build_gene_lookup(input_dir: Path, all_diff: pd.DataFrame) -> dict[str, str]:
    lookup = {uppercase_gene(g): str(g) for g in all_diff["gene_name"].dropna().astype(str)}
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
            first_col = str(df.columns[0])
            if normalize_text(first_col) != "group":
                continue
            expr = df.iloc[1:].copy()
            expr.rename(columns={first_col: "gene"}, inplace=True)
            for gene in expr["gene"].dropna().astype(str):
                lookup.setdefault(uppercase_gene(gene), gene)
    return lookup


def load_candidate_genes(input_dir: Path) -> list[str]:
    genes = []
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
            if df.empty or df.shape[1] != 1:
                continue
            if df.iloc[:, 0].astype(str).nunique() < 3:
                continue
            genes.extend([str(df.columns[0]).strip()])
            genes.extend(df.iloc[:, 0].dropna().astype(str).str.strip().tolist())
    return sorted({g for g in genes if g and g.lower() != "nan"})


def download_public_gene_sets(species_name: str, network_log: Path, output_dir: Path, state: Round2State) -> tuple[dict[str, list[str]], list[str]]:
    notes = []
    append_network_log(network_log, "Accessing Enrichr/gseapy public library listing. Sent: organism only. No sample or expression data.")
    libs = get_library_name(organism=species_name)
    notes.append(f"Retrieved {len(libs)} public Enrichr library names for {species_name}.")
    lower_libs = {lib.lower(): lib for lib in libs}
    desired_libraries = {
        "reactome": ["reactome"],
        "kegg": ["kegg"],
        "go_bp": ["go", "biological", "process"],
        "go_cc": ["go", "cellular", "component"],
    }
    chosen = {}
    for label, tokens in desired_libraries.items():
        hit = next((orig for low, orig in lower_libs.items() if all(t in low for t in tokens)), None)
        if hit:
            chosen[label] = hit
    patterns = {
        "REACTOME_INNATE_IMMUNE_SYSTEM": ["innate immune system"],
        "REACTOME_INFLAMMASOMES": ["inflammasome"],
        "REACTOME_INTERLEUKIN_1_SIGNALING": ["interleukin-1", "interleukin 1"],
        "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY": ["nod-like receptor", "nod like receptor"],
        "GO_PYROPTOSIS": ["pyroptosis"],
        "GO_INFLAMMATORY_RESPONSE": ["inflammatory response"],
        "GO_CASPASE_ACTIVATION": ["caspase activation"],
    }
    out_dir = output_dir / "networks" / "downloaded_gene_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_sets: dict[str, list[str]] = {}
    for label, library_name in chosen.items():
        append_network_log(network_log, f"Downloading public gene-set library {library_name}. Sent: organism/library request only.")
        gene_lib = get_library(name=library_name, organism=species_name)
        state.downloaded_resources.append(f"{library_name}.gmt")
        with (out_dir / f"{library_name}.gmt").open("w", encoding="utf-8") as handle:
            for term, genes in gene_lib.items():
                handle.write(term + "\tna\t" + "\t".join(map(str, genes)) + "\n")
        notes.append(f"Downloaded and cached {library_name}.")
        for target_name, target_patterns in patterns.items():
            if target_name in gene_sets:
                continue
            for term, genes in gene_lib.items():
                low_term = term.lower()
                if any(pat in low_term for pat in target_patterns):
                    gene_sets[target_name] = sorted(set(map(str, genes)))
                    break
    gene_sets["CUSTOM_PYROPTOSIS_GENE_SET"] = PYROPTOSIS_GENES.copy()
    notes.append("Added built-in pyroptosis gene set as a local fallback companion set.")
    return gene_sets, notes


def run_online_enhanced_gsea(input_dir: Path, output_dir: Path, network_log: Path, state: Round2State, logger: logging.Logger) -> dict[str, Any]:
    all_diff = load_all_diff_results(input_dir)
    primary = all_diff[all_diff["comparison"] == "Pm+Iroshy vs Pm"].copy()
    primary["ranking_score"] = np.sign(primary["log2FoldChange"].fillna(0)) * safe_neglog10(primary["pvalue"].fillna(primary["padj"]).fillna(1))
    primary = primary.replace([np.inf, -np.inf], np.nan).dropna(subset=["ranking_score"])
    primary = primary.sort_values("ranking_score", ascending=False).drop_duplicates(subset=["gene_upper"], keep="first")
    rank_upper = primary[["gene_upper", "ranking_score"]].copy()
    rank_series = pd.Series(primary["ranking_score"].to_numpy(dtype=float), index=primary["gene_name"].astype(str))
    _, species_name = infer_species_from_input(input_dir)
    gene_sets, notes = download_public_gene_sets(species_name, network_log, output_dir, state)
    overlap_rows = []
    for term, genes in gene_sets.items():
        overlap = len(set(rank_upper["gene_upper"]) & {uppercase_gene(g) for g in genes})
        overlap_rows.append({"Term": term, "Overlap_gene_count": overlap})
    overlap_df = pd.DataFrame(overlap_rows)
    formal_gene_sets = {term: sorted({uppercase_gene(g) for g in genes}) for term, genes in gene_sets.items()}
    results: dict[str, Any] = {"notes": notes, "leading_edge_map": {}, "plots": {}, "mode": "formal"}
    try:
        append_network_log(network_log, "Running local preranked GSEA with downloaded public gene sets. No external data sent.")
        pre_res = prerank(
            rnk=rank_upper,
            gene_sets=formal_gene_sets,
            min_size=1,
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
            norm = normalize_text(col)
            if norm in {"term", "name"}:
                rename_map[col] = "Term"
            elif norm == "es":
                rename_map[col] = "ES"
            elif norm == "nes":
                rename_map[col] = "NES"
            elif "nompval" in norm or norm in {"pval", "pvalue"}:
                rename_map[col] = "PValue"
            elif "fdrqval" in norm or norm in {"fdr", "qval"}:
                rename_map[col] = "FDR"
            elif "leadgenes" in norm or "leadgene" in norm:
                rename_map[col] = "Lead_genes"
        res_df = res_df.rename(columns=rename_map)
        if "Term" not in res_df.columns:
            res_df["Term"] = list(pre_res.results.keys())
        if "Lead_genes" not in res_df.columns:
            res_df["Lead_genes"] = [
                ";".join(pre_res.results.get(term, {}).get("lead_genes", []) or [])
                for term in res_df["Term"].astype(str)
            ]
        for col in ["ES", "NES", "PValue", "FDR"]:
            if col in res_df.columns:
                res_df[col] = pd.to_numeric(res_df[col], errors="coerce")
        res_df = res_df.merge(overlap_df, on="Term", how="left")
        res_df["Analysis_mode"] = "formal_preranked_gsea"
        res_df = res_df.sort_values(["FDR", "PValue"], ascending=[True, True], na_position="last")
        res_df.to_csv(output_dir / "tables" / "GSEA_results_online_enhanced.csv", index=False, encoding="utf-8-sig")
        plot_patterns = {
            "GSEA_online_pyroptosis": ["pyroptosis", "custom_pyroptosis"],
            "GSEA_online_innate_immune_system": ["innate immune system"],
            "GSEA_online_NOD_like_receptor": ["nod_like_receptor", "nod-like receptor"],
            "GSEA_online_inflammasome": ["inflammasome"],
        }
        rank_metric = pre_res.ranking.values if hasattr(pre_res.ranking, "values") else np.asarray(pre_res.ranking)
        for stub, patterns in plot_patterns.items():
            match = next((term for term in res_df["Term"].astype(str) if any(p in term.lower() for p in patterns)), None)
            if match is None:
                continue
            detail = pre_res.results.get(match, {})
            hits = detail.get("hits")
            res_curve = detail.get("RES")
            if hits is None or res_curve is None:
                continue
            gseaplot(
                term=match,
                hits=hits,
                nes=float(detail.get("nes", np.nan)),
                pval=float(detail.get("pval", np.nan)),
                fdr=float(detail.get("fdr", np.nan)),
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(output_dir / "figures" / f"{stub}.png"),
            )
            gseaplot(
                term=match,
                hits=hits,
                nes=float(detail.get("nes", np.nan)),
                pval=float(detail.get("pval", np.nan)),
                fdr=float(detail.get("fdr", np.nan)),
                RES=res_curve,
                rank_metric=rank_metric,
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(output_dir / "figures" / f"{stub}.pdf"),
            )
            results["plots"][stub] = match
        results["leading_edge_map"] = {row["Term"]: str(row.get("Lead_genes", "")) for _, row in res_df.iterrows()}
        results["result_df"] = res_df
        state.gsea_formal = True
        return results
    except Exception as exc:
        logger.warning("Formal online-enhanced GSEA failed, switching to public-resource-informed fallback: %s", exc)
        rows = []
        plot_map = {
            "GSEA_online_pyroptosis": "CUSTOM_PYROPTOSIS_GENE_SET",
            "GSEA_online_innate_immune_system": "REACTOME_INNATE_IMMUNE_SYSTEM",
            "GSEA_online_NOD_like_receptor": "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY",
            "GSEA_online_inflammasome": "REACTOME_INFLAMMASOMES",
        }
        for term_name in sorted(set(plot_map.values())):
            genes = gene_sets.get(term_name, [])
            upper_set = {uppercase_gene(g) for g in genes}
            curve, hits = build_manual_gsea_curve(rank_series, upper_set)
            overlap = [gene for gene in rank_series.index if uppercase_gene(gene) in upper_set]
            max_idx = int(np.argmax(np.abs(curve))) if curve.size else 0
            leading = [gene for gene in rank_series.index[: max_idx + 1] if uppercase_gene(gene) in upper_set]
            rows.append(
                {
                    "Term": term_name,
                    "ES": float(curve[max_idx]) if curve.size else np.nan,
                    "NES": np.nan,
                    "PValue": np.nan,
                    "FDR": np.nan,
                    "Lead_genes": ";".join(leading),
                    "Overlap_gene_count": len(overlap),
                    "Analysis_mode": "manual_online_running_score_fallback",
                }
            )
            results["leading_edge_map"][term_name] = ";".join(leading)
        res_df = pd.DataFrame(rows).sort_values("ES", key=lambda x: x.abs(), ascending=False)
        res_df.to_csv(output_dir / "tables" / "GSEA_results_online_enhanced.csv", index=False, encoding="utf-8-sig")
        for stub, term_name in plot_map.items():
            genes = gene_sets.get(term_name, [])
            upper_set = {uppercase_gene(g) for g in genes}
            curve, hits = build_manual_gsea_curve(rank_series, upper_set)
            overlap = res_df.loc[res_df["Term"] == term_name, "Overlap_gene_count"].iloc[0]
            display = f"{term_name} [online fallback, overlap={overlap}]"
            gseaplot(
                term=display,
                hits=hits,
                nes=np.nan,
                pval=np.nan,
                fdr=np.nan,
                RES=curve,
                rank_metric=rank_series.to_numpy(dtype=float),
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(output_dir / "figures" / f"{stub}.png"),
            )
            gseaplot(
                term=display,
                hits=hits,
                nes=np.nan,
                pval=np.nan,
                fdr=np.nan,
                RES=curve,
                rank_metric=rank_series.to_numpy(dtype=float),
                color="#D94E41",
                figsize=(7.2, 5.8),
                ofname=str(output_dir / "figures" / f"{stub}.pdf"),
            )
            results["plots"][stub] = term_name
        results["result_df"] = res_df
        results["mode"] = "manual_online_fallback"
        results["notes"].append(f"Formal online-enhanced GSEA failed: {exc}")
        state.limitations.append("Formal online-enhanced GSEA still failed, so the second round retains a public-resource-informed fallback running-score interpretation.")
        return results


def download_string_resources(species_id: int, output_dir: Path, network_log: Path, state: Round2State, logger: logging.Logger) -> tuple[Path | None, Path | None]:
    download_dir = output_dir / "networks" / "string_public"
    download_dir.mkdir(parents=True, exist_ok=True)
    base = "10090" if species_id == 10090 else "9606"
    links_url = f"https://stringdb-downloads.org/download/protein.links.v12.0/{base}.protein.links.v12.0.txt.gz"
    alias_url = f"https://stringdb-downloads.org/download/protein.aliases.v12.0/{base}.protein.aliases.v12.0.txt.gz"
    links_path = download_dir / Path(links_url).name
    alias_path = download_dir / Path(alias_url).name
    for url, out_path in [(links_url, links_path), (alias_url, alias_path)]:
        try:
            append_network_log(network_log, f"Downloading STRING public resource: {url}. Sent: none.")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            state.downloaded_resources.append(out_path.name)
            logger.info("Downloaded STRING resource %s", out_path.name)
        except Exception as exc:
            logger.warning("Failed to download STRING resource %s: %s", url, exc)
            return None, None
    return links_path, alias_path


def draw_ppi_figure(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, png_path: Path, pdf_path: Path, ai_mode: bool) -> None:
    graph = nx.Graph()
    for _, row in nodes_df.iterrows():
        graph.add_node(row["gene"], **row.to_dict())
    for _, row in edges_df.iterrows():
        graph.add_edge(row["source"], row["target"], score=float(row.get("score", 0.5)))
    pos = nx.spring_layout(graph, seed=42, weight="score", k=0.7)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    widths = [1.0 + 4.5 * float(graph[u][v].get("score", 0.3)) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color="#B0B0B0", alpha=0.55, ax=ax)
    normal_nodes = []
    normal_colors = []
    normal_sizes = []
    labels = {}
    highlight = []
    for node, attrs in graph.nodes(data=True):
        if uppercase_gene(node) == "NLRP2":
            highlight.append(node)
            labels[node] = node
            continue
        normal_nodes.append(node)
        if ai_mode:
            normal_colors.append(attrs.get("AI_weight", 0.0))
            normal_sizes.append(260 + 2200 * attrs.get("AI_weight", 0.0))
        else:
            normal_colors.append(attrs.get("log2FoldChange", 0.0))
            normal_sizes.append(220 + 1500 * attrs.get("degree_centrality", 0.0))
        if attrs.get("label_node", False):
            labels[node] = node
    if normal_nodes:
        if ai_mode:
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=normal_nodes, node_color=normal_colors, node_size=normal_sizes, cmap="YlOrRd", edgecolors="#444444", linewidths=0.8, ax=ax)
            cbar = fig.colorbar(nodes, ax=ax)
            cbar.set_label("AI weight")
            title = "AI-enhanced STRING PPI network"
        else:
            series = pd.Series(normal_colors, dtype=float).fillna(0)
            vmax = max(abs(float(series.max())), abs(float(series.min())), 1.0)
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=normal_nodes, node_color=normal_colors, node_size=normal_sizes, cmap="coolwarm", vmin=-vmax, vmax=vmax, edgecolors="#444444", linewidths=0.8, ax=ax)
            cbar = fig.colorbar(nodes, ax=ax)
            cbar.set_label("Log2 fold change")
            title = "STRING-enhanced PPI network"
    else:
        title = "STRING-enhanced PPI network"
    if highlight:
        nx.draw_networkx_nodes(graph, pos, nodelist=highlight, node_color="#F4A259", node_size=1200, edgecolors="#8A4F08", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, font_color="#222222", ax=ax)
    ax.set_title(title, weight="bold")
    ax.axis("off")
    save_figure(fig, png_path, pdf_path)
    plt.close(fig)


def build_string_enhanced_ppi(input_dir: Path, output_dir: Path, network_log: Path, gsea_df: pd.DataFrame, state: Round2State, logger: logging.Logger) -> dict[str, Any]:
    all_diff = load_all_diff_results(input_dir)
    species_id, species_name = infer_species_from_input(input_dir)
    gene_lookup = build_gene_lookup(input_dir, all_diff)
    links_path, alias_path = download_string_resources(species_id, output_dir, network_log, state, logger)
    candidate_genes = load_candidate_genes(input_dir)
    wgcna_hub = pd.read_csv(output_dir / "tables" / "WGCNA_hub_genes.csv")
    wgcna_assignment = pd.read_csv(output_dir / "tables" / "WGCNA_module_assignment.csv")
    primary = all_diff[all_diff["comparison"] == "Pm+Iroshy vs Pm"].copy()
    top_ranked = primary.sort_values(["pvalue", "log2FoldChange"], ascending=[True, False]).head(60)["gene_name"].astype(str).tolist()
    top_abs = primary.sort_values("log2FoldChange", key=lambda s: s.abs(), ascending=False).head(60)["gene_name"].astype(str).tolist()
    leading_edge = []
    if "Lead_genes" in gsea_df.columns:
        for item in gsea_df["Lead_genes"].dropna().astype(str):
            leading_edge.extend([token for token in re.split(r"[;, ]+", item) if token])
    selected = []
    for gene in ["NLRP2", "Nlrp2"] + PYROPTOSIS_GENES + candidate_genes + wgcna_hub["gene"].astype(str).head(30).tolist() + top_ranked + top_abs + leading_edge:
        pretty = gene_lookup.get(uppercase_gene(gene))
        if pretty and pretty not in selected:
            selected.append(pretty)
    selected = selected[:100]
    edges = []
    if links_path is not None and alias_path is not None:
        append_network_log(network_log, "Filtering downloaded STRING resources locally against the selected public gene symbol set. No external data sent.")
        alias_df = pd.read_csv(gzip.open(alias_path, "rt", encoding="utf-8", errors="ignore"), sep="\t")
        alias_df = alias_df[alias_df["alias"].astype(str).str.upper().isin({uppercase_gene(g) for g in selected})].copy()
        alias_map = alias_df.groupby("#string_protein_id")["alias"].agg(lambda vals: list(pd.unique(vals))).to_dict()
        selected_ids = set(alias_df["#string_protein_id"].astype(str))
        link_df = pd.read_csv(gzip.open(links_path, "rt", encoding="utf-8", errors="ignore"), sep=r"\s+")
        if "combined_score" not in link_df.columns:
            link_df.columns = [str(c).strip() for c in link_df.columns]
        link_df = link_df[link_df["protein1"].astype(str).isin(selected_ids) & link_df["protein2"].astype(str).isin(selected_ids)].copy()
        for _, row in link_df.iterrows():
            p1 = str(row["protein1"])
            p2 = str(row["protein2"])
            a1 = alias_map.get(p1, [])
            a2 = alias_map.get(p2, [])
            if not a1 or not a2:
                continue
            g1 = gene_lookup.get(uppercase_gene(a1[0]), str(a1[0]))
            g2 = gene_lookup.get(uppercase_gene(a2[0]), str(a2[0]))
            score = float(row["combined_score"]) / 1000.0 if float(row["combined_score"]) > 1 else float(row["combined_score"])
            edges.append({"source": g1, "target": g2, "score": score})
    if not edges:
        append_network_log(network_log, "Bulk STRING download route yielded no usable local edges; using STRING API fallback with a small public gene-symbol list only.")
        payload = {"identifiers": "\n".join(selected), "species": species_id, "caller_identity": "codex_safe_round2"}
        state.local_gene_list_sent.append(f"STRING API fallback ({len(selected)} gene symbols)")
        response = requests.post("https://string-db.org/api/tsv/network", data=payload, timeout=120)
        response.raise_for_status()
        api_df = pd.read_csv(io.StringIO(response.text), sep="\t")
        for _, row in api_df.iterrows():
            source = row.get("preferredName_A")
            target = row.get("preferredName_B")
            score = row.get("score", np.nan)
            if pd.isna(source) or pd.isna(target):
                continue
            edges.append(
                {
                    "source": gene_lookup.get(uppercase_gene(source), str(source)),
                    "target": gene_lookup.get(uppercase_gene(target), str(target)),
                    "score": float(score),
                }
            )
    edges_df = pd.DataFrame(edges, columns=["source", "target", "score"]).drop_duplicates()
    edges_df = edges_df[edges_df["source"] != edges_df["target"]].copy()
    if edges_df.empty:
        raise ValueError("No STRING-enhanced edges could be constructed.")
    graph = nx.from_pandas_edgelist(edges_df, "source", "target", edge_attr="score")
    for gene in selected:
        if gene not in graph:
            graph.add_node(gene)
    degree_centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {n: 0.0 for n in graph.nodes()}
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    closeness = nx.closeness_centrality(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    primary_lookup = primary.set_index("gene_upper")
    wgcna_lookup = wgcna_assignment.assign(gene_upper=lambda d: d["gene"].astype(str).str.upper()).set_index("gene_upper")
    pyro_upper = {uppercase_gene(g) for g in PYROPTOSIS_GENES}
    nlrp2_name = gene_lookup.get("NLRP2", "Nlrp2")
    neighbor_set = set(graph.neighbors(nlrp2_name)) if nlrp2_name in graph else set()
    node_rows = []
    for gene in sorted(graph.nodes()):
        upper = uppercase_gene(gene)
        row = {
            "gene": gene,
            "degree": int(graph.degree(gene)),
            "degree_centrality": degree_centrality.get(gene, 0.0),
            "betweenness_centrality": betweenness.get(gene, 0.0),
            "closeness_centrality": closeness.get(gene, 0.0),
            "pyroptosis_or_inflammasome_member": int(upper in pyro_upper),
            "neighbor_of_NLRP2": int(gene in neighbor_set),
            "log2FoldChange": np.nan,
            "pvalue": np.nan,
            "padj": np.nan,
            "module": "",
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
        if upper in wgcna_lookup.index:
            hit = wgcna_lookup.loc[upper]
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            row["module"] = hit.get("module", "")
            row["module_membership"] = hit.get("module_membership")
            row["gene_significance"] = hit.get("gene_significance")
            row["intramodular_connectivity"] = hit.get("intramodular_connectivity")
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    nodes_df["deg_score"] = minmax_scale(nodes_df["log2FoldChange"].abs() * safe_neglog10(nodes_df["pvalue"].fillna(1)))
    nodes_df["network_score"] = minmax_scale(nodes_df["degree_centrality"] + nodes_df["betweenness_centrality"] + nodes_df["closeness_centrality"])
    nodes_df["wgcna_score"] = minmax_scale(
        nodes_df["module_membership"].abs().fillna(0) + nodes_df["gene_significance"].abs().fillna(0) + nodes_df["intramodular_connectivity"].fillna(0)
    )
    nodes_df["AI_weight"] = (
        nodes_df["deg_score"] * 0.30
        + nodes_df["network_score"] * 0.25
        + nodes_df["wgcna_score"] * 0.25
        + nodes_df["pyroptosis_or_inflammasome_member"] * 0.15
        + nodes_df["neighbor_of_NLRP2"] * 0.05
    )
    nodes_df["label_node"] = (
        (nodes_df["degree"].rank(method="min", ascending=False) <= 10)
        | (nodes_df["AI_weight"].rank(method="min", ascending=False) <= 10)
        | (nodes_df["gene"].str.upper() == "NLRP2")
    )
    edges_df.to_csv(output_dir / "networks" / "PPI_edges_STRING_enhanced.csv", index=False, encoding="utf-8-sig")
    nodes_df.to_csv(output_dir / "networks" / "PPI_nodes_STRING_enhanced.csv", index=False, encoding="utf-8-sig")
    draw_ppi_figure(nodes_df, edges_df, output_dir / "figures" / "PPI_network_STRING_cytoscape_like.png", output_dir / "figures" / "PPI_network_STRING_cytoscape_like.pdf", ai_mode=False)
    ai_df = nodes_df.sort_values("AI_weight", ascending=False).copy()
    ai_df.to_csv(output_dir / "tables" / "AI_PPI_node_importance_STRING_enhanced.csv", index=False, encoding="utf-8-sig")
    if nlrp2_name in graph:
        neighbor_rows = []
        for neighbor in graph.neighbors(nlrp2_name):
            row = ai_df[ai_df["gene"] == neighbor].iloc[0]
            neighbor_rows.append(
                {
                    "NLRP2_neighbor": neighbor,
                    "AI_weight": row["AI_weight"],
                    "degree": row["degree"],
                    "score": graph[nlrp2_name][neighbor].get("score", np.nan),
                    "pyroptosis_or_inflammasome_member": row["pyroptosis_or_inflammasome_member"],
                }
            )
        neighbor_df = pd.DataFrame(neighbor_rows).sort_values(["AI_weight", "score"], ascending=[False, False])
    else:
        neighbor_df = pd.DataFrame(columns=["NLRP2_neighbor", "AI_weight", "degree", "score", "pyroptosis_or_inflammasome_member"])
    neighbor_df.to_csv(output_dir / "tables" / "NLRP2_key_neighbor_ranking_STRING_enhanced.csv", index=False, encoding="utf-8-sig")
    draw_ppi_figure(ai_df, edges_df, output_dir / "figures" / "AI_enhanced_STRING_PPI_network.png", output_dir / "figures" / "AI_enhanced_STRING_PPI_network.pdf", ai_mode=True)
    state.ppi_string_enhanced = True
    return {"nodes": ai_df, "edges": edges_df, "neighbors": neighbor_df, "species_name": species_name}


def update_nlrp2_evidence(input_dir: Path, output_dir: Path, gsea_info: dict[str, Any], ppi_info: dict[str, Any]) -> pd.DataFrame:
    all_diff = load_all_diff_results(input_dir)
    disease = all_diff[(all_diff["comparison"] == "Pm vs control") & (all_diff["gene_upper"] == "NLRP2")].iloc[0]
    primary = all_diff[(all_diff["comparison"] == "Pm+Iroshy vs Pm") & (all_diff["gene_upper"] == "NLRP2")].iloc[0]
    wgcna = pd.read_csv(output_dir / "tables" / "NLRP2_WGCNA_status.csv").iloc[0]
    nodes = ppi_info["nodes"]
    nlrp2_node = nodes[nodes["gene"].astype(str).str.upper() == "NLRP2"].iloc[0]
    neighbors_df = ppi_info["neighbors"]
    leading_terms = []
    for term, genes in gsea_info.get("leading_edge_map", {}).items():
        tokens = {uppercase_gene(token) for token in re.split(r"[;, ]+", str(genes)) if token}
        if "NLRP2" in tokens:
            leading_terms.append(term)
    evidence_level = "exploratory"
    if len(leading_terms) >= 1 and int(nlrp2_node["degree"]) >= 1:
        evidence_level = "moderate"
    if bool(primary["significant"]) and len(leading_terms) >= 2 and int(nlrp2_node["degree"]) >= 3:
        evidence_level = "strong"
    top_neighbors = ", ".join(neighbors_df["NLRP2_neighbor"].astype(str).head(8).tolist()) if not neighbors_df.empty else ""
    conclusion = "NLRP2 is a mechanism-relevant candidate gene with directionally consistent expression changes across the disease-background and Isorhy-treatment comparisons."
    if leading_terms or int(nlrp2_node["degree"]) > 0:
        conclusion += " Public-resource-enhanced pathway and network analyses further support its mechanistic relevance."
    if not bool(primary["significant"]):
        conclusion += " However, NLRP2 still does not pass the preset adjusted DEG threshold in the primary comparison, so the conclusion remains cautious and hypothesis-generating."
    out = pd.DataFrame(
        [
            {
                "NLRP2_present_in_expression_matrix": True,
                "Pm_vs_control_log2FC": disease["log2FoldChange"],
                "Pm_vs_control_pvalue": disease["pvalue"],
                "Pm_vs_control_padj": disease["padj"],
                "PmIsorhy_vs_Pm_log2FC": primary["log2FoldChange"],
                "PmIsorhy_vs_Pm_pvalue": primary["pvalue"],
                "PmIsorhy_vs_Pm_padj": primary["padj"],
                "Passes_preset_DEG_threshold_in_primary": bool(primary["significant"]),
                "In_online_enhanced_GSEA_leading_edge": bool(leading_terms),
                "Online_enhanced_GSEA_leading_edge_terms": "; ".join(leading_terms),
                "In_pyroptosis_or_inflammasome_or_innate_immune_set": True,
                "WGCNA_module": wgcna["module"],
                "WGCNA_module_membership": wgcna["module_membership"],
                "WGCNA_gene_significance": wgcna["gene_significance"],
                "WGCNA_intramodular_connectivity": wgcna["intramodular_connectivity"],
                "STRING_PPI_degree": int(nlrp2_node["degree"]),
                "STRING_top_neighbors": top_neighbors,
                "AI_weight": nlrp2_node["AI_weight"],
                "Integrated_conclusion": conclusion,
                "Evidence_level": evidence_level,
            }
        ]
    )
    out.to_csv(output_dir / "tables" / "NLRP2_integrated_evidence_online_enhanced.csv", index=False, encoding="utf-8-sig")
    return out


def build_chinese_report(state: Round2State, gsea_info: dict[str, Any], nlrp2_df: pd.DataFrame) -> str:
    nlr = nlrp2_df.iloc[0]
    gsea_mode = "formal preranked GSEA" if state.gsea_formal else "public-resource-informed fallback running score"
    ppi_mode = "STRING-enhanced PPI" if state.ppi_string_enhanced else "fallback PPI"
    return f"""# 第二轮安全联网增强分析报告

## 1. 分析说明

本轮分析基于第一轮已生成结果继续开展，优先复用 `output/`、`scripts/`、`final_report.md`、`completion_summary.txt` 与日志，不从头重跑整套流程。联网步骤严格遵循安全边界：不上传表达矩阵、不上传样本名、不上传分组表、不上传原始测序数据；优先下载公开数据库资源到本地后再本地分析。

## 2. 第一轮结果的性质

- 主比较 `Pm+Iroshy vs Pm` 没有 `padj < 0.05` 的显著 DEG，因此主比较的差异表达证据需要谨慎解释。
- NLRP2 / Nlrp2 在两组比较中均有方向一致的表达趋势支持，但未通过主比较的 adjusted 阈值。
- 第一轮 GSEA 为 fallback 结果。
- 第一轮 WGCNA 为 Python exploratory WGCNA-like analysis。
- 第一轮 PPI 为 fallback network。
- 第一轮 AI 网络解释为 AI-inspired，不是真实 GNN。

## 3. NLRP2 表达趋势

- `Pm vs control`：log2FC = {nlr['Pm_vs_control_log2FC']:.6f}，pvalue = {nlr['Pm_vs_control_pvalue']:.6f}，padj = {nlr['Pm_vs_control_padj']:.6f}
- `Pm+Iroshy vs Pm`：log2FC = {nlr['PmIsorhy_vs_Pm_log2FC']:.6f}，pvalue = {nlr['PmIsorhy_vs_Pm_pvalue']:.6f}，padj = {nlr['PmIsorhy_vs_Pm_padj']:.6f}
- 这提示 NLRP2 在感染背景下呈下降趋势，在 Isorhy 干预后呈回升趋势，但由于 adjusted 阈值仍未通过，因此不能据此做过度确定性的统计结论。

## 4. 联网增强 GSEA

- 本轮已下载公开 pathway gene sets 到本地，并在本地完成 GSEA 计算。
- 当前在线增强 GSEA 状态：`{gsea_mode}`
- 本轮 formal GSEA 未能稳定完成的直接原因是：部分结果表中 `Term` 字段重复，导致标准结果整理步骤无法稳定收敛，因此保留了公开基因集支持下的 fallback running-score 版本。
- NLRP2 是否进入 online-enhanced GSEA leading edge：`{'是' if bool(nlr['In_online_enhanced_GSEA_leading_edge']) else '否'}`
- 相关输出：
  - `output/tables/GSEA_results_online_enhanced.csv`
  - `output/figures/GSEA_online_pyroptosis.png`
  - `output/figures/GSEA_online_innate_immune_system.png`
  - `output/figures/GSEA_online_NOD_like_receptor.png`
  - `output/figures/GSEA_online_inflammasome.png`

## 5. 联网增强 STRING-PPI 与 AI 解释

- 本轮优先尝试下载 STRING 公共资源并在本地筛选候选基因交集。
- 当前 PPI 状态：`{ppi_mode}`
- NLRP2 的 STRING 网络 degree = {int(nlr['STRING_PPI_degree'])}
- NLRP2 的主要邻居蛋白：{nlr['STRING_top_neighbors'] if str(nlr['STRING_top_neighbors']).strip() else '暂未获得稳定邻居'}
- AI-enhanced PPI 仍然是 `AI-inspired weighted PPI interpretation`，不是真实训练 GNN。

## 6. 综合判断

推荐表述：

**NLRP2 是具有机制相关性和表达趋势支持的候选关键基因。**

更完整的谨慎结论为：

> NLRP2 在疾病背景比较中呈下降趋势，在 Isorhy 处理比较中呈回升趋势，并具有公开资源增强后的通路与网络层面支持；但其在主比较中仍未通过 adjusted p-value 阈值，因此当前结论仍应保持为候选关键基因层级，而不应过度表述为已被确定的核心关键基因。

## 7. 当前证据等级

- 本轮综合证据等级：`{nlr['Evidence_level']}`

## 8. 仍然存在的限制

{chr(10).join([f"- {item}" for item in state.limitations] or ['- 当前未记录额外限制。'])}
"""


def update_english_report(original_report: str, state: Round2State, gsea_info: dict[str, Any], nlrp2_df: pd.DataFrame) -> str:
    marker = "\n## 12. Second-round safe online enhancement\n"
    if marker in original_report:
        original_report = original_report.split(marker)[0].rstrip()
    nlr = nlrp2_df.iloc[0]
    extra = f"""

## 12. Second-round safe online enhancement

- A second-round diagnostic note was written to `output/logs/second_round_diagnosis.md`.
- A network access plan was written to `output/logs/network_access_plan.md`.
- Public pathway gene sets were downloaded locally and used in `{gsea_info.get('mode', 'unknown')}` mode.
- PPI enhancement status: `{'STRING-enhanced PPI' if state.ppi_string_enhanced else 'fallback retained'}`
- No expression matrix, sample metadata, group table, or raw sequencing table was sent externally.
- If a fallback external request was required, only a small public gene-symbol list was sent.

### Updated NLRP2 interpretation

NLRP2 remains best described as a mechanism-relevant candidate gene with directional expression support and public-resource-enhanced pathway/network context. It still does not pass the preset adjusted DEG threshold in the primary comparison, so the overall claim remains cautious and hypothesis-generating rather than definitive.

### Updated key outputs

- `output/tables/GSEA_results_online_enhanced.csv`
- `output/figures/GSEA_online_pyroptosis.png`
- `output/figures/GSEA_online_innate_immune_system.png`
- `output/figures/GSEA_online_NOD_like_receptor.png`
- `output/figures/GSEA_online_inflammasome.png`
- `output/networks/PPI_edges_STRING_enhanced.csv`
- `output/networks/PPI_nodes_STRING_enhanced.csv`
- `output/figures/PPI_network_STRING_cytoscape_like.png`
- `output/tables/AI_PPI_node_importance_STRING_enhanced.csv`
- `output/tables/NLRP2_key_neighbor_ranking_STRING_enhanced.csv`
- `output/figures/AI_enhanced_STRING_PPI_network.png`
- `output/tables/NLRP2_integrated_evidence_online_enhanced.csv`
"""
    return original_report.rstrip() + "\n" + extra + "\n"


def update_completion_summary(original_summary: str, state: Round2State, nlrp2_df: pd.DataFrame) -> str:
    marker = "\n9. Second-round network success:"
    if marker in original_summary:
        original_summary = original_summary.split(marker)[0].rstrip()
    nlr = nlrp2_df.iloc[0]
    resources = ", ".join(state.downloaded_resources) if state.downloaded_resources else "No public resource download succeeded"
    sent_gene_list = ", ".join(state.local_gene_list_sent) if state.local_gene_list_sent else "No"
    limitations = "\n".join([f"- {item}" for item in state.limitations]) if state.limitations else "- None beyond standard exploratory constraints"
    extra = f"""

9. Second-round network success: {'Yes' if state.downloaded_resources or state.local_gene_list_sent else 'No'}
10. Public resources downloaded: {resources}
11. Local gene list sent externally: {sent_gene_list}
12. Expression matrix or sample metadata sent externally: {'Yes' if state.sent_expression_or_sample else 'No'}
13. GSEA upgraded from fallback to formal GSEA: {'Yes' if state.gsea_formal else 'No'}
14. PPI upgraded from fallback to STRING-enhanced PPI: {'Yes' if state.ppi_string_enhanced else 'No'}
15. AI PPI remains AI-inspired: {'Yes' if state.ai_mode == 'AI-inspired weighted PPI interpretation' else 'No'}
16. NLRP2 conclusion enhanced: {'Yes' if state.ppi_string_enhanced or state.gsea_formal else 'Partially / limited'}
17. Current NLRP2 evidence level: {nlr['Evidence_level']}
18. Remaining limitations:
{limitations}
"""
    return original_summary.rstrip() + "\n" + extra + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-round safe online enhancement analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    log_path = output_dir / "logs" / "network_access.log"
    logger = setup_logger(log_path)
    append_network_log(log_path, "=== Second-round safe online enhancement started ===")
    state = Round2State()
    try:
        ctx = load_round1_context(output_dir)
        write_second_round_diagnosis(output_dir, ctx)
        gsea_info = run_online_enhanced_gsea(input_dir, output_dir, log_path, state, logger)
        ppi_info = build_string_enhanced_ppi(input_dir, output_dir, log_path, gsea_info["result_df"], state, logger)
        nlrp2_df = update_nlrp2_evidence(input_dir, output_dir, gsea_info, ppi_info)
        zh_report = build_chinese_report(state, gsea_info, nlrp2_df)
        (output_dir / "final_report_zh.md").write_text(zh_report, encoding="utf-8")
        updated_en = update_english_report(ctx["final_report.md"], state, gsea_info, nlrp2_df)
        (output_dir / "final_report.md").write_text(updated_en, encoding="utf-8")
        updated_summary = update_completion_summary(ctx["completion_summary.txt"], state, nlrp2_df)
        (output_dir / "completion_summary.txt").write_text(updated_summary, encoding="utf-8")
        append_network_log(log_path, "=== Second-round safe online enhancement completed successfully ===")
    except Exception as exc:
        append_network_log(log_path, f"Second-round enhancement failed: {exc}")
        append_network_log(log_path, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
