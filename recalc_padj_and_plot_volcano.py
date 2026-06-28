from __future__ import annotations

from pathlib import Path

import math
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "差异分析结果表.xlsx"
OUTPUT_DIR = ROOT / "output" / "volcano_recalc"
OUTPUT_XLSX = OUTPUT_DIR / "差异分析结果表_padj重算_含火山图结果.xlsx"
OUTPUT_PNG = OUTPUT_DIR / "volcano_plots_recalc_pvalue.png"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary_stats.csv"

PVALUE_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
LABEL_TOP_N = 8

PANEL_CONFIGS = [
    {"sheet": "Pm vs control", "title": "Pm vs Control"},
    {"sheet": "Pm+Iroshy vs Pm", "title": "Pm+Iroshy vs Pm"},
]


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return pd.Series(out, index=pvalues.index)
    order = np.argsort(pv)
    ranks = np.arange(1, pv.size + 1, dtype=float)
    adjusted = pv[order] * pv.size / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(pv)
    restored[order] = adjusted
    out[mask] = restored
    return pd.Series(out, index=pvalues.index)


def prepare_sheet(df: pd.DataFrame, sheet_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    out["pvalue_used"] = pd.to_numeric(out["pvalue"], errors="coerce")
    out["padj_original"] = pd.to_numeric(out["padj"], errors="coerce")
    out["padj_recalc"] = bh_adjust(out["pvalue_used"])
    out["log2FoldChange_num"] = pd.to_numeric(out["log2FoldChange"], errors="coerce")
    out["minus_log10_pvalue"] = -np.log10(out["pvalue_used"].clip(lower=np.nextafter(0, 1)))

    out["sig_by_pvalue"] = out["pvalue_used"] < PVALUE_THRESHOLD
    out["sig_by_padj_recalc"] = out["padj_recalc"] < 0.05
    out["abs_log2fc_ge_1"] = out["log2FoldChange_num"].abs() >= LOG2FC_THRESHOLD

    conditions = [
        out["sig_by_pvalue"] & (out["log2FoldChange_num"] >= LOG2FC_THRESHOLD),
        out["sig_by_pvalue"] & (out["log2FoldChange_num"] <= -LOG2FC_THRESHOLD),
    ]
    choices = ["Up", "Down"]
    out["volcano_group"] = np.select(conditions, choices, default="Not significant")

    padj_na = int(out["padj_original"].isna().sum())
    padj_na_rate = padj_na / len(out) if len(out) else math.nan
    summary = {
        "sheet": sheet_name,
        "rows": int(len(out)),
        "padj_original_na": padj_na,
        "padj_original_na_rate": padj_na_rate,
        "pvalue_lt_0_05": int(out["sig_by_pvalue"].sum()),
        "padj_recalc_lt_0_05": int(out["sig_by_padj_recalc"].sum()),
        "abs_log2fc_ge_1": int(out["abs_log2fc_ge_1"].sum()),
        "up_by_pvalue": int((out["volcano_group"] == "Up").sum()),
        "down_by_pvalue": int((out["volcano_group"] == "Down").sum()),
        "not_significant_by_pvalue": int((out["volcano_group"] == "Not significant").sum()),
        "padj_recalc_min": float(out["padj_recalc"].min()),
        "padj_recalc_max": float(out["padj_recalc"].max()),
        "note": (
            "Input appears pre-filtered because all rows satisfy pvalue<0.05 and |log2FoldChange|>=1 "
            "within this workbook sheet."
        ),
    }
    return out, summary


def choose_labels(df: pd.DataFrame, top_n: int = LABEL_TOP_N) -> pd.DataFrame:
    labeled = df.loc[df["volcano_group"] != "Not significant"].copy()
    if labeled.empty:
        return labeled
    labeled["label_score"] = labeled["minus_log10_pvalue"] * labeled["log2FoldChange_num"].abs()
    labeled = labeled.sort_values(
        by=["label_score", "minus_log10_pvalue", "log2FoldChange_num"],
        ascending=[False, False, False],
    )
    labeled = labeled.drop_duplicates(subset=["gene_name"], keep="first")
    return labeled.head(top_n)


def draw_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    colors = {
        "Up": "#f4a582",
        "Down": "#92c5de",
        "Not significant": "#d9d9d9",
    }

    for group in ["Not significant", "Down", "Up"]:
        sub = df[df["volcano_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["log2FoldChange_num"],
            sub["minus_log10_pvalue"],
            s=6,
            c=colors[group],
            edgecolors="none",
            alpha=0.75 if group != "Not significant" else 0.6,
            label=group,
            rasterized=True,
        )

    ax.axvline(-LOG2FC_THRESHOLD, color="#bdbdbd", linestyle="--", linewidth=0.9)
    ax.axvline(0, color="#8c8c8c", linestyle=":", linewidth=0.9)
    ax.axvline(LOG2FC_THRESHOLD, color="#bdbdbd", linestyle="--", linewidth=0.9)
    ax.axhline(-np.log10(PVALUE_THRESHOLD), color="#8c8c8c", linestyle="--", linewidth=0.9)

    texts = []
    for _, row in choose_labels(df).iterrows():
        gene = row["gene_name"] if pd.notna(row["gene_name"]) and str(row["gene_name"]).strip() else row["Geneid"]
        color = "#c65b7c" if row["volcano_group"] == "Up" else "#5f5aa2"
        texts.append(
            ax.text(
                row["log2FoldChange_num"],
                row["minus_log10_pvalue"],
                str(gene),
                fontsize=8,
                color=color,
                weight="bold",
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#8f8f8f", lw=0.7),
            expand_points=(1.2, 1.3),
            expand_text=(1.2, 1.3),
            force_points=0.2,
            force_text=0.4,
        )

    up_n = int((df["volcano_group"] == "Up").sum())
    down_n = int((df["volcano_group"] == "Down").sum())
    ns_n = int((df["volcano_group"] == "Not significant").sum())
    legend_labels = [
        f"Up (p<{PVALUE_THRESHOLD}, n={up_n})",
        f"Down (p<{PVALUE_THRESHOLD}, n={down_n})",
        f"Not significant (n={ns_n})",
    ]
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors["Up"], markersize=6),
        plt.Line2D([], [], marker="o", linestyle="", color=colors["Down"], markersize=6),
        plt.Line2D([], [], marker="o", linestyle="", color=colors["Not significant"], markersize=6),
    ]
    ax.legend(handles, legend_labels, loc="upper right", frameon=True, framealpha=0.95, fontsize=8)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel(r"log$_2$ Fold Change", fontsize=11)
    ax.set_ylabel(r"-log$_{10}$(p-value)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)


def write_excel(results: dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for sheet_name, df in results.items():
            export_df = df.copy()
            helper_cols = [
                "pvalue_used",
                "padj_original",
                "padj_recalc",
                "log2FoldChange_num",
                "minus_log10_pvalue",
                "sig_by_pvalue",
                "sig_by_padj_recalc",
                "abs_log2fc_ge_1",
                "volcano_group",
            ]
            preferred = [
                c
                for c in [
                    "Geneid",
                    "Chr",
                    "gene_name",
                    "baseMean",
                    "log2FoldChange",
                    "log2FoldChange_num",
                    "pvalue",
                    "pvalue_used",
                    "padj",
                    "padj_original",
                    "padj_recalc",
                    "minus_log10_pvalue",
                    "sig_by_pvalue",
                    "sig_by_padj_recalc",
                    "abs_log2fc_ge_1",
                    "volcano_group",
                ]
                if c in export_df.columns
            ]
            remaining = [c for c in export_df.columns if c not in preferred]
            export_df = export_df[preferred + remaining]
            safe_sheet = sheet_name[:31]
            export_df.to_excel(writer, sheet_name=safe_sheet, index=False)

            ws = writer.book[safe_sheet]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 24)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workbook = pd.ExcelFile(INPUT_XLSX)
    results: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for config in PANEL_CONFIGS:
        sheet = config["sheet"]
        df = pd.read_excel(INPUT_XLSX, sheet_name=sheet)
        prepared, summary = prepare_sheet(df, sheet)
        results[sheet] = prepared
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    write_excel(results, summary_df)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=220)
    fig.suptitle("Differential Expression: Volcano Plots", fontsize=16, weight="bold", y=0.98)

    for ax, config in zip(axes, PANEL_CONFIGS):
        draw_panel(ax, results[config["sheet"]], config["title"])

    note = textwrap.fill(
        "Volcano plots use p-value for significance coloring because the original padj column contains many NA values. "
        "A new padj_recalc column was recomputed from the p-value column using Benjamini-Hochberg correction within each sheet. "
        "Because this workbook appears pre-filtered, all plotted points are already significant by p<0.05 and |log2FC|>=1.",
        width=120,
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9, color="#4d4d4d")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved Excel: {OUTPUT_XLSX}")
    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
