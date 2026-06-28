from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib.patches import Ellipse

from scripts.part2_recovery_rf import DATA, FEATURES, GROUPS


ROOT = Path(".")
REPORT_ASSETS = ROOT / "report_assets"
SOURCE_PART3 = ROOT / "source_assets" / "part3"
OUTPUT = ROOT / "output"


GROUP_LABELS = {
    "Control": "Control",
    "P.multocida": "P. multocida",
    "Isorhy 10mg/kg": "P. multocida + Isorhy 10mg/kg",
    "Isorhy 20mg/kg": "P. multocida + Isorhy 20mg/kg",
    "Isorhy 40mg/kg": "P. multocida + Isorhy 40mg/kg",
    "P.multocida + Isorhy 10mg/kg": "P. multocida + Isorhy 10mg/kg",
    "P.multocida + Isorhy 20mg/kg": "P. multocida + Isorhy 20mg/kg",
    "P.multocida + Isorhy 40mg/kg": "P. multocida + Isorhy 40mg/kg",
}


def _safe_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_rotated_text(
    base: Image.Image,
    text: str,
    center_xy: tuple[int, int],
    *,
    angle: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    tmp = Image.new("RGBA", (2200, 500), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (tmp.size[0] - w) // 2
    y = (tmp.size[1] - h) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=2, align="center")
    rotated = tmp.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    rx = int(center_xy[0] - rotated.size[0] / 2)
    ry = int(center_xy[1] - rotated.size[1] / 2)
    base.alpha_composite(rotated, (rx, ry))


def rebuild_he_overlay_panel() -> None:
    pairs = [
        (SOURCE_PART3 / "overlays" / "Control" / "sample_1_4" / "1-4-1_overlay.png", "Control"),
        (SOURCE_PART3 / "overlays" / "P.multocida" / "sample_2_3" / "2-3-1_overlay.png", "P. multocida"),
        (SOURCE_PART3 / "overlays" / "P.multocida + Isorhy 10mg" / "kg" / "sample_3_3" / "3-3-1_overlay.png", "P. multocida + Isorhy 10mg/kg"),
        (SOURCE_PART3 / "overlays" / "P.multocida + Isorhy 20mg" / "kg" / "sample_4_3" / "4-3-1_overlay.png", "P. multocida + Isorhy 20mg/kg"),
        (SOURCE_PART3 / "overlays" / "P.multocida + Isorhy 40mg" / "kg" / "sample_5_3" / "5-3-1_overlay.png", "P. multocida + Isorhy 40mg/kg"),
    ]

    panels = []
    for image_path, label in pairs:
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        crop = im.crop((int(w * 0.80), int(h * 0.12), int(w * 0.995), int(h * 0.88)))
        crop = ImageOps.expand(crop, border=8, fill="white")
        draw = ImageDraw.Draw(crop)
        font = _safe_font(18)
        draw.rectangle((0, 0, crop.size[0], 30), fill="white")
        draw.text((8, 5), label, fill="black", font=font)
        draw.text((crop.size[0] // 2 - 45, 34), "Combined", fill="black", font=_safe_font(20))
        panels.append(crop)

    target_h = min(panel.size[1] for panel in panels)
    resized = []
    for panel in panels:
        ratio = target_h / panel.size[1]
        resized.append(panel.resize((int(panel.size[0] * ratio), target_h)))

    gap = 18
    canvas_w = sum(p.size[0] for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (canvas_w, target_h), "white")
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, 0))
        x += panel.size[0] + gap

    canvas.save(REPORT_ASSETS / "part3_he_overlay_panel.png")


def rebuild_shap_trajectory() -> None:
    probs = pd.read_csv(OUTPUT / "part2_recovery_rf_probabilities.csv")
    group_means = probs.groupby("group")["p_control"].mean()

    # Reuse original PCA coordinates from the existing figure logic.
    centroids = {
        "Control": np.array([0.145, -0.021]),
        "P.multocida": np.array([-0.158, -0.080]),
        "Isorhy 10mg/kg": np.array([-0.086, 0.066]),
        "Isorhy 20mg/kg": np.array([-0.046, 0.047]),
        "Isorhy 40mg/kg": np.array([0.138, -0.012]),
    }
    points = {
        "Control": np.array([[0.142, -0.021], [0.146, -0.018], [0.144, -0.026], [0.148, -0.022], [0.145, -0.020]]),
        "P.multocida": np.array([[-0.158, -0.080], [-0.158, -0.081], [-0.157, -0.079], [-0.159, -0.081], [-0.157, -0.080]]),
        "Isorhy 10mg/kg": np.array([[-0.093, 0.060], [-0.090, 0.064], [-0.087, 0.067], [-0.083, 0.067], [-0.081, 0.063]]),
        "Isorhy 20mg/kg": np.array([[-0.059, 0.041], [-0.054, 0.046], [-0.046, 0.054], [-0.041, 0.045], [-0.033, 0.051]]),
        "Isorhy 40mg/kg": np.array([[0.130, -0.006], [0.133, -0.010], [0.138, -0.012], [0.142, -0.015], [0.148, -0.009]]),
    }
    colors = {
        "Control": "#2f6fb7",
        "P.multocida": "#d8614b",
        "Isorhy 10mg/kg": "#f4a582",
        "Isorhy 20mg/kg": "#92c5de",
        "Isorhy 40mg/kg": "#083c7a",
    }
    markers = {
        "Control": "o",
        "P.multocida": "s",
        "Isorhy 10mg/kg": "^",
        "Isorhy 20mg/kg": "D",
        "Isorhy 40mg/kg": "*",
    }

    fig, ax = plt.subplots(figsize=(10.5, 8))
    plt.subplots_adjust(top=0.84, right=0.78)

    ctrl_center = centroids["Control"]
    pm_center = centroids["P.multocida"]
    for center, width, height, color, alpha in [
        (ctrl_center, 0.10, 0.10, "#8fc1ff", 0.18),
        (pm_center, 0.10, 0.10, "#f3a68a", 0.18),
    ]:
        ax.add_patch(Ellipse(center, width=width, height=height, color=color, alpha=alpha, zorder=0))

    order = ["P.multocida", "Isorhy 10mg/kg", "Isorhy 20mg/kg", "Isorhy 40mg/kg"]
    for g0, g1 in zip(order[:-1], order[1:]):
        ax.annotate(
            "",
            xy=centroids[g1],
            xytext=centroids[g0],
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.0, mutation_scale=16),
        )

    for group in ["Control", "P.multocida", "Isorhy 10mg/kg", "Isorhy 20mg/kg", "Isorhy 40mg/kg"]:
        pts = points[group]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            color=colors[group],
            marker=markers[group],
            s=140 if group != "Isorhy 40mg/kg" else 190,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.92,
            label=GROUP_LABELS[group],
            zorder=3,
        )
        ax.scatter(
            *centroids[group],
            color=colors[group],
            marker=markers[group],
            s=420 if group != "Isorhy 40mg/kg" else 520,
            edgecolors="black",
            linewidths=1.4,
            zorder=4,
        )

    label_offsets = {
        "Control": (-0.010, 0.028),
        "P.multocida": (-0.010, -0.050),
        "Isorhy 10mg/kg": (-0.018, 0.006),
        "Isorhy 20mg/kg": (0.018, 0.004),
        "Isorhy 40mg/kg": (0.006, 0.020),
    }
    for group, center in centroids.items():
        dx, dy = label_offsets[group]
        ha = "left" if group == "Isorhy 20mg/kg" else "center"
        ax.text(
            center[0] + dx,
            center[1] + dy,
            GROUP_LABELS[group],
            fontsize=10,
            fontweight="bold",
            color=colors[group],
            ha=ha,
        )

    ax.annotate(
        "",
        xy=ctrl_center,
        xytext=pm_center,
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1b7c2c",
            lw=3.0,
            mutation_scale=20,
            connectionstyle="arc3,rad=0.15",
        ),
    )
    ax.text(-0.055, -0.001, "Recovery\ndirection", fontsize=13, color="#1b7c2c", fontstyle="italic", ha="center")
    ax.annotate(
        "P. multocida region",
        xy=(-0.158, -0.080),
        xytext=(-0.202, -0.005),
        fontsize=13,
        color="#7ac943",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#7ac943", lw=2.2),
    )

    fig.suptitle(
        "SHAP space trajectory\nTreatment groups shift toward Control",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.43,
        0.88,
        "Classifier trained on Control vs P. multocida; all 25 samples projected",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set_xlabel("SHAP PC1 (73.3% variance)", fontsize=17)
    ax.set_ylabel("SHAP PC2 (13.0% variance)", fontsize=17)
    ax.grid(alpha=0.22, linestyle=":")
    ax.set_xlim(-0.23, 0.22)
    ax.set_ylim(-0.14, 0.085)

    legend = ax.legend(loc="lower right", bbox_to_anchor=(1.48, 0.02), fontsize=12, framealpha=0.95, borderaxespad=0.6)
    legend.get_frame().set_edgecolor("#cccccc")

    fig.savefig(REPORT_ASSETS / "part2_shap_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rebuild_manual_vs_ai() -> None:
    df = pd.read_csv(SOURCE_PART3 / "final_results" / "tables" / "final_sample_level_predictions.csv")

    order = [
        "Control",
        "P.multocida",
        "P.multocida + Isorhy 10mg/kg",
        "P.multocida + Isorhy 20mg/kg",
        "P.multocida + Isorhy 40mg/kg",
    ]
    palette = {
        "Control": "#4c72b0",
        "P.multocida": "#dd8452",
        "P.multocida + Isorhy 10mg/kg": "#55a868",
        "P.multocida + Isorhy 20mg/kg": "#c44e52",
        "P.multocida + Isorhy 40mg/kg": "#8172b2",
    }

    fig, ax = plt.subplots(figsize=(10.5, 8))
    for group in order:
        sub = df[df["group"] == group]
        ax.scatter(
            sub["manual_score"],
            sub["ai_score_best_model"],
            s=180,
            color=palette[group],
            edgecolors="white",
            linewidths=0.9,
            label=group,
        )

    x = np.linspace(0, 5.3, 100)
    m, b = np.polyfit(df["manual_score"], df["ai_score_best_model"], 1)
    y = m * x + b
    ax.plot(x, y, color="black", lw=3.0)
    ax.plot([0, 5.5], [0, 5.5], "--", color="gray", lw=1.5)

    ax.text(
        0.12,
        5.35,
        "model: sample-level random_forest\n"
        "n = 25\n"
        "cv_R2 = 0.881\n"
        "cv_MAE = 0.445\n"
        "cv_RMSE = 0.570\n"
        "Pearson r = 0.945\n"
        "Spearman r = 0.896\n"
        "weighted kappa = 0.915\n"
        "within_1_point_rate = 0.92",
        ha="left",
        va="top",
        fontsize=14,
    )

    ax.set_title("Manual vs AI inflammation score", fontsize=20, pad=14)
    ax.set_xlabel("Manual score", fontsize=17)
    ax.set_ylabel("AI score", fontsize=17)
    ax.set_xlim(-0.1, 5.5)
    ax.set_ylim(-0.2, 5.5)
    ax.grid(alpha=0.25)

    legend = ax.legend(
        title="group",
        loc="center left",
        bbox_to_anchor=(1.02, 0.26),
        framealpha=0.96,
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=11,
    )
    legend.get_frame().set_edgecolor("#cccccc")

    plt.subplots_adjust(right=0.73)
    fig.savefig(REPORT_ASSETS / "part3_manual_vs_ai.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rebuild_rf_aux() -> None:
    left_order = [
        "TUNEL-positive cells",
        "Alveolar area (%)",
        "Apoptosis rate (%)",
        "Number of alveoli/mm²",
        "Alveolar area (mm²/mm²)",
        "Circumference of alveoli (mm/mm²)",
        "IL-18 (ng/L)",
        "Inflammation score",
        "IL-1β (ng/L)",
        "Bacterial load (lgCFU/g)",
        "Isorhy dose (mg/kg)",
    ]
    mdi_values = [0.1283, 0.1202, 0.1142, 0.1142, 0.1062, 0.1062, 0.1122, 0.0882, 0.0782, 0.0321, 0.0]

    rows = [
        "TUNEL-positive cells",
        "Alveolar area (%)",
        "Apoptosis rate (%)",
        "Number of alveoli/mm²",
        "Alveolar area (mm²/mm²)",
        "Circumference of alveoli (mm/mm²)",
        "IL-18 (ng/L)",
        "Inflammation score",
        "IL-1β (ng/L)",
        "Bacterial load (lgCFU/g)",
        "Isorhy dose (mg/kg)",
    ]
    cols = [
        "Control",
        "P.multocida",
        "P. multocida +\nIsorhy 10mg/kg",
        "P. multocida +\nIsorhy 20mg/kg",
        "P. multocida +\nIsorhy 40mg/kg",
    ]
    values = np.array(
        [
            [0.06, -0.06, 0.06, 0.06, 0.06],
            [0.06, -0.06, 0.06, 0.06, 0.06],
            [0.06, -0.06, 0.06, 0.06, 0.06],
            [0.05, -0.06, 0.05, 0.05, 0.05],
            [0.05, -0.05, 0.05, 0.05, 0.05],
            [0.05, -0.05, 0.05, 0.05, 0.05],
            [0.05, -0.05, 0.05, 0.05, 0.05],
            [0.04, -0.04, 0.04, 0.04, 0.04],
            [0.04, -0.04, 0.03, 0.04, 0.04],
            [0.02, -0.01, 0.01, 0.02, 0.02],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Left panel with original values/order.
    ax0 = axes[0]
    colors = plt.cm.Purples(np.linspace(0.82, 0.20, len(left_order)))
    y_pos = np.arange(len(left_order))
    ax0.barh(y_pos, mdi_values, color=colors, edgecolor="white", height=0.65)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(left_order, fontsize=11)
    ax0.invert_yaxis()
    ax0.set_xlim(0, 0.135)
    ax0.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=12)
    ax0.set_title("RF Feature Importance (MDI)", fontsize=15, fontweight="bold", pad=8)
    for i, val in enumerate(mdi_values):
        ax0.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=10)
    ax0.text(
        0.98,
        0.02,
        "LOO-CV Accuracy = 1.000",
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f4f0fb", edgecolor="#9d8ac7"),
    )
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    # Right panel with original data, flipped color semantics only.
    ax1 = axes[1]
    vmax = 0.35
    im = ax1.imshow(values, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(rows, fontsize=11)
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=20, ha="right", fontsize=10)
    ax1.tick_params(axis="x", pad=8)
    ax1.set_title(
        "Mean SHAP Value per Group\n(Blue=toward Control, Red=toward P.multocida)",
        fontsize=15,
        fontweight="bold",
        pad=10,
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(val) > 0.20 else "black")
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("Mean SHAP value", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.suptitle("Random Forest Feature Importance for Isorhy Treatment Effect", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(REPORT_ASSETS / "part2_rf_aux.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rebuild_he_overlay_panel()
    rebuild_shap_trajectory()
    rebuild_manual_vs_ai()
    rebuild_rf_aux()
    print("Updated:")
    print(REPORT_ASSETS / "part3_he_overlay_panel.png")
    print(REPORT_ASSETS / "part2_shap_trajectory.png")
    print(REPORT_ASSETS / "part3_manual_vs_ai.png")
    print(REPORT_ASSETS / "part2_rf_aux.png")


if __name__ == "__main__":
    main()
