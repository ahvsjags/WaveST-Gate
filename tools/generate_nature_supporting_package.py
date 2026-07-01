from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nature_supporting_package"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"

for folder in (FIG_DIR, TABLE_DIR, SOURCE_DIR, QA_DIR):
    folder.mkdir(parents=True, exist_ok=True)


COLORS = {
    "navy": "#1f2937",
    "blue": "#4c78a8",
    "cyan": "#72b7b2",
    "teal": "#5f9ea0",
    "green": "#59a14f",
    "gold": "#e2a23a",
    "orange": "#f28e2b",
    "rose": "#b65b6a",
    "purple": "#8f79b8",
    "grey": "#6f7782",
    "light": "#f6f8fa",
    "line": "#d9e1e7",
    "ink": "#17202a",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.4,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


manifest_rows: list[dict[str, str]] = []
missing_rows: list[dict[str, str]] = []


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def note_missing(artifact: str, reason: str, server_hint: str = "/mnt/WaveST-Gate") -> None:
    missing_rows.append({"artifact": artifact, "reason": reason, "server_hint": server_hint})


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        note_missing(str(path), "source CSV missing")
        return pd.DataFrame()
    return pd.read_csv(p, **kwargs)


def read_json(path: str | Path) -> dict:
    p = ROOT / path
    if not p.exists():
        note_missing(str(path), "source JSON missing")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_source(path: str | Path) -> str:
    p = ROOT / path
    if not p.exists():
        note_missing(str(path), "source artifact missing")
        return ""
    dest = SOURCE_DIR / p.name
    if p.is_file():
        shutil.copy2(p, dest)
    return rel(dest)


def save_table(df: pd.DataFrame, name: str, title: str, source: str) -> None:
    if df is None or df.empty:
        df = pd.DataFrame([{"status": "source missing or empty", "source": source}])
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Source: `{source}`\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    manifest_rows.append(
        {
            "type": "table",
            "id": name,
            "title": title,
            "csv": rel(csv_path),
            "markdown": rel(md_path),
            "source": source,
        }
    )


def save_fig(fig: plt.Figure, name: str, title: str, source: str) -> None:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    svg = FIG_DIR / f"{name}.svg"
    tif = FIG_DIR / f"{name}.tif"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(tif, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest_rows.append(
        {
            "type": "figure",
            "id": name,
            "title": title,
            "png": rel(png),
            "pdf": rel(pdf),
            "svg": rel(svg),
            "tif": rel(tif),
            "source": source,
        }
    )


def panel_label(ax, label: str, x: float = -0.045, y: float = 1.055) -> None:
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def title_box(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    clean_title = title.replace("Supplementary Fig. ", "Fig. ")
    fig.text(0.012, 0.985, clean_title, ha="left", va="top", fontsize=8.2, fontweight="normal", color=COLORS["ink"])
    if subtitle:
        wrapped = "\n".join(textwrap.wrap(subtitle, width=118))
        fig.text(0.012, 0.962, wrapped, ha="left", va="top", fontsize=6.5, color=COLORS["grey"], linespacing=1.12)


def clean_ax(ax) -> None:
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.45, alpha=0.7)
    ax.tick_params(colors=COLORS["ink"])


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def draw_mini_table(
    ax,
    rows: list[tuple[str, str]],
    title: str,
    left_width: float = 0.56,
    row_height: float = 0.105,
    fontsize: float = 6.8,
) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, fontweight="bold", color=COLORS["ink"])
    y = 0.86
    for i, (key, value) in enumerate(rows):
        fill = COLORS["light"] if i % 2 == 0 else "white"
        ax.add_patch(
            FancyBboxPatch(
                (0.0, y - row_height + 0.01),
                0.98,
                row_height,
                boxstyle="round,pad=0.006,rounding_size=0.008",
                transform=ax.transAxes,
                linewidth=0.4,
                edgecolor=COLORS["line"],
                facecolor=fill,
            )
        )
        ax.text(0.025, y - 0.045, str(key), transform=ax.transAxes, ha="left", va="center", fontsize=fontsize, color=COLORS["grey"])
        ax.text(left_width, y - 0.045, str(value), transform=ax.transAxes, ha="left", va="center", fontsize=fontsize, color=COLORS["ink"], fontweight="bold")
        y -= row_height


def load_image(path: Path, max_side: int = 1200) -> Image.Image | None:
    if not path.exists():
        note_missing(rel(path), "image source missing")
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def show_image(ax, path: str | Path, title: str = "") -> None:
    im = load_image(ROOT / path if not isinstance(path, Path) else path)
    ax.axis("off")
    if im is None:
        ax.text(0.5, 0.5, "missing image", ha="center", va="center", color=COLORS["rose"])
    else:
        ax.imshow(im)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", color=COLORS["ink"])


def horizontal_bar(ax, labels, values, color=COLORS["blue"], title="", xlabel="") -> None:
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ax.barh(np.array(labels)[order], values[order], color=color, alpha=0.88)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    clean_ax(ax)


def save_all_tables() -> None:
    bench = read_json("results/nature_benchmark_datasheet/benchmark_datasheet.json")
    overview = bench.get("overview", {})
    cell_type_summary = pd.DataFrame(bench.get("cell_type_summary", []))
    split_summary = pd.DataFrame(bench.get("split_summary", []))
    qc = bench.get("qc_summary", {})

    # S1 Dataset/sample/cohort metadata
    core_manifest = read_json("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json")
    data_manifest = ROOT / "data_manifest" / "wavestgate_breast_core.yaml"
    s1 = pd.DataFrame(
        [
            {"item": "primary benchmark", "value": "10x Janesick breast cancer CytAssist/Xenium Rep2", "evidence": rel(data_manifest)},
            {"item": "spots", "value": overview.get("num_spots", core_manifest.get("num_spots", "")), "evidence": "benchmark datasheet"},
            {"item": "Xenium-supervised spots", "value": overview.get("num_spots_with_ground_truth", core_manifest.get("num_spots_with_ground_truth", "")), "evidence": "benchmark datasheet"},
            {"item": "typed Xenium cells", "value": overview.get("num_cells", core_manifest.get("num_cells", "")), "evidence": "benchmark datasheet"},
            {"item": "cell types", "value": overview.get("num_cell_types", core_manifest.get("num_cell_types", "")), "evidence": "benchmark datasheet"},
            {"item": "spot radius", "value": overview.get("spot_radius", core_manifest.get("spot_radius", "")), "evidence": "benchmark manifest"},
        ]
    )
    save_table(s1, "supplementary_table_s1_dataset_metadata", "Supplementary Table S1. Dataset and sample metadata", "data_manifest + benchmark datasheet")

    # S2 Benchmark inventory
    rows = []
    rows += [{"item": k, "value": v} for k, v in overview.items() if not isinstance(v, (dict, list))]
    if split_summary is not None and not split_summary.empty:
        for _, r in split_summary.iterrows():
            rows.append({"item": f"split_{r['split']}_spots", "value": r.get("num_spots")})
            rows.append({"item": f"split_{r['split']}_gt_spots", "value": r.get("num_spots_with_ground_truth")})
    save_table(pd.DataFrame(rows), "supplementary_table_s2_benchmark_inventory", "Supplementary Table S2. Xenium-to-Visium benchmark inventory", "results/nature_benchmark_datasheet")

    # S3 Cell-type dictionary
    save_table(cell_type_summary, "supplementary_table_s3_cell_type_dictionary", "Supplementary Table S3. Cell-type dictionary and benchmark support", "benchmark_datasheet cell_type_summary")

    # S4 Model architecture and hyperparameters
    methods = read_json("results/nature_manuscript_methods/manuscript_methods.json")
    method_rows = []
    for key, value in methods.items():
        if isinstance(value, (str, int, float, bool)):
            method_rows.append({"field": key, "value": value})
        elif isinstance(value, list):
            method_rows.append({"field": key, "value": "; ".join(map(str, value[:8]))})
        elif isinstance(value, dict):
            for kk, vv in value.items():
                if isinstance(vv, (str, int, float, bool)):
                    method_rows.append({"field": f"{key}.{kk}", "value": vv})
    if not method_rows:
        method_rows = [
            {"field": "framework", "value": "WaveST-Gate"},
            {"field": "inputs", "value": "H&E image patches; Visium expression; scRNA/reference prototypes"},
            {"field": "outputs", "value": "cell-type proportions; reconstructed expression; gates; uncertainty; niche assignments"},
        ]
    save_table(pd.DataFrame(method_rows), "supplementary_table_s4_model_hyperparameters", "Supplementary Table S4. Model architecture and hyperparameter summary", "results/nature_manuscript_methods/manuscript_methods.json")

    # S5-S19 mostly direct curated evidence tables
    direct_tables = [
        ("results/nature_manuscript_tables/table_2_main_model_performance.csv", "supplementary_table_s5_main_model_metrics", "Supplementary Table S5. Main model metrics"),
        ("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison_manifest.json", "supplementary_table_s6_baseline_configuration", "Supplementary Table S6. Baseline configuration and protocol"),
        ("results/nature_manuscript_tables/table_3_baseline_comparison.csv", "supplementary_table_s7_baseline_results", "Supplementary Table S7. Full baseline results and paired statistics"),
        ("results/nature_manuscript_tables/table_4_ablation_delta.csv", "supplementary_table_s8_ablation_results", "Supplementary Table S8. Ablation results"),
        ("results/nature_manuscript_tables/table_5_reliability_boundary_niche.csv", "supplementary_table_s9_reliability_calibration", "Supplementary Table S9. Reliability, calibration and risk coverage"),
        ("results/nature_manuscript_tables/table_9_imagegate_supplement.csv", "supplementary_table_s10_he_morphology_contribution", "Supplementary Table S10. H&E morphology contribution controls"),
        ("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_marker_validation.csv", "supplementary_table_s11_boundary_marker_validation", "Supplementary Table S11. Boundary preservation and marker validation"),
        ("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_biological_summary.csv", "supplementary_table_s12_niche_validation", "Supplementary Table S12. Niche composition and biological validation"),
        ("results/nature_manuscript_tables/table_6_external_generalization.csv", "supplementary_table_s13_external_generalization", "Supplementary Table S13. External no-retuning and matched-GT transfer"),
        ("results/nature_manuscript_tables/table_11_rep1_retune_budget_curve.csv", "supplementary_table_s14_rep1_retuning_budget", "Supplementary Table S14. Rep1 minimal-retuning budget curve"),
        ("results/nature_manuscript_tables/table_7_robustness_summary.csv", "supplementary_table_s15_robustness_stress_tests", "Supplementary Table S15. Robustness stress tests"),
        ("results/nature_manuscript_tables/table_10_benchmark_sensitivity.csv", "supplementary_table_s16_benchmark_sensitivity", "Supplementary Table S16. Benchmark sensitivity"),
        ("results/nature_leakage_fairness_audit/leakage_fairness_checklist.csv", "supplementary_table_s17_leakage_fairness_audit", "Supplementary Table S17. Leakage and fairness audit"),
        ("results/nature_release/release_upload_manifest.csv", "supplementary_table_s18_source_data_release_manifest", "Supplementary Table S18. Source data, release and checksum manifest"),
    ]
    for source, name, title in direct_tables:
        p = ROOT / source
        if p.suffix == ".json":
            js = read_json(source)
            rows = []
            for k, v in js.items():
                if isinstance(v, (str, int, float, bool)):
                    rows.append({"field": k, "value": v})
                elif isinstance(v, list):
                    rows.append({"field": k, "value": f"{len(v)} entries"})
                elif isinstance(v, dict):
                    rows.append({"field": k, "value": json.dumps(v, ensure_ascii=False)[:500]})
            save_table(pd.DataFrame(rows), name, title, source)
        else:
            save_table(read_csv(source), name, title, source)

    env = read_json("results/nature_release/environment_report.json")
    baseline = read_csv("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv")
    compute_rows = []
    if not baseline.empty:
        for _, r in baseline.iterrows():
            compute_rows.append(
                {
                    "method": r.get("method"),
                    "runtime_seconds": r.get("runtime_seconds", ""),
                    "peak_cuda_memory_mb": r.get("peak_cuda_memory_mb", ""),
                    "device": r.get("device", ""),
                    "source": "baseline_comparison",
                }
            )
    torch_info = env.get("torch", {})
    for dev in torch_info.get("devices", []):
        compute_rows.append({"method": "hardware", "runtime_seconds": "", "peak_cuda_memory_mb": dev.get("total_memory_mb"), "device": dev.get("name"), "source": "environment_report"})
    save_table(pd.DataFrame(compute_rows), "supplementary_table_s19_compute_hardware_software", "Supplementary Table S19. Compute cost, hardware and software versions", "environment_report + baseline runtime")


def figure_s1() -> None:
    qc = read_csv("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_ground_truth_qc.csv")
    splits = read_csv("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_splits.csv")
    spot_coords = read_csv("data/processed/cytassist_xenium_rep2_common297/spot_coords.csv")
    counts = read_csv("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_cell_counts.csv")
    bench = read_json("results/nature_benchmark_datasheet/benchmark_datasheet.json")
    cell_summary = pd.DataFrame(bench.get("cell_type_summary", []))
    fig = plt.figure(figsize=(11, 7.5), constrained_layout=False)
    title_box(fig, "Supplementary Fig. 1 | Benchmark construction and QC", "Xenium-to-Visium aggregation defines a fixed supervised evaluation target.")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.055, right=0.98, bottom=0.07, top=0.90, hspace=0.38, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    ax.axis("off")
    steps = ["H&E tissue\nregistration", "typed Xenium\nsingle cells", "matched Visium\nspot grid", "55 um radius\naggregation", "spot-level\ncell-type GT"]
    y = np.linspace(0.82, 0.24, len(steps))
    for i, (yy, s) in enumerate(zip(y, steps)):
        box = FancyBboxPatch((0.18, yy - 0.055), 0.64, 0.11, boxstyle="round,pad=0.012,rounding_size=0.006", facecolor="white", edgecolor="#9ab5bf", linewidth=0.8)
        ax.add_patch(box)
        ax.text(0.50, yy, s, ha="center", va="center", fontsize=6.7, color=COLORS["ink"], fontweight="bold", linespacing=1.05)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.50, y[i + 1] + 0.080), xytext=(0.50, yy - 0.080), arrowprops=dict(arrowstyle="->", lw=0.8, color=COLORS["grey"]))
    ax.text(0.05, 0.08, "Single-cell annotations are aggregated into matched Visium spot neighborhoods.", fontsize=6.6, color=COLORS["grey"], wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    if not spot_coords.empty and "x" in spot_coords and "y" in spot_coords:
        sample = spot_coords.sample(min(1600, len(spot_coords)), random_state=4)
        ax.scatter(sample["x"], sample["y"], s=2, c=COLORS["line"], alpha=0.7, label="all spots")
        if not qc.empty and "spot_id" in qc and "has_xenium_ground_truth" in qc:
            merged = spot_coords.merge(qc[["spot_id", "has_xenium_ground_truth"]], on="spot_id", how="left")
            gt = merged[merged["has_xenium_ground_truth"].astype(bool)]
            ax.scatter(gt["x"], gt["y"], s=5, c=COLORS["rose"], alpha=0.8, label="supervised")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Supervised spot coverage", loc="left", fontweight="bold")
    clean_ax(ax)

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "C")
    if not splits.empty:
        split_counts = splits["split"].value_counts().reindex(["train", "val", "test"]).dropna()
        ax.bar(split_counts.index, split_counts.values, color=[COLORS["teal"], COLORS["gold"], COLORS["rose"]])
    ax.set_title("Deterministic spatial split", loc="left", fontweight="bold")
    ax.set_ylabel("spots")
    clean_ax(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "D")
    if not qc.empty:
        gt_counts = numeric(qc, "xenium_cell_count")
        ax.hist(gt_counts[gt_counts > 0], bins=35, color=COLORS["blue"], alpha=0.82)
    ax.set_title("Xenium cells per supervised spot", loc="left", fontweight="bold")
    ax.set_xlabel("aggregated cells")
    ax.set_ylabel("spots")
    clean_ax(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "E")
    if not qc.empty:
        ent = numeric(qc, "ground_truth_entropy")
        ax.hist(ent[ent > 0], bins=35, color=COLORS["purple"], alpha=0.78)
    ax.set_title("Ground-truth entropy", loc="left", fontweight="bold")
    ax.set_xlabel("entropy")
    clean_ax(ax)

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "F")
    if not cell_summary.empty:
        top = cell_summary.sort_values("total_xenium_cells", ascending=False).head(10)
        horizontal_bar(ax, top["cell_type"], top["total_xenium_cells"], color=COLORS["green"], title="Top typed Xenium cells", xlabel="cells")
    save_fig(fig, "supplementary_figure_s1_benchmark_qc", "Supplementary Figure S1. Benchmark construction and QC", "benchmark datasheet; spot QC; spot coordinates")


def figure_s2() -> None:
    sens = read_csv("results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_cell_count_sensitivity.csv")
    cov = read_csv("results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_coverage_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.0))
    title_box(fig, "Supplementary Fig. 2 | Aggregation sensitivity", "Radius and cell-count thresholds define a transparent coverage-performance trade-off.")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.84, hspace=0.52, wspace=0.30)
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not cov.empty:
        xcol = "radius" if "radius" in cov else cov.columns[0]
        ycol = (
            "num_spots_with_cells"
            if "num_spots_with_cells" in cov.columns
            else "num_spots_passing_threshold"
            if "num_spots_passing_threshold" in cov.columns
            else cov.select_dtypes(include=[np.number]).columns[-1]
        )
        axs[0].plot(numeric(cov, xcol), numeric(cov, ycol), marker="o", linewidth=1.5, color=COLORS["teal"])
        axs[0].set_title("GT coverage by radius", loc="left", fontweight="bold")
        axs[0].set_xlabel("radius")
        axs[0].set_ylabel("covered spots")
    if not sens.empty:
        for metric, ax in [("jsd", axs[1]), ("spotwise_cosine", axs[2]), ("mean_celltype_pearson", axs[3])]:
            if metric in sens:
                pivot = sens.pivot_table(index="radius", columns="min_xenium_cells", values=metric, aggfunc="mean")
                for col in pivot.columns:
                    ax.plot(pivot.index, pivot[col], marker="o", linewidth=1, label=f"min {col}")
                ax.set_title(metric.replace("_", " "), loc="left", fontweight="bold")
                ax.set_xlabel("radius")
                ax.legend(frameon=False, ncol=2, fontsize=5.5)
    for ax in axs:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s2_aggregation_sensitivity", "Supplementary Figure S2. Aggregation sensitivity", "benchmark_sensitivity")


def figure_s3() -> None:
    pred = read_csv("results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv")
    coords = read_csv("data/processed/cytassist_xenium_rep2_common297/spot_coords.csv")
    summary = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/cell_type_proportion_summary.csv")
    fig = plt.figure(figsize=(13.5, 10.2))
    title_box(fig, "Supplementary Fig. 3 | Full cell-type atlas", "All predicted cell-type maps are shown on the same tissue coordinate frame.")
    gs = gridspec.GridSpec(5, 4, figure=fig, left=0.035, right=0.985, bottom=0.040, top=0.910, hspace=0.34, wspace=0.16)
    merged = pd.DataFrame()
    cell_types: list[str] = []
    if not pred.empty and not coords.empty and "spot_id" in pred.columns and "spot_id" in coords.columns:
        merged = coords[["spot_id", "x", "y"]].merge(pred, on="spot_id", how="inner")
        if not summary.empty and "cell_type" in summary.columns:
            ordered = summary.sort_values(["group", "mean_fraction"], ascending=[True, False])["cell_type"].astype(str).tolist()
            cell_types = [c for c in ordered if c in merged.columns]
        if not cell_types:
            cell_types = [c for c in pred.columns if c != "spot_id"]
    xlim = (merged["x"].min(), merged["x"].max()) if not merged.empty else (0, 1)
    ylim = (merged["y"].min(), merged["y"].max()) if not merged.empty else (0, 1)
    for i in range(20):
        ax = fig.add_subplot(gs[i])
        if i < len(cell_types) and not merged.empty:
            ct = cell_types[i]
            values = pd.to_numeric(merged[ct], errors="coerce").fillna(0)
            vmax = float(np.nanpercentile(values, 99.2)) if values.max() > 0 else 1.0
            vmax = max(vmax, float(values.max()) * 0.35, 1e-6)
            ax.scatter(merged["x"], merged["y"], s=1.4, c="#d6dbe0", alpha=0.28, linewidths=0, rasterized=True)
            sc = ax.scatter(merged["x"], merged["y"], s=2.2, c=np.clip(values, 0, vmax), cmap="magma", vmin=0, vmax=vmax, linewidths=0, rasterized=True)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ct.replace("+", "+ ").replace("  ", " "), loc="left", fontsize=6.5, fontweight="bold", pad=2)
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.01)
            cb.ax.tick_params(labelsize=4.7, length=1.5, pad=1)
            cb.outline.set_linewidth(0.35)
        elif i == 19 and not summary.empty:
            ax.axis("off")
            top = summary.sort_values("mean_fraction", ascending=False).head(7)
            rows = [(r["cell_type"], f"{float(r['mean_fraction']):.3f}") for _, r in top.iterrows()]
            draw_mini_table(ax, rows, "Mean fraction summary", left_width=0.68, row_height=0.092, fontsize=5.8)
        else:
            ax.axis("off")
    save_fig(fig, "supplementary_figure_s3_full_celltype_atlas", "Supplementary Figure S3. Full cell-type atlas", "predicted_proportions.csv + spot_coords.csv")


def figure_s4() -> None:
    base = read_csv("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv")
    boot = read_csv("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_split_bootstrap_summary.csv")
    paired = read_csv("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_bootstrap_paired_improvement.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7))
    title_box(fig, "Supplementary Fig. 4 | Baseline fairness package", "Formal comparisons use matched supervised spots and shared metrics.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not base.empty:
        b = base.sort_values("jsd", ascending=False)
        axs[0].barh(b["method"], numeric(b, "jsd"), color=[COLORS["rose"] if m == "WaveST-Gate" else COLORS["grey"] for m in b["method"]])
        axs[0].set_title("Matched benchmark JSD", loc="left", fontweight="bold")
        axs[0].set_xlabel("JSD (lower is better)")
        rt = base.dropna(subset=["runtime_seconds"]) if "runtime_seconds" in base else pd.DataFrame()
        if not rt.empty:
            sizes = np.clip(pd.to_numeric(rt.get("mean_celltype_pearson", pd.Series([0.2] * len(rt))), errors="coerce").fillna(0.1).add(0.05) * 130, 20, 120)
            colors = rt["source"].map({"formal_baseline": COLORS["blue"], "simple_baseline": COLORS["green"], "main_model": COLORS["rose"]}).fillna(COLORS["grey"])
            axs[1].scatter(numeric(rt, "runtime_seconds"), numeric(rt, "jsd"), s=sizes, c=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
            for _, r in rt.sort_values("jsd").head(6).iterrows():
                axs[1].text(float(r["runtime_seconds"]), float(r["jsd"]), str(r["method"])[:14], fontsize=5.0, ha="left", va="center")
            axs[1].set_xscale("log")
            axs[1].set_xlabel("runtime seconds (log)")
            axs[1].set_ylabel("JSD")
            axs[1].set_title("Runtime-accuracy Pareto view", loc="left", fontweight="bold")
    if not paired.empty:
        p = paired.sort_values("mean_jsd_improvement_vs_wavestgate", ascending=True).copy()
        y = np.arange(len(p))
        x = pd.to_numeric(p["mean_jsd_improvement_vs_wavestgate"], errors="coerce")
        low = pd.to_numeric(p.get("ci95_low", x), errors="coerce")
        high = pd.to_numeric(p.get("ci95_high", x), errors="coerce")
        axs[2].hlines(y, low, high, color=COLORS["grey"], linewidth=0.9)
        axs[2].scatter(x, y, color=COLORS["green"], s=22, zorder=3)
        axs[2].axvline(0, color=COLORS["ink"], linewidth=0.7)
        axs[2].set_yticks(y)
        axs[2].set_yticklabels(p["method"].astype(str).str.slice(0, 26), fontsize=5.4)
        axs[2].set_xlabel("JSD improvement vs WaveST-Gate")
        axs[2].set_title("Bootstrap paired improvement", loc="left", fontweight="bold")
    if not boot.empty:
        b = boot.sort_values("jsd_mean", ascending=True).copy()
        y = np.arange(len(b))
        x = pd.to_numeric(b["jsd_mean"], errors="coerce")
        sd = pd.to_numeric(b.get("jsd_std", pd.Series([0] * len(b))), errors="coerce").fillna(0)
        axs[3].hlines(y, x - 1.96 * sd, x + 1.96 * sd, color=COLORS["grey"], linewidth=0.9)
        axs[3].scatter(x, y, color=[COLORS["rose"] if m == "WaveST-Gate" else COLORS["purple"] for m in b["method"]], s=22, zorder=3)
        axs[3].set_yticks(y)
        axs[3].set_yticklabels(b["method"].astype(str).str.slice(0, 26), fontsize=5.4)
        axs[3].set_xlabel("bootstrap mean JSD +/- 1.96 SD")
        axs[3].set_title("Bootstrap stability", loc="left", fontweight="bold")
    for ax in axs:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s4_baseline_fairness", "Supplementary Figure S4. Baseline fairness package", "baseline_comparison")


def figure_s5() -> None:
    fair = read_csv("results/nature_leakage_fairness_audit/baseline_fairness_table.csv")
    perm = read_csv("results/nature_leakage_fairness_audit/label_permutation_control.csv")
    check = read_csv("results/nature_leakage_fairness_audit/leakage_fairness_checklist.csv")
    audit = read_json("results/nature_leakage_fairness_audit/leakage_fairness_audit.json")
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.8))
    title_box(fig, "Supplementary Fig. 5 | Leakage and negative controls", "Fairness and leakage checks test whether the benchmark advantage is artifactual.")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.860, hspace=0.46, wspace=0.36)
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not fair.empty and "recomputed_jsd" in fair:
        f = fair.sort_values("recomputed_jsd", ascending=True).copy()
        y = np.arange(len(f))
        x = numeric(f, "recomputed_jsd")
        axs[0].hlines(y, 0, x, color=COLORS["line"], linewidth=1.0)
        point_colors = [COLORS["rose"] if m == "WaveST-Gate" else COLORS["blue"] for m in f["method"]]
        axs[0].scatter(x, y, c=point_colors, s=28, edgecolor="white", linewidth=0.45, zorder=3)
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(f["method"].astype(str).str.slice(0, 24), fontsize=5.7)
        axs[0].set_xlabel("audited JSD (lower is better)")
        axs[0].set_title("Recomputed prediction error", loc="left", fontweight="bold")
    if not perm.empty:
        cols = perm.select_dtypes(include=[np.number]).columns
        if len(cols):
            value_col = "jsd" if "jsd" in perm.columns else cols[-1]
            axs[1].hist(pd.to_numeric(perm[value_col], errors="coerce").dropna(), bins=24, color=COLORS["rose"], alpha=0.85)
            observed = audit.get("permutation_control", {}).get("observed_jsd")
            if observed is not None:
                axs[1].axvline(float(observed), color=COLORS["ink"], linewidth=1.2, linestyle="--", label="observed")
                axs[1].legend(frameon=False)
            axs[1].set_xlabel("JSD under shuffled labels")
            axs[1].set_title("Label permutation control", loc="left", fontweight="bold")
    if not check.empty:
        status_col = "status" if "status" in check else check.columns[-1]
        c = check.copy()
        status_score = c[status_col].astype(str).str.lower().map({"pass": 1.0, "warn": 0.0, "warning": 0.0, "fail": -1.0}).fillna(0.0)
        mat = status_score.to_numpy()[:, None]
        axs[2].imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        labels = c["check"].astype(str).str.replace("_", " ", regex=False).str.slice(0, 32) if "check" in c else pd.Series([f"check {i+1}" for i in range(len(c))])
        axs[2].set_yticks(np.arange(len(c)))
        axs[2].set_yticklabels(labels, fontsize=5.9)
        axs[2].set_xticks([0])
        axs[2].set_xticklabels(["verdict"], fontsize=6.2)
        for yi, row in c.iterrows():
            metric = str(row.get("metric", ""))
            metric = metric[:32] + ("..." if len(metric) > 32 else "")
            axs[2].text(0, yi, str(row.get(status_col, "")).upper(), ha="center", va="center", fontsize=5.6, fontweight="bold", color=COLORS["ink"])
            axs[2].text(0.58, yi, metric, ha="left", va="center", fontsize=4.8, color=COLORS["grey"], transform=axs[2].get_yaxis_transform())
        axs[2].set_xlim(-0.5, 1.55)
        axs[2].set_title("Leakage and fairness audit matrix", loc="left", fontweight="bold")
        axs[2].tick_params(length=0)
        for spine in axs[2].spines.values():
            spine.set_visible(False)
    if not fair.empty and {"num_missing_required_spots", "num_missing_cell_types", "abs_jsd_delta"}.issubset(fair.columns):
        f = fair.sort_values("recomputed_jsd", ascending=True).copy()
        raw = pd.DataFrame({
            "missing spots": numeric(f, "num_missing_required_spots"),
            "missing types": numeric(f, "num_missing_cell_types"),
            "log10 JSD delta": np.log10(numeric(f, "abs_jsd_delta").clip(lower=1e-12)),
        })
        norm = raw.copy()
        for col in norm.columns:
            span = float(norm[col].max() - norm[col].min())
            norm[col] = 0.0 if span == 0 else (norm[col] - norm[col].min()) / span
        im = axs[3].imshow(norm.to_numpy(), aspect="auto", cmap="cividis_r", vmin=0, vmax=1)
        axs[3].set_yticks(np.arange(len(f)))
        axs[3].set_yticklabels(f["method"].astype(str).str.slice(0, 22), fontsize=5.5)
        axs[3].set_xticks(np.arange(len(raw.columns)))
        axs[3].set_xticklabels(raw.columns, rotation=25, ha="right", fontsize=5.8)
        for yi in range(len(f)):
            for xi, col in enumerate(raw.columns):
                val = raw.iloc[yi, xi]
                txt = f"{val:.0f}" if col != "log10 JSD delta" else f"{val:.1f}"
                axs[3].text(xi, yi, txt, ha="center", va="center", fontsize=4.7, color=COLORS["ink"])
        cb = fig.colorbar(im, ax=axs[3], fraction=0.045, pad=0.012)
        cb.ax.tick_params(labelsize=5, length=2)
        cb.set_label("normalized audit burden", fontsize=5.5)
        axs[3].set_title("Prediction-file integrity", loc="left", fontweight="bold")
        axs[3].tick_params(length=0)
        for spine in axs[3].spines.values():
            spine.set_visible(False)
    for ax in axs[:2]:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s5_leakage_negative_controls", "Supplementary Figure S5. Leakage and negative controls", "nature_leakage_fairness_audit")


def figure_s6() -> None:
    ab = read_csv("results/nature_main/cytassist_rep2_radius55/ablations250/ablation_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
    title_box(fig, "Supplementary Fig. 6 | Ablation mechanism wall", "Module and modality removals quantify necessity across accuracy, uncertainty and boundary-aware objectives.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not ab.empty:
        full = float(ab.loc[ab["ablation"].eq("full"), "jsd"].iloc[0]) if (ab["ablation"].eq("full")).any() else float(numeric(ab, "jsd").min())
        ab = ab.assign(delta_jsd=numeric(ab, "jsd") - full).sort_values("delta_jsd")
        axs[0].barh(ab["ablation"], ab["delta_jsd"], color=[COLORS["green"] if v <= 0 else COLORS["rose"] for v in ab["delta_jsd"]])
        axs[0].set_title("JSD delta versus full", loc="left", fontweight="bold")
        axs[1].scatter(numeric(ab, "spotwise_cosine"), numeric(ab, "jsd"), c=COLORS["teal"], s=50)
        for _, r in ab.iterrows():
            axs[1].text(float(r["spotwise_cosine"]), float(r["jsd"]), str(r["ablation"]).replace("_", " ")[:16], fontsize=5.3)
        axs[1].set_title("Accuracy trade-off", loc="left", fontweight="bold")
        if "uncertainty_error_pearson" in ab:
            axs[2].barh(ab["ablation"], numeric(ab, "uncertainty_error_pearson"), color=COLORS["purple"])
            axs[2].set_title("Uncertainty-error correlation", loc="left", fontweight="bold")
        loss_cols = [c for c in ab.columns if c.startswith("loss_")][:6]
        if loss_cols:
            heat = ab.set_index("ablation")[loss_cols].apply(pd.to_numeric, errors="coerce")
            axs[3].imshow(heat.fillna(0), aspect="auto", cmap="YlGnBu")
            axs[3].set_yticks(range(len(heat.index)))
            axs[3].set_yticklabels([x[:20] for x in heat.index], fontsize=5.3)
            axs[3].set_xticks(range(len(loss_cols)))
            axs[3].set_xticklabels([c.replace("loss_", "") for c in loss_cols], rotation=45, ha="right")
            axs[3].set_title("Loss components", loc="left", fontweight="bold")
    for ax in axs[:3]:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s6_ablation_mechanism", "Supplementary Figure S6. Ablation mechanism wall", "ablations250/ablation_summary.csv")


def figure_s7() -> None:
    per = read_csv("results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution/image_contribution_per_spot.csv")
    coords = read_csv("data/processed/cytassist_xenium_rep2_common297/spot_coords.csv")
    run = read_csv("results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution/imagegate_run_comparison.csv")
    fig = plt.figure(figsize=(11.0, 7.2))
    title_box(fig, "Supplementary Fig. 7 | H&E morphology contribution", "Per-spot paired controls localize morphology-dependent gains beyond run-level averages.")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.060, right=0.985, bottom=0.075, top=0.875, hspace=0.42, wspace=0.34)
    axs = [fig.add_subplot(gs[i]) for i in range(6)]
    for ax, label in zip(axs, "ABCDEF"):
        panel_label(ax, label)

    merged = pd.DataFrame()
    if not per.empty and not coords.empty and {"spot_id", "paired_jsd_improvement"}.issubset(per.columns):
        merged = coords[["spot_id", "x", "y"]].merge(per, on="spot_id", how="inner")

    if not merged.empty:
        delta = pd.to_numeric(merged["paired_jsd_improvement"], errors="coerce")
        limit = float(np.nanpercentile(np.abs(delta.dropna()), 98)) if delta.notna().any() else 0.01
        limit = max(limit, 0.004)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        axs[0].scatter(coords["x"], coords["y"], s=1.0, c="#d8dde3", alpha=0.22, linewidths=0, rasterized=True)
        sc = axs[0].scatter(merged["x"], merged["y"], s=8, c=np.clip(delta, -limit, limit), cmap="PRGn", norm=norm, linewidths=0, rasterized=True)
        axs[0].invert_yaxis()
        axs[0].set_aspect("equal")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("spatial paired gain", loc="left", fontweight="bold")
        cb = fig.colorbar(sc, ax=axs[0], fraction=0.046, pad=0.015)
        cb.ax.tick_params(labelsize=5.0, length=2)
        cb.set_label("no-image - image JSD", fontsize=5.6)

    if not per.empty and {"noimage_jsd", "imagegate_jsd", "patch_texture"}.issubset(per.columns):
        x = pd.to_numeric(per["noimage_jsd"], errors="coerce")
        y = pd.to_numeric(per["imagegate_jsd"], errors="coerce")
        c = pd.to_numeric(per["patch_texture"], errors="coerce")
        valid = x.notna() & y.notna()
        lim_max = float(np.nanpercentile(pd.concat([x[valid], y[valid]]), 98.5)) if valid.any() else 0.03
        lim_max = max(lim_max, 0.02)
        axs[1].plot([0, lim_max], [0, lim_max], color=COLORS["grey"], lw=0.8, ls="--")
        sc = axs[1].scatter(x[valid], y[valid], c=c[valid], cmap="viridis", s=13, alpha=0.72, linewidths=0, rasterized=True)
        axs[1].set_xlim(0, lim_max)
        axs[1].set_ylim(0, lim_max)
        axs[1].set_aspect("equal", adjustable="box")
        axs[1].set_xlabel("no-image JSD")
        axs[1].set_ylabel("image-gate JSD")
        axs[1].set_title("paired spot-level error", loc="left", fontweight="bold")
        cb = fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.015)
        cb.ax.tick_params(labelsize=5.0, length=2)
        cb.set_label("texture", fontsize=5.6)

    if not per.empty and "paired_jsd_improvement" in per.columns:
        delta = pd.to_numeric(per["paired_jsd_improvement"], errors="coerce").dropna()
        bins = np.linspace(float(delta.quantile(0.01)), float(delta.quantile(0.99)), 34)
        axs[2].hist(delta, bins=bins, color="#9aa6b2", alpha=0.72, density=True)
        axs[2].axvline(0, color=COLORS["ink"], lw=0.8)
        axs[2].axvline(delta.mean(), color=COLORS["rose"], lw=1.0)
        axs[2].text(0.03, 0.94, f"n={len(delta)}\nmean={delta.mean():+.4f}", transform=axs[2].transAxes, va="top", fontsize=5.9)
        axs[2].set_xlabel("no-image - image JSD")
        axs[2].set_ylabel("density")
        axs[2].set_title("paired gain distribution", loc="left", fontweight="bold")

    if not per.empty and {"patch_texture", "paired_jsd_improvement"}.issubset(per.columns):
        tmp = per[["patch_texture", "paired_jsd_improvement"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not tmp.empty:
            tmp["texture_bin"] = pd.qcut(tmp["patch_texture"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            cats = [c for c in ["Q1", "Q2", "Q3", "Q4"] if c in set(tmp["texture_bin"].astype(str))]
            data = [tmp.loc[tmp["texture_bin"].astype(str).eq(c), "paired_jsd_improvement"].values for c in cats]
            bp = axs[3].boxplot(data, tick_labels=cats, widths=0.52, patch_artist=True, showfliers=False, medianprops=dict(color=COLORS["ink"], lw=0.9))
            for patch in bp["boxes"]:
                patch.set(facecolor="#eef2f4", edgecolor=COLORS["grey"], linewidth=0.8)
            rng = np.random.default_rng(7)
            for j, vals in enumerate(data, start=1):
                if len(vals):
                    idx = rng.choice(np.arange(len(vals)), size=min(90, len(vals)), replace=False)
                    axs[3].scatter(np.full(len(idx), j) + rng.normal(0, 0.045, len(idx)), vals[idx], s=6, color=COLORS["teal"], alpha=0.45, linewidths=0, rasterized=True)
            axs[3].axhline(0, color=COLORS["ink"], lw=0.7)
            axs[3].set_xlabel("patch texture quartile")
            axs[3].set_ylabel("no-image - image JSD")
            axs[3].set_title("texture-stratified paired gain", loc="left", fontweight="bold")

    if not per.empty and {"patch_texture", "image_gate", "paired_jsd_improvement"}.issubset(per.columns):
        tmp = per[["patch_texture", "image_gate", "paired_jsd_improvement"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not tmp.empty:
            vmin = float(tmp["paired_jsd_improvement"].quantile(0.02))
            vmax = float(tmp["paired_jsd_improvement"].quantile(0.98))
            if vmin >= 0 or vmax <= 0:
                vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
            sc = axs[4].scatter(tmp["patch_texture"], tmp["image_gate"], c=tmp["paired_jsd_improvement"], cmap="PRGn", norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax), s=12, alpha=0.66, linewidths=0, rasterized=True)
            bins = pd.qcut(tmp["patch_texture"], 8, duplicates="drop")
            med = tmp.groupby(bins, observed=True).agg(texture=("patch_texture", "median"), gate=("image_gate", "median"))
            axs[4].plot(med["texture"], med["gate"], color=COLORS["ink"], lw=1.0)
            axs[4].set_xlabel("patch texture")
            axs[4].set_ylabel("image gate")
            axs[4].set_title("gate response to morphology", loc="left", fontweight="bold")
            cb = fig.colorbar(sc, ax=axs[4], fraction=0.046, pad=0.015)
            cb.ax.tick_params(labelsize=5.0, length=2)

    if not run.empty and "setting" in run.columns:
        metrics = ["jsd", "mean_celltype_pearson", "spotwise_cosine", "mean_image_gate"]
        present = [m for m in metrics if m in run.columns]
        if present:
            setting_map = {
                "baseline_main": "baseline",
                "imagegate_enhanced": "image-gate",
                "matched_no_image_control": "no-image",
            }
            metric_map = {
                "jsd": "JSD",
                "mean_celltype_pearson": "Pearson",
                "spotwise_cosine": "cosine",
                "mean_image_gate": "image gate",
            }
            settings = run["setting"].astype(str).map(setting_map).fillna(run["setting"].astype(str))
            mat = run[present].apply(pd.to_numeric, errors="coerce")
            scaled = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0, np.nan)
            im = axs[5].imshow(scaled.fillna(0).T.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
            axs[5].set_xticks(range(len(settings)))
            axs[5].set_xticklabels(settings, fontsize=5.6)
            axs[5].set_yticks(range(len(present)))
            axs[5].set_yticklabels([metric_map.get(m, m) for m in present], fontsize=5.8)
            axs[5].set_title("run-level control profile", loc="left", fontweight="bold")
            for yi, m in enumerate(present):
                for xi, value in enumerate(mat[m]):
                    axs[5].text(xi, yi, f"{value:.3f}", ha="center", va="center", fontsize=4.8, color=COLORS["ink"])
            cb = fig.colorbar(im, ax=axs[5], fraction=0.046, pad=0.015)
            cb.ax.tick_params(labelsize=5.0, length=2)

    for ax in axs:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s7_he_morphology_contribution", "Supplementary Figure S7. H&E morphology contribution", "image_contribution per-spot paired controls")


def figure_s8() -> None:
    err = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/reliability_spot_errors.csv")
    bins = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/uncertainty_calibration_bins.csv")
    risk = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/risk_coverage_curve.csv")
    fail = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/failure_case_candidates.csv")
    fig, axs = plt.subplots(2, 2, figsize=(9.8, 6.5))
    title_box(fig, "Supplementary Fig. 8 | Reliability calibration details", "Uncertainty links to spot-level error, calibration bins, risk coverage and failure candidates.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not err.empty:
        sample = err.sample(min(900, len(err)), random_state=7)
        axs[0].scatter(numeric(sample, "spot_uncertainty"), numeric(sample, "jsd"), s=8, alpha=0.5, c=COLORS["teal"])
        axs[0].set_xlabel("spot uncertainty")
        axs[0].set_ylabel("JSD")
        axs[0].set_title("Uncertainty-error scatter", loc="left", fontweight="bold")
    if not bins.empty:
        if {"mean_uncertainty", "mean_jsd"}.issubset(bins.columns):
            size = pd.to_numeric(bins.get("num_spots", pd.Series([45] * len(bins))), errors="coerce").fillna(45)
            axs[1].plot(bins["mean_uncertainty"], bins["mean_jsd"], color=COLORS["purple"], linewidth=1.0)
            axs[1].scatter(bins["mean_uncertainty"], bins["mean_jsd"], s=np.clip(size, 20, 70), color=COLORS["purple"], alpha=0.85)
            axs[1].set_xlabel("mean uncertainty")
            axs[1].set_ylabel("mean JSD")
        else:
            num_cols = bins.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                axs[1].plot(bins[num_cols[0]], bins[num_cols[1]], marker="o", color=COLORS["purple"])
        axs[1].set_title("Calibration bins", loc="left", fontweight="bold")
    if not risk.empty:
        if {"coverage", "mean_jsd"}.issubset(risk.columns):
            axs[2].plot(risk["coverage"], risk["mean_jsd"], color=COLORS["rose"], marker="o")
            axs[2].set_xlabel("coverage retained")
            axs[2].set_ylabel("mean JSD")
        else:
            cols = risk.select_dtypes(include=[np.number]).columns
            if len(cols) >= 2:
                axs[2].plot(risk[cols[0]], risk[cols[1]], color=COLORS["rose"], marker="o")
        axs[2].set_title("Risk coverage", loc="left", fontweight="bold")
    if not fail.empty:
        cols = [c for c in ["jsd", "spot_uncertainty"] if c in fail]
        if len(cols) >= 2:
            axs[3].scatter(numeric(fail, cols[1]), numeric(fail, cols[0]), c=COLORS["orange"], s=28)
            axs[3].set_xlabel(cols[1])
            axs[3].set_ylabel(cols[0])
        else:
            axs[3].text(0.1, 0.5, f"{len(fail)} failure candidates", transform=axs[3].transAxes)
        axs[3].set_title("Failure candidates", loc="left", fontweight="bold")
    for ax in axs:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s8_reliability_calibration", "Supplementary Figure S8. Reliability calibration details", "nature_analysis reliability files")


def figure_s9() -> None:
    fig = plt.figure(figsize=(10.5, 6.8))
    title_box(fig, "Supplementary Fig. 9 | Boundary preservation details", "Boundary maps, typed transitions and marker validation test over-smoothing risks.")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.05, right=0.98, bottom=0.06, top=0.90, hspace=0.28, wspace=0.25)
    imgs = [
        ("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_sharpness_map.png", "boundary sharpness map"),
        ("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_he_overlay.png", "H&E boundary overlay"),
        ("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_map.png", "niche map context"),
    ]
    for i, (path, title) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i])
        panel_label(ax, chr(65 + i))
        show_image(ax, path, title)
    marker = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_marker_validation.csv")
    btype = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_type_summary.csv")
    jumps = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_edge_jumps.csv")
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "D")
    if not marker.empty:
        top = marker.sort_values("mean_expression", ascending=False).head(10)
        horizontal_bar(ax, top["marker_set"] + " | " + top["boundary_type"], numeric(top, "mean_expression"), COLORS["teal"], "Marker validation", "mean expression")
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "E")
    if not btype.empty:
        y = btype.iloc[:, 0].astype(str)
        num = btype.select_dtypes(include=[np.number]).iloc[:, 0]
        ax.barh(y, num, color=COLORS["purple"])
    ax.set_title("Typed boundary summary", loc="left", fontweight="bold")
    clean_ax(ax)
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "F")
    if not jumps.empty:
        nums = jumps.select_dtypes(include=[np.number]).columns
        if len(nums):
            ax.hist(jumps[nums[0]].dropna(), bins=40, color=COLORS["rose"], alpha=0.8)
    ax.set_title("Boundary edge jump distribution", loc="left", fontweight="bold")
    clean_ax(ax)
    save_fig(fig, "supplementary_figure_s9_boundary_preservation", "Supplementary Figure S9. Boundary preservation details", "nature_analysis boundary files")


def figure_s10() -> None:
    comp = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_composition.csv")
    bio = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_biological_summary.csv")
    marker = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_marker_enrichment.csv")
    xen = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_xenium_neighborhood_summary.csv")
    gate = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/gate_reliability_by_niche.csv")
    agent = read_csv("results/nature_main/cytassist_rep2_radius55/nature_analysis/agent_attention_by_niche.csv")
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 6.8))
    title_box(fig, "Supplementary Fig. 10 | Niche biological validation", "Niche identity is supported by composition, marker enrichment, Xenium neighborhoods and reliability profiles.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCDEF"):
        panel_label(ax, label)
    show_image(axs[0], "results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_map.png", "niche map")
    if not bio.empty:
        label_col = "niche_label" if "niche_label" in bio.columns else "niche"
        size_df = bio.copy()
        size_df["label"] = size_df[label_col].astype(str).str.replace(" niche", "", regex=False)
        axs[1].scatter(
            pd.to_numeric(size_df["niche"], errors="coerce"),
            pd.to_numeric(size_df["num_spots"], errors="coerce"),
            s=np.clip(pd.to_numeric(size_df["num_spots"], errors="coerce") / 4, 80, 360),
            color=COLORS["blue"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, r in size_df.iterrows():
            axs[1].text(float(r["niche"]), float(r["num_spots"]), str(r["label"])[:18], fontsize=5.6, ha="center", va="center", color="white")
        axs[1].set_xlabel("niche")
        axs[1].set_ylabel("spots")
        axs[1].set_title("Niche sizes and labels", loc="left", fontweight="bold")
    if not comp.empty:
        mean_cols = [c for c in comp.columns if c.startswith("mean_")]
        if mean_cols:
            heat = comp.set_index(comp["niche"].astype(str))[mean_cols].rename(columns=lambda c: c.replace("mean_", "")).T
            top = heat.loc[heat.max(axis=1).sort_values(ascending=False).head(10).index]
            im = axs[2].imshow(top.values, aspect="auto", cmap="viridis")
            axs[2].set_yticks(range(len(top.index)))
            axs[2].set_yticklabels(top.index, fontsize=5.5)
            axs[2].set_xticks(range(len(top.columns)))
            axs[2].set_xticklabels([f"N{x}" for x in top.columns])
            axs[2].set_title("Cell-type composition heatmap", loc="left", fontweight="bold")
            cb = fig.colorbar(im, ax=axs[2], fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=5.2, length=2)
    if not marker.empty:
        if {"marker_set", "niche", "mean_expression"}.issubset(marker.columns):
            m = marker.copy()
            m["niche"] = m["niche"].astype(str)
            m["mean_expression"] = pd.to_numeric(m["mean_expression"], errors="coerce")
            xcats = sorted(m["niche"].unique(), key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else str(x))
            ycats = list(dict.fromkeys(m["marker_set"].astype(str)))
            xmap = {x: i for i, x in enumerate(xcats)}
            ymap = {y: i for i, y in enumerate(ycats)}
            sizes = np.clip(m["mean_expression"].fillna(0) * 2.8, 18, 260)
            sc = axs[3].scatter(m["niche"].map(xmap), m["marker_set"].map(ymap), s=sizes, c=m["mean_expression"], cmap="magma", alpha=0.86, edgecolor="white", linewidth=0.35)
            axs[3].set_xticks(range(len(xcats)))
            axs[3].set_xticklabels([f"N{x}" for x in xcats])
            axs[3].set_yticks(range(len(ycats)))
            axs[3].set_yticklabels(ycats, fontsize=5.7)
            axs[3].set_title("Marker enrichment dot plot", loc="left", fontweight="bold")
            cb = fig.colorbar(sc, ax=axs[3], fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=5.2, length=2)
    if not xen.empty:
        if {"niche", "spot_group_agreement_rate", "neighborhood_group_agreement_rate", "num_spots_with_xenium_neighbors"}.issubset(xen.columns):
            x = pd.to_numeric(xen["spot_group_agreement_rate"], errors="coerce")
            y = pd.to_numeric(xen["neighborhood_group_agreement_rate"], errors="coerce")
            s = np.clip(pd.to_numeric(xen["num_spots_with_xenium_neighbors"], errors="coerce").fillna(1) * 1.2, 40, 300)
            axs[4].scatter(x, y, s=s, color=COLORS["green"], alpha=0.82, edgecolor="white", linewidth=0.8)
            for _, r in xen.iterrows():
                axs[4].text(float(r["spot_group_agreement_rate"]), float(r["neighborhood_group_agreement_rate"]), f"N{int(r['niche'])}", fontsize=6, ha="center", va="center", color="white")
            axs[4].set_xlim(-0.04, 1.04)
            axs[4].set_ylim(-0.04, 1.04)
            axs[4].set_xlabel("spot group agreement")
            axs[4].set_ylabel("neighborhood agreement")
            axs[4].set_title("Xenium neighborhood support", loc="left", fontweight="bold")
    if not gate.empty:
        gate_cols = [c for c in ["mean_image_gate", "mean_expression_gate", "mean_reference_gate"] if c in gate.columns]
        if gate_cols:
            matrix = gate.set_index(gate["niche"].astype(str))[gate_cols].T
            if not agent.empty and {"niche", "top_agent_attention"}.issubset(agent.columns):
                matrix.loc["top_agent_attention"] = agent.set_index(agent["niche"].astype(str))["top_agent_attention"].reindex(matrix.columns)
            im = axs[5].imshow(matrix.astype(float).values, aspect="auto", cmap="cividis")
            axs[5].set_xticks(range(len(matrix.columns)))
            axs[5].set_xticklabels([f"N{x}" for x in matrix.columns])
            axs[5].set_yticks(range(len(matrix.index)))
            axs[5].set_yticklabels([x.replace("mean_", "").replace("_", " ") for x in matrix.index], fontsize=5.8)
            axs[5].set_title("Gate and agent profile", loc="left", fontweight="bold")
            cb = fig.colorbar(im, ax=axs[5], fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=5.2, length=2)
    for ax in (axs[1], axs[4]):
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s10_niche_biological_validation", "Supplementary Figure S10. Niche biological validation", "nature_analysis niche files")


def figure_s11() -> None:
    ext = read_csv("results/nature_external_no_retuning/external_no_retuning_summary.csv")
    matched = read_csv("results/nature_external_matched_gt/external_matched_gt_summary.csv")
    budget = read_csv("results/nature_manuscript_tables/table_11_rep1_retune_budget_curve.csv")
    multi = read_csv("results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    title_box(fig, "Supplementary Fig. 11 | External generalization", "No-retuning transfer, matched-GT Rep1 evaluation and minimal adaptation define cross-sample behavior.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not ext.empty:
        e = ext.sort_values("expression_log1p_rmse", ascending=True).copy()
        y = np.arange(len(e))
        x = pd.to_numeric(e["expression_log1p_rmse"], errors="coerce")
        sizes = np.clip(pd.to_numeric(e.get("num_spots", pd.Series([1000] * len(e))), errors="coerce").fillna(1000) / 18, 45, 260)
        colors = pd.to_numeric(e.get("mean_spot_uncertainty", pd.Series([0] * len(e))), errors="coerce").fillna(0)
        axs[0].hlines(y, 0, x, color=COLORS["line"], linewidth=1.0)
        sc = axs[0].scatter(x, y, s=sizes, c=colors, cmap="viridis", edgecolor="white", linewidth=0.7, alpha=0.9)
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(e["dataset"].astype(str).str.replace("_xenium_common301", "", regex=False), fontsize=5.3)
        axs[0].set_xlabel("expression RMSE")
        axs[0].set_title("External no-retuning transfer", loc="left", fontweight="bold")
        cb = fig.colorbar(sc, ax=axs[0], fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.2, length=2)
    if not matched.empty:
        r = matched.iloc[0]
        points = [
            ("no retune\nWaveST-Gate", r.get("no_retuning_model_jsd", np.nan), COLORS["grey"]),
            ("no retune\nbest baseline", r.get("no_retuning_best_baseline_jsd", np.nan), COLORS["purple"]),
            ("minimal retune\nWaveST-Gate", r.get("minimal_retune_test_jsd", np.nan), COLORS["rose"]),
            ("minimal retune\nbest baseline", r.get("minimal_retune_best_baseline_jsd", np.nan), COLORS["orange"]),
        ]
        xs = [0, 0, 1, 1]
        ys = [float(p[1]) for p in points]
        axs[1].plot([0, 1], [ys[0], ys[2]], color=COLORS["rose"], linewidth=1.5, marker="o", label="WaveST-Gate")
        axs[1].plot([0, 1], [ys[1], ys[3]], color=COLORS["purple"], linewidth=1.5, marker="o", label="best baseline")
        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(["no retune", "minimal retune"])
        axs[1].set_ylabel("JSD")
        axs[1].legend(frameon=False, fontsize=5.8)
        axs[1].set_title("Rep1 matched-GT adaptation", loc="left", fontweight="bold")
    if not budget.empty:
        axs[2].plot(numeric(budget, "budget_steps"), numeric(budget, "jsd"), marker="o", color=COLORS["rose"])
        if "best_baseline_jsd" in budget:
            axs[2].axhline(float(numeric(budget, "best_baseline_jsd").dropna().iloc[0]), ls="--", color=COLORS["grey"], lw=1)
        axs[2].set_xscale("symlog")
        axs[2].set_title("Rep1 minimal-retuning budget", loc="left", fontweight="bold")
        axs[2].set_xlabel("steps")
        axs[2].set_ylabel("JSD")
    if not multi.empty:
        if {"method", "jsd_mean"}.issubset(multi.columns):
            m = multi.sort_values("jsd_mean", ascending=False).copy()
            y = np.arange(len(m))
            x = pd.to_numeric(m["jsd_mean"], errors="coerce")
            xerr = pd.to_numeric(m.get("jsd_std", pd.Series([0] * len(m))), errors="coerce").fillna(0)
            colors = [COLORS["rose"] if name == "WaveST-Gate" else COLORS["green"] for name in m["method"]]
            axs[3].errorbar(x, y, xerr=xerr, fmt="none", ecolor=COLORS["grey"], elinewidth=0.8, capsize=2)
            axs[3].scatter(x, y, c=colors, s=35, edgecolor="white", linewidth=0.5, zorder=3)
            axs[3].set_yticks(y)
            axs[3].set_yticklabels(m["method"].astype(str).str.slice(0, 24), fontsize=5.4)
            axs[3].set_xlabel("mean JSD +/- SD")
        axs[3].set_title("Multi-sample minimal-retune forest", loc="left", fontweight="bold")
    for ax in axs:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s11_external_generalization", "Supplementary Figure S11. External generalization", "external summaries")


def figure_s12() -> None:
    rob = read_csv("results/nature_main/cytassist_rep2_radius55/robustness/robustness_summary.csv")
    patch = read_csv("results/nature_main/cytassist_rep2_radius55/patch_size_robustness/patch_size_summary.csv")
    split = read_csv("results/nature_main/cytassist_rep2_radius55/split_sensitivity/split_sensitivity_aggregate.csv")
    sens = read_csv("results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_cell_count_sensitivity.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    title_box(fig, "Supplementary Fig. 12 | Robustness stress tests", "Perturbation, split, patch-size and benchmark sensitivity analyses define stability.")
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not rob.empty:
        metric = "jsd_max" if "jsd_max" in rob.columns else "jsd" if "jsd" in rob.columns else None
        if metric is not None and "scenario" in rob.columns:
            r = rob.copy()
            r[metric] = pd.to_numeric(r[metric], errors="coerce")
            order = r.groupby("scenario")[metric].max().sort_values().index.tolist()
            ymap = {name: i for i, name in enumerate(order)}
            axs[0].hlines(range(len(order)), r[metric].min(), r.groupby("scenario")[metric].max().reindex(order), color=COLORS["line"], linewidth=0.9)
            axs[0].scatter(r[metric], r["scenario"].map(ymap), s=32, color=COLORS["rose"], alpha=0.74, edgecolor="white", linewidth=0.4)
            clean = r.loc[r["scenario"].eq("clean"), metric]
            if not clean.empty:
                axs[0].axvline(float(clean.iloc[0]), color=COLORS["ink"], linestyle="--", linewidth=1.0, label="clean")
                axs[0].legend(frameon=False, fontsize=5.8)
            axs[0].set_yticks(range(len(order)))
            axs[0].set_yticklabels([str(x).replace("_", " ") for x in order], fontsize=5.8)
            axs[0].set_title("Stress-test JSD by scenario", loc="left", fontweight="bold")
            axs[0].set_xlabel(metric)
    if not patch.empty:
        x = patch.iloc[:, 0]
        y = patch["jsd"] if "jsd" in patch else patch.select_dtypes(include=[np.number]).iloc[:, -1]
        axs[1].plot(x, y, marker="o", color=COLORS["blue"])
        axs[1].set_title("Patch-size robustness", loc="left", fontweight="bold")
    if not split.empty:
        split_metric = "mean_jsd" if "mean_jsd" in split.columns else "jsd" if "jsd" in split.columns else None
        if split_metric is not None:
            x = split["split"].astype(str) if "split" in split.columns else split.index.astype(str)
            y = pd.to_numeric(split[split_metric], errors="coerce")
            yerr = pd.to_numeric(split.get("sd_jsd", pd.Series([0] * len(split))), errors="coerce").fillna(0)
            axs[2].errorbar(x, y, yerr=yerr, fmt="o", color=COLORS["green"], ecolor=COLORS["grey"], capsize=3, markersize=5)
            axs[2].plot(x, y, color=COLORS["green"], linewidth=0.9, alpha=0.65)
            axs[2].set_ylabel(split_metric)
        axs[2].set_title("GT-stratified split sensitivity", loc="left", fontweight="bold")
    if not sens.empty and "jsd" in sens:
        pivot = sens.pivot_table(index="radius", columns="min_xenium_cells", values="jsd", aggfunc="mean")
        im = axs[3].imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
        axs[3].set_xticks(range(len(pivot.columns)))
        axs[3].set_xticklabels([str(int(c)) for c in pivot.columns])
        axs[3].set_yticks(range(len(pivot.index)))
        axs[3].set_yticklabels([str(int(r)) for r in pivot.index])
        axs[3].set_xlabel("min Xenium cells")
        axs[3].set_ylabel("radius")
        axs[3].set_title("Radius/cell-count JSD heatmap", loc="left", fontweight="bold")
        cb = fig.colorbar(im, ax=axs[3], fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.2, length=2)
    for ax in axs[:3]:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s12_robustness_stress_tests", "Supplementary Figure S12. Robustness stress tests", "robustness, patch, split, benchmark sensitivity")


def figure_s13() -> None:
    cls = read_csv("results/nature_external_pathology_validation/pathology_class_summary.csv")
    patient = read_csv("results/nature_external_pathology_validation/pathology_patient_summary.csv")
    niche = read_csv("results/nature_external_pathology_validation/pathology_niche_summary.csv")
    fig = plt.figure(figsize=(10.5, 6.4))
    title_box(fig, "Supplementary Fig. 13 | External pathology validation", "Niche-pathology correspondence is summarized by class, heatmap and patient-level support.")
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.055, right=0.98, bottom=0.07, top=0.90, hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    show_image(ax, "results/nature_external_pathology_validation/pathology_niche_by_class_heatmap.png", "pathology-niche heatmap")
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    if not cls.empty:
        top = cls.copy()
        top["classification"] = top["classification"].fillna("unclassified").astype(str)
        top["agreement_rate"] = pd.to_numeric(top.get("agreement_rate"), errors="coerce")
        top = top.dropna(subset=["agreement_rate"]).sort_values("num_spots", ascending=True)
        ax.barh(top["classification"], top["agreement_rate"], color=COLORS["green"])
        ax.set_xlim(0, 1.05)
        ax.set_title("Agreement by pathology class", loc="left", fontweight="bold")
        clean_ax(ax)
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    if not niche.empty:
        cols = [
            ("mean_tumor_fraction", "tumor", COLORS["rose"]),
            ("mean_immune_fraction", "immune", COLORS["purple"]),
            ("mean_stromal_fraction", "stromal", COLORS["green"]),
            ("mean_other_fraction", "other", COLORS["grey"]),
        ]
        y = np.arange(len(niche))
        left = np.zeros(len(niche))
        labels = []
        for col, label, color in cols:
            if col in niche.columns:
                values = pd.to_numeric(niche[col], errors="coerce").fillna(0).to_numpy()
                ax.barh(y, values, left=left, color=color, label=label, alpha=0.88)
                left += values
        for _, r in niche.iterrows():
            labels.append(f"N{int(r['external_niche'])} | {str(r.get('dominant_pathology_class', ''))[:26]}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("mean group fraction")
        ax.set_title("Niche-level pathology composition", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=4, fontsize=5.6, loc="lower right")
        clean_ax(ax)
    else:
        show_image(ax, "results/nature_external_pathology_validation/pathology_group_composition.png", "pathology group composition")
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    if not patient.empty:
        if {"patientid", "agreement_rate", "num_spots"}.issubset(patient.columns):
            tmp = patient.copy()
            tmp["weighted_agreement"] = pd.to_numeric(tmp["agreement_rate"], errors="coerce") * pd.to_numeric(tmp["num_spots"], errors="coerce")
            agg = tmp.groupby("patientid", as_index=False).agg(num_spots=("num_spots", "sum"), weighted_agreement=("weighted_agreement", "sum"))
            agg["agreement_rate"] = agg["weighted_agreement"] / agg["num_spots"].replace(0, np.nan)
            agg = agg.dropna(subset=["agreement_rate"]).sort_values("agreement_rate")
            ax.barh(agg["patientid"].astype(str), agg["agreement_rate"], color=COLORS["blue"])
            ax.set_xlim(0, 1.05)
        else:
            y = patient.iloc[:, 0].fillna("unclassified").astype(str)
            nums = patient.select_dtypes(include=[np.number])
            if not nums.empty:
                ax.barh(y, nums.iloc[:, 0], color=COLORS["blue"])
    ax.set_title("Patient-level support", loc="left", fontweight="bold")
    clean_ax(ax)
    save_fig(fig, "supplementary_figure_s13_external_pathology_validation", "Supplementary Figure S13. External pathology validation", "nature_external_pathology_validation")


def figure_s14() -> None:
    relv = read_json("results/nature_release/release_verification.json")
    env = read_json("results/nature_release/environment_report.json")
    doi = read_json("results/nature_release/zenodo_deposition_result.json")
    upload = read_csv("results/nature_release/release_upload_manifest.csv")
    checksums = read_csv("results/nature_submission_readiness/release_file_checksums.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10.6, 6.8))
    title_box(fig, "Supplementary Fig. 14 | Reproducibility and release map", "Source data, scripts, checksums, environment and Zenodo release form a reproducible handoff.")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.095, top=0.860, hspace=0.48, wspace=0.36)
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    def release_category(path: str) -> str:
        p = str(path).replace("\\", "/")
        if p.startswith("data/"):
            return "data"
        if p.startswith("results/"):
            return "results"
        if p.startswith(("wavestgate/", "tools/", "experiments/")) or p.endswith(".py"):
            return "code"
        if p.startswith("docs/") or p.endswith((".md", ".rst")):
            return "docs"
        return "metadata"

    if not upload.empty and {"path", "bytes"}.issubset(upload.columns):
        u = upload.copy()
        u["category"] = u["path"].map(release_category)
        u["bytes"] = pd.to_numeric(u["bytes"], errors="coerce").fillna(0)
        u = u.sort_values("bytes", ascending=True).reset_index(drop=True)
        palette = {"data": COLORS["green"], "results": COLORS["blue"], "code": COLORS["rose"], "docs": COLORS["purple"], "metadata": COLORS["grey"]}
        for cat, group in u.groupby("category", sort=False):
            axs[0].scatter(group.index, np.log10(group["bytes"] + 1), s=14, alpha=0.72, color=palette.get(cat, COLORS["grey"]), label=f"{cat} ({len(group)})", edgecolor="white", linewidth=0.25)
        axs[0].set_xlabel("release manifest files ranked by size")
        axs[0].set_ylabel("log10(bytes + 1)")
        axs[0].set_title("Release manifest size spectrum", loc="left", fontweight="bold")
        axs[0].legend(frameon=False, fontsize=5.2, ncol=2, loc="upper left", handletextpad=0.2, columnspacing=0.8)
        clean_ax(axs[0])
    else:
        axs[0].axis("off")
    checks = relv.get("checks", [])
    if checks:
        chk = pd.DataFrame(checks)
        score = chk.get("status", pd.Series([""] * len(chk))).astype(str).str.lower().map({"pass": 1.0, "passed": 1.0, "warn": 0.0, "warning": 0.0, "fail": -1.0, "failed": -1.0}).fillna(0.0)
        axs[1].imshow(score.to_numpy()[:, None], aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        labels = chk.get("name", pd.Series([f"check {i+1}" for i in range(len(chk))])).astype(str).str.replace("_", " ", regex=False).str.slice(0, 36)
        axs[1].set_yticks(np.arange(len(chk)))
        axs[1].set_yticklabels(labels, fontsize=4.9)
        axs[1].set_xticks([0])
        axs[1].set_xticklabels(["status"], fontsize=5.6)
        for yi, row in chk.iterrows():
            metric = str(row.get("metric", ""))
            metric = metric[:30] + ("..." if len(metric) > 30 else "")
            axs[1].text(0, yi, str(row.get("status", "")).upper(), ha="center", va="center", fontsize=4.8, fontweight="bold", color=COLORS["ink"])
            if metric:
                axs[1].text(0.60, yi, metric, ha="left", va="center", fontsize=4.2, color=COLORS["grey"], transform=axs[1].get_yaxis_transform())
        axs[1].set_xlim(-0.5, 1.65)
        axs[1].set_title("Release verification matrix", loc="left", fontweight="bold")
        axs[1].tick_params(length=0)
        for spine in axs[1].spines.values():
            spine.set_visible(False)
    else:
        axs[1].axis("off")
    if not checksums.empty and {"path", "status", "bytes"}.issubset(checksums.columns):
        c = checksums.copy()
        c["category"] = c["path"].map(release_category)
        c["bytes"] = pd.to_numeric(c["bytes"], errors="coerce").fillna(0)
        c = c.sort_values("bytes", ascending=True).reset_index(drop=True)
        palette = {"data": COLORS["green"], "results": COLORS["blue"], "code": COLORS["rose"], "docs": COLORS["purple"], "metadata": COLORS["grey"]}
        for cat, group in c.groupby("category", sort=False):
            axs[2].vlines(group.index, 0, np.log10(group["bytes"] + 1), color=palette.get(cat, COLORS["grey"]), alpha=0.22, linewidth=0.75)
            axs[2].scatter(group.index, np.log10(group["bytes"] + 1), s=28, color=palette.get(cat, COLORS["grey"]), alpha=0.82, edgecolor="white", linewidth=0.35, label=f"{cat} ({len(group)})")
        axs[2].set_xlabel("core checksum artifacts ranked by size")
        axs[2].set_ylabel("log10(bytes + 1)")
        axs[2].set_title("Checksum-covered core artifacts", loc="left", fontweight="bold")
        axs[2].legend(frameon=False, fontsize=5.5, loc="upper left", handletextpad=0.2)
        clean_ax(axs[2])
    else:
        axs[2].axis("off")
    packages = env.get("packages", {})
    pkg_rows = []
    for key in ["python", "torch", "numpy", "pandas", "matplotlib", "scanpy", "anndata"]:
        value = packages.get(key, "") if isinstance(packages, dict) else ""
        if isinstance(value, dict):
            value = value.get("version", "available")
        if value:
            pkg_rows.append((key, str(value)[:30]))
    release_rows = [
        ("DOI status", str(relv.get("doi_status", doi.get("status", "")))),
        ("DOI", str(doi.get("doi", relv.get("doi", ""))).replace("https://doi.org/", "")),
        ("bundle MB", f"{float(relv.get('bundle_bytes', 0)) / 1e6:.1f}"),
        ("tar members", str(relv.get("tar_member_count", ""))),
        ("manifest rows", str(relv.get("upload_manifest_rows", ""))),
        ("critical artifacts", str(relv.get("critical_artifacts_checked", ""))),
    ]
    rows = release_rows + pkg_rows[:6]
    draw_mini_table(axs[3], rows, "Release identity and environment", left_width=0.43, row_height=0.070, fontsize=5.9)
    save_fig(fig, "supplementary_figure_s14_reproducibility_release", "Supplementary Figure S14. Reproducibility and release map", "nature_release")


def figure_s15() -> None:
    base = read_csv("results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv")
    train = read_csv("results/nature_main/cytassist_rep2_radius55/training_history.csv")
    env = read_json("results/nature_release/environment_report.json")
    metrics = read_csv("results/nature_main/cytassist_rep2_radius55/metrics.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6))
    title_box(fig, "Supplementary Fig. 15 | Compute and scalability", "Runtime-accuracy and memory profiles provide methods-oriented reproducibility context.")
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.090, top=0.865, hspace=0.42, wspace=0.32)
    axs = axs.ravel()
    for ax, label in zip(axs, "ABCD"):
        panel_label(ax, label)
    if not base.empty:
        rt = base.dropna(subset=["runtime_seconds"]) if "runtime_seconds" in base else base
        if not rt.empty:
            colors = rt["source"].map({"formal_baseline": COLORS["blue"], "simple_baseline": COLORS["green"], "main_model": COLORS["rose"]}).fillna(COLORS["grey"])
            sizes = np.clip(pd.to_numeric(rt.get("mean_celltype_pearson", pd.Series([0.2] * len(rt))), errors="coerce").fillna(0).add(0.05) * 130, 24, 135)
            axs[0].scatter(pd.to_numeric(rt["runtime_seconds"], errors="coerce"), pd.to_numeric(rt["jsd"], errors="coerce"), s=sizes, c=colors, alpha=0.82, edgecolor="white", linewidth=0.45)
            for _, r in rt.sort_values("jsd").head(7).iterrows():
                axs[0].text(float(r["runtime_seconds"]), float(r["jsd"]), str(r["method"])[:16], fontsize=5.1, ha="left", va="center")
            axs[0].set_xscale("log")
            axs[0].set_xlabel("runtime seconds (log)")
            axs[0].set_ylabel("JSD")
            axs[0].set_title("runtime-accuracy envelope", loc="left", fontweight="bold")
        mem = base.dropna(subset=["runtime_seconds", "peak_cuda_memory_mb"]) if {"runtime_seconds", "peak_cuda_memory_mb"}.issubset(base.columns) else pd.DataFrame()
        if not mem.empty:
            x = pd.to_numeric(mem["runtime_seconds"], errors="coerce")
            y = pd.to_numeric(mem["peak_cuda_memory_mb"], errors="coerce")
            c = pd.to_numeric(mem["jsd"], errors="coerce")
            sc = axs[1].scatter(x, y, c=c, cmap="viridis_r", s=72, alpha=0.82, edgecolor="white", linewidth=0.5)
            axs[1].set_xscale("log")
            axs[1].set_xlabel("runtime seconds (log)")
            axs[1].set_ylabel("peak CUDA memory (MB)")
            axs[1].set_title("memory-runtime profile", loc="left", fontweight="bold")
            cb = fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.015)
            cb.ax.tick_params(labelsize=5.0, length=2)
            cb.set_label("JSD", fontsize=5.6)
    if not train.empty:
        for col, color in [("loss_total", COLORS["rose"]), ("loss_expr", COLORS["blue"]), ("loss_prop", COLORS["green"]), ("loss_uncertainty", COLORS["purple"])]:
            if col in train.columns:
                y = pd.to_numeric(train[col], errors="coerce").rolling(9, min_periods=1).mean()
                axs[2].plot(pd.to_numeric(train["step"], errors="coerce"), y, lw=1.0, color=color, label=col.replace("loss_", ""))
        axs[2].set_yscale("log")
        axs[2].set_title("training loss components", loc="left", fontweight="bold")
        axs[2].set_xlabel("step")
        axs[2].set_ylabel("loss (log)")
        axs[2].legend(frameon=False, ncol=2, fontsize=5.5)
    torch = env.get("torch", {})
    devices = torch.get("devices", [])
    hw = devices[0] if devices else {}
    final = metrics.iloc[0].to_dict() if not metrics.empty else {}
    rows = [
        ("GPU", str(hw.get("name", ""))[:34]),
        ("GPU memory", f"{hw.get('total_memory_mb', '')} MB"),
        ("CUDA", str(torch.get("cuda_version", ""))),
        ("final JSD", f"{float(final.get('jsd', np.nan)):.4f}" if final else ""),
        ("spotwise cosine", f"{float(final.get('spotwise_cosine', np.nan)):.4f}" if final else ""),
        ("supervised spots", str(int(final.get("num_supervised_spots", 0))) if final else ""),
    ]
    draw_mini_table(axs[3], rows, "compute context", left_width=0.44, row_height=0.105, fontsize=6.2)
    for ax in axs[:3]:
        clean_ax(ax)
    save_fig(fig, "supplementary_figure_s15_compute_scalability", "Supplementary Figure S15. Compute and scalability", "baseline runtime + training history + environment")


def write_manifests() -> None:
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT / "supporting_package_manifest.csv", index=False, encoding="utf-8")
    manifest.to_markdown(OUT / "supporting_package_manifest.md", index=False)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(OUT / "missing_or_server_needed.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["artifact", "reason", "server_hint"]).to_csv(OUT / "missing_or_server_needed.csv", index=False, encoding="utf-8")
    with (OUT / "README.md").open("w", encoding="utf-8") as f:
        f.write("# WaveST-Gate Nature Supporting Package\n\n")
        f.write("This package contains first-pass, reproducible supporting figures S1-S15 and supplementary tables S1-S19.\n\n")
        f.write("Folders:\n\n")
        f.write("- `figures/`: PDF, SVG, PNG and TIFF exports.\n")
        f.write("- `tables/`: CSV and Markdown exports.\n")
        f.write("- `source_data/`: copied source-data references where compact.\n")
        f.write("- `qa/`: reserved for render and consistency checks.\n\n")
        f.write("The first-pass outputs prioritize completeness, evidence traceability and consistent visual style. Manual panel-level polishing can follow.\n")


def main() -> None:
    save_all_tables()
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
    figure_s7()
    figure_s8()
    figure_s9()
    figure_s10()
    figure_s11()
    figure_s12()
    figure_s13()
    figure_s14()
    figure_s15()
    write_manifests()
    print(f"Wrote supporting package to {OUT}")


if __name__ == "__main__":
    main()
