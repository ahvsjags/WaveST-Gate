from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from wavestgate.evaluation.manuscript_figures import build_manuscript_figures


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 480), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 80, 560, 400), outline=(255, 255, 255), width=12)
    draw.line((80, 400, 560, 80), fill=(30, 30, 30), width=8)
    img.save(path)


def test_build_manuscript_figures_outputs_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nature_dir = run_dir / "nature_analysis"
    prop_dir = nature_dir / "proportion_maps"
    tables_dir = tmp_path / "tables"
    pathology_dir = tmp_path / "pathology"

    for path, color in [
        (prop_dir / "predicted_top_celltypes_panel.png", (110, 150, 210)),
        (prop_dir / "predicted_tumor_immune_stromal_group_panel.png", (120, 180, 140)),
        (nature_dir / "image_gate_map.png", (180, 140, 210)),
        (nature_dir / "expression_gate_map.png", (210, 180, 120)),
        (nature_dir / "reference_gate_map.png", (120, 210, 210)),
        (nature_dir / "spot_uncertainty_map.png", (210, 120, 150)),
        (nature_dir / "uncertainty_calibration.png", (160, 160, 220)),
        (nature_dir / "risk_coverage_curve.png", (220, 160, 160)),
        (nature_dir / "boundary_he_overlay.png", (180, 200, 160)),
        (nature_dir / "boundary_sharpness_map.png", (200, 180, 160)),
        (nature_dir / "niche_map.png", (160, 200, 200)),
        (pathology_dir / "pathology_group_composition.png", (150, 170, 190)),
        (pathology_dir / "pathology_niche_by_class_heatmap.png", (190, 150, 170)),
    ]:
        _image(path, color)

    tables_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"method": "WaveST-Gate", "jsd": 0.01},
            {"method": "Baseline", "jsd": 0.2},
        ]
    ).to_csv(tables_dir / "table_3_baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "clean", "jsd_mean": 0.01},
            {"scenario": "dropout", "jsd_mean": 0.05},
        ]
    ).to_csv(tables_dir / "table_7_robustness_summary.csv", index=False)

    manifest = build_manuscript_figures(
        output_dir=tmp_path / "figures",
        run_dir=run_dir,
        tables_dir=tables_dir,
        external_pathology_dir=pathology_dir,
    )

    assert manifest["num_figures"] == 6
    assert manifest["num_fail"] == 0
    assert Path(manifest["figure_manifest_csv"]).exists()
    assert Path(manifest["figure_manifest_md"]).exists()
    frame = pd.read_csv(manifest["figure_manifest_csv"])
    assert set(frame["figure"]) >= {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"}
