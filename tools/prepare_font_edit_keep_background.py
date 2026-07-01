from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "results" / "nature_manuscript_figures"
SOURCE_ROOT = FIG_ROOT / "FINAL_SUBMISSION_KEEP_BACKGROUND"
OUT_ROOT = FIG_ROOT / "FONT_EDIT_KEEP_BACKGROUND"


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree_contents(item, target)
        else:
            shutil.copy2(item, target)


def count_svg_text(svg_path: Path) -> int:
    return svg_path.read_text(encoding="utf-8", errors="replace").count("<text")


def main() -> int:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(SOURCE_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in ("pdf_exact_background", "svg_exact_background", "png_reference_original"):
        copy_tree_contents(SOURCE_ROOT / folder, OUT_ROOT / folder)

    rows: list[dict[str, str | int]] = []
    svg_dir = OUT_ROOT / "svg_exact_background"
    pdf_dir = OUT_ROOT / "pdf_exact_background"
    png_dir = OUT_ROOT / "png_reference_original"
    for png_path in sorted(png_dir.glob("*.png")):
        stem = png_path.stem
        svg_path = svg_dir / f"{stem}.svg"
        pdf_path = pdf_dir / f"{stem}.pdf"
        rows.append(
            {
                "figure": stem,
                "png_reference": str(png_path.relative_to(OUT_ROOT)),
                "editable_svg_keep_background": str(svg_path.relative_to(OUT_ROOT)) if svg_path.exists() else "",
                "editable_pdf_keep_background": str(pdf_path.relative_to(OUT_ROOT)) if pdf_path.exists() else "",
                "svg_text_objects": count_svg_text(svg_path) if svg_path.exists() else 0,
                "png_bytes": png_path.stat().st_size,
                "svg_bytes": svg_path.stat().st_size if svg_path.exists() else 0,
                "pdf_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
            }
        )

    manifest = OUT_ROOT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme = """# FONT_EDIT_KEEP_BACKGROUND

Use this folder when you only need to edit labels/fonts but must not lose H&E images, heatmaps, shaded backgrounds, or any approved visual content.

Recommended files:

- `svg_exact_background/`: SVG with the approved PNG visible as the locked/background layer and editable text/vector objects in a separate hidden layer.
- `pdf_exact_background/`: PDF with the same exact background plus editable text/vector objects in an optional content layer.
- `png_reference_original/`: visual reference PNGs for checking that nothing changed.

Editing workflow:

1. Open the SVG/PDF in Illustrator, Inkscape, Affinity Designer, or a PDF editor that supports layers.
2. Keep the exact background layer visible and lock it.
3. Turn on the editable objects/text layer only when you need to select labels.
4. Edit fonts or wording.
5. Before export, compare against `png_reference_original/` and make sure no panels/backgrounds disappeared.

Avoid `recommended_editable_*` for this specific job. Those are easier to edit, but they may simplify or omit background/image layers.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Prepared {OUT_ROOT}")
    print(f"Figures: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
