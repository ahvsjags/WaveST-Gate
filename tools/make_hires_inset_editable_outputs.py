from __future__ import annotations

import base64
import csv
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageOps
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import make_reportlab_dejavu_editable_pdf as dejavu_pdf


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "results" / "nature_manuscript_figures"
SOURCE_ROOT = FIG_ROOT / "TRUE_EDITABLE_TEXT_NO_VISUAL_CHANGE"
OUT_ROOT = FIG_ROOT / "TRUE_EDITABLE_TEXT_HIRES_FULL_CHECK"
SOURCE_SVG = SOURCE_ROOT / "svg"
SOURCE_BG = SOURCE_ROOT / "background_no_text_png"
OUT_SVG = OUT_ROOT / "svg"
OUT_BG = OUT_ROOT / "background_no_text_png"
OUT_PDF = OUT_ROOT / "pdf_dejavu_embedded"
OUT_INSETS = OUT_ROOT / "hires_inset_png"
QA_DIR = OUT_ROOT / "_qa"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


HIRES_INSETS_BY_STEM = {
    "figure_1_workflow_schematic": [
        # Main workflow output cards in panel A.
        {
            "label": "method_image_gate",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/image_gate_map.png",
            "box": (2686, 446, 2910, 588),
        },
        {
            "label": "method_reference_gate",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/reference_gate_map.png",
            "box": (2946, 446, 3170, 588),
        },
        {
            "label": "method_uncertainty",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/spot_uncertainty_map.png",
            "box": (3206, 446, 3430, 588),
        },
        {
            "label": "method_boundary",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_he_overlay.png",
            "box": (2686, 682, 3056, 830),
        },
        # Small source-image thumbnails elsewhere in Figure 1.
        {
            "label": "he_tissue",
            "path": ROOT / "data/raw/wavestgate_breast_core/10x/visium/extracted/spatial/tissue_hires_image.png",
            "box": (118, 1489, 304, 1639),
        },
        {
            "label": "registered_he",
            "path": ROOT / "data/raw/wavestgate_breast_core/10x/visium/extracted/spatial/tissue_hires_image.png",
            "box": (120, 2118, 436, 2324),
        },
        # Interpretation cards in panel G.
        {
            "label": "interpretation_gate",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/image_gate_map.png",
            "box": (2386, 2948, 2554, 3066),
        },
        {
            "label": "interpretation_boundary",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_he_overlay.png",
            "box": (2584, 2948, 2857, 3066),
        },
        {
            "label": "interpretation_niche",
            "path": ROOT / "results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_map.png",
            "box": (2386, 3130, 2554, 3208),
        },
        {
            "label": "interpretation_pathology",
            "path": ROOT / "results/nature_external_pathology_validation/pathology_niche_by_class_heatmap.png",
            "box": (2584, 3130, 2857, 3208),
        },
    ],
}

HIRES_SCALE = 6


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def hires_asset(stem: str, item: dict[str, object]) -> Path:
    x0, y0, x1, y1 = item["box"]  # type: ignore[misc]
    display_size = (int(x1) - int(x0), int(y1) - int(y0))
    out = OUT_INSETS / f"{stem}_{item['label']}_hires.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(item["path"]).convert("RGB")  # type: ignore[arg-type]
    source_aspect = image.width / max(image.height, 1)
    target_aspect = display_size[0] / max(display_size[1], 1)
    if target_aspect >= source_aspect:
        max_width = image.width
        max_height = max(1, round(max_width / target_aspect))
    else:
        max_height = image.height
        max_width = max(1, round(max_height * target_aspect))
    target_size = (display_size[0] * HIRES_SCALE, display_size[1] * HIRES_SCALE)
    if target_size[0] > max_width or target_size[1] > max_height:
        scale = max(1.0, min(max_width / display_size[0], max_height / display_size[1]))
        target_size = (max(display_size[0], round(display_size[0] * scale)), max(display_size[1], round(display_size[1] * scale)))
    ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS).save(out)
    return out


def add_svg_hires_insets(svg_path: Path, out_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    layer = ET.Element("g", {"id": "hires_raster_insets_LOCKED_DO_NOT_EDIT"})
    for item in HIRES_INSETS_BY_STEM.get(svg_path.stem, []):
        x0, y0, x1, y1 = item["box"]
        group = ET.SubElement(layer, "g", {"id": f"hires_{item['label']}"})
        href = data_uri(hires_asset(svg_path.stem, item))
        image = ET.SubElement(
            group,
            "image",
            {
                "x": str(x0),
                "y": str(y0),
                "width": str(x1 - x0),
                "height": str(y1 - y0),
                "preserveAspectRatio": "none",
                "href": href,
            },
        )
        image.set(f"{{{XLINK_NS}}}href", href)
    # Put hires insets above the no-text background but below editable text.
    children = list(root)
    insert_at = 0
    for idx, child in enumerate(children):
        if child.attrib.get("id") == "editable_text_LAYER_VISIBLE":
            insert_at = idx
            break
    root.insert(insert_at, layer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def write_pdf_with_hires_background(stem: str) -> Path:
    bg = Image.open(OUT_BG / f"{stem}.png").convert("RGB")
    width, height = bg.size
    out = OUT_PDF / f"{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_bg = QA_DIR / f"{stem}_hires_background_composite.png"
    tmp_bg.parent.mkdir(parents=True, exist_ok=True)
    composite = bg.convert("RGBA")
    for item in HIRES_INSETS_BY_STEM.get(stem, []):
        x0, y0, x1, y1 = item["box"]
        inset = Image.open(hires_asset(stem, item)).convert("RGBA").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        composite.alpha_composite(inset, (x0, y0))
    composite.convert("RGB").save(tmp_bg)

    # Reuse the XML/text renderer by temporarily swapping only the new output
    # background image. The original editable output directory is not touched.
    original_bg = OUT_BG / f"{stem}.png"
    backup = QA_DIR / f"{stem}_original_background_backup.png"
    shutil.copy2(original_bg, backup)
    try:
        shutil.copy2(tmp_bg, original_bg)
        produced = dejavu_pdf.write_pdf(OUT_SVG / f"{stem}.svg")
        if produced != out:
            produced.replace(out)
    finally:
        shutil.copy2(backup, original_bg)
    return out


def main() -> int:
    for folder in (OUT_BG, OUT_SVG, OUT_PDF, OUT_INSETS, QA_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for path in SOURCE_BG.glob("figure_*.png"):
        shutil.copy2(path, OUT_BG / path.name)
    for path in SOURCE_SVG.glob("figure_*.svg"):
        out = OUT_SVG / path.name
        if path.stem in HIRES_INSETS_BY_STEM:
            add_svg_hires_insets(path, out)
        else:
            shutil.copy2(path, out)

    dejavu_pdf.SOURCE_ROOT = OUT_ROOT
    dejavu_pdf.SVG_DIR = OUT_SVG
    dejavu_pdf.BACKGROUND_DIR = OUT_BG
    dejavu_pdf.OUT_DIR = OUT_PDF

    regular = dejavu_pdf.find_font("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf")
    bold = dejavu_pdf.find_font("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold)))

    rows = []
    for svg in sorted(OUT_SVG.glob("figure_*.svg")):
        if svg.stem in HIRES_INSETS_BY_STEM:
            out_pdf = write_pdf_with_hires_background(svg.stem)
        else:
            out_pdf = dejavu_pdf.write_pdf(svg)
        rows.append({"figure": svg.stem, "svg": str(svg.relative_to(OUT_ROOT)), "pdf": str(out_pdf.relative_to(OUT_ROOT)), "bytes": out_pdf.stat().st_size})

    with (OUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "svg", "pdf", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT_ROOT / "README.md").write_text(
        "# TRUE_EDITABLE_TEXT_HIRES_INSETS\n\n"
        "Same editable text layer as TRUE_EDITABLE_TEXT_NO_VISUAL_CHANGE, with high-resolution raster inset layers over all Figure 1 source-image thumbnails found in the manuscript figure generator.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} figures to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
