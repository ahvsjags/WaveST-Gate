"""Export reference-style fully vector PDF files.

The reference PDFs in results/nature_manuscript_figures/参考PDF are pure PDF
content streams: text plus vector paths, with no image XObjects.  This exporter
uses the edit-priority SVGs and replaces embedded raster panels with editable
vector color tiles so PDF editors can select/move all visible elements.
"""

from __future__ import annotations

import base64
import csv
import io
import math
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image

ROOT = Path("/mnt/WaveST-Gate")
sys.path.insert(0, str(ROOT / "tools"))

from export_editable_pdf_from_svg import (  # noqa: E402
    PDF,
    Renderer,
    _apply,
    _f,
    _fmt,
    _len,
    _parse_transform,
    _pt,
    _render_element,
    _scale,
)


SVG_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_edit_priority"
OUT_DIR = ROOT / "results/nature_manuscript_figures/editable_pdf_reference_style_all_vector"


class ReferenceStyleRenderer(Renderer):
    """Renderer that turns SVG images into vector tile rectangles."""

    target_tiles_per_image = 18000

    def draw_image(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        href = attrs.get("href") or attrs.get("{http://www.w3.org/1999/xlink}href") or attrs.get("xlink:href")
        if not href or not href.startswith("data:image"):
            return
        x, y = _apply(m, _f(attrs.get("x")), _f(attrs.get("y")))
        w = _f(attrs.get("width")) * _scale(m)
        h = _f(attrs.get("height")) * _scale(m)
        if w <= 0 or h <= 0:
            return
        header, encoded = href.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")

        step = max(1, int(math.ceil(math.sqrt((w * h) / self.target_tiles_per_image))))
        cols = max(1, int(math.ceil(w / step)))
        rows = max(1, int(math.ceil(h / step)))
        # Resize to the displayed tile grid; each sampled pixel becomes a vector
        # rectangle. Quantizing improves run-length merging and edit performance.
        sample = image.resize((cols, rows), Image.Resampling.BOX)
        rgba = sample.getdata()
        qdata: list[tuple[int, int, int, int]] = []
        for r, g, b, a in rgba:
            if a < 10:
                qdata.append((0, 0, 0, 0))
            else:
                qdata.append((round(r / 8) * 8, round(g / 8) * 8, round(b / 8) * 8, 255))

        last_color: tuple[int, int, int, int] | None = None
        self.content.append(b"q\n")
        for row in range(rows):
            col = 0
            while col < cols:
                color = qdata[row * cols + col]
                if color[3] == 0:
                    col += 1
                    continue
                run = 1
                while col + run < cols and qdata[row * cols + col + run] == color:
                    run += 1
                if color != last_color:
                    self.content.append(
                        f"{_fmt(color[0] / 255)} {_fmt(color[1] / 255)} {_fmt(color[2] / 255)} rg\n".encode("ascii")
                    )
                    last_color = color
                tx = x + col * step
                ty = y + row * step
                tw = min(step * run, w - col * step)
                th = min(step, h - row * step)
                px, py_top = _pt(self.page_h_px, tx, ty)
                self.content.append(
                    f"{_fmt(px)} {_fmt(py_top - _len(th))} {_fmt(_len(tw))} {_fmt(_len(th))} re\nf\n".encode("ascii")
                )
                col += run
        self.content.append(b"Q\n")


def convert(svg_path: Path, pdf_path: Path) -> dict[str, Any]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    view_box = root.attrib.get("viewBox", "")
    if view_box:
        _, _, width, height = [float(x) for x in view_box.split()]
    else:
        width = _f(root.attrib.get("width"))
        height = _f(root.attrib.get("height"))
    pdf = PDF()
    renderer = ReferenceStyleRenderer(pdf=pdf, page_h_px=height, page_w_px=width, content=[])
    for child in list(root):
        _render_element(renderer, child, _parse_transform(root.attrib.get("transform")))
    content = b"".join(renderer.content)
    pdf.write(pdf_path, _len(width), _len(height), content)
    return {
        "asset": svg_path.stem,
        "pdf": str(pdf_path),
        "width_px": int(width),
        "height_px": int(height),
        "pdf_bytes": pdf_path.stat().st_size,
        "image_xobjects": len(pdf.image_cache),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for svg_path in sorted(SVG_DIR.glob("*.svg")):
        pdf_path = OUT_DIR / f"{svg_path.stem}.pdf"
        print(f"converting {svg_path.name} -> {pdf_path.name}", flush=True)
        rows.append(convert(svg_path, pdf_path))
    manifest = OUT_DIR / "editable_pdf_reference_style_all_vector_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Editable PDF Reference-Style All-Vector Exports",
        "",
        "These PDFs mimic the reference PDF structure: no image XObjects are embedded.",
        "",
        "- Text is PDF text.",
        "- Lines, shapes, panels, and sampled raster panels are vector drawing operations.",
        "- Former image panels are converted to editable vector color tiles, so appearance is approximate rather than pixel-identical.",
        "- Use `../editable_pdf_exact_overlay/` when exact first-view appearance is more important than pure-vector editability.",
        "",
        "| Asset | Size | Image XObjects | PDF |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['asset']} | {row['width_px']}x{row['height_px']} | {row['image_xobjects']} | `{Path(row['pdf']).name}` |")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} reference-style vector PDFs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
