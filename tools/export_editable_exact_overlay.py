"""Build exact-view editable overlay SVG/PDF exports.

These files prioritize visual identity with the approved final PNGs.  The
original PNG is kept as the visible exact background, while editable source
objects are stored in a separate layer that can be enabled for editing.
"""

from __future__ import annotations

import base64
import csv
import re
import sys
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path("/mnt/WaveST-Gate")
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from export_editable_pdf_from_svg import PDF, Renderer, _f, _len, _render_element  # noqa: E402


FINAL_PNG_DIR = ROOT / "results/nature_manuscript_figures"
EDIT_SVG_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_edit_priority"
SVG_OUT_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_exact_overlay"
PDF_OUT_DIR = ROOT / "results/nature_manuscript_figures/editable_pdf_exact_overlay"

SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.S)
GROUP_RE_TEMPLATE = r'  <g id="{group_id}">\n(?P<body>.*?)\n  </g>'


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _inject_layer_namespaces(svg_open: str) -> str:
    if "xmlns:inkscape" not in svg_open:
        svg_open = svg_open.replace(
            "<svg ",
            '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ',
            1,
        )
    return svg_open


def _extract_svg_open(text: str) -> str:
    match = SVG_OPEN_RE.search(text)
    if not match:
        raise ValueError("No SVG root found")
    return _inject_layer_namespaces("<svg" + match.group(1) + ">")


def _extract_group_body(text: str, group_id: str) -> str:
    pattern = GROUP_RE_TEMPLATE.format(group_id=re.escape(group_id))
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"No {group_id} group found")
    return match.group("body")


def _svg_size(svg_path: Path) -> tuple[int, int]:
    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.split()]
        return int(parts[2]), int(parts[3])
    return int(_f(root.attrib.get("width"))), int(_f(root.attrib.get("height")))


def export_svg(svg_path: Path, png_path: Path) -> dict[str, int | str]:
    text = svg_path.read_text(encoding="utf-8", errors="ignore")
    svg_open = _extract_svg_open(text)
    width, height = _svg_size(svg_path)
    href = _data_uri(png_path)
    groups = {
        "embedded_image_panels": _extract_group_body(text, "embedded_image_panels"),
        "editable_vector_objects": _extract_group_body(text, "editable_vector_objects"),
        "editable_text_objects": _extract_group_body(text, "editable_text_objects"),
    }
    out_path = SVG_OUT_DIR / svg_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = f'''<?xml version="1.0" encoding="UTF-8"?>
{svg_open}
  <title>{svg_path.stem}</title>
  <desc>Exact-view editable overlay. The approved PNG is visible for identical appearance; editable objects are in a hidden layer that can be turned on for editing.</desc>
  <g id="exact_original_png_background" inkscape:groupmode="layer" inkscape:label="EXACT original PNG background" style="display:inline" pointer-events="none">
    <image x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" href="{href}" xlink:href="{href}"/>
  </g>
  <g id="editable_objects_turn_on_for_editing" inkscape:groupmode="layer" inkscape:label="EDITABLE objects - turn on, then hide/lock background" style="display:none">
    <g id="embedded_image_panels">
{groups["embedded_image_panels"]}
    </g>
    <g id="editable_vector_objects">
{groups["editable_vector_objects"]}
    </g>
    <g id="editable_text_objects">
{groups["editable_text_objects"]}
    </g>
  </g>
</svg>
'''
    out_path.write_text(out, encoding="utf-8")
    return {
        "asset": svg_path.stem,
        "width_px": width,
        "height_px": height,
        "editable_svg": str(out_path),
        "text_objects": out.count("<text "),
        "image_objects": out.count("<image "),
        "svg_bytes": out_path.stat().st_size,
    }


def _pdf_escape_name(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})".encode("ascii", errors="replace")


def _add_lossless_pdf_image(pdf: PDF, png_path: Path) -> tuple[str, int, int]:
    image = Image.open(png_path).convert("RGBA")
    width, height = image.size
    alpha = image.getchannel("A")
    has_alpha = alpha.getextrema() != (255, 255)
    smask_ref = ""
    if has_alpha:
        mask_payload = zlib.compress(alpha.tobytes(), 6)
        smask_id = pdf.add(
            (
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length {len(mask_payload)} >>\nstream\n"
            ).encode("ascii")
            + mask_payload
            + b"\nendstream"
        )
        smask_ref = f" /SMask {smask_id} 0 R"
    rgb_payload = zlib.compress(image.convert("RGB").tobytes(), 6)
    image_id = pdf.add(
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode{smask_ref} /Length {len(rgb_payload)} >>\nstream\n"
        ).encode("ascii")
        + rgb_payload
        + b"\nendstream"
    )
    name = f"Im{len(pdf.image_cache) + 1}"
    pdf.image_cache[f"lossless:{png_path}"] = (name, image_id)
    return name, width, height


def _draw_lossless_fullpage_background(pdf: PDF, png_path: Path, width_px: float, height_px: float) -> bytes:
    name, _, _ = _add_lossless_pdf_image(pdf, png_path)
    return (
        b"q\n"
        + f"{_len(width_px):.4f} 0 0 {_len(height_px):.4f} 0 0 cm\n/{name} Do\nQ\n".encode("ascii")
    )


def _write_layered_pdf(
    pdf: PDF,
    path: Path,
    page_w_pt: float,
    page_h_pt: float,
    background_content: bytes,
    editable_content: bytes,
) -> None:
    background_layer_id = pdf.add(b"<< /Type /OCG /Name " + _pdf_escape_name("EXACT original PNG background") + b" >>")
    editable_layer_id = pdf.add(b"<< /Type /OCG /Name " + _pdf_escape_name("EDITABLE objects - turn on for editing") + b" >>")
    content = (
        b"/OC /BG BDC\n"
        + background_content
        + b"EMC\n"
        + b"/OC /EDIT BDC\n"
        + editable_content
        + b"EMC\n"
    )
    compressed = zlib.compress(content, 6)
    content_id = pdf.add(
        f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
        + compressed
        + b"\nendstream"
    )
    font_regular = pdf.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold = pdf.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    xobjects = " ".join(f"/{name} {oid} 0 R" for name, oid in pdf.image_cache.values())
    extg = " ".join(f"/{name} {oid} 0 R" for name, oid in pdf.alpha_cache.values())
    resources = f"<< /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >>"
    if xobjects:
        resources += f" /XObject << {xobjects} >>"
    if extg:
        resources += f" /ExtGState << {extg} >>"
    resources += f" /Properties << /BG {background_layer_id} 0 R /EDIT {editable_layer_id} 0 R >> >>"

    pages_id = len(pdf.objects) + 2
    page_id = pdf.add(
        (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {page_w_pt:.4f} {page_h_pt:.4f}] "
            f"/Resources {resources} /Contents {content_id} 0 R >>"
        ).encode("ascii")
    )
    pages_id = pdf.add(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("ascii"))
    oc_properties = (
        f"/OCProperties << /OCGs [{background_layer_id} 0 R {editable_layer_id} 0 R] "
        f"/D << /Name (Layers) /Order [{background_layer_id} 0 R {editable_layer_id} 0 R] "
        f"/ON [{background_layer_id} 0 R] /OFF [{editable_layer_id} 0 R] >> >>"
    )
    catalog_id = pdf.add(f"<< /Type /Catalog /Pages {pages_id} 0 R {oc_properties} >>".encode("ascii"))

    offsets: list[int] = [0]
    output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    for i, obj in enumerate(pdf.objects, 1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(pdf.objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(pdf.objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def export_pdf(svg_path: Path, png_path: Path) -> dict[str, int | str]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    view_box = root.attrib.get("viewBox", "")
    if view_box:
        _, _, width, height = [float(x) for x in view_box.split()]
    else:
        width = _f(root.attrib.get("width"))
        height = _f(root.attrib.get("height"))

    pdf = PDF()
    background_content = _draw_lossless_fullpage_background(pdf, png_path, width, height)

    editable_renderer = Renderer(pdf=pdf, page_h_px=height, page_w_px=width, content=[], skip_large_images=True)
    for child in list(root):
        _render_element(editable_renderer, child, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    out_path = PDF_OUT_DIR / f"{svg_path.stem}.pdf"
    _write_layered_pdf(
        pdf,
        out_path,
        _len(width),
        _len(height),
        background_content,
        b"".join(editable_renderer.content),
    )
    return {
        "asset": svg_path.stem,
        "width_px": int(width),
        "height_px": int(height),
        "editable_pdf": str(out_path),
        "pdf_bytes": out_path.stat().st_size,
    }


def write_readme(svg_rows: list[dict[str, int | str]], pdf_rows: list[dict[str, int | str]]) -> None:
    for out_dir, kind, rows in ((SVG_OUT_DIR, "SVG", svg_rows), (PDF_OUT_DIR, "PDF", pdf_rows)):
        manifest = out_dir / f"editable_{kind.lower()}_exact_overlay_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            f"# Editable {kind} Exact-Overlay Exports",
            "",
            "These files open with the approved final PNG as the visible background, so the first view matches the original figure exactly.",
            "",
            "- The original PNG background is preserved and visible.",
            "- Editable objects are stored in a separate layer named `EDITABLE objects - turn on for editing`.",
            "- For editing, turn on the editable layer, then hide or lock the exact PNG background layer.",
            "- This is the only way to keep a pixel-identical first view while also carrying editable objects.",
            "",
            f"| Asset | Size | {kind} |",
            "| --- | ---: | --- |",
        ]
        path_key = "editable_svg" if kind == "SVG" else "editable_pdf"
        for row in rows:
            lines.append(f"| {row['asset']} | {row['width_px']}x{row['height_px']} | `{Path(str(row[path_key])).name}` |")
        (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SVG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_rows: list[dict[str, int | str]] = []
    pdf_rows: list[dict[str, int | str]] = []
    for svg_path in sorted(EDIT_SVG_DIR.glob("*.svg")):
        png_path = FINAL_PNG_DIR / f"{svg_path.stem}.png"
        if not png_path.exists():
            raise FileNotFoundError(png_path)
        print(f"overlaying {svg_path.stem}", flush=True)
        svg_rows.append(export_svg(svg_path, png_path))
        pdf_rows.append(export_pdf(svg_path, png_path))
    write_readme(svg_rows, pdf_rows)
    print(f"Wrote exact-overlay SVGs to {SVG_OUT_DIR}", flush=True)
    print(f"Wrote exact-overlay PDFs to {PDF_OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
