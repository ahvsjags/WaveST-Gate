from __future__ import annotations

import base64
import csv
import html
import re
import shutil
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "nature_manuscript_figures"
TEXT_SVG_DIR = FIG_DIR / "editable_svg_edit_priority"
OUT_DIR = FIG_DIR / "UNCHANGED_VISIBLE_EDITABLE_TEXT"
OUT_SVG = OUT_DIR / "svg"
OUT_PDF = OUT_DIR / "pdf"
OUT_REF = OUT_DIR / "png_reference"
FIGURES = [
    "figure_1_workflow_schematic",
    "figure_2_spatial_cell_composition",
    "figure_3_baseline_performance",
    "figure_4_reliability_calibration",
    "figure_5_boundary_niche_pathology",
]
DPI = 600


def fnum(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def svg_size(root: ET.Element) -> tuple[int, int]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(x) for x in view_box.replace(",", " ").split()]
        return int(parts[2]), int(parts[3])
    return int(fnum(root.attrib.get("width"))), int(fnum(root.attrib.get("height")))


def attrs_to_str(attrs: dict[str, str]) -> str:
    return " ".join(f'{key}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items())


def element_inner_xml(element: ET.Element) -> str:
    text = element.text or ""
    children = "".join(ET.tostring(child, encoding="unicode") for child in list(element))
    return text + children


def extract_text_elements(svg_path: Path) -> list[ET.Element]:
    root = ET.parse(svg_path).getroot()
    texts = []
    for element in root.iter():
        if element.tag.split("}", 1)[-1] == "text":
            texts.append(element)
    return texts


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_svg(figure: str, png_path: Path, text_svg_path: Path) -> dict[str, int | str]:
    root = ET.parse(text_svg_path).getroot()
    width, height = svg_size(root)
    href = data_uri(png_path)
    text_items = extract_text_elements(text_svg_path)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f"  <title>{figure}</title>",
        "  <desc>Approved unchanged PNG background plus visible editable SVG text layer.</desc>",
        '  <g id="approved_png_background_LOCK_DO_NOT_EDIT" inkscape:groupmode="layer" inkscape:label="LOCKED approved unchanged PNG background">',
        f'    <image x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" href="{href}" xlink:href="{href}"/>',
        "  </g>",
        '  <g id="editable_text_LAYER_VISIBLE" inkscape:groupmode="layer" inkscape:label="EDITABLE TEXT visible - edit fonts here">',
    ]
    for item in text_items:
        attrs = dict(item.attrib)
        inner = element_inner_xml(item)
        lines.append(f"    <text {attrs_to_str(attrs)}>{inner}</text>")
    lines.extend(["  </g>", "</svg>", ""])
    out_path = OUT_SVG / f"{figure}.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "figure": figure,
        "width_px": width,
        "height_px": height,
        "text_objects": len(text_items),
        "svg": str(out_path),
        "svg_bytes": out_path.stat().st_size,
    }


class PDF:
    def __init__(self) -> None:
        self.objects: list[bytes] = []

    def add(self, payload: bytes) -> int:
        self.objects.append(payload)
        return len(self.objects)


def pdf_escape(text: str) -> bytes:
    text = html.unescape(text).replace("✓", "check")
    raw = text.encode("latin-1", errors="replace")
    raw = raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + raw + b")"


def rgb(value: str | None) -> tuple[float, float, float]:
    if not value or value == "none":
        return 0.0, 0.0, 0.0
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[return-value]
    if value.startswith("rgb"):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", value)[:3]]
        return tuple(n / 255.0 for n in nums)  # type: ignore[return-value]
    return 0.0, 0.0, 0.0


def len_pt(value: float) -> float:
    return value * 72.0 / DPI


def point(page_h_px: float, x: float, y: float) -> tuple[float, float]:
    return len_pt(x), len_pt(page_h_px - y)


def text_content(element: ET.Element) -> str:
    return "".join(element.itertext())


def add_background_image(pdf: PDF, png_path: Path) -> tuple[str, int, int, int]:
    image = Image.open(png_path).convert("RGB")
    width, height = image.size
    payload = zlib.compress(image.tobytes(), 6)
    image_id = pdf.add(
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(payload)} >>\nstream\n"
        ).encode("ascii")
        + payload
        + b"\nendstream"
    )
    return "Im1", image_id, width, height


def draw_pdf_text(element: ET.Element, page_h_px: float) -> bytes:
    text = text_content(element)
    if not text:
        return b""
    x = fnum(element.attrib.get("x"))
    y = fnum(element.attrib.get("y"))
    size = fnum(element.attrib.get("font-size"), 12.0)
    weight = str(element.attrib.get("font-weight", "400"))
    fill = element.attrib.get("fill", "#000000")
    opacity = fnum(element.attrib.get("fill-opacity"), 1.0)
    r, g, b = rgb(fill)
    font = "F2" if weight in {"600", "700", "800", "bold"} else "F1"
    # SVG uses dominant-baseline=hanging; approximate its visual top with PDF baseline.
    px, py = point(page_h_px, x, y + size * 0.86)
    commands = [
        "q",
        f"/GS{max(0, min(1000, int(round(opacity * 1000))))} gs",
        "BT",
        f"{fmt(r)} {fmt(g)} {fmt(b)} rg",
        f"/{font} {fmt(len_pt(size))} Tf",
        f"{fmt(px)} {fmt(py)} Td",
    ]
    return ("\n".join(commands) + "\n").encode("ascii") + pdf_escape(text) + b" Tj\nET\nQ\n"


def write_pdf(figure: str, png_path: Path, text_svg_path: Path) -> dict[str, int | str]:
    pdf = PDF()
    bg_name, bg_id, width, height = add_background_image(pdf, png_path)
    font_regular = pdf.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold = pdf.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    gs_entries = []
    for value in range(0, 1001):
        alpha = value / 1000
        gs_id = pdf.add(f"<< /Type /ExtGState /ca {fmt(alpha)} /CA {fmt(alpha)} >>".encode("ascii"))
        gs_entries.append((value, gs_id))

    content = bytearray()
    content.extend(b"q\n")
    content.extend(f"{fmt(len_pt(width))} 0 0 {fmt(len_pt(height))} 0 0 cm\n/{bg_name} Do\nQ\n".encode("ascii"))
    text_items = extract_text_elements(text_svg_path)
    for item in text_items:
        content.extend(draw_pdf_text(item, height))
    compressed = zlib.compress(bytes(content), 6)
    content_id = pdf.add(
        f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
        + compressed
        + b"\nendstream"
    )

    extg = " ".join(f"/GS{value} {oid} 0 R" for value, oid in gs_entries)
    resources = (
        f"<< /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> "
        f"/XObject << /{bg_name} {bg_id} 0 R >> /ExtGState << {extg} >> >>"
    )
    pages_id = len(pdf.objects) + 2
    page_id = pdf.add(
        (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {fmt(len_pt(width))} {fmt(len_pt(height))}] "
            f"/Resources {resources} /Contents {content_id} 0 R >>"
        ).encode("ascii")
    )
    pages_id = pdf.add(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("ascii"))
    catalog_id = pdf.add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(pdf.objects, 1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(pdf.objects)+1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(pdf.objects)+1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    out_path = OUT_PDF / f"{figure}.pdf"
    out_path.write_bytes(output)
    return {
        "figure": figure,
        "width_px": width,
        "height_px": height,
        "text_objects": len(text_items),
        "pdf": str(out_path),
        "pdf_bytes": out_path.stat().st_size,
    }


def main() -> int:
    for folder in (OUT_SVG, OUT_PDF, OUT_REF):
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for figure in FIGURES:
        png_path = FIG_DIR / f"{figure}.png"
        text_svg_path = TEXT_SVG_DIR / f"{figure}.svg"
        if not png_path.exists():
            raise FileNotFoundError(png_path)
        if not text_svg_path.exists():
            raise FileNotFoundError(text_svg_path)
        shutil.copy2(png_path, OUT_REF / png_path.name)
        svg_row = write_svg(figure, png_path, text_svg_path)
        pdf_row = write_pdf(figure, png_path, text_svg_path)
        rows.append(
            {
                "figure": figure,
                "png_reference": f"png_reference/{figure}.png",
                "editable_svg": f"svg/{figure}.svg",
                "editable_pdf": f"pdf/{figure}.pdf",
                "width_px": svg_row["width_px"],
                "height_px": svg_row["height_px"],
                "text_objects": svg_row["text_objects"],
                "svg_bytes": svg_row["svg_bytes"],
                "pdf_bytes": pdf_row["pdf_bytes"],
            }
        )
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    readme = """# UNCHANGED_VISIBLE_EDITABLE_TEXT

Purpose: keep the approved figure appearance unchanged while making labels/fonts selectable as vector text.

What this version does:

- Uses the approved final PNG as the full-page background, so image panels, heatmaps, H&E, shadows, and layout do not change.
- Adds a visible editable text layer above the background.
- The text layer is PDF/SVG text, so fonts and wording can be selected and edited in vector editors.
- The non-text scientific image content remains high-resolution raster, because H&E/heatmap/photo panels cannot become true vector without changing the science image.

Use:

- `pdf/` for PDF editors, Illustrator, Affinity, or Inkscape.
- `svg/` for SVG editing.
- `png_reference/` to confirm the approved underlying figure.

Do not use `final_clear_editable_vector/recommended_editable_pdf` for final appearance matching; it is an edit-priority reconstruction and differs from the approved PNG.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(rows)} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
