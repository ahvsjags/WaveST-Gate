from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "results" / "nature_manuscript_figures"
SOURCE_ROOT = FIG_ROOT / "TRUE_EDITABLE_TEXT_NO_VISUAL_CHANGE"
SVG_DIR = SOURCE_ROOT / "svg"
BACKGROUND_DIR = SOURCE_ROOT / "background_no_text_png"
OUT_DIR = SOURCE_ROOT / "pdf_dejavu_embedded"
DPI = 600


def find_font(*names: str) -> Path:
    candidates = [
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "native"
        / "poppler"
        / "Library"
        / "share"
        / "fonts",
        Path("C:/Windows/Fonts"),
    ]
    for root in candidates:
        if root.exists():
            for name in names:
                matches = list(root.rglob(name))
                if matches:
                    return matches[0]
    raise FileNotFoundError(", ".join(names))


def parse_transform(value: str | None) -> tuple[float, float, float, float, float, float]:
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not value:
        return matrix
    for name, raw in re.findall(r"(translate|scale|matrix)\(([^)]*)\)", value):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", raw)]
        if name == "translate":
            op = (1.0, 0.0, 0.0, 1.0, nums[0] if nums else 0.0, nums[1] if len(nums) > 1 else 0.0)
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            op = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            op = tuple(nums[:6]) if len(nums) >= 6 else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        matrix = multiply(matrix, op)  # type: ignore[arg-type]
    return matrix


def multiply(
    m1: tuple[float, float, float, float, float, float],
    m2: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply_matrix(m: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def scale_from_matrix(m: tuple[float, float, float, float, float, float]) -> float:
    a, b, c, d, _, _ = m
    sx = (a * a + b * b) ** 0.5
    sy = (c * c + d * d) ** 0.5
    return (sx + sy) / 2 if sx or sy else 1.0


def clean_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def rgb(value: str | None) -> tuple[float, float, float]:
    if not value or value == "none":
        return 0.0, 0.0, 0.0
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]
    if value.startswith("rgb"):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", value)[:3]]
        return tuple(n / 255 for n in nums)  # type: ignore[return-value]
    return 0.0, 0.0, 0.0


def read_size(root: ET.Element) -> tuple[int, int]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(x) for x in view_box.split()]
        return int(parts[2]), int(parts[3])
    return int(float(root.attrib["width"])), int(float(root.attrib["height"]))


def iter_texts(element: ET.Element, matrix: tuple[float, float, float, float, float, float]):
    local = multiply(matrix, parse_transform(element.attrib.get("transform")))
    if clean_tag(element.tag) == "text":
        yield element, local
    for child in list(element):
        yield from iter_texts(child, local)


def iter_images(element: ET.Element, matrix: tuple[float, float, float, float, float, float]):
    local = multiply(matrix, parse_transform(element.attrib.get("transform")))
    if clean_tag(element.tag) == "image":
        yield element, local
    for child in list(element):
        yield from iter_images(child, local)


def image_href(element: ET.Element) -> str:
    for key, value in element.attrib.items():
        if key == "href" or key.endswith("}href"):
            return value
    return ""


def data_image_reader(href: str) -> ImageReader | None:
    if not href.startswith("data:image/") or "," not in href:
        return None
    _, payload = href.split(",", 1)
    return ImageReader(io.BytesIO(base64.b64decode(payload)))


def px_to_pt(value: float) -> float:
    return value * 72 / DPI


def write_pdf(svg_path: Path) -> Path:
    root = ET.parse(svg_path).getroot()
    width, height = read_size(root)
    stem = svg_path.stem
    bg_path = BACKGROUND_DIR / f"{stem}.png"
    out_path = OUT_DIR / f"{stem}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_path), pagesize=(px_to_pt(width), px_to_pt(height)))
    c.drawImage(str(bg_path), 0, 0, width=px_to_pt(width), height=px_to_pt(height), mask="auto")
    for element, matrix in iter_images(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)):
        href = image_href(element)
        reader = data_image_reader(href)
        if reader is None:
            continue
        x, y = apply_matrix(matrix, float(element.attrib.get("x", 0)), float(element.attrib.get("y", 0)))
        image_w = float(element.attrib.get("width", 0)) * scale_from_matrix(matrix)
        image_h = float(element.attrib.get("height", 0)) * scale_from_matrix(matrix)
        if image_w <= 0 or image_h <= 0:
            continue
        is_full_background = abs(x) < 0.5 and abs(y) < 0.5 and abs(image_w - width) < 0.5 and abs(image_h - height) < 0.5
        if is_full_background:
            continue
        c.drawImage(
            reader,
            px_to_pt(x),
            px_to_pt(height - (y + image_h)),
            width=px_to_pt(image_w),
            height=px_to_pt(image_h),
            mask="auto",
        )
    for element, matrix in iter_texts(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)):
        label = html.unescape("".join(element.itertext()))
        if not label:
            continue
        x, y = apply_matrix(matrix, float(element.attrib.get("x", 0)), float(element.attrib.get("y", 0)))
        size = float(str(element.attrib.get("font-size", "12")).replace("px", "")) * scale_from_matrix(matrix)
        weight = str(element.attrib.get("font-weight", "400"))
        font_name = "DejaVuSans-Bold" if weight in {"600", "700", "800", "bold"} else "DejaVuSans"
        r, g, b = rgb(element.attrib.get("fill"))
        opacity = float(element.attrib.get("fill-opacity", "1"))
        c.saveState()
        if hasattr(c, "setFillAlpha"):
            c.setFillAlpha(max(0.0, min(1.0, opacity)))
        c.setFillColor(Color(r, g, b))
        c.setFont(font_name, px_to_pt(size))
        c.drawString(px_to_pt(x), px_to_pt(height - (y + size * 0.86)), label)
        c.restoreState()
    c.showPage()
    c.save()
    return out_path


def main() -> int:
    regular = find_font("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf")
    bold = find_font("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold)))
    rows = []
    for svg_path in sorted(SVG_DIR.glob("figure_*.svg")):
        out = write_pdf(svg_path)
        rows.append(f"{svg_path.stem},{out.relative_to(SOURCE_ROOT)},{out.stat().st_size}")
    (OUT_DIR / "manifest.csv").write_text("figure,pdf,bytes\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} PDFs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
