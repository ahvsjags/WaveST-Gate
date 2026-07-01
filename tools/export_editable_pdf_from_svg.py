"""Convert rebuilt editable SVG manuscript figures into editable PDFs.

This intentionally avoids rasterizing the page.  SVG text and drawing
primitives are written as PDF text/vector operators, while photographic or
heatmap SVG image objects are embedded as PDF image XObjects.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path("/mnt/WaveST-Gate")
SVG_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_rebuild"
SVG_EDIT_PRIORITY_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_edit_priority"
PDF_DIR = ROOT / "results/nature_manuscript_figures/editable_pdf_rebuild"
PDF_EDIT_PRIORITY_DIR = ROOT / "results/nature_manuscript_figures/editable_pdf_edit_priority"
DPI = 600


def _clean_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _f(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    for suffix in ("px", "pt", "in"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return float(text or 0)


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _pdf_name(name: str) -> bytes:
    return f"/{name}".encode("ascii")


def _escape_text(text: str) -> bytes:
    # Base-14 Helvetica is used so the text remains editable in PDF editors.
    # The generated figure text is ASCII except a few check marks; substitute
    # those with an ASCII equivalent rather than turning all text into paths.
    text = html.unescape(text).replace("✓", "check")
    raw = text.encode("latin-1", errors="replace")
    raw = raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + raw + b")"


def _rgb(value: str | None) -> tuple[float, float, float] | None:
    if not value or value == "none":
        return None
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[return-value]
    if value.startswith("rgb"):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\\.?[0-9]+", value)[:3]]
        return tuple(n / 255.0 for n in nums)  # type: ignore[return-value]
    named = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 0.5, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }
    return named.get(value.lower(), (0.0, 0.0, 0.0))


def _style(element: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    style = element.attrib.get("style", "")
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    out.update({k: v for k, v in element.attrib.items() if not k.startswith("{")})
    return out


def _hidden(element: ET.Element) -> bool:
    style = _style(element)
    return style.get("display") == "none" or "display:none" in style.get("style", "")


def _matmul(m1: tuple[float, float, float, float, float, float], m2: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
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


def _parse_transform(text: str | None) -> tuple[float, float, float, float, float, float]:
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not text:
        return matrix
    for name, raw in re.findall(r"(translate|scale|matrix)\\(([^)]*)\\)", text):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\\.?[0-9]+", raw)]
        if name == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            op = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            op = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            op = tuple(nums[:6]) if len(nums) >= 6 else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        matrix = _matmul(matrix, op)  # type: ignore[arg-type]
    return matrix


def _apply(m: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _scale(m: tuple[float, float, float, float, float, float]) -> float:
    a, b, c, d, _, _ = m
    sx = math.hypot(a, b)
    sy = math.hypot(c, d)
    return (sx + sy) / 2.0 if sx or sy else 1.0


def _pt(page_h_px: float, x: float, y: float) -> tuple[float, float]:
    return x * 72.0 / DPI, (page_h_px - y) * 72.0 / DPI


def _len(value: float) -> float:
    return value * 72.0 / DPI


class PDF:
    def __init__(self) -> None:
        self.objects: list[bytes] = []
        self.image_cache: dict[str, tuple[str, int]] = {}
        self.alpha_cache: dict[float, tuple[str, int]] = {}

    def add(self, payload: bytes) -> int:
        self.objects.append(payload)
        return len(self.objects)

    def alpha(self, alpha: float) -> str:
        alpha = max(0.0, min(1.0, round(alpha, 3)))
        if alpha >= 0.999:
            return ""
        if alpha not in self.alpha_cache:
            name = f"GS{len(self.alpha_cache) + 1}"
            oid = self.add(f"<< /Type /ExtGState /ca {_fmt(alpha)} /CA {_fmt(alpha)} >>".encode("ascii"))
            self.alpha_cache[alpha] = (name, oid)
        return self.alpha_cache[alpha][0]

    def image(self, data_uri: str) -> tuple[str, int, int]:
        digest = hashlib.sha1(data_uri.encode("ascii", errors="ignore")).hexdigest()
        if digest in self.image_cache:
            name, oid = self.image_cache[digest]
            # Width/height are not stored in the cache tuple to keep resource map
            # simple; callers only need them for placement from SVG attrs.
            return name, 0, 0
        header, encoded = data_uri.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = im.size
        alpha = im.getchannel("A")
        has_alpha = alpha.getextrema() != (255, 255)

        rgb = im.convert("RGB")
        rgb_buf = io.BytesIO()
        rgb.save(rgb_buf, format="JPEG", quality=95, optimize=True)
        rgb_payload = rgb_buf.getvalue()
        smask_ref = ""
        if has_alpha:
            mask_payload = zlib.compress(alpha.tobytes(), 6)
            smask_id = self.add(
                (
                    f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                    f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length {len(mask_payload)} >>\nstream\n"
                ).encode("ascii")
                + mask_payload
                + b"\nendstream"
            )
            smask_ref = f" /SMask {smask_id} 0 R"
        image_id = self.add(
            (
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode{smask_ref} /Length {len(rgb_payload)} >>\nstream\n"
            ).encode("ascii")
            + rgb_payload
            + b"\nendstream"
        )
        name = f"Im{len(self.image_cache) + 1}"
        self.image_cache[digest] = (name, image_id)
        return name, width, height

    def write(self, path: Path, page_w: float, page_h: float, content: bytes) -> None:
        content_id = self.add(
            f"<< /Length {len(zlib.compress(content, 6))} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + zlib.compress(content, 6)
            + b"\nendstream"
        )
        font_regular = self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold = self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        xobjects = " ".join(f"/{name} {oid} 0 R" for name, oid in self.image_cache.values())
        extg = " ".join(f"/{name} {oid} 0 R" for name, oid in self.alpha_cache.values())
        resources = f"<< /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >>"
        if xobjects:
            resources += f" /XObject << {xobjects} >>"
        if extg:
            resources += f" /ExtGState << {extg} >>"
        resources += " >>"
        pages_id = len(self.objects) + 2
        page_id = self.add(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_fmt(page_w)} {_fmt(page_h)}] "
                f"/Resources {resources} /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        pages_id = self.add(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("ascii"))
        catalog_id = self.add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

        offsets: list[int] = [0]
        output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        for i, obj in enumerate(self.objects, 1):
            offsets.append(len(output))
            output.extend(f"{i} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(self.objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            output.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        output.extend(
            f"trailer\n<< /Size {len(self.objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(output)


@dataclass
class Renderer:
    pdf: PDF
    page_h_px: float
    page_w_px: float
    content: list[bytes]
    skip_large_images: bool = False

    def _set_color(self, color: tuple[float, float, float], stroke: bool = False) -> None:
        op = "RG" if stroke else "rg"
        self.content.append(f"{_fmt(color[0])} {_fmt(color[1])} {_fmt(color[2])} {op}\n".encode("ascii"))

    def _set_alpha(self, alpha: float) -> None:
        name = self.pdf.alpha(alpha)
        if name:
            self.content.append(f"/{name} gs\n".encode("ascii"))

    def _path_rect(self, x: float, y: float, w: float, h: float, rx: float = 0) -> bytes:
        if rx <= 0:
            px, py_top = _pt(self.page_h_px, x, y)
            return f"{_fmt(px)} {_fmt(py_top - _len(h))} {_fmt(_len(w))} {_fmt(_len(h))} re\n".encode("ascii")
        # Rounded rectangle path in PDF coordinates.
        k = 0.5522847498
        r = min(rx, w / 2, h / 2)
        x0, y0 = _pt(self.page_h_px, x, y + h)
        x1, y1 = _pt(self.page_h_px, x + w, y)
        r = _len(r)
        parts = [
            f"{_fmt(x0 + r)} {_fmt(y0)} m",
            f"{_fmt(x1 - r)} {_fmt(y0)} l",
            f"{_fmt(x1 - r + k*r)} {_fmt(y0)} {_fmt(x1)} {_fmt(y0 + r - k*r)} {_fmt(x1)} {_fmt(y0 + r)} c",
            f"{_fmt(x1)} {_fmt(y1 - r)} l",
            f"{_fmt(x1)} {_fmt(y1 - r + k*r)} {_fmt(x1 - r + k*r)} {_fmt(y1)} {_fmt(x1 - r)} {_fmt(y1)} c",
            f"{_fmt(x0 + r)} {_fmt(y1)} l",
            f"{_fmt(x0 + r - k*r)} {_fmt(y1)} {_fmt(x0)} {_fmt(y1 - r + k*r)} {_fmt(x0)} {_fmt(y1 - r)} c",
            f"{_fmt(x0)} {_fmt(y0 + r)} l",
            f"{_fmt(x0)} {_fmt(y0 + r - k*r)} {_fmt(x0 + r - k*r)} {_fmt(y0)} {_fmt(x0 + r)} {_fmt(y0)} c",
            "h",
        ]
        return ("\n".join(parts) + "\n").encode("ascii")

    def _fill_stroke_path(
        self,
        path: bytes,
        fill: tuple[float, float, float] | None,
        stroke: tuple[float, float, float] | None,
        stroke_width: float,
        fill_alpha: float,
        stroke_alpha: float,
    ) -> None:
        if fill:
            self._set_alpha(fill_alpha)
            self._set_color(fill, stroke=False)
            self.content.append(path + b"f\n")
        if stroke:
            self._set_alpha(stroke_alpha)
            self._set_color(stroke, stroke=True)
            self.content.append(f"{_fmt(_len(stroke_width))} w\n".encode("ascii"))
            self.content.append(path + b"S\n")
        if fill_alpha < 0.999 or stroke_alpha < 0.999:
            self._set_alpha(1.0)

    def draw_rect(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        x, y = _f(attrs.get("x")), _f(attrs.get("y"))
        w, h = _f(attrs.get("width")), _f(attrs.get("height"))
        sx = _scale(m)
        x, y = _apply(m, x, y)
        w *= sx
        h *= sx
        rx = _f(attrs.get("rx")) * sx
        fill = _rgb(attrs.get("fill"))
        stroke = _rgb(attrs.get("stroke"))
        stroke_w = _f(attrs.get("stroke-width") or 1) * sx
        self._fill_stroke_path(
            self._path_rect(x, y, w, h, rx),
            fill,
            stroke,
            stroke_w,
            _f(attrs.get("fill-opacity") or 1),
            _f(attrs.get("stroke-opacity") or 1),
        )

    def draw_ellipse(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        cx, cy = _apply(m, _f(attrs.get("cx")), _f(attrs.get("cy")))
        rx = _f(attrs.get("rx")) * _scale(m)
        ry = _f(attrs.get("ry")) * _scale(m)
        px, py = _pt(self.page_h_px, cx, cy)
        rxp, ryp = _len(rx), _len(ry)
        k = 0.5522847498
        parts = [
            f"{_fmt(px + rxp)} {_fmt(py)} m",
            f"{_fmt(px + rxp)} {_fmt(py + k*ryp)} {_fmt(px + k*rxp)} {_fmt(py + ryp)} {_fmt(px)} {_fmt(py + ryp)} c",
            f"{_fmt(px - k*rxp)} {_fmt(py + ryp)} {_fmt(px - rxp)} {_fmt(py + k*ryp)} {_fmt(px - rxp)} {_fmt(py)} c",
            f"{_fmt(px - rxp)} {_fmt(py - k*ryp)} {_fmt(px - k*rxp)} {_fmt(py - ryp)} {_fmt(px)} {_fmt(py - ryp)} c",
            f"{_fmt(px + k*rxp)} {_fmt(py - ryp)} {_fmt(px + rxp)} {_fmt(py - k*ryp)} {_fmt(px + rxp)} {_fmt(py)} c",
            "h",
        ]
        path = ("\n".join(parts) + "\n").encode("ascii")
        self._fill_stroke_path(
            path,
            _rgb(attrs.get("fill")),
            _rgb(attrs.get("stroke")),
            _f(attrs.get("stroke-width") or 1) * _scale(m),
            _f(attrs.get("fill-opacity") or 1),
            _f(attrs.get("stroke-opacity") or 1),
        )

    def _parse_points(self, raw: str, m: tuple[float, float, float, float, float, float]) -> list[tuple[float, float]]:
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\\.?[0-9]+", raw)]
        pts = []
        for i in range(0, len(nums) - 1, 2):
            pts.append(_apply(m, nums[i], nums[i + 1]))
        return pts

    def draw_poly(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float], close: bool) -> None:
        pts = self._parse_points(attrs.get("points", ""), m)
        if len(pts) < 2:
            return
        commands = []
        for idx, (x, y) in enumerate(pts):
            px, py = _pt(self.page_h_px, x, y)
            commands.append(f"{_fmt(px)} {_fmt(py)} {'m' if idx == 0 else 'l'}")
        if close:
            commands.append("h")
        path = ("\n".join(commands) + "\n").encode("ascii")
        self._fill_stroke_path(
            path,
            _rgb(attrs.get("fill")) if close else None,
            _rgb(attrs.get("stroke")) or (_rgb(attrs.get("fill")) if not close else None),
            _f(attrs.get("stroke-width") or 1) * _scale(m),
            _f(attrs.get("fill-opacity") or 1),
            _f(attrs.get("stroke-opacity") or attrs.get("fill-opacity") or 1),
        )

    def draw_path_arc(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\\.?[0-9]+", attrs.get("d", ""))]
        if len(nums) < 9:
            return
        sx, sy, rx, ry, _rot, large, sweep, ex, ey = nums[:9]
        # Sample the SVG elliptical arc into editable PDF line segments.
        points = _sample_arc(sx, sy, rx, ry, bool(int(large)), bool(int(sweep)), ex, ey, 36)
        transformed = [_apply(m, x, y) for x, y in points]
        commands = []
        for idx, (x, y) in enumerate(transformed):
            px, py = _pt(self.page_h_px, x, y)
            commands.append(f"{_fmt(px)} {_fmt(py)} {'m' if idx == 0 else 'l'}")
        stroke = _rgb(attrs.get("stroke")) or (0.0, 0.0, 0.0)
        self._set_alpha(_f(attrs.get("stroke-opacity") or 1))
        self._set_color(stroke, stroke=True)
        self.content.append(f"{_fmt(_len(_f(attrs.get('stroke-width') or 1) * _scale(m)))} w\n".encode("ascii"))
        self.content.append(("\n".join(commands) + "\nS\n").encode("ascii"))

    def draw_text(self, element: ET.Element, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        text = "".join(element.itertext())
        x, y = _apply(m, _f(attrs.get("x")), _f(attrs.get("y")))
        font_size = _f(attrs.get("font-size") or 12) * _scale(m)
        color = _rgb(attrs.get("fill")) or (0.0, 0.0, 0.0)
        px, py = _pt(self.page_h_px, x, y + font_size * 0.86)
        self._set_alpha(_f(attrs.get("fill-opacity") or 1))
        self._set_color(color, stroke=False)
        font = "F2" if attrs.get("font-weight") in {"700", "bold", "Bold"} else "F1"
        self.content.append(
            b"BT\n"
            + f"/{font} {_fmt(_len(font_size))} Tf\n".encode("ascii")
            + f"{_fmt(px)} {_fmt(py)} Td\n".encode("ascii")
            + _escape_text(text)
            + b" Tj\nET\n"
        )

    def draw_image(self, attrs: dict[str, str], m: tuple[float, float, float, float, float, float]) -> None:
        href = attrs.get("href") or attrs.get("{http://www.w3.org/1999/xlink}href") or attrs.get("xlink:href")
        if not href or not href.startswith("data:image"):
            return
        x, y = _apply(m, _f(attrs.get("x")), _f(attrs.get("y")))
        w = _f(attrs.get("width")) * _scale(m)
        h = _f(attrs.get("height")) * _scale(m)
        if self.skip_large_images and w >= self.page_w_px * 0.82 and h >= self.page_h_px * 0.82:
            return
        name, _, _ = self.pdf.image(href)
        px, py = _pt(self.page_h_px, x, y + h)
        self.content.append(b"q\n")
        self.content.append(f"{_fmt(_len(w))} 0 0 {_fmt(_len(h))} {_fmt(px)} {_fmt(py)} cm\n/{name} Do\nQ\n".encode("ascii"))


def _sample_arc(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    large_arc: bool,
    sweep: bool,
    x2: float,
    y2: float,
    steps: int,
) -> list[tuple[float, float]]:
    if rx == 0 or ry == 0:
        return [(x1, y1), (x2, y2)]
    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2
    rx = abs(rx)
    ry = abs(ry)
    lam = dx * dx / (rx * rx) + dy * dy / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    sign = -1 if large_arc == sweep else 1
    num = max(0.0, rx * rx * ry * ry - rx * rx * dy * dy - ry * ry * dx * dx)
    den = max(1e-9, rx * rx * dy * dy + ry * ry * dx * dx)
    coef = sign * math.sqrt(num / den)
    cxp = coef * rx * dy / ry
    cyp = coef * -ry * dx / rx
    cx = cxp + (x1 + x2) / 2
    cy = cyp + (y1 + y2) / 2

    def angle(u: tuple[float, float], v: tuple[float, float]) -> float:
        dot = u[0] * v[0] + u[1] * v[1]
        det = u[0] * v[1] - u[1] * v[0]
        return math.atan2(det, dot)

    v1 = ((dx - cxp) / rx, (dy - cyp) / ry)
    v2 = ((-dx - cxp) / rx, (-dy - cyp) / ry)
    theta1 = angle((1, 0), v1)
    delta = angle(v1, v2)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    if sweep and delta < 0:
        delta += 2 * math.pi
    n = max(4, int(abs(delta) / (2 * math.pi) * steps))
    return [(cx + rx * math.cos(theta1 + delta * i / n), cy + ry * math.sin(theta1 + delta * i / n)) for i in range(n + 1)]


def _render_element(
    renderer: Renderer,
    element: ET.Element,
    matrix: tuple[float, float, float, float, float, float],
    pass_name: str = "all",
) -> None:
    if _hidden(element):
        return
    attrs = _style(element)
    matrix = _matmul(matrix, _parse_transform(attrs.get("transform")))
    tag = _clean_tag(element.tag)
    if tag == "g":
        for child in list(element):
            _render_element(renderer, child, matrix, pass_name=pass_name)
    elif pass_name == "images" and tag != "image":
        return
    elif pass_name == "vectors" and tag == "image":
        return
    elif tag == "rect":
        renderer.draw_rect(attrs, matrix)
    elif tag == "ellipse":
        renderer.draw_ellipse(attrs, matrix)
    elif tag == "polyline":
        renderer.draw_poly(attrs, matrix, close=False)
    elif tag == "polygon":
        renderer.draw_poly(attrs, matrix, close=True)
    elif tag == "path":
        renderer.draw_path_arc(attrs, matrix)
    elif tag == "text":
        renderer.draw_text(element, attrs, matrix)
    elif tag == "image":
        renderer.draw_image(attrs, matrix)


def convert(svg_path: Path, pdf_path: Path, *, edit_priority: bool = False) -> dict[str, Any]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    view_box = root.attrib.get("viewBox", "")
    if view_box:
        _, _, width, height = [float(x) for x in view_box.split()]
    else:
        width = _f(root.attrib.get("width"))
        height = _f(root.attrib.get("height"))
    pdf = PDF()
    renderer = Renderer(
        pdf=pdf,
        page_h_px=height,
        page_w_px=width,
        content=[],
        skip_large_images=edit_priority,
    )
    if edit_priority:
        # Small raster panels first, then editable text/vector content on top.
        # This avoids full-page raster composites blocking selection.
        for child in list(root):
            _render_element(renderer, child, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), pass_name="images")
        for child in list(root):
            _render_element(renderer, child, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), pass_name="vectors")
    else:
        for child in list(root):
            _render_element(renderer, child, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    content = b"".join(renderer.content)
    pdf.write(pdf_path, _len(width), _len(height), content)
    return {
        "asset": svg_path.stem,
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "width_px": int(width),
        "height_px": int(height),
        "pdf_bytes": pdf_path.stat().st_size,
        "images": len(pdf.image_cache),
        "alpha_states": len(pdf.alpha_cache),
    }


def main(argv: list[str] | None = None) -> None:
    edit_priority = bool(argv and "--edit-priority" in argv)
    output_dir = PDF_EDIT_PRIORITY_DIR if edit_priority else PDF_DIR
    source_svg_dir = SVG_EDIT_PRIORITY_DIR if edit_priority and SVG_EDIT_PRIORITY_DIR.exists() else SVG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(source_svg_dir.glob("*.svg"))
    rows = []
    for svg_path in svg_paths:
        pdf_path = output_dir / f"{svg_path.stem}.pdf"
        print(f"converting {svg_path.name} -> {pdf_path.name}", flush=True)
        rows.append(convert(svg_path, pdf_path, edit_priority=edit_priority))
    manifest_csv = output_dir / "editable_pdf_rebuild_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Editable PDF Rebuild Exports",
        "",
        f"These PDFs are generated from `{source_svg_dir.name}/` without page rasterization.",
        "",
        "- Text is written as editable PDF text using Helvetica/Helvetica-Bold.",
        "- Lines, rectangles, ellipses, polygons, and arcs are PDF vector operators.",
        "- Real H&E/photographic/heatmap portions are embedded as image XObjects.",
        "- For exact pixel appearance without object editing, use `../vector_exports_exact/`.",
    ]
    if edit_priority:
        lines.extend(
            [
                "- This edit-priority variant skips full-page raster composites that can block selection in PDF editors.",
                "- Small image panels are kept as separate image objects underneath editable text/vector layers.",
            ]
        )
    lines.extend(
        [
            "",
            "| Asset | Size | Images | PDF |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(f"| {row['asset']} | {row['width_px']}x{row['height_px']} | {row['images']} | `{Path(row['pdf']).name}` |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} editable PDFs to {output_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
