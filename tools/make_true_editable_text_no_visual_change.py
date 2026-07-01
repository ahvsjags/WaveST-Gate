from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
import re
import sys
import zlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "results" / "nature_manuscript_figures"
TEXT_SOURCE_DIR = FIG_DIR / "editable_svg_edit_priority"
OUT_DIR = FIG_DIR / "TRUE_EDITABLE_TEXT_NO_VISUAL_CHANGE"
BACKGROUND_DIR = OUT_DIR / "background_no_text_png"
SVG_DIR = OUT_DIR / "svg"
PDF_DIR = OUT_DIR / "pdf"
QA_DIR = OUT_DIR / "_qa"
TARGET_STEMS = {
    "figure_1_workflow_schematic",
    "figure_2_spatial_cell_composition",
    "figure_3_baseline_performance",
    "figure_4_reliability_calibration",
    "figure_5_boundary_niche_pathology",
}
DPI = 600
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


@dataclass
class TextEditReport:
    figure: str
    action: str
    label: str
    detail: str


TEXT_EDIT_REPORTS: list[TextEditReport] = []


class RecorderState:
    def __init__(self) -> None:
        self.meta: dict[int, dict[str, Any]] = {}
        self.saved: list[dict[str, Any]] = []
        self.saved_meta: dict[str, dict[str, Any]] = {}
        self.original = {
            "Image.new": Image.new,
            "Image.open": Image.open,
            "ImageDraw.Draw": ImageDraw.Draw,
            "Image.Image.save": Image.Image.save,
            "Image.Image.copy": Image.Image.copy,
            "Image.Image.convert": Image.Image.convert,
            "Image.Image.resize": Image.Image.resize,
            "Image.Image.thumbnail": Image.Image.thumbnail,
            "Image.Image.crop": Image.Image.crop,
            "Image.Image.rotate": Image.Image.rotate,
            "Image.Image.filter": Image.Image.filter,
            "Image.Image.putalpha": Image.Image.putalpha,
            "Image.Image.alpha_composite": Image.Image.alpha_composite,
            "Image.Image.paste": Image.Image.paste,
            "ImageEnhance._Enhance.enhance": ImageEnhance._Enhance.enhance,
        }

    def get(self, image: Image.Image, *, default_raster: bool = False) -> dict[str, Any]:
        key = id(image)
        if key not in self.meta:
            self.meta[key] = {
                "width": image.width,
                "height": image.height,
                "elements": [],
                "text_elements": [],
                "raster": default_raster,
            }
        else:
            self.meta[key]["width"] = image.width
            self.meta[key]["height"] = image.height
        return self.meta[key]

    def set_meta(
        self,
        image: Image.Image,
        elements: list[str] | None = None,
        text_elements: list[str] | None = None,
        *,
        raster: bool = False,
    ) -> dict[str, Any]:
        meta = {
            "width": image.width,
            "height": image.height,
            "elements": elements or [],
            "text_elements": text_elements or [],
            "raster": raster,
        }
        self.meta[id(image)] = meta
        return meta


STATE = RecorderState()


def num(value: Any) -> float:
    return float(value)


def fmt(value: Any) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def color(value: Any) -> tuple[str | None, float]:
    if value is None:
        return None, 1.0
    if isinstance(value, str):
        return value, 1.0
    if isinstance(value, int):
        value = (value, value, value)
    if isinstance(value, (tuple, list)):
        if not value:
            return None, 1.0
        rgb = tuple(max(0, min(255, int(v))) for v in value[:3])
        opacity = 1.0
        if len(value) >= 4:
            opacity = max(0.0, min(1.0, float(value[3]) / 255.0))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", opacity
    return str(value), 1.0


def style(fill: Any = None, outline: Any = None, width: Any = None) -> str:
    attrs: list[str] = []
    fill_color, fill_opacity = color(fill)
    if fill_color is None:
        attrs.append('fill="none"')
    else:
        attrs.append(f'fill="{fill_color}"')
        if fill_opacity < 1:
            attrs.append(f'fill-opacity="{fmt(fill_opacity)}"')
    stroke_color, stroke_opacity = color(outline)
    if stroke_color is None:
        attrs.append('stroke="none"')
    else:
        attrs.append(f'stroke="{stroke_color}"')
        if stroke_opacity < 1:
            attrs.append(f'stroke-opacity="{fmt(stroke_opacity)}"')
        attrs.append(f'stroke-width="{fmt(width or 1)}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')
    return " ".join(attrs)


def points(values: Any) -> list[tuple[float, float]]:
    if isinstance(values, tuple) and len(values) == 4 and all(isinstance(v, (int, float)) for v in values):
        return [(num(values[0]), num(values[1])), (num(values[2]), num(values[3]))]
    out: list[tuple[float, float]] = []
    if isinstance(values, (list, tuple)):
        if values and all(isinstance(v, (int, float)) for v in values):
            flat = list(values)
            out = [(num(flat[i]), num(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
        else:
            for point in values:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    out.append((num(point[0]), num(point[1])))
    return out


def transform(snippet: str, op: str) -> str:
    return f'<g transform="{op}">{snippet}</g>' if snippet else snippet


def clean_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text_label(element: ET.Element) -> str:
    return " ".join(html.unescape("".join(element.itertext())).split())


def font_size(element: ET.Element) -> float:
    return float(str(element.attrib.get("font-size", "12")).replace("px", ""))


def font_weight(element: ET.Element) -> str:
    return str(element.attrib.get("font-weight", "400"))


def find_font_path(*names: str) -> Path | None:
    roots = [
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
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            matches = list(root.rglob(name))
            if matches:
                return matches[0]
    return None


REGULAR_FONT_PATH = find_font_path("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf")
BOLD_FONT_PATH = find_font_path("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf") or REGULAR_FONT_PATH
FONT_CACHE: dict[tuple[int, str], ImageFont.ImageFont] = {}


def pil_font(size: float, weight: str = "400") -> ImageFont.ImageFont:
    key = (max(1, int(round(size))), str(weight))
    if key not in FONT_CACHE:
        path = BOLD_FONT_PATH if str(weight) in {"600", "700", "800", "bold"} else REGULAR_FONT_PATH
        if path and path.exists():
            FONT_CACHE[key] = ImageFont.truetype(str(path), size=key[0])
        else:
            FONT_CACHE[key] = ImageFont.load_default()
    return FONT_CACHE[key]


def text_bbox(element: ET.Element) -> tuple[float, float, float, float]:
    label = text_label(element)
    size = font_size(element)
    font = pil_font(size, font_weight(element))
    bbox = font.getbbox(label)
    x = float(element.attrib.get("x", "0"))
    y = float(element.attrib.get("y", "0"))
    return (x + bbox[0], y + bbox[1], x + bbox[2], y + max(bbox[3], size * 1.05))


def set_attr_num(element: ET.Element, key: str, value: float) -> None:
    element.set(key, fmt(value))


def element_visibility_score(element: ET.Element, diff: Image.Image) -> float:
    label = text_label(element)
    if not label:
        return 0.0
    x0, y0, x1, y1 = text_bbox(element)
    pad = max(4, int(round(font_size(element) * 0.35)))
    x0i = max(0, int(np.floor(x0)) - pad)
    y0i = max(0, int(np.floor(y0)) - pad)
    x1i = min(diff.width, int(np.ceil(x1)) + pad)
    y1i = min(diff.height, int(np.ceil(y1)) + pad)
    if x1i <= x0i or y1i <= y0i:
        return 0.0
    mask = Image.new("L", (x1i - x0i, y1i - y0i), 0)
    draw = STATE.original["ImageDraw.Draw"](mask)
    draw.text(
        (float(element.attrib.get("x", "0")) - x0i, float(element.attrib.get("y", "0")) - y0i),
        label,
        font=pil_font(font_size(element), font_weight(element)),
        fill=255,
    )
    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    if float(mask_arr.sum()) < 1:
        return 0.0
    crop = np.asarray(diff.crop((x0i, y0i, x1i, y1i)), dtype=np.float32)
    weighted_delta = float((crop * mask_arr).sum() / mask_arr.sum())
    coverage = float(((crop > 10) & (mask_arr > 0.05)).sum() / max(1, int((mask_arr > 0.05).sum())))
    return weighted_delta * coverage


def render_improvement_score(stem: str, element: ET.Element, approved: Image.Image, background: Image.Image) -> float:
    label = html.unescape("".join(element.itertext()))
    if not label:
        return 0.0
    x0, y0, x1, y1 = text_bbox(element)
    pad = max(8, int(round(font_size(element) * 0.7)))
    x0i = max(0, int(np.floor(x0)) - pad)
    y0i = max(0, int(np.floor(y0)) - pad)
    x1i = min(background.width, int(np.ceil(x1)) + pad)
    y1i = min(background.height, int(np.ceil(y1)) + pad)
    if x1i <= x0i or y1i <= y0i:
        return 0.0
    approved_crop = np.asarray(approved.crop((x0i, y0i, x1i, y1i)), dtype=np.float32)
    before_image = background.crop((x0i, y0i, x1i, y1i)).convert("RGBA")
    before = np.asarray(before_image.convert("RGB"), dtype=np.float32)
    overlay = Image.new("RGBA", before_image.size, (0, 0, 0, 0))
    draw = STATE.original["ImageDraw.Draw"](overlay)
    fill, opacity = color(element.attrib.get("fill"))
    r, g, b = rgb_tuple(fill)
    alpha = int(max(0.0, min(1.0, float(element.attrib.get("fill-opacity", "1")) * opacity)) * 255)
    draw.text(
        (float(element.attrib.get("x", "0")) - x0i, float(element.attrib.get("y", "0")) - y0i),
        label,
        font=pil_font(font_size(element), font_weight(element)),
        fill=(r, g, b, alpha),
    )
    after_image = before_image.copy()
    after_image.alpha_composite(overlay)
    after = np.asarray(after_image.convert("RGB"), dtype=np.float32)
    before_mse = float(((before - approved_crop) ** 2).mean())
    after_mse = float(((after - approved_crop) ** 2).mean())
    return before_mse - after_mse


def rgb_tuple(value: str | None) -> tuple[int, int, int]:
    if not value or value == "none":
        return (0, 0, 0)
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    if value.startswith("rgb"):
        nums = [int(float(x)) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", value)[:3]]
        if len(nums) == 3:
            return tuple(max(0, min(255, n)) for n in nums)  # type: ignore[return-value]
    return (0, 0, 0)


def remove_element(root: ET.Element, target: ET.Element) -> bool:
    for parent in root.iter():
        children = list(parent)
        if target in children:
            parent.remove(target)
            return True
    return False


def append_report(figure: str, action: str, element_or_label: ET.Element | str, detail: str) -> None:
    label = text_label(element_or_label) if isinstance(element_or_label, ET.Element) else str(element_or_label)
    TEXT_EDIT_REPORTS.append(TextEditReport(figure=figure, action=action, label=label, detail=detail))


def image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    STATE.original["Image.Image.save"](image.convert("RGBA"), buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def embed_image(image: Image.Image, x: float, y: float, width: float, height: float) -> str:
    href = image_data_uri(image)
    return (
        f'<image x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}" '
        f'preserveAspectRatio="none" href="{href}" xlink:href="{href}"/>'
    )


class RecordingDraw:
    def __init__(self, image: Image.Image, mode: str | None = None) -> None:
        self.image = image
        self._draw = STATE.original["ImageDraw.Draw"](image, mode) if mode is not None else STATE.original["ImageDraw.Draw"](image)
        STATE.get(image)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._draw, name)

    def textbbox(self, *args: Any, **kwargs: Any) -> Any:
        return self._draw.textbbox(*args, **kwargs)

    def textlength(self, *args: Any, **kwargs: Any) -> Any:
        return self._draw.textlength(*args, **kwargs)

    def rounded_rectangle(self, xy: Any, radius: int = 0, fill: Any = None, outline: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width, **kwargs)
        x0, y0, x1, y1 = map(num, xy)
        STATE.get(self.image)["elements"].append(
            f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" '
            f'rx="{fmt(radius)}" ry="{fmt(radius)}" {style(fill, outline, width)}/>'
        )

    def rectangle(self, xy: Any, fill: Any = None, outline: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.rectangle(xy, fill=fill, outline=outline, width=width, **kwargs)
        x0, y0, x1, y1 = map(num, xy)
        STATE.get(self.image)["elements"].append(
            f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" {style(fill, outline, width)}/>'
        )

    def ellipse(self, xy: Any, fill: Any = None, outline: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.ellipse(xy, fill=fill, outline=outline, width=width, **kwargs)
        x0, y0, x1, y1 = map(num, xy)
        STATE.get(self.image)["elements"].append(
            f'<ellipse cx="{fmt((x0 + x1) / 2)}" cy="{fmt((y0 + y1) / 2)}" '
            f'rx="{fmt((x1 - x0) / 2)}" ry="{fmt((y1 - y0) / 2)}" {style(fill, outline, width)}/>'
        )

    def line(self, xy: Any, fill: Any = None, width: int = 1, joint: Any = None, **kwargs: Any) -> None:
        self._draw.line(xy, fill=fill, width=width, joint=joint, **kwargs)
        pts = points(xy)
        if len(pts) < 2:
            return
        stroke, opacity = color(fill)
        attrs = [
            'fill="none"',
            f'stroke="{stroke or "#000000"}"',
            f'stroke-width="{fmt(width)}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if opacity < 1:
            attrs.append(f'stroke-opacity="{fmt(opacity)}"')
        d = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)
        STATE.get(self.image)["elements"].append(f'<polyline points="{d}" {" ".join(attrs)}/>')

    def polygon(self, xy: Any, fill: Any = None, outline: Any = None, **kwargs: Any) -> None:
        self._draw.polygon(xy, fill=fill, outline=outline, **kwargs)
        pts = points(xy)
        if pts:
            d = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)
            STATE.get(self.image)["elements"].append(f'<polygon points="{d}" {style(fill, outline, 1)}/>')

    def arc(self, xy: Any, start: float, end: float, fill: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.arc(xy, start=start, end=end, fill=fill, width=width, **kwargs)
        x0, y0, x1, y1 = map(num, xy)
        stroke, opacity = color(fill)
        attrs = [f'fill="none"', f'stroke="{stroke or "#000000"}"', f'stroke-width="{fmt(width)}"']
        if opacity < 1:
            attrs.append(f'stroke-opacity="{fmt(opacity)}"')
        STATE.get(self.image)["elements"].append(
            f'<path d="M {fmt(x0)} {fmt((y0 + y1) / 2)} A {fmt((x1 - x0) / 2)} {fmt((y1 - y0) / 2)} 0 0 1 {fmt(x1)} {fmt((y0 + y1) / 2)}" {" ".join(attrs)}/>'
        )

    def text(self, xy: Any, text: Any, fill: Any = None, font: Any = None, anchor: Any = None, **kwargs: Any) -> None:
        # Intentionally do not draw text to the background PNG. Record it as editable SVG/PDF text.
        x, y = xy
        size = getattr(font, "size", 12) if font is not None else 12
        fill_color, opacity = color(fill)
        attrs = [
            f'x="{fmt(x)}"',
            f'y="{fmt(y)}"',
            f'fill="{fill_color or "#000000"}"',
            'font-family="DejaVu Sans"',
            f'font-size="{fmt(size)}px"',
            f'font-weight="{"700" if "Bold" in str(getattr(font, "path", "")) else "400"}"',
            'dominant-baseline="hanging"',
        ]
        if opacity < 1:
            attrs.append(f'fill-opacity="{fmt(opacity)}"')
        if anchor:
            attrs.append(f'text-anchor="{html.escape(str(anchor))}"')
        safe = html.escape(str(text))
        STATE.get(self.image)["text_elements"].append(f'<text {" ".join(attrs)}>{safe}</text>')


def copy_elements(src: Image.Image, dst: Image.Image, op: str | None = None) -> None:
    src_meta = STATE.get(src, default_raster=True)
    dst_meta = STATE.get(dst)
    for snippet in src_meta["elements"]:
        dst_meta["elements"].append(transform(snippet, op) if op else snippet)
    for snippet in src_meta["text_elements"]:
        dst_meta["text_elements"].append(transform(snippet, op) if op else snippet)


def install_recorder() -> None:
    def image_new(mode: str, size: Any, color_value: Any = 0) -> Image.Image:
        image = STATE.original["Image.new"](mode, size, color_value)
        meta = STATE.set_meta(image)
        fill, opacity = color(color_value)
        if fill is not None and opacity > 0 and image.mode not in {"L", "1"}:
            attrs = f'fill="{fill}"'
            if opacity < 1:
                attrs += f' fill-opacity="{fmt(opacity)}"'
            meta["elements"].append(f'<rect x="0" y="0" width="{image.width}" height="{image.height}" {attrs} stroke="none"/>')
        return image

    def image_open(fp: Any, *args: Any, **kwargs: Any) -> Image.Image:
        image = STATE.original["Image.open"](fp, *args, **kwargs)
        STATE.set_meta(image, raster=True)
        return image

    def draw_factory(image: Image.Image, mode: str | None = None) -> RecordingDraw:
        return RecordingDraw(image, mode)

    def copy_method(self: Image.Image) -> Image.Image:
        out = STATE.original["Image.Image.copy"](self)
        src = STATE.get(self, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), deepcopy(src["text_elements"]), raster=bool(src.get("raster")))
        return out

    def convert_method(self: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.convert"](self, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), deepcopy(src["text_elements"]), raster=bool(src.get("raster")))
        return out

    def resize_method(self: Image.Image, size: Any, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.resize"](self, size, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        sx = out.width / max(1, self.width)
        sy = out.height / max(1, self.height)
        elements = [transform(snippet, f"scale({fmt(sx)} {fmt(sy)})") for snippet in src["elements"]]
        texts = [transform(snippet, f"scale({fmt(sx)} {fmt(sy)})") for snippet in src["text_elements"]]
        STATE.set_meta(out, elements, texts, raster=bool(src.get("raster")))
        return out

    def thumbnail_method(self: Image.Image, size: Any, *args: Any, **kwargs: Any) -> None:
        old_w, old_h = self.width, self.height
        result = STATE.original["Image.Image.thumbnail"](self, size, *args, **kwargs)
        meta = STATE.get(self, default_raster=True)
        sx = self.width / max(1, old_w)
        sy = self.height / max(1, old_h)
        meta["elements"] = [transform(snippet, f"scale({fmt(sx)} {fmt(sy)})") for snippet in meta["elements"]]
        meta["text_elements"] = [transform(snippet, f"scale({fmt(sx)} {fmt(sy)})") for snippet in meta["text_elements"]]
        return result

    def crop_method(self: Image.Image, box: Any = None) -> Image.Image:
        out = STATE.original["Image.Image.crop"](self, box)
        if box:
            x0, y0 = num(box[0]), num(box[1])
            src = STATE.get(self, default_raster=True)
            elements = [transform(snippet, f"translate({fmt(-x0)} {fmt(-y0)})") for snippet in src["elements"]]
            texts = [transform(snippet, f"translate({fmt(-x0)} {fmt(-y0)})") for snippet in src["text_elements"]]
            STATE.set_meta(out, elements, texts, raster=bool(src.get("raster")))
        return out

    def rotate_method(self: Image.Image, angle: Any, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.rotate"](self, angle, *args, **kwargs)
        STATE.set_meta(out, [embed_image(out, 0, 0, out.width, out.height)], [], raster=True)
        return out

    def filter_method(self: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.filter"](self, *args, **kwargs)
        STATE.set_meta(out, [embed_image(out, 0, 0, out.width, out.height)], [], raster=True)
        return out

    def putalpha_method(self: Image.Image, alpha: Any) -> None:
        result = STATE.original["Image.Image.putalpha"](self, alpha)
        STATE.get(self)["raster"] = True
        return result

    def enhance_method(self: Any, factor: float) -> Image.Image:
        out = STATE.original["ImageEnhance._Enhance.enhance"](self, factor)
        src = STATE.get(self.image, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), deepcopy(src["text_elements"]), raster=bool(src.get("raster")))
        return out

    def alpha_composite_method(self: Image.Image, im: Image.Image, dest: Any = (0, 0), source: Any = (0, 0)) -> None:
        result = STATE.original["Image.Image.alpha_composite"](self, im, dest, source)
        dst = STATE.get(self)
        src = STATE.get(im, default_raster=True)
        dx, dy = (dest if isinstance(dest, tuple) else (0, 0))
        if src.get("raster") or not src["elements"]:
            dst["elements"].append(embed_image(im, dx, dy, im.width, im.height))
        for snippet in src["elements"]:
            dst["elements"].append(transform(snippet, f"translate({fmt(dx)} {fmt(dy)})"))
        for snippet in src["text_elements"]:
            dst["text_elements"].append(transform(snippet, f"translate({fmt(dx)} {fmt(dy)})"))
        return result

    def paste_method(self: Image.Image, im: Any, box: Any = None, mask: Any = None) -> None:
        result = STATE.original["Image.Image.paste"](self, im, box, mask)
        dst = STATE.get(self)
        if isinstance(box, tuple):
            if len(box) == 2:
                dx, dy = num(box[0]), num(box[1])
                dw = im.width if isinstance(im, Image.Image) else 0
                dh = im.height if isinstance(im, Image.Image) else 0
            else:
                dx, dy = num(box[0]), num(box[1])
                dw, dh = num(box[2]) - dx, num(box[3]) - dy
        else:
            dx = dy = 0
            dw = im.width if isinstance(im, Image.Image) else self.width
            dh = im.height if isinstance(im, Image.Image) else self.height
        if isinstance(im, Image.Image):
            src = STATE.get(im, default_raster=True)
            if src.get("raster") or not src["elements"]:
                if mask is not None:
                    masked = STATE.original["Image.new"]("RGBA", (im.width, im.height), (0, 0, 0, 0))
                    STATE.original["Image.Image.paste"](masked, im.convert("RGBA"), (0, 0), mask)
                    dst["elements"].append(embed_image(masked, dx, dy, dw, dh))
                else:
                    dst["elements"].append(embed_image(im, dx, dy, dw, dh))
            for snippet in src["elements"]:
                dst["elements"].append(transform(snippet, f"translate({fmt(dx)} {fmt(dy)})"))
            for snippet in src["text_elements"]:
                dst["text_elements"].append(transform(snippet, f"translate({fmt(dx)} {fmt(dy)})"))
        elif isinstance(box, tuple) and len(box) == 4:
            fill, opacity = color(im)
            if fill is not None and opacity > 0:
                attrs = f'fill="{fill}"'
                if opacity < 1:
                    attrs += f' fill-opacity="{fmt(opacity)}"'
                dst["elements"].append(f'<rect x="{fmt(dx)}" y="{fmt(dy)}" width="{fmt(dw)}" height="{fmt(dh)}" {attrs} stroke="none"/>')
        return result

    def save_method(self: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> Any:
        path = Path(fp) if isinstance(fp, (str, Path)) else None
        if path is not None and path.suffix.lower() == ".png" and path.stem in TARGET_STEMS:
            path = BACKGROUND_DIR / path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            result = STATE.original["Image.Image.save"](self, path, *args, **kwargs)
            STATE.saved.append({"stem": path.stem, "png": str(path)})
            STATE.saved_meta[path.stem] = deepcopy(STATE.get(self))
            return result
        return STATE.original["Image.Image.save"](self, fp, *args, **kwargs)

    Image.new = image_new  # type: ignore[assignment]
    Image.open = image_open  # type: ignore[assignment]
    ImageDraw.Draw = draw_factory  # type: ignore[assignment]
    Image.Image.save = save_method  # type: ignore[assignment]
    Image.Image.copy = copy_method  # type: ignore[assignment]
    Image.Image.convert = convert_method  # type: ignore[assignment]
    Image.Image.resize = resize_method  # type: ignore[assignment]
    Image.Image.thumbnail = thumbnail_method  # type: ignore[assignment]
    Image.Image.crop = crop_method  # type: ignore[assignment]
    Image.Image.rotate = rotate_method  # type: ignore[assignment]
    Image.Image.filter = filter_method  # type: ignore[assignment]
    Image.Image.putalpha = putalpha_method  # type: ignore[assignment]
    Image.Image.alpha_composite = alpha_composite_method  # type: ignore[assignment]
    Image.Image.paste = paste_method  # type: ignore[assignment]
    ImageEnhance._Enhance.enhance = enhance_method  # type: ignore[assignment]


def strip_svg_namespace(element: ET.Element) -> None:
    if clean_tag(element.tag) == "text":
        element.tag = "text"
    elif "}" in element.tag:
        element.tag = clean_tag(element.tag)
    for key in list(element.attrib):
        if key.startswith("{"):
            value = element.attrib.pop(key)
            element.attrib[clean_tag(key)] = value
    for child in list(element):
        strip_svg_namespace(child)


def visible_text_elements(stem: str, elements: list[ET.Element]) -> list[ET.Element]:
    approved_path = FIG_DIR / f"{stem}.png"
    background_path = BACKGROUND_DIR / f"{stem}.png"
    if not approved_path.exists() or not background_path.exists():
        return elements
    approved = Image.open(approved_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")
    diff = ImageChops.difference(approved, background).convert("L")
    kept: list[ET.Element] = []
    for element in elements:
        label = text_label(element)
        if not label:
            continue
        score = element_visibility_score(element, diff)
        improvement = render_improvement_score(stem, element, approved, background)
        size = font_size(element)
        # Low-score and harmful text is usually an obsolete draft layer that was later
        # painted over. Keep subtle real labels even when exact font metrics differ.
        if score < 2.0 and improvement < -20.0:
            append_report(stem, "remove-hidden-or-obsolete", element, f"visibility_score={score:.2f}; improvement={improvement:.2f}")
            continue
        if size >= 12 and improvement < -900.0 and score < 30.0:
            append_report(stem, "remove-obsolete-layer", element, f"visibility_score={score:.2f}; improvement={improvement:.2f}")
            continue
        kept.append(element)
    return kept


def remove_confirmed_obsolete_text(stem: str, elements: list[ET.Element]) -> list[ET.Element]:
    kept: list[ET.Element] = []
    for element in elements:
        label = text_label(element)
        x = float(element.attrib.get("x", "0"))
        y = float(element.attrib.get("y", "0"))
        size = font_size(element)
        remove_reason: str | None = None
        if (
            stem == "figure_3_baseline_performance"
            and (
                (label == "Figure 3" and 80 <= x <= 95 and 50 <= y <= 60 and size >= 21)
                or (label == "WaveST-Gate" and 80 <= x <= 95 and 85 <= y <= 95 and 40 <= size <= 48)
                or (label == "#1" and 500 <= x <= 560 and 95 <= y <= 110)
                or (label in {"SCALE", "#1/13"} and 130 <= x <= 285 and 145 <= y <= 165)
            )
        ):
            remove_reason = "obsolete Figure 3 header candidate hidden by final header"
        if (
            stem == "figure_3_baseline_performance"
            and label.startswith("Xenium cells are aggregated into matched Visium spots")
        ):
            remove_reason = "obsolete panel-a title hidden by final Xenium-derived ground-truth title"
        if stem == "figure_3_baseline_performance" and label.startswith("same benchmark logic:"):
            remove_reason = "obsolete heading subtitle hidden by final Rep1 transfer claim"
        if stem == "figure_3_baseline_performance" and label.startswith("Rep1 domain shift is recovered"):
            remove_reason = "obsolete Rep1 title hidden by final transfer-weakness claim"
        if (
            stem == "figure_3_baseline_performance"
            and 300 <= x <= 1250
            and 2570 <= y <= 2625
            and not label.startswith("Rep1 transfer weakness")
            and not label.startswith("explained by module-removal")
        ):
            remove_reason = "obsolete Rep1 overlay text hidden by final transfer-weakness claim"
        if stem == "figure_3_baseline_performance" and label.startswith("Fairness, sensitivity, permutation"):
            remove_reason = "obsolete robustness title hidden by final stress-test claim"
        if (
            stem == "figure_3_baseline_performance"
            and 400 <= x <= 1500
            and 3500 <= y <= 3580
            and not label.startswith("The same benchmark result is stress-tested")
            and not label.startswith("label-null separation")
        ):
            remove_reason = "obsolete robustness overlay text hidden by final stress-test claim"
        if stem == "figure_3_baseline_performance" and 120 <= x <= 720 and 180 <= y <= 200:
            remove_reason = "obsolete route labels hidden by final header"
        if stem == "figure_3_baseline_performance" and 330 <= x <= 400 and 145 <= y <= 160:
            remove_reason = "obsolete scale chip label hidden by final header"
        if stem == "figure_3_baseline_performance" and label == "BENCHMARK":
            remove_reason = "obsolete watermark hidden by final panel foreground"
        if stem == "figure_3_baseline_performance" and 650 <= x <= 740 and 1160 <= y <= 1310:
            remove_reason = "obsolete rank-list labels hidden by final list"
        if (
            stem == "figure_3_baseline_performance"
            and 140 <= x <= 330
            and 1100 <= y <= 1305
            and (
                (label == "#1" and size < 90)
                or (label == "WaveST-Gate" and x < 330)
                or label in {"02", "BayesPrism", "CARD"}
            )
        ):
            remove_reason = "obsolete primary-result card text hidden by final card"
        if (
            stem == "figure_3_baseline_performance"
            and 560 <= x <= 1450
            and 1030 <= y <= 1130
            and label in {"winner lane", "0.01293", "0.238", "18.4x", "**"}
        ):
            remove_reason = "obsolete rank-cliff overlay hidden by final rank panel"
        if (
            stem == "figure_3_baseline_performance"
            and 1430 <= x <= 2050
            and 2570 <= y <= 2620
            and (
                label in {"1", "2", "3", "direct", "small budget", "recovered", "mechanism", "JSD 0.366"}
                or (label in {"25 steps", "0.039", "+0.0087"} and size <= 10)
            )
        ):
            remove_reason = "obsolete transfer metric duplicate hidden by final metric cards"
        if (
            stem == "figure_3_baseline_performance"
            and 130 <= x <= 290
            and 3515 <= y <= 3605
            and (label.startswith("REVIEWER-PROOF") or label in {"audit deck", "validation deck"})
        ):
            remove_reason = "obsolete audit title hidden by final audit-gauntlet title"
        if stem == "figure_3_baseline_performance" and label == "Comprehensive Assessment":
            remove_reason = "obsolete assessment title hidden by final wall title"
        if (
            stem == "figure_3_baseline_performance"
            and label.startswith("A-F evidence enters one locked ranking")
        ):
            remove_reason = "obsolete assessment subtitle hidden by final wall title"
        if (
            stem == "figure_4_reliability_calibration"
            and 900 <= x <= 2400
            and 1200 <= y <= 1405
            and label != "Reliability-state atlas validates calibrated trust over Xenium-supervised tissue space"
        ):
            remove_reason = "obsolete Figure 4 title candidate hidden by final atlas title"
        if (
            stem == "figure_4_reliability_calibration"
            and 70 <= x <= 1200
            and 1320 <= y <= 1410
            and not label.startswith("Xenium-supervised spatial validation field")
            and not label.startswith("spot-level JSD is projected")
        ):
            remove_reason = "obsolete Figure 4 panel-B title candidate hidden by final title"
        if (
            stem == "figure_4_reliability_calibration"
            and 2450 <= x <= 3500
            and 1320 <= y <= 1410
            and not label.startswith("biological trust proof wall")
            and not label.startswith("niche-to-modality flow")
        ):
            remove_reason = "obsolete Figure 4 panel-D title candidate hidden by final title"
        if stem == "figure_4_reliability_calibration" and 2450 <= x <= 3250 and 2550 <= y <= 2665:
            if x < 2620 or label == "H&E local support remains sparse but localized: mean=0.0006, q99=0.0016":
                remove_reason = "obsolete Figure 4 bottom annotation hidden by final inset labels"
        if (
            stem == "figure_4_reliability_calibration"
            and 2100 <= x <= 2260
            and 1430 <= y <= 1470
            and label in {"n=485; r=0.53", "n=485; r=0.53; rho=0.60"}
        ):
            remove_reason = "obsolete short Figure 4 correlation label hidden by final Pearson/Spearman label"
        if (
            stem == "figure_4_reliability_calibration"
            and 95 <= x <= 100
            and 2550 <= y <= 2590
            and label in {"High", "1,664 spots"}
        ):
            remove_reason = "obsolete Figure 4 state label duplicate hidden by final state labels"
        if (
            stem == "figure_4_reliability_calibration"
            and (
                (label == "0.9987" and 3035 <= x <= 3055 and 1545 <= y <= 1560)
                or (label in {"ST", "scRNA"} and 3020 <= x <= 3050 and 1630 <= y <= 1785)
                or (label in {"trusted", "review", "flag"} and 3420 <= x <= 3435 and 1540 <= y <= 1790)
                or (label in {"N0 HER2/tumor-as.", "N1 stromal remod."} and 2555 <= x <= 2575 and 1515 <= y <= 1600)
                or (label == "scRNA" and 3395 <= x <= 3405 and 2595 <= y <= 2610)
                or (label in {"full atlas", "higher review need"} and 3370 <= x <= 3385 and 2510 <= y <= 2590)
            )
        ):
            remove_reason = "obsolete Figure 4 duplicate label hidden by final trust-flow/frontier labels"
        if (
            stem == "figure_4_reliability_calibration"
            and (
                (label == "selective reporting frontier" and 2735 <= x <= 2755 and 2545 <= y <= 2560)
                or (label == "less review" and 2630 <= x <= 2645 and 2575 <= y <= 2590)
                or (
                    label in {"N3 stromal remodelin.", "N4 HER2/tumor-associ."}
                    and 2748 <= x <= 2764
                    and 2600 <= y <= 2630
                )
            )
        ):
            remove_reason = "obsolete Figure 4 frontier/legend duplicate hidden by final frontier labels"
        if remove_reason:
            append_report(stem, "remove-confirmed-obsolete", element, remove_reason)
            continue
        kept.append(element)
    return kept


def fix_known_text_conflicts(stem: str, elements: list[ET.Element]) -> list[ET.Element]:
    for element in elements:
        label = text_label(element)
        x = float(element.attrib.get("x", "0"))
        y = float(element.attrib.get("y", "0"))
        if stem == "figure_1_workflow_schematic" and label == "scRNA prototype agents" and 980 <= x <= 1080 and 740 <= y <= 830:
            old = (x, y)
            set_attr_num(element, "x", 1042)
            set_attr_num(element, "y", 456)
            append_report(stem, "move-conflict-label", element, f"from {old} to (1042, 456)")
        if stem == "figure_1_workflow_schematic" and label == "ST expression" and 990 <= x <= 1060 and 760 <= y <= 820:
            # Keep this one inside the green rounded label box.
            old = (x, y)
            set_attr_num(element, "x", 1032)
            set_attr_num(element, "y", 786)
            element.set("font-size", "13px")
            append_report(stem, "align-label-in-box", element, f"from {old}, 14px to (1032, 786), 13px")
    return elements


def text_elements_from_edit_priority(stem: str) -> list[str]:
    source = TEXT_SOURCE_DIR / f"{stem}.svg"
    root = ET.parse(source).getroot()
    elements: list[ET.Element] = []
    for element in root.iter():
        if clean_tag(element.tag) == "text":
            copied = deepcopy(element)
            strip_svg_namespace(copied)
            elements.append(copied)
    elements = visible_text_elements(stem, elements)
    elements = remove_confirmed_obsolete_text(stem, elements)
    elements = fix_known_text_conflicts(stem, elements)
    return [ET.tostring(element, encoding="unicode", short_empty_elements=False) for element in elements]


def svg_file(stem: str, meta: dict[str, Any], text_elements: list[str]) -> Path:
    width, height = meta["width"], meta["height"]
    bg_href = "data:image/png;base64," + base64.b64encode((BACKGROUND_DIR / f"{stem}.png").read_bytes()).decode("ascii")
    text_body = "\n".join(text_elements)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width / DPI:.6f}in" height="{height / DPI:.6f}in" viewBox="0 0 {width} {height}" version="1.1">
  <title>{html.escape(stem)}</title>
  <desc>No-text regenerated background plus visible editable text. Non-text scientific panels remain unchanged raster/vector objects from the source render.</desc>
  <g id="background_without_text_LOCK_DO_NOT_EDIT">
    <image x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" href="{bg_href}" xlink:href="{bg_href}"/>
  </g>
  <g id="editable_text_LAYER_VISIBLE">
{text_body}
  </g>
</svg>
'''
    path = SVG_DIR / f"{stem}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def len_pt(value: float) -> float:
    return value * 72.0 / DPI


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
    return 0.0, 0.0, 0.0


def parse_attrs(fragment: str) -> dict[str, str]:
    return dict(re.findall(r'([\w:.-]+)="([^"]*)"', fragment))


def strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def iter_text_fragments(text_elements: list[str]) -> list[ET.Element]:
    out: list[ET.Element] = []
    for text_svg in text_elements:
        try:
            element = ET.fromstring(text_svg)
        except ET.ParseError:
            match = re.search(r"<(?:\w+:)?text\s+([^>]*)>(.*?)</(?:\w+:)?text>", text_svg, flags=re.S)
            if not match:
                continue
            element = ET.Element("text", parse_attrs(match.group(1)))
            element.text = html.unescape(strip_tags(match.group(2)))
        if clean_tag(element.tag) == "text":
            out.append(element)
    return out


def write_pdf(stem: str, meta: dict[str, Any], text_elements: list[str]) -> Path:
    bg = Image.open(BACKGROUND_DIR / f"{stem}.png").convert("RGB")
    width, height = bg.size
    objects: list[bytes] = []

    def add(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    bg_payload = zlib.compress(bg.tobytes(), 6)
    bg_id = add(
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(bg_payload)} >>\nstream\n"
        ).encode("ascii")
        + bg_payload
        + b"\nendstream"
    )
    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    gs_ids = {}
    for alpha_i in sorted({1000} | {int(round(float(a) * 1000)) for a in re.findall(r'fill-opacity="([0-9.]+)"', "\n".join(text_elements))}):
        alpha = max(0, min(1000, alpha_i)) / 1000
        gs_ids[alpha_i] = add(f"<< /Type /ExtGState /ca {fmt(alpha)} /CA {fmt(alpha)} >>".encode("ascii"))

    content = bytearray()
    content.extend(b"q\n")
    content.extend(f"{fmt(len_pt(width))} 0 0 {fmt(len_pt(height))} 0 0 cm\n/Im1 Do\nQ\n".encode("ascii"))
    for element in iter_text_fragments(text_elements):
        attrs = element.attrib
        label = html.unescape("".join(element.itertext()))
        x = float(attrs.get("x", "0"))
        y = float(attrs.get("y", "0"))
        size = float(attrs.get("font-size", "12").replace("px", ""))
        fill = attrs.get("fill")
        opacity = int(round(float(attrs.get("fill-opacity", "1")) * 1000))
        r, g, b = rgb(fill)
        font = "F2" if attrs.get("font-weight") in {"600", "700", "800", "bold"} else "F1"
        px = len_pt(x)
        py = len_pt(height - (y + size * 0.86))
        content.extend(
            (
                "q\n"
                f"/GS{opacity} gs\n"
                "BT\n"
                f"{fmt(r)} {fmt(g)} {fmt(b)} rg\n"
                f"/{font} {fmt(len_pt(size))} Tf\n"
                f"{fmt(px)} {fmt(py)} Td\n"
            ).encode("ascii")
            + pdf_escape(label)
            + b" Tj\nET\nQ\n"
        )

    compressed = zlib.compress(bytes(content), 6)
    content_id = add(f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii") + compressed + b"\nendstream")
    extg = " ".join(f"/GS{k} {v} 0 R" for k, v in gs_ids.items())
    resources = (
        f"<< /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> "
        f"/XObject << /Im1 {bg_id} 0 R >> /ExtGState << {extg} >> >>"
    )
    pages_id = len(objects) + 2
    page_id = add(
        (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {fmt(len_pt(width))} {fmt(len_pt(height))}] "
            f"/Resources {resources} /Contents {content_id} 0 R >>"
        ).encode("ascii")
    )
    pages_id = add(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("ascii"))
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(f"trailer\n<< /Size {len(objects)+1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    path = PDF_DIR / f"{stem}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


def write_text_edit_report() -> None:
    rows = sorted(TEXT_EDIT_REPORTS, key=lambda item: (item.figure, item.action, item.label, item.detail))
    csv_path = QA_DIR / "text_layer_edit_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "action", "label", "detail"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"figure": row.figure, "action": row.action, "label": row.label, "detail": row.detail})
    summary: dict[tuple[str, str], int] = {}
    for row in rows:
        summary[(row.figure, row.action)] = summary.get((row.figure, row.action), 0) + 1
    lines = ["# Text Layer Edit Report", "", "Automatic edits applied to the visible editable text layer only.", ""]
    for (figure, action), count in sorted(summary.items()):
        lines.append(f"- {figure}: {action} = {count}")
    lines.extend(["", f"Full row-level report: `{csv_path.name}`", ""])
    (QA_DIR / "TEXT_LAYER_EDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def generate() -> list[dict[str, Any]]:
    for folder in (BACKGROUND_DIR, SVG_DIR, PDF_DIR, QA_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    install_recorder()
    from wavestgate.evaluation import manuscript_figures

    font_dir = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "native" / "poppler" / "Library" / "share" / "fonts"
    dejavu_regular = next(font_dir.rglob("DejaVuSans.ttf"), None) if font_dir.exists() else None
    dejavu_bold = next(font_dir.rglob("DejaVuSans-Bold.ttf"), None) if font_dir.exists() else None
    original_font = manuscript_figures._font

    def local_font(size: int, bold: bool = False) -> Any:
        candidate = dejavu_bold if bold else dejavu_regular
        if candidate and candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
        return original_font(size, bold=bold)

    from PIL import ImageFont

    manuscript_figures._font = local_font
    manuscript_figures.build_manuscript_figures(output_dir=BACKGROUND_DIR)
    rows = []
    for item in STATE.saved:
        stem = item["stem"]
        if stem not in TARGET_STEMS:
            continue
        # Find the image metadata by locating the saved image dimensions and stem in last saved order.
        png = Path(item["png"])
        image_meta = STATE.saved_meta.get(stem)
        if image_meta is None:
            continue
        source_text = text_elements_from_edit_priority(stem)
        svg_path = svg_file(stem, image_meta, source_text)
        pdf_path = write_pdf(stem, image_meta, source_text)
        rows.append(
            {
                "figure": stem,
                "background_no_text_png": str(png.relative_to(OUT_DIR)),
                "editable_svg": str(svg_path.relative_to(OUT_DIR)),
                "editable_pdf": str(pdf_path.relative_to(OUT_DIR)),
                "width_px": image_meta["width"],
                "height_px": image_meta["height"],
                "text_objects": len(source_text),
                "pdf_bytes": pdf_path.stat().st_size,
                "svg_bytes": svg_path.stat().st_size,
            }
        )
    rows = sorted({row["figure"]: row for row in rows}.values(), key=lambda row: row["figure"])
    write_text_edit_report()
    return rows


def main() -> int:
    rows = generate()
    if not rows:
        raise SystemExit("No figure rows generated")
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "README.md").write_text(
        """# TRUE_EDITABLE_TEXT_NO_VISUAL_CHANGE

This is the closest correct version for editing labels without changing the figure design.

- `background_no_text_png/` is regenerated from the same figure code with text drawing suppressed.
- `svg/` places editable SVG text over that no-text background.
- `pdf/` places editable PDF text over that no-text background.
- H&E/heatmap/photo panels remain high-resolution raster, because those scientific image panels cannot be true vector without changing the data image.

Use these files when the goal is changing font/labels while preserving all non-text visual content.
""",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
