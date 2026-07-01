"""Generate source-rebuilt editable SVG exports for manuscript figures.

The existing final manuscript figures are rendered with Pillow.  This helper
reruns that renderer while recording drawing primitives into SVG.  It keeps
photographic and heatmap-like content as embedded image objects, while text,
lines, rectangles, ellipses, polygons, and arcs become editable SVG elements.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path("/mnt/WaveST-Gate")
sys.path.insert(0, str(ROOT))
SOURCE_DIR = ROOT / "results/nature_manuscript_figures"
REBUILD_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_rebuild"
TEMP_PNG_DIR = REBUILD_DIR / "_regenerated_png"

TARGET_STEMS = {
    "editor_first_glance_contact_sheet",
    "figure_1_editorial_graphical_abstract",
    "figure_1_workflow_schematic",
    "figure_2_spatial_cell_composition",
    "figure_3_baseline_performance",
    "figure_4_reliability_calibration",
    "figure_5_boundary_niche_pathology",
    "supplementary_figure_s1_robustness",
}

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
DPI = 600


class RecorderState:
    def __init__(self) -> None:
        self.meta: dict[int, dict[str, Any]] = {}
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
        self.saved: list[dict[str, Any]] = []

    def get(self, image: Image.Image, *, default_raster: bool = False) -> dict[str, Any]:
        key = id(image)
        if key not in self.meta:
            self.meta[key] = {
                "width": image.width,
                "height": image.height,
                "elements": [],
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
        *,
        raster: bool = False,
    ) -> dict[str, Any]:
        meta = {
            "width": image.width,
            "height": image.height,
            "elements": elements or [],
            "raster": raster,
        }
        self.meta[id(image)] = meta
        return meta


STATE = RecorderState()


def _num(value: Any) -> float:
    return float(value)


def _fmt(value: Any) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _color(value: Any) -> tuple[str | None, float]:
    if value is None:
        return None, 1.0
    if isinstance(value, str):
        return value, 1.0
    if isinstance(value, int):
        value = (value, value, value)
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) == 0:
            return None, 1.0
        rgb = tuple(max(0, min(255, int(v))) for v in value[:3])
        opacity = 1.0
        if len(value) >= 4:
            opacity = max(0.0, min(1.0, float(value[3]) / 255.0))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}", opacity
    return str(value), 1.0


def _style(fill: Any = None, outline: Any = None, width: Any = None) -> str:
    attrs: list[str] = []
    fill_color, fill_opacity = _color(fill)
    if fill_color is None:
        attrs.append('fill="none"')
    else:
        attrs.append(f'fill="{fill_color}"')
        if fill_opacity < 1:
            attrs.append(f'fill-opacity="{_fmt(fill_opacity)}"')
    stroke_color, stroke_opacity = _color(outline)
    if stroke_color is None:
        attrs.append('stroke="none"')
    else:
        attrs.append(f'stroke="{stroke_color}"')
        if stroke_opacity < 1:
            attrs.append(f'stroke-opacity="{_fmt(stroke_opacity)}"')
        attrs.append(f'stroke-width="{_fmt(width or 1)}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')
    return " ".join(attrs)


def _points(points: Any) -> list[tuple[float, float]]:
    if isinstance(points, tuple) and len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
        return [(_num(points[0]), _num(points[1])), (_num(points[2]), _num(points[3]))]
    out: list[tuple[float, float]] = []
    if isinstance(points, (list, tuple)):
        if points and all(isinstance(v, (int, float)) for v in points):
            flat = list(points)
            out = [(_num(flat[i]), _num(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
        else:
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    out.append((_num(point[0]), _num(point[1])))
    return out


def _transform(snippet: str, transform: str) -> str:
    if not snippet:
        return snippet
    return f'<g transform="{transform}">{snippet}</g>'


def _copy_elements(src: Image.Image, dst: Image.Image, transform: str | None = None) -> None:
    src_meta = STATE.get(src, default_raster=True)
    dst_meta = STATE.get(dst)
    for snippet in src_meta["elements"]:
        dst_meta["elements"].append(_transform(snippet, transform) if transform else snippet)


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    # Use the original PIL save implementation so the recorder does not recurse.
    STATE.original["Image.Image.save"](image.convert("RGBA"), buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _embed_image(image: Image.Image, x: float, y: float, width: float, height: float) -> str:
    href = _image_data_uri(image)
    return (
        f'<image x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" height="{_fmt(height)}" '
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
        x0, y0, x1, y1 = map(_num, xy)
        STATE.get(self.image)["elements"].append(
            f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(x1 - x0)}" height="{_fmt(y1 - y0)}" '
            f'rx="{_fmt(radius)}" ry="{_fmt(radius)}" {_style(fill, outline, width)}/>'
        )

    def rectangle(self, xy: Any, fill: Any = None, outline: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.rectangle(xy, fill=fill, outline=outline, width=width, **kwargs)
        x0, y0, x1, y1 = map(_num, xy)
        STATE.get(self.image)["elements"].append(
            f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(x1 - x0)}" height="{_fmt(y1 - y0)}" {_style(fill, outline, width)}/>'
        )

    def ellipse(self, xy: Any, fill: Any = None, outline: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.ellipse(xy, fill=fill, outline=outline, width=width, **kwargs)
        x0, y0, x1, y1 = map(_num, xy)
        STATE.get(self.image)["elements"].append(
            f'<ellipse cx="{_fmt((x0 + x1) / 2)}" cy="{_fmt((y0 + y1) / 2)}" '
            f'rx="{_fmt((x1 - x0) / 2)}" ry="{_fmt((y1 - y0) / 2)}" {_style(fill, outline, width)}/>'
        )

    def line(self, xy: Any, fill: Any = None, width: int = 1, joint: Any = None, **kwargs: Any) -> None:
        self._draw.line(xy, fill=fill, width=width, joint=joint, **kwargs)
        pts = _points(xy)
        if len(pts) < 2:
            return
        color, opacity = _color(fill)
        attrs = [
            'fill="none"',
            f'stroke="{color or "#000000"}"',
            f'stroke-width="{_fmt(width)}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if opacity < 1:
            attrs.append(f'stroke-opacity="{_fmt(opacity)}"')
        d = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        STATE.get(self.image)["elements"].append(f'<polyline points="{d}" {" ".join(attrs)}/>')

    def polygon(self, xy: Any, fill: Any = None, outline: Any = None, **kwargs: Any) -> None:
        self._draw.polygon(xy, fill=fill, outline=outline, **kwargs)
        pts = _points(xy)
        if not pts:
            return
        d = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        STATE.get(self.image)["elements"].append(f'<polygon points="{d}" {_style(fill, outline, 1)}/>')

    def arc(self, xy: Any, start: float, end: float, fill: Any = None, width: int = 1, **kwargs: Any) -> None:
        self._draw.arc(xy, start=start, end=end, fill=fill, width=width, **kwargs)
        x0, y0, x1, y1 = map(_num, xy)
        rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        import math

        sx = cx + rx * math.cos(math.radians(start))
        sy = cy + ry * math.sin(math.radians(start))
        ex = cx + rx * math.cos(math.radians(end))
        ey = cy + ry * math.sin(math.radians(end))
        large = 1 if abs(end - start) > 180 else 0
        color, opacity = _color(fill)
        attrs = [
            'fill="none"',
            f'stroke="{color or "#000000"}"',
            f'stroke-width="{_fmt(width)}"',
            'stroke-linecap="round"',
        ]
        if opacity < 1:
            attrs.append(f'stroke-opacity="{_fmt(opacity)}"')
        STATE.get(self.image)["elements"].append(
            f'<path d="M {_fmt(sx)} {_fmt(sy)} A {_fmt(rx)} {_fmt(ry)} 0 {large} 1 {_fmt(ex)} {_fmt(ey)}" {" ".join(attrs)}/>'
        )

    def text(self, xy: Any, text: Any, fill: Any = None, font: Any = None, anchor: Any = None, **kwargs: Any) -> None:
        self._draw.text(xy, text, fill=fill, font=font, anchor=anchor, **kwargs)
        x, y = _num(xy[0]), _num(xy[1])
        color, opacity = _color(fill)
        size = getattr(font, "size", 16) if font is not None else 16
        family = "DejaVu Sans"
        weight = "700" if "Bold" in str(getattr(font, "path", "")) else "400"
        attrs = [
            f'x="{_fmt(x)}"',
            f'y="{_fmt(y)}"',
            f'fill="{color or "#000000"}"',
            f'font-family="{family}"',
            f'font-size="{_fmt(size)}px"',
            f'font-weight="{weight}"',
            'dominant-baseline="hanging"',
        ]
        if opacity < 1:
            attrs.append(f'fill-opacity="{_fmt(opacity)}"')
        if anchor:
            attrs.append(f'data-pil-anchor="{html.escape(str(anchor))}"')
        safe = html.escape(str(text))
        STATE.get(self.image)["elements"].append(f'<text {" ".join(attrs)}>{safe}</text>')


def install_recorder() -> None:
    def image_new(mode: str, size: Any, color: Any = 0) -> Image.Image:
        image = STATE.original["Image.new"](mode, size, color)
        meta = STATE.set_meta(image)
        fill, opacity = _color(color)
        if fill is not None and opacity > 0 and image.mode not in {"L", "1"}:
            attrs = f'fill="{fill}"'
            if opacity < 1:
                attrs += f' fill-opacity="{_fmt(opacity)}"'
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
        STATE.set_meta(out, deepcopy(src["elements"]), raster=bool(src.get("raster")))
        return out

    def convert_method(self: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.convert"](self, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), raster=bool(src.get("raster")))
        return out

    def resize_method(self: Image.Image, size: Any, *args: Any, **kwargs: Any) -> Image.Image:
        old_w, old_h = self.width, self.height
        out = STATE.original["Image.Image.resize"](self, size, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        sx = out.width / max(old_w, 1)
        sy = out.height / max(old_h, 1)
        elements = [_transform(snippet, f"scale({_fmt(sx)} {_fmt(sy)})") for snippet in src["elements"]]
        STATE.set_meta(out, elements, raster=bool(src.get("raster")))
        return out

    def thumbnail_method(self: Image.Image, size: Any, *args: Any, **kwargs: Any) -> None:
        old_w, old_h = self.width, self.height
        old_meta = deepcopy(STATE.get(self, default_raster=True))
        result = STATE.original["Image.Image.thumbnail"](self, size, *args, **kwargs)
        sx = self.width / max(old_w, 1)
        sy = self.height / max(old_h, 1)
        elements = [_transform(snippet, f"scale({_fmt(sx)} {_fmt(sy)})") for snippet in old_meta["elements"]]
        STATE.set_meta(self, elements, raster=bool(old_meta.get("raster")))
        return result

    def crop_method(self: Image.Image, box: Any = None) -> Image.Image:
        out = STATE.original["Image.Image.crop"](self, box)
        src = STATE.get(self, default_raster=True)
        if box:
            x0, y0 = _num(box[0]), _num(box[1])
            elements = [_transform(snippet, f"translate({_fmt(-x0)} {_fmt(-y0)})") for snippet in src["elements"]]
        else:
            elements = deepcopy(src["elements"])
        STATE.set_meta(out, elements, raster=bool(src.get("raster")))
        return out

    def rotate_method(self: Image.Image, angle: Any, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.rotate"](self, angle, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        # Keep rotated labels and texture snippets visually faithful as raster,
        # but also preserve source SVG elements where possible for inspection.
        STATE.set_meta(out, deepcopy(src["elements"]), raster=True)
        return out

    def filter_method(self: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        out = STATE.original["Image.Image.filter"](self, *args, **kwargs)
        src = STATE.get(self, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), raster=True)
        return out

    def putalpha_method(self: Image.Image, alpha: Any) -> None:
        result = STATE.original["Image.Image.putalpha"](self, alpha)
        meta = STATE.get(self, default_raster=True)
        if bool(meta.get("raster")):
            meta["raster"] = True
        return result

    def enhance_method(self: Any, factor: float) -> Image.Image:
        out = STATE.original["ImageEnhance._Enhance.enhance"](self, factor)
        src = STATE.get(self.image, default_raster=True)
        STATE.set_meta(out, deepcopy(src["elements"]), raster=True if src.get("raster") else bool(src.get("raster")))
        return out

    def alpha_composite_method(self: Image.Image, im: Image.Image, dest: Any = (0, 0), source: Any = (0, 0)) -> None:
        result = STATE.original["Image.Image.alpha_composite"](self, im, dest, source)
        dst_meta = STATE.get(self)
        src_meta = STATE.get(im, default_raster=True)
        dx, dy = (dest if isinstance(dest, tuple) else (0, 0))
        if src_meta.get("raster") or not src_meta["elements"]:
            dst_meta["elements"].append(_embed_image(im, dx, dy, im.width, im.height))
        for snippet in src_meta["elements"]:
            dst_meta["elements"].append(_transform(snippet, f"translate({_fmt(dx)} {_fmt(dy)})"))
        return result

    def paste_method(self: Image.Image, im: Any, box: Any = None, mask: Any = None) -> None:
        result = STATE.original["Image.Image.paste"](self, im, box, mask)
        dst_meta = STATE.get(self)
        if isinstance(box, tuple):
            if len(box) == 2:
                dx, dy = _num(box[0]), _num(box[1])
                dw = im.width if isinstance(im, Image.Image) else 0
                dh = im.height if isinstance(im, Image.Image) else 0
            else:
                dx, dy = _num(box[0]), _num(box[1])
                dw, dh = _num(box[2]) - dx, _num(box[3]) - dy
        else:
            dx = dy = 0
            dw = im.width if isinstance(im, Image.Image) else self.width
            dh = im.height if isinstance(im, Image.Image) else self.height
        if isinstance(im, Image.Image):
            src_meta = STATE.get(im, default_raster=True)
            if src_meta.get("raster") or not src_meta["elements"]:
                if mask is not None:
                    masked = STATE.original["Image.new"]("RGBA", (im.width, im.height), (0, 0, 0, 0))
                    STATE.original["Image.Image.paste"](masked, im.convert("RGBA"), (0, 0), mask)
                    dst_meta["elements"].append(_embed_image(masked, dx, dy, dw, dh))
                else:
                    dst_meta["elements"].append(_embed_image(im, dx, dy, dw, dh))
            for snippet in src_meta["elements"]:
                dst_meta["elements"].append(_transform(snippet, f"translate({_fmt(dx)} {_fmt(dy)})"))
        elif isinstance(box, tuple) and len(box) == 4:
            fill, opacity = _color(im)
            if fill is not None and opacity > 0:
                attrs = f'fill="{fill}"'
                if opacity < 1:
                    attrs += f' fill-opacity="{_fmt(opacity)}"'
                dst_meta["elements"].append(
                    f'<rect x="{_fmt(dx)}" y="{_fmt(dy)}" width="{_fmt(dw)}" height="{_fmt(dh)}" {attrs} stroke="none"/>'
                )
        return result

    def save_method(self: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> Any:
        result = STATE.original["Image.Image.save"](self, fp, *args, **kwargs)
        path = Path(fp) if isinstance(fp, (str, Path)) else None
        if path is not None and path.suffix.lower() == ".png" and path.stem in TARGET_STEMS:
            svg_path = REBUILD_DIR / f"{path.stem}.svg"
            source_png = SOURCE_DIR / f"{path.stem}.png"
            _write_svg(self, svg_path, source_png if source_png.exists() else path)
            STATE.saved.append({"png": str(path), "svg": str(svg_path), "source": str(source_png)})
        return result

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


def _write_svg(image: Image.Image, svg_path: Path, reference_png: Path) -> None:
    meta = STATE.get(image)
    width, height = image.width, image.height
    ref_href = "data:image/png;base64," + base64.b64encode(reference_png.read_bytes()).decode("ascii")
    body = "\n".join(meta["elements"])
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" width="{width / DPI:.6f}in" height="{height / DPI:.6f}in" viewBox="0 0 {width} {height}" version="1.1">
  <title>{html.escape(svg_path.stem)}</title>
  <desc>Source-rebuilt editable SVG export. Text and drawing primitives are editable SVG elements; photographic/heatmap raster content is embedded as image objects. A hidden exact reference layer is included for alignment checks.</desc>
  <g id="exact_original_reference_hidden" style="display:none">
    <image x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" href="{ref_href}" xlink:href="{ref_href}"/>
  </g>
  <g id="editable_source_rebuild">
{body}
  </g>
</svg>
'''
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")


def write_manifest() -> None:
    rows: list[dict[str, Any]] = []
    for item in STATE.saved:
        svg = Path(item["svg"])
        source = Path(item["source"])
        if not svg.exists():
            continue
        with Image.open(source) as im:
            width, height = im.size
        text = svg.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "asset": svg.stem,
                "source_png": str(source),
                "editable_svg": str(svg),
                "width_px": width,
                "height_px": height,
                "svg_bytes": svg.stat().st_size,
                "text_objects": text.count("<text "),
                "rect_objects": text.count("<rect "),
                "ellipse_objects": text.count("<ellipse "),
                "line_or_path_objects": text.count("<polyline ") + text.count("<path "),
                "polygon_objects": text.count("<polygon "),
                "image_objects": text.count("<image "),
            }
        )
    rows = sorted(rows, key=lambda r: r["asset"])
    manifest_csv = REBUILD_DIR / "editable_svg_rebuild_manifest.csv"
    if rows:
        with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    manifest_json = REBUILD_DIR / "editable_svg_rebuild_manifest.json"
    manifest_json.write_text(json.dumps({"assets": rows}, indent=2), encoding="utf-8")
    lines = [
        "# Editable SVG Rebuild Exports",
        "",
        "These SVG files are regenerated from the Pillow figure source with drawing operations recorded as SVG objects.",
        "",
        "- Text, lines, rectangles, ellipses, polygons, and arcs are editable SVG elements.",
        "- Real H&E/photographic/heatmap content remains embedded as image objects.",
        "- Each SVG includes a hidden `exact_original_reference_hidden` layer containing the final approved PNG for visual alignment checks.",
        "- These are for object editing. Use `../vector_exports_exact/` when exact pixel appearance is the only requirement.",
        "",
        "| Asset | Text | Rect | Ellipse | Line/path | Polygon | Images | SVG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['asset']} | {row['text_objects']} | {row['rect_objects']} | {row['ellipse_objects']} | "
            f"{row['line_or_path_objects']} | {row['polygon_objects']} | {row['image_objects']} | `{Path(row['editable_svg']).name}` |"
        )
    (REBUILD_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REBUILD_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PNG_DIR.mkdir(parents=True, exist_ok=True)
    install_recorder()
    from wavestgate.evaluation.manuscript_figures import build_manuscript_figures

    build_manuscript_figures(output_dir=TEMP_PNG_DIR)
    write_manifest()
    print(f"Wrote editable SVG rebuilds to {REBUILD_DIR}")


if __name__ == "__main__":
    main()
