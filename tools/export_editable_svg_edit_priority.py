"""Create edit-priority SVGs from source-rebuilt editable SVG exports.

The source-rebuilt SVGs preserve a hidden exact PNG reference and may also
contain full-page raster composites captured from intermediate Pillow images.
Those large images are visually useful, but in vector editors they can make it
feel like the text and panels are not selectable.  This post-processor removes
page-sized raster blockers and rewrites the file with explicit image, vector,
and text layers.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path("/mnt/WaveST-Gate")
SOURCE_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_rebuild"
OUTPUT_DIR = ROOT / "results/nature_manuscript_figures/editable_svg_edit_priority"

ATTR_RE = re.compile(r'([\w:.-]+)="([^"]*)"')
IMAGE_TAG_RE = re.compile(r"<image\b([^>]*)>", re.S)
SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.S)


def _attrs(fragment: str) -> dict[str, str]:
    return dict(ATTR_RE.findall(fragment))


def _number(value: str | None) -> float:
    if not value:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else 0.0


def _svg_size(svg_open: str) -> tuple[float, float]:
    attrs = _attrs(svg_open)
    view_box = attrs.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    return _number(attrs.get("width")), _number(attrs.get("height"))


def _is_large_image_line(line: str, page_w: float, page_h: float) -> bool:
    """Return true when the line contains a page-sized image element."""
    for match in IMAGE_TAG_RE.finditer(line):
        attrs = _attrs(match.group(1))
        width = _number(attrs.get("width"))
        height = _number(attrs.get("height"))
        if width >= page_w * 0.82 and height >= page_h * 0.82:
            return True
    return False


def _indent(item: str) -> str:
    return "\n".join(f"    {line}" for line in item.splitlines())


def _read_editable_body(path: Path) -> tuple[str, list[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    svg_match = SVG_OPEN_RE.search(text)
    if not svg_match:
        raise ValueError(f"No <svg> root found in {path}")
    svg_open = "<svg" + svg_match.group(1) + ">"
    body_match = re.search(
        r'<g id="editable_source_rebuild">\n(?P<body>.*)\n  </g>\n</svg>\n?\Z',
        text,
        flags=re.S,
    )
    if not body_match:
        raise ValueError(f"No editable_source_rebuild group found in {path}")
    items: list[str] = []
    pending: list[str] = []
    for raw_line in body_match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if pending:
            pending.append(line)
            if "</text>" in line:
                items.append("\n".join(pending))
                pending = []
            continue
        if line.startswith("<text ") and "</text>" not in line:
            pending = [line]
            continue
        items.append(line)
    if pending:
        items.append("\n".join(pending))
    return svg_open, items


def convert_one(path: Path) -> dict[str, int | str]:
    svg_open, lines = _read_editable_body(path)
    page_w, page_h = _svg_size(svg_open)

    image_lines: list[str] = []
    vector_lines: list[str] = []
    text_lines: list[str] = []
    skipped_large_images = 0

    for line in lines:
        item = line.strip()
        if not item:
            continue
        if "<image" in item:
            if _is_large_image_line(item, page_w, page_h):
                skipped_large_images += len(IMAGE_TAG_RE.findall(item))
                continue
            image_lines.append(item)
        elif "<text " in item:
            text_lines.append(item)
        else:
            vector_lines.append(item)

    out_path = OUTPUT_DIR / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = path.stem
    out_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        svg_open,
        f"  <title>{title}</title>",
        "  <desc>Edit-priority SVG: page-sized raster blockers removed; embedded image panels, vector objects, and text are separated into editable layers.</desc>",
        '  <g id="embedded_image_panels">',
        *[_indent(line) for line in image_lines],
        "  </g>",
        '  <g id="editable_vector_objects">',
        *[_indent(line) for line in vector_lines],
        "  </g>",
        '  <g id="editable_text_objects">',
        *[_indent(line) for line in text_lines],
        "  </g>",
        "</svg>",
        "",
    ]
    out_path.write_text("\n".join(out_lines), encoding="utf-8")

    return {
        "asset": path.stem,
        "editable_svg": str(out_path),
        "width_px": int(page_w),
        "height_px": int(page_h),
        "svg_bytes": out_path.stat().st_size,
        "text_objects": len(text_lines),
        "vector_lines": len(vector_lines),
        "image_objects": len(image_lines),
        "skipped_large_images": skipped_large_images,
    }


def write_readme(rows: list[dict[str, int | str]]) -> None:
    manifest = OUTPUT_DIR / "editable_svg_edit_priority_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Editable SVG Edit-Priority Exports",
        "",
        "These SVG files are cleaned from `../editable_svg_rebuild/` for easier object selection in vector editors.",
        "",
        "- Page-sized raster composites and the hidden exact PNG reference layer are removed.",
        "- Remaining H&E/photo/heatmap panels are separate embedded image objects.",
        "- Vector objects and text are separated into `editable_vector_objects` and `editable_text_objects` layers.",
        "- Use these files when you need to move or edit text and figure objects.",
        "- Use `../editable_svg_rebuild/` only when you want the hidden exact-reference layer for alignment checks.",
        "",
        "| Asset | Text | Vector lines | Images | Removed page images | SVG |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['asset']} | {row['text_objects']} | {row['vector_lines']} | "
            f"{row['image_objects']} | {row['skipped_large_images']} | `{Path(str(row['editable_svg'])).name}` |"
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [convert_one(path) for path in sorted(SOURCE_DIR.glob("*.svg"))]
    if not rows:
        raise SystemExit(f"No SVG files found in {SOURCE_DIR}")
    write_readme(rows)
    print(f"Wrote {len(rows)} edit-priority SVGs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
