# Static Table Geometry Contract

This is the handoff contract for any downstream repo that consumes static table geometry from this repo.

## Source Of Truth

Use the rectangle export / UI JSON, not `bbox`, not mask hulls, and not mask crop thumbnails.

The worker and review flow in this repo load static table metadata from `approved_table_rectangles.json` when present, otherwise `approved_tables.json`. For geometry, downstream consumers must read `tight_rect` and `zone_rect`.

Per table, the contract is:

```json
{
  "mask_id": 3,
  "label": "table_3",
  "bbox": [x, y, w, h],
  "tight_rect": {
    "center_x": 1034.2,
    "center_y": 582.7,
    "width_px": 418.0,
    "height_px": 96.0,
    "angle_deg": 14.3,
    "polygon": [[834,520],[1240,624],[1194,713],[788,609]]
  },
  "zone_rect": {
    "center_x": 1034.2,
    "center_y": 582.7,
    "width_px": 462.0,
    "height_px": 134.0,
    "angle_deg": 14.3,
    "polygon": [[812,499],[1260,614],[1210,731],[762,616]]
  }
}
```

Use:
- `tight_rect` for the tight crop
- `zone_rect` for the expanded crop

Important:
- `polygon` is the authoritative geometry.
- `bbox` is only useful for badge center / metadata.
- `width_px` is the long-axis length.
- `height_px` is the short-axis length.
- Crop from the original full-resolution frame, not a resized UI image.

## Exact Backend Math

The rectangle corner order must match the backend implementation in [segmentation.py](/Users/huntercameronkuperman/Documents/ScreenRecord_v1/backend/segmentation.py#L701) and the zone-rect serialization path in [segmentation.py](/Users/huntercameronkuperman/Documents/ScreenRecord_v1/backend/segmentation.py#L2748).

If `polygon` is present, use it directly.

If `polygon` is missing, reconstruct it exactly like this:

```python
from math import cos, radians, sin

theta = radians(angle_deg)

ux = cos(theta)
uy = sin(theta)

vx = -sin(theta)
vy = cos(theta)

half_long = width_px / 2.0
half_short = height_px / 2.0

corners_local = [
    (-half_long, -half_short),
    ( half_long, -half_short),
    ( half_long,  half_short),
    (-half_long,  half_short),
]

polygon = [
    (
        center_x + long_off * ux + short_off * vx,
        center_y + long_off * uy + short_off * vy,
    )
    for (long_off, short_off) in corners_local
]
```

That is the exact corner order used here.

The expanded rect is built from the same center and angle as `tight_rect`, then serialized as `zone_rect`.

## Exact Crop To Match This Repo

If the downstream repo wants the crop to match the rectangle exactly, preserving camera perspective and keeping only the pixels inside the rotated box, use an RGBA masked crop:

```python
import cv2
import numpy as np
from math import ceil, cos, floor, radians, sin


def rect_polygon(rect):
    poly = rect.get("polygon") or []
    if len(poly) >= 4:
        return np.array(poly, dtype=np.float32)

    cx = float(rect["center_x"])
    cy = float(rect["center_y"])
    w = float(rect["width_px"])
    h = float(rect["height_px"])
    theta = radians(float(rect["angle_deg"]))

    ux, uy = cos(theta), sin(theta)
    vx, vy = -sin(theta), cos(theta)

    half_long = w / 2.0
    half_short = h / 2.0
    corners_local = [
        (-half_long, -half_short),
        ( half_long, -half_short),
        ( half_long,  half_short),
        (-half_long,  half_short),
    ]

    return np.array([
        [cx + lo * ux + so * vx, cy + lo * uy + so * vy]
        for lo, so in corners_local
    ], dtype=np.float32)


def crop_rect_rgba(image_bgr, rect):
    pts = rect_polygon(rect)

    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, int(floor(np.min(pts[:, 0]))))
    y0 = max(0, int(floor(np.min(pts[:, 1]))))
    x1 = min(w_img - 1, int(ceil(np.max(pts[:, 0]))))
    y1 = min(h_img - 1, int(ceil(np.max(pts[:, 1]))))

    crop = image_bgr[y0:y1 + 1, x0:x1 + 1].copy()

    local_pts = pts - np.array([x0, y0], dtype=np.float32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(local_pts).astype(np.int32)], 255)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


tight_crop_rgba = crop_rect_rgba(frame_bgr, table["tight_rect"])
expanded_crop_rgba = crop_rect_rgba(frame_bgr, table["zone_rect"])
```

That yields:
- a tight RGBA crop
- an expanded RGBA crop
- transparency outside the rotated rectangle

If the downstream repo wants a black background instead of transparency:

```python
masked = cv2.bitwise_and(crop, crop, mask=mask)
```

## Deskewed Top-Down Crop

Use a perspective warp only if the downstream repo wants a normalized straightened crop instead of a crop that matches the UI geometry:

```python
def warp_rect(image_bgr, rect):
    src = rect_polygon(rect).astype(np.float32)
    w = max(1, int(round(float(rect["width_px"]))))
    h = max(1, int(round(float(rect["height_px"]))))
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


tight_warp = warp_rect(frame_bgr, table["tight_rect"])
expanded_warp = warp_rect(frame_bgr, table["zone_rect"])
```

If the downstream repo wants the crop to match the UI exactly, use the RGBA masked crop, not the warp.

## UI Rendering Rules

To match this repo's geometry in overlays:
- draw `zone_rect` first
- use a dashed outline for `zone_rect`
- then draw `tight_rect`
- use a solid outline for `tight_rect`
- if `polygon` exists, draw from `polygon`
- only fall back to `center_x`, `center_y`, `width_px`, `height_px`, `angle_deg` if `polygon` is missing
- do not recompute from `bbox`

## Minimum Required Fields

Per camera:
- `camera_id`
- `image_width`
- `image_height`

Per table:
- `mask_id`
- `label`
- `bbox`
- `tight_rect`
- `zone_rect`

Per rect:
- `center_x`
- `center_y`
- `width_px`
- `height_px`
- `angle_deg`
- `polygon`

## One-Line Rule

If the downstream repo uses `tight_rect.polygon` and `zone_rect.polygon` directly on the original frame, its crops and overlays will match this repo. If it uses `bbox` or rebuilds the rectangle differently, it will drift.
