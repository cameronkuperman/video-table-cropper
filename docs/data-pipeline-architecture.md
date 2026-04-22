# Data Pipeline Architecture

End-to-end flow from raw video to labeled training data.

---

## Pipeline Stages

```
Raw Video (Drive)
      |
      v
[1] Frame Extraction ---- ffmpeg, 1 frame every 3s
      |
      v
[2] YOLO Detection ------ YOLOv8m-seg on FULL FRAMES (not crops)
      |
      v
[3] Tracking ------------ match same person across 3 frames by centroid
      |
      v
[4] Per-Table Cropping --- perspective warp each frame to each table's zone_poly
      |
      v
[5] Perception Build ----- filter YOLO detections to people overlapping this table
      |
      v
[6] Upload to Drive ------ crops + perception.json → unlabeled/{folder}/
      |
      v
[7] Human Labeling ------- Flask UI, reviewer picks clean/dirty/occupied
      |
      v
[8] Export (folder move) - Drive folder moves from unlabeled/ → {label}/
```

---

## Stage Details

### [1] Frame Extraction

- **Tool:** ffmpeg via `extract_frames.py`
- **Input:** video file (mp4/mov/avi/mkv/webm)
- **Output:** full-resolution JPEG frames, one every 3 seconds
- **Naming:** `frame_XXXX_{timestamp}.jpg`
- **Quality:** ffmpeg `-q:v 2` (high quality)

### [2] YOLO Detection

**YOLO runs on full uncropped frames. Not on table crops.**

- **Model:** `yolov8m-seg.pt` (YOLOv8 medium, segmentation variant)
- **Size:** ~52 MB, auto-downloads on first run
- **Classes:** person only (class 0)
- **Confidence threshold:** 0.15
- **Per-frame output:** list of person detections, each containing:

```python
{
    "mask":          np.ndarray(bool, shape=(H, W)),  # pixel-level segmentation mask
    "bbox_xyxy":     [x1, y1, x2, y2],               # bounding box, full-frame pixel coords
    "centroid_norm":  (cx, cy),                       # center of mass, normalized 0-1
    "score":         float,                           # detection confidence 0-1
    "track_id":      None,                            # assigned in step 3
}
```

The mask is a full-resolution boolean array (same H x W as the frame). It is used for
pixel-level overlap computation but is NOT serialized to JSON.

### [3] Within-Triplet Tracking

Frames are grouped into non-overlapping **triplets** of 3 consecutive frames
(frames 0-1-2, then 3-4-5, etc.). Within each triplet:

- **Algorithm:** greedy nearest-centroid matching
- **Max distance:** 0.2 (normalized, where diagonal = sqrt(2))
- **Frame 0:** each person gets a new ID (`t0`, `t1`, ...)
- **Frames 1-2:** match to previous frame's people by centroid distance
- **If distance > 0.2:** person gets a new track ID (treated as a different person)

**Effect:** seated customers keep the same `track_id` across all 3 frames.
Fast-moving waiters exceed the distance threshold and get new IDs each frame.

### [4] Per-Table Cropping

For each table in the camera config, each frame is cropped:

- **Crop region:** `zone_poly` (expanded zone, NOT tight_rect)
- **Method:** PIL QUAD perspective transform (deskews rotated tables)
- **Fallback:** axis-aligned bbox crop if polygon != 4 points
- **Format:** JPEG quality 90, RGB
- **Output size:** determined by the polygon edge lengths (not fixed)

### [5] Perception Build

For each table, YOLO detections from step 2 are filtered:

**Inclusion criteria** (a person is included if either is true):
1. Person's segmentation mask has pixel overlap with the table's `tight_poly` mask
2. Person's bounding box intersects the table's polygon (even if mask doesn't overlap)

Then overlap metrics, distances, and displacement are computed per included person.

### [6] Upload

Each triplet x table produces one Drive folder:

```
unlabeled/{video_stem}_{table_id}_t{triplet_idx:04d}/
    frame_0.jpg           # perspective-warped crop of table zone
    frame_1.jpg           # 3 seconds later
    frame_2.jpg           # 6 seconds later
    perception.json       # person-table interaction data (if YOLO available)
```

A parallel debug output also goes to:

```
temp_processing/{video_stem}_t{triplet_idx:04d}/
    frame_0.jpg           # full frame with polygon overlays + person bbox drawings
    frame_1.jpg
    frame_2.jpg
```

### [7] Human Labeling

Flask web UI (`app.py`) shows the 3 cropped frames from `unlabeled/`.
Reviewer assigns one of:

| Action       | What happens                                     |
|-------------|--------------------------------------------------|
| **clean**    | folder moved to `clean/`                         |
| **dirty**    | folder moved to `dirty/`                         |
| **occupied** | folder moved to `occupied/`                      |
| **label_later** | folder moved to `label_later/`                |
| **discard**  | folder moved to `discarded/`                     |

### [8] Export

Labeling IS the export. The label action is a **Drive folder move**:

```
unlabeled/{folder_name}/  →  {label}/{folder_name}/
```

The folder contents are identical regardless of label. The label is encoded
by which parent directory the folder lives in.

---

## Data Structures

### Table Geometry (`approved_table_rectangles.json`)

```json
{
    "cameras": [
        {
            "camera_id": "IPC11",
            "camera_name": "IPC11",
            "camera_number": 11,
            "image_width": 1280,
            "image_height": 720,
            "expansion_strength": 4.5,
            "tables": [
                {
                    "mask_id": 4,
                    "label": "table top_5",
                    "color": "#af52de",
                    "score": 0.815,
                    "bbox": [1102, 322, 100, 86],
                    "tight_rect": {
                        "center_x": 1151.2,
                        "center_y": 364.1,
                        "width_px": 90.2,
                        "height_px": 62.8,
                        "angle_deg": 26.4,
                        "polygon": [[1125, 316], [1206, 356], [1178, 412], [1097, 372]]
                    },
                    "zone_rect": {
                        "center_x": 1151.2,
                        "center_y": 364.1,
                        "width_px": 137.2,
                        "height_px": 140.4,
                        "angle_deg": 26.4,
                        "polygon": [[1121, 271], [1244, 332], [1181, 457], [1059, 396]]
                    }
                }
            ]
        }
    ]
}
```

**tight_rect** — snug boundary around the table surface.
Used for: overlap computation with person masks.

**zone_rect** — expanded region around the table (controlled by `expansion_strength`).
Used for: cropping frames. Includes surrounding context (chairs, floor near table).

Both share the same `center_x/y` and `angle_deg`. The zone is just wider/taller.

Polygon coordinates are in the reference resolution (`image_width` x `image_height`).
If the actual video frame resolution differs, all coordinates are scaled proportionally
(angle is NOT scaled — it's rotation-invariant).

### perception.json

One file per table per triplet. Written only if YOLO was available at processing time.

```json
{
    "people": [
        {
            "frame_index": 0,
            "track_id": "t0",
            "centroid_x": 0.4231,
            "centroid_y": 0.6102,
            "bbox_xyxy": [320.0, 410.0, 480.5, 690.2],
            "overlap_frac_of_person": 0.7812,
            "overlap_frac_of_table": 0.3104,
            "distance_norm": 0.0823,
            "displacement_from_prev": null,
            "score": 0.943
        },
        {
            "frame_index": 1,
            "track_id": "t0",
            "centroid_x": 0.4248,
            "centroid_y": 0.6089,
            "bbox_xyxy": [322.1, 408.3, 482.0, 688.7],
            "overlap_frac_of_person": 0.7956,
            "overlap_frac_of_table": 0.3201,
            "distance_norm": 0.0791,
            "displacement_from_prev": 0.0021,
            "score": 0.951
        }
    ],
    "scalars": {
        "person_count": [1, 1, 1],
        "max_overlap_frac_of_person": [0.7812, 0.7956, 0.8010],
        "overlap_sum": [0.7812, 0.7956, 0.8010],
        "min_distance_norm": [0.0823, 0.0791, 0.0774],
        "mean_displacement": 0.0019,
        "max_displacement": 0.0021,
        "entering": [0, 0],
        "leaving": [0, 0],
        "persistent_count": 1
    }
}
```

#### `people` array — one entry per (person x frame)

| Field | Type | Description |
|-------|------|-------------|
| `frame_index` | int (0/1/2) | Which frame in the triplet |
| `track_id` | string | Same ID across frames = same physical person. Format: `"t0"`, `"t1"`, etc. |
| `centroid_x` | float 0-1 | Person center-of-mass X, normalized by frame width |
| `centroid_y` | float 0-1 | Person center-of-mass Y, normalized by frame height |
| `bbox_xyxy` | [x1, y1, x2, y2] | Bounding box in **full-frame pixel coordinates** (not crop coordinates) |
| `overlap_frac_of_person` | float 0-1 | What fraction of this person's segmentation mask overlaps the table polygon. 0.78 = 78% of person is over the table. |
| `overlap_frac_of_table` | float 0-1 | What fraction of the table area this person covers. 0.31 = person covers 31% of table surface. |
| `distance_norm` | float 0+ | Euclidean distance from person centroid to table centroid, normalized by frame dimensions. Lower = closer to table center. |
| `displacement_from_prev` | float or null | How far the person moved since the previous frame (normalized). `null` for the person's first appearance. Low values = stationary (seated). |
| `score` | float 0-1 | YOLO detection confidence |

#### `scalars` object — aggregate signals per triplet

| Field | Type | Description |
|-------|------|-------------|
| `person_count` | [int, int, int] | Number of people overlapping this table in each frame |
| `max_overlap_frac_of_person` | [float, float, float] | Highest person-overlap in each frame |
| `overlap_sum` | [float, float, float] | Sum of all person overlaps per frame |
| `min_distance_norm` | [float, float, float] | Closest person to table center per frame. Defaults to 1.0 if no people. |
| `mean_displacement` | float | Average movement across all tracked people. 0.0 if no displacement data. |
| `max_displacement` | float | Maximum single-frame movement across all people |
| `entering` | [int, int] | [new people in frame 1 not in frame 0, new people in frame 2 not in frame 1] |
| `leaving` | [int, int] | [people gone in frame 1 that were in frame 0, gone in frame 2 that were in frame 1] |
| `persistent_count` | int | Number of unique track IDs present in ALL 3 frames |

---

## Exported Folder Structure (after labeling)

```
Drive project root/
├── raw_videos/                            # input videos
│   └── IPC11_2024-01-15.mp4
│
├── temp_processing/                       # debug: full frames with overlays
│   └── IPC11_2024-01-15_t0000/
│       ├── frame_0.jpg                    #   full frame + polygon overlays + person bboxes
│       ├── frame_1.jpg
│       └── frame_2.jpg
│
├── unlabeled/                             # awaiting human review
│   └── IPC11_2024-01-15_table_top_5_t0003/
│       ├── frame_0.jpg                    #   perspective-warped crop of zone_poly
│       ├── frame_1.jpg
│       ├── frame_2.jpg
│       └── perception.json
│
├── clean/                                 # LABELED: table is clean/empty
│   └── IPC11_2024-01-15_table_top_5_t0000/
│       ├── frame_0.jpg                    #   same cropped frames, unchanged
│       ├── frame_1.jpg
│       ├── frame_2.jpg
│       └── perception.json               #   same perception data, unchanged
│
├── dirty/                                 # LABELED: table is dirty/needs clearing
│   └── IPC11_2024-01-15_table_top_5_t0001/
│       ├── frame_0.jpg
│       ├── frame_1.jpg
│       ├── frame_2.jpg
│       └── perception.json
│
└── occupied/                              # LABELED: table has person(s) seated
    └── IPC11_2024-01-15_table_top_5_t0002/
        ├── frame_0.jpg
        ├── frame_1.jpg
        ├── frame_2.jpg
        └── perception.json
```

**The folder contents are identical across all 3 label categories.**
The label is encoded solely by which parent directory (`clean/`, `dirty/`, `occupied/`) the folder sits in.

---

## Coordinate Spaces

There are two coordinate spaces in this system:

| Space | Where used | Origin | Units |
|-------|-----------|--------|-------|
| **Full-frame pixels** | `bbox_xyxy` in perception.json, table polygon coords | top-left (0,0) | pixels at actual frame resolution |
| **Normalized 0-1** | `centroid_x/y`, `distance_norm`, `displacement_from_prev` | top-left (0,0) | fraction of frame width/height |

The cropped frame images (frame_0/1/2.jpg) are in their own local pixel space
after the perspective warp. There is no coordinate mapping stored from crop space
back to full-frame space — the crops are visual data only. All structured
coordinates in perception.json reference the **original full frame**.

---

## Key Design Decisions

1. **YOLO on full frames, not crops** — detections in full-frame coordinates let you
   compute overlap with any table polygon. Running YOLO per-crop would miss people
   partially outside the crop boundary and duplicate detections of people near
   multiple tables.

2. **zone_poly for crops, tight_poly for overlap** — crops use the expanded zone
   to include visual context (chairs, nearby floor). Overlap metrics use the tight
   polygon for precise "is this person actually at the table" measurement.

3. **Triplets not individual frames** — 3 frames at 3-second intervals capture
   temporal signal. A seated person appears in all 3 (persistent_count=1+,
   low displacement). A passing waiter appears in 1-2 frames (entering/leaving
   counts > 0, high displacement).

4. **Label = folder location** — no metadata database for labels. The Drive folder
   hierarchy IS the label store. Simple, human-browsable, and works with any
   downstream pipeline that reads directories.

---

## Session / Provenance Tracking

The current simple labeling system (`app.py`) stores **no metadata about the labeling
event** — no timestamp, no reviewer ID, no session. The label is purely which Drive
folder the sample was moved into.

However, the **folder name itself** encodes the full source provenance:

```
IPC11_2024-01-15_table_top_5_t0003
  │       │            │        │
  │       │            │        └─ triplet index (temporal position in video)
  │       │            └─ table ID from approved_table_rectangles.json
  │       └─ video filename stem (= recording session)
  └─ camera ID (IPC number)
```

### Parsing the folder name

| Component | How to extract | What it tells you |
|-----------|---------------|-------------------|
| **Camera ID** | Leading `IPC\d+` | Physical camera angle/location |
| **Video stem** | Everything before the table ID | Source recording session (one continuous video = one session) |
| **Table ID** | Between video stem and `_t\d{4}` | Which table in the camera's field of view |
| **Triplet index** | Trailing `_t\d{4}` | Temporal position: `t0000` = first 9s of video, `t0001` = next 9s, etc. |

The **video stem is the session identifier.** All folders sharing the same video stem
came from the same continuous recording and share the same lighting, camera angle,
time of day, and background.

### Train/Val/Test Splits for Generalization

To prevent data leakage (model memorizing specific scenes rather than learning
generalizable features), **never split at the individual sample level.** Split at a
higher grouping level so correlated samples stay together.

Three levels of split strictness:

#### Level 1: Split by video stem (minimum)

All triplets from the same video go into the same split.

```python
from collections import defaultdict
import re

def video_stem_from_folder(folder_name: str) -> str:
    """Strip _tXXXX suffix to get video_stem + table_id."""
    return re.sub(r"_t\d{4}$", "", folder_name)

def video_session_from_folder(folder_name: str) -> str:
    """Extract just the video stem (recording session), without table_id.

    For 'IPC11_2024-01-15_table_top_5_t0003' this returns
    something like 'IPC11_2024-01-15' — the unique recording.

    Since table IDs vary in format, the safest approach is to group
    by everything before the last known table label pattern.
    """
    # Strip triplet index
    base = re.sub(r"_t\d{4}$", "", folder_name)
    return base

# Group all samples by their source video
by_video = defaultdict(list)
for folder_name, label in all_labeled_samples:
    session = video_session_from_folder(folder_name)
    by_video[session].append((folder_name, label))

# Split at the VIDEO level, not sample level
video_keys = sorted(by_video.keys())
# e.g. 70/15/15 split across video sessions
```

**Prevents:** same-frame temporal leakage. Two triplets 9 seconds apart from the
same video will never end up in different splits.

#### Level 2: Split by camera + date (better)

Group by camera AND recording date so the same day's footage from one camera
never spans splits.

```python
def camera_date_key(folder_name: str) -> str:
    """Extract camera + date as a grouping key."""
    m = re.match(r"(IPC\d+)[_-](\d{4}-\d{2}-\d{2})", folder_name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # fallback: use the full video stem
    return video_session_from_folder(folder_name)
```

**Prevents:** same-day lighting/scene leakage within a camera.

#### Level 3: Split by camera ID (strictest)

Entire cameras go into one split. Train on IPC3, IPC7, IPC11; test on IPC5, IPC9.

```python
def camera_id_from_folder(folder_name: str) -> str:
    m = re.match(r"(IPC\d+)", folder_name)
    return m.group(1) if m else "unknown"
```

**Prevents:** model memorizing table geometry, camera angle, or background
appearance for a specific camera. This is the strongest test of generalization
but requires enough distinct cameras to have meaningful splits.

#### Recommended approach

Use **Level 1 (video stem)** as the minimum. If you have enough cameras (5+),
also validate with **Level 3 (camera holdout)** to check cross-camera generalization.

```
Train:  IPC3 (all videos), IPC7 (all videos), IPC11 (all videos)
Val:    IPC5 (all videos)
Test:   IPC9 (all videos)  ← never seen camera angle during training
```

If the model performs well on a held-out camera it has never seen, it has learned
table state features rather than memorizing specific scenes.
