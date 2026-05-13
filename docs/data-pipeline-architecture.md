# Data Pipeline Architecture

End-to-end flow from raw video to labeled training data.

---

## Pipeline Stages

```
Raw Video (Drive)
      |
      v
[1] Frame Extraction ---- ffmpeg, 1 frame every 3s (N=10 → ~30s window per group)
      |
      v
[2] YOLO Detection ------ YOLOv8m-seg on FULL FRAMES (not crops)
      |
      v
[3] Tracking ------------ match same person across N frames (centroid + IoU, K=3 lookback)
      |
      v
[4] Per-Table Cropping --- perspective warp each frame to each table's zone_poly
      |
      v
[5] Perception Build ----- filter YOLO detections to people overlapping this table
      |
      v
[6] Upload to Drive ------ sampled crops + perception artifact → unlabeled/{folder}/
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

### [3] Within-Group Tracking

Frames are grouped into non-overlapping **groups** of N consecutive frames
(default N=10 today; legacy N=3 still supported). Group size is detected
per-folder at runtime. Within each group:

- **Algorithm:** greedy centroid + bbox-IoU matching with a K=3 frame lookback
- **Max distance:** 0.2 normalized centroid distance per inter-frame gap
- **Frame 0:** each person gets a new ID (`t0`, `t1`, ...)
- **Frames 1..N-1:** for each detection, build candidate set from tracks last
  seen in any of the previous K frames (more recent preferred); score by
  `-distance + IoU_bonus * iou(bbox)`; greedy 1-to-1 assignment.
- **No match:** person gets a new track ID (treated as a different person).

**Effect:** seated customers keep the same `track_id` across all N frames,
and a single missed YOLO detection (one frame in 10) is recoverable instead
of starting a fresh track. Fast-moving waiters exceed the distance threshold
and get new IDs each frame. The `gap_count` and `disjoint_track_count` fields
in `perception.json` expose tracker noise rather than smoothing it away.

> The "triplet" name is preserved on the wire/Drive protocol — folder suffix
> `_t{idx:04d}`, app-property `autolabel_preprocess_triplets`, edge config
> `frames_per_triplet` — for backwards compatibility with deployed clients
> and existing labeled data.

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

Each group x table produces one Drive folder:

```
unlabeled/{video_stem}_{table_id}_t{idx:04d}/
    frame_0.jpg           # perspective-warped crop of table zone
    frame_1.jpg           # 3 seconds later
    ...
    frame_{N-1}.jpg       # 3*(N-1) seconds later (27s at N=10)
    perception_v2.json      # person-table interaction data for 10-frame groups
```

A parallel debug output also goes to:

```
temp_processing/{video_stem}_t{idx:04d}/
    frame_0.jpg           # full frame with polygon overlays + person bbox drawings
    frame_1.jpg
    ...
    frame_{N-1}.jpg
```

### [7] Human Labeling

Flask web UI (`app.py`) shows the N cropped frames from `unlabeled/`.
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

One file per table per group. Written only if YOLO was available at processing
time. See [PERCEPTION.md](../PERCEPTION.md) for the full schema-v2 specification
(tracks[] array, tenure features, sitting-vs-passing geometry, gap-filled
transitions). Below is an abbreviated quick-reference.

```jsonc
{
    "schema_version": 2,
    "n_frames": 10,
    "people": [
        {
            "frame_index": 0,
            "track_id": "t0",
            "centroid_x": 0.4231, "centroid_y": 0.6102,
            "bbox_xyxy": [320.0, 410.0, 480.5, 690.2],
            "overlap_frac_of_person": 0.7812,
            "overlap_frac_of_table": 0.3104,
            "distance_norm": 0.0823,
            "displacement_from_prev": null,
            "score": 0.943,
            "bbox_below_table_frac": 0.42,   // legs under table → strong sit signal
            "bbox_aspect_ratio": 1.18         // h/w; standing >2.5, sitting ~1.0–1.5
        }
        // ... one row per (person × frame) pair
    ],
    "scalars": {
        // Per-frame lists, length n_frames
        "person_count": [...],
        "max_overlap_frac_of_person": [...],
        "overlap_sum": [...],
        "min_distance_norm": [...],
        // Per-transition lists, length n_frames - 1
        "entering": [...],            // gap-filled (single-frame YOLO miss tolerated)
        "leaving":  [...],
        "raw_entering": [...],        // raw, no gap-fill
        "raw_leaving":  [...],
        // Aggregate displacements
        "mean_displacement": 0.0019,
        "max_displacement":  0.0021,
        // Tenure (auto-scaled thresholds: ceil(0.8N), floor(0.2N))
        "persistent_count":           1,    // strict: present in ALL frames
        "mostly_persistent_count":    1,    // primary occupied signal
        "transient_count":            0,    // primary waiter signal
        "max_track_tenure":           10,
        "max_consecutive_tenure":     10,
        // Group-level dwelling
        "frames_with_any_person":     10,
        "frames_with_dwelling_person": 10,  // overlap_frac_of_person > 0.5
        "primary_track_id":           "t0",
        "primary_track_dwell_frames": 10,
        // Tracker-noise indicator
        "disjoint_track_count":       0
    },
    "tracks": [
        {
            "track_id": "t0",
            "frames_present": 10,
            "max_consecutive": 10,
            "first_frame": 0, "last_frame": 9,
            "presence_ratio": 1.0,
            "consecutive_ratio": 1.0,
            "gap_count": 0,
            "dwell_frames": 10,
            "mean_overlap_frac_of_person": 0.795,
            // ... see PERCEPTION.md for the complete per-track field list
        }
    ]
}
```

The schema is **strictly the occupied/unoccupied signal.** Dirty-vs-clean is
computed elsewhere from the cropped images themselves.

Old `perception.json` files written under schema v1 (no `schema_version` field,
N=3) are still valid; readers should default `n_frames` to
`len(scalars.person_count)` when missing.

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
│       ├── ...
│       └── frame_{N-1}.jpg                #   N=10 today; N=3 for legacy data
│
├── unlabeled/                             # awaiting human review
│   └── IPC11_2024-01-15_table_top_5_t0003/
│       ├── frame_0.jpg                    #   perspective-warped crop of zone_poly
│       ├── frame_1.jpg
│       ├── ...
│       ├── frame_{N-1}.jpg
│       └── perception_v2.json
│
├── clean/                                 # LABELED: table is clean/empty
│   └── IPC11_2024-01-15_table_top_5_t0000/
│       ├── frame_0.jpg                    #   same cropped frames, unchanged
│       ├── ...
│       ├── frame_{N-1}.jpg
│       └── perception_v2.json            #   same perception data, unchanged
│
├── dirty/                                 # LABELED: table is dirty/needs clearing
│   └── IPC11_2024-01-15_table_top_5_t0001/
│       └── ...
│
└── occupied/                              # LABELED: table has person(s) seated
    └── IPC11_2024-01-15_table_top_5_t0002/
        └── ...
```

**The folder contents are identical across all 3 label categories.**
The label is encoded solely by which parent directory (`clean/`, `dirty/`, `occupied/`) the folder sits in. The `_t{idx:04d}` suffix in folder names is preserved
from the legacy "triplet" terminology even though groups now contain N frames.

---

## Coordinate Spaces

There are two coordinate spaces in this system:

| Space | Where used | Origin | Units |
|-------|-----------|--------|-------|
| **Full-frame pixels** | `bbox_xyxy` in the perception artifact, table polygon coords | top-left (0,0) | pixels at actual frame resolution |
| **Normalized 0-1** | `centroid_x/y`, `distance_norm`, `displacement_from_prev` | top-left (0,0) | fraction of frame width/height |

The sampled cropped frame images are in their own local pixel space after the
perspective warp. There is no coordinate mapping stored from crop space back to
full-frame space — the crops are visual data only. All structured coordinates
in the perception artifact reference the **original full frame**.

---

## Key Design Decisions

1. **YOLO on full frames, not crops** — detections in full-frame coordinates let you
   compute overlap with any table polygon. Running YOLO per-crop would miss people
   partially outside the crop boundary and duplicate detections of people near
   multiple tables.

2. **zone_poly for crops, tight_poly for overlap** — crops use the expanded zone
   to include visual context (chairs, nearby floor). Overlap metrics use the tight
   polygon for precise "is this person actually at the table" measurement.

3. **Groups not individual frames** — N frames at 3-second intervals capture
   temporal signal. At N=10 (current target) the window covers ~30 seconds of
   table activity. A seated person appears in most/all N frames
   (`mostly_persistent_count >= 1`, low displacement, high
   `bbox_below_table_frac`). A passing waiter appears in 1-2 frames
   (`transient_count >= 1`, high displacement, low `bbox_below_table_frac`).

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
