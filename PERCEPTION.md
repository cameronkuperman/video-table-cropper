# Perception Data — How It Works

Every labeled subfolder can contain a perception artifact alongside the sampled
images in the group. Legacy 3-frame groups use `perception.json`; current
10-frame groups use `perception_10frame.json`. It is generated during
`--process` if `ultralytics` is installed, and is carried through untouched
when the human labels the sample. Training code can use both the images and
this file.

`perception.json` is **strictly the occupied/unoccupied signal.** Dirty-vs-clean
classification is computed elsewhere from the cropped image itself.

The current capture target is **N = 10 frames per group, 3 s/frame** (about 27 s
elapsed between the first and last frame, 30 s nominal window). Older folders
on Drive may have N = 3; the schema and readers are N-aware so both work.

---

## The problem it solves

A table region might have a person in the camera frame. But is that person
**sitting at the table** (→ occupied) or a **waiter walking past** (→ unoccupied)?

A single frame can't tell the difference. Across 10 frames at 3 s intervals:
- A **seated customer** barely moves → same position, high overlap, low
  displacement, bbox extends below the table polygon's bottom edge (legs
  under table).
- A **waiter walking past** moves quickly → different position each frame,
  low/zero overlap, high displacement, often present in only 1–3 frames,
  bbox fully above the table edge if walking *behind* the table.

10 frames over 30 s also lets the model see arrival and departure mid-window
that 3 frames over 9 s couldn't capture.

---

## How it is generated

```
For each source group of N frames:

  Step 1 — Detection (full frame, per frame)
    Run YOLOv8-seg on each full frame.
    For every detected person, store: mask, bbox, centroid (normalized 0–1), confidence.

  Step 2 — Tracking (across N frames)
    Match people between frames by centroid distance + bbox-IoU, with a K=3
    frame lookback so a single missed detection doesn't break the track.
    If the same person's centroid stays within ~20% of frame diagonal per
    inter-frame gap → same track_id. Larger jumps → new track_id.

  Step 3 — Per-table filtering and perception building
    For each table in the group (e.g. 7 tables → 7 separate perception files):
      Filter to only people whose mask OR bbox overlaps this table's polygon.
      For each kept (person × frame) pair, compute:
        overlap_frac_of_person   — what % of the person's body is over the table
        overlap_frac_of_table    — what % of the table they cover
        distance_norm            — centroid distance to table centroid (0–1)
        displacement_from_prev   — how far they moved since previous frame (0–1)
        bbox_below_table_frac    — fraction of bbox below the table's bottom edge
        bbox_aspect_ratio        — h/w; standing ~2.5+, sitting ~1.0–1.5
      Aggregate per-track and per-group, write the perception artifact.
```

---

## perception artifact structure (schema v2)

```json
{
  "schema_version": 2,
  "n_frames": 10,

  "people": [
    {
      "frame_index": 0,
      "track_id": "t0",
      "centroid_x": 0.42,
      "centroid_y": 0.61,
      "bbox_xyxy": [320.0, 410.0, 480.0, 690.0],
      "overlap_frac_of_person": 0.78,
      "overlap_frac_of_table": 0.31,
      "distance_norm": 0.08,
      "displacement_from_prev": null,
      "score": 0.94,
      "bbox_below_table_frac": 0.42,
      "bbox_aspect_ratio": 1.18
    }
  ],

  "scalars": {
    "person_count":               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "max_overlap_frac_of_person": [0.78, 0.80, 0.81, 0.81, 0.79, 0.80, 0.79, 0.81, 0.80, 0.79],
    "overlap_sum":                [0.78, 0.80, 0.81, 0.81, 0.79, 0.80, 0.79, 0.81, 0.80, 0.79],
    "min_distance_norm":          [0.08, 0.07, 0.07, 0.07, 0.08, 0.07, 0.08, 0.07, 0.07, 0.08],
    "mean_displacement":          0.006,
    "max_displacement":           0.012,

    "entering":     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "leaving":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "raw_entering": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "raw_leaving":  [0, 0, 0, 0, 0, 0, 0, 0, 0],

    "persistent_count":           1,
    "mostly_persistent_count":    1,
    "transient_count":            0,
    "max_track_tenure":           10,
    "max_consecutive_tenure":     10,

    "frames_with_any_person":     10,
    "frames_with_dwelling_person": 10,
    "primary_track_id":           "t0",
    "primary_track_dwell_frames": 10,

    "disjoint_track_count":       0
  },

  "tracks": [
    {
      "track_id":                    "t0",
      "frames_present":              10,
      "max_consecutive":             10,
      "first_frame":                 0,
      "last_frame":                  9,
      "presence_ratio":              1.0,
      "consecutive_ratio":           1.0,
      "gap_count":                   0,
      "dwell_frames":                10,
      "mean_overlap_frac_of_person": 0.795,
      "max_overlap_frac_of_person":  0.81,
      "overlap_stddev":              0.0102,
      "mean_distance_norm":          0.074,
      "min_distance_norm":           0.07,
      "centroid_spread":             0.008,
      "mean_displacement":           0.006,
      "max_displacement":            0.012,
      "mean_score":                  0.94,
      "min_score":                   0.93,
      "mean_bbox_below_table_frac":  0.41,
      "mean_bbox_aspect_ratio":      1.20
    }
  ]
}
```

### Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Bumps when the schema makes a non-additive change. Currently 2. |
| `n_frames` | The N this file describes. Scalar list lengths and `frame_index` ranges follow this. |

### `people` — one entry per (person × frame they appear in)

| Field | Meaning |
|---|---|
| `frame_index` | Which of the N frames (0..N-1) |
| `track_id` | Same ID across frames = same physical person (subject to tracker limits — see below) |
| `centroid_x/y` | Person centroid, normalized 0–1 by frame width/height |
| `bbox_xyxy` | Bounding box in full-frame pixel coords |
| `overlap_frac_of_person` | What fraction of **their body** is inside the table polygon |
| `overlap_frac_of_table` | What fraction of **the table area** they cover |
| `distance_norm` | Centroid distance from table centroid, normalized by frame diagonal |
| `displacement_from_prev` | How far they moved since the previous frame (normalized). `null` on first appearance. |
| `score` | YOLOv8 detection confidence |
| `bbox_below_table_frac` | Fraction of the bbox vertical extent that lies below the table's bottom edge. **Strong sitting signal**: legs under the table. |
| `bbox_aspect_ratio` | Bbox h/w. Standing ~2.5+, sitting ~1.0–1.5, leaning/bending ~0.5–1.0. |

### `scalars` — pre-aggregated summaries

Per-frame lists are length N. Per-transition lists are length N−1.

| Field | Meaning |
|---|---|
| `person_count` | How many people overlap this table in each frame |
| `max_overlap_frac_of_person` | Largest single overlap fraction per frame |
| `overlap_sum` | Sum of all overlap fractions per frame |
| `min_distance_norm` | Closest person's distance per frame |
| `mean_displacement` / `max_displacement` | Across all (person × frame) pairs that have a previous frame |
| `entering[i]` | Tracks new in frame `i+1` that weren't in frame `i`, after gap-filling single-frame YOLO misses (max gap 1) |
| `leaving[i]` | Tracks in frame `i` that weren't in frame `i+1`, gap-filled |
| `raw_entering` / `raw_leaving` | Same as above but **without** gap-filling — exposes the YOLO/tracker noise for debugging |
| `persistent_count` | Tracks present in **all N** frames. High-precision but low-recall. |
| `mostly_persistent_count` | Tracks present in `>=` 80% of frames (`>=8/10`). **Primary occupied signal.** |
| `transient_count` | Tracks present in `<=` 20% of frames (`<=2/10`). **Primary waiter signal.** |
| `max_track_tenure` | Largest `frames_present` among all tracks |
| `max_consecutive_tenure` | Longest run of *consecutive* frames any track was present |
| `frames_with_any_person` | Number of frames in which any person overlaps this table |
| `frames_with_dwelling_person` | Number of frames with at least one person at `overlap_frac_of_person > 0.5`. Stronger than `person_count` for occupied detection. |
| `primary_track_id` | Track with the highest summed `overlap_frac_of_person` (or `null` if no people detected) |
| `primary_track_dwell_frames` | `dwell_frames` of that primary track |
| `disjoint_track_count` | Pairs of tracks with disjoint frame sets and mean centroids within 0.1 normalized distance. High value = tracker probably split one customer into multiple IDs. |

### `tracks` — per-track summary (denormalized from `people`)

One entry per unique `track_id`. Lets downstream code apply its own tenure
threshold without re-scanning `people[]`.

| Field | Meaning |
|---|---|
| `frames_present` / `max_consecutive` | Tenure counts |
| `presence_ratio` / `consecutive_ratio` | The above as fractions of N (model-friendly) |
| `gap_count` | Missing-frame runs strictly between `first_frame` and `last_frame`. 0 = continuous. >0 = sporadic. |
| `dwell_frames` | Frames where this track had `overlap_frac_of_person > 0.5` |
| `mean_overlap_frac_of_person` / `max_overlap_frac_of_person` / `overlap_stddev` | Overlap stability across the track's lifetime |
| `mean_distance_norm` / `min_distance_norm` | Distance to table centroid |
| `centroid_spread` | Spatial stddev of the track's centroid. Seated tight, walker spreads. |
| `mean_displacement` / `max_displacement` | Per-frame motion (preferred over `total_displacement`, which just rewards longer tracks) |
| `mean_score` / `min_score` | Detection-confidence aggregates. Low `min_score` flags reflection/shadow false positives. |
| `mean_bbox_below_table_frac` | Aggregate sitting signal |
| `mean_bbox_aspect_ratio` | Aggregate posture signal |

---

## Waiter vs customer patterns at N=10

**Seated customer:**
```
person_count:                  [1,1,1,1,1,1,1,1,1,1]
mostly_persistent_count:       1
persistent_count:              1
transient_count:               0
max_consecutive_tenure:        10
primary_track_dwell_frames:    10
mean_displacement:             ~0.005
tracks[0].centroid_spread:     ~0.005          ← stayed put
tracks[0].mean_bbox_below_table_frac: ~0.40    ← legs under table
```

**Waiter walking past behind the table:**
```
person_count:                  [0,1,1,0,0,0,0,0,0,0]
mostly_persistent_count:       0
transient_count:               1
max_consecutive_tenure:        2
mean_displacement:             ~0.06
tracks[0].centroid_spread:     ~0.05            ← moved across
tracks[0].mean_bbox_below_table_frac: 0.0       ← entirely above table
tracks[0].mean_bbox_aspect_ratio: ~2.7           ← tall narrow standing
```

**Customer arriving mid-window (sat down at frame 4):**
```
person_count:                  [0,0,0,0,1,1,1,1,1,1]
mostly_persistent_count:       0
mostly_persistent_count tolerates 80%: would need >=8/10 → 0
max_consecutive_tenure:        6
tracks[0].first_frame:         4
tracks[0].consecutive_ratio:   0.6
tracks[0].mean_bbox_below_table_frac: ~0.40   ← still sat
```

**Flaky-seated (real customer, YOLO missed one frame):**
```
person_count:                  [1,1,1,0,1,1,1,1,1,1]
persistent_count:              0   (strict — fails because of frame 3)
mostly_persistent_count:       1   (>=8/10 → still counts)
tracks[0].gap_count:           1
entering / leaving:            [0]*9   (gap-filled)
raw_entering / raw_leaving:    show the spurious flap
```

**Empty table:**
```
person_count:                  [0]*10
people:                        []
tracks:                        []
frames_with_any_person:        0
primary_track_id:              null
```

---

## Notes on tracker reliability

`track_id` is produced by a centroid + bbox-IoU matcher with a 3-frame
lookback. It tolerates one-frame YOLO drops and small displacements but is not
a true re-identification model. Practical limits:

- Two customers seated within ~20% of frame diagonal of each other can
  occasionally swap IDs across frames.
- A customer who stands up, walks away, and a different customer who sits
  in the same chair can be merged into one track if the centroids land close.
- A customer dropped for 4+ consecutive frames will get a new track ID on
  return.

`disjoint_track_count` and per-track `gap_count` expose this noise to the
model rather than smoothing it away. Treat `persistent_count` as a
high-confidence/low-recall feature; prefer `mostly_persistent_count`,
`max_consecutive_tenure`, and `primary_track_dwell_frames` for the main
occupied signal.

---

## Schema-version compatibility

Old `perception.json` files on Drive (no `schema_version` or `n_frames` field)
are still valid. Readers should:

- Default `n_frames` to `len(scalars.person_count)` if missing (or 3 if the
  file is otherwise pre-schema).
- Tolerate variable-length lists rather than indexing `[0]`/`[1]`/`[2]`.
- Treat new fields (`tracks`, `mostly_persistent_count`, `bbox_below_table_frac`,
  etc.) as optional.

---

## Enabling person detection

By default `--process` runs without person detection (fast).
To enable, install ultralytics:

```bash
pip install ultralytics
```

Then re-run `--process`. YOLOv8m-seg weights (~52 MB) auto-download on first
run. Smaller variants (`yolov8n-seg.pt`, `yolov8s-seg.pt`) are faster and less
accurate; edit `load_yolo_model()` in `person_detector.py` to switch.

---

## Note on multiple tables sharing the same N frames

Person detection runs **once per group** on the full frames, then the results
are filtered per-table. This means:

- The subfolders from one group (one per table) share the same underlying
  detections.
- Each `perception.json` only contains people relevant to **that table**.
- A person at table 1 will not appear in table 3's perception.json.
