# Perception Data — How It Works

Every labeled subfolder can contain a `perception.json` alongside the 3 images.
It is generated during `--process` if `ultralytics` is installed, and is carried
through untouched when the human labels the sample. Training code can use both
the images and this file.

---

## The problem it solves

A table region might have a person in the camera frame. But is that person
**sitting at the table** (→ occupied) or a **waiter walking past** (→ unoccupied)?

A single frame can't tell the difference. Across 3 frames (~20-30 seconds apart):
- A **seated customer** barely moves → same position all 3 frames, high overlap, low displacement
- A **waiter** moves quickly → different position each frame, low/zero overlap, high displacement (or just 1 of 3 frames)

---

## How it is generated

```
For each triplet of 3 frames:

  Step 1 — Detection (full frame, per frame)
    Run YOLOv8-seg on each full frame.
    For every detected person, store: mask, bbox, centroid (normalized 0–1), confidence.

  Step 2 — Tracking (across 3 frames)
    Match people between frames by nearest centroid.
    If the same person's centroid moves < 20% of the frame diagonal between frames
    → same track_id.
    If they move more, or disappear → new track_id (different person or waiter).

  Step 3 — Per-table filtering and perception building
    For each table in the triplet (7 tables → 7 separate perception.json files):
      Filter to only people whose mask OR bbox overlaps this table's polygon.
      For each kept (person × frame) pair, compute:
        overlap_frac_of_person   — what % of the person's body is over the table
        overlap_frac_of_table    — what % of the table they cover
        distance_norm            — centroid distance to table centroid (0–1)
        displacement_from_prev   — how far they moved since previous frame (0–1)
      Write perception.json.
```

---

## perception.json structure

```json
{
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
      "score": 0.94
    },
    {
      "frame_index": 1,
      "track_id": "t0",
      "centroid_x": 0.43,
      "centroid_y": 0.61,
      "bbox_xyxy": [322.0, 412.0, 481.0, 689.0],
      "overlap_frac_of_person": 0.80,
      "overlap_frac_of_table": 0.33,
      "distance_norm": 0.07,
      "displacement_from_prev": 0.009,
      "score": 0.93
    },
    {
      "frame_index": 2,
      "track_id": "t0",
      "centroid_x": 0.43,
      "centroid_y": 0.62,
      "bbox_xyxy": [321.0, 411.0, 482.0, 691.0],
      "overlap_frac_of_person": 0.81,
      "overlap_frac_of_table": 0.34,
      "distance_norm": 0.07,
      "displacement_from_prev": 0.003,
      "score": 0.95
    }
  ],
  "scalars": {
    "person_count":               [1, 1, 1],
    "max_overlap_frac_of_person": [0.78, 0.80, 0.81],
    "overlap_sum":                [0.78, 0.80, 0.81],
    "min_distance_norm":          [0.08, 0.07, 0.07],
    "mean_displacement":          0.006,
    "max_displacement":           0.009,
    "entering":                   [0, 0],
    "leaving":                    [0, 0],
    "persistent_count":           1
  }
}
```

### `people` — one entry per (person × frame)

| Field | What it means |
|---|---|
| `frame_index` | Which of the 3 frames (0, 1, 2) |
| `track_id` | Same ID across frames = same physical person. New ID = different person or re-entry. |
| `centroid_x/y` | Person centroid, normalized 0–1 by frame width/height |
| `bbox_xyxy` | Bounding box in full-frame pixel coords |
| `overlap_frac_of_person` | What fraction of **their body** is inside the table polygon |
| `overlap_frac_of_table` | What fraction of **the table area** they cover |
| `distance_norm` | Centroid distance from table centroid, normalized by frame diagonal |
| `displacement_from_prev` | How far they moved since the previous frame (normalized). `null` if first appearance. |
| `score` | YOLOv8 detection confidence |

### `scalars` — pre-aggregated summaries

All list values have 3 elements, one per frame.

| Field | What it means |
|---|---|
| `person_count` | How many people overlap this table in each frame |
| `max_overlap_frac_of_person` | Largest single overlap fraction per frame |
| `overlap_sum` | Sum of all overlap fractions per frame |
| `min_distance_norm` | Closest person's distance per frame |
| `mean_displacement` | Average displacement across all (person × frame) pairs with a previous frame |
| `max_displacement` | Largest single displacement |
| `entering[0]` | Tracks new in frame 1 that weren't in frame 0 |
| `entering[1]` | Tracks new in frame 2 that weren't in frame 1 |
| `leaving[0]` | Tracks in frame 0 that weren't in frame 1 |
| `leaving[1]` | Tracks in frame 1 that weren't in frame 2 |
| `persistent_count` | Number of unique track_ids present in **all 3** frames (seated customer signal) |

---

## Waiter vs customer patterns

**Seated customer:**
```
person_count:      [1, 1, 1]     ← present all 3 frames
persistent_count:  1              ← same track all 3 frames
mean_displacement: ~0.005         ← barely moved
leaving:           [0, 0]         ← never left
entering:          [0, 0]         ← was there from the start
```

**Waiter walking through:**
```
person_count:      [0, 1, 0]     ← only in one frame
persistent_count:  0              ← no persistent tracks
mean_displacement: null           ← can't compute
leaving:           [0, 1]         ← gone by frame 2
entering:          [1, 0]         ← appeared in frame 1
```

**Table with no one near it:**
```
person_count:      [0, 0, 0]
people:            []             ← empty list
persistent_count:  0
```

---

## Enabling person detection

By default `--process` runs without person detection (fast).
To enable, install ultralytics:

```bash
pip install ultralytics
```

Then re-run `--process`. YOLOv8n-seg weights (~6 MB) auto-download on first run.
To use a more accurate (but slower) model, edit `load_yolo_model()` in
`person_detector.py` and change `"yolov8n-seg.pt"` to `"yolov8s-seg.pt"`.

---

## Note on 7 tables sharing the same 3 frames

Person detection runs **once per triplet** on the full frame, then the results
are filtered per-table. This means:

- The 7 subfolders from one triplet share the same underlying detections
- But each `perception.json` only contains people relevant to **that table**
- A person at table 1 will not appear in table 3's perception.json
