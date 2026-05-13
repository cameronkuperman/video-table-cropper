# How to build `perception_v2.json` (preprocessing repo guide)

This document is a **self-contained specification** for producing
`perception_v2.json` files in any repository — language-agnostic, implementation-
agnostic. Drop this guide into the preprocessing repo and an engineer can
build a compatible producer without reading the labeling-side code.

If you change the schema in any non-additive way, bump `schema_version` here
and in every reader.

---

## 1. What `perception_v2.json` is

The perception artifact is a small JSON file that lives **next to the sampled
cropped table images** in a Drive folder or local directory. Current 10-frame
groups use `perception_v2.json`; legacy folders may still contain
`perception.json` or `perception_10frame.json`. There is exactly one perception artifact per
`(table × group)` — i.e. for each combination of one physical table and one
group of N frames sampled from a camera looking at that table.

It is **not** the cropped images themselves and it is **not** any kind of raw
detection log. It is a *summary* of the people YOLO detected near that one
specific table, flattened across N frames into rows + scalar aggregates
designed for downstream training and labeling.

A typical file is < 5 KB and contains:

- A list of per-(person × frame) detection rows.
- A flat `scalars` block of pre-aggregated signals across the N frames.
- A `tracks` array with one entry per unique track ID (denormalized summary).
- Two top-level fields: `schema_version` (currently `2`) and `n_frames`.

---

## 2. Why `perception.json` exists (the problem it solves)

The downstream binary task is **occupied vs unoccupied**: is there a customer
seated at this table?

A single image of a restaurant table is not enough to answer that, because:

- A **waiter walking past** can sit inside the table polygon for one frame and
  look identical to a seated customer in that frame.
- A **person walking *behind* the table** can have a bbox that overlaps the
  table polygon from the camera's POV but be 4 m past it.
- **Reflections** in glass partitions occasionally produce phantom detections.
- A **YOLO false negative** can drop a real customer for one frame.

Across N frames sampled at 3-second intervals, these confusable cases each
produce different temporal signatures:

- A **seated customer** is present in (almost) every frame, barely moves,
  has high overlap, and their bounding box extends *below* the table polygon
  (legs under the table).
- A **waiter walking past** is present in 1–2 frames, displaces fast, and has
  a tall narrow bbox entirely above the table polygon's bottom edge.
- A **flaky-seated customer** (real customer + one missed YOLO frame) looks
  like the seated case minus a single hole.

`perception.json` is the artifact that captures these temporal signatures in
a form a downstream model (or a human labeler in the meantime) can use, **without
having to re-run YOLO at training time.**

The labeling UI also reads it for hints. Training reads both the cropped
images and `perception.json`.

> **Strict scope.** `perception.json` is **only for the occupied/unoccupied
> signal.** Dirty-vs-clean classification happens elsewhere from the cropped
> images themselves. Don't put dish/cutlery/object detections in here.

---

## 3. Inputs

To build `perception.json` for one `(table × group)` you need:

1. **N full-resolution frames** of the same camera, sampled at a fixed
   interval (3 seconds in our pipeline). N is currently 10 → 30 s nominal
   window. The build code must accept any N in `[2, 16]` so it scales.

2. **For each frame: a list of person detections** from a YOLO segmentation
   model (we use YOLOv8m-seg, class=`person`, confidence threshold 0.15).
   Each detection includes:
   - A pixel-level boolean **mask** the size of the full frame.
   - A **bounding box** `[x1, y1, x2, y2]` in full-frame pixel coordinates.
   - A normalized **centroid** `(cx, cy)` in `[0, 1]` (mean of the mask in
     normalized coords).
   - A **confidence score** in `[0, 1]`.

3. **The table polygon** in full-frame pixel coordinates. Convex/quad
   typical, but any simple polygon works. The polygon describes the
   table-top *footprint* (the "tight" polygon, not the expanded crop zone).

4. The **frame image dimensions** `(H, W)` for normalization.

YOLO must be run **once per group on the full uncropped frames** — never
per-cropped-image. This is critical: a person partially outside one table's
crop is a real detection that must be visible to nearby tables. Also, we
filter the same set of detections through *every* table polygon in the
group, so running YOLO once amortizes the cost across all tables that share
the group.

---

## 4. Algorithm overview

```
Step 1 — Per-frame detection:
    For each of the N frames, run YOLO and produce a list of person dicts.

Step 2 — Within-group tracking:
    Assign track IDs across the N frames so the same physical person keeps
    the same ID. Centroid + bbox-IoU matching with a K-frame lookback.

Step 3 — Per-table perception build:
    For each table:
      a. Filter the N frame-detection lists to only people whose mask OR
         bbox overlaps this table's polygon.
      b. For each surviving (person × frame) pair, compute geometric
         features (overlap, distance, displacement, sitting indicators).
      c. Aggregate per-frame, per-transition, per-track, and per-group.
      d. Emit perception.json.

Step 4 — Write:
    Serialize as UTF-8 JSON (with indent=2 for human-debuggability) and
    store next to the sampled crop images.
```

Steps 1, 2 happen once per group (shared across every table in the group).
Step 3 happens once per `(table × group)`. A camera with 7 tables on one
group produces 7 files, all built from the same YOLO output.

---

## 5. Step 2 — tracking algorithm in full

Use this exact algorithm for compatibility with our reference implementation.

**Tunables** (in our reference: `TRACK_LOOKBACK = 3`,
`TRACK_MAX_DIST = 0.2`, `TRACK_IOU_BONUS = 0.5`):

```python
TRACK_LOOKBACK = 3      # how many previous frames to consider candidates from
TRACK_MAX_DIST = 0.2    # max normalized centroid distance per inter-frame gap
TRACK_IOU_BONUS = 0.5   # weight added to score for overlapping bbox IoU
```

**Procedure:**

1. **Frame 0:** every detection gets a fresh ID `t0`, `t1`, … and is recorded
   in `last_seen[track_id] = (frame_idx=0, person_dict)`.

2. **Frame i (1 ≤ i < N):**
   a. Build the candidate set: every track in `last_seen` whose
      `seen_frame ∈ [i - TRACK_LOOKBACK, i - 1]`.
   b. For each pair `(curr_detection, candidate)`, compute:
      - `dist` = Euclidean distance between their normalized centroids.
      - `age` = `i - candidate.seen_frame` (1, 2, or 3).
      - **Reject** if `dist > TRACK_MAX_DIST × age`. (Threshold loosens
        proportionally with age — a customer drifts more in 3 frames than 1.)
      - `iou` = bbox IoU of `curr_detection.bbox` and `candidate.bbox`.
      - `score = -dist + TRACK_IOU_BONUS × iou - 0.01 × (age - 1)`.
        The recency penalty (`-0.01 × (age - 1)`) breaks ties in favor of
        the more recent frame.
   c. Sort all `(curr_index, track_id, score)` proposals by descending
      score. Greedily assign 1-to-1: each `curr_index` and each `track_id`
      can only be claimed once. Higher-scoring proposals win.
   d. Any `curr_index` not assigned gets a new track ID `t{next_id++}`.
   e. Update `last_seen` for every detection in this frame (refresh
      recency for matched IDs, register new IDs for unmatched ones).

This recovers **single-frame YOLO drops** — a customer missed for one frame
gets re-matched to its old ID via a 2-frame-old candidate. It does not
recover 4+ consecutive drops; that's by design (the customer probably did
leave and a new customer sat down).

The IoU tiebreaker is what stops **two seated customers near each other from
swapping IDs** when a 1-pixel centroid drift would otherwise pick the wrong
neighbor.

---

## 6. Step 3 — per-table perception build

For each `(group, table)`:

### 6.1 Filter to people overlapping this table

For each frame `f` and each detection `p` in that frame:

1. Rasterize the table polygon to a boolean mask the size of the full frame
   (`table_mask`, shape `(H, W)`).
2. Compute `intersection_px = popcount(p.mask & table_mask)`.
3. **Inclusion rule:**
   - If `intersection_px > 0` → include the detection.
   - Else, check if the detection's bbox overlaps `table_mask` at all
     (any non-zero pixel inside the bbox region of the table mask). If yes,
     include with `overlap_frac = 0` (bbox-touch only). If no, drop the
     detection — truly unrelated to this table.

### 6.2 Compute per-row features

For each surviving `(person × frame)` pair, append a row to `people[]`:

| Field | Computation |
|---|---|
| `frame_index` | `f` |
| `track_id` | from Step 2 |
| `centroid_x`, `centroid_y` | normalized centroid of the mask, rounded to 4 d.p. |
| `bbox_xyxy` | full-frame pixel coords, rounded to 1 d.p. |
| `overlap_frac_of_person` | `intersection_px / max(person_mask_pixel_count, 1)` |
| `overlap_frac_of_table` | `intersection_px / max(table_mask_pixel_count, 1)` |
| `distance_norm` | `hypot(centroid_x - table_centroid_x, centroid_y - table_centroid_y)` where table centroid is the mean of the mask coords, normalized |
| `displacement_from_prev` | `hypot(d_centroid)` from this same `track_id`'s previous-frame centroid (any earlier frame this track was seen at the table). `null` if first appearance at this table. |
| `score` | YOLO confidence, rounded to 3 d.p. |
| `bbox_below_table_frac` | `max(0, bbox_y2 - table_polygon_max_y) / max(bbox_y2 - bbox_y1, ε)`, clipped to `[0, 1]`. **The single most discriminative sitting feature.** Legs under the table → high. |
| `bbox_aspect_ratio` | `(bbox_y2 - bbox_y1) / max(bbox_x2 - bbox_x1, 1)`. Standing ≈ 2.5+, sitting ≈ 1.0–1.5, leaning ≈ 0.5–1.0. |

Notes:
- "Table polygon's max y" = the largest y-coordinate among the polygon's
  vertices in image-pixel coordinates (image y grows downward, so max y is
  the bottom edge).
- `displacement_from_prev` compares against this track's previous-frame
  centroid *as filtered through this table*. If a track exits the table's
  polygon and re-enters, displacement is null on re-entry (no continuity
  through a missing frame at this table).

### 6.3 Compute scalars (group-level aggregates)

All thresholds derive from N so they auto-scale:
```
PERSISTENT_THRESHOLD = N                          # strict
MOSTLY_THRESHOLD     = ceil(0.8 * N)              # 8/10 at N=10
TRANSIENT_MAX        = floor(0.2 * N)             # 2/10 at N=10
DWELL_OVERLAP        = 0.5                        # fraction of body over table
```

**Per-frame lists** (length N):
```
person_count[f]               = count of rows with frame_index == f
max_overlap_frac_of_person[f] = max(overlap_frac_of_person for rows in f)
overlap_sum[f]                = sum(overlap_frac_of_person for rows in f)
min_distance_norm[f]          = min(distance_norm for rows in f), default 1.0
```

**Per-transition lists** (length N − 1):
```
For each track t with frame set frames(t):
  filled = gap_fill(frames(t), max_gap=1)        # see below
  for i in range(N - 1):
    if (i+1) in frames(t) and i not in frames(t): raw_entering[i] += 1
    if i in frames(t) and (i+1) not in frames(t): raw_leaving[i]  += 1
    if (i+1) in filled and i not in filled:       entering[i]      += 1
    if i in filled and (i+1) not in filled:       leaving[i]       += 1
```

`gap_fill(frames, max_gap=k)` = the input frame set with any gap of size
`≤ k` between two present frames filled in. Example with `k=1`:
`{0,1,2,4,5} → {0,1,2,3,4,5}`. `{0,1,5,6}` is unchanged (gap of 3).

The gap-filled lists suppress single-frame YOLO misses from producing fake
enter/leave events. The raw lists are kept so the model can see when the
detector was unreliable.

**Aggregate displacements** (over every row that has a non-null
`displacement_from_prev`):
```
mean_displacement = mean of those displacements, 0.0 if none
max_displacement  = max of those displacements,  0.0 if none
```

**Tenure signals** (one pass over the per-track frame sets):
```
For each track t:
  tenure       = |frames(t)|
  consecutive  = longest run of consecutive integers in frames(t)
  max_track_tenure       = max(max_track_tenure, tenure)
  max_consecutive_tenure = max(max_consecutive_tenure, consecutive)
  if tenure >= PERSISTENT_THRESHOLD:  persistent_count        += 1
  if tenure >= MOSTLY_THRESHOLD:      mostly_persistent_count += 1
  if tenure <= TRANSIENT_MAX:         transient_count         += 1
```

**Group-level dwelling signals:**
```
frames_with_any_person      = count of frames where person_count[f] > 0
frames_with_dwelling_person = count of frames f where some row has
                              frame_index == f AND
                              overlap_frac_of_person > DWELL_OVERLAP

primary_track_id           = track_id with max sum(overlap_frac_of_person),
                             or null if no rows
primary_track_dwell_frames = the dwell_frames of that track
                             (per-track count of rows with overlap > 0.5)
```

**ID-swap defense:**
```
disjoint_track_count = count of unordered pairs (a, b) of tracks where
  frames(a) ∩ frames(b) is empty AND
  hypot(mean_centroid(a), mean_centroid(b)) <= 0.1
```
A high value = the tracker probably split one customer across multiple IDs.

### 6.4 Compute `tracks[]` (one entry per unique track_id)

Sort the array by `track_id` ascending so the file is diffable.

For each track:

| Field | Computation |
|---|---|
| `track_id` | the ID itself |
| `frames_present` | `|frames(t)|` |
| `max_consecutive` | longest run of consecutive integers in `frames(t)` |
| `first_frame`, `last_frame` | min and max of `frames(t)` |
| `presence_ratio` | `frames_present / N`, rounded to 4 d.p. |
| `consecutive_ratio` | `max_consecutive / N`, rounded to 4 d.p. |
| `gap_count` | number of missing-frame *runs* strictly between `first_frame` and `last_frame`. `[0,1,2,4,5,7,8,9]` → 2 runs (frame 3 alone, frame 6 alone). |
| `dwell_frames` | count of this track's rows with `overlap_frac_of_person > 0.5` |
| `mean_overlap_frac_of_person`, `max_overlap_frac_of_person`, `overlap_stddev` | mean/max/std over this track's rows |
| `mean_distance_norm`, `min_distance_norm` | mean/min over this track's rows |
| `centroid_spread` | `hypot(stddev(centroid_x), stddev(centroid_y))` over this track's rows. Seated → small (~0.005), walker → large (~0.05). |
| `mean_displacement`, `max_displacement` | over this track's non-null displacements; both 0.0 if track has no consecutive frames. |
| `mean_score`, `min_score` | over this track's rows. Low `min_score` flags reflection/shadow false positives. |
| `mean_bbox_below_table_frac` | mean over this track's rows |
| `mean_bbox_aspect_ratio` | mean over this track's rows |

All floats rounded to 4 d.p. except scores (3 d.p.) for diff stability.

### 6.5 Assemble the output dict

```jsonc
{
  "schema_version": 2,
  "n_frames": <N>,
  "people": [...],
  "scalars": { ... },
  "tracks": [...]
}
```

Serialize with `indent=2, sort_keys=False`. UTF-8 encoding. Use
`perception_v2.json` for current 10-frame groups. Place it in the same folder
as the sampled crop images.

---

## 7. Output schema reference

Use this as the contract for readers and writers. **Every field must be
present** unless explicitly marked optional.

```jsonc
{
  "schema_version": 2,                  // int, current = 2
  "n_frames": 10,                       // int, in [2, 16]

  "people": [
    {
      "frame_index": 0,                 // int 0..n_frames-1
      "track_id": "t0",                 // string "tN"
      "centroid_x": 0.4231,             // float 0..1
      "centroid_y": 0.6102,
      "bbox_xyxy": [320.0, 410.0, 480.5, 690.2],   // [x1, y1, x2, y2] full-frame px
      "overlap_frac_of_person": 0.7812, // float 0..1
      "overlap_frac_of_table":  0.3104, // float 0..1
      "distance_norm":          0.0823, // float 0+
      "displacement_from_prev": null,   // float or null
      "score":                  0.943,  // float 0..1
      "bbox_below_table_frac":  0.42,   // float 0..1
      "bbox_aspect_ratio":      1.18    // float 0+
    }
    // ... zero or more rows; can be empty if no people overlapped this table
  ],

  "scalars": {
    "person_count":                [...],   // int[N]
    "max_overlap_frac_of_person":  [...],   // float[N]
    "overlap_sum":                 [...],   // float[N]
    "min_distance_norm":           [...],   // float[N], default 1.0 when no people
    "mean_displacement":           0.0019,  // float, 0.0 if no displacements
    "max_displacement":            0.0021,  // float, 0.0 if no displacements
    "entering":                    [...],   // int[N-1], gap-filled
    "leaving":                     [...],   // int[N-1], gap-filled
    "raw_entering":                [...],   // int[N-1]
    "raw_leaving":                 [...],   // int[N-1]
    "persistent_count":            1,       // int >= 0
    "mostly_persistent_count":     1,       // int >= 0
    "transient_count":             0,       // int >= 0
    "max_track_tenure":            10,      // int 0..N
    "max_consecutive_tenure":      10,      // int 0..N
    "frames_with_any_person":      10,      // int 0..N
    "frames_with_dwelling_person": 10,      // int 0..N
    "primary_track_id":            "t0",    // string or null
    "primary_track_dwell_frames":  10,      // int 0..N
    "disjoint_track_count":        0        // int >= 0
  },

  "tracks": [
    {
      "track_id":                    "t0",   // string, sorted ascending in array
      "frames_present":              10,     // int 1..N
      "max_consecutive":             10,     // int 1..N
      "first_frame":                 0,      // int 0..N-1
      "last_frame":                  9,      // int 0..N-1
      "presence_ratio":              1.0,    // float 0..1
      "consecutive_ratio":           1.0,    // float 0..1
      "gap_count":                   0,      // int >= 0
      "dwell_frames":                10,     // int 0..N
      "mean_overlap_frac_of_person": 0.795,
      "max_overlap_frac_of_person":  0.81,
      "overlap_stddev":              0.0102,
      "mean_distance_norm":          0.074,
      "min_distance_norm":           0.07,
      "centroid_spread":             0.008,
      "mean_displacement":           0.006,  // 0.0 if track has < 2 frames
      "max_displacement":            0.012,
      "mean_score":                  0.94,
      "min_score":                   0.93,
      "mean_bbox_below_table_frac":  0.41,
      "mean_bbox_aspect_ratio":      1.20
    }
  ]
}
```

Empty cases:
- No people overlap the table → `people: []`, `tracks: []`,
  `primary_track_id: null`, `primary_track_dwell_frames: 0`, all per-frame
  lists are zeros (or 1.0 for `min_distance_norm`).

---

## 8. Backwards compatibility

Older schema-v1 files (no `schema_version`, no `n_frames`, no `tracks[]`,
N=3 implied) are still produced by some legacy data on Drive. Readers should:

- Default `n_frames` to `len(scalars.person_count)` if missing, or 3 if the
  file is otherwise pre-schema.
- Tolerate variable-length lists rather than indexing `[0]`/`[1]`/`[2]`.
- Treat new fields as optional and substitute reasonable defaults when
  absent (e.g. `mostly_persistent_count` defaults to `persistent_count`,
  `tracks` defaults to `[]`, `bbox_below_table_frac` defaults to 0.0).

When you bump the schema again:
- Increment `schema_version`.
- Document new fields in this guide.
- Make changes additive when possible. If breaking, accept both versions
  on the read side until all writers are upgraded.

---

## 9. Validation harness

Before shipping a new producer, run this end-to-end test. Construct
synthetic detections to cover every code path; assert on the output:

```python
def make_detection(track_id, cx, cy, bbox, score=0.9, H=720, W=1280):
    """Build a minimal person dict with a bbox-shaped mask."""
    import numpy as np
    mask = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask[y1:y2, x1:x2] = True
    return {
        "mask": mask, "bbox_xyxy": list(bbox),
        "centroid_norm": (cx, cy), "score": score, "track_id": track_id,
    }

# Table at the center, with bottom edge at y=435.
table_polygon = [(540, 285), (740, 285), (740, 435), (540, 435)]

# Three tracks across N=10:
#   t0 — seated, present in all 10 frames
#   t1 — flaky-seated, present in [0,1,2,4,5,6,7,8,9] (missing frame 3)
#   t2 — waiter, present in frame 4 only, bbox entirely above table
seated_bbox = (560, 290, 720, 470)            # mostly inside table; some legs below
waiter_bbox = (560, 100, 620, 420)            # entirely above table_max_y=435

frames = []
for i in range(10):
    persons = [make_detection("t0", 0.49, 0.55, seated_bbox)]
    if i != 3:
        persons.append(make_detection("t1", 0.51, 0.56, (565, 292, 725, 472)))
    if i == 4:
        persons.append(make_detection("t2", 0.55, 0.30, waiter_bbox))
    frames.append(persons)

result = build_perception_for_table(frames, table_polygon, (720, 1280), n_frames=10)

S = result["scalars"]
assert result["schema_version"] == 2
assert result["n_frames"] == 10
assert len(S["person_count"]) == 10
assert len(S["entering"])     == 9
assert len(S["leaving"])      == 9

assert S["persistent_count"]        == 1   # only t0 in all 10 frames
assert S["mostly_persistent_count"] == 2   # t0 + t1 (>= 8/10)
assert S["transient_count"]         == 1   # t2 (1 frame <= 2/10)

# Gap-filled enter/leave should hide t1's flake but show the real waiter event:
assert S["entering"][3] == 1 and sum(S["entering"]) == 1   # t2 entering frame 4
assert S["leaving"][4]  == 1 and sum(S["leaving"])  == 1   # t2 leaving frame 5

# Raw should expose the t1 flake at frames 2→3 and 3→4:
assert S["raw_entering"][3] == 2   # t1 reappear + t2 enter
assert S["raw_leaving"][2]  == 1   # t1 disappear

# Per-track summaries:
t0 = next(t for t in result["tracks"] if t["track_id"] == "t0")
t1 = next(t for t in result["tracks"] if t["track_id"] == "t1")
t2 = next(t for t in result["tracks"] if t["track_id"] == "t2")
assert t0["frames_present"]            == 10 and t0["gap_count"] == 0
assert t1["frames_present"]            ==  9 and t1["gap_count"] == 1
assert t2["frames_present"]            ==  1
assert t2["mean_bbox_below_table_frac"] == 0.0    # waiter entirely above table
assert t0["mean_bbox_below_table_frac"]  > 0.0    # seated has legs below
assert S["primary_track_id"]           == "t0"
```

Also test the **N=3 backwards-compat path** with the same builder — ensure
list lengths are 3 / 2 and `persistent_count` matches the strict semantic.

---

## 10. Tunables — consolidated

All knobs in one place. Match these values exactly for compatibility with our
reference implementation.

| Constant | Value | What it does |
|---|---|---|
| `MIN_FRAMES_PER_GROUP` | 2 | Reject groups smaller than this |
| `MAX_FRAMES_PER_GROUP` | 16 | Reject groups larger than this |
| `FRAME_INTERVAL_SECONDS` | 3 | Sampling interval (informational; producer must enforce) |
| `PERSISTENT_THRESHOLD_FRAC` | 1.0 | Strict: present in every frame |
| `MOSTLY_THRESHOLD_FRAC` | 0.8 | "Mostly persistent": ceil(0.8N) frames |
| `TRANSIENT_THRESHOLD_FRAC` | 0.2 | "Transient": floor(0.2N) frames |
| `DWELL_OVERLAP_THRESHOLD` | 0.5 | overlap fraction for "dwelling" |
| `MAX_GAP_TO_FILL` | 1 | gap-fill window for entering/leaving |
| `DISJOINT_TRACK_DISTANCE_THRESHOLD` | 0.1 | normalized centroid distance for disjoint_track_count |
| `TRACK_LOOKBACK` | 3 | K — frames considered as candidates in tracking |
| `TRACK_MAX_DIST` | 0.2 | max normalized centroid distance per inter-frame gap |
| `TRACK_IOU_BONUS` | 0.5 | weight added to track-match score for bbox IoU |
| YOLO model | `yolov8m-seg.pt` | medium variant; nano/small are faster, less accurate |
| YOLO confidence | 0.15 | per-detection minimum |
| YOLO classes | `[0]` | person only |

---

## 11. Output handoff conventions

- **Filename:** `perception_v2.json` for current 10-frame groups. Legacy
  readers may still accept `perception.json` or `perception_10frame.json`, but
  new writers should not upload those names.
- **Location:** same directory as the sampled crop images.
- **Encoding:** UTF-8.
- **Pretty-printed:** `indent=2`. Pretty-printing is for human debuggability;
  readers must accept any whitespace.
- **Atomic write recommended:** write to `perception_v2.json.tmp` then rename.
  Half-written files crash readers.
- **Idempotent:** producing the same input twice should produce
  byte-identical output (modulo Python `numpy` non-determinism in float
  reductions; round to 4 d.p. before serializing).

---

## 12. What this file is *not*

To avoid scope creep, here's what **must not** go into perception.json,
even if tempting:

- ❌ Object detections (cups, dishes, bottles, cutlery). Those belong to the
  dirty-vs-clean signal computed elsewhere, in a separate file if needed.
- ❌ Action classes (eating, drinking, signing check). Out of scope.
- ❌ Pose / keypoint data. The bbox + bbox_below_table_frac + aspect ratio
  are sufficient for sitting-vs-standing.
- ❌ Cropped table image embeddings. Those go in a separate feature file.
- ❌ Time-of-day, camera identity, restaurant identity. Those go in
  `metadata.json` (also adjacent to the frames).
- ❌ Labels of any kind. Labels are encoded by which Drive folder the files
  end up in.

Keep this file single-purpose: **person presence and behavior near this
specific table, across N frames.**

---

## 13. Recommended ML usage

Notes for downstream model training, not requirements on the producer:

- **Primary occupied features:** `mostly_persistent_count`,
  `max_consecutive_tenure`, `frames_with_dwelling_person`,
  `primary_track_dwell_frames`.
- **Primary waiter features:** `transient_count`, per-track
  `mean_displacement`, `centroid_spread`, low `mean_bbox_below_table_frac`,
  high `mean_bbox_aspect_ratio`.
- **Noise indicators (let the model learn around them):** `gap_count`,
  `disjoint_track_count`, `min_score`, the gap between `entering` and
  `raw_entering`.
- **High-precision sit signal:** `bbox_below_table_frac` consistently > 0.3
  across a `mostly_persistent` track.
- Treat `persistent_count` as **high-confidence but low-recall.** Prefer
  `mostly_persistent_count` for the main feature.
