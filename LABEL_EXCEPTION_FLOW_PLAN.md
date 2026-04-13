# Label Exception Flow Plan

## Goal

Add a dedicated action in the labeler for triplets that should not be assigned to `clean`, `dirty`, or `occupied`.

This is for cases like:
- bad segmentation
- unusable crops
- broken or incomplete triplets
- uncertain items that should be reviewed later instead of skipped forever

The user-facing goal is to replace repeated skipping with an explicit destination such as `trash` or `label_later`.

## Recommended Product Decision

Use two separate destinations instead of one:

- `label_later`
  - for valid-looking triplets that need manual review later
  - keeps uncertain data recoverable and easy to revisit

- `trash`
  - for bad segmentation, unusable crops, corrupted triplets, or obvious junk
  - keeps low-quality data out of the active labeling loop

If we want the smallest first implementation, start with only `trash`.

## Drive / Data Model Changes

No Google Cloud reconfiguration should be needed.

Expected Drive changes:
- auto-create one or two additional subfolders under the existing project root
  - `label_later`
  - optionally `trash`
- treat them the same way the app already treats `clean`, `dirty`, and `occupied`

The existing move-on-label behavior can be reused:
- source: `unlabeled`
- destination: `label_later` or `trash`

## Backend Implementation Plan

### 1. Folder bootstrap

Update the folder bootstrap logic in `app.py` so `_folder_ids()` ensures the new destination folders exist:
- add `label_later`
- optionally add `trash`

Also update any comments or docs that currently describe the fixed 6-folder layout.

### 2. Label API support

Extend `POST /api/label` to accept the new label values:
- `label_later`
- optionally `trash`

Validation changes:
- update the allowed label set
- map each new label to its destination folder ID

Behavior should stay the same otherwise:
- move the folder out of `unlabeled`
- remove it from in-memory unlabeled queue caches
- return the chosen destination in the response

### 3. Stats support

Extend `/api/stats` to return counts for the new folders.

Frontend stats display can then show:
- unlabeled
- clean
- dirty
- occupied
- label later
- optionally trash

### 4. Queue behavior

No major queue logic change should be required.

The current queue already removes labeled folders after a successful move, so once a triplet is sent to `label_later` or `trash` it should naturally leave the active queue.

## Frontend Implementation Plan

### 1. Add new button(s)

Add one or two new actions beside the existing label buttons:
- `Label Later`
- optionally `Trash`

Suggested shortcuts:
- `4` = `Label Later`
- `5` = `Trash`

The new button should visually read as non-primary but still intentional.

Suggested styles:
- `Label Later`: muted blue or slate
- `Trash`: darker gray or desaturated red

### 2. Update keyboard handling

Extend the existing keydown handler so the new actions are as fast as the current label flow.

### 3. Update optimistic stats

Extend the local optimistic stats update logic so these new destinations update counts immediately in the UI without forcing a stats refresh.

### 4. Update helper text

Update the footer hint text so all shortcuts are visible.

Example:
- `1 Occupied`
- `2 Dirty`
- `3 Clean`
- `4 Later`
- `5 Trash`
- `Right Arrow / Space Skip`

## UX Recommendations

### Option A: Replace skip for bad data

Keep `Skip` for “not now”, but train operators to use:
- `Label Later` for uncertain-but-usable items
- `Trash` for bad segmentation / junk

This is the safest workflow because skip still exists as a temporary navigation tool.

### Option B: Make skip less prominent

If the team wants stricter data hygiene:
- keep skip available
- visually demote it
- encourage operators to classify every item into a real destination whenever possible

### Option C: Add an undo window later

If accidental trashing is a concern, later we can add:
- a client-side undo for the last move
- or a dedicated review page for `label_later` / `trash`

That should be treated as a separate follow-up feature, not part of the first implementation.

## Recommended Implementation Order

1. Add `label_later` only
2. Add backend support in `app.py`
3. Add the new button + shortcut in `static/app.js`
4. Update stats and hint text
5. Test Drive moves and queue removal
6. Decide whether `trash` should be added as a second destination

This keeps the first pass small and reduces the chance of adding too many operator choices at once.

## Edge Cases To Handle

- folder move fails after optimistic UI advance
- multi-user sessions where stats are only approximately current
- previously skipped items that remain in the queue
- malformed folders that should maybe go directly to `trash`
- accidental use of the new destination on a valid item

## Testing Checklist

- labeling to `label_later` moves the folder out of `unlabeled`
- the item disappears from the current queue immediately
- stats update correctly in UI and backend response
- refresh does not bring the item back
- keyboard shortcut triggers the same move path as the button
- `skip` still works and does not move anything
- cache/prewarm behavior is unaffected by the new destination

## Tradeoffs

- More destination folders means slightly more operator complexity
- `label_later` can become a dumping ground if there is no later review process
- `trash` is operationally useful, but mistakes are harder to notice unless there is an audit path
- Stats/header area may get more crowded
- Any future analytics that assume only `clean`, `dirty`, and `occupied` will need to be updated

## Follow-Up Ideas

- dedicated review mode for `label_later`
- bulk move from `label_later` back into `unlabeled`
- automatic routing of obviously malformed triplets into `trash`
- audit log of the last N label actions
