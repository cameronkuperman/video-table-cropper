-- Add columns for asynchronous Drive export tracking.
-- The label endpoint now returns immediately; a background worker
-- picks up items with export_status='pending' and processes them.

ALTER TABLE video_review_items
    ADD COLUMN IF NOT EXISTS export_status TEXT NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS export_error TEXT NULL,
    ADD COLUMN IF NOT EXISTS export_attempts INTEGER NOT NULL DEFAULT 0;

-- Constrain valid states at the database level.
ALTER TABLE video_review_items
    ADD CONSTRAINT chk_export_status
        CHECK (export_status IN ('none', 'pending', 'in_progress', 'completed', 'failed'));

-- Index for the background worker's claim query.
CREATE INDEX IF NOT EXISTS idx_video_review_items_export_pending
    ON video_review_items (export_status, export_attempts)
    WHERE export_status IN ('pending', 'failed') AND export_attempts < 3;
