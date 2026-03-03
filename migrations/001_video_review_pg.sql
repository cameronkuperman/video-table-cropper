CREATE TABLE IF NOT EXISTS video_review_sessions (
    id TEXT PRIMARY KEY,
    review_root_folder_id TEXT NOT NULL,
    pending_root_folder_id TEXT NOT NULL,
    batch_limit INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_review_items (
    id BIGSERIAL PRIMARY KEY,
    sample_id TEXT NOT NULL UNIQUE,
    sample_folder_id TEXT NOT NULL UNIQUE,
    sample_folder_name TEXT NOT NULL,
    review_root_folder_id TEXT NOT NULL,
    source_parent_folder_id TEXT NOT NULL,
    source_video_drive_file_id TEXT NOT NULL,
    source_video_name TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    table_track_id TEXT NOT NULL,
    anchor_time_seconds INTEGER NOT NULL,
    preview_anchor_file_id TEXT,
    preview_t_minus_10_file_id TEXT,
    preview_t_minus_20_file_id TEXT,
    tight_anchor_file_id TEXT,
    perception_file_id TEXT,
    sample_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'labeled', 'skipped', 'trashed')),
    label TEXT NULL CHECK (label IN ('clean', 'dirty', 'occupied') OR label IS NULL),
    exported_folder_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    archived_parent_folder_id TEXT NULL,
    claimed_by_session_id TEXT NULL,
    claim_expires_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_review_items_status_id
    ON video_review_items(status, id);

CREATE INDEX IF NOT EXISTS idx_video_review_items_claim
    ON video_review_items(claimed_by_session_id, claim_expires_at);

CREATE INDEX IF NOT EXISTS idx_video_review_items_camera_video_anchor
    ON video_review_items(camera_id, source_video_name, anchor_time_seconds);

CREATE TABLE IF NOT EXISTS video_review_actions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    queue_item_id BIGINT NOT NULL REFERENCES video_review_items(id),
    action_type TEXT NOT NULL CHECK (action_type IN ('label', 'skip', 'trash')),
    prev_status TEXT,
    new_status TEXT,
    prev_label TEXT,
    new_label TEXT,
    exported_folder_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    moved_folder_id TEXT,
    archive_parent_folder_id TEXT,
    undone BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_review_actions_session_undone
    ON video_review_actions(session_id, undone, id);

CREATE TABLE IF NOT EXISTS worker_status (
    worker_id TEXT PRIMARY KEY,
    event_seq BIGINT NOT NULL,
    state TEXT NOT NULL,
    message TEXT NOT NULL,
    worker_running BOOLEAN NOT NULL,
    current_video JSONB NULL,
    last_error TEXT NULL,
    stop_reason TEXT NULL,
    counters JSONB NOT NULL,
    config JSONB NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_videos (
    video_id TEXT PRIMARY KEY,
    video_name TEXT NOT NULL,
    source_folder_id TEXT NULL,
    source_folder_name TEXT NULL,
    created_samples INTEGER NOT NULL,
    source_video_trashed BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
