PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS titles (
    id TEXT PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    notes_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS title_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_heading TEXT,
    line_start INTEGER,
    line_end INTEGER,
    context_excerpt TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (title_id) REFERENCES titles(id) ON DELETE CASCADE,
    UNIQUE(title_id, source_path, line_start, line_end)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    media_type TEXT NOT NULL DEFAULT 'mixed',
    primary_path TEXT NOT NULL UNIQUE,
    signature TEXT NOT NULL DEFAULT '',
    preview_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifact_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    member_path TEXT NOT NULL,
    member_kind TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
    UNIQUE(artifact_id, member_path)
);

CREATE TABLE IF NOT EXISTS fragments (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_title TEXT NOT NULL DEFAULT '',
    source_heading TEXT,
    excerpt TEXT NOT NULL DEFAULT '',
    context_before TEXT,
    context_after TEXT,
    line_start INTEGER,
    line_end INTEGER,
    normalized_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS constellations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL DEFAULT '',
    invocation TEXT,
    state TEXT NOT NULL DEFAULT 'latent',
    body_reading TEXT NOT NULL DEFAULT '',
    mind_reading TEXT NOT NULL DEFAULT '',
    spirit_reading TEXT NOT NULL DEFAULT '',
    source_note_path TEXT NOT NULL DEFAULT '',
    promoted_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, relation_type, target_id)
);

CREATE TABLE IF NOT EXISTS session_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sandbox_items (
    id TEXT PRIMARY KEY,
    item_kind TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_id TEXT,
    source_kind TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recent_summons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    selected_id TEXT,
    selected_kind TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_title_occurrences_title_id ON title_occurrences(title_id);
CREATE INDEX IF NOT EXISTS idx_title_occurrences_source_path ON title_occurrences(source_path);
CREATE INDEX IF NOT EXISTS idx_artifact_members_artifact_id ON artifact_members(artifact_id);
CREATE INDEX IF NOT EXISTS idx_fragments_source_path ON fragments(source_path);
CREATE INDEX IF NOT EXISTS idx_fragments_hash ON fragments(normalized_hash);
CREATE INDEX IF NOT EXISTS idx_constellations_state ON constellations(state);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id, source_kind);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id, target_kind);
CREATE INDEX IF NOT EXISTS idx_sandbox_items_sort_order ON sandbox_items(sort_order);
CREATE INDEX IF NOT EXISTS idx_recent_summons_created_at ON recent_summons(created_at DESC);
