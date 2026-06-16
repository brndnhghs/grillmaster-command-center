"""SQLite bootstrap and derived refresh wiring for the GRILLMASTER index."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import SQLITE_DB_PATH, VAULT_ROOT
from vault.artifacts import discover_artifacts
from vault.constellations import discover_constellations
from vault.fragments import discover_fragments
from vault.scanner import scan_vault
from vault.titles import discover_titles

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "task5-bootstrap-v1"


@dataclass(slots=True)
class IndexRefreshResult:
    """Serializable status object for bootstrap/refresh flows."""

    db_path: str
    schema_path: str
    status: str
    refreshed_at: str
    schema_applied: bool = True
    placeholder: bool = False
    counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    resolved = Path(db_path or SQLITE_DB_PATH).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def open_index(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the local sqlite index and enforce expected connection settings."""

    resolved_path = _resolve_db_path(db_path)
    connection = sqlite3.connect(resolved_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def load_schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def apply_schema(connection: sqlite3.Connection) -> None:
    """Apply the idempotent sqlite schema script."""

    connection.executescript(load_schema_sql())
    connection.commit()


def _set_session_value(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO session_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, value),
    )


def bootstrap_index(db_path: str | Path | None = None) -> IndexRefreshResult:
    """Ensure the sqlite file exists and the recovered schema has been applied."""

    started_at = _utc_now()
    resolved_path = _resolve_db_path(db_path)
    with open_index(resolved_path) as connection:
        apply_schema(connection)
        _set_session_value(connection, "schema_version", SCHEMA_VERSION)
        _set_session_value(connection, "bootstrap_completed_at", started_at)
        connection.commit()

    return IndexRefreshResult(
        db_path=str(resolved_path),
        schema_path=str(SCHEMA_PATH),
        status="SQLite schema ready for corpus refresh.",
        refreshed_at=started_at,
        placeholder=True,
        counts={},
    )


def _clear_derived_tables(connection: sqlite3.Connection) -> None:
    for table in (
        "relations",
        "artifact_members",
        "title_occurrences",
        "artifacts",
        "fragments",
        "constellations",
        "titles",
    ):
        connection.execute(f"DELETE FROM {table}")


def _insert_titles(connection: sqlite3.Connection, *, vault_root: Path) -> dict[str, int]:
    snapshot = discover_titles(vault_root)
    connection.executemany(
        """
        INSERT INTO titles (id, canonical_text, slug, description, aliases_json, notes_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.id,
                record.canonical_text,
                record.slug,
                record.description,
                json.dumps(record.aliases, sort_keys=True),
                record.notes_count,
            )
            for record in snapshot.records
        ],
    )
    connection.executemany(
        """
        INSERT INTO title_occurrences (
            title_id, source_path, source_heading, line_start, line_end, context_excerpt
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                occurrence.title_id,
                occurrence.source_path,
                occurrence.source_heading,
                occurrence.line_start,
                occurrence.line_end,
                occurrence.context_excerpt,
            )
            for occurrence in snapshot.occurrences
        ],
    )
    return {
        "titles": len(snapshot.records),
        "title_occurrences": len(snapshot.occurrences),
    }


def _artifact_metadata(record: Any) -> str:
    payload = {
        "companion_note_paths": record.companion_note_paths,
        "member_paths": record.member_paths,
        "source_paths": record.source_paths,
        "related_ids": record.related_ids,
    }
    return json.dumps(payload, sort_keys=True)


def _insert_artifacts(connection: sqlite3.Connection, *, vault_root: Path) -> dict[str, int]:
    snapshot = discover_artifacts(vault_root)
    connection.executemany(
        """
        INSERT INTO artifacts (
            id, title, media_type, primary_path, signature, preview_path, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.id,
                record.title,
                record.media_type,
                record.primary_path,
                record.signature,
                record.preview_path,
                _artifact_metadata(record),
            )
            for record in snapshot.records
        ],
    )

    member_rows: list[tuple[str, str, str, int]] = []
    for record in snapshot.records:
        ordered_members = list(dict.fromkeys([record.primary_path, *record.companion_note_paths, *record.member_paths]))
        for index, member_path in enumerate(ordered_members):
            member_kind = "primary"
            if member_path in record.companion_note_paths:
                member_kind = "note"
            elif member_path in record.member_paths and member_path != record.primary_path:
                member_kind = "member"
            member_rows.append((record.id, member_path, member_kind, index))

    connection.executemany(
        """
        INSERT INTO artifact_members (artifact_id, member_path, member_kind, sort_order)
        VALUES (?, ?, ?, ?)
        """,
        member_rows,
    )
    return {
        "artifacts": len(snapshot.records),
        "artifact_members": len(member_rows),
    }


def _insert_fragments(connection: sqlite3.Connection, *, vault_root: Path) -> dict[str, int]:
    snapshot = discover_fragments(vault_root)
    connection.executemany(
        """
        INSERT INTO fragments (
            id, source_path, source_title, source_heading, excerpt,
            context_before, context_after, line_start, line_end, normalized_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.id,
                record.source_path,
                record.source_title,
                record.heading,
                record.excerpt,
                record.context_before,
                record.context_after,
                record.line_start,
                record.line_end,
                record.normalized_hash,
            )
            for record in snapshot.records
        ],
    )
    return {"fragments": len(snapshot.records)}


def _insert_constellations(connection: sqlite3.Connection, *, vault_root: Path) -> dict[str, int]:
    snapshot = discover_constellations(vault_root)
    connection.executemany(
        """
        INSERT INTO constellations (
            id, title, slug, summary, invocation, state,
            body_reading, mind_reading, spirit_reading, source_note_path, promoted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.id,
                record.title,
                record.slug,
                record.summary,
                record.invocation,
                record.state,
                record.body_reading,
                record.mind_reading,
                record.spirit_reading,
                record.source_note_path,
                record.promoted_at,
            )
            for record in snapshot.records
        ],
    )
    return {"constellations": len(snapshot.records)}


def _scan_counts(vault_root: str | Path | None = None) -> dict[str, int]:
    return scan_vault(vault_root or VAULT_ROOT).counts()


def refresh_index(
    db_path: str | Path | None = None,
    *,
    vault_root: str | Path | None = None,
) -> IndexRefreshResult:
    """Populate the local sqlite index from the vault-canonical source data."""

    bootstrap_result = bootstrap_index(db_path)
    resolved_vault_root = Path(vault_root or VAULT_ROOT).expanduser().resolve()
    counts: dict[str, int] = _scan_counts(resolved_vault_root)
    counts.update({
        "titles": 0,
        "title_occurrences": 0,
        "artifacts": 0,
        "artifact_members": 0,
        "fragments": 0,
        "constellations": 0,
    })
    completed_at = _utc_now()

    with open_index(db_path or bootstrap_result.db_path) as connection:
        apply_schema(connection)
        _clear_derived_tables(connection)
        counts.update(_insert_titles(connection, vault_root=resolved_vault_root))
        counts.update(_insert_artifacts(connection, vault_root=resolved_vault_root))
        counts.update(_insert_fragments(connection, vault_root=resolved_vault_root))
        counts.update(_insert_constellations(connection, vault_root=resolved_vault_root))

        payload = {
            "status": "placeholder_refresh_complete",
            "schema_version": SCHEMA_VERSION,
            "vault_root": str(resolved_vault_root),
            "counts": counts,
            "refreshed_at": completed_at,
        }
        _set_session_value(connection, "refresh_status", payload["status"])
        _set_session_value(connection, "refresh_payload", json.dumps(payload, sort_keys=True))
        _set_session_value(connection, "refresh_completed_at", completed_at)
        connection.commit()

    status = (
        "Index refresh complete: "
        f"{counts['titles']} titles, {counts['artifacts']} artifacts, "
        f"{counts['fragments']} fragments, {counts['constellations']} constellations "
        f"from {counts['supported']} supported files."
    )

    return IndexRefreshResult(
        db_path=bootstrap_result.db_path,
        schema_path=bootstrap_result.schema_path,
        status=status,
        refreshed_at=completed_at,
        counts=counts,
    )


if __name__ == "__main__":
    print(json.dumps(refresh_index().to_dict(), indent=2, sort_keys=True))
