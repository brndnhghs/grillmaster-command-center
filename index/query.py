from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from core.models import EntityKind, SummonResult
from index.build import open_index

VALID_KINDS: tuple[EntityKind, ...] = ("title", "artifact", "fragment", "constellation")


def _normalize_limit(limit: int | None) -> int:
    if limit is None:
        return 20
    return max(1, min(int(limit), 100))


def _normalize_kinds(kinds: Iterable[EntityKind] | None) -> tuple[EntityKind, ...]:
    if kinds is None:
        return VALID_KINDS
    normalized = tuple(kind for kind in kinds if kind in VALID_KINDS)
    return normalized or VALID_KINDS


def _tokens(query_text: str) -> list[str]:
    return [token.casefold() for token in query_text.strip().split() if token.strip()]


def _like_patterns(tokens: list[str]) -> list[str]:
    return [f"%{token}%" for token in tokens]


def _score_text(*values: str | None, tokens: list[str]) -> float:
    haystack = " ".join(value or "" for value in values).casefold()
    score = 0.0
    for token in tokens:
        if token == haystack.strip():
            score += 8.0
        if haystack.startswith(token):
            score += 3.0
        if token in haystack:
            score += 1.0
    return score


def _decode_json(payload: str | None, default):
    if not payload:
        return default
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default


def _split_grouped(value: str | None) -> list[str]:
    if not value:
        return []
    separator = "||" if "||" in value else ","
    return [item for item in dict.fromkeys(part for part in value.split(separator) if part)]



def _query_titles(connection: sqlite3.Connection, *, tokens: list[str], limit: int) -> list[SummonResult]:
    if not tokens:
        rows = connection.execute(
            """
            SELECT
                t.id,
                t.canonical_text,
                t.slug,
                t.description,
                t.notes_count,
                GROUP_CONCAT(DISTINCT o.source_path) AS grouped_source_paths,
                GROUP_CONCAT(DISTINCT o.source_heading) AS grouped_headings,
                GROUP_CONCAT(DISTINCT o.context_excerpt) AS grouped_excerpts
            FROM titles t
            LEFT JOIN title_occurrences o ON o.title_id = t.id
            GROUP BY t.id, t.canonical_text, t.slug, t.description, t.notes_count
            ORDER BY t.canonical_text COLLATE NOCASE ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        where = " AND ".join(
            [
                "(" 
                "lower(t.canonical_text) LIKE ? OR "
                "lower(t.slug) LIKE ? OR "
                "lower(COALESCE(t.description, '')) LIKE ? OR "
                "lower(COALESCE(o.source_heading, '')) LIKE ? OR "
                "lower(COALESCE(o.context_excerpt, '')) LIKE ?"
                ")"
                for _ in tokens
            ]
        )
        params: list[object] = []
        for pattern in _like_patterns(tokens):
            params.extend((pattern, pattern, pattern, pattern, pattern))
        params.append(limit)
        rows = connection.execute(
            f"""
            SELECT
                t.id,
                t.canonical_text,
                t.slug,
                t.description,
                t.notes_count,
                GROUP_CONCAT(DISTINCT o.source_path) AS grouped_source_paths,
                GROUP_CONCAT(DISTINCT o.source_heading) AS grouped_headings,
                GROUP_CONCAT(DISTINCT o.context_excerpt) AS grouped_excerpts
            FROM titles t
            LEFT JOIN title_occurrences o ON o.title_id = t.id
            WHERE {where}
            GROUP BY t.id, t.canonical_text, t.slug, t.description, t.notes_count
            ORDER BY t.canonical_text COLLATE NOCASE ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    results: list[SummonResult] = []
    for row in rows:
        source_paths = _split_grouped(row["grouped_source_paths"])
        headings = _split_grouped(row["grouped_headings"])
        excerpts = _split_grouped(row["grouped_excerpts"])
        label = row["canonical_text"]
        score = _score_text(label, row["description"], " ".join(headings), tokens=tokens)
        results.append(
            SummonResult(
                id=row["id"],
                kind="title",
                label=label,
                score=score,
                description=row["description"],
                snippet=excerpts[0] if excerpts else row["description"],
                source_path=source_paths[0] if source_paths else None,
                source_paths=source_paths,
                metadata={
                    "canonical_text": label,
                    "slug": row["slug"],
                    "notes_count": row["notes_count"],
                    "occurrence_headings": headings,
                    "occurrence_paths": source_paths,
                    "occurrence_excerpts": excerpts,
                },
                matched_terms=tokens,
            )
        )
    return results



def _query_artifacts(connection: sqlite3.Connection, *, tokens: list[str], limit: int) -> list[SummonResult]:
    if not tokens:
        rows = connection.execute(
            """
            SELECT
                a.id,
                a.title,
                a.media_type,
                a.primary_path,
                a.preview_path,
                a.metadata_json,
                GROUP_CONCAT(am.member_path, '||') AS grouped_member_paths,
                GROUP_CONCAT(am.member_kind, '||') AS grouped_member_kinds
            FROM artifacts a
            LEFT JOIN artifact_members am ON am.artifact_id = a.id
            GROUP BY a.id, a.title, a.media_type, a.primary_path, a.preview_path, a.metadata_json
            ORDER BY a.title COLLATE NOCASE ASC, a.primary_path COLLATE NOCASE ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        where = " AND ".join(
            [
                "(" 
                "lower(a.title) LIKE ? OR "
                "lower(a.primary_path) LIKE ? OR "
                "lower(a.metadata_json) LIKE ? OR "
                "lower(COALESCE(am.member_path, '')) LIKE ?"
                ")"
                for _ in tokens
            ]
        )
        params: list[object] = []
        for pattern in _like_patterns(tokens):
            params.extend((pattern, pattern, pattern, pattern))
        params.append(limit)
        rows = connection.execute(
            f"""
            SELECT
                a.id,
                a.title,
                a.media_type,
                a.primary_path,
                a.preview_path,
                a.metadata_json,
                GROUP_CONCAT(am.member_path, '||') AS grouped_member_paths,
                GROUP_CONCAT(am.member_kind, '||') AS grouped_member_kinds
            FROM artifacts a
            LEFT JOIN artifact_members am ON am.artifact_id = a.id
            WHERE {where}
            GROUP BY a.id, a.title, a.media_type, a.primary_path, a.preview_path, a.metadata_json
            ORDER BY a.title COLLATE NOCASE ASC, a.primary_path COLLATE NOCASE ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    results: list[SummonResult] = []
    for row in rows:
        decoded = _decode_json(row["metadata_json"], {})
        member_paths = _split_grouped(row["grouped_member_paths"])
        member_kinds = _split_grouped(row["grouped_member_kinds"])
        if not member_paths:
            member_paths = list(decoded.get("member_paths", []))
        source_paths = [path for path in dict.fromkeys([row["primary_path"], *decoded.get("source_paths", []), *member_paths]) if path]
        snippet = row["primary_path"]
        score = _score_text(row["title"], row["primary_path"], json.dumps(decoded, sort_keys=True), tokens=tokens)
        results.append(
            SummonResult(
                id=row["id"],
                kind="artifact",
                label=row["title"] or Path(row["primary_path"]).stem,
                score=score,
                description=f"{row['media_type']} artifact",
                snippet=snippet,
                source_path=row["primary_path"],
                source_paths=source_paths,
                metadata={
                    "title": row["title"],
                    "media_type": row["media_type"],
                    "primary_path": row["primary_path"],
                    "preview_path": row["preview_path"],
                    "member_paths": member_paths,
                    "member_kinds": member_kinds,
                    "companion_note_paths": list(decoded.get("companion_note_paths", [])),
                    "related_ids": list(decoded.get("related_ids", [])),
                },
                matched_terms=tokens,
            )
        )
    return results



def _query_fragments(connection: sqlite3.Connection, *, tokens: list[str], limit: int) -> list[SummonResult]:
    if not tokens:
        rows = connection.execute(
            """
            SELECT id, source_title, source_heading, excerpt, context_before, context_after, source_path, line_start, line_end
            FROM fragments
            ORDER BY source_path COLLATE NOCASE ASC, line_start ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        where = " AND ".join(
            [
                "(" 
                "lower(source_title) LIKE ? OR "
                "lower(COALESCE(source_heading, '')) LIKE ? OR "
                "lower(excerpt) LIKE ? OR "
                "lower(source_path) LIKE ? OR "
                "lower(COALESCE(context_before, '')) LIKE ? OR "
                "lower(COALESCE(context_after, '')) LIKE ?"
                ")"
                for _ in tokens
            ]
        )
        params: list[object] = []
        for pattern in _like_patterns(tokens):
            params.extend((pattern, pattern, pattern, pattern, pattern, pattern))
        params.append(limit)
        rows = connection.execute(
            f"""
            SELECT id, source_title, source_heading, excerpt, context_before, context_after, source_path, line_start, line_end
            FROM fragments
            WHERE {where}
            ORDER BY source_path COLLATE NOCASE ASC, line_start ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    results: list[SummonResult] = []
    for row in rows:
        label = row["source_heading"] or row["source_title"] or row["source_path"]
        score = _score_text(label, row["excerpt"], row["source_path"], row["context_before"], row["context_after"], tokens=tokens)
        results.append(
            SummonResult(
                id=row["id"],
                kind="fragment",
                label=label,
                score=score,
                description=row["source_title"],
                snippet=row["excerpt"],
                source_path=row["source_path"],
                source_paths=[row["source_path"]],
                line_start=row["line_start"],
                line_end=row["line_end"],
                metadata={
                    "heading": row["source_heading"],
                    "source_title": row["source_title"],
                    "excerpt": row["excerpt"],
                    "context_before": row["context_before"],
                    "context_after": row["context_after"],
                },
                matched_terms=tokens,
            )
        )
    return results



def _query_constellations(connection: sqlite3.Connection, *, tokens: list[str], limit: int) -> list[SummonResult]:
    if not tokens:
        rows = connection.execute(
            """
            SELECT id, title, slug, summary, invocation, state, body_reading, mind_reading, spirit_reading, source_note_path
            FROM constellations
            ORDER BY title COLLATE NOCASE ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        where = " AND ".join(
            [
                "(" 
                "lower(title) LIKE ? OR "
                "lower(summary) LIKE ? OR "
                "lower(COALESCE(invocation, '')) LIKE ? OR "
                "lower(source_note_path) LIKE ? OR "
                "lower(state) LIKE ? OR "
                "lower(body_reading) LIKE ? OR "
                "lower(mind_reading) LIKE ? OR "
                "lower(spirit_reading) LIKE ?"
                ")"
                for _ in tokens
            ]
        )
        params: list[object] = []
        for pattern in _like_patterns(tokens):
            params.extend((pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern))
        params.append(limit)
        rows = connection.execute(
            f"""
            SELECT id, title, slug, summary, invocation, state, body_reading, mind_reading, spirit_reading, source_note_path
            FROM constellations
            WHERE {where}
            ORDER BY title COLLATE NOCASE ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    results: list[SummonResult] = []
    for row in rows:
        score = _score_text(
            row["title"],
            row["summary"],
            row["invocation"],
            row["body_reading"],
            row["mind_reading"],
            row["spirit_reading"],
            tokens=tokens,
        )
        results.append(
            SummonResult(
                id=row["id"],
                kind="constellation",
                label=row["title"],
                score=score,
                description=row["summary"],
                snippet=row["invocation"] or row["summary"],
                source_path=row["source_note_path"],
                source_paths=[row["source_note_path"]] if row["source_note_path"] else [],
                metadata={
                    "title": row["title"],
                    "slug": row["slug"],
                    "summary": row["summary"],
                    "invocation": row["invocation"],
                    "state": row["state"],
                    "body_reading": row["body_reading"],
                    "mind_reading": row["mind_reading"],
                    "spirit_reading": row["spirit_reading"],
                    "source_note_path": row["source_note_path"],
                },
                matched_terms=tokens,
            )
        )
    return results



def search_index(
    query_text: str,
    *,
    kinds: Iterable[EntityKind] | None = None,
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> list[SummonResult]:
    normalized_limit = _normalize_limit(limit)
    normalized_kinds = _normalize_kinds(kinds)
    tokens = _tokens(query_text)

    with open_index(db_path) as connection:
        per_kind_limit = max(normalized_limit, 5)
        results: list[SummonResult] = []
        if "title" in normalized_kinds:
            results.extend(_query_titles(connection, tokens=tokens, limit=per_kind_limit))
        if "artifact" in normalized_kinds:
            results.extend(_query_artifacts(connection, tokens=tokens, limit=per_kind_limit))
        if "fragment" in normalized_kinds:
            results.extend(_query_fragments(connection, tokens=tokens, limit=per_kind_limit))
        if "constellation" in normalized_kinds:
            results.extend(_query_constellations(connection, tokens=tokens, limit=per_kind_limit))

    results.sort(key=lambda item: (-item.score, item.kind, item.label.casefold(), item.id))
    return results[:normalized_limit]


search_summons = search_index

__all__ = ["search_index", "search_summons"]
