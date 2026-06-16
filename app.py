"""Grillmaster Command Center — streamlined.
   Scan vault → search → collect → assemble → promote.
   Now with pipeline generation."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from core.bms import score_text
from core.config import VAULT_ROOT, SQLITE_DB_PATH
from core.models import SummonResult
from index.query import search_index as summon
from vault.constellations import (
    render_constellation_markdown,
    write_constellation_note,
    discover_constellations,
)

# ── helpers ──────────────────────────────────────────────────────────────

CACHE_DIR = Path(str(VAULT_ROOT or ".")) / ".cache"


def _refresh_index():
    from index.build import refresh_index

    result = refresh_index()
    counts = result.counts or {}
    st.toast(f"Indexed {counts.get('titles', 0)} titles, "
             f"{counts.get('artifacts', 0)} artifacts, "
             f"{counts.get('fragments', 0)} fragments")
    st.session_state["gm_last_refresh"] = counts
    return counts


def _init():
    st.session_state.setdefault("gm_tray", [])
    st.session_state.setdefault("gm_draft_title", "")
    st.session_state.setdefault("gm_draft_summary", "")
    st.session_state.setdefault("gm_draft_state", "latent")
    st.session_state.setdefault("gm_draft_items", [])  # [(kind, id, label)]
    st.session_state.setdefault("gm_last_refresh", None)
    st.session_state.setdefault("gm_preview_md", None)
    st.session_state.setdefault("gm_pipeline_preview", None)  # (arr, method_id, params, seed)


def _collect(item: dict[str, Any]):
    key = (item["kind"], item["id"])
    existing = {(i["kind"], i["id"]) for i in st.session_state["gm_tray"]}
    if key not in existing:
        st.session_state["gm_tray"].append(item)
        st.toast(f"Collected: {item.get('label', item['id'])}")


def _add_to_draft(item: dict[str, Any]):
    key = (item["kind"], item["id"])
    existing = {(i["kind"], i["id"]) for i in st.session_state["gm_draft_items"]}
    if key not in existing:
        st.session_state["gm_draft_items"].append(item)
        st.toast(f"Added to draft: {item.get('label', item['id'])}")


def _promote():
    draft_items = st.session_state["gm_draft_items"]
    from core.ids import make_fragment_id

    payload = {
        "title": st.session_state["gm_draft_title"] or "Untitled",
        "state": st.session_state["gm_draft_state"],
        "summary": st.session_state["gm_draft_summary"],
        "title_ids": [i["id"] for i in draft_items if i["kind"] == "title"],
        "artifact_ids": [i["id"] for i in draft_items if i["kind"] == "artifact"],
        "fragment_ids": [i["id"] for i in draft_items if i["kind"] == "fragment"],
        "unresolved": [i for i in draft_items if i["kind"] not in ("title", "artifact", "fragment")],
    }
    if not payload["title_ids"] and not payload["artifact_ids"] and not payload["fragment_ids"]:
        st.error("Add at least one item to the draft before promoting.")
        return
    try:
        result = write_constellation_note(payload, vault_root=VAULT_ROOT)
        st.success(f"Promoted → {result.path}")
        _reset_draft()
    except ValueError as exc:
        st.error(str(exc))


def _reset_draft():
    st.session_state["gm_draft_title"] = ""
    st.session_state["gm_draft_summary"] = ""
    st.session_state["gm_draft_state"] = "latent"
    st.session_state["gm_draft_items"] = []
    st.session_state["gm_preview_md"] = None


# ── UI components ────────────────────────────────────────────────────────

def _tab_search():
    """Search the vault index."""
    query = st.text_input("Summon", key="gm_search_q", placeholder="search titles, artifacts, fragments…")
    if not query:
        return

    results: list[SummonResult] = summon(query, db_path=SQLITE_DB_PATH)
    if not results:
        st.caption("Nothing found.")
        return

    st.caption(f"{len(results)} result(s)")
    for r in results:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{r.label}**  `{r.kind}`")
            if r.description:
                st.caption(r.description[:200])
            if r.snippet:
                st.caption(r.snippet[:200])
        with col2:
            if st.button("Collect", key=f"col_{r.kind}_{r.id}"):
                _collect(asdict(r))
        with col3:
            if st.button("→ Draft", key=f"dft_{r.kind}_{r.id}"):
                _add_to_draft(asdict(r))
        st.divider()


def _tab_browse():
    """Browse vault contents by kind."""
    kind = st.selectbox("Kind", ["all", "title", "artifact", "fragment", "constellation"])
    results: list[SummonResult] = summon("", db_path=SQLITE_DB_PATH)
    filtered = [r for r in results if kind == "all" or r.kind == kind]
    if not filtered:
        st.info("Refresh the index first using the scan button above.")
        return
    st.caption(f"{len(filtered)} item(s)")
    for r in filtered[:100]:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{r.label}**  `{r.kind}`")
            if r.snippet:
                st.caption(r.snippet[:200])
        with col2:
            if st.button("Collect", key=f"bc_{r.kind}_{r.id}"):
                _collect(asdict(r))
        with col3:
            if st.button("→ Draft", key=f"bd_{r.kind}_{r.id}"):
                _add_to_draft(asdict(r))
        st.divider()


def _tab_generate():
    """Generate images using the pipeline, preview, and promote to vault."""
    from pipeline_bridge.catalog import list_categories, list_methods, get_method
    from pipeline_bridge.generate import generate
    from pipeline_bridge.promote import promote_still

    st.markdown("### Generate")

    # ── Method selector ──
    categories = list_categories()
    cat = st.selectbox("Category", ["all"] + categories, key="gm_gen_cat")

    methods = list_methods(category=cat if cat != "all" else None)
    if not methods:
        st.info("No methods found.")
        return

    method_options = {f"{m['id']}  {m['name']}": m for m in methods}
    selected_label = st.selectbox(
        "Method",
        list(method_options.keys()),
        key="gm_gen_method",
    )
    method = method_options[selected_label]
    method_id = method["id"]

    # ── Parameter controls ──
    params = method.get("params", {})
    param_values = {}
    if params:
        st.markdown("**Parameters**")
        for pname, pspec in params.items():
            choices = pspec.get("choices")
            default = pspec.get("default", "")
            if choices:
                param_values[pname] = st.selectbox(
                    pname, choices,
                    index=choices.index(default) if default in choices else 0,
                    key=f"gm_p_{pname}",
                )
            elif "min" in pspec and "max" in pspec:
                v_min = float(pspec["min"])
                v_max = float(pspec["max"])
                v_default = float(pspec.get("default", (v_min + v_max) / 2))
                param_values[pname] = st.slider(
                    pname, v_min, v_max, v_default,
                    key=f"gm_p_{pname}",
                )
            elif isinstance(default, bool):
                param_values[pname] = st.checkbox(pname, value=default, key=f"gm_p_{pname}")
            else:
                param_values[pname] = st.text_input(pname, value=str(default), key=f"gm_p_{pname}")

    # ── Seed ──
    seed = st.number_input("Seed", value=42069, step=1, key="gm_gen_seed")

    # ── Action buttons ──
    col1, col2, col3 = st.columns(3)
    with col1:
        gen_clicked = st.button("Generate", use_container_width=True, key="gm_gen_go")
    with col2:
        anim_clicked = st.button("Animate 5s", use_container_width=True, key="gm_gen_anim")
    with col3:
        promote_clicked = st.button("Promote → Vault", use_container_width=True, key="gm_gen_promote")

    # ── Preview area ──
    preview = st.session_state.get("gm_pipeline_preview")
    if preview:
        arr, prev_mid, prev_params, prev_seed = preview
        st.image(arr, caption=f"Method {prev_mid} — seed {prev_seed}", use_container_width=True)

    # ── Generate action ──
    if gen_clicked:
        with st.spinner("Generating..."):
            try:
                arr = generate(method_id, params=param_values or None, seed=seed)
                st.session_state["gm_pipeline_preview"] = (arr, method_id, param_values, seed)
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")

    # ── Animate action ──
    if anim_clicked:
        from pipeline_bridge.animate import animate
        with st.spinner("Animating..."):
            try:
                mp4_path = animate(method_id, params=param_values or None, seed=seed, duration=5.0, fps=10)
                st.video(mp4_path)
                # Also add to tray as a pipeline artifact
                _collect({
                    "kind": "artifact",
                    "id": f"pipeline_anim_{method_id}_{seed}",
                    "label": f"Animated {method['name']} (seed {seed})",
                    "source_path": mp4_path,
                })
            except Exception as e:
                st.error(f"Animation failed: {e}")

    # ── Promote action ──
    if promote_clicked:
        title_input = st.text_input(
            "Artifact title",
            value=f"{method['name']} — {seed}",
            key="gm_gen_title_input",
        )
        if title_input:
            with st.spinner("Promoting to vault..."):
                try:
                    result = promote_still(method_id, title_input, params=param_values or None, seed=seed)
                    # Add to tray
                    _collect({
                        "kind": "artifact",
                        "id": f"pipeline_{method_id}_{seed}",
                        "label": title_input,
                        "source_path": result["path"],
                        "description": f"Pipeline {method_id} ({method['name']}), seed {seed}",
                    })
                    st.success(f"Promoted → `{result['vault_path']}`")
                except Exception as e:
                    st.error(f"Promotion failed: {e}")


def _tray_pane():
    st.markdown("### Tray")
    items = st.session_state["gm_tray"]
    if not items:
        st.info("Collect items from the search results.")
        return
    for item in list(items):
        cols = st.columns([3, 1, 1])
        cols[0].markdown(f"**{item.get('label', item['id'])}**  `{item.get('kind')}`")
        if cols[1].button("→Draft", key=f"tr_{item['kind']}_{item['id']}"):
            _add_to_draft(item)
        if cols[2].button("✕", key=f"trx_{item['kind']}_{item['id']}"):
            st.session_state["gm_tray"].remove(item)
            st.rerun()
    if st.button("Clear Tray"):
        st.session_state["gm_tray"] = []
        st.rerun()


def _draft_pane():
    st.markdown("### Draft")
    st.text_input("Title", key="gm_draft_title", placeholder="Constellation title")
    st.text_area("Summary", key="gm_draft_summary", placeholder="One-line summary")
    st.selectbox("State", ["latent", "manifested", "stalled"], key="gm_draft_state")

    draft_items = st.session_state["gm_draft_items"]
    if not draft_items:
        st.caption("Add items from the tray or search results.")
    else:
        st.caption(f"{len(draft_items)} item(s) in draft")
        for item in list(draft_items):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"- {item.get('label', item['id'])}  `{item.get('kind')}`")
            if c2.button("✕", key=f"drx_{item['kind']}_{item['id']}"):
                st.session_state["gm_draft_items"].remove(item)
                st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Preview"):
        _preview_draft()
    if c2.button("Promote → Vault"):
        _promote()

    preview_md = st.session_state.get("gm_preview_md")
    if preview_md:
        with st.expander("Preview markdown", expanded=True):
            st.code(preview_md, language="markdown")


def _preview_draft():
    draft_items = st.session_state["gm_draft_items"]
    from core.ids import make_fragment_id

    payload = {
        "title": st.session_state["gm_draft_title"] or "Untitled",
        "state": st.session_state["gm_draft_state"],
        "summary": st.session_state["gm_draft_summary"],
        "title_ids": [i["id"] for i in draft_items if i["kind"] == "title"],
        "artifact_ids": [i["id"] for i in draft_items if i["kind"] == "artifact"],
        "fragment_ids": [i["id"] for i in draft_items if i["kind"] == "fragment"],
        "unresolved": [i for i in draft_items if i["kind"] not in ("title", "artifact", "fragment")],
    }
    try:
        md = render_constellation_markdown(payload, vault_root=VAULT_ROOT)
        st.session_state["gm_preview_md"] = md
    except ValueError as exc:
        st.error(str(exc))


# ── main ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Grillmaster", layout="wide")
    _init()
    st.title("⏦ Grillmaster Command Center")
    st.caption("Scan → search → generate → collect → assemble → promote to vault")

    # ── header bar ──
    cols = st.columns([3, 1, 1, 1])
    with cols[0]:
        st.caption(f"Vault: `{VAULT_ROOT}`")
    with cols[1]:
        if st.button("Refresh Index (full scan)", use_container_width=True):
            _refresh_index()
    with cols[2]:
        n_tray = len(st.session_state["gm_tray"])
        st.caption(f"Tray: {n_tray}")
    with cols[3]:
        n_draft = len(st.session_state["gm_draft_items"])
        st.caption(f"Draft: {n_draft}")

    st.divider()

    # ── three columns ──
    left, mid, right = st.columns([1.2, 1.2, 1.0])

    with left:
        st.markdown("### Search / Browse / Generate")
        tab_s, tab_b, tab_g = st.tabs(["Search", "Browse", "Generate"])
        with tab_s:
            _tab_search()
        with tab_b:
            _tab_browse()
        with tab_g:
            _tab_generate()

    with mid:
        _tray_pane()
        st.divider()
        _draft_pane()

    with right:
        st.markdown("### BMS Lens")
        # Show BMS balance for draft
        draft_items = st.session_state["gm_draft_items"]
        title = st.session_state["gm_draft_title"] or ""
        summary = st.session_state["gm_draft_summary"] or ""
        if title or summary or draft_items:
            body_hints = " ".join(i.get("label", "") for i in draft_items if i["kind"] == "artifact")
            mind_hints = " ".join(i.get("label", "") for i in draft_items if i["kind"] == "title")
            spirit_hints = f"{len(draft_items)} items"
            label = title or "draft"
        else:
            body_hints = ""
            mind_hints = ""
            spirit_hints = ""
            label = "draft"

        # Show BMS visualization
        if label:
            bal = score_text(title, summary, summary + " " + body_hints + " " + mind_hints, st.session_state["gm_draft_state"])
            _render_bms_indicator(bal)

        st.divider()

        # ── last refresh info ──
        last = st.session_state.get("gm_last_refresh")
        if last:
            st.caption(
                f"Index: {last.get('titles', 0)} titles · "
                f"{last.get('artifacts', 0)} artifacts · "
                f"{last.get('fragments', 0)} fragments"
            )


def _render_bms_indicator(bal) -> None:
    """Compact BMS indicator."""
    b = bal.body if hasattr(bal, 'body') else getattr(bal, 'body_score', 0)
    m = bal.mind if hasattr(bal, 'mind') else getattr(bal, 'mind_score', 0)
    s = bal.spirit if hasattr(bal, 'spirit') else getattr(bal, 'spirit_score', 0)
    st.markdown(
        f"""
        <div style="display:flex;gap:0.5rem;font-size:0.85rem;">
            <div style="background:#433;padding:0.3rem 0.6rem;border-radius:4px;flex:1;text-align:center;">
                <strong>Body</strong><br>{b:.1f}
            </div>
            <div style="background:#343;padding:0.3rem 0.6rem;border-radius:4px;flex:1;text-align:center;">
                <strong>Mind</strong><br>{m:.1f}
            </div>
            <div style="background:#334;padding:0.3rem 0.6rem;border-radius:4px;flex:1;text-align:center;">
                <strong>Spirit</strong><br>{s:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
