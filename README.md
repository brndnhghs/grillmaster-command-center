# GRILLMASTER Command Center

GRILLMASTER Command Center is an interactive production hub for the semi-autonomous GRILLMASTER multimedia art project. It organizes artifacts and statements from the GRILLMASTER Obsidian vault, supports discovery and development, and helps move source material toward stronger outputs.

## Prototype status

This repository is currently in an architectural transition.

- The old `tabs/` and `utils/` prototype modules are still present for reference.
- The new target architecture is a persistent shell built around vault canon, a derived local index, and restorable session state.
- Current work is focused on replacing the tabbed dashboard entrypoint with a production-oriented constellation-engine bootstrap.

## Current direction

The recovered architecture centers on:

- a single-page Streamlit shell for production, discovery, and synthesis
- three-pane navigation for discovering, developing, and previewing work
- local cache and SQLite-backed derived state
- vault-backed constellation promotion under `Grillmaster/Constellations/`

## Product intent

The app should not behave like a passive archive or generic dashboard. It should function as a creative operations console for Grillmaster: a place to surface meaningful artifacts and statements, organize them into working sets, discover productive combinations, shape drafts, and stimulate progress through interaction.

In practical terms, the app should help answer:

- What has already been made?
- What matters most right now?
- What belongs together?
- What should be developed next?
- How do we turn raw material into a stronger finished output?

This repo should be treated as an in-progress rebuild rather than a finished app.
