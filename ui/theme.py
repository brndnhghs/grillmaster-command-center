"""Theme and CSS helpers for the GRILLMASTER shell."""

from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    """Apply the shell theme for the constellation-engine bootstrap."""
    st.markdown(
        """
        <style>
            :root {
                --gm-bg: #0b0c0f;
                --gm-panel: #13161b;
                --gm-panel-alt: #171b21;
                --gm-border: #2a313c;
                --gm-text: #ece6d9;
                --gm-muted: #9f9383;
                --gm-accent: #d4541a;
                --gm-accent-soft: rgba(212, 84, 26, 0.16);
                --gm-mind: #7f5af0;
                --gm-spirit: #2cb1bc;
            }

            .stApp {
                background:
                    radial-gradient(circle at top, rgba(212, 84, 26, 0.12), transparent 30%),
                    linear-gradient(180deg, #0d1015 0%, var(--gm-bg) 100%);
                color: var(--gm-text);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.6rem;
                max-width: 100%;
            }

            .gm-shell-header {
                margin-bottom: 1rem;
                padding: 1rem 1.15rem 0.8rem;
                border: 1px solid var(--gm-border);
                border-radius: 12px;
                background: linear-gradient(180deg, rgba(19, 22, 27, 0.96), rgba(14, 16, 20, 0.96));
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
            }

            .gm-kicker {
                color: var(--gm-accent);
                text-transform: uppercase;
                letter-spacing: 0.28em;
                font-size: 0.72rem;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }

            .gm-title {
                color: var(--gm-text);
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 2.3rem;
                font-weight: 900;
                line-height: 1;
                margin: 0;
            }

            .gm-subtitle {
                margin-top: 0.45rem;
                color: var(--gm-muted);
                font-size: 0.95rem;
                max-width: 70rem;
            }

            .gm-shell-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 0.3rem;
            }

            .gm-pane {
                min-height: 31rem;
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid var(--gm-border);
                background: linear-gradient(180deg, var(--gm-panel-alt), var(--gm-panel));
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
            }

            .gm-pane-label {
                color: var(--gm-accent);
                text-transform: uppercase;
                letter-spacing: 0.22em;
                font-size: 0.72rem;
                font-weight: 800;
                margin-bottom: 0.55rem;
            }

            .gm-pane-title {
                color: var(--gm-text);
                font-size: 1.2rem;
                font-weight: 800;
                margin-bottom: 0.4rem;
            }

            .gm-pane-copy,
            .gm-list,
            .gm-note {
                color: var(--gm-muted);
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .gm-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin: 0.9rem 0 1rem;
            }

            .gm-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                border: 1px solid rgba(212, 84, 26, 0.35);
                background: var(--gm-accent-soft);
                color: var(--gm-text);
                padding: 0.28rem 0.65rem;
                font-size: 0.75rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                font-weight: 700;
            }

            .gm-rule {
                height: 1px;
                margin: 0.85rem 0;
                border: none;
                background: linear-gradient(90deg, var(--gm-accent), transparent 78%);
            }

            .gm-note {
                border-left: 3px solid var(--gm-accent);
                padding-left: 0.8rem;
            }

            .gm-list {
                margin: 0.35rem 0 0;
                padding-left: 1rem;
            }

            .gm-list li {
                margin-bottom: 0.35rem;
            }

            .gm-flow-card {
                margin: 0.9rem 0 1rem;
                padding: 0.85rem 0.9rem;
                border-radius: 12px;
                border: 1px solid rgba(212, 84, 26, 0.26);
                background: linear-gradient(180deg, rgba(212, 84, 26, 0.08), rgba(255, 255, 255, 0.02));
            }

            .gm-flow-title {
                color: var(--gm-text);
                font-size: 0.95rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                margin-bottom: 0.65rem;
                text-transform: uppercase;
            }

            .gm-flow-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.55rem;
                align-items: stretch;
            }

            .gm-flow-step {
                border-radius: 10px;
                border: 1px solid var(--gm-border);
                background: rgba(11, 12, 15, 0.55);
                padding: 0.7rem;
                min-height: 8.4rem;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }

            .gm-flow-step strong {
                color: var(--gm-text);
                font-size: 0.84rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .gm-flow-step span {
                color: var(--gm-muted);
                font-size: 0.84rem;
                line-height: 1.45;
            }

            .gm-flow-arrow {
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--gm-accent);
                font-size: 1.35rem;
                font-weight: 900;
            }

            .gm-flow-note {
                margin-top: 0.7rem;
                padding-left: 0.8rem;
                border-left: 3px solid var(--gm-accent);
                color: var(--gm-muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .gm-flow-detail {
                margin-top: 0.7rem;
                border-radius: 10px;
                border: 1px solid var(--gm-border);
                background: rgba(11, 12, 15, 0.48);
                padding: 0.75rem 0.85rem;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }

            .gm-flow-detail strong {
                color: var(--gm-text);
                font-size: 0.88rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .gm-flow-detail span {
                color: var(--gm-muted);
                font-size: 0.87rem;
                line-height: 1.5;
            }

            .gm-bms {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.5rem;
                margin-top: 1rem;
            }

            .gm-bms-card {
                border: 1px solid var(--gm-border);
                border-radius: 10px;
                padding: 0.7rem;
                background: rgba(255, 255, 255, 0.02);
            }

            .gm-bms-card strong {
                display: block;
                color: var(--gm-text);
                margin-bottom: 0.25rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.78rem;
            }

            .gm-bms-card span {
                color: var(--gm-muted);
                font-size: 0.84rem;
            }

            .gm-bms-card.body strong { color: var(--gm-accent); }
            .gm-bms-card.mind strong { color: #a68cff; }
            .gm-bms-card.spirit strong { color: #67d5de; }

            div[data-testid="stButton"] > button {
                width: 100%;
                border-radius: 10px;
                border: 1px solid var(--gm-border);
                background: rgba(212, 84, 26, 0.10);
                color: var(--gm-text);
                font-weight: 700;
            }

            div[data-testid="stButton"] > button:hover {
                border-color: var(--gm-accent);
                color: var(--gm-accent);
                background: rgba(212, 84, 26, 0.16);
            }

            @media (max-width: 1100px) {
                .gm-shell-grid {
                    grid-template-columns: 1fr;
                }

                .gm-pane {
                    min-height: auto;
                }

                .gm-flow-grid {
                    grid-template-columns: 1fr;
                }

                .gm-flow-arrow {
                    transform: rotate(90deg);
                    min-height: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
