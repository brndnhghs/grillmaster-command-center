"""Tuning mode — directed brief → agent-built graph → critique → learned node-craft.

The *directed inverse* of `shootout/`: instead of generating random graphs and
learning a taste regressor from star ratings, a human brief drives the in-app
Hermes agent to deliberately build a node-graph, the human critiques it, and the
system distills durable *craft knowledge* (which nodes / params / combinations
produce which visual effects) into a growing `playbook.md` that is fed back into
every future build. See docs/plans (jiggly-floating-snowglobe) for the design.
"""
