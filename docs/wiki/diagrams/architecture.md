# Architecture Diagram

This is the system-level flowchart for Grillmaster Command Center. The prose version with component descriptions and data-flow steps lives in [`../architecture.md`](../architecture.md).

```mermaid
flowchart TD
    User(["User / Browser"])
    Dashboard["Dashboard :7870"]
    Pipeline["Image Pipeline :7860"]
    Chord["Chord Bot :7861"]
    Sidecar["3D Sidecar :7862"]
    GraphDoc[("Shared Graph Doc")]
    JobQueue[("Job Queue")]
    Cache[("Frame Cache")]
    Disk[("Output Disk")]

    User --> Dashboard
    Dashboard -->|spawn/monitor| Pipeline
    Dashboard -->|spawn/monitor| Chord
    Dashboard -->|spawn/monitor| Sidecar
    User -.->|direct REST/SSE/WS| Pipeline
    User -.->|direct REST| Chord

    Pipeline --> GraphDoc
    Pipeline --> JobQueue
    Pipeline --> Cache
    Pipeline --> Disk
    Pipeline -->|proxied iframe| Dashboard
    Chord --> Disk
```

## Notes

- **Solid arrows** are process supervision (Dashboard spawns and monitors the services). **Dotted arrows** are direct client→server traffic — the browser talks straight to the Image Pipeline and Chord Bot, not through the Dashboard.
- The **Shared Graph Doc** is the single source of truth for the live simulation loop: the running loop re-reads it every frame, so an edited graph is absorbed without restarting.
- The **Frame Cache** is keyed by node-id + parameter hash + frame; Architecture-A simulation methods cache their full frame list here.
