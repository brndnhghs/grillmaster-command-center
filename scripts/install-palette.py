#!/usr/bin/env python3
"""
Install palettes into the Grillmaster image pipeline from external sources.

Usage:
  python3 scripts/install-palette.py list                          # list all registered palettes
  python3 scripts/install-palette.py list --category matplotlib    # only matplotlib colormaps
  python3 scripts/install-palette.py list --category builtin       # only built-in palettes
  python3 scripts/install-palette.py list --category user          # only user-installed
  python3 scripts/install-palette.py info <name>                   # show palette details
  python3 scripts/install-palette.py url <url> [name]              # install from url
  python3 scripts/install-palette.py vscode-search <query>        # search VS Code marketplace
  python3 scripts/install-palette.py vscode-install <publisher.ext>  # install a marketplace theme
  python3 scripts/install-palette.py remove <name>                 # remove a user-installed palette
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def _add_path():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))


_add_path()


# ── Helpers ──────────────────────────────────────────────────────────────


def _print_table(rows, headers):
    """Simple text table."""
    col_widths = [
        max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))
    ]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(sep.join(str(r[i]).ljust(w) for i, (r, w) in enumerate(zip(row, col_widths))))


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_list(args):
    from image_pipeline.core.palette_registry import (
        PALETTES,
        list_builtins,
        list_matplotlib,
        list_user,
    )

    if args.category == "matplotlib":
        names = list_matplotlib()
    elif args.category == "builtin":
        names = list_builtins()
    elif args.category == "user":
        names = list_user()
    else:
        names = list(PALETTES.keys())

    print(f"\n{'─' * 60}")
    print(f"  {len(names)} palettes ({args.category})")
    print(f"{'─' * 60}")
    # Group by first letter for readability
    for i, name in enumerate(names):
        count = len(PALETTES.get(name, []))
        print(f"  {name:<40s} {count:>3d} swatches")


def cmd_info(args):
    from image_pipeline.core.palette_registry import get, get_source

    swatches = get(args.name)
    if swatches is None:
        print(f"Palette '{args.name}' not found.")
        sys.exit(1)

    print(f"\n  Name:     {args.name}")
    print(f"  Source:   {get_source(args.name)}")
    print(f"  Count:    {len(swatches)} colors")
    print(f"  Swatches: [{', '.join(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in swatches[:8])}{' ...' if len(swatches) > 8 else ''}]")


def cmd_url(args):
    from image_pipeline.core.palette_registry import register, extract_colors_from_vscode_theme

    url = args.url
    name_override = args.name or ""

    print(f"Fetching {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            theme = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Failed: {exc}")
        sys.exit(1)

    swatches = extract_colors_from_vscode_theme(theme)
    if len(swatches) < 2:
        print(f"Only {len(swatches)} unique colors extracted — need at least 2.")
        sys.exit(1)

    palette_name = name_override or theme.get("name", "unknown").lower().replace(" ", "-")
    register(palette_name, swatches)
    print(f"\n✓ Registered '{palette_name}' ({len(swatches)} swatches)")
    print(f"  Source: {url}")
    print(f"  Preview: [{', '.join(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in swatches[:5])} ...]")


def cmd_vscode_search(args):
    url = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery"
    payload = {
        "filters": [
            {
                "criteria": [
                    {"filterType": 5, "value": "Themes"},
                    {"filterType": 10, "value": args.query},
                ],
                "pageNumber": 1,
                "pageSize": 20,
                "sortBy": 4,
                "sortOrder": 0,
            }
        ],
        "flags": 0x304,  # IncludeLatestVersionOnly | IncludeStatistics | IncludeCategoryAndTags
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json;api-version=7.2-preview.1",
    }

    print(f"Searching VS Code Marketplace for '{args.query}' ...\n")
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Search failed: {exc}")
        sys.exit(1)

    rows = []
    for ext in data.get("results", [{}])[0].get("extensions", []):
        stats = {s["statisticName"]: s.get("value", 0) for s in ext.get("statistics", [])}
        ext_id = f"{ext.get('publisher', {}).get('publisherName', '')}.{ext.get('extensionName', '')}"
        rows.append((
            ext_id,
            ext.get("displayName", ""),
            round(stats.get("averagerating", 0), 1),
            int(stats.get("install", 0)),
            ext.get("versions", [{}])[0].get("version", ""),
        ))

    _print_table(
        rows,
        headers=["ID (publisher.ext)", "Display Name", "Rating", "Installs", "Version"],
    )
    print(f"\n{len(rows)} results. Install with: python3 scripts/install-palette.py vscode-install <publisher.ext>")

    # Also show install command hints at the bottom
    for ext_id, display, *_ in rows[:5]:
        print(f"  → python3 scripts/install-palette.py vscode-install {ext_id}")


def cmd_vscode_install(args):
    from image_pipeline.core.palette_registry import register, extract_colors_from_vscode_theme

    ext_id = args.extension_id
    parts = ext_id.split(".", 1)
    if len(parts) != 2:
        print(f"Invalid extension ID '{ext_id}'. Expected format: publisher.extensionName (e.g. binaryify.one-dark-pro)")
        sys.exit(1)

    publisher, ext_name = parts

    # Fetch extension via FilterType 7 (ExtensionName with full qualified name)
    url = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery"
    payload = {
        "filters": [
            {
                "criteria": [
                    {"filterType": 7, "value": ext_id},
                ],
                "pageNumber": 1,
                "pageSize": 1,
                "sortBy": 0,
                "sortOrder": 0,
            }
        ],
        "flags": 0x312,  # IncludeVersions | IncludeFiles | IncludeStatistics | IncludeLatestVersionOnly
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json;api-version=7.2-preview.1",
    }

    print(f"Looking up '{ext_id}' on VS Code Marketplace ...")
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Lookup failed: {exc}")
        sys.exit(1)

    # Find the extension (filterType=7 returns exact match)
    extensions = data.get("results", [{}])[0].get("extensions", [])
    if not extensions:
        print(f"Extension '{ext_id}' not found on VS Code Marketplace.")
        sys.exit(1)
    ext = extensions[0]

    display_name = ext.get("displayName", ext_name)
    version_data = ext.get("versions", [{}])[0]
    files = version_data.get("files", [])

    # Find the VSIX URL
    vsix_url = None
    for f in files:
        if f.get("assetType") == "Microsoft.VisualStudio.Services.VSIXPackage":
            vsix_url = f.get("source", "")
            break

    if not vsix_url:
        print("No VSIX download URL found for this extension.")
        sys.exit(1)

    # Download VSIX and extract the first theme JSON
    import zipfile
    import io

    print(f"Downloading VSIX for '{display_name}' ...")
    try:
        with urllib.request.urlopen(vsix_url, timeout=30) as resp:
            vsix_data = resp.read()
    except Exception as exc:
        print(f"Download failed: {exc}")
        sys.exit(1)

    theme_json_data = None
    with zipfile.ZipFile(io.BytesIO(vsix_data)) as zf:
        # Find JSON files that look like theme files
        candidates = sorted(
            p for p in zf.namelist()
            if p.endswith(".json") and ("theme" in p.lower() or "color" in p.lower())
        )
        if not candidates:
            # Try any JSON file that's not package.json
            candidates = sorted(
                p for p in zf.namelist()
                if p.endswith(".json") and "package" not in p.lower()
            )
        for candidate in candidates:
            try:
                raw = zf.read(candidate)
                parsed = json.loads(raw)
                # Must have either a "colors" dict or a "tokenColors" array
                if isinstance(parsed, dict) and ("colors" in parsed or "tokenColors" in parsed):
                    theme_json_data = parsed
                    print(f"  Extracted: {candidate}")
                    break
            except Exception:
                continue

    if theme_json_data is None:
        print("Could not find a parsable theme JSON in the VSIX.")
        sys.exit(1)

    swatches = extract_colors_from_vscode_theme(theme_json_data)
    if len(swatches) < 2:
        print(f"Only {len(swatches)} unique colors extracted — need at least 2.")
        sys.exit(1)

    palette_name = ext_name.replace("-", "_").replace(".", "_")
    register(palette_name, swatches)
    print(f"\n✓ Installed '{palette_name}' ({len(swatches)} swatches)")
    print(f"  Source: VS Code Marketplace — {display_name} ({ext_id})")
    print(f"  First colors: {', '.join(f'#{r:02x}{g:02x}{b:02x}' for r, g, b in swatches[:5])} ...")
    print(f"\n  Now available as palette_name='{palette_name}' in any node.")


def cmd_remove(args):
    from image_pipeline.core.palette_registry import remove

    if remove(args.name):
        print(f"✓ Removed palette '{args.name}'")
    else:
        print(f"Palette '{args.name}' is not a user-installed palette (can't remove built-ins).")
        sys.exit(1)


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Manage Grillmaster palettes")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List registered palettes")
    p_list.add_argument("--category", default="all", choices=["all", "builtin", "matplotlib", "user"])

    p_info = sub.add_parser("info", help="Show palette details")
    p_info.add_argument("name")

    p_url = sub.add_parser("url", help="Install palette from a URL")
    p_url.add_argument("url")
    p_url.add_argument("name", nargs="?", default="")

    p_vs = sub.add_parser("vscode-search", help="Search VS Code Marketplace")
    p_vs.add_argument("query")

    p_vi = sub.add_parser("vscode-install", help="Install a theme from VS Code Marketplace by publisher.extensionName")
    p_vi.add_argument("extension_id")

    p_rm = sub.add_parser("remove", help="Remove a user-installed palette")
    p_rm.add_argument("name")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {
        "list": cmd_list,
        "info": cmd_info,
        "url": cmd_url,
        "vscode-search": cmd_vscode_search,
        "vscode-install": cmd_vscode_install,
        "remove": cmd_remove,
    }[args.command](args)


if __name__ == "__main__":
    main()
