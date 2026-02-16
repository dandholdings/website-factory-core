#!/usr/bin/env python3
"""
generate_taxonomy.py — Populate data/site.yaml taxonomy from override file or blueprint.

Modes:
  1) Override:   --override-name energy-efficiency.yaml  (from taxonomy_overrides/)
  2) Blueprint:  --niche "home energy efficiency" [--family energy-efficiency]

Usage:
    python _core/scripts/generate_taxonomy.py --site-root . --override-name energy-efficiency.yaml
    python _core/scripts/generate_taxonomy.py --site-root . --niche "home energy efficiency"
    python _core/scripts/generate_taxonomy.py --site-root . --family energy-efficiency --niche "energy"
    python _core/scripts/generate_taxonomy.py --site-root . --override-name x.yaml --force

Outputs:
    data/site.yaml               taxonomy.hubs updated
    data/taxonomy_receipt.json    mode + source + hub summary
    content/hubs/<id>/_index.md   one per hub (idempotent, never overwrites)
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from taxonomy_blueprints import select_blueprint, BLUEPRINTS

# Override search paths (resolved at runtime relative to core checkout)
_OVERRIDE_SEARCH_PATHS = [
    Path("taxonomy_overrides"),
    SCRIPT_DIR.parent / "taxonomy_overrides",
]

_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ── validation ───────────────────────────────────────────────────────────────

def validate_taxonomy(hubs: list) -> list[str]:
    """Return error strings.  Empty list = valid."""
    errs: list[str] = []
    if not isinstance(hubs, list) or not hubs:
        return ["taxonomy.hubs is empty or not a list"]

    hub_ids: set[str] = set()
    for i, h in enumerate(hubs):
        if not isinstance(h, dict):
            errs.append(f"hub[{i}] not a dict"); continue
        hid = h.get("id", "")
        if not hid:
            errs.append(f"hub[{i}] missing 'id'")
        elif not _KEBAB.match(hid):
            errs.append(f"hub '{hid}' not kebab-case")
        elif hid in hub_ids:
            errs.append(f"duplicate hub id '{hid}'")
        hub_ids.add(hid)
        if not h.get("label"):
            errs.append(f"hub '{hid}' missing 'label'")

        cids: set[str] = set()
        for j, c in enumerate(h.get("clusters") or []):
            cid = c.get("id", "")
            if not cid:
                errs.append(f"hub '{hid}' cluster[{j}] missing 'id'")
            elif not _KEBAB.match(cid):
                errs.append(f"hub '{hid}' cluster '{cid}' not kebab-case")
            elif cid in cids:
                errs.append(f"hub '{hid}' duplicate cluster '{cid}'")
            cids.add(cid)

    for h in hubs:
        for r in h.get("related_hubs") or []:
            if r not in hub_ids:
                errs.append(f"hub '{h.get('id')}' related_hub '{r}' not found")

    # soft warnings only
    if len(hubs) < 6 or len(hubs) > 10:
        print(f"  [warn] {len(hubs)} hubs (recommended 6-10)")
    for h in hubs:
        cc = len(h.get("clusters") or [])
        if cc and (cc < 3 or cc > 7):
            print(f"  [warn] hub '{h.get('id')}' has {cc} clusters (recommended 4-6)")

    return errs


# ── override loader ──────────────────────────────────────────────────────────

def find_override(name: str) -> Path | None:
    p = Path(name)
    if p.is_file():
        return p.resolve()
    for base in _OVERRIDE_SEARCH_PATHS:
        c = base / name
        if c.is_file():
            return c.resolve()
    return None


def load_override(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print(f"ERROR: override is not a YAML mapping: {path}"); sys.exit(1)
    hubs = None
    if "taxonomy" in raw and isinstance(raw["taxonomy"], dict):
        hubs = raw["taxonomy"].get("hubs")
    if not hubs:
        hubs = raw.get("hubs")
    if not hubs:
        print(f"ERROR: override has no taxonomy.hubs: {path}"); sys.exit(1)
    errs = validate_taxonomy(hubs)
    if errs:
        print(f"ERROR: override validation failed ({path}):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    return hubs


# ── hub index pages ──────────────────────────────────────────────────────────

def ensure_hub_index_pages(hubs: list):
    root = Path("content/hubs")
    root.mkdir(parents=True, exist_ok=True)
    ri = root / "_index.md"
    if not ri.exists():
        ri.write_text('---\ntitle: "Topics"\ndescription: "Browse guides by topic."\n---\n\n', encoding="utf-8")

    for h in hubs or []:
        hid = str(h.get("id", "")).strip()
        if not hid:
            continue
        label = str(h.get("label", hid)).strip()
        desc = str(h.get("description", f"Guides about {label.lower()}.")).strip()
        d = root / hid
        d.mkdir(parents=True, exist_ok=True)
        idx = d / "_index.md"
        if idx.exists():
            print(f"  [skip] {hid}/_index.md exists")
            continue
        idx.write_text(f'---\ntitle: "{label}"\ndescription: "{desc}"\n---\n\n', encoding="utf-8")
        print(f"  [created] content/hubs/{hid}/_index.md")


# ── receipt ──────────────────────────────────────────────────────────────────

def write_receipt(mode, hubs, *, override_file="", family_id="", niche=""):
    r = {
        "mode": mode, "date": date.today().isoformat(),
        "override_file": override_file, "family_id": family_id,
        "niche_input": niche, "hub_count": len(hubs),
        "hubs": [{"id": h.get("id"), "label": h.get("label"),
                  "clusters": [c.get("id") for c in h.get("clusters") or []]}
                 for h in hubs],
    }
    p = Path("data/taxonomy_receipt.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Receipt: {p.resolve()}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate taxonomy (override or blueprint)")
    ap.add_argument("--site-root", default="", help="Site repo root (must contain data/)")
    ap.add_argument("--override-name", default="", help="Override filename in taxonomy_overrides/")
    ap.add_argument("--override-file", default="", help="Explicit path to override YAML")
    ap.add_argument("--niche", default="", help="Niche for blueprint auto-selection")
    ap.add_argument("--family", default="", help="Explicit blueprint family ID")
    ap.add_argument("--hub-count", type=int, default=8)
    ap.add_argument("--clusters-per-hub", type=int, default=5)
    ap.add_argument("--site-yaml", default="data/site.yaml")
    ap.add_argument("--force", action="store_true", help="Overwrite existing taxonomy")
    args = ap.parse_args()

    # resolve site root
    if args.site_root:
        root = Path(args.site_root).resolve()
        if not root.is_dir():
            print(f"ERROR: --site-root '{root}' not a directory"); sys.exit(1)
        os.chdir(root)
    print(f"Site root: {Path.cwd()}")

    syp = Path(args.site_yaml)
    cfg = load_yaml(syp)
    print(f"Site YAML: {syp.resolve()}")

    # idempotency
    existing = (cfg.get("taxonomy") or {}).get("hubs") or []
    if existing and not args.force:
        print(f"Taxonomy already has {len(existing)} hubs — skipping (use --force to overwrite).")
        return

    # resolve mode
    override_path = None
    if args.override_file:
        p = Path(args.override_file)
        if not p.is_file():
            print(f"ERROR: --override-file not found: {p}"); sys.exit(1)
        override_path = p.resolve()
    elif args.override_name:
        override_path = find_override(args.override_name)
        if not override_path:
            print(f"ERROR: override '{args.override_name}' not found.")
            for base in _OVERRIDE_SEARCH_PATHS:
                if base.is_dir():
                    avail = sorted(f.name for f in base.glob("*.yaml"))
                    if avail:
                        print(f"  Available in {base}: {', '.join(avail)}")
            sys.exit(1)

    family_id = ""
    if override_path:
        print(f"Mode: override\nFile: {override_path}")
        hubs = load_override(override_path)
    else:
        if not args.niche and not args.family:
            print("ERROR: provide --override-name, --niche, or --family"); sys.exit(1)
        if args.family:
            family_id = args.family
            if family_id not in BLUEPRINTS:
                print(f"Unknown family '{family_id}'. Available: {', '.join(BLUEPRINTS.keys())}"); sys.exit(1)
        else:
            family_id = select_blueprint(args.niche)
        bp = BLUEPRINTS[family_id]
        print(f"Mode: blueprint\nFamily: {family_id} ({bp['family_label']})")
        hubs = []
        used_ids = {h["id"] for h in bp["hubs"][:args.hub_count]}
        for h in bp["hubs"][:args.hub_count]:
            hub = dict(h)
            hub["clusters"] = hub.get("clusters", [])[:args.clusters_per_hub]
            hub["related_hubs"] = [r for r in hub.get("related_hubs", []) if r in used_ids][:3]
            hubs.append(hub)

    errs = validate_taxonomy(hubs)
    if errs:
        print("Validation errors:"); [print(f"  - {e}") for e in errs]; sys.exit(1)

    cfg.setdefault("taxonomy", {})["hubs"] = hubs
    save_yaml(syp, cfg)
    print(f"Updated: {syp.resolve()} ({len(hubs)} hubs)")

    ensure_hub_index_pages(hubs)
    write_receipt(
        mode="override" if override_path else "blueprint", hubs=hubs,
        override_file=str(override_path or ""), family_id=family_id, niche=args.niche,
    )

    tc = sum(len(h.get("clusters") or []) for h in hubs)
    print(f"\n✅ Taxonomy: {len(hubs)} hubs, {tc} clusters")
    for h in hubs:
        print(f"   {h['id']}: {h['label']} ({len(h.get('clusters') or [])} clusters)")


if __name__ == "__main__":
    main()
