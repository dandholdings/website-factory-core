#!/usr/bin/env python3
"""
generate_taxonomy.py — Populate data/site.yaml taxonomy from a blueprint.

Usage:
    python scripts/generate_taxonomy.py --niche "home energy efficiency"
    python scripts/generate_taxonomy.py --niche "cognitive biases and decisions" --hub-count 8
    python scripts/generate_taxonomy.py --family energy-efficiency   # explicit family

Outputs:
    - data/site.yaml  taxonomy.hubs updated
    - scripts/taxonomy_receipt.json  reasoning + chosen taxonomy
    - content/hubs/<id>/_index.md  created for each hub
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from datetime import date

import yaml

# Allow running from thin-repo or core
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from taxonomy_blueprints import select_blueprint, BLUEPRINTS


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "site"


def ensure_hub_index_pages(hubs: list):
    """Create content/hubs/<id>/_index.md for each hub."""
    hubs_root = Path("content") / "hubs"
    hubs_root.mkdir(parents=True, exist_ok=True)

    # Root index
    root_idx = hubs_root / "_index.md"
    if not root_idx.exists():
        root_idx.write_text(
            '---\ntitle: "Topics"\ndescription: "Browse guides by topic."\n---\n\n',
            encoding="utf-8",
        )

    for h in hubs or []:
        hid = str(h.get("id", "")).strip()
        label = str(h.get("label", hid)).strip()
        desc = str(h.get("description", f"Guides about {label.lower()}.")).strip()
        if not hid:
            continue
        p = hubs_root / hid
        p.mkdir(parents=True, exist_ok=True)
        idx = p / "_index.md"
        if idx.exists():
            print(f"  [skip] {hid}/_index.md already exists")
            continue
        idx.write_text(
            f'---\ntitle: "{label}"\ndescription: "{desc}"\n---\n\n',
            encoding="utf-8",
        )
        print(f"  [created] content/hubs/{hid}/_index.md")


def build_receipt(family_id: str, blueprint: dict, niche: str, hub_count: int) -> dict:
    """Build a human-readable taxonomy receipt."""
    hubs_used = blueprint["hubs"][:hub_count]
    reasoning = {
        "family_id": family_id,
        "family_label": blueprint["family_label"],
        "niche_input": niche,
        "date": date.today().isoformat(),
        "hub_count": len(hubs_used),
        "why_this_family": f"The niche '{niche}' matched the '{family_id}' blueprint based on keyword overlap.",
        "hubs": [],
    }
    for h in hubs_used:
        hub_entry = {
            "id": h["id"],
            "label": h["label"],
            "includes": h.get("description", ""),
            "cluster_count": len(h.get("clusters", [])),
            "clusters": [c["id"] for c in h.get("clusters", [])],
            "cluster_reasoning": f"These {len(h.get('clusters', []))} clusters divide '{h['label']}' into distinct subtopic areas for topical authority.",
        }
        reasoning["hubs"].append(hub_entry)
    return reasoning


def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy from blueprint")
    parser.add_argument("--niche", required=True, help="Niche description (e.g. 'home energy efficiency')")
    parser.add_argument("--family", default="", help="Explicit family ID (skip auto-detection)")
    parser.add_argument("--hub-count", type=int, default=8, help="Number of hubs to use (default 8)")
    parser.add_argument("--clusters-per-hub", type=int, default=5, help="Clusters per hub (default 5)")
    parser.add_argument("--site-yaml", default="data/site.yaml", help="Path to site.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing taxonomy")
    args = parser.parse_args()

    site_yaml_path = Path(args.site_yaml)
    cfg = load_yaml(site_yaml_path)

    # Check if taxonomy already exists
    existing_hubs = (cfg.get("taxonomy", {}) or {}).get("hubs", [])
    if existing_hubs and not args.force:
        print(f"Taxonomy already has {len(existing_hubs)} hubs. Use --force to overwrite.")
        return

    # Select blueprint
    if args.family:
        family_id = args.family
        if family_id not in BLUEPRINTS:
            print(f"Unknown family '{family_id}'. Available: {', '.join(BLUEPRINTS.keys())}")
            sys.exit(1)
    else:
        family_id = select_blueprint(args.niche)

    bp = BLUEPRINTS[family_id]
    print(f"Selected blueprint: {family_id} ({bp['family_label']})")

    # Trim hubs/clusters to requested counts
    hubs = []
    for h in bp["hubs"][:args.hub_count]:
        hub = dict(h)
        clusters = hub.get("clusters", [])[:args.clusters_per_hub]
        hub["clusters"] = clusters
        # Ensure related_hubs only reference hubs we're using
        used_ids = {hh["id"] for hh in bp["hubs"][:args.hub_count]}
        hub["related_hubs"] = [r for r in hub.get("related_hubs", []) if r in used_ids][:3]
        hubs.append(hub)

    # Update site.yaml
    if "taxonomy" not in cfg:
        cfg["taxonomy"] = {}
    cfg["taxonomy"]["hubs"] = hubs

    save_yaml(site_yaml_path, cfg)
    print(f"Updated {site_yaml_path} with {len(hubs)} hubs")

    # Create hub _index.md files
    ensure_hub_index_pages(hubs)

    # Write receipt
    receipt = build_receipt(family_id, bp, args.niche, args.hub_count)
    receipt_path = Path("scripts") / "taxonomy_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Receipt written to {receipt_path}")

    # Summary
    print(f"\n✅ Taxonomy generated: {len(hubs)} hubs, {sum(len(h.get('clusters', [])) for h in hubs)} clusters")
    for h in hubs:
        c_count = len(h.get("clusters", []))
        print(f"   {h['id']}: {h['label']} ({c_count} clusters)")


if __name__ == "__main__":
    main()
