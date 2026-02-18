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
from llm_client import slugify
from sanitize import sanitize_niche_input


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _resolve_override_path(name: str) -> Path | None:
    """Find an override YAML by filename in common locations."""
    if not name:
        return None
    candidates = []
    # 1) Relative to this script's parent (core checkout): <core>/taxonomy_overrides/<name>
    candidates.append((SCRIPT_DIR.parent / "taxonomy_overrides" / name).resolve())
    # 2) If running from thin-repo root with core checked out into _core/
    candidates.append((Path("_core") / "taxonomy_overrides" / name).resolve())
    # 3) If taxonomy_overrides is copied into site repo root
    candidates.append((Path("taxonomy_overrides") / name).resolve())
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_override_taxonomy(override_name: str) -> dict | None:
    """Load override YAML and return a dict with at least {'hubs': [...]} or {'taxonomy': {'hubs': [...]}}."""
    p = _resolve_override_path(override_name)
    if not p:
        return None
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Override file {p} is not a YAML mapping/object")
    return raw
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


def build_receipt(family_id: str, blueprint: dict, niche: str, hub_count: int, plan: dict = None) -> dict:
    """Build a human-readable taxonomy receipt."""
    hubs_used = blueprint.get("hubs", [])[:hub_count] if blueprint else []
    
    if plan:
        # Plan-based receipt
        reasoning = {
            "family_id": family_id,
            "family_label": blueprint.get("family_label", "Custom Family") if blueprint else "Custom Family",
            "niche_input": niche,
            "date": date.today().isoformat(),
            "hub_count": len(hubs_used),
            "generation_method": "plan_based",
            "why_this_family": f"Generated from a custom plan for niche '{niche}'.",
            "ontology": plan.get("ontology", {}),
            "hubs": [],
        }
    else:
        # Blueprint-based receipt
        reasoning = {
            "family_id": family_id,
            "family_label": blueprint.get("family_label", ""),
            "niche_input": niche,
            "date": date.today().isoformat(),
            "hub_count": len(hubs_used),
            "generation_method": "blueprint_based",
            "why_this_family": f"The niche '{niche}' matched the '{family_id}' blueprint based on keyword overlap.",
            "hubs": [],
        }
    
    for h in hubs_used:
        hub_entry = {
            "id": h.get("id", ""),
            "label": h.get("label", ""),
            "includes": h.get("description", ""),
            "cluster_count": len(h.get("clusters", [])),
            "clusters": [c.get("id", "") for c in h.get("clusters", [])],
            "cluster_reasoning": f"These {len(h.get('clusters', []))} clusters divide '{h.get('label', '')}' into distinct subtopic areas for topical authority.",
        }
        reasoning["hubs"].append(hub_entry)
    
    return reasoning


def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy from blueprint or plan")
    parser.add_argument("--niche", required=False, help="Niche description (e.g. 'home energy efficiency')")
    parser.add_argument("--family", default="", help="Explicit family ID (skip auto-detection)")
    parser.add_argument("--hub-count", type=int, default=8, help="Number of hubs to use (default 8)")
    parser.add_argument("--clusters-per-hub", type=int, default=5, help="Clusters per hub (default 5)")
    parser.add_argument("--pages-per-hub", type=int, default=3, help="Pages per hub (default 3)")
    parser.add_argument("--site-root", default="", help="Site root directory (e.g. sites/<slug>)")
    parser.add_argument("--override-name", default="", help="Override YAML filename in taxonomy_overrides/")
    parser.add_argument("--plan-file", default="", help="Use pre-generated plan JSON file")
    parser.add_argument("--generate-plan", action="store_true", help="Generate a new plan instead of using blueprint")
    parser.add_argument("--site-yaml", default="data/site.yaml", help="Path to site.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing taxonomy")
    args = parser.parse_args()

    if args.site_root:
        os.chdir(args.site_root)

    site_yaml_path = Path(args.site_yaml)
    cfg = load_yaml(site_yaml_path)

    # Sanitize niche input if provided
    if args.niche:
        args.niche = sanitize_niche_input(args.niche)

    if not (args.niche or args.override_name or args.plan_file):
        print("--niche is required unless --override-name or --plan-file is provided")
        sys.exit(2)

    # Check if taxonomy already exists
    existing_hubs = (cfg.get("taxonomy", {}) or {}).get("hubs", [])
    if existing_hubs and not args.force:
        print(f"Taxonomy already has {len(existing_hubs)} hubs. Use --force to overwrite.")
        return

    # Load plan if provided
    plan = None
    if args.plan_file:
        plan_path = Path(args.plan_file)
        if not plan_path.exists():
            print(f"Plan file not found: {plan_path}")
            sys.exit(1)
        try:
            import json
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
            plan = plan_data.get("plan") if isinstance(plan_data, dict) and "plan" in plan_data else plan_data
            print(f"Loaded plan from {plan_path}")
        except Exception as e:
            print(f"Failed to load plan: {e}")
            sys.exit(1)

    # Generate new plan if requested
    elif args.generate_plan and args.niche:
        try:
            from plan_generator import generate_plan_contract, save_plan
            print(f"Generating plan for niche: {args.niche}")
            plan = generate_plan_contract(
                niche=args.niche,
                hub_count=args.hub_count,
                clusters_per_hub=args.clusters_per_hub,
                pages_per_hub=args.pages_per_hub,
                family_id=args.family if args.family else None
            )
            # Save the generated plan
            plan_path = Path("scripts") / "generated_plan.json"
            save_plan(plan, plan_path)
            print(f"Plan generated and saved to {plan_path}")
        except ImportError as e:
            print(f"Cannot import plan_generator: {e}")
            print("Falling back to blueprint-based generation")
            plan = None
        except Exception as e:
            print(f"Failed to generate plan: {e}")
            print("Falling back to blueprint-based generation")
            plan = None

    # Load override taxonomy if provided (and no plan)
    override = None
    if args.override_name and not plan:
        override = load_override_taxonomy(args.override_name)
        if not override:
            print(f"Override '{args.override_name}' not found in taxonomy_overrides.")
            sys.exit(1)

    if plan:
        # Use plan-based taxonomy
        family_id = plan.get("family_id", "new-family")
        family_label = plan.get("family_label", "Custom Family")
        
        # Extract hubs from plan
        hubs = []
        for hub_data in plan.get("hubs", []):
            hub = {
                "id": hub_data.get("id", ""),
                "label": hub_data.get("label", ""),
                "description": hub_data.get("description", ""),
                "clusters": hub_data.get("clusters", []),
                "related_hubs": []  # Will be populated based on plan relationships
            }
            hubs.append(hub)
        
        bp = {"family_label": family_label, "hubs": hubs}
        print(f"Using plan-based taxonomy: {family_label} ({len(hubs)} hubs)")
        
    elif override:
        # Accept either top-level hubs: [...] or taxonomy: {hubs:[...]}
        raw_hubs = None
        if isinstance(override.get("taxonomy"), dict):
            raw_hubs = override["taxonomy"].get("hubs")
        if raw_hubs is None:
            raw_hubs = override.get("hubs")
        if not isinstance(raw_hubs, list) or not raw_hubs:
            print("Override YAML must contain hubs (either top-level 'hubs' or 'taxonomy.hubs').")
            sys.exit(1)
        hubs = raw_hubs
        family_id = override.get("family_id") or (args.family or "override")
        bp = {"family_label": override.get("family_label") or "Override", "hubs": hubs}
        print(f"Using override taxonomy: {args.override_name} ({len(hubs)} hubs)")
    else:
        # Select blueprint (traditional approach)
        if args.family:
            family_id = args.family
            if family_id not in BLUEPRINTS:
                print(f"Unknown family '{family_id}'. Available: {', '.join(BLUEPRINTS.keys())}")
                sys.exit(1)
        else:
            family_id = select_blueprint(args.niche)

        # Handle "new-family" case - generate a custom plan
        if family_id == "new-family":
            print(f"No existing blueprint matches niche '{args.niche}'. Generating custom plan...")
            try:
                from plan_generator import generate_plan_contract, save_plan
                plan = generate_plan_contract(
                    niche=args.niche,
                    hub_count=args.hub_count,
                    clusters_per_hub=args.clusters_per_hub,
                    pages_per_hub=args.pages_per_hub,
                    family_id="new-family"
                )
                # Use plan-based taxonomy
                family_id = plan.get("family_id", "new-family")
                family_label = plan.get("family_label", "Custom Family")
                
                # Extract hubs from plan
                hubs = []
                for hub_data in plan.get("hubs", []):
                    hub = {
                        "id": hub_data.get("id", ""),
                        "label": hub_data.get("label", ""),
                        "description": hub_data.get("description", ""),
                        "clusters": hub_data.get("clusters", []),
                        "related_hubs": []
                    }
                    hubs.append(hub)
                
                bp = {"family_label": family_label, "hubs": hubs}
                print(f"Generated custom plan: {family_label} ({len(hubs)} hubs)")
                
            except ImportError as e:
                print(f"Cannot import plan_generator: {e}")
                print("Falling back to default blueprint (energy-efficiency)")
                family_id = "energy-efficiency"
                bp = BLUEPRINTS[family_id]
                print(f"Selected default blueprint: {family_id} ({bp['family_label']})")
                
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
            except Exception as e:
                print(f"Failed to generate custom plan: {e}")
                print("Falling back to default blueprint (energy-efficiency)")
                family_id = "energy-efficiency"
                bp = BLUEPRINTS[family_id]
                print(f"Selected default blueprint: {family_id} ({bp['family_label']})")
                
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
        else:
            # Use existing blueprint
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
    receipt = build_receipt(family_id, bp, args.niche or args.override_name, len(hubs) if 'hubs' in locals() else args.hub_count)
    receipt_path = Path("scripts") / "taxonomy_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Receipt written to {receipt_path}")

    # Summary
    try:
        print(f"\n✅ Taxonomy generated: {len(hubs)} hubs, {sum(len(h.get('clusters', [])) for h in hubs)} clusters")
    except UnicodeEncodeError:
        print(f"\n[OK] Taxonomy generated: {len(hubs)} hubs, {sum(len(h.get('clusters', [])) for h in hubs)} clusters")
    for h in hubs:
        c_count = len(h.get("clusters", []))
        print(f"   {h['id']}: {h['label']} ({c_count} clusters)")


if __name__ == "__main__":
    main()