#!/usr/bin/env python3
"""
fill_hub_content.py — Generate hub pillar content using Gemini Flash.

Fills:
  1) Hub pillar intro markdown (600-900 words) → content/hubs/<hub>/_index.md
  2) Cluster intros (1-3 sentences each) → data/site.yaml clusters[].intro
  3) Hub FAQs (4-8 Q/A) → data/site.yaml hubs[].faqs

Usage:
    python scripts/fill_hub_content.py                       # fill all hubs missing content
    python scripts/fill_hub_content.py --hub electricity-basics  # fill one hub
    python scripts/fill_hub_content.py --force                   # overwrite existing

Requires: GEMINI_API_KEY env var
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SITE_YAML_PATH = Path("data/site.yaml")
HUBS_CONTENT_ROOT = Path("content/hubs")

REQUEST_TIMEOUT = 120
MAX_RETRIES = 3


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_frontmatter(md_text: str):
    """Parse frontmatter + body from markdown."""
    if not md_text.startswith("---"):
        return {}, md_text
    after_open = md_text[4:]
    parts = after_open.split("\n---", 1)
    if len(parts) < 2:
        return {}, md_text
    fm_raw = parts[0]
    body = parts[1].lstrip("\n").lstrip("\r")
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except Exception:
        fm = {}
    return fm, body


def write_markdown(fm: dict, body: str) -> str:
    """Combine frontmatter + body into markdown string."""
    fm_str = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False).strip()
    return f"---\n{fm_str}\n---\n\n{body}\n"


def call_gemini(system: str, user: str) -> dict:
    """Call Gemini API and return parsed JSON."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"  API {r.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()

            data = r.json()
            text = ""
            for c in data.get("candidates", []):
                for p in c.get("content", {}).get("parts", []):
                    text += p.get("text", "")

            # Clean and parse JSON
            text = text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

    raise RuntimeError("Gemini API retries exhausted")


def generate_hub_content(hub: dict, site_cfg: dict) -> dict:
    """Generate pillar intro, cluster intros, and FAQs for one hub."""
    site = site_cfg.get("site", {}) or {}
    brand = site.get("brand") or site.get("title") or "Evergreen Site"
    tagline = site.get("tagline") or site.get("default_meta_description") or ""

    clusters = hub.get("clusters", [])
    cluster_list = "\n".join([f"- {c.get('label', c.get('id', ''))}" for c in clusters])

    system = f"""You write calm, neutral, informational content for the site "{brand}".
{tagline}

STRICT RULES:
- Informational only. No diagnosis, no treatment instructions, no professional advice.
- No first-person (I, we, my, mine, me). Use "people", "a person", "you" instead.
- No dates, no year references, no "recently", "currently", "nowadays".
- No guarantees or absolute claims.
- Global-safe: no country-specific regulations or assumptions.
- Calm, clear, evergreen prose. No hype, no fear.
- One gentle disclaimer is OK where relevant, but do NOT spam "consult a professional".
Return ONLY valid JSON. No markdown fences. No extra text."""

    prompt = f"""Generate content for the hub "{hub['label']}" ({hub['id']}).
Hub description: {hub.get('description', '')}

This hub has these clusters (subtopics):
{cluster_list}

Return a JSON object with these fields:

1) "hub_intro_markdown" — A pillar-style introduction to this topic area.
   - 600-900 words of markdown
   - Use H2 (##) and H3 (###) headings to structure the content
   - Cover: what this topic area is about, why it matters, key concepts, how subtopics connect
   - Informational and encyclopedic tone
   - Include 2-3 internal links in format: [anchor text](/pages/slug-here/) — use plausible slugs related to the hub
   - Every paragraph must be 2-3 sentences maximum

2) "clusters" — An array of objects, one per cluster:
   [{{"id": "<cluster-id>", "intro": "<1-3 sentence intro for this cluster>"}}]
   Each intro should explain what the cluster covers and why it's a distinct subtopic.

3) "faqs" — An array of 6 FAQ objects:
   [{{"q": "<question>", "a": "<answer 2-4 sentences>"}}]
   Questions should be things a beginner would ask about this topic area.
   Answers must be informational, not advisory."""

    return call_gemini(system, prompt)


def update_hub_index_md(hub_id: str, intro_md: str, force: bool = False):
    """Write hub pillar intro to content/hubs/<hub>/_index.md."""
    idx_path = HUBS_CONTENT_ROOT / hub_id / "_index.md"
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text("---\ntitle: \"\"\ndescription: \"\"\n---\n\n", encoding="utf-8")

    raw = idx_path.read_text(encoding="utf-8")
    fm, body = read_frontmatter(raw)

    # Only overwrite if body is essentially empty or force
    body_stripped = body.strip()
    if body_stripped and not force:
        print(f"  [skip] {hub_id}/_index.md already has content ({len(body_stripped)} chars)")
        return False

    md = write_markdown(fm, intro_md.strip())
    idx_path.write_text(md, encoding="utf-8")
    print(f"  [written] content/hubs/{hub_id}/_index.md ({len(intro_md)} chars)")
    return True


def update_site_yaml_hub(cfg: dict, hub_id: str, cluster_intros: list, faqs: list) -> bool:
    """Update cluster intros and FAQs in data/site.yaml for one hub."""
    hubs = (cfg.get("taxonomy", {}) or {}).get("hubs", [])
    updated = False

    for h in hubs:
        if h.get("id") != hub_id:
            continue

        # Update cluster intros
        if cluster_intros:
            intro_map = {c["id"]: c["intro"] for c in cluster_intros if c.get("id") and c.get("intro")}
            for c in h.get("clusters", []):
                cid = c.get("id", "")
                if cid in intro_map and (not c.get("intro") or not c["intro"].strip()):
                    c["intro"] = intro_map[cid]
                    updated = True

        # Update FAQs
        if faqs and not h.get("faqs"):
            h["faqs"] = faqs
            updated = True

        break

    return updated


def main():
    parser = argparse.ArgumentParser(description="Fill hub content using Gemini Flash")
    parser.add_argument("--site-root", default="", help="Site root directory (e.g. sites/<slug>)")
    parser.add_argument("--hub", default="", help="Fill only this hub (by id)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing content")
    parser.add_argument("--site-yaml", default="data/site.yaml", help="Path to site.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    args = parser.parse_args()

    if args.site_root:
        os.chdir(args.site_root)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Export it first.")
        sys.exit(1)

    site_yaml_path = Path(args.site_yaml)
    cfg = load_yaml(site_yaml_path)
    hubs = (cfg.get("taxonomy", {}) or {}).get("hubs", [])

    if not hubs:
        print("No hubs found in site.yaml. Run generate_taxonomy.py first.")
        sys.exit(1)

    # Filter to specific hub if requested
    if args.hub:
        hubs = [h for h in hubs if h.get("id") == args.hub]
        if not hubs:
            print(f"Hub '{args.hub}' not found in site.yaml")
            sys.exit(1)

    yaml_changed = False
    filled = 0

    for hub in hubs:
        hub_id = hub.get("id", "")
        print(f"\n--- Hub: {hub_id} ({hub.get('label', '')}) ---")

        # Check if already filled
        idx_path = HUBS_CONTENT_ROOT / hub_id / "_index.md"
        has_content = False
        if idx_path.exists():
            _, body = read_frontmatter(idx_path.read_text(encoding="utf-8"))
            has_content = len(body.strip()) > 50

        has_faqs = bool(hub.get("faqs"))
        has_cluster_intros = all(c.get("intro") for c in hub.get("clusters", []))

        if has_content and has_faqs and has_cluster_intros and not args.force:
            print(f"  [skip] Already complete")
            continue

        if args.dry_run:
            print(f"  [dry-run] Would generate: intro={'yes' if not has_content else 'skip'}, "
                  f"faqs={'yes' if not has_faqs else 'skip'}, "
                  f"cluster_intros={'yes' if not has_cluster_intros else 'skip'}")
            continue

        # Generate content
        try:
            result = generate_hub_content(hub, cfg)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        # Write hub intro markdown
        intro_md = result.get("hub_intro_markdown", "").strip()
        if intro_md:
            update_hub_index_md(hub_id, intro_md, force=args.force)

        # Update cluster intros in site.yaml
        cluster_intros = result.get("clusters", [])
        faqs = result.get("faqs", [])

        if update_site_yaml_hub(cfg, hub_id, cluster_intros, faqs):
            yaml_changed = True

        filled += 1
        # Rate limit between hubs
        if filled < len(hubs):
            time.sleep(2)

    # Save updated site.yaml
    if yaml_changed:
        save_yaml(site_yaml_path, cfg)
        print(f"\n✅ Updated {site_yaml_path} with cluster intros + FAQs")

    print(f"\n✅ Filled content for {filled} hubs")


if __name__ == "__main__":
    main()
