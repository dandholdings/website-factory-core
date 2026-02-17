from __future__ import annotations

import os
import sys
import argparse
import re
import json
import time
import hashlib
import random
from pathlib import Path
from typing import Optional, Dict, Any

import requests
import yaml

# --- site-root support (early) --------------------------------------------
def _apply_site_root_early():
    """Allow running core scripts from inside thin-repo."""

    def _slugify(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        s = re.sub(r"-+", "-", s).strip("-")
        return s

    if "--site-root" in sys.argv:
        i = sys.argv.index("--site-root")
        if i + 1 < len(sys.argv) and sys.argv[i + 1]:
            os.chdir(sys.argv[i + 1])
            return

    slug = ""
    if "--site-slug" in sys.argv:
        i = sys.argv.index("--site-slug")
        if i + 1 < len(sys.argv):
            slug = (sys.argv[i + 1] or "").strip()

    if not slug:
        slug = (os.getenv("SITE_SLUG") or "").strip()

    if not slug:
        slug = _slugify(os.getenv("BOOTSTRAP_NICHE") or os.getenv("NICHE") or "")

    if slug:
        root = Path("sites") / slug
        root.mkdir(parents=True, exist_ok=True)
        os.chdir(root)
        return

_apply_site_root_early()

def _sr(rel: str) -> Path:
    return (Path.cwd() / rel).resolve()


# --- LLM provider configuration ----------------------------------------
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
MOONSHOT_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _llm_provider() -> str:
    if GEMINI_API_KEY:
        return "gemini"
    if MOONSHOT_API_KEY:
        return "moonshot"
    return ""

PROVIDER = _llm_provider()

HEADERS = {"Content-Type": "application/json"}
def _safe_int(val, default: int) -> int:
    """Parse an int from a string, returning default on any failure (empty string, None, garbage)."""
    try:
        v = (val or "").strip() if isinstance(val, str) else val
        return int(v) if v else default
    except (ValueError, TypeError):
        return default

REQUEST_TIMEOUT = _safe_int(os.getenv("REQUEST_TIMEOUT"), 180)
CONNECT_TIMEOUT = _safe_int(os.getenv("CONNECT_TIMEOUT"), 20)

# FIX: Bootstrap needs much more tokens for 300 titles.
# At ~8 tokens per title, 300 titles = ~2400+ tokens. Default 1600 will always truncate.
# Use a dedicated higher cap for bootstrap, overridable via env.
MAX_OUTPUT_TOKENS = _safe_int(os.getenv("MAX_OUTPUT_TOKENS") or os.getenv("BOOTSTRAP_MAX_TOKENS"), 8192)

# FIX: Temperature — only clamp for Moonshot/Kimi. Let Gemini use any value.
try:
    _t_raw = os.getenv("TEMPERATURE", "0.7").strip()
    TEMPERATURE = float(_t_raw)
except Exception:
    TEMPERATURE = 0.7

# Only force temperature=1 for Moonshot/Kimi (legacy constraint).
if PROVIDER == "moonshot" and TEMPERATURE != 1:
    TEMPERATURE = 1

HTTP_MAX_TRIES = _safe_int(os.getenv("HTTP_MAX_TRIES") or os.getenv("KIMI_HTTP_MAX_TRIES"), 6)
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", os.getenv("KIMI_BACKOFF_BASE", "1.7")))

EMPTY_BACKOFF_BASE = float(os.getenv("EMPTY_BACKOFF_BASE", os.getenv("KIMI_EMPTY_BACKOFF_BASE", "1.25")))
EMPTY_BACKOFF_CAP = float(os.getenv("EMPTY_BACKOFF_CAP", os.getenv("KIMI_EMPTY_BACKOFF_CAP", "3.0")))

SITE_PATH = Path(os.getenv("SITE_CONFIG", "data/site.yaml"))
HUGO_PATH = Path("hugo.yaml")
TITLES_POOL_PATH = _sr("scripts/titles_pool.txt")
MANIFEST_PATH = Path("scripts/manifest.json")

TITLE_COUNT = _safe_int(os.getenv("TITLE_COUNT"), 300)
PAGES_NOW = _safe_int(os.getenv("PAGES_NOW"), 0)

NICHE = (os.getenv("BOOTSTRAP_NICHE") or "").strip()
TONE = (os.getenv("BOOTSTRAP_TONE") or "").strip()

THEME_PACKS = [
  "calm-paper","charcoal-gold","clinic-clean","earthy-trail","editorial","forest-hush",
  "lavender-dusk","maker","matcha-cream","midnight-plum","minimal-mono","modern-sans",
  "night-ink","ocean-mist","playful-soft","ruby-graphite","sandstone","steel-blue",
  "sunset-clay","warm-sunrise"
]

DEFAULT_OUTLINE_H2 = [
  "Intro",
  "Definitions and key terms",
  "Why this topic exists",
  "How people usually experience this",
  "How it typically works",
  "When this topic tends to come up",
  "Clarifying examples",
  "Common misconceptions",
  "Why this topic gets misunderstood online",
  "Related situations that feel similar",
  "Related topics and deeper reading",
  "Neutral summary",
  "FAQs",
]

def slugify(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:60].strip("-") or "site"


def _titlecase_for_display(t: str) -> str:
    """Title-case a string for display, preserving small words."""
    small = {"a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "in", "of", "is", "it", "vs"}
    words = t.strip().split()
    if not words:
        return t
    result = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() not in small:
            result.append(w.capitalize())
        else:
            result.append(w.lower())
    return " ".join(result)


def _normalize_quotes_list(titles: list[str]) -> list[str]:
    """Strip smart quotes and normalize whitespace in titles."""
    out = []
    for t in titles:
        t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.append(t)
    return out


def _dedupe_preserve_order(titles: list[str]) -> list[str]:
    """Remove duplicate titles (case-insensitive), keeping first occurrence."""
    seen: set[str] = set()
    out = []
    for t in titles:
        key = t.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_hub_index_pages(hubs: list[dict]):
    """Create content/hubs/<id>/_index.md so hub landing pages exist for navigation."""
    hubs_root = Path("content") / "hubs"
    hubs_root.mkdir(parents=True, exist_ok=True)

    # Root hubs index if missing
    root_index = hubs_root / "_index.md"
    if not root_index.exists():
        root_index.write_text(
            "---\n"
            "title: \"Topics\"\n"
            "description: \"Browse guides by topic.\"\n"
            "---\n\n",
            encoding="utf-8",
        )

    for h in hubs or []:
        hid = str(h.get("id") or "").strip()
        label = str(h.get("label") or hid).strip()
        if not hid:
            continue
        p = hubs_root / hid
        p.mkdir(parents=True, exist_ok=True)
        idx = p / "_index.md"
        if idx.exists():
            continue
        idx.write_text(
            "---\n"
            f"title: \"{label}\"\n"
            f"description: \"Guides and checklists about {label.lower()}.\"\n"
            "---\n\n",
            encoding="utf-8",
        )

def parse_json_strict_or_extract(raw: str) -> dict:
    """Parse a JSON object from model output. Robust against fences, prose, truncation."""
    if raw is None:
        raw = ""

    if isinstance(raw, (dict, list)):
        try:
            return raw if isinstance(raw, dict) else (raw[0] if raw and isinstance(raw[0], dict) else json.loads(json.dumps(raw)))
        except Exception:
            raw = json.dumps(raw)

    raw = str(raw)
    raw = raw.replace("\ufeff", "").replace("\x00", "").strip()

    if not raw:
        raise json.JSONDecodeError("Empty model output (expected JSON object)", raw, 0)

    def _strip_code_fences(s: str) -> str:
        s2 = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
        s2 = re.sub(r"\s*```\s*$", "", s2.strip())
        return s2.strip()

    cand = _strip_code_fences(raw)

    try:
        obj = json.loads(cand)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Top-level JSON must be an object", cand, 0)
        return obj
    except json.JSONDecodeError:
        pass

    # Extract the first balanced JSON object from the text.
    s = cand
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found in model output", cand, 0)

    depth = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        raise json.JSONDecodeError("JSON object appears truncated (missing closing brace)", cand, 0)

    snippet = s[start:end].strip()
    snippet = _strip_code_fences(snippet)

    obj = json.loads(snippet)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Top-level JSON must be an object", snippet, 0)
    return obj


def _safe_write_kimi_dump(
    kind: str,
    attempt: int,
    *,
    content: str = "",
    envelope: Optional[Dict[str, Any]] = None,
    http_status: Optional[int] = None,
    http_text: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    """Write a debug dump. Never includes API keys."""
    try:
        log_dir = Path("scripts") / "_logs" / "kimi"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = hashlib.sha1(f"{ts}-{kind}-{attempt}-{random.random()}".encode("utf-8")).hexdigest()[:10]
        p = log_dir / f"{ts}-{kind}-a{attempt+1}-{rid}.json"

        def _trim(s: str, n: int) -> str:
            s = "" if s is None else str(s)
            s = s.replace("\x00", "")
            return s[:n]

        # FIX: Redact all sensitive keys including x-goog-api-key
        safe_payload = None
        if isinstance(payload, dict):
            safe_payload = {}
            for k, v in payload.items():
                if k.lower() in ("api_key", "authorization", "x-goog-api-key"):
                    safe_payload[k] = "[REDACTED]"
                else:
                    safe_payload[k] = v

        # FIX: Scrub any key= params from http_text
        safe_http_text = _trim(http_text, 4000) if http_text else ""
        safe_http_text = re.sub(r'key=[A-Za-z0-9_-]+', 'key=[REDACTED]', safe_http_text)

        dump = {
            "kind": kind,
            "attempt": attempt + 1,
            "model": GEMINI_MODEL if PROVIDER == "gemini" else MOONSHOT_MODEL,
            "http_status": http_status,
            "content_preview": _trim(content, 4000),
            "http_text_preview": safe_http_text,
            "payload": safe_payload,
            "envelope": envelope,
        }
        p.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def kimi_json(system: str, user: str, temperature: float = 0.7, max_tokens: int = 1400) -> dict:
    """Request a JSON object from the configured LLM provider."""
    provider = PROVIDER
    if not provider:
        raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY (recommended) or MOONSHOT_API_KEY (legacy).")

    # FIX: Only clamp temperature for Moonshot/Kimi, not Gemini.
    temp = temperature
    if provider == "moonshot" and "kimi" in (MOONSHOT_MODEL or "").lower():
        temp = 1

    http_max_tries = HTTP_MAX_TRIES
    backoff_base = BACKOFF_BASE

    def _sleep(attempt: int) -> float:
        return min(60.0, (backoff_base ** attempt) + random.random())

    last_err = None

    def _gemini_request_payload() -> tuple:
        # FIX: Use x-goog-api-key header instead of URL param (security)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

        prompt = (
            system.strip()
            + "\n\n"
            + "CRITICAL: Output MUST be a single valid JSON object. No markdown. No code fences. No commentary.\n"
            + user.strip()
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temp),
                "maxOutputTokens": int(max_tokens),
                # FIX: Use responseMimeType to force structured JSON output
                "responseMimeType": "application/json",
                # FIX: Disable thinking — Gemini 2.5 Flash spends ~3900 tokens
                # on internal reasoning, leaving only ~150 for actual output.
                # Content generation doesn't need chain-of-thought.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        }
        return url, headers, payload

    def _moonshot_request_payload() -> tuple:
        url = f"{MOONSHOT_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {MOONSHOT_API_KEY}", "Content-Type": "application/json"}

        payload = {
            "model": MOONSHOT_MODEL,
            "temperature": int(temp) if isinstance(temp, (int, float)) and int(temp) == 1 else float(temp),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return url, headers, payload

    def _extract_text_from_response(provider_name: str, data: dict) -> str:
        if provider_name == "gemini":
            try:
                cands = data.get("candidates") or []
                if not cands:
                    return ""
                content = (cands[0].get("content") or {})
                parts = content.get("parts") or []
                out = []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        out.append(p["text"])
                return "".join(out).strip()
            except Exception:
                return ""

        try:
            msg = (data.get("choices", [{}])[0].get("message") or {})
            content = msg.get("content")

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if "text" in part and isinstance(part.get("text"), str):
                            parts.append(part.get("text"))
                        elif "content" in part and isinstance(part.get("content"), str):
                            parts.append(part.get("content"))
                    elif isinstance(part, str):
                        parts.append(part)
                content = "".join(parts).strip()

            if content is None:
                content = ""

            if not str(content).strip():
                tool_calls = msg.get("tool_calls") or []
                if isinstance(tool_calls, list) and tool_calls:
                    fn = (tool_calls[0].get("function") or {})
                    content = fn.get("arguments") or ""

            if not str(content).strip():
                fn_call = (msg.get("function_call") or {})
                content = fn_call.get("arguments") or ""

            if not str(content).strip() and isinstance(msg.get("json"), (dict, str)):
                content = msg.get("json")

            return str(content or "").strip()
        except Exception:
            return ""

    # Main retry loop
    for attempt in range(http_max_tries):
        if provider == "gemini":
            url, headers, payload = _gemini_request_payload()
        else:
            url, headers, payload = _moonshot_request_payload()

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == http_max_tries - 1:
                raise
            sleep = _sleep(attempt)
            print(f"HTTP error: {type(e).__name__} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        if r.status_code in (408, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}: {r.text[:2000]}"
            _safe_write_kimi_dump(f"{provider}_http_{r.status_code}", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == http_max_tries - 1:
                break
            sleep = _sleep(attempt)
            print(f"HTTP {r.status_code} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        if r.status_code >= 400:
            last_err = f"HTTP {r.status_code}: {r.text[:2000]}"
            _safe_write_kimi_dump(f"{provider}_http_error", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            break

        try:
            data = r.json()
        except Exception as e:
            last_err = f"Bad JSON response envelope: {type(e).__name__}: {e}"
            _safe_write_kimi_dump(f"{provider}_bad_envelope", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == http_max_tries - 1:
                raise
            sleep = _sleep(attempt)
            print(f"Response parse error — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        content = _extract_text_from_response(provider, data)

        if not str(content).strip():
            last_err = "Empty model output"
            _safe_write_kimi_dump(f"{provider}_empty", attempt, content="", envelope=data, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == http_max_tries - 1:
                break
            sleep = _sleep(attempt)
            print(f"Model returned empty output (attempt {attempt+1}/{http_max_tries}) — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        try:
            return parse_json_strict_or_extract(content)
        except json.JSONDecodeError as e:
            last_err = f"Model JSON decode error: {e}"
            preview = (content or "").strip().replace("\n", " ")[:240]
            print(f"Model returned non-JSON content (attempt {attempt+1}/{http_max_tries}): {preview}")
            _safe_write_kimi_dump(f"{provider}_json_decode", attempt, content=content, envelope=data, http_status=r.status_code, http_text=r.text, payload=payload)

            if attempt == http_max_tries - 1:
                break
            sleep = _sleep(attempt)
            print(f"Retrying after non-JSON output in {sleep:.1f}s")
            time.sleep(sleep)
            continue

    raise RuntimeError(last_err or f"{provider} API retries exhausted")


def ensure_manifest_reset():
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"used_titles": [], "generated_this_run": []}, indent=2), encoding="utf-8")


def write_titles_pool(path: Path, titles: list[str], title_hub_overrides: dict[str, str] | None = None) -> None:
    """
    Writes titles_pool.txt.
    Back-compat: plain lines are titles.
    New: if title_hub_overrides provides a hub_id for a title, write "<hub_id>\t<Title>".
    """
    title_hub_overrides = title_hub_overrides or {}

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    seen_titles: set[str] = set()

    for t in titles:
        if not isinstance(t, str):
            continue
        title = t.strip()
        if not title:
            continue

        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        hub = (title_hub_overrides.get(title) or "").strip()
        if hub:
            lines.append(f"{hub}\t{title}")
        else:
            lines.append(title)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
def patch_hugo_yaml(site_cfg: dict):
    """Keep hugo.yaml minimal but aligned to site identity for Cloudflare Pages."""
    if not HUGO_PATH.exists():
        return
    cfg = yaml.safe_load(HUGO_PATH.read_text(encoding="utf-8")) or {}

    site = site_cfg.get("site", {}) if isinstance(site_cfg, dict) else {}
    brand = site.get("brand") or site.get("title") or cfg.get("title") or "Site"
    base_url = site.get("base_url") or cfg.get("baseURL") or "https://YOUR-SITE.pages.dev/"
    lang = site.get("language_code") or cfg.get("languageCode") or "en-us"

    cfg["baseURL"] = str(base_url)
    cfg["languageCode"] = str(lang)
    cfg["title"] = str(site.get("title") or brand)

    params = cfg.get("params") or {}
    factory = params.get("factory") or {}
    factory["brand"] = str(brand)
    params["factory"] = factory

    cfg["params"] = params

    HUGO_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

def _deterministic_bootstrap_fallback(niche: str, title_count: int) -> Dict[str, Any]:
    """Deterministic fallback bootstrap content when the LLM is unavailable."""
    n = (niche or "").strip()
    words = [w for w in re.split(r"\s+", re.sub(r"[^a-zA-Z0-9\s]", " ", n)) if w]
    title_words = words[:4] if words else ["Evergreen", "Guides"]
    site_title = " ".join([w.capitalize() for w in title_words])[:40].strip() or "Evergreen Guides"
    brand = site_title.split(" ")[0:3]
    brand = " ".join([w.capitalize() for w in brand]).strip() or site_title

    seed = int(hashlib.sha1(n.encode("utf-8")).hexdigest()[:8], 16) if n else 0
    theme_pack = THEME_PACKS[seed % len(THEME_PACKS)]

    tagline = f"Practical explanations for {n.lower()} — neutral, simple, no hype." if n else "Practical explanations — neutral, simple, no hype."
    meta = tagline[:155]

    hubs = [
        {"id": "basics", "label": "Basics", "description": f"Foundational concepts of {n.lower()}."},
        {"id": "how-it-works", "label": "How It Works", "description": f"Mechanisms and processes behind {n.lower()}."},
        {"id": "types-and-variations", "label": "Types & Variations", "description": f"Different forms and categories of {n.lower()}."},
        {"id": "common-situations", "label": "Common Situations", "description": f"Everyday scenarios involving {n.lower()}."},
        {"id": "misconceptions", "label": "Misconceptions", "description": f"Myths and misunderstandings about {n.lower()}."},
        {"id": "comparisons", "label": "Comparisons", "description": f"Comparing related aspects of {n.lower()}."},
        {"id": "deeper-concepts", "label": "Deeper Concepts", "description": f"Advanced topics in {n.lower()}."},
    ]

    base = n.lower().strip() or "this topic"
    templates = [
        "What is {x}?",
        "How does {x} work?",
        "Types of {x} explained",
        "Common misconceptions about {x}",
        "{x}: key terms explained",
        "Why {x} matters in everyday life",
        "How {x} affects {y}",
        "Signs of {x} that people overlook",
        "A simple guide to understanding {x}",
        "Common questions about {x}",
        "Is {x} the same as {z}?",
        "How {x} differs from {z}",
        "What causes {x}?",
        "Where does {x} happen most often?",
        "How to recognize {x} in {y}",
        "Does {x} always involve {y}?",
        "When is {x} most likely to occur?",
        "What makes {x} stronger or weaker?",
        "Can {x} happen without awareness?",
        "How {y} influences {x}",
        "The role of {y} in {x}",
        "{x} in {y}: what to know",
        "How {x} spreads between {y}",
        "Why {x} is misunderstood",
        "What experts mean by {x}",
        "How children experience {x}",
        "How culture shapes {x}",
        "The difference between {x} and {z}",
        "How technology affects {x}",
        "How groups experience {x} differently",
    ]
    ys = ["relationships", "groups", "families", "workplaces", "social settings", "daily life", "conversations"]
    zs = ["empathy", "sympathy", "mood", "influence", "suggestion", "imitation", "mimicry"]

    titles = []
    rnd = random.Random(seed)
    attempts = 0
    while len(titles) < max(40, min(title_count, 600)) and attempts < 2000:
        attempts += 1
        t = rnd.choice(templates)
        title = t.format(x=base, y=rnd.choice(ys), z=rnd.choice(zs))
        titles.append(title)

    uniq = []
    seen = set()
    for t in titles:
        s = slugify(t)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(t)
        if len(uniq) >= title_count:
            break

    # Build catalog with round-robin hub assignment for fallback
    hub_ids = [h["id"] for h in hubs]
    catalog = [{"title": t, "hub": hub_ids[i % len(hub_ids)]} for i, t in enumerate(uniq)]

    return {
        "site_title": site_title,
        "brand": brand,
        "tagline": tagline[:120],
        "default_meta_description": meta,
        "theme_pack": theme_pack,
        "taxonomy": {"hubs": hubs},
        "catalog": catalog,
        "titles_pool": uniq,
    }

def main(site_slug: str = "", force_reset: bool = False):
    """Bootstrap a new site (or re-run safely)."""

    if Path(SITE_PATH).exists() and Path(TITLES_POOL_PATH).exists() and not force_reset:
        print(f"[bootstrap] Existing bootstrap detected for '{site_slug or Path.cwd().name}'. Skipping LLM calls.")
        return

    Path("scripts").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    if not NICHE:
        raise SystemExit("BOOTSTRAP_NICHE is required (e.g. 'work anxiety', 'caravan towing safety', etc).")

    if force_reset:
        for p in (Path(SITE_PATH), Path(TITLES_POOL_PATH), Path(MANIFEST_PATH)):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    existing = load_yaml(SITE_PATH)

    preserved_hubs = None
    if isinstance(existing, dict):
        preserved_hubs = (existing.get('taxonomy', {}) or {}).get('hubs')
    preserve_taxonomy = bool(preserved_hubs) and not force_reset

    # Only call Phase 1 LLM if we need identity fields or we are generating taxonomy.
    identity_missing = True
    if isinstance(existing, dict):
        site_sec = existing.get('site', {}) or {}
        theme_sec = existing.get('theme', {}) or {}
        identity_missing = not all([
            bool(site_sec.get('title')),
            bool(site_sec.get('brand')),
            bool(site_sec.get('tagline')),
            bool(site_sec.get('default_meta_description')),
            bool(theme_sec.get('pack')),
        ])


    phase1 = {}
    if (not preserve_taxonomy) or identity_missing:
        system = (
            "You are a careful site-bootstrapper for an evergreen informational website.\n"
            "Hard rules:\n"
            "- No dates/years or time-sensitive words (recent/currently/this year/today/now).\n"
            "- No prices/cost claims, no statistics, no 'studies show', no numbers-as-facts.\n"
            "- No medical/legal/financial advice. No guarantees. No first-person.\n"
            "- Output a single valid JSON object only. No markdown fences. No extra text.\n"
        )

        # ── Phase 1: Site identity + hubs (small payload, reliable) ──
        phase1_user = {
            "task": "Create site identity and topic taxonomy for an evergreen informational website.",
            "inputs": {
                "niche": NICHE,
                "tone": TONE or "neutral, calm, beginner-friendly",
            },
            "allowed_theme_packs": THEME_PACKS,
            "required_json": {
                "site_title": "string (2-4 words, no punctuation)",
                "brand": "string (same as title or shorter)",
                "tagline": "string (8-14 words, no hype, no promises)",
                "default_meta_description": "string (<= 155 chars, neutral)",
                "theme_pack": "one of allowed_theme_packs",
                "taxonomy": {
                    "hubs": [
                        {"id": "slug-style-id", "label": "Human Label", "description": "1-sentence description"}
                    ]
                },
            },
            "notes": [
                "Hub constraints: 6-10 hubs total.",
                "Hub ids must be lowercase slug-style (e.g. 'understanding-basics', 'types-of-spread').",
                "Each hub should cover a distinct aspect of the niche.",
            ],
        }

        print("[bootstrap] Phase 1: Site identity + taxonomy...")
        try:
            phase1 = kimi_json(
                system=system,
                user=json.dumps(phase1_user, ensure_ascii=False),
                temperature=TEMPERATURE,
                max_tokens=2048,
            )
        except Exception as e:
            print(f"[bootstrap] Phase 1 failed ({type(e).__name__}: {e}). Using deterministic fallback.")
            phase1 = _deterministic_bootstrap_fallback(NICHE, TITLE_COUNT)
    else:
        print("[bootstrap] Phase 1: Skipped taxonomy/identity generation (existing taxonomy + identity present).")

    # Prefer existing identity fields if present (do not overwrite on reruns)
    existing_site = (existing.get("site", {}) or {}) if isinstance(existing, dict) else {}
    existing_theme = (existing.get("theme", {}) or {}) if isinstance(existing, dict) else {}

    theme_pack = (existing_theme.get("pack") or "").strip() if preserve_taxonomy else ""
    if not theme_pack:
        theme_pack = (phase1.get("theme_pack") or "").strip()
    if theme_pack not in THEME_PACKS:
        theme_pack = "modern-sans"

    site_title = (existing_site.get("title") or phase1.get("site_title") or "Evergreen Site").strip()
    brand = (existing_site.get("brand") or phase1.get("brand") or site_title).strip()
    tagline = (existing_site.get("tagline") or phase1.get("tagline") or "Calm, practical explanations — not advice.").strip()
    meta = (existing_site.get("default_meta_description") or phase1.get("default_meta_description") or tagline).strip()

    base_url = (existing_site.get("base_url") or "").strip()
    if not base_url:
        base_url = (os.getenv("BOOTSTRAP_BASE_URL") or "https://YOUR-SITE.pages.dev/").strip()

    # Use existing taxonomy if present unless force_reset was requested
    if preserve_taxonomy and isinstance(preserved_hubs, list) and preserved_hubs:
        hubs = preserved_hubs
    else:
        # Sanitize hubs from taxonomy.hubs (new) or top-level hubs (old/fallback)
        raw_hubs = None
        taxonomy = phase1.get("taxonomy")
        if isinstance(taxonomy, dict):
            raw_hubs = taxonomy.get("hubs")
        if not raw_hubs:
            raw_hubs = phase1.get("hubs")
        hubs = None
        if isinstance(raw_hubs, list) and raw_hubs:
            # Validate each hub is a dict with id and label
            valid_hubs = []
            for h in raw_hubs:
                if isinstance(h, dict) and h.get("id") and h.get("label"):
                    valid_hubs.append({"id": str(h["id"]).strip(), "label": str(h["label"]).strip()})
            if len(valid_hubs) >= 3:
                hubs = valid_hubs

        if not hubs:
            hubs = (existing.get("taxonomy", {}) or {}).get("hubs") if isinstance(existing, dict) else None

        if not hubs or not isinstance(hubs, list) or len(hubs) < 3:
            hubs = [
                {"id": "basics", "label": "Basics"},
                {"id": "how-it-works", "label": "How It Works"},
                {"id": "gear-setup", "label": "Gear & Setup"},
                {"id": "troubleshooting", "label": "Troubleshooting"},
                {"id": "comparisons", "label": "Comparisons"},
            ]

    # ── Phase 2: Generate catalog in batches ──
    # 300 titles at ~20 tokens each won't fit in one call.
    # Generate in batches of 100, each batch gets its own LLM call.
    hub_ids = [h["id"] for h in hubs]
    hub_labels_str = ", ".join([f'{h["id"]} ({h["label"]})' for h in hubs])
    batch_size = 100
    num_batches = max(1, (TITLE_COUNT + batch_size - 1) // batch_size)

    all_catalog: list[tuple[str, str]] = []
    seen_titles: set[str] = set()

    # Check if phase1 already returned some catalog/titles (fallback case)
    existing_catalog = phase1.get("catalog") or []
    existing_titles = phase1.get("titles_pool") or []

    if isinstance(existing_catalog, list) and existing_catalog:
        for item in existing_catalog:
            if isinstance(item, dict):
                t = (item.get("title") or "").strip()
                h = (item.get("hub") or "").strip()
                if t and h and t.lower() not in seen_titles:
                    seen_titles.add(t.lower())
                    all_catalog.append((t, h))
    elif isinstance(existing_titles, list) and existing_titles:
        # Fallback had flat titles — assign round-robin
        for i, t in enumerate(existing_titles):
            if isinstance(t, str) and t.strip() and t.strip().lower() not in seen_titles:
                seen_titles.add(t.strip().lower())
                all_catalog.append((t.strip(), hub_ids[i % len(hub_ids)]))

    remaining = TITLE_COUNT - len(all_catalog)
    if remaining > 0:
        print(f"[bootstrap] Phase 2: Generating {remaining} catalog titles in {num_batches} batches...")
        for batch_i in range(num_batches):
            batch_need = min(batch_size, TITLE_COUNT - len(all_catalog))
            if batch_need <= 0:
                break

            # Per-hub target for this batch
            per_hub = max(batch_need // len(hub_ids), 3)
            catalog_prompt = {
                "task": f"Generate {batch_need} unique page titles assigned to topic hubs.",
                "niche": NICHE,
                "hubs": hub_labels_str,
                "rules": [
                    "Each title: question-style, evergreen, beginner-friendly, globally relevant.",
                    "No dates/years, prices, stats, brand names, advice framing.",
                    f"Distribute evenly: aim for ~{per_hub} titles per hub. No hub should have less than {per_hub - 2}.",
                    "Do NOT repeat any of these already-used titles: " + json.dumps([t for t, _ in all_catalog[-50:]]) if all_catalog else "No prior titles yet.",
                ],
                "required_json": {
                    "catalog": [{"title": "string", "hub": "one of the hub ids listed above"}],
                },
            }

            try:
                batch_out = kimi_json(
                    system=system,
                    user=json.dumps(catalog_prompt, ensure_ascii=False),
                    temperature=TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception as e:
                print(f"[bootstrap] Batch {batch_i+1} failed ({e}), skipping.")
                continue

            batch_items = batch_out.get("catalog") or batch_out.get("titles_pool") or []
            added = 0
            if isinstance(batch_items, list):
                for item in batch_items:
                    if isinstance(item, dict):
                        t = (item.get("title") or "").strip()
                        h = (item.get("hub") or "").strip()
                        if t and h and t.lower() not in seen_titles and h in hub_ids:
                            seen_titles.add(t.lower())
                            all_catalog.append((t, h))
                            added += 1
                    elif isinstance(item, str) and item.strip():
                        # Flat title — assign round-robin
                        t = item.strip()
                        if t.lower() not in seen_titles:
                            seen_titles.add(t.lower())
                            all_catalog.append((t, hub_ids[len(all_catalog) % len(hub_ids)]))
                            added += 1

            print(f"[bootstrap] Batch {batch_i+1}: added {added} titles (total: {len(all_catalog)})")
            if len(all_catalog) >= TITLE_COUNT:
                break
            time.sleep(1)  # Rate-limit courtesy

    # Clean up catalog
    cleaned: list[tuple[str, str]] = []
    for t, h in all_catalog:
        t = _titlecase_for_display(_normalize_quotes_list([t])[0] if t else t)
        cleaned.append((t, h))
    cleaned = cleaned[:TITLE_COUNT]

    titles = [t for t, _ in cleaned]
    title_hub_overrides = {t: h for t, h in cleaned}

    # Log catalog summary
    if title_hub_overrides:
        hub_counts: dict[str, int] = {}
        for h in title_hub_overrides.values():
            hub_counts[h] = hub_counts.get(h, 0) + 1
        print(f"[bootstrap] Catalog: {len(titles)} titles across {len(hub_counts)} hubs")
        for hid, cnt in sorted(hub_counts.items(), key=lambda x: x[0]):
            print(f"  Hub '{hid}': {cnt} titles")
    else:
        print(f"[bootstrap] Titles: {len(titles)} (no hub assignments)")

    wc_min, wc_max, ideal_min, ideal_max = 900, 1900, 1100, 1600

    site_cfg = existing if isinstance(existing, dict) else {}
    site_cfg.setdefault("site", {})
    site_cfg.setdefault("theme", {})
    site_cfg.setdefault("taxonomy", {})
    site_cfg.setdefault("generation", {})
    site_cfg.setdefault("internal_linking", {})
    site_cfg.setdefault("ads", {})
    site_cfg.setdefault("gates", {})

    site_cfg["site"].update({
        "title": site_title,
        "brand": brand,
        "language_code": site_cfg["site"].get("language_code") or "en-us",
        "base_url": base_url,
        "default_meta_description": meta,
        "tagline": tagline,
        "niche": NICHE,
    })

    site_cfg["theme"].update({
        "pack": theme_pack,
        "font_sans": site_cfg["theme"].get("font_sans") or "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif",
        "font_serif": site_cfg["theme"].get("font_serif") or "ui-serif, Georgia, Cambria, 'Times New Roman', Times, serif",
        "content_max": site_cfg["theme"].get("content_max") or "74ch",
        "radius": site_cfg["theme"].get("radius") or "16px",
    })

    site_cfg["taxonomy"]["hubs"] = hubs

    gen = site_cfg["generation"]
    gen.setdefault("forbidden_words", [])
    core_forbidden = [
        "diagnose", "diagnosis", "prescribed", "guaranteed", "sue",
        "treatment", "treat", "cure", "therapist", "lawyer", "accountant",
    ]
    merged = list(dict.fromkeys((gen.get("forbidden_words") or []) + core_forbidden))
    gen["forbidden_words"] = merged
    gen["page_types"] = gen.get("page_types") or ["explainer", "checklist", "myth-vs-reality", "comparison", "troubleshooting"]
    gen["outline_h2"] = DEFAULT_OUTLINE_H2
    gen["wordcount"] = {"min": wc_min, "ideal_min": ideal_min, "ideal_max": ideal_max, "max": wc_max}

    il = site_cfg["internal_linking"]
    il.setdefault("enabled", True)
    il["min_links"] = max(int(il.get("min_links") or 6), 6)
    il["forbid_external"] = True

    gates = site_cfg["gates"]
    gates["wordcount_min"] = wc_min
    gates["wordcount_max"] = wc_max
    gates["min_internal_links"] = 6
    gates["forbid_external_links"] = True
    # FIX: Add faq_min/faq_max to gates output (was missing)
    gates["faq_min"] = gates.get("faq_min", 4)
    gates["faq_max"] = gates.get("faq_max", 8)

    save_yaml(SITE_PATH, site_cfg)
    # Create hub landing pages for navigation (content/hubs/<id>/_index.md)
    ensure_hub_index_pages(hubs)
    patch_hugo_yaml(site_cfg)
    # Trim to TITLE_COUNT (the model may return more than requested)
    if len(titles) > TITLE_COUNT:
        titles = titles[:TITLE_COUNT]

    write_titles_pool(Path(TITLES_POOL_PATH), titles, title_hub_overrides)

    ensure_manifest_reset()

    receipt = {
        "niche": NICHE,
        "tone": TONE,
        "site_title": site_title,
        "theme_pack": theme_pack,
        "title_count_written": len(titles),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_hash": hashlib.sha256(("|".join(DEFAULT_OUTLINE_H2) + f"|{wc_min}-{wc_max}").encode("utf-8")).hexdigest()[:16],
        "llm_used": bool(phase1 and phase1.get("site_title")),
    }
    rc = Path("scripts/bootstrap_receipt.json")
    if not rc.exists():
        rc.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n===== BOOTSTRAP SUMMARY =====")
    print(f"Niche: {NICHE}")
    print(f"Site title: {site_title}")
    print(f"Theme pack: {theme_pack}")
    print(f"Titles written: {len(titles)} (target {TITLE_COUNT})")
    print("Receipt: scripts/bootstrap_receipt.json")
    print("=============================\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-slug", default="", help="Folder under sites/, e.g. home-espresso-basics")
    ap.add_argument("--force-reset", action="store_true", help="Wipe existing site.yaml / titles pool / bootstrap receipt before bootstrapping")
    args = ap.parse_args()
    main(site_slug=args.site_slug.strip(), force_reset=bool(args.force_reset))

