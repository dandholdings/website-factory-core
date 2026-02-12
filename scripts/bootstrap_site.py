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

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

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

def write_titles_pool(titles: list):
    TITLES_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    uniq = []
    seen = set()
    for t in titles:
        t = (t or "").strip()
        if not t:
            continue
        key = re.sub(r"\s+", " ", t.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    TITLES_POOL_PATH.write_text("\n".join(uniq) + "\n", encoding="utf-8")

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
        {"id": "basics", "label": "Basics"},
        {"id": "how-it-works", "label": "How It Works"},
        {"id": "gear-setup", "label": "Gear & Setup"},
        {"id": "troubleshooting", "label": "Troubleshooting"},
        {"id": "comparisons", "label": "Comparisons"},
    ]

    base = n.lower().strip() or "this topic"
    templates = [
        "What is {x}?",
        "How does {x} work?",
        "Beginner mistakes with {x}",
        "Common misconceptions about {x}",
        "{x}: key terms explained",
        "How to choose {y} for {x}",
        "{y} vs {z} for {x}",
        "Signs you are overcomplicating {x}",
        "A simple checklist for {x}",
        "Troubleshooting {x}: common problems",
        "What to do when {x} tastes bitter",
        "What to do when {x} tastes sour",
        "How to dial in {x} without chasing perfect",
        "How grind size affects {x}",
        "How water temperature affects {x}",
        "How dose and yield affect {x}",
        "How to keep {x} consistent day to day",
        "How to clean and maintain {y} for {x}",
        "How to set up a simple {x} routine",
        "How to read feedback from taste in {x}",
    ]
    ys = ["a grinder", "a machine", "beans", "water", "a scale", "a workflow"]
    zs = ["a manual grinder", "an electric grinder", "dark roast", "light roast", "tap water", "filtered water"]

    titles = []
    rnd = random.Random(seed)
    while len(titles) < max(40, min(title_count, 600)):
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

    return {
        "site_title": site_title,
        "brand": brand,
        "tagline": tagline[:120],
        "default_meta_description": meta,
        "theme_pack": theme_pack,
        "hubs": hubs,
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

    system = (
        "You are a careful site-bootstrapper for an evergreen informational website.\n"
        "Hard rules:\n"
        "- No dates/years or time-sensitive words (recent/currently/this year/today/now).\n"
        "- No prices/cost claims, no statistics, no 'studies show', no numbers-as-facts.\n"
        "- No medical/legal/financial advice. No guarantees. No first-person.\n"
        "- Output a single valid JSON object only. No markdown fences. No extra text.\n"
    )

    user = {
        "task": "Create site identity + theme choice + titles pool for an evergreen website.",
        "inputs": {
            "niche": NICHE,
            "tone": TONE or "neutral, calm, beginner-friendly",
            "title_count": TITLE_COUNT,
        },
        "allowed_theme_packs": THEME_PACKS,
        "required_json": {
            "site_title": "string (2-4 words, no punctuation)",
            "brand": "string (same as title or shorter)",
            "tagline": "string (8-14 words, no hype, no promises)",
            "default_meta_description": "string (<= 155 chars, neutral)",
            "theme_pack": "one of allowed_theme_packs",
            "hubs": [
                {"id": "work-career|money-stress|burnout-load|milestones|social-norms", "label": "string"}
            ],
            "titles_pool": [f"list of {TITLE_COUNT} unique page titles, question-style, evergreen, global-friendly"],
        },
        "notes": [
            "Titles must avoid dates/years, prices, stats, brand names, and advice framing.",
            "Prefer novice-friendly, definitional and comparison topics.",
            "Keep titles short and specific; no clickbait.",
            f"You MUST produce at least {TITLE_COUNT} titles in the titles_pool array.",
        ],
    }

    try:
        out = kimi_json(
            system=system,
            user=json.dumps(user, ensure_ascii=False),
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        print(f"[bootstrap] WARNING: LLM bootstrap failed ({type(e).__name__}: {e}). Using deterministic fallback.")
        out = _deterministic_bootstrap_fallback(NICHE, TITLE_COUNT)

    theme_pack = out.get("theme_pack")
    if theme_pack not in THEME_PACKS:
        theme_pack = "modern-sans"

    site_title = (out.get("site_title") or "Evergreen Site").strip()
    brand = (out.get("brand") or site_title).strip()
    tagline = (out.get("tagline") or "Calm, practical explanations — not advice.").strip()
    meta = (out.get("default_meta_description") or tagline).strip()

    base_url = (existing.get("site", {}) or {}).get("base_url") if isinstance(existing, dict) else None
    if not base_url:
        base_url = (os.getenv("BOOTSTRAP_BASE_URL") or "https://YOUR-SITE.pages.dev/").strip()

    # FIX: Sanitize hubs — LLM sometimes returns a dict, a string, or malformed list items.
    raw_hubs = out.get("hubs")
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

    # FIX: Sanitize titles_pool — ensure it's a list of strings, backfill if too few.
    raw_titles = out.get("titles_pool")
    titles_from_llm = []
    if isinstance(raw_titles, list):
        for t in raw_titles:
            if isinstance(t, str) and t.strip():
                titles_from_llm.append(t.strip())

    # If LLM gave us fewer than 50 titles (truncation/failure), backfill with deterministic ones
    if len(titles_from_llm) < 50:
        print(f"[bootstrap] LLM returned only {len(titles_from_llm)} titles (need ≥50). Backfilling with deterministic titles.")
        fallback = _deterministic_bootstrap_fallback(NICHE, TITLE_COUNT)
        fallback_titles = fallback.get("titles_pool") or []
        # Merge: LLM titles first, then deterministic ones to fill up
        seen = set(t.lower().strip() for t in titles_from_llm)
        for t in fallback_titles:
            if t.lower().strip() not in seen:
                titles_from_llm.append(t)
                seen.add(t.lower().strip())
            if len(titles_from_llm) >= TITLE_COUNT:
                break
    titles = titles_from_llm

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
    il["min_links"] = max(int(il.get("min_links") or 3), 3)
    il["forbid_external"] = True

    gates = site_cfg["gates"]
    gates["wordcount_min"] = wc_min
    gates["wordcount_max"] = wc_max
    gates["min_internal_links"] = 3
    gates["forbid_external_links"] = True
    # FIX: Add faq_min/faq_max to gates output (was missing)
    gates["faq_min"] = gates.get("faq_min", 4)
    gates["faq_max"] = gates.get("faq_max", 8)

    save_yaml(SITE_PATH, site_cfg)
    patch_hugo_yaml(site_cfg)

    write_titles_pool(titles)

    ensure_manifest_reset()

    receipt = {
        "niche": NICHE,
        "tone": TONE,
        "site_title": site_title,
        "theme_pack": theme_pack,
        "title_count_written": len(titles),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_hash": hashlib.sha256(("|".join(DEFAULT_OUTLINE_H2) + f"|{wc_min}-{wc_max}").encode("utf-8")).hexdigest()[:16],
        "llm_used": bool(out and out.get("titles_pool")),
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
