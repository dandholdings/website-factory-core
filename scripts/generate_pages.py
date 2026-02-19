import os
import sys
import argparse
import json
import time
import re
import random
import hashlib
import requests
from datetime import date
from pathlib import Path
import yaml

# Shared LLM client module — canonical implementations of llm_json(),
# parse_json_strict_or_extract(), etc. live here. The inline copies below
# are kept for backward compatibility but should be migrated to use
# llm_client directly in a future cleanup pass.
try:
    from llm_client import (
        llm_json as _shared_llm_json,
        parse_json_strict_or_extract as _shared_parse_json,
        safe_write_llm_dump as _shared_write_dump,
        read_markdown_frontmatter as _shared_read_fm,
        slugify as _shared_slugify,
    )
    _HAS_SHARED = True
except ImportError:
    _HAS_SHARED = False

START_TIME = time.time()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "site"


def _apply_site_root_early():
    """Allow running core scripts from inside thin-repo.

    Priority:
      1) --site-root <path>
      2) --site-slug <slug>  -> chdir sites/<slug>
      3) SITE_SLUG env (if set) -> chdir sites/<SITE_SLUG>

    If slug is blank, we derive it from BOOTSTRAP_NICHE/NICHE.
    """
    # 1) explicit --site-root
    if "--site-root" in sys.argv:
        i = sys.argv.index("--site-root")
        if i + 1 < len(sys.argv) and sys.argv[i + 1]:
            os.chdir(sys.argv[i + 1])
            return

    # 2) --site-slug
    slug = ""
    if "--site-slug" in sys.argv:
        i = sys.argv.index("--site-slug")
        if i + 1 < len(sys.argv):
            slug = (sys.argv[i + 1] or "").strip()

    # 3) env
    if not slug:
        slug = (os.getenv("SITE_SLUG", "") or "").strip()

    if not slug:
        niche = (os.getenv("BOOTSTRAP_NICHE", "") or os.getenv("NICHE", "")).strip()
        slug = _slugify(niche)

    if slug:
        root = Path("sites") / slug
        root.mkdir(parents=True, exist_ok=True)
        os.chdir(root)


_apply_site_root_early()


def _parse_site_root() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--site-root", default=".")
    p.add_argument("--site-slug", default="")
    args, _ = p.parse_known_args()
    return Path(args.site_root).resolve()

SITE_ROOT = Path(_parse_site_root()).resolve()

def _sr(rel: str) -> Path:
    return (SITE_ROOT / rel).resolve()


# --- Paths & runtime knobs --------------------------------------------
CONTENT_ROOT: Path = _sr("content/pages")
DATA_DIR: Path = _sr("data")
SCRIPTS_DIR: Path = _sr("scripts")

# Common files written by bootstrap / consumed by generator
SITE_CONFIG_PATH_DEFAULT: Path = DATA_DIR / "site.yaml"
TITLES_POOL_PATH: Path = SCRIPTS_DIR / "titles_pool.txt"
MANIFEST_PATH: Path = SCRIPTS_DIR / "manifest.json"
PLAN_PATH: Path = DATA_DIR / "plan.yaml"

# Diagnostics / retry bookkeeping
FAILED_TITLES_PATH: Path = SCRIPTS_DIR / "failed_titles.tsv"
RETRY_TITLES_PATH: Path = SCRIPTS_DIR / "retry_titles.tsv"

def _safe_int(val, default: int) -> int:
    """Parse an int from a string, returning default on any failure."""
    try:
        v = (val or "").strip() if isinstance(val, str) else val
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


# Generation behavior
GEN_VERSION: str = os.getenv("GEN_VERSION", "1")
FACTORY_MODE: str = (os.getenv("FACTORY_MODE", "generate") or "generate").strip().lower()
BACKFILL_METADATA: bool = (os.getenv("BACKFILL_METADATA", "0").strip() == "1")
FAIL_STOP: int = _safe_int(os.getenv("FAIL_STOP"), 15)

# Retry / pacing
HTTP_MAX_TRIES: int = _safe_int(os.getenv("HTTP_MAX_TRIES") or os.getenv("KIMI_HTTP_MAX_TRIES"), 6)
BACKOFF_BASE: float = float(os.getenv("BACKOFF_BASE", os.getenv("KIMI_BACKOFF_BASE", "1.7")) or "1.7")
MAX_ATTEMPTS: int = _safe_int(os.getenv("MAX_ATTEMPTS"), 25)
SLEEP_SECONDS: float = float(os.getenv("SLEEP_SECONDS", "0.3") or "0.3")

# Per-run limits
PAGES_PER_RUN: int = _safe_int(os.getenv("PAGES_PER_RUN"), 5)
PER_TITLE_CAP: int = _safe_int(os.getenv("PER_TITLE_CAP"), 2)

# Regen filters (optional)
REGEN_RULE: str = (os.getenv("REGEN_RULE", "") or "").strip()
REGEN_HUB: str = (os.getenv("REGEN_HUB", "") or "").strip()
REGEN_SLUGS: str = (os.getenv("REGEN_SLUGS", "") or "").strip()


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

# --- Runtime knobs (shared across providers) -----------------------------
REQUEST_TIMEOUT = _safe_int(os.getenv("REQUEST_TIMEOUT"), 180)
CONNECT_TIMEOUT = _safe_int(os.getenv("CONNECT_TIMEOUT"), 20)

MAX_OUTPUT_TOKENS = _safe_int(os.getenv("MAX_OUTPUT_TOKENS"), 4096)

# FIX: Temperature — only clamp for Moonshot/Kimi. Let Gemini use any value.
try:
    _t_raw = os.getenv("TEMPERATURE", "0.25").strip()  # Lower default for more deterministic JSON
    TEMPERATURE = float(_t_raw)
except Exception:
    TEMPERATURE = 0.25

# Only force temperature=1 for Moonshot/Kimi (legacy constraint).
# Gemini Flash 2.5 works fine with 0.7 or any valid float.
if PROVIDER == "moonshot" and TEMPERATURE != 1:
    TEMPERATURE = 1

EMPTY_BACKOFF_BASE = float(os.getenv("EMPTY_BACKOFF_BASE", os.getenv("KIMI_EMPTY_BACKOFF_BASE", "1.25")))
EMPTY_BACKOFF_CAP = float(os.getenv("EMPTY_BACKOFF_CAP", os.getenv("KIMI_EMPTY_BACKOFF_CAP", "3.0")))

HEADERS = {"Content-Type": "application/json"}


def _safe_write_kimi_dump(
    kind: str,
    attempt: int,
    *,
    content: str = "",
    envelope=None,
    http_status=None,
    http_text=None,
    payload=None,
):
    """Write a debug dump to help diagnose provider hiccups.

    Never includes API keys. Keeps content size bounded.
    """
    try:
        import hashlib as _hl
        log_dir = Path("scripts") / "_logs" / "llm"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = _hl.sha1(f"{ts}-{kind}-{attempt}-{random.random()}".encode("utf-8")).hexdigest()[:10]
        p = log_dir / f"{ts}-{kind}-a{attempt+1}-{rid}.json"

        def _trim(s, n: int) -> str:
            s = "" if s is None else str(s)
            s = s.replace("\x00", "")
            return s[:n]

        # FIX: Redact all sensitive keys from payload including x-goog-api-key
        safe_payload = None
        if isinstance(payload, dict):
            safe_payload = {}
            for k, v in payload.items():
                if k.lower() in ("api_key", "authorization", "x-goog-api-key"):
                    safe_payload[k] = "[REDACTED]"
                else:
                    safe_payload[k] = v

        # FIX: Also scrub any key= params from http_text that might contain API keys
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


def resolve_site_config_path() -> str:
    """Prefer the single contract at data/site.yaml.
    Backward compatible: fall back to scripts/site_config.yaml if needed.
    """
    p = (os.getenv("SITE_CONFIG_PATH") or "data/site.yaml").strip()
    if os.path.isfile(p):
        return p
    fallback = "scripts/site_config.yaml"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Site config not found: {p} (and no {fallback} fallback)")

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_manifest_shape(m: dict) -> dict:
    if not isinstance(m, dict):
        return {"used_titles": [], "generated_this_run": []}
    m.setdefault("used_titles", [])
    m.setdefault("generated_this_run", [])
    return m

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {"used_titles": [], "generated_this_run": []}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return ensure_manifest_shape(json.load(f))

def save_manifest(m):
    m = ensure_manifest_shape(m)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

def slugify(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:80].strip("-")

def load_titles():
    """Load titles from titles_pool.txt. Supports both plain and hub-pinned formats.
    Returns list of plain title strings. Use load_titles_with_hubs() for hub info."""
    if not TITLES_POOL_PATH.exists():
        print(f"[warn] titles_pool.txt not found at {TITLES_POOL_PATH}")
        return []
    try:
        with open(TITLES_POOL_PATH, "r", encoding="utf-8") as f:
            titles = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "hub_id\tTitle" or plain "Title"
                if "\t" in line:
                    _, title = line.split("\t", 1)
                    titles.append(title.strip())
                else:
                    titles.append(line)
            return titles
    except Exception as e:
        print(f"[warn] Failed to read titles_pool.txt: {e}")
        return []


def load_titles_with_hubs():
    """Load titles with hub assignments. Returns dict {title: hub_id}."""
    if not TITLES_POOL_PATH.exists():
        return {}
    try:
        mapping = {}
        with open(TITLES_POOL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    hub, title = line.split("\t", 1)
                    mapping[title.strip()] = hub.strip()
        return mapping
    except Exception:
        return {}

def load_plan(path: str) -> dict:
    if not os.path.isfile(path):
        return {"items": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"items": []}

def save_plan(path: str, plan: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan or {"items": []}, f, sort_keys=False, allow_unicode=True)

def parse_json_strict_or_extract(raw: str) -> dict:
    """Parse a JSON object from model output. Robust against fences, prose, truncation."""
    raw = (raw or "").strip()
    if not raw:
        raise json.JSONDecodeError("Empty model output", raw, 0)

    def _strip_fences(s: str) -> str:
        s2 = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
        s2 = re.sub(r"\s*```\s*$", "", s2.strip())
        return s2.strip()

    cand = _strip_fences(raw)

    # Try direct parse first
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass

    # Walk for the first balanced JSON object
    s = cand
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", cand, 0)

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

    snippet = s[start:end]
    obj = json.loads(snippet)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Top-level JSON must be an object", snippet, 0)
    return obj

def kimi_json(system: str, user: str, temperature: float = 0.7, max_tokens: int = 1400) -> dict:
    """Request a JSON object from the configured LLM provider.

    Supports:
      - Gemini (preferred) via GEMINI_API_KEY / GEMINI_MODEL
      - Moonshot/Kimi (legacy) via MOONSHOT_API_KEY / MOONSHOT_BASE_URL / KIMI_MODEL
    """
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
        # FIX: Use x-goog-api-key header instead of URL param (security + no key in logs)
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
                # FIX: Use responseMimeType to force JSON output from Gemini
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


def call_kimi(system: str, prompt: str) -> dict:
    """Backwards-compatible wrapper for the site factory."""
    return kimi_json(system=system, user=prompt, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)

def build_internal_link_hints(content_root="content/pages", limit: int = 40) -> str:
    """Build a curated list of existing internal links for the model to use.
    
    Groups by hub so the model can prioritise same-hub links.
    """
    root = Path(content_root)
    items = []
    for md in root.glob("*/index.md"):
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _ = read_markdown_frontmatter(raw)
        slug = fm.get("slug") or md.parent.name
        title = fm.get("title") or slug.replace("-", " ").title()
        hub = fm.get("hub") or "general"
        items.append((str(title).strip(), str(slug).strip(), str(hub).strip()))
    items = sorted(items, key=lambda x: x[2])  # group by hub
    items = items[:limit]
    
    # Group by hub for clearer model instructions
    by_hub = {}
    for t, s, h in items:
        if t and s:
            by_hub.setdefault(h, []).append(f"- [{t}](/pages/{s}/)")
    
    lines = []
    for hub_id in sorted(by_hub.keys()):
        lines.append(f"### Hub: {hub_id}")
        lines.extend(by_hub[hub_id])
    return "\n".join(lines)


def _generate_sibling_link_hints(titles_pool_path: Path, limit: int = 20) -> str:
    """Generate internal link hints from the titles pool for fresh bootstraps."""
    if not titles_pool_path.exists():
        return ""
    try:
        raw_lines = [t.strip() for t in titles_pool_path.read_text(encoding="utf-8").splitlines() if t.strip()]
    except Exception:
        return ""
    if not raw_lines:
        return ""
    # Parse hub\tTitle format
    titles = []
    for line in raw_lines:
        if "\t" in line:
            _, title = line.split("\t", 1)
            titles.append(title.strip())
        else:
            titles.append(line)
    # Pick a random sample and convert to slug format
    sample = random.sample(titles, min(limit, len(titles)))
    items = []
    for t in sample:
        s = slugify(t)
        if s:
            items.append(f"- [{t}](/pages/{s}/)")
    return "\n".join(items)


def build_prompts(cfg: dict):
    site = cfg.get("site", {}) if isinstance(cfg, dict) else {}
    taxonomy = cfg.get("taxonomy", {}) if isinstance(cfg, dict) else {}
    generation = cfg.get("generation", {}) if isinstance(cfg, dict) else {}

    brand = site.get("brand") or site.get("title") or "Evergreen Site"
    hub_defs = taxonomy.get("hubs") or []
    hubs = [h.get("id") for h in hub_defs if isinstance(h, dict) and h.get("id")] or [
        "basics", "how-it-works", "contexts", "misconceptions", "related-concepts"
    ]

    # Build hub → clusters lookup for cluster-aware generation
    hub_clusters = {}
    for h in hub_defs:
        if isinstance(h, dict) and h.get("id") and h.get("clusters"):
            cluster_ids = [c.get("id") for c in h["clusters"] if isinstance(c, dict) and c.get("id")]
            if cluster_ids:
                hub_clusters[h["id"]] = cluster_ids

    page_types = generation.get("page_types") or [
        "is-it-normal", "checklist", "red-flags", "myth-vs-reality", "explainer"
    ]

    wc = generation.get("wordcount") or {}
    wc_min = int(wc.get("min") or 900)
    wc_ideal_min = int(wc.get("ideal_min") or 1100)
    wc_ideal_max = int(wc.get("ideal_max") or 1600)
    wc_max = int(wc.get("max") or 1900)

    forbidden = generation.get("forbidden_words") or []
    forbidden_str = ", ".join(forbidden) if forbidden else "diagnose, diagnosis, prescribed, guaranteed, sue"

    outline = generation.get("outline_h2") or [
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
    outline_md = "\n".join([f"{i+1}. ## {h}" for i, h in enumerate(outline)])

    closing_templates = generation.get("closing_reassurance_templates") or []
    closing_hint = ""
    if closing_templates:
        closing_hint = "Choose ONE closing reassurance line in a similar style to these:\n- " + "\n- ".join(closing_templates[:3])

    # FIX: Stronger JSON-only instruction for Gemini Flash 2.5
    system = f"""You write calm, reassuring evergreen content for the site "{brand}".
NO medical, legal, or financial advice. Avoid diagnosing. Avoid giving instructions like a professional.
CRITICAL: Never use first-person pronouns (I, we, we're, we've, my, mine, me). Use "people", "a person", "individuals", or "you" instead.
CRITICAL: Every paragraph must be 2-3 sentences maximum. Never write 4+ sentence paragraphs.
Forbidden words/phrases: {forbidden_str}.
Return a single valid JSON object only. No markdown fences. No extra text before or after the JSON.
"""

    # Build cluster instruction block
    cluster_instruction = ""
    if hub_clusters:
        cluster_lines = []
        for hid, cids in hub_clusters.items():
            cluster_lines.append(f"  Hub \"{hid}\" clusters: {' | '.join(cids)}")
        cluster_instruction = f"""
cluster (REQUIRED if hub has clusters defined below, else omit this field):
{chr(10).join(cluster_lines)}
If your chosen hub has clusters, you MUST set cluster to one of its allowed values."""

    page_prompt = f"""Return ONLY a JSON object with these keys:
title
summary (one sentence reassurance; also used as meta description)
description (<= 160 chars, no quotes)
hub (one of: { " | ".join(hubs) })
page_type (one of: { " | ".join(page_types) })
{cluster_instruction}
closing_reassurance (one short, gentle line; NOT advice)
body_md (markdown only; must include the exact H2 headings below)

Use these H2 sections IN THIS EXACT ORDER (do NOT reorder, do NOT skip any, do NOT add extras, do NOT duplicate any):
{outline_md}

CRITICAL: Use each H2 heading EXACTLY ONCE. Do NOT repeat any heading. The article must have exactly {len(outline)} H2 sections, no more, no less.

=== UNIQUE PAYLOAD BLOCK (CRITICAL — pages rejected without this) ===

Immediately after the "## Intro" section (before "## Definitions and key terms"), you MUST insert exactly one "payload block" that is specific to the topic. Choose the best format:

A) "Quick-start snapshot" — 5-7 short bullet points summarising the key takeaways
B) "If X, try Y" chooser — a markdown table with 3-6 rows matching situations to approaches
C) "Self-check checklist" — 7-12 items as a checkbox-style list using "- [ ]" syntax
D) "Starter templates" — 3-5 short template sentences people can adapt

Rules for the payload block:
- Start it with an H3 heading: ### Quick-start snapshot, ### If–then chooser, ### Self-check checklist, or ### Starter templates
- 80-180 words
- Must include at least 1 internal link to another page in the same hub
- Must be specific to the topic (not generic advice)
- Must NOT reuse wording from other sections

=== HARD RULES (violation = rejected page) ===

INTERNAL LINKS (CRITICAL — pages are rejected if this fails):
- You MUST include at least 6 internal links total in body_md.
- CONTEXTUAL LINKS: Place at least 3 internal links naturally within paragraph text across the body sections (Intro through Common misconceptions). These must flow naturally in-sentence, e.g. "This pattern resembles [emotional mirroring in groups](/pages/emotional-mirroring-in-groups/) and can intensify over time." Do NOT cluster links together — spread them across different sections.
- RELATED SECTION: The "Related topics and deeper reading" section should include 3+ additional internal links as a bulleted list.
- HUB PREFERENCE: Use 3-4 links to pages in the SAME hub + 1-2 links to pages in a different hub. This creates strong topical clusters.
- Use ONLY relative URLs in this format: [anchor text](/pages/slug-here/)
- Anchor text must be descriptive (NEVER "click here", "learn more", "read more").
- ZERO external links allowed. No https:// URLs anywhere.
- NEVER link to the same target twice. Every link URL must be unique.
- NEVER link to the page itself (self-link).

FORBIDDEN LANGUAGE (any occurrence = rejected):
- NO dates or time words. BANNED WORDS: "recent", "recently", "currently", "nowadays", "today", "now", "this year", "last year", "in 20XX", "at the time of writing", "as of", "latest", "emerging", "new research", "growing", "increasingly", "modern", "contemporary". Do not use ANY year numbers (2020, 2024, 2025, etc).
- NO first-person pronouns anywhere in the text. This is a STRICT rule. BANNED WORDS: "I", "I'm", "I've", "we", "we're", "we've", "my", "mine", "me". Write in third person or second person ("you") instead.
  WRONG: "We all experience emotional contagion." → RIGHT: "People experience emotional contagion."
  WRONG: "When we feel sadness, it spreads." → RIGHT: "When a person feels sadness, it spreads."
  WRONG: "Our emotions influence others." → RIGHT: "A person's emotions influence others."
  WRONG: "It connects us as humans." → RIGHT: "It connects people as humans."
  Note: "us" is acceptable ONLY in fixed phrases like "around us" or "between us" where it cannot be rewritten.
- NO guarantees: "guarantee", "100%", "will definitely", "will always", "will never". Avoid absolute claims.
- NO medical/legal terms: "diagnose", "diagnosis", "prescribed", "treatment", "cure", "therapy", "sue", "consult a doctor".
- NO advice framing: "you should", "try this", "make sure to", "it is recommended", "experts say".
- NO external links or URLs starting with http.
- NO affiliate/commercial language: "best", "worst", "buy", "sign up", "download", "review", "sponsored".

TONE & STRUCTURE:
- Neutral, encyclopedic, beginner-friendly. No hype, no fear.
- CRITICAL PARAGRAPH RULE: Every paragraph must be 2–3 sentences maximum. NEVER write a paragraph with 4 or more sentences. If you need more detail, start a new paragraph. This is a hard limit that causes immediate rejection.
- Use ONLY H2 (##) and H3 (###) headings. No H1, no H4+.
- FAQs: 4-6 Q&As using ### headings for each question.
- WORD COUNT (CRITICAL): You MUST write at least {wc_min} words in body_md. Target {wc_ideal_min}–{wc_ideal_max} words. Pages under {wc_min} words are automatically rejected. Each H2 section should have at least 50–80 words. Do NOT be brief — expand with examples, distinctions, and context.
- Do not include the closing reassurance inside body_md; put it in closing_reassurance.
{closing_hint}
"""
    return system, page_prompt, hub_clusters

def choose_close(data: dict, cfg: dict) -> str:
    close = (data.get("closing_reassurance") or "").strip()
    if close:
        return close

    templates = (cfg.get("generation", {}) or {}).get("closing_reassurance_templates") or []
    if templates:
        return random.choice(templates).strip()
    return "If this hit close to home, you're not alone — and you're not failing."

def compute_contract_hash(site_config_path: str) -> str:
    try:
        with open(site_config_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return "unknown"

def read_markdown_frontmatter(md_text: str):
    if not md_text.startswith("---"):
        return {}, md_text
    # Strip the leading "---\n" then split on the closing "\n---"
    after_open = md_text[4:]  # skip "---\n"
    parts = after_open.split("\n---", 1)
    if len(parts) < 2:
        return {}, md_text
    fm_raw = parts[0]
    # Body starts after the closing "---" and optional newlines
    body = parts[1].lstrip("\n").lstrip("\r")
    try:
        fm = yaml.safe_load(fm_raw) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, body

def write_markdown_with_frontmatter(front: dict, body: str) -> str:
    fm_txt = yaml.safe_dump(front or {}, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_txt}\n---\n\n{body.lstrip() if body else ''}"

def iter_content_pages(root_dir) -> list:
    pages = []
    for dirpath, dirnames, filenames in os.walk(str(root_dir)):
        if "index.md" in filenames:
            pages.append(os.path.join(dirpath, "index.md"))
    return pages

def backfill_page_metadata(content_root, contract_hash: str) -> int:
    updated = 0
    for path in iter_content_pages(content_root):
        try:
            raw = open(path, "r", encoding="utf-8").read()
            fm, body = read_markdown_frontmatter(raw)
            if not fm:
                continue
            changed = False
            if "gen_version" not in fm:
                fm["gen_version"] = GEN_VERSION
                changed = True
            if "contract_hash" not in fm or str(fm.get("contract_hash")) != str(contract_hash):
                fm["contract_hash"] = contract_hash
                changed = True
            if "prompt_hash" not in fm:
                fm["prompt_hash"] = "backfilled"
                changed = True
            if changed:
                open(path, "w", encoding="utf-8").write(write_markdown_with_frontmatter(fm, body))
                updated += 1
        except Exception:
            continue
    return updated

def parse_regen_rule(rule: str) -> dict:
    rule = (rule or "").strip()
    if not rule:
        return {}
    if ":" in rule:
        k, v = rule.split(":", 1)
        return {"type": k.strip(), "value": v.strip()}
    return {"type": rule, "value": ""}

def select_pages_for_regen(content_root, contract_hash: str) -> list:
    targets = []
    slugs_set = set([s.strip() for s in REGEN_SLUGS.split(",") if s.strip()]) if REGEN_SLUGS else set()
    rule = parse_regen_rule(REGEN_RULE)
    for path in iter_content_pages(content_root):
        try:
            raw = open(path, "r", encoding="utf-8").read()
            fm, _ = read_markdown_frontmatter(raw)
            if not fm:
                continue
            slug = str(fm.get("slug") or "").strip()
            hub = str(fm.get("hub") or "").strip()
            gv = fm.get("gen_version", 0)
            try:
                gv = int(gv)
            except Exception:
                gv = 0

            if slugs_set:
                if slug and slug in slugs_set:
                    targets.append({"path": path, "fm": fm})
                continue
            if REGEN_HUB:
                if hub.lower() == REGEN_HUB.lower():
                    targets.append({"path": path, "fm": fm})
                continue

            if not rule:
                continue
            rtype = rule.get("type")
            rval = rule.get("value")
            if rtype == "version_lt":
                try:
                    n = int(rval)
                except Exception:
                    n = 0
                if gv < n:
                    targets.append({"path": path, "fm": fm})
            elif rtype == "contract_mismatch":
                if str(fm.get("contract_hash", "")) != str(contract_hash):
                    targets.append({"path": path, "fm": fm})
        except Exception:
            continue

    return targets

def generate_one_page(title: str, system: str, page_prompt: str, cfg: dict, pinned_hub: str = "", pinned_page_type: str = ""):
    """Returns (ok, data_dict)."""
    generation = (cfg.get("generation") or {})
    wc_cfg = generation.get("wordcount") or {}
    wc_min = int(wc_cfg.get("min") or 900)
    wc_max = int(wc_cfg.get("max") or 1900)

    extra = ""
    if pinned_hub:
        extra += f"\nHub (must use exactly): {pinned_hub}"
    if pinned_page_type:
        extra += f"\nPage type (must use exactly): {pinned_page_type}"

    try:
        resp = call_kimi(system, f"{page_prompt}\n\nTitle: {title}{extra}")
        data = resp if isinstance(resp, dict) else parse_json_strict_or_extract(str(resp))
    except Exception as e:
        print(f"  LLM call failed for '{title}': {type(e).__name__}: {e}")
        return False, {}

    body = (data.get("body_md") or "").strip()
    required_h2 = (cfg.get("generation", {}) or {}).get("outline_h2", [])
    if required_h2:
        # Auto-fix common H2 variations before checking
        h2_aliases = {
            "Related topics": "Related topics and deeper reading",
            "Related Topics": "Related topics and deeper reading",
            "Related topics and further reading": "Related topics and deeper reading",
            "Deeper reading": "Related topics and deeper reading",
            "Further reading": "Related topics and deeper reading",
            "Summary": "Neutral summary",
            "Frequently asked questions": "FAQs",
            "FAQ": "FAQs",
            "Misconceptions": "Common misconceptions",
            "Key terms": "Definitions and key terms",
            "Definitions": "Definitions and key terms",
            "Examples": "Clarifying examples",
            "Introduction": "Intro",
        }
        for alias, canonical in h2_aliases.items():
            body = re.sub(rf'^## {re.escape(alias)}\s*$', f'## {canonical}', body, flags=re.MULTILINE)

        # Auto-dedup: if a heading appears twice, remove the second occurrence and its content
        seen_h2 = set()
        deduped_lines = []
        skip_until_next_h2 = False
        for line in body.split('\n'):
            h2_match = re.match(r'^## (.+)$', line)
            if h2_match:
                heading = h2_match.group(1).strip()
                if heading in seen_h2:
                    # Duplicate — skip this heading and its content until next H2
                    skip_until_next_h2 = True
                    continue
                seen_h2.add(heading)
                skip_until_next_h2 = False
            elif skip_until_next_h2:
                continue
            deduped_lines.append(line)
        body = '\n'.join(deduped_lines)

        # Extract H2 headings in order from body
        got_h2 = re.findall(r'^## (.+)$', body, re.MULTILINE)
        got_h2 = [h.strip() for h in got_h2]
        if got_h2 != required_h2:
            missing = [h for h in required_h2 if h not in got_h2]
            extra_h2 = [h for h in got_h2 if h not in required_h2]
            if missing:
                print(f"  Missing H2 sections: {missing}")
            if extra_h2:
                print(f"  Extra H2 sections: {extra_h2}")
            if not missing and not extra_h2:
                print(f"  H2 order wrong. Expected: ...{required_h2[-4:]}  Got: ...{got_h2[-4:]}")
            return False, {}
    else:
        if body.count("## ") < 6:
            print(f"  Too few H2 sections (found {body.count('## ')})")
            return False, {}

    required = ["title", "summary", "description", "hub", "page_type"]
    if any((k not in data or not str(data[k]).strip()) for k in required):
        missing_keys = [k for k in required if k not in data or not str(data[k]).strip()]
        print(f"  Missing required keys: {missing_keys}")
        return False, {}

    data["body_md"] = body

    # Pre-write validation: internal links (need 6+: 3 contextual + 3 in related)
    all_link_matches = re.findall(r'\[([^\]]*)\]\((/pages/[a-z0-9-]+/)\)', body)
    internal_link_count = len(all_link_matches)
    if internal_link_count < 6:
        print(f"  Too few internal links: {internal_link_count} (need 6+)")
        return False, {}

    # Check for duplicate link targets
    link_urls = [url for _, url in all_link_matches]
    unique_urls = set(link_urls)
    if len(unique_urls) < len(link_urls):
        dupes = [u for u in unique_urls if link_urls.count(u) > 1]
        print(f"  Duplicate internal links: {dupes[:3]}")
        return False, {}

    # Check for self-links
    page_slug_from_data = slugify(data.get("title", ""))
    self_url_patterns = [f"/pages/{page_slug_from_data}/"]
    if any(url in self_url_patterns for url in link_urls):
        print(f"  Self-link detected — rejected")
        return False, {}

    # Split contextual (in-body) vs related section links
    related_heading_pat = re.compile(r'^## Related topics and deeper reading\s*$', re.MULTILINE)
    related_m = related_heading_pat.search(body)
    if related_m:
        body_before_related = body[:related_m.start()]
        related_section_text = body[related_m.end():]
        # Trim related section at next H2
        next_h2 = re.search(r'^## ', related_section_text, re.MULTILINE)
        if next_h2:
            related_section_text = related_section_text[:next_h2.start()]
    else:
        body_before_related = body
        related_section_text = ""

    contextual_links = re.findall(r'\[.*?\]\(/pages/[a-z0-9-]+/\)', body_before_related)
    related_links = re.findall(r'\[.*?\]\(/pages/[a-z0-9-]+/\)', related_section_text)

    # Stage-aware gating: only enforce strict minimums when candidate pool is sufficient
    existing_page_count = len(list(Path("content/pages").glob("*/index.md"))) if Path("content/pages").exists() else 0
    strict_linking = existing_page_count >= 8  # Early sites get a pass

    if strict_linking:
        if len(contextual_links) < 3:
            print(f"  Too few contextual (in-body) links: {len(contextual_links)} (need 3+)")
            return False, {}
        if len(related_links) < 3:
            print(f"  Too few related section links: {len(related_links)} (need 3+)")
            return False, {}
    else:
        # Relaxed: just need some links total
        if internal_link_count < 3:
            print(f"  Too few internal links for early-stage site: {internal_link_count} (need 3+)")
            return False, {}

    # Pre-write validation: payload block exists (exactly one)
    payload_headings = re.findall(
        r'^### (?:Quick-start snapshot|If–then chooser|Self-check checklist|Starter templates)\s*$',
        body, re.MULTILINE
    )
    if len(payload_headings) == 0:
        print(f"  Missing unique payload block (no ### Quick-start/If-then/Checklist/Templates heading)")
        return False, {}
    if len(payload_headings) > 1:
        print(f"  Multiple payload blocks found ({len(payload_headings)}) — need exactly 1")
        return False, {}

    # Pre-write validation: no external links
    external_links = re.findall(r'https?://', body)
    if external_links:
        print(f"  External links found ({len(external_links)}) — rejected")
        return False, {}

    # Pre-write validation: no date/recency language (must match quality_gates.py exactly)
    recency_patterns = [
        r'\b(19|20)\d{2}\b',           # years
        r'\brecently\b',
        r'\bcurrently\b',
        r'\bthis\s+year\b',
        r'\blast\s+year\b',
        r'(?<!\bto)\btoday\b',         # avoid "up to today" but catch standalone
        r'\bright\s+now\b',
        r'\bas\s+of\s+now\b',
    ]
    for pat in recency_patterns:
        m = re.search(pat, body, re.MULTILINE)
        if m:
            print(f"  Date/recency language found: '{m.group().strip()[:40]}' — rejected")
            return False, {}

    # Pre-write validation: no first-person
    # "us" and "our" are too aggressive — catches "around us", "our emotions" which is common
    # in encyclopedic writing about emotional/psychological topics.
    # Only flag strong first-person voice: I, we, my, me, mine.
    # Check body + summary + description + closing_reassurance to match quality_gates coverage.
    text_to_check_fp = body + "\n" + str(data.get("summary", "")) + "\n" + str(data.get("description", "")) + "\n" + str(data.get("closing_reassurance", ""))
    first_person_m = re.search(r"(?<![/\w])\b(I|I'm|I've|my|mine|me|we|we're|we've)\b(?![/\w])", text_to_check_fp)
    if first_person_m:
        print(f"  First-person language found: '{first_person_m.group()}' — rejected")
        return False, {}

    # Pre-write validation: no guarantee/promise language
    guarantee_patterns = [
        r'\bguarantee[ds]?\b',
        r'\b100%\b',
        r'\bwill\s+definitely\b',
        r'\bwill\s+always\b',
        r'\bwill\s+never\b',
    ]
    for pat in guarantee_patterns:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            print(f"  Guarantee language found: '{m.group()}' — rejected")
            return False, {}

    # Pre-write validation: paragraph length (max 3 sentences per paragraph)
    max_sent = 3
    bad_paras = 0
    for para in re.split(r'\n{2,}', body.strip()):
        para = para.strip()
        if not para or para.startswith('#') or para.startswith('-') or para.startswith('*') or re.match(r'^\d+\.', para) or para.startswith('```'):
            continue
        sc = len(re.findall(r'[.!?](?:\s|$)', para))
        if sc > max_sent:
            bad_paras += 1
    if bad_paras > 0:
        print(f"  {bad_paras} paragraphs exceed {max_sent} sentences — rejected")
        return False, {}

    # Pre-write validation: word count (must match quality_gates thresholds)
    wc = len(re.findall(r'\b[\w\']+\b', body))
    if wc < wc_min:
        print(f"  Word count too low: {wc} (min {wc_min}) — rejected")
        return False, {}
    if wc > wc_max:
        print(f"  Word count too high: {wc} (max {wc_max}) — rejected")
        return False, {}

    # Pre-write validation: required sections have enough content (must match quality_gates)
    required_sections = ["Intro", "Definitions and key terms", "How it typically works", "Clarifying examples", "Neutral summary"]
    for sec_name in required_sections:
        # Extract text between this H2 and the next H2
        pat = re.compile(rf"^##\s+{re.escape(sec_name)}\s*$", re.M)
        m = pat.search(body)
        if not m:
            print(f"  Missing required section: \"{sec_name}\" — rejected")
            return False, {}
        start = m.end()
        rest = body[start:]
        m2 = re.search(r"^##\s+", rest, flags=re.M)
        sec_text = (rest[:m2.start()] if m2 else rest).strip()
        sec_wc = len(re.findall(r'\b[\w\']+\b', sec_text))
        if sec_wc < 40:
            print(f"  Section \"{sec_name}\" too thin: {sec_wc} words (min 40) — rejected")
            return False, {}

    # Pre-write validation: FAQ count (must match quality_gates faq_min=4)
    faq_section_pat = re.compile(r"^##\s+FAQs?\s*$", re.M)
    faq_m = faq_section_pat.search(body)
    if faq_m:
        faq_text = body[faq_m.end():]
        faq_next = re.search(r"^##\s+", faq_text, flags=re.M)
        if faq_next:
            faq_text = faq_text[:faq_next.start()]
        faq_q_count = len(re.findall(r"^###\s+.+", faq_text, flags=re.M))
        if faq_q_count == 0:
            faq_q_count = len(re.findall(r"^\*\*Q[:\s].+\*\*", faq_text, flags=re.M))
        if faq_q_count == 0:
            faq_q_count = len(re.findall(r"^\*\*.+\?\*\*", faq_text, flags=re.M))
        if faq_q_count < 4:
            print(f"  Too few FAQs: {faq_q_count} (min 4) — rejected")
            return False, {}

    return True, data

def write_page(slug: str, data: dict, close: str, contract_hash: str, prompt_hash: str, tags: list = None) -> Path:
    """Create a content page folder and write index.md."""
    page_slug = (slug or "").strip()
    if not page_slug:
        raise ValueError("write_page: empty slug")

    page_dir = Path("content") / "pages" / page_slug
    page_dir.mkdir(parents=True, exist_ok=True)

    title = (data.get("title") or page_slug.replace("-", " ").title()).strip()
    hub = (data.get("hub") or "").strip()
    page_type = (data.get("page_type") or data.get("type") or "guide").strip()
    description = (data.get("description") or data.get("summary") or "").strip()
    summary = (data.get("summary") or "").strip()

    front = {
        "title": title,
        "slug": page_slug,
        "description": description,
        "summary": summary,
        "hub": hub,
        "page_type": page_type,
        "date": data.get("date") or date.today().isoformat(),
        "draft": False,
        "gen_version": GEN_VERSION,
        "contract_hash": contract_hash,
        "prompt_hash": prompt_hash,
    }
    # Add cluster if present (for hub pillar page grouping)
    cluster = (data.get("cluster") or "").strip()
    if cluster:
        front["cluster"] = cluster
    # Add tags for Hugo related-content matching
    if tags:
        front["tags"] = tags

    body = (data.get("body_md") or "").rstrip()
    close_txt = (close or "").strip()
    if close_txt:
        if body:
            body += "\n\n"
        body += close_txt + "\n"

    md = write_markdown_with_frontmatter(front, body)
    (page_dir / "index.md").write_text(md, encoding="utf-8")
    return page_dir


def _remove_one(path: Path, title: str) -> None:
    if not path.exists():
        return
    lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines()]
    out = []
    removed = False
    for l in lines:
        if (not removed) and l.strip() == title:
            removed = True
            continue
        out.append(l)
    path.write_text("\n".join(out).strip() + ("\n" if out else ""), encoding="utf-8")


def mark_failed(title: str, reason: str) -> None:
    FAILED_TITLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_TITLES_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{title}\t{reason}\n")
    existing = set()
    if RETRY_TITLES_PATH.exists():
        existing = {l.strip() for l in RETRY_TITLES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    if title not in existing:
        with RETRY_TITLES_PATH.open("a", encoding="utf-8") as f:
            f.write(title + "\n")
    _remove_one(TITLES_POOL_PATH, title)


def mark_done(title: str) -> None:
    t = (title or "").strip()
    if not t:
        return

    _remove_one(Path("scripts") / "titles_pool.txt", t)
    _remove_one(Path("scripts") / "retry_titles.txt", t)

    try:
        plan = load_plan(str(PLAN_PATH))
        q = plan.get("queue") or []
        if isinstance(q, list) and t in q:
            plan["queue"] = [x for x in q if x != t]
            save_plan(str(PLAN_PATH), plan)
    except Exception:
        pass


# --- Frontmatter cache for performance optimization ---
_FRONTMATTER_CACHE = {}
_FRONTMATTER_CACHE_TIMESTAMPS = {}

def _get_cached_frontmatter(file_path: Path) -> tuple:
    """Get frontmatter from cache or parse and cache it."""
    global _FRONTMATTER_CACHE, _FRONTMATTER_CACHE_TIMESTAMPS
    
    try:
        mtime = file_path.stat().st_mtime
        cached = _FRONTMATTER_CACHE.get(str(file_path))
        cached_time = _FRONTMATTER_CACHE_TIMESTAMPS.get(str(file_path))
        
        # Return cached if still valid
        if cached is not None and cached_time == mtime:
            return cached
        
        # Parse and cache
        raw = file_path.read_text(encoding="utf-8")
        fm, body = read_markdown_frontmatter(raw)
        
        _FRONTMATTER_CACHE[str(file_path)] = (fm, body)
        _FRONTMATTER_CACHE_TIMESTAMPS[str(file_path)] = mtime
        
        return fm, body
    except Exception:
        return {}, ""

def _clear_frontmatter_cache():
    """Clear the frontmatter cache."""
    global _FRONTMATTER_CACHE, _FRONTMATTER_CACHE_TIMESTAMPS
    _FRONTMATTER_CACHE.clear()
    _FRONTMATTER_CACHE_TIMESTAMPS.clear()

def _count_pages_per_hub(content_root: Path) -> dict:
    """Count existing published pages per hub."""
    counts = {}
    for md in content_root.glob("*/index.md"):
        try:
            fm, _ = _get_cached_frontmatter(md)
            hub = str(fm.get("hub", "")).strip()
            if hub:
                counts[hub] = counts.get(hub, 0) + 1
        except Exception:
            continue
    return counts


def _select_hub_batched_titles(todo_items: list, content_root: Path, hub_min: int = 10, max_hubs_per_batch: int = 3) -> list:
    """Select titles prioritizing hubs that need more pages.

    Strategy: fill each hub to hub_min pages before moving to the next.
    This ensures every hub has enough siblings for internal linking.
    
    OPTIMIZATION: Process hubs incrementally in batches to avoid overwhelming the system.
    """
    hub_counts = _count_pages_per_hub(content_root)

    # Group todo items by hub
    by_hub = {}
    for it in todo_items:
        hub = str(it.get("hub", "")).strip() or "general"
        by_hub.setdefault(hub, []).append(it)

    # Prioritize hubs under the minimum, then hubs with the fewest pages
    ordered_titles = []
    
    # Sort hubs by priority: those under hub_min first, then by count
    hub_priority = []
    for hub in by_hub.keys():
        count = hub_counts.get(hub, 0)
        if count < hub_min:
            hub_priority.append((0, count, hub))  # Highest priority: under minimum
        else:
            hub_priority.append((1, count, hub))  # Lower priority: already at minimum
    
    hub_priority.sort(key=lambda x: (x[0], x[1]))
    
    # Take only max_hubs_per_batch hubs at a time for incremental processing
    selected_hubs = [hub for _, _, hub in hub_priority[:max_hubs_per_batch]]
    
    for hub in selected_hubs:
        items = by_hub[hub]
        # Take up to hub_min - current_count items from this hub
        current_count = hub_counts.get(hub, 0)
        needed = max(0, hub_min - current_count)
        
        if needed > 0:
            # Take needed items or all items if fewer available
            items_to_take = min(needed, len(items))
            ordered_titles.extend([it.get("title", "").strip() for it in items[:items_to_take] if it.get("title")])
        else:
            # Hub already at minimum, take a few items anyway
            items_to_take = min(5, len(items))  # Small batch for maintenance
            ordered_titles.extend([it.get("title", "").strip() for it in items[:items_to_take] if it.get("title")])

    return ordered_titles


def link_injection_pass(content_root: Path, limit: int = 50):
    """Second pass: ensure every page has at least 3 internal links.

    Reads all existing pages, finds those with fewer than 3 internal links,
    and injects links into their 'Related topics and deeper reading' section.

    Only links to pages that have valid frontmatter (title, slug, hub) to avoid
    injecting links to pages that quality_gates will later delete.
    
    OPTIMIZATION: Uses frontmatter cache and optimized algorithm.
    """
    import time
    start_time = time.time()
    
    # Build catalog of all existing pages (only those with valid frontmatter)
    all_pages = {}  # slug -> {title, hub, tags, path}
    hub_pages = {}  # hub -> list of slugs for faster lookup
    
    # First pass: collect all pages using cache
    for md in content_root.glob("*/index.md"):
        try:
            fm, body = _get_cached_frontmatter(md)
            slug = fm.get("slug") or md.parent.name
            
            # Skip pages missing required frontmatter
            if not fm.get("title") or not fm.get("hub"):
                continue
                
            hub = fm.get("hub", "")
            tags = set(fm.get("tags", []))
            
            all_pages[slug] = {
                "title": fm.get("title", slug.replace("-", " ").title()),
                "hub": hub,
                "tags": tags,
                "path": md,
                "body": body,
                "fm": fm,
            }
            
            # Build hub index for faster lookup
            hub_pages.setdefault(hub, []).append(slug)
        except Exception:
            continue

    if len(all_pages) < 4:
        # Not enough pages to do meaningful linking
        return 0

    fixed = 0
    processed = 0
    
    # Pre-compute tag sets for faster intersection
    tag_sets = {slug: info["tags"] for slug, info in all_pages.items()}
    
    # Process pages in order of need (those with fewest links first)
    pages_to_process = []
    for slug, info in all_pages.items():
        body = info["body"]
        existing_links = re.findall(r'\[.*?\]\(/pages/([a-z0-9-]+)/\)', body)
        link_count = len(existing_links)
        if link_count < 3:
            pages_to_process.append((link_count, slug, info, existing_links))
    
    # Sort by fewest links first
    pages_to_process.sort(key=lambda x: x[0])
    
    for link_count, slug, info, existing_links in pages_to_process[:limit]:
        processed += 1
        body = info["body"]
        hub = info["hub"]
        
        if link_count >= 3:
            continue

        # OPTIMIZED: Find best link targets using hub index
        candidates = []
        
        # First, try pages from the same hub (fast lookup)
        same_hub_slugs = hub_pages.get(hub, [])
        for other_slug in same_hub_slugs:
            if other_slug == slug or other_slug in existing_links:
                continue
            other = all_pages[other_slug]
            score = 10 + len(info["tags"] & other["tags"])  # Base score for same hub
            candidates.append((score, other_slug, other["title"]))
        
        # If we need more candidates, try other hubs
        if len(candidates) < (3 - link_count):
            for other_slug, other in all_pages.items():
                if other_slug == slug or other_slug in existing_links or other_slug in same_hub_slugs:
                    continue
                score = len(info["tags"] & other["tags"])
                candidates.append((score, other_slug, other["title"]))
        
        candidates.sort(key=lambda x: -x[0])
        needed = 3 - link_count
        picks = candidates[:needed]

        if not picks:
            continue

        # Build link block
        links_md = "\n".join([f"- [{title}](/pages/{s}/)" for _, s, title in picks])

        # Inject into "Related topics and deeper reading" section if it exists
        related_heading = "## Related topics and deeper reading"
        if related_heading in body:
            # Append links after the heading
            parts = body.split(related_heading, 1)
            # Find the next ## heading after related section
            after = parts[1]
            next_h2 = re.search(r'\n## ', after)
            if next_h2:
                insert_pos = next_h2.start()
                new_after = after[:insert_pos].rstrip() + "\n\n" + links_md + "\n" + after[insert_pos:]
            else:
                new_after = after.rstrip() + "\n\n" + links_md + "\n"
            body = parts[0] + related_heading + new_after
        else:
            # Append at the very end
            body = body.rstrip() + f"\n\n{related_heading}\n\n{links_md}\n"

        # Write back
        md_text = write_markdown_with_frontmatter(info["fm"], body)
        info["path"].write_text(md_text, encoding="utf-8")
        fixed += 1
    
    # Clear cache after modifications
    _clear_frontmatter_cache()
    
    end_time = time.time()
    if processed > 0:
        print(f"[PERF] link_injection_pass processed {processed} pages, fixed {fixed} in {end_time-start_time:.2f}s")
    
    return fixed


def main():
    site_cfg_path = resolve_site_config_path()
    cfg = load_yaml(site_cfg_path)
    system, page_prompt, hub_clusters = build_prompts(cfg)

    # FIX: PERF_MAX_PAGES from empty workflow input will be empty string, which crashes int().
    _perf_raw = (os.getenv("PERF_MAX_PAGES") or "").strip()
    max_pages = int(_perf_raw) if _perf_raw else int((cfg.get("gates") or {}).get("max_pages", 1000))

    existing_pages = len(list(CONTENT_ROOT.glob("*/index.md")))
    if existing_pages >= max_pages:
        print(f"✅ Page cap reached ({existing_pages}/{max_pages}). Nothing to do.")
        return

    # Provide internal link candidates so the model can reliably include them
    link_hints = build_internal_link_hints(CONTENT_ROOT, limit=40)
    if link_hints:
        page_prompt = page_prompt + f"\n\n=== AVAILABLE INTERNAL LINKS (grouped by hub — prefer same-hub links) ===\n{link_hints}\n\nYou MUST use links from this list. Prefer 3-4 links from YOUR hub + 1-2 from other hubs. Do NOT invent slugs. Do NOT use external URLs.\n"
    else:
        # Fresh bootstrap: no pages exist yet. Generate plausible sibling slugs from the titles pool.
        # NOTE: These pages don't exist yet — they'll be created in subsequent runs.
        # Early-stage pages get relaxed link validation (see strict_linking in generate_one_page).
        sibling_slugs = _generate_sibling_link_hints(TITLES_POOL_PATH, limit=20)
        if sibling_slugs:
            page_prompt = page_prompt + f"\n\n=== PLANNED INTERNAL LINKS (these pages will be created alongside yours — use at least 6) ===\n{sibling_slugs}\n\nYou MUST use links from this list. These are sibling pages being generated in the same batch. Do NOT invent slugs. Do NOT use external URLs.\n"

    os.makedirs(CONTENT_ROOT, exist_ok=True)
    contract_hash = compute_contract_hash(site_cfg_path)
    manifest = load_manifest()

    if BACKFILL_METADATA:
        backfilled = backfill_page_metadata(CONTENT_ROOT, contract_hash)
        if backfilled:
            print(f"[metadata] backfilled gen_version/contract_hash/prompt_hash on {backfilled} pages")

    prompt_hash = hashlib.sha1((system + "\n" + page_prompt).encode("utf-8")).hexdigest()

    # Regen mode
    if FACTORY_MODE == "regen":
        targets = select_pages_for_regen(CONTENT_ROOT, contract_hash)
        if not targets:
            print("[regen] no pages matched the regeneration criteria")
            return

        print(f"[regen] matched {len(targets)} pages; regenerating up to {PAGES_PER_RUN}")

        regen_count = 0
        attempts = 0

        for t in targets:
            if regen_count >= PAGES_PER_RUN or attempts >= MAX_ATTEMPTS:
                break
            fm = t["fm"]
            title = str(fm.get("title") or "").strip()
            slug = str(fm.get("slug") or "").strip()
            hub = str(fm.get("hub") or "").strip()
            page_type = str(fm.get("page_type") or "").strip()
            if not title or not slug:
                continue

            attempts += 1
            print(f"[regen] {slug}: {title}")

            ok, data = generate_one_page(
                title=title,
                system=system,
                page_prompt=page_prompt,
                cfg=cfg,
                pinned_hub=hub,
                pinned_page_type=page_type,
            )
            if not ok:
                continue

            close = choose_close(data, cfg)
            write_page(slug=slug, data=data, close=close, contract_hash=contract_hash, prompt_hash=prompt_hash)

            regen_count += 1
            manifest.setdefault("generated_this_run", []).append(slug)
            time.sleep(SLEEP_SECONDS)

        save_manifest(manifest)
        return

    # Generate mode: consume plan todos first, else fall back to titles_pool.
    plan = load_plan(str(PLAN_PATH))
    plan_items = plan.get("items", []) if isinstance(plan, dict) else []
    todo_items = [it for it in plan_items if isinstance(it, dict) and str(it.get("status", "todo")).lower() == "todo"]

    titles = []
    hub_map = {}  # title -> hub_id from titles_pool.txt
    if todo_items:
        # Hub-batched selection: prioritize hubs that need more pages
        # Use incremental processing: max 3 hubs per batch
        titles = _select_hub_batched_titles(todo_items, CONTENT_ROOT, hub_min=10, max_hubs_per_batch=3)
    else:
        titles = load_titles()
        hub_map = load_titles_with_hubs()
        
        # Apply Concrete Intent Validator (CIV) to ensure concrete, high-intent titles
        try:
            from concrete_intent_validator import enforce_concrete_titles
            
            # Get niche from environment or config
            niche = (os.getenv("BOOTSTRAP_NICHE", "") or os.getenv("NICHE", "")).strip()
            if not niche:
                # Try to get niche from site config
                site_cfg = load_yaml(site_cfg_path)
                niche = site_cfg.get("niche", "").strip()
            
            if niche and titles:
                site_root = Path.cwd()
                validated_titles = enforce_concrete_titles(titles, niche, site_root)
                
                # Check if any titles were rewritten
                original_count = len(titles)
                validated_count = len(validated_titles)
                rewritten_count = sum(1 for i in range(min(original_count, validated_count))
                                    if titles[i] != validated_titles[i])
                
                if rewritten_count > 0:
                    print(f"✅ Applied CIV validation: {rewritten_count}/{original_count} titles rewritten for concrete intent")
                    # Show examples of rewritten titles
                    print(f"   Examples of CIV improvements:")
                    for i in range(min(3, rewritten_count)):
                        if i < len(titles) and i < len(validated_titles) and titles[i] != validated_titles[i]:
                            print(f"     - '{titles[i]}' → '{validated_titles[i]}'")
                    
                    titles = validated_titles
                else:
                    print(f"✅ All {original_count} titles passed CIV validation")
            else:
                print(f"⚠️  CIV: No niche found or no titles to validate")
                
        except ImportError as e:
            print(f"⚠️  CIV not available: {e}. Proceeding without concrete intent validation.")
        except Exception as e:
            print(f"⚠️  CIV validation failed: {e}. Proceeding without concrete intent validation.")
        
        random.shuffle(titles)

    if not titles:
        print("✅ No titles available to generate (pool is empty or not yet bootstrapped). Nothing to do.")
        return

    produced = 0
    attempts = 0
    deletes = 0
    consec_fail = 0

    manifest["generated_this_run"] = []
    used = set(manifest.get("used_titles", []))

    per_title_fail = {}

    for title in titles:
        if produced >= PAGES_PER_RUN or attempts >= MAX_ATTEMPTS:
            break

        plan_item = None
        if todo_items:
            for it in todo_items:
                if it.get("title", "").strip() == title and str(it.get("status", "todo")).lower() == "todo":
                    plan_item = it
                    break

        slug = (plan_item.get("slug") if isinstance(plan_item, dict) and plan_item.get("slug") else None) or slugify(title)

        if slug in used:
            continue

        if per_title_fail.get(slug, 0) >= PER_TITLE_CAP:
            continue

        attempts += 1
        print(f"\n--- Generating: {title} (slug: {slug}) ---")

        pinned_hub = ""
        pinned_type = ""
        if isinstance(plan_item, dict):
            pinned_hub = str(plan_item.get("hub") or "").strip()
            pinned_type = str(plan_item.get("page_type") or "").strip()
        if not pinned_hub and title in hub_map:
            pinned_hub = hub_map[title]

        ok, data = generate_one_page(
            title=title,
            system=system,
            page_prompt=page_prompt,
            cfg=cfg,
            pinned_hub=pinned_hub,
            pinned_page_type=pinned_type,
        )
        if not ok:
            deletes += 1
            consec_fail += 1
            per_title_fail[slug] = per_title_fail.get(slug, 0) + 1
            mark_failed(title, "quality_check_failed")
            if consec_fail >= FAIL_STOP:
                print(f"\nStop early: hit {consec_fail} consecutive failures (FAIL_STOP={FAIL_STOP}).")
                break
            continue

        close = choose_close(data, cfg)
        plan_tags = plan_item.get("tags", []) if isinstance(plan_item, dict) else []
        write_page(slug=slug, data=data, close=close, contract_hash=contract_hash, prompt_hash=prompt_hash, tags=plan_tags)

        if isinstance(plan_item, dict):
            plan_item["slug"] = slug
            plan_item["status"] = "done"
            plan_item["generated_date"] = date.today().isoformat()

        produced += 1
        consec_fail = 0
        used.add(slug)
        manifest.setdefault("used_titles", []).append(slug)
        manifest.setdefault("generated_this_run", []).append(slug)
        mark_done(title)
        time.sleep(SLEEP_SECONDS)

    save_manifest(manifest)

    if todo_items:
        save_plan(str(PLAN_PATH), plan)

    # Two-pass linking: fix internal links on all pages after generation batch
    if produced > 0:
        fixed = link_injection_pass(CONTENT_ROOT)
        if fixed:
            print(f"\n[link-pass] Injected internal links into {fixed} pages")

    duration = int(time.time() - START_TIME)
    print("\n===== FACTORY SUMMARY =====")
    print(f"Pages attempted: {attempts}")
    print(f"Pages produced: {produced}")
    print(f"Pages rejected: {deletes}")
    print(f"Duration: {duration // 60}m {duration % 60}s")
    print("===========================\n")

if __name__ == "__main__":
    main()
