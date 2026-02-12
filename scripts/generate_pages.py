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
FAIL_STOP: int = _safe_int(os.getenv("FAIL_STOP"), 6)

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
    _t_raw = os.getenv("TEMPERATURE", "0.7").strip()
    TEMPERATURE = float(_t_raw)
except Exception:
    TEMPERATURE = 0.7

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
    if not TITLES_POOL_PATH.exists():
        print(f"[warn] titles_pool.txt not found at {TITLES_POOL_PATH}")
        return []
    try:
        with open(TITLES_POOL_PATH, "r", encoding="utf-8") as f:
            return [t.strip() for t in f if t.strip()]
    except Exception as e:
        print(f"[warn] Failed to read titles_pool.txt: {e}")
        return []

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
    """Build a curated list of existing internal links for the model to use."""
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
        items.append((str(title).strip(), str(slug).strip()))
    items = sorted(items, key=lambda x: x[1])
    items = items[:limit]
    return "\n".join([f"- [{t}](/pages/{s}/)" for t, s in items if t and s])


def build_prompts(cfg: dict):
    site = cfg.get("site", {}) if isinstance(cfg, dict) else {}
    taxonomy = cfg.get("taxonomy", {}) if isinstance(cfg, dict) else {}
    generation = cfg.get("generation", {}) if isinstance(cfg, dict) else {}

    brand = site.get("brand") or site.get("title") or "Reality Checks"
    hubs = [h.get("id") for h in (taxonomy.get("hubs") or []) if isinstance(h, dict) and h.get("id")] or [
        "work-career", "money-stress", "burnout-load", "milestones", "social-norms"
    ]
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
        "What this feeling usually means",
        "Common reasons",
        "What makes it worse",
        "What helps (non-advice)",
        "When it might signal a bigger issue",
        "FAQs",
    ]
    outline_md = "\n".join([f"## {h}" for h in outline])

    closing_templates = generation.get("closing_reassurance_templates") or []
    closing_hint = ""
    if closing_templates:
        closing_hint = "Choose ONE closing reassurance line in a similar style to these:\n- " + "\n- ".join(closing_templates[:3])

    # FIX: Stronger JSON-only instruction for Gemini Flash 2.5
    system = f"""You write calm, reassuring evergreen content for the site "{brand}".
NO medical, legal, or financial advice. Avoid diagnosing. Avoid giving instructions like a professional.
Forbidden words/phrases: {forbidden_str}.
Return a single valid JSON object only. No markdown fences. No extra text before or after the JSON.
"""

    page_prompt = f"""Return ONLY a JSON object with these keys:
title
summary (one sentence reassurance; also used as meta description)
description (<= 160 chars, no quotes)
hub (one of: { " | ".join(hubs) })
page_type (one of: { " | ".join(page_types) })
closing_reassurance (one short, gentle line; NOT advice)
body_md (markdown only; must include the exact H2 headings below)

Use these H2 sections exactly:
{outline_md}

Rules:
- Neutral, encyclopedic tone (beginner-friendly). No hype, no fear framing.
- No medical, legal, or financial advice.
- No dates or time-sensitive language (no years, "recent", "currently", "this year", "today", "now").
- No prices, costs, or financial claims.
- No guarantees/promises ("always", "never", "100%", "will definitely", "guarantee").
- No first-person language ("I", "we", "our", "my").
- No calls-to-action or directive language ("you should", "try this", "make sure to", "sign up", "buy", "download").
- No affiliate/product review language (affiliate, sponsored, review, coupon, discount).
- Comparisons must be neutral (avoid superlatives like "best", "worst", "better than").
- Short paragraphs: 2–3 sentences max.
- Use ONLY H2 (##) and H3 (###) headings. No H1, no H4+.
- Include at least 3 contextually relevant internal links using ONLY relative URLs like /pages/<slug>/ (no external links).
- Wordcount: minimum {wc_min} words, target {wc_ideal_min}–{wc_ideal_max}, maximum {wc_max}.
- Keep tone grounded and human, not clinical.
- No "diagnose/diagnosis/prescribed/guaranteed/sue".
- FAQs: 4-6 Q&As using ### headings for each question.
- Do not include the closing reassurance inside body_md; put it in closing_reassurance.
{closing_hint}
"""
    return system, page_prompt

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
    parts = md_text.split("\n---\n", 2)
    if len(parts) < 3:
        return {}, md_text
    fm_raw = parts[1]
    body = parts[2]
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
        missing = [h for h in required_h2 if f"## {h}" not in body]
        if missing:
            print(f"  Missing H2 sections: {missing}")
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
    return True, data

def write_page(slug: str, data: dict, close: str, contract_hash: str, prompt_hash: str) -> Path:
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


def main():
    site_cfg_path = resolve_site_config_path()
    cfg = load_yaml(site_cfg_path)
    system, page_prompt = build_prompts(cfg)

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
        page_prompt = page_prompt + "\n\nInternal links you MAY use (choose at least 3; do not invent links; no external links):\n" + link_hints + "\n"

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
    if todo_items:
        titles = [it.get("title", "").strip() for it in todo_items if it.get("title")]
    else:
        titles = load_titles()
        random.shuffle(titles)

    if not titles:
        print("✅ No titles available to generate (pool is empty or not yet bootstrapped). Nothing to do.")
        return

    produced = 0
    attempts = 0
    retries = 0
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
        write_page(slug=slug, data=data, close=close, contract_hash=contract_hash, prompt_hash=prompt_hash)

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

    duration = int(time.time() - START_TIME)
    print("\n===== FACTORY SUMMARY =====")
    print(f"Pages attempted: {attempts}")
    print(f"Pages produced: {produced}")
    print(f"Retries: {retries}")
    print(f"Deletes: {deletes}")
    print(f"Duration: {duration // 60}m {duration % 60}s")
    print("===========================\n")

if __name__ == "__main__":
    main()
