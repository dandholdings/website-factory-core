"""
llm_client.py — Shared LLM provider client for the website factory.

Used by: bootstrap_site.py, generate_pages.py, fill_hub_content.py
Eliminates code duplication of kimi_json(), parse_json_strict_or_extract(),
_safe_write_kimi_dump(), and site-root helpers.
"""

import os
import re
import sys
import json
import time
import random
import hashlib
from pathlib import Path

import requests


# --- LLM provider configuration ----------------------------------------
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
MOONSHOT_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _safe_int(val, default: int) -> int:
    """Parse an int from a string, returning default on any failure."""
    try:
        v = (val or "").strip() if isinstance(val, str) else val
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


def llm_provider() -> str:
    if GEMINI_API_KEY:
        return "gemini"
    if MOONSHOT_API_KEY:
        return "moonshot"
    return ""


PROVIDER = llm_provider()

# --- Runtime knobs -------------------------------------------------------
REQUEST_TIMEOUT = _safe_int(os.getenv("REQUEST_TIMEOUT"), 180)
CONNECT_TIMEOUT = _safe_int(os.getenv("CONNECT_TIMEOUT"), 20)
HTTP_MAX_TRIES = _safe_int(os.getenv("HTTP_MAX_TRIES") or os.getenv("KIMI_HTTP_MAX_TRIES"), 6)
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", os.getenv("KIMI_BACKOFF_BASE", "1.7")) or "1.7")

try:
    _t_raw = os.getenv("TEMPERATURE", "0.25").strip()  # Lower default for more deterministic JSON
    TEMPERATURE = float(_t_raw)
except Exception:
    TEMPERATURE = 0.25

# Only force temperature=1 for Moonshot/Kimi (legacy constraint).
if PROVIDER == "moonshot" and TEMPERATURE != 1:
    TEMPERATURE = 1


# --- Slugify (canonical version) -----------------------------------------
def slugify(s: str, max_len: int = 80) -> str:
    """Canonical slug function. All scripts should use this."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:max_len].strip("-") or "site"


# --- Site root helper -----------------------------------------------------
def apply_site_root_early():
    """Allow running core scripts from inside thin-repo.

    Priority:
      1) --site-root <path>
      2) --site-slug <slug>  -> chdir sites/<slug>
      3) SITE_SLUG env (if set) -> chdir sites/<SITE_SLUG>
    """
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
        slug = (os.getenv("SITE_SLUG", "") or "").strip()

    if not slug:
        niche = (os.getenv("BOOTSTRAP_NICHE", "") or os.getenv("NICHE", "")).strip()
        slug = slugify(niche, max_len=60)

    if slug:
        root = Path("sites") / slug
        root.mkdir(parents=True, exist_ok=True)
        os.chdir(root)


# --- JSON parsing ---------------------------------------------------------
def parse_json_strict_or_extract(raw: str) -> dict:
    """Parse a JSON object from model output. Robust against fences, prose, truncation."""
    if raw is None:
        raw = ""

    if isinstance(raw, (dict, list)):
        try:
            return raw if isinstance(raw, dict) else (raw[0] if raw and isinstance(raw[0], dict) else json.loads(json.dumps(raw)))
        except Exception:
            raw = json.dumps(raw)

    raw = str(raw).replace("\ufeff", "").replace("\x00", "").strip()

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
        # If it's a list, wrap it in a dict with key "data"
        if isinstance(obj, list):
            return {"data": obj}
    except json.JSONDecodeError:
        pass

    # Walk for the first balanced JSON object or array
    s = cand
    start_obj = s.find("{")
    start_arr = s.find("[")
    
    # Determine which comes first and what type we're looking for
    if start_obj == -1 and start_arr == -1:
        raise json.JSONDecodeError("No JSON object or array found", cand, 0)
    
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        # Looking for object
        start = start_obj
        open_char = "{"
        close_char = "}"
    else:
        # Looking for array
        start = start_arr
        open_char = "["
        close_char = "]"

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
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise json.JSONDecodeError(f"JSON {open_char}{close_char} appears truncated (missing closing {close_char})", cand, 0)

    snippet = _strip_fences(s[start:end])
    obj = json.loads(snippet)
    
    # Handle arrays by wrapping in dict
    if isinstance(obj, list):
        return {"data": obj}
    
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Top-level JSON must be an object or array", snippet, 0)
    
    return obj


# --- Debug dump -----------------------------------------------------------
def safe_write_llm_dump(
    kind: str,
    attempt: int,
    *,
    content: str = "",
    envelope=None,
    http_status=None,
    http_text=None,
    payload=None,
):
    """Write a debug dump to help diagnose provider hiccups. Never includes API keys."""
    try:
        log_dir = Path("scripts") / "_logs" / "llm"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = hashlib.sha1(f"{ts}-{kind}-{attempt}-{random.random()}".encode("utf-8")).hexdigest()[:10]
        p = log_dir / f"{ts}-{kind}-a{attempt+1}-{rid}.json"

        def _trim(s, n: int) -> str:
            s = "" if s is None else str(s)
            s = s.replace("\x00", "")
            return s[:n]

        # Redact all sensitive keys from payload
        safe_payload = None
        if isinstance(payload, dict):
            safe_payload = {}
            for k, v in payload.items():
                if k.lower() in ("api_key", "authorization", "x-goog-api-key"):
                    safe_payload[k] = "[REDACTED]"
                else:
                    safe_payload[k] = v

        # Scrub any key= params from http_text
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


# --- Main LLM call --------------------------------------------------------
def llm_json(system: str, user: str, temperature: float = None, max_tokens: int = 4096) -> dict:
    """Request a JSON object from the configured LLM provider.

    Supports:
      - Gemini (preferred) via GEMINI_API_KEY / GEMINI_MODEL
      - Moonshot/Kimi (legacy) via MOONSHOT_API_KEY / MOONSHOT_BASE_URL / KIMI_MODEL
    """
    provider = PROVIDER
    if not provider:
        raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY (recommended) or MOONSHOT_API_KEY (legacy).")

    temp = temperature if temperature is not None else TEMPERATURE
    if provider == "moonshot" and "kimi" in (MOONSHOT_MODEL or "").lower():
        temp = 1

    def _sleep(attempt: int) -> float:
        return min(60.0, (BACKOFF_BASE ** attempt) + random.random())

    last_err = None

    def _gemini_request_payload() -> tuple:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        # Combine system and user with clear JSON-only instructions
        # Use a more positive, less restrictive approach to avoid empty responses
        prompt = (
            system.strip()
            + "\n\n"
            + "IMPORTANT: Please output ONLY a valid JSON object or array.\n"
            + "- Begin with '{' or '[' and end with '}' or ']'\n"
            + "- Do not include explanatory text, markdown, or code fences\n"
            + "- Example of correct output: {\"data\": [\"item1\", \"item2\"]} or [\"item1\", \"item2\"]\n\n"
            + user.strip()
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temp),
                "maxOutputTokens": int(max_tokens),
                "responseMimeType": "application/json",
                "stopSequences": ["```", "Here is the JSON"],  # Only block the most problematic phrases
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
        # Clear system prompt for JSON-only output (less restrictive)
        enhanced_system = (
            system.strip()
            + "\n\nIMPORTANT: Output ONLY a valid JSON object or array. "
            + "Do not include explanatory text, markdown, or code fences. "
            + "Begin with '{' or '[' and end with '}' or ']'. "
            + "Example: {\"data\": [\"item1\", \"item2\"]} or [\"item1\", \"item2\"]"
        )
        payload = {
            "model": MOONSHOT_MODEL,
            "temperature": int(temp) if isinstance(temp, (int, float)) and int(temp) == 1 else float(temp),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": enhanced_system},
                {"role": "user", "content": user},
            ],
        }
        return url, headers, payload

    def _extract_text(provider_name: str, data: dict) -> str:
        if provider_name == "gemini":
            try:
                cands = data.get("candidates") or []
                if not cands:
                    return ""
                parts = (cands[0].get("content") or {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            except Exception:
                return ""

        try:
            msg = (data.get("choices", [{}])[0].get("message") or {})
            content = msg.get("content")

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", part.get("content", "")))
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
    for attempt in range(HTTP_MAX_TRIES):
        if provider == "gemini":
            url, headers, payload = _gemini_request_payload()
        else:
            url, headers, payload = _moonshot_request_payload()

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == HTTP_MAX_TRIES - 1:
                raise
            sleep = _sleep(attempt)
            print(f"HTTP error: {type(e).__name__} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        if r.status_code in (408, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}: {r.text[:2000]}"
            safe_write_llm_dump(f"{provider}_http_{r.status_code}", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == HTTP_MAX_TRIES - 1:
                break
            sleep = _sleep(attempt)
            print(f"HTTP {r.status_code} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        if r.status_code >= 400:
            last_err = f"HTTP {r.status_code}: {r.text[:2000]}"
            safe_write_llm_dump(f"{provider}_http_error", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            break

        try:
            data = r.json()
        except Exception as e:
            last_err = f"Bad JSON response envelope: {type(e).__name__}: {e}"
            safe_write_llm_dump(f"{provider}_bad_envelope", attempt, content="", envelope=None, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == HTTP_MAX_TRIES - 1:
                raise
            sleep = _sleep(attempt)
            print(f"Response parse error — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        content = _extract_text(provider, data)

        if not str(content).strip():
            last_err = "Empty model output"
            safe_write_llm_dump(f"{provider}_empty", attempt, content="", envelope=data, http_status=r.status_code, http_text=r.text, payload=payload)
            if attempt == HTTP_MAX_TRIES - 1:
                break
            sleep = _sleep(attempt)
            print(f"Model returned empty output (attempt {attempt+1}/{HTTP_MAX_TRIES}) — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        try:
            return parse_json_strict_or_extract(content)
        except json.JSONDecodeError as e:
            last_err = f"Model JSON decode error: {e}"
            preview = (content or "").strip().replace("\n", " ")[:240]
            print(f"Model returned non-JSON content (attempt {attempt+1}/{HTTP_MAX_TRIES}): {preview}")
            safe_write_llm_dump(f"{provider}_json_decode", attempt, content=content, envelope=data, http_status=r.status_code, http_text=r.text, payload=payload)

            if attempt == HTTP_MAX_TRIES - 1:
                break
            sleep = _sleep(attempt)
            print(f"Retrying after non-JSON output in {sleep:.1f}s")
            time.sleep(sleep)
            continue

    raise RuntimeError(last_err or f"{provider} API retries exhausted")


# --- Frontmatter helpers --------------------------------------------------
def read_markdown_frontmatter(md_text: str):
    """Parse YAML frontmatter and body from a markdown string.

    Canonical implementation — all scripts should use this.
    Splits on "\\n---" (newline before closing ---), strips leading newlines from body.
    Handles empty frontmatter (---\\n---) correctly.
    """
    if not md_text.startswith("---"):
        return {}, md_text
    
    # Skip the opening "---\n"
    after_open = md_text[4:]  # skip "---\n"
    
    # Check for empty frontmatter case: "---\n---" at the start
    if after_open.startswith("---"):
        # Empty frontmatter
        body_start = after_open.find("\n", 3)  # Skip the closing "---"
        if body_start == -1:
            return {}, ""  # No body after empty frontmatter
        body = after_open[body_start:].lstrip("\n").lstrip("\r")
        return {}, body
    
    # Normal case: look for "\n---" as closing marker
    parts = after_open.split("\n---", 1)
    if len(parts) < 2:
        return {}, md_text
    
    fm_raw = parts[0]
    body = parts[1].lstrip("\n").lstrip("\r")
    
    try:
        import yaml
        fm = yaml.safe_load(fm_raw) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    
    return fm, body


def write_markdown_with_frontmatter(front: dict, body: str) -> str:
    import yaml
    fm_txt = yaml.safe_dump(front or {}, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_txt}\n---\n\n{body.lstrip() if body else ''}"
