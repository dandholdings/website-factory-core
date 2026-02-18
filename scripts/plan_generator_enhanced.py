#!/usr/bin/env python3
"""
plan_generator.py — Generate a complete taxonomy plan contract for a niche.

This implements the "single plan contract" approach:
1. Takes niche, hub_count, clusters_per_hub, pages_per_hub as inputs
2. Generates a complete plan with strict JSON schema
3. Includes ontology, blueprint/family selection, hubs, clusters, pages
4. Validates and repairs JSON deterministically
5. Outputs plan.json for downstream steps

ENHANCED VERSION with:
- Better JSON repair with deterministic fallback
- Strict schema validation with keyword relevance checks
- Improved blueprint selection to prevent wrong matches
- Adaptive temperature settings for different tasks
- Comprehensive error handling and debugging
"""

import argparse
import json
import re
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

# Allow running from thin-repo or core
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from llm_client import llm_json, parse_json_strict_or_extract, safe_write_llm_dump
from taxonomy_blueprints import select_blueprint, BLUEPRINTS, FAMILY_KEYWORDS


def json_repair_enhanced(raw_text: str, max_repair_attempts: int = 2) -> Dict[str, Any]:
    """Repair invalid JSON deterministically with multiple fallback strategies.
    
    Steps:
    1. Strip leading/trailing junk
    2. Find first '{' and last '}'
    3. Attempt json.loads
    4. If failure, try parse_json_strict_or_extract (from llm_client)
    5. If still failure, run repair LLM prompt
    6. Retry parse with repaired JSON
    
    Returns parsed dict or raises RuntimeError.
    """
    # Step 1: Clean input
    cleaned = raw_text.strip()
    
    # Step 2: Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # Continue to repair steps
    
    # Step 3: Try extract JSON from text
    try:
        from llm_client import parse_json_strict_or_extract
        extracted = parse_json_strict_or_extract(cleaned)
        if extracted:
            return extracted
    except (ImportError, Exception):
        pass
    
    # Step 4: Find JSON boundaries
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    
    if start == -1 or end == -1 or end <= start:
        # No JSON found, try repair with LLM
        return _repair_with_llm_enhanced(cleaned, "No JSON boundaries found")
    
    json_str = cleaned[start:end+1]
    
    # Step 5: Attempt parse of extracted JSON
    for attempt in range(max_repair_attempts):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if attempt == max_repair_attempts - 1:
                # Last attempt: use LLM repair
                return _repair_with_llm_enhanced(json_str, f"JSON decode error: {e}")
            
            # Try simple repairs
            json_str = _simple_json_repair(json_str)
    
    # Should not reach here
    raise RuntimeError(f"Failed to repair JSON after {max_repair_attempts} attempts")


def _simple_json_repair(json_str: str) -> str:
    """Apply simple deterministic JSON repairs."""
    # Fix common issues
    repaired = json_str
    
    # 1. Fix unclosed quotes
    lines = repaired.split('\n')
    for i, line in enumerate(lines):
        # Count quotes in line
        quote_count = line.count('"')
        if quote_count % 2 == 1:
            # Odd number of quotes, add closing quote at end
            lines[i] = line + '"'
    
    repaired = '\n'.join(lines)
    
    # 2. Fix trailing commas in arrays/objects
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    
    # 3. Fix missing commas between object items
    repaired = re.sub(r'"\s*"\s*:', '", ":', repaired)
    
    return repaired


def _repair_with_llm_enhanced(broken_json: str, error_msg: str = "") -> Dict[str, Any]:
    """Use LLM to repair JSON syntax only, without altering content."""
    system = """You are a JSON syntax repair assistant. Fix ONLY the JSON syntax errors.
Do NOT change the content, values, or structure. Only fix:
- Missing or extra commas
- Unclosed quotes
- Unclosed brackets or braces
- Trailing commas
- Invalid escape sequences
- Unterminated strings
- Missing colons between keys and values

Return ONLY the repaired JSON string, no commentary, no markdown fences."""

    try:
        response = llm_json(
            system=system,
            user=f"Repair this JSON (error: {error_msg}):\n\n{broken_json}",
            temperature=0.1,
            max_tokens=len(broken_json) + 1000
        )
        
        # Extract repaired JSON from response
        repaired = ""
        if isinstance(response, dict):
            # Try to find JSON in dict values
            for value in response.values():
                if isinstance(value, str) and '{' in value and '}' in value:
                    repaired = value
                    break
            if not repaired:
                repaired = json.dumps(response)
        else:
            repaired = str(response)
        
        # Clean up markdown fences
        repaired = repaired.strip()
        if repaired.startswith('```json'):
            repaired = repaired[7:]
        if repaired.startswith('```'):
            repaired = repaired[3:]
        if repaired.endswith('```'):
            repaired = repaired[:-3]
        repaired = repaired.strip()
        
        # Parse the repaired JSON
        return json.loads(repaired)
        
    except Exception as e:
        # Save failure for debugging
        debug_dir = Path("scripts") / "_logs" / "json_repair"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"repair_failure_{int(time.time())}.txt"
        debug_file.write_text(f"Original: {broken_json}\n\nError: {e}\n\nLLM Error: {error_msg}", encoding="utf-8")
        raise RuntimeError(f"LLM JSON repair failed: {e}")


def validate_plan_schema_enhanced(plan: Dict[str, Any], niche: str = "") -> Tuple[bool, List[str]]:
    """Enhanced plan validation with keyword relevance checks.
    
    Returns (is_valid, error_messages)
    """
    errors = []
    
    # Required top-level keys
    required_keys = ["niche", "family_id", "family_label", "hubs", "ontology"]
    for key in required_keys:
        if key not in plan:
            errors.append(f"Missing required key: {key}")
    
    # Validate niche matches
    if niche and "niche" in plan:
        plan_niche = plan["niche"].lower()
        input_niche = niche.lower()
        # Basic check: niche should be similar
        if input_niche not in plan_niche and plan_niche not in input_niche:
            errors.append(f"Niche mismatch: plan has '{plan['niche']}', input was '{niche}'")
    
    # Validate ontology
    if "ontology" in plan:
        ontology = plan["ontology"]
        if not isinstance(ontology, dict):
            errors.append("ontology must be a dict")
        else:
            if "core_keywords" not in ontology or not isinstance(ontology["core_keywords"], list):
                errors.append("ontology.core_keywords must be a list")
            if "not_allowed_domains" not in ontology or not isinstance(ontology["not_allowed_domains"], list):
                errors.append("ontology.not_allowed_domains must be a list")
    
    # Validate hubs
    if "hubs" in plan:
        hubs = plan["hubs"]
        if not isinstance(hubs, list):
            errors.append("hubs must be a list")
        else:
            hub_ids = set()
            hub_labels = set()
            
            for i, hub in enumerate(hubs):
                if not isinstance(hub, dict):
                    errors.append(f"Hub {i} must be a dict")
                    continue
                
                # Check required hub keys
                hub_required = ["id", "label", "description", "clusters", "pages"]
                for key in hub_required:
                    if key not in hub:
                        errors.append(f"Hub {i} missing key: {key}")
                
                # Check uniqueness
                hub_id = hub.get("id", "")
                hub_label = hub.get("label", "")
                
                if hub_id in hub_ids:
                    errors.append(f"Hub {i} duplicate id: {hub_id}")
                else:
                    hub_ids.add(hub_id)
                
                if hub_label in hub_labels:
                    errors.append(f"Hub {i} duplicate label: {hub_label}")
                else:
                    hub_labels.add(hub_label)
                
                # Validate clusters
                if "clusters" in hub and isinstance(hub["clusters"], list):
                    cluster_ids = set()
                    for j, cluster in enumerate(hub["clusters"]):
                        if not isinstance(cluster, dict):
                            errors.append(f"Hub {i} cluster {j} must be a dict")
                            continue
                        
                        if "id" not in cluster or "label" not in cluster:
                            errors.append(f"Hub {i} cluster {j} missing id or label")
                            continue
                        
                        cluster_id = cluster.get("id", "")
                        if cluster_id in cluster_ids:
                            errors.append(f"Hub {i} cluster {j} duplicate id: {cluster_id}")
                        else:
                            cluster_ids.add(cluster_id)
                
                # Validate pages
                if "pages" in hub and isinstance(hub["pages"], list):
                    page_slugs = set()
                    page_titles = set()
                    
                    for j, page in enumerate(hub["pages"]):
                        if not isinstance(page, dict):
                            errors.append(f"Hub {i} page {j} must be a dict")
                            continue
                        
                        # Check required page keys
                        page_required = ["title", "slug", "page_type", "internal_links"]
                        for key in page_required:
                            if key not in page:
                                errors.append(f"Hub {i} page {j} missing key: {key}")
                        
                        # Validate page_type
                        if "page_type" in page:
                            valid_types = ["explainer", "howto", "checklist", "comparison", 
                                         "troubleshooting", "guide", "reference", "overview"]
                            if page["page_type"] not in valid_types:
                                errors.append(f"Hub {i} page {j} invalid page_type: {page['page_type']}. Must be one of: {valid_types}")
                        
                        # Check uniqueness
                        page_slug = page.get("slug", "")
                        page_title = page.get("title", "")
                        
                        if page_slug in page_slugs:
                            errors.append(f"Hub {i} page {j} duplicate slug: {page_slug}")
                        else:
                            page_slugs.add(page_slug)
                        
                        if page_title in page_titles:
                            errors.append(f"Hub {i} page {j} duplicate title: {page_title}")
                        else:
                            page_titles.add(page_title)
                        
                        # Validate internal_links
                        if "internal_links" in page and isinstance(page["internal_links"], list):
                            # Check for duplicates within same page
                            link_set = set()
                            for link in page["internal_links"]:
                                if link in link_set:
                                    errors.append(f"Hub {i} page {j} duplicate internal link: {link}")
                                else:
                                    link_set.add(link)
    
    return len(errors) == 0, errors


def validate_keyword_relevance(plan: Dict[str, Any], niche: str) -> Tuple[bool, List[str]]:
    """Validate that hubs, clusters, and pages contain relevant keywords from niche ontology."""
    warnings = []
    
    if "ontology" not in plan or "core_keywords" not in plan["ontology"]:
        return True, warnings  # Skip if no ontology
    
    core_keywords = [kw.lower() for kw in plan["ontology"].get("core_keywords", [])]
    if not core_keywords:
        return True, warnings
    
    niche_lower = niche.lower()
    
    # Check hubs
    if "hubs" in plan:
        for i, hub in enumerate(plan["hubs"]):
            hub_text = f"{hub.get('label', '')} {hub.get('description', '')}".lower()
            
            # Check if hub contains any core keywords
            has_keyword = any(keyword in hub_text for keyword in core_keywords)
            if not has_keyword:
                # Check if hub text contains niche words
                niche_words = set(re.findall(r'\w+', niche_lower))
                hub_words = set(re.findall(r'\w+', hub_text))
                overlap = len(niche_words & hub_words)
                
                if overlap < 2:
                    warnings.append(f"Hub {i} ('{hub.get('label', '')}') may not be relevant to niche. Contains keywords: {[kw for kw in core_keywords if kw in hub_text]}")
    
    return len(warnings) == 0, warnings


def select_blueprint_enhanced(niche: str, min_overlap_threshold: int = 3) -> str:
    """Enhanced blueprint selection with stricter relevance checking.
    
    Fixes the "oceanliving → energy-efficiency" issue by:
    1. Requiring higher keyword overlap
    2. Checking semantic relevance
    3. Creating new-family when uncertain
    """
    niche_lower = niche.lower()
    niche_words = set(re.findall(r'\w+', niche_lower))
    
    # Calculate overlap scores
    scores = {}
    overlaps = {}
    
    for family_id, keywords in FAMILY_KEYWORDS.items():
        score = 0
        family_keyword_set = set()
        
        # Build set of all keyword words for this family
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in niche_lower:
                # Exact phrase match
                score += len(kw) * 2
            # Add individual words from keyword
            family_keyword_set.update(re.findall(r'\w+', kw_lower))
        
        # Calculate word overlap
        word_overlap = len(niche_words & family_keyword_set)
        overlaps[family_id] = word_overlap
        scores[family_id] = score + (word_overlap * 15)  # Weight word overlap higher
    
    # Find best match
    if not scores:
        return "new-family"
    
    best = max(scores, key=scores.get)
    best_overlap = overlaps.get(best, 0)
    best_score = scores.get(best, 0)
    
    # Apply strict relevance guard
    if best_overlap < min_overlap_threshold:
        print(f"  [Blueprint] Not enough overlap ({best_overlap} < {min_overlap_threshold}) for '{best}', creating new-family")
        return "new-family"
    
    # Also check if the best score is too low
    if best_score < 10:  # Higher threshold for low scores
        print(f"  [Blueprint] Score too low ({best_score} < 10) for '{best}', creating new-family")
        return "new-family"
    
    # Special case: check for obvious mismatches
    niche_theme = _detect_niche_theme(niche_lower)
    family_theme = _get_family_theme(best)
    
    if niche_theme and family_theme and niche_theme != family_theme:
        print(f"  [Blueprint] Theme mismatch: niche '{niche_theme}' vs family '{family_theme}', creating new-family")
        return "new-family"
    
    print(f"  [Blueprint] Selected '{best}' with overlap {best_overlap}, score {best_score}")
    return best


def _detect_niche_theme(niche_lower: str) -> str:
    """Detect broad theme of niche for mismatch detection."""
    themes = {
        "ocean": ["ocean", "sea", "marine", "water", "beach", "coast", "fishing", "sailing"],
        "energy": ["energy", "electric", "power", "solar", "wind", "battery", "efficiency"],
        "home": ["home", "house", "garden", "yard", "kitchen", "bathroom", "renovation"],
        "health": ["health", "fitness", "exercise", "diet", "nutrition", "wellness"],
        "tech": ["tech", "digital", "software", "app", "computer", "phone", "internet"],
        "finance": ["finance", "money", "invest", "save", "budget", "debt", "retirement"],
    }
    
    for theme, keywords in themes.items():
        for keyword in keywords:
            if keyword in niche_lower:
                return theme
    
    return ""


def _get_family_theme(family_id: str) -> str:
    """Get broad theme of a family."""
    theme_map = {
        "energy-efficiency": "energy",
        "home-systems": "home",
        "decision-science": "tech",  # cognitive/decision making
        "digital-habits": "tech",
    }
    return theme_map.get(family_id, "")


def generate_plan_contract_enhanced(
    niche: str,
    hub_count: int = 8,
    clusters_per_hub: int = 5,
    pages_per_hub: int = 3,
    family_id: Optional[str] = None,
    temperature: float = 0.25
) -> Dict[str, Any]:
    """Generate a complete taxonomy plan contract with enhanced validation.
    
    Returns a dict with the complete plan including:
    - niche ontology
    - blueprint/family selection (or creation)
    -