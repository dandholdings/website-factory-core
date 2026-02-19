#!/usr/bin/env python3
"""
NicheSpec pre-pass system for programmatic SEO site generator.

Generates a structured specification for any niche to ensure concrete,
high-intent content generation. Used as foundation for taxonomy, hubs,
clusters, and page titles.
"""

import os
import sys
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import llm_json, parse_json_strict_or_extract

logger = logging.getLogger(__name__)


@dataclass
class NicheSpec:
    """Structured specification for a niche."""
    niche: str
    audiences: List[str]
    contexts: List[str]  # climate/geography/domain context
    core_entities: List[str]  # tools/objects/materials/places (10-25)
    core_problems: List[str]  # real pains (10-20)
    constraints: List[str]  # budget/safety/legal/seasonal/skill (6-15)
    monetizable_adjacencies: List[str]  # product/service categories (8-15)
    red_lines: List[str]  # medical/legal/financial boundaries if relevant
    keywords: List[str]  # high-signal terms (20-50)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NicheSpec':
        return cls(**data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate NicheSpec meets minimum requirements."""
        errors = []
        
        # Required field checks
        if not self.niche or not isinstance(self.niche, str):
            errors.append("Niche must be a non-empty string")
        
        if len(self.core_entities) < 10:
            errors.append(f"core_entities must have at least 10 items (got {len(self.core_entities)})")
        
        if len(self.core_problems) < 10:
            errors.append(f"core_problems must have at least 10 items (got {len(self.core_problems)})")
        
        if len(self.keywords) < 20:
            errors.append(f"keywords must have at least 20 items (got {len(self.keywords)})")
        
        # Empty field checks
        for field_name in ['audiences', 'contexts', 'constraints', 'monetizable_adjacencies']:
            field_value = getattr(self, field_name)
            if not field_value or len(field_value) == 0:
                errors.append(f"{field_name} must not be empty")
        
        # Quality checks
        if any(len(str(item).strip()) == 0 for item in self.core_entities):
            errors.append("core_entities contains empty items")
        
        if any(len(str(item).strip()) == 0 for item in self.core_problems):
            errors.append("core_problems contains empty items")
        
        return len(errors) == 0, errors


class NicheSpecGenerator:
    """Generates and validates NicheSpec for any niche."""
    
    def __init__(self, site_root: Optional[Path] = None):
        self.site_root = site_root or Path.cwd()
        self.cache_dir = self.site_root / "data" / "niche_specs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, niche: str, max_attempts: int = 3) -> NicheSpec:
        """
        Generate NicheSpec for a niche with validation and retry logic.
        
        Args:
            niche: The niche to generate spec for
            max_attempts: Maximum number of LLM attempts
            
        Returns:
            Validated NicheSpec
            
        Raises:
            RuntimeError: If generation fails after all attempts
        """
        logger.info(f"Generating NicheSpec for niche: {niche}")
        
        # Check cache first
        cached = self._load_from_cache(niche)
        if cached:
            logger.info(f"Using cached NicheSpec for {niche}")
            return cached
        
        for attempt in range(max_attempts):
            try:
                spec_dict = self._call_llm_for_spec(niche, attempt)
                niche_spec = NicheSpec.from_dict(spec_dict)
                
                # Validate
                is_valid, errors = niche_spec.validate()
                if is_valid:
                    logger.info(f"NicheSpec generated successfully for {niche}")
                    self._save_to_cache(niche, niche_spec)
                    return niche_spec
                else:
                    logger.warning(f"NicheSpec validation failed (attempt {attempt + 1}/{max_attempts}): {errors}")
                    if attempt < max_attempts - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
            
            except Exception as e:
                logger.warning(f"NicheSpec generation failed (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        
        # All attempts failed, try fallback
        logger.warning(f"All {max_attempts} attempts failed, using fallback NicheSpec")
        return self._generate_fallback_spec(niche)
    
    def _call_llm_for_spec(self, niche: str, attempt: int = 0) -> Dict[str, Any]:
        """Call LLM to generate NicheSpec JSON."""
        system_prompt = """You are a domain expert creating structured niche specifications for content generation.

CRITICAL REQUIREMENTS:
1. Output MUST be a single valid JSON object matching the exact schema below
2. NO markdown, NO code fences, NO explanatory text
3. All arrays must have minimum items as specified
4. All items must be concrete, specific, and actionable

JSON SCHEMA:
{
  "niche": "string (the input niche)",
  "audiences": ["array of target audience segments"],
  "contexts": ["climate/geography/domain contexts relevant to niche"],
  "core_entities": ["10-25 tools/objects/materials/places/products"],
  "core_problems": ["10-20 real pains, challenges, needs"],
  "constraints": ["6-15 budget/safety/legal/seasonal/skill constraints"],
  "monetizable_adjacencies": ["8-15 product/service categories"],
  "red_lines": ["medical/legal/financial boundaries if relevant"],
  "keywords": ["20-50 high-signal search terms"]
}

EXAMPLE for "home energy efficiency":
{
  "niche": "home energy efficiency",
  "audiences": ["homeowners", "renters", "property managers", "DIY enthusiasts"],
  "contexts": ["cold climates", "hot climates", "urban areas", "suburban homes"],
  "core_entities": ["solar panels", "insulation", "thermostat", "heat pump", "LED bulbs", "energy audit", "window film", "weather stripping", "power strip", "smart meter", "water heater", "HVAC system", "attic fan", "programmable timer", "energy monitor"],
  "core_problems": ["high electricity bills", "drafty windows", "uneven room temperatures", "old inefficient appliances", "poor insulation", "air leaks", "peak hour pricing", "rising energy costs", "carbon footprint concerns", "government rebate confusion"],
  "constraints": ["budget under $500", "DIY skill level", "local building codes", "HOA restrictions", "winter preparation timeline", "summer heat waves", "rental property limitations", "historic home preservation rules"],
  "monetizable_adjacencies": ["solar installation", "home insulation services", "smart home devices", "energy audits", "HVAC maintenance", "window replacement", "appliance upgrades", "government rebate consulting"],
  "red_lines": ["no medical advice", "no electrical work without licensed professional", "no guarantee of specific savings"],
  "keywords": ["reduce energy bill", "home insulation tips", "smart thermostat installation", "DIY weather stripping", "solar panel cost", "energy efficient appliances", "window insulation film", "attic ventilation", "HVAC maintenance", "power strip energy savings"]
}"""
        
        user_prompt = f"""Generate a complete NicheSpec for the niche: "{niche}"

IMPORTANT:
- core_entities: MUST have 10-25 concrete tools/objects/materials/places/products
- core_problems: MUST have 10-20 real pains, challenges, needs
- keywords: MUST have 20-50 high-signal search terms
- All arrays must be non-empty
- Be specific and concrete, not generic

Return ONLY the JSON object, no other text."""

        # Adjust temperature based on attempt
        temperature = 0.7 if attempt == 0 else 0.9  # More creative on retry
        
        response = llm_json(
            system=system_prompt,
            user=user_prompt,
            temperature=temperature,
            max_tokens=4000
        )
        
        # Ensure response is dict
        if isinstance(response, dict):
            # Add niche if missing
            if "niche" not in response:
                response["niche"] = niche
            return response
        else:
            raise ValueError(f"Expected dict from LLM, got {type(response)}")
    
    def _generate_fallback_spec(self, niche: str) -> NicheSpec:
        """Generate a fallback NicheSpec when LLM fails."""
        logger.warning(f"Generating fallback NicheSpec for {niche}")
        
        # Simple deterministic fallback
        words = niche.lower().split()
        base_word = words[0] if words else "topic"
        
        return NicheSpec(
            niche=niche,
            audiences=["general audience", "enthusiasts", "professionals"],
            contexts=["general context", "various applications"],
            core_entities=[f"{base_word} tool", f"{base_word} equipment", f"{base_word} material", 
                          f"{base_word} system", f"{base_word} device", f"{base_word} product",
                          f"{base_word} service", f"{base_word} technique", f"{base_word} method",
                          f"{base_word} process", f"{base_word} approach", f"{base_word} solution"],
            core_problems=[f"{base_word} cost", f"{base_word} efficiency", f"{base_word} maintenance",
                          f"{base_word} installation", f"{base_word} selection", f"{base_word} troubleshooting",
                          f"{base_word} optimization", f"{base_word} safety", f"{base_word} reliability",
                          f"{base_word} performance"],
            constraints=["budget constraints", "time constraints", "skill level", "safety requirements",
                        "legal regulations", "seasonal factors", "availability"],
            monetizable_adjacencies=[f"{base_word} consulting", f"{base_word} services", f"{base_word} products",
                                    f"{base_word} training", f"{base_word} maintenance", f"{base_word} installation"],
            red_lines=["no medical advice", "no legal advice", "consult professionals for critical decisions"],
            keywords=[f"{base_word} guide", f"{base_word} tutorial", f"{base_word} how-to", f"{base_word} tips",
                     f"{base_word} best practices", f"{base_word} troubleshooting", f"{base_word} installation",
                     f"{base_word} maintenance", f"{base_word} cost", f"{base_word} benefits"]
        )
    
    def _save_to_cache(self, niche: str, spec: NicheSpec) -> None:
        """Save NicheSpec to cache file."""
        cache_file = self.cache_dir / f"{self._slugify(niche)}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_from_cache(self, niche: str) -> Optional[NicheSpec]:
        """Load NicheSpec from cache if exists and recent (< 7 days)."""
        cache_file = self.cache_dir / f"{self._slugify(niche)}.json"
        if not cache_file.exists():
            return None
        
        try:
            # Check file age (7 days max)
            import time
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > 7 * 24 * 3600:  # 7 days in seconds
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            spec = NicheSpec.from_dict(data)
            # Quick validation
            if len(spec.core_entities) >= 5 and len(spec.core_problems) >= 5:
                return spec
        
        except Exception as e:
            logger.warning(f"Failed to load cached NicheSpec: {e}")
        
        return None
    
    def _slugify(self, text: str) -> str:
        """Simple slugify function."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_-]+', '-', text)
        return text.strip('-')


def generate_niche_spec(niche: str, site_root: Optional[Path] = None) -> NicheSpec:
    """
    Convenience function to generate NicheSpec.
    
    Args:
        niche: The niche to generate spec for
        site_root: Optional site root path for cache
        
    Returns:
        Validated NicheSpec
    """
    generator = NicheSpecGenerator(site_root)
    return generator.generate(niche)


def generate_taxonomy_from_niche_spec(
    niche_spec: NicheSpec,
    hub_count: int = 8,
    clusters_per_hub: int = 5,
    pages_per_hub: int = 3
) -> Dict[str, Any]:
    """
    Generate taxonomy (hubs and clusters) from a NicheSpec.
    
    Args:
        niche_spec: The NicheSpec to generate taxonomy from
        hub_count: Number of hubs to generate
        clusters_per_hub: Number of clusters per hub
        pages_per_hub: Number of pages per hub
        
    Returns:
        Dictionary with taxonomy structure: {
            "family_id": "niche-spec",
            "family_label": "Custom {niche} Family",
            "hubs": [...]
        }
    """
    logger.info(f"Generating taxonomy from NicheSpec for: {niche_spec.niche}")
    
    # Use NicheSpec data to create concrete, high-intent hubs
    hubs = []
    
    # Create hubs from core entities, problems, and contexts
    all_hub_sources = []
    
    # 1. Core entities as hubs (most concrete)
    for entity in niche_spec.core_entities[:hub_count//2]:
        all_hub_sources.append(("entity", entity))
    
    # 2. Core problems as hubs (pain-focused)
    for problem in niche_spec.core_problems[:hub_count//3]:
        all_hub_sources.append(("problem", problem))
    
    # 3. Contexts as hubs (situational)
    for context in niche_spec.contexts[:hub_count//4]:
        all_hub_sources.append(("context", context))
    
    # 4. Monetizable adjacencies as hubs (commercial)
    for adjacency in niche_spec.monetizable_adjacencies[:hub_count//4]:
        all_hub_sources.append(("adjacency", adjacency))
    
    # Ensure we have enough hub sources
    while len(all_hub_sources) < hub_count:
        # Add more from core entities
        for entity in niche_spec.core_entities[len(all_hub_sources):len(all_hub_sources)+2]:
            if entity not in [src[1] for src in all_hub_sources]:
                all_hub_sources.append(("entity", entity))
    
    # Intent types for clusters
    intent_types = ["howto", "troubleshoot", "cost_decision", "checklist_safety", "local_seasonal"]
    
    # Create hubs from selected sources
    for i, (source_type, source_text) in enumerate(all_hub_sources[:hub_count]):
        hub_id = _slugify_hub_id(niche_spec.niche, source_text, i+1)
        
        # Generate anchor terms for this hub (3-6 highly relevant terms)
        anchor_terms = []
        
        # 1. First, find core entities that are semantically related to the hub
        hub_keywords = source_text.lower().split()
        for entity in niche_spec.core_entities:
            entity_lower = entity.lower()
            
            # Check for direct match or strong semantic relationship
            # Score based on multiple criteria
            score = 0
            
            # Direct substring match (strongest)
            if any(keyword in entity_lower for keyword in hub_keywords):
                score += 3
            
            # Shared words (medium)
            entity_words = set(entity_lower.split())
            hub_words = set(hub_keywords)
            shared_words = entity_words.intersection(hub_words)
            if shared_words:
                score += len(shared_words) * 2
            
            # Length similarity (shorter entities are more concrete)
            if len(entity_lower.split()) <= 3:
                score += 1
            
            # Contains niche keywords (from niche spec keywords)
            if any(keyword in entity_lower for keyword in niche_spec.keywords):
                score += 1
            
            # Only add if score is high enough (>= 3)
            if score >= 3 and entity not in anchor_terms:
                anchor_terms.append(entity)
                if len(anchor_terms) >= 4:  # Get 4 good entities
                    break
        
        # 2. Add core problems that are relevant
        if len(anchor_terms) < 6:
            for problem in niche_spec.core_problems:
                problem_lower = problem.lower()
                # Check if problem relates to hub
                if any(keyword in problem_lower for keyword in hub_keywords):
                    if problem not in anchor_terms:
                        anchor_terms.append(problem)
                        if len(anchor_terms) >= 6:
                            break
        
        # 3. Add constraints only if we still need more
        if len(anchor_terms) < 6:
            for constraint in niche_spec.constraints:
                constraint_lower = constraint.lower()
                # Check if constraint is concrete (not too vague)
                vague_indicators = ["constraints", "requirements", "factors", "considerations"]
                if not any(indicator in constraint_lower for indicator in vague_indicators):
                    if constraint not in anchor_terms:
                        anchor_terms.append(constraint)
                        if len(anchor_terms) >= 6:
                            break
        
        # Ensure we have at least 3 anchor terms
        if len(anchor_terms) < 3:
            # Fallback: take top core entities
            for entity in niche_spec.core_entities[:3]:
                if entity not in anchor_terms:
                    anchor_terms.append(entity)
                    if len(anchor_terms) >= 3:
                        break
        
        # Create concrete, high-intent hub title (no generic labels)
        if source_type == "entity":
            hub_label = f"{source_text.title()}: Selection, Installation & Maintenance"
            hub_desc = f"Practical guide to choosing, installing, and maintaining {source_text.lower()} for optimal performance."
        elif source_type == "problem":
            hub_label = f"How to Fix {source_text}: Step-by-Step Solutions"
            hub_desc = f"Diagnose and solve {source_text.lower()} with proven troubleshooting methods and expert advice."
        elif source_type == "context":
            hub_label = f"{source_text.title()}: Practical Applications & Solutions"
            hub_desc = f"Real-world applications and tailored solutions for {source_text.lower()} scenarios in {niche_spec.niche}."
        elif source_type == "adjacency":
            hub_label = f"{source_text.title()}: Professional Services & Solutions"
            hub_desc = f"Expert {source_text.lower()} services, providers, and solutions for {niche_spec.niche}."
        else:
            hub_label = f"{source_text.title()}: Practical Guide"
            hub_desc = f"Actionable guidance and solutions for {source_text.lower()} in {niche_spec.niche}."
        
        # Create clusters for this hub with proper intent types
        clusters = []
        for j in range(clusters_per_hub):
            # Determine intent type for this cluster (rotate through intent types)
            intent_type = intent_types[j % len(intent_types)]
            
            # Generate cluster title based on intent type and hub source
            if intent_type == "howto":
                if source_type == "entity":
                    cluster_title = f"How to Choose the Right {source_text}"
                elif source_type == "problem":
                    cluster_title = f"How to Solve {source_text} Step by Step"
                else:
                    cluster_title = f"How to Apply {source_text} Effectively"
                promise = f"Learn the exact steps to successfully implement {source_text.lower()}."
                
            elif intent_type == "troubleshoot":
                if source_type == "entity":
                    cluster_title = f"{source_text} Troubleshooting: Fix Common Problems"
                elif source_type == "problem":
                    cluster_title = f"Diagnosing {source_text}: Root Causes & Solutions"
                else:
                    cluster_title = f"Troubleshooting {source_text} Issues"
                promise = f"Identify and fix common problems with {source_text.lower()} quickly and effectively."
                
            elif intent_type == "cost_decision":
                if source_type == "entity":
                    cluster_title = f"{source_text} Cost Analysis: Budget & ROI"
                elif source_type == "problem":
                    cluster_title = f"Cost to Fix {source_text}: Professional vs DIY"
                else:
                    cluster_title = f"{source_text} Cost & Pricing Guide"
                promise = f"Make informed budget decisions about {source_text.lower()} with clear cost breakdowns."
                
            elif intent_type == "checklist_safety":
                if source_type == "entity":
                    cluster_title = f"{source_text} Safety Checklist: Essential Precautions"
                elif source_type == "problem":
                    cluster_title = f"Preventing {source_text}: Safety Measures"
                else:
                    cluster_title = f"{source_text} Safety & Best Practices"
                promise = f"Ensure safety and avoid common mistakes when working with {source_text.lower()}."
                
            elif intent_type == "local_seasonal":
                if source_type == "entity":
                    cluster_title = f"{source_text} for Your Climate & Region"
                elif source_type == "problem":
                    cluster_title = f"{source_text} Solutions for Local Conditions"
                else:
                    cluster_title = f"{source_text} Regional Adaptations"
                promise = f"Adapt {source_text.lower()} solutions to your local climate and seasonal conditions."
            
            # Fix cluster title punctuation and casing
            cluster_title = _fix_cluster_title_punctuation(cluster_title, intent_type)
            
            # Generate anchors for this cluster (2-4 concrete terms)
            cluster_anchors = []
            if anchor_terms:
                # Use first 2-4 anchor terms
                cluster_anchors = anchor_terms[:min(4, len(anchor_terms))]
            else:
                # Fallback to core entities
                cluster_anchors = niche_spec.core_entities[:min(4, len(niche_spec.core_entities))]
            
            cluster_id = f"{hub_id}-{_slugify(cluster_title)}"
            cluster_slug = _slugify(cluster_title)
            
            clusters.append({
                "id": cluster_id,
                "slug": cluster_slug,
                "title": cluster_title,
                "intent_type": intent_type,
                "anchors": cluster_anchors,
                "promise": promise
            })
        
        hubs.append({
            "id": hub_id,
            "slug": hub_id,
            "title": hub_label,
            "description": hub_desc,
            "facet": source_type,
            "anchor_terms": anchor_terms[:6],  # Limit to 6 terms
            "clusters": clusters,
            "related_hubs": []  # Will be populated later
        })
    
    # Create related hub relationships
    hub_ids = [hub["id"] for hub in hubs]
    for i, hub in enumerate(hubs):
        # Connect to 2-3 other hubs (avoid self)
        possible_connections = [hid for hid in hub_ids if hid != hub["id"]]
        related_count = min(3, len(possible_connections))
        if related_count > 0:
            hub["related_hubs"] = possible_connections[:related_count]
    
    return {
        "family_id": "niche-spec",
        "family_label": f"Custom {niche_spec.niche.title()} Family",
        "hubs": hubs
    }


def _slugify_hub_id(niche: str, text: str, index: int) -> str:
    """Generate a slugified hub ID."""
    import re
    # Clean the text
    clean_text = text.lower().strip()
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    clean_text = re.sub(r'[\s_-]+', '-', clean_text)
    clean_text = clean_text.strip('-')
    
    # Use niche prefix if text is short
    if len(clean_text) < 3:
        niche_slug = niche.lower().replace(' ', '-')
        clean_text = f"{niche_slug}-{clean_text}"
    
    # Add index for uniqueness
    return f"{clean_text}-{index}"


def _slugify(text: str) -> str:
    """Simple slugify function."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-')


def _fix_cluster_title_punctuation(title: str, intent_type: str) -> str:
    """
    Fix punctuation and casing issues in cluster titles.
    
    Args:
        title: The cluster title to fix
        intent_type: The intent type (howto, troubleshoot, etc.)
        
    Returns:
        Fixed title with proper punctuation and casing
    """
    import re
    
    # Check if title already has a colon with subtitle
    has_colon_subtitle = ':' in title
    
    # Ensure title ends with proper punctuation
    if not title.endswith(('.', '!', '?', ':')):
        # Add punctuation based on intent type, but only if no colon subtitle already exists
        if not has_colon_subtitle:
            # Check if title already ends with the phrase we're about to add
            if intent_type == "howto" and not title.endswith("Step by Step"):
                title = title + " Step by Step"
            elif intent_type == "troubleshoot" and not title.endswith("Common Problems & Solutions"):
                title = title + ": Common Problems & Solutions"
            elif intent_type == "cost_decision" and not title.endswith("Budget & ROI Analysis"):
                title = title + ": Budget & ROI Analysis"
            elif intent_type == "checklist_safety" and not title.endswith("Essential Precautions"):
                title = title + ": Essential Precautions"
            elif intent_type == "local_seasonal" and not title.endswith("for Your Region"):
                title = title + " for Your Region"
        else:
            # Title has colon but doesn't end with punctuation - add period
            title = title + "."
    
    # Fix casing issues (title case for main parts)
    # Split by colon to handle subtitles
    if ':' in title:
        parts = title.split(':', 1)
        main_part = parts[0].strip()
        subtitle = parts[1].strip()
        
        # Apply title case to main part (but keep certain words lowercase)
        main_part_words = main_part.split()
        fixed_main = []
        for i, word in enumerate(main_part_words):
            if i == 0 or word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'by', 'with', 'in', 'of']:
                fixed_main.append(word.capitalize())
            else:
                fixed_main.append(word.lower())
        
        main_part = ' '.join(fixed_main)
        title = f"{main_part}: {subtitle}"
    else:
        # Simple title case for titles without colons
        words = title.split()
        fixed_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'by', 'with', 'in', 'of']:
                fixed_words.append(word.capitalize())
            else:
                fixed_words.append(word.lower())
        title = ' '.join(fixed_words)
    
    # Ensure proper spacing around colons
    title = re.sub(r'\s*:\s*', ': ', title)
    
    # Fix common issues
    title = title.replace('Step By Step', 'Step by Step')
    title = title.replace('DiY', 'DIY')
    title = title.replace('Roi', 'ROI')
    
    return title


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NicheSpec for a niche")
    parser.add_argument("niche", help="The niche to generate spec for")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        spec = generate_niche_spec(args.niche)
        
        print(f"[OK] NicheSpec generated successfully for: {args.niche}")
        print(f"   Core entities: {len(spec.core_entities)} items")
        print(f"   Core problems: {len(spec.core_problems)} items")
        print(f"   Keywords: {len(spec.keywords)} items")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(spec.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved to: {args.output}")
        else:
            print("\n" + json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"[FAIL] Failed to generate NicheSpec: {e}")
        sys.exit(1)