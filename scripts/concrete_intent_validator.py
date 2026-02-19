#!/usr/bin/env python3
"""
Concrete Intent Validator (CIV)
Global, niche-agnostic enforcement system that prevents vague hubs/clusters/page titles
and guarantees concrete, high-intent titles for ANY niche.
"""

import re
import json
import time
import random
from typing import List, Tuple, Dict, Set, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Global list of action/problem verbs (high-intent)
ACTION_VERBS = {
    "fix", "prevent", "choose", "compare", "install", "clean", "replace",
    "troubleshoot", "cost", "setup", "calculate", "avoid", "improve",
    "measure", "maintain", "diagnose", "plan", "build", "install", "repair",
    "optimize", "secure", "configure", "upgrade", "debug", "test", "monitor",
    "analyze", "evaluate", "select", "buy", "sell", "negotiate", "budget",
    "save", "invest", "protect", "insure", "document", "backup", "restore",
    "migrate", "integrate", "automate", "schedule", "track", "audit"
}

# Vague-only phrases to reject (unless paired with concrete noun)
VAGUE_PHRASES = {
    "influence", "global influences", "benefits of monitoring", "impacts on",
    "journey", "lifestyle", "mindset", "community", "wellness", "sustainability",
    "awareness", "consciousness", "transformation", "evolution", "paradigm",
    "ecosystem", "synergy", "holistic", "integral", "essence", "being",
    "presence", "flow", "harmony", "balance", "connection", "relationship",
    "dynamics", "factors", "aspects", "elements", "dimensions", "perspectives",
    "approaches", "methodologies", "frameworks", "models", "concepts", "ideas",
    "principles", "values", "beliefs", "attitudes", "perceptions", "experiences"
}

# Stopwords for filtering
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "don", "should",
    "now", "also", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "from", "through", "until", "while"
}

# Universal templates for high-intent titles
TEMPLATES = {
    "how_to": "How to {verb} {noun}: {qualifier}",
    "troubleshooting": "{noun} {verb} Problems: {solution}",
    "cost": "How Much Does {noun} {verb} Cost? {context}",
    "checklist": "{noun} {verb} Checklist: {steps}",
    "vs": "{noun1} vs {noun2}: Which is Better for {verb}?",
    "best_for": "Best {noun} for {verb}: {criteria}",
    "mistakes": "Common {noun} {verb} Mistakes to Avoid",
    "guide": "Complete Guide to {verb} {noun}: {scope}",
    "review": "{noun} {verb} Review: {evaluation}",
    "tips": "{number} Tips for {verb} {noun} {better}"
}

TEMPLATE_IDS = list(TEMPLATES.keys())


class ConcreteIntentValidator:
    """Validates titles for concrete, high-intent content."""
    
    def __init__(self):
        self.action_verbs = ACTION_VERBS
        self.vague_phrases = VAGUE_PHRASES
        self.stopwords = STOPWORDS
        self.templates = TEMPLATES
        self.template_ids = TEMPLATE_IDS
        
        # Compile regex patterns
        self.word_pattern = re.compile(r'\b[\w\-]+\b')
        self.vague_pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in VAGUE_PHRASES) + r')\b', re.IGNORECASE)
        
    def extract_words(self, text: str) -> List[str]:
        """Extract words from text, lowercased."""
        return [w.lower() for w in self.word_pattern.findall(text)]
    
    def has_concrete_noun(self, words: List[str], noun_bank: Optional[Set[str]] = None) -> bool:
        """
        Check if text contains at least one concrete noun/phrase.
        A concrete noun is:
        - Not in stopwords
        - Not a vague phrase
        - Preferably in noun_bank (if provided)
        - Not a single character
        - Contains at least one non-stopword that suggests concreteness
        """
        for word in words:
            if len(word) <= 1:
                continue
            if word in self.stopwords:
                continue
            if word in self.vague_phrases:
                continue
            
            # Check if it looks like a concrete noun
            # Concrete nouns often: not ending in -ing (gerunds), not abstract suffixes
            if word.endswith(('ing', 'ness', 'ment', 'tion', 'sion', 'ity', 'ance', 'ence')):
                # Could still be concrete if it's in noun_bank
                if noun_bank and word in noun_bank:
                    return True
                # Otherwise, check further
                continue
            
            # If we have a noun bank, check if word is in it
            if noun_bank and word in noun_bank:
                return True
            
            # Basic heuristic: word length > 3, not all common
            if len(word) > 3 and word not in {'thing', 'stuff', 'item', 'object', 'element'}:
                return True
        
        return False
    
    def has_action_verb(self, words: List[str]) -> bool:
        """Check if text contains at least one action/problem verb."""
        for word in words:
            if word in self.action_verbs:
                return True
        return False
    
    def has_vague_only(self, text: str, words: List[str]) -> Tuple[bool, str]:
        """
        Check if text contains vague-only phrases without concrete pairing.
        Returns (has_vague_only, phrase_found)
        """
        # Check for vague phrases
        vague_match = self.vague_pattern.search(text.lower())
        if vague_match:
            vague_phrase = vague_match.group(0)
            
            # Check if there's a concrete noun nearby (within 3 words)
            words_lower = [w.lower() for w in self.word_pattern.findall(text)]
            try:
                vague_idx = words_lower.index(vague_phrase.split()[0])
            except (ValueError, IndexError):
                vague_idx = -1
            
            if vague_idx >= 0:
                # Look for concrete nouns in surrounding words
                start = max(0, vague_idx - 3)
                end = min(len(words_lower), vague_idx + 4)
                context = words_lower[start:end]
                
                # Check context for potential concrete nouns
                has_concrete_in_context = False
                for word in context:
                    if (len(word) > 3 and word not in self.stopwords and 
                        word not in self.vague_phrases and not word.endswith(('ing', 'ness', 'ment'))):
                        has_concrete_in_context = True
                        break
                
                if not has_concrete_in_context:
                    return True, vague_phrase
        
        return False, ""
    
    def validate_title(self, title: str, noun_bank: Optional[Set[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate a title against CIV rules.
        Returns (is_valid, reasons)
        """
        reasons = []
        
        if not title or len(title.strip()) < 10:
            reasons.append("Title too short or empty")
            return False, reasons
        
        words = self.extract_words(title)
        
        # Rule 1: Must contain at least ONE concrete noun/phrase
        if not self.has_concrete_noun(words, noun_bank):
            reasons.append("No concrete noun/phrase found")
        
        # Rule 2: Must contain at least ONE action/problem verb
        if not self.has_action_verb(words):
            reasons.append("No action/problem verb found")
        
        # Rule 3: Must not contain vague-only phrases without concrete pairing
        has_vague, vague_phrase = self.has_vague_only(title, words)
        if has_vague:
            reasons.append(f"Contains vague-only phrase: '{vague_phrase}'")
        
        # Additional check: title should be specific, not generic
        word_count = len(words)
        stopword_count = sum(1 for w in words if w in self.stopwords)
        if word_count > 0 and stopword_count / word_count > 0.6:
            reasons.append("Too many stopwords (title too generic)")
        
        is_valid = len(reasons) == 0
        return is_valid, reasons
    
    def score_title(self, title: str, noun_bank: Optional[Set[str]] = None) -> Dict:
        """
        Score a title and return detailed analysis.
        """
        words = self.extract_words(title)
        
        # Find concrete nouns
        concrete_nouns = []
        for word in words:
            if len(word) <= 1 or word in self.stopwords or word in self.vague_phrases:
                continue
            if noun_bank and word in noun_bank:
                concrete_nouns.append(word)
            elif len(word) > 3 and not word.endswith(('ing', 'ness', 'ment', 'tion', 'ity')):
                concrete_nouns.append(word)
        
        # Find action verbs
        action_verbs_found = [w for w in words if w in self.action_verbs]
        
        # Check for vague phrases
        has_vague, vague_phrase = self.has_vague_only(title, words)
        
        # Calculate score
        score = 0
        if concrete_nouns:
            score += 2
        if action_verbs_found:
            score += 2
        if not has_vague:
            score += 1
        if len(words) >= 5:
            score += 1
        
        max_score = 6
        normalized_score = score / max_score
        
        return {
            "title": title,
            "score": normalized_score,
            "concrete_nouns": concrete_nouns,
            "action_verbs": action_verbs_found,
            "has_vague": has_vague,
            "vague_phrase": vague_phrase if has_vague else "",
            "word_count": len(words),
            "is_valid": normalized_score >= 0.7  # 70% threshold
        }
    
    def generate_high_intent_title(self, 
                                  niche: str, 
                                  hub: str = "", 
                                  cluster: str = "", 
                                  noun_bank: Optional[Set[str]] = None,
                                  template_id: Optional[str] = None) -> str:
        """
        Generate a high-intent title using templates and noun bank.
        """
        if not noun_bank:
            noun_bank = set()
        
        # Select template
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
        else:
            template = random.choice(list(self.templates.values()))
        
        # Get nouns from bank or generate fallback
        if noun_bank:
            nouns = list(noun_bank)
            if len(nouns) >= 2:
                noun1 = random.choice(nouns)
                noun2 = random.choice([n for n in nouns if n != noun1])
            else:
                noun1 = niche.split()[0] if niche else "topic"
                noun2 = "alternative"
        else:
            noun1 = niche.split()[0] if niche else "topic"
            noun2 = "alternative"
        
        # Get action verb
        verb = random.choice(list(self.action_verbs))
        
        # Fill template
        title = template
        title = title.replace("{verb}", verb)
        title = title.replace("{noun}", noun1)
        title = title.replace("{noun1}", noun1)
        title = title.replace("{noun2}", noun2)
        title = title.replace("{qualifier}", f"for {niche}")
        title = title.replace("{solution}", "Solved")
        title = title.replace("{context}", "in 2024")
        title = title.replace("{steps}", "Step-by-Step")
        title = title.replace("{criteria}", "Based on Performance")
        title = title.replace("{scope}", "Everything You Need to Know")
        title = title.replace("{evaluation}", "Pros and Cons")
        title = title.replace("{better}", "Effectively")
        title = title.replace("{number}", str(random.randint(5, 15)))
        
        return title


class NounBankGenerator:
    """Generates and manages noun banks for niches."""
    
    def __init__(self, site_root: Path):
        self.site_root = site_root
        self.noun_bank_path = site_root / "data" / "noun_bank.json"
        self.noun_bank_path.parent.mkdir(parents=True, exist_ok=True)
    
    def generate_noun_bank(self, niche: str, hub: str = "", cluster: str = "") -> Set[str]:
        """
        Generate a noun bank for a niche/hub/cluster using LLM.
        This should be called once per niche/hub/cluster combination.
        """
        # Import here to avoid circular imports
        try:
            from llm_client import llm_json
        except ImportError:
            # Fallback for testing
            logger.warning("llm_client not available, using mock noun bank")
            return self._mock_noun_bank(niche, hub, cluster)
        
        prompt = f"""
        Generate a list of 50-80 concrete nouns and noun phrases for the niche "{niche}"{f" with focus on hub: {hub}" if hub else ""}{f" and cluster: {cluster}" if cluster else ""}.
        
        Requirements:
        1. Only concrete, tangible, measurable nouns (tools, objects, devices, materials, products, services, techniques, methods)
        2. No abstract concepts (mindset, journey, awareness, transformation, ecosystem)
        3. No vague terms (things, stuff, items, elements)
        4. Include domain-specific terminology
        5. Include both single words and short phrases (2-3 words max)
        6. Prioritize high-intent, problem-solving nouns
        
        Format as a JSON array of strings.
        Example for "home energy efficiency":
        ["solar panels", "insulation", "thermostat", "LED bulbs", "heat pump", "energy audit", "window film", "weather stripping", "power strip", "smart meter"]
        """
        
        try:
            response = llm_json(
                system="You are a domain expert generating concrete noun lists for content generation.",
                user=prompt,
                temperature=0.3
            )
            
            if isinstance(response, dict) and "nouns" in response:
                nouns = response["nouns"]
            elif isinstance(response, list):
                nouns = response
            else:
                nouns = []
            
            # Filter and clean nouns
            filtered_nouns = self._filter_nouns(nouns)
            
            # Save to cache
            self._save_noun_bank(niche, hub, cluster, filtered_nouns)
            
            return set(filtered_nouns)
            
        except Exception as e:
            logger.error(f"Failed to generate noun bank: {e}")
            return self._fallback_noun_bank(niche, hub, cluster)
    
    def _filter_nouns(self, nouns: List[str]) -> List[str]:
        """Filter noun list to remove duplicates, abstracts, etc."""
        filtered = []
        seen = set()
        civ = ConcreteIntentValidator()
        
        for noun in nouns:
            if not noun or len(noun.strip()) == 0:
                continue
            
            noun_lower = noun.lower().strip()
            
            # Remove duplicates
            if noun_lower in seen:
                continue
            
            # Remove single characters
            if len(noun_lower) <= 1:
                continue
            
            # Remove stopwords-only
            words = civ.extract_words(noun_lower)
            if all(w in civ.stopwords for w in words):
                continue
            
            # Remove vague phrases
            if any(vague in noun_lower for vague in civ.vague_phrases):
                continue
            
            # Remove abstract suffixes
            if noun_lower.endswith(('ness', 'ment', 'tion', 'sion', 'ity', 'ance', 'ence', 'ism', 'ship')):
                # Check if it might still be concrete
                if not any(word in noun_lower for word in ['install', 'repair', 'build', 'tool', 'device', 'system']):
                    continue
            
            filtered.append(noun_lower)
            seen.add(noun_lower)
        
        return filtered[:80]  # Limit to 80 items
    
    def _mock_noun_bank(self, niche: str, hub: str = "", cluster: str = "") -> Set[str]:
        """Mock noun bank for testing."""
        # Generic nouns that work for many niches
        generic_nouns = {
            "guide", "manual", "tutorial", "checklist", "calculator", "tool",
            "software", "app", "device", "system", "equipment", "material",
            "product", "service", "technique", "method", "process", "procedure",
            "template", "worksheet", "spreadsheet", "database", "inventory",
            "schedule", "calendar", "budget", "estimate", "quote", "contract",
            "warranty", "guarantee", "certificate", "license", "permit"
        }
        
        # Niche-specific additions
        niche_lower = niche.lower()
        if "energy" in niche_lower or "efficiency" in niche_lower:
            generic_nouns.update(["solar panel", "insulation", "thermostat", "heat pump", "LED bulb"])
        elif "health" in niche_lower or "fitness" in niche_lower:
            generic_nouns.update(["exercise", "diet", "supplement", "equipment", "routine"])
        elif "finance" in niche_lower or "money" in niche_lower:
            generic_nouns.update(["budget", "investment", "account", "loan", "credit"])
        
        return generic_nouns
    
    def _fallback_noun_bank(self, niche: str, hub: str = "", cluster: str = "") -> Set[str]:
        """Fallback noun bank when LLM fails."""
        return self._mock_noun_bank(niche, hub, cluster)
    
    def _save_noun_bank(self, niche: str, hub: str, cluster: str, nouns: List[str]):
        """Save noun bank to JSON cache."""
        key = f"{niche}_{hub}_{cluster}" if hub or cluster else niche
        
        data = {}
        if self.noun_bank_path.exists():
            try:
                with open(self.noun_bank_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        
        data[key] = {
            "niche": niche,
            "hub": hub,
            "cluster": cluster,
            "nouns": nouns,
            "generated_at": time.time()
        }
        
        with open(self.noun_bank_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_noun_bank(self, niche: str, hub: str = "", cluster: str = "") -> Optional[Set[str]]:
        """Load noun bank from cache if exists and not expired (7 days)."""
        key = f"{niche}_{hub}_{cluster}" if hub or cluster else niche
        
        if not self.noun_bank_path.exists():
            return None
        
        try:
            with open(self.noun_bank_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if key in data:
                entry = data[key]
                generated_at = entry.get("generated_at", 0)
                # Check if cache is still valid (7 days)
                if time.time() - generated_at < 7 * 24 * 3600:
                    return set(entry.get("nouns", []))
        except Exception as e:
            logger.warning(f"Failed to load noun bank cache: {e}")
        
        return None
    
    def get_noun_bank(self, niche: str, hub: str = "", cluster: str = "") -> Set[str]:
        """Get noun bank, generating if needed."""
        # Try to load from cache first
        cached = self.load_noun_bank(niche, hub, cluster)
        if cached:
            return cached
        
        # Generate new noun bank
        return self.generate_noun_bank(niche, hub, cluster)


class TitleGenerator:
    """Generates and validates titles with rewrite-on-fail loop."""
    
    def __init__(self, site_root: Path):
        self.site_root = site_root
        self.civ = ConcreteIntentValidator()
        self.noun_bank_gen = NounBankGenerator(site_root)
    
    def generate_title(self,
                      niche: str,
                      hub: str = "",
                      cluster: str = "",
                      max_attempts: int = 3,
                      use_llm: bool = True) -> Tuple[str, Dict]:
        """
        Generate a concrete, high-intent title with rewrite-on-fail loop.
        Returns (title, metadata)
        """
        # Get noun bank for this niche/hub/cluster
        noun_bank = self.noun_bank_gen.get_noun_bank(niche, hub, cluster)
        
        attempts = 0
        best_title = ""
        best_score = 0
        best_metadata = {}
        
        while attempts < max_attempts:
            attempts += 1
            
            # Try different generation strategies
            if use_llm and attempts == 1:
                title = self._generate_with_llm(niche, hub, cluster, noun_bank)
            else:
                # Fallback to template-based generation
                template_id = random.choice(self.civ.template_ids) if attempts > 1 else None
                title = self.civ.generate_high_intent_title(niche, hub, cluster, noun_bank, template_id)
            
            # Validate the title
            is_valid, reasons = self.civ.validate_title(title, noun_bank)
            score_data = self.civ.score_title(title, noun_bank)
            
            metadata = {
                "attempt": attempts,
                "is_valid": is_valid,
                "reasons": reasons,
                "score": score_data["score"],
                "generation_method": "llm" if (use_llm and attempts == 1) else "template"
            }
            
            # Keep track of best title
            if score_data["score"] > best_score:
                best_title = title
                best_score = score_data["score"]
                best_metadata = metadata
            
            # If title is valid, return it immediately
            if is_valid:
                return title, metadata
        
        # If we exhausted attempts, return the best we found
        if not best_title:
            # Ultimate fallback
            best_title = f"How to Fix Common {niche.split()[0].title()} Problems: Step-by-Step Guide"
            best_metadata = {
                "attempt": attempts,
                "is_valid": True,
                "reasons": ["Fallback title generated"],
                "score": 0.8,
                "generation_method": "fallback"
            }
        
        return best_title, best_metadata
    
    def _generate_with_llm(self, niche: str, hub: str = "", cluster: str = "", noun_bank: Set[str] = None) -> str:
        """Generate title using LLM."""
        try:
            from llm_client import llm_json
        except ImportError:
            # Fallback to template
            return self.civ.generate_high_intent_title(niche, hub, cluster, noun_bank)
        
        # Create prompt with noun bank examples
        noun_examples = list(noun_bank)[:10] if noun_bank and len(noun_bank) > 0 else ["equipment", "system", "process"]
        
        prompt = f"""
        Generate ONE concrete, high-intent title for content about "{niche}"{f" with focus on hub: {hub}" if hub else ""}{f" and cluster: {cluster}" if cluster else ""}.
        
        Requirements:
        1. MUST include at least ONE concrete noun from this list: {', '.join(noun_examples)}
        2. MUST include at least ONE action/problem verb (fix, prevent, choose, compare, install, troubleshoot, etc.)
        3. MUST NOT contain vague phrases like: journey, lifestyle, mindset, community, wellness, sustainability, awareness, transformation, ecosystem
        4. Title should be specific, actionable, and problem-solving
        5. Title length: 10-20 words
        
        Examples of good titles:
        - "How to Install Solar Panels for Maximum Energy Efficiency"
        - "Troubleshooting Common Thermostat Problems in Cold Weather"
        - "Cost Comparison: LED Bulbs vs Traditional Lighting for Home Savings"
        
        Return ONLY the title as a string, no JSON, no quotes, no explanation.
        """
        
        try:
            response = llm_json(
                system="You are an expert content title generator focused on concrete, actionable titles.",
                user=prompt,
                temperature=0.7
            )
            
            # Extract title from response
            if isinstance(response, dict):
                title = response.get("title", "")
            elif isinstance(response, str):
                title = response
            else:
                title = str(response)
            
            # Clean up the title
            title = title.strip().strip('"').strip("'")
            if title:
                return title
        
        except Exception as e:
            logger.warning(f"LLM title generation failed: {e}")
        
        # Fallback
        return self.civ.generate_high_intent_title(niche, hub, cluster, noun_bank)
    
    def validate_and_rewrite(self, title: str, niche: str, hub: str = "", cluster: str = "") -> Tuple[str, Dict]:
        """
        Validate a title and rewrite if needed.
        Returns (rewritten_title, metadata)
        """
        noun_bank = self.noun_bank_gen.get_noun_bank(niche, hub, cluster)
        
        # Validate the existing title
        is_valid, reasons = self.civ.validate_title(title, noun_bank)
        
        if is_valid:
            return title, {
                "valid": True,
                "reasons": [],
                "rewritten": False,
                "original_title": title
            }
        
        # Title is invalid, try to rewrite it
        rewritten_title, metadata = self.generate_title(niche, hub, cluster, max_attempts=2, use_llm=False)
        
        return rewritten_title, {
            "valid": True,
            "reasons": reasons,
            "rewritten": True,
            "original_title": title,
            "rewritten_title": rewritten_title,
            "generation_metadata": metadata
        }


# Global utility functions
def get_civ() -> ConcreteIntentValidator:
    """Get a shared CIV instance."""
    return ConcreteIntentValidator()

def get_title_generator(site_root: Path) -> TitleGenerator:
    """Get a title generator for the given site root."""
    return TitleGenerator(site_root)

def enforce_concrete_titles(titles: List[str], niche: str, site_root: Path) -> List[str]:
    """
    Enforce concrete, high-intent titles for a list of titles.
    Returns validated/rewritten titles.
    """
    generator = get_title_generator(site_root)
    validated_titles = []
    metadata_list = []
    
    for title in titles:
        new_title, metadata = generator.validate_and_rewrite(title, niche)
        validated_titles.append(new_title)
        metadata_list.append(metadata)
    
    return validated_titles

def validate_hub_structure(hubs: List[Dict], niche: str, site_root: Path) -> Tuple[List[Dict], List[str]]:
    """
    Validate hub titles and clusters for concrete intent.
    Returns (validated_hubs, warnings)
    """
    civ = get_civ()
    noun_bank_gen = NounBankGenerator(site_root)
    warnings = []
    
    for hub in hubs:
        hub_title = hub.get("title", "")
        hub_id = hub.get("id", "")
        
        # Get noun bank for this hub
        noun_bank = noun_bank_gen.get_noun_bank(niche, hub_title)
        
        # Validate hub title
        is_valid, reasons = civ.validate_title(hub_title, noun_bank)
        if not is_valid:
            warnings.append(f"Hub '{hub_id}' title '{hub_title}' is vague: {', '.join(reasons)}")
            # Try to generate a better title
            generator = TitleGenerator(site_root)
            new_title, _ = generator.generate_title(niche, hub_title, "", max_attempts=1, use_llm=False)
            hub["title"] = new_title
        
        # Validate cluster titles
        clusters = hub.get("clusters", [])
        for cluster in clusters:
            cluster_title = cluster.get("title", "")
            is_valid, reasons = civ.validate_title(cluster_title, noun_bank)
            if not is_valid:
                warnings.append(f"Cluster '{cluster_title}' in hub '{hub_id}' is vague: {', '.join(reasons)}")
                # Try to generate a better title
                generator = TitleGenerator(site_root)
                new_title, _ = generator.generate_title(niche, hub_title, cluster_title, max_attempts=1, use_llm=False)
                cluster["title"] = new_title
    
    return hubs, warnings
