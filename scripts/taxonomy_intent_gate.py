#!/usr/bin/env python3
"""
taxonomy_intent_gate.py — Split "concreteness" into two distinct gates:
1. Taxonomy Intent Gate: ensures hub/cluster titles are user-intentful and niche-specific
2. Title Rewrite Gate: allows ONLY constrained edits and must preserve meaning

This replaces the monolithic concreteness validation with more targeted,
domain-aware validation that prevents nonsense outputs and domain drift.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from niche_resolver import NicheBreadthResolver, NicheAnalysis, DomainCategory


class IntentType(Enum):
    """Types of user intent for content."""
    HOW_TO = "how_to"           # How to do something
    COMPARISON = "comparison"   # Compare options
    PROBLEM_SOLUTION = "problem_solution"  # Solve a problem
    BUYING_GUIDE = "buying_guide"  # What to buy
    TROUBLESHOOTING = "troubleshooting"  # Fix issues
    BEGINNER_GUIDE = "beginner_guide"  # Getting started
    ADVANCED_TECHNIQUE = "advanced_technique"  # Expert techniques
    REVIEW = "review"           # Product/service review


@dataclass
class TaxonomyIntentResult:
    """Result of taxonomy intent validation."""
    is_valid: bool
    intent_type: Optional[IntentType]
    confidence_score: float
    violations: List[str]
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "intent_type": self.intent_type.value if self.intent_type else None,
            "confidence_score": self.confidence_score,
            "violations": self.violations,
            "suggestions": self.suggestions
        }


@dataclass
class TitleRewriteResult:
    """Result of title rewrite validation."""
    is_valid: bool
    semantic_similarity: float
    shared_keywords: List[str]
    domain_alignment: float
    violations: List[str]
    suggested_fix: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "semantic_similarity": self.semantic_similarity,
            "shared_keywords": self.shared_keywords,
            "domain_alignment": self.domain_alignment,
            "violations": self.violations,
            "suggestified_fix": self.suggested_fix
        }


class TaxonomyIntentGate:
    """Validates that hub/cluster titles are user-intentful and niche-specific."""
    
    def __init__(self):
        self.niche_resolver = NicheBreadthResolver()
        
        # Intent patterns for different intent types
        self.intent_patterns = {
            IntentType.HOW_TO: [
                r"how to", r"step.*by.*step", r"guide to", r"tutorial on",
                r"learn how", r"master.*technique", r"complete guide"
            ],
            IntentType.COMPARISON: [
                r"vs\.?", r"versus", r"comparison", r"which.*better",
                r"pros.*cons", r"advantages.*disadvantages", r"difference between"
            ],
            IntentType.PROBLEM_SOLUTION: [
                r"fix", r"solve", r"troubleshoot", r"repair", r"debug",
                r"problem.*solution", r"issue.*resolve", r"error.*correct"
            ],
            IntentType.BUYING_GUIDE: [
                r"buy", r"purchase", r"choose", r"select", r"best.*for",
                r"top.*pick", r"recommended", r"buying guide", r"shopping"
            ],
            IntentType.TROUBLESHOOTING: [
                r"troubleshoot", r"diagnose", r"debug", r"fix.*issue",
                r"common.*problem", r"error.*message", r"won't.*work"
            ],
            IntentType.BEGINNER_GUIDE: [
                r"beginner", r"starter", r"getting started", r"introduction",
                r"basics", r"fundamentals", r"first.*steps", r"new to"
            ],
            IntentType.ADVANCED_TECHNIQUE: [
                r"advanced", r"expert", r"pro.*tip", r"master.*technique",
                r"optimize", r"maximize", r"professional", r"power user"
            ],
            IntentType.REVIEW: [
                r"review", r"tested", r"hands.*on", r"evaluation",
                r"rating", r"score", r"verdict", r"our.*take"
            ]
        }
        
        # Vague patterns that indicate low intent (softened to conditional fail)
        self.vague_patterns = [
            r"\b(things|stuff|items|elements|aspects|factors)\b",
            r"\b(understanding|exploring|discovering|learning about)\b",
            r"\b(basics|fundamentals|principles|concepts)\b",
            r"\b(introduction|overview|summary|recap)\b",
            r"\b(importance|significance|value|benefits)\b",
            r"\b(journey|path|process|transformation)\b",
            r"\b(mindset|philosophy|approach|perspective)\b"
        ]
        
        # Concreteness markers that can redeem vague phrases
        self.concreteness_markers = [
            r"how to", r"step.*by.*step", r"checklist", r"templates?", r"examples?",
            r"settings", r"tips", r"mistakes", r"troubleshoot", r"fix",
            r"best", r"vs\.?", r"versus", r"review", r"cost",
            r"in \d{4}", r"for \w+", r"with \w+"
        ]
        
        # Domain keyword sets for domain-aware keyword satisfaction
        self.domain_keywords = {
            DomainCategory.PHOTOGRAPHY: {
                "photography", "photo", "photograph", "photographer", "camera", "lens",
                "shoot", "exposure", "aperture", "shutter", "composition", "iso",
                "white balance", "framing", "focus", "raw", "lightroom", "photoshop",
                "editing", "retouching", "color correction", "cropping", "filter"
            },
            DomainCategory.ANIMALS: {
                "animal", "wildlife", "species", "mammal", "bird", "reptile",
                "amphibian", "fish", "insect", "habitat", "behavior", "conservation",
                "taxonomy", "pet", "care", "training", "health", "nutrition"
            },
            DomainCategory.OUTDOORS: {
                "nature", "outdoors", "hiking", "trail", "wilderness", "ecology",
                "environment", "plants", "weather", "terrain", "camping", "backpacking",
                "survival", "navigation", "gear", "equipment"
            },
            DomainCategory.HOME: {
                "home", "house", "interior", "decor", "furniture", "appliance",
                "renovation", "maintenance", "cleaning", "organization", "storage",
                "improvement", "repair", "diy", "gardening", "landscaping"
            },
            DomainCategory.SOFTWARE: {
                "software", "app", "application", "program", "code", "programming",
                "development", "coding", "debugging", "testing", "deployment",
                "framework", "library", "api", "database", "algorithm"
            }
        }
        
        # Theme families for semantic matching (expanded with common themes)
        self.theme_families = {
            "photography": ["photography", "photograph", "photographer", "photo", "camera",
                           "lens", "shoot", "exposure", "aperture", "shutter", "composition",
                           "editing", "lighting", "framing", "focus", "iso", "white balance"],
            "equipment": ["equipment", "gear", "tools", "devices", "apparatus",
                         "instruments", "hardware", "supplies", "kit"],
            "techniques": ["techniques", "methods", "approaches", "strategies",
                          "procedures", "practices", "skills", "tactics"],
            "safety": ["safety", "security", "protection", "precautions",
                      "safeguards", "safety measures", "risk management"],
            "installation": ["installation", "setup", "configuration", "assembly",
                            "implementation", "deployment", "setup process"],
            "maintenance": ["maintenance", "upkeep", "care", "servicing",
                           "preservation", "repair", "cleaning", "troubleshooting"],
            "optimization": ["optimization", "improvement", "enhancement", "tuning",
                            "refinement", "performance", "efficiency", "speed"],
            "troubleshooting": ["troubleshooting", "diagnosis", "debugging",
                               "problem-solving", "fixing", "repair", "error handling"],
            "comparison": ["comparison", "evaluation", "assessment", "analysis",
                          "review", "contrast", "versus", "vs"],
            "beginners": ["beginners", "novices", "starters", "newcomers",
                         "first-timers", "learning", "introductory", "basic"],
            "advanced": ["advanced", "expert", "professional", "master",
                        "specialist", "complex", "sophisticated", "pro"],
            "editing": ["editing", "post-processing", "retouching", "color correction",
                       "cropping", "filter", "effects", "software"],
            "software": ["software", "application", "app", "program", "tool",
                        "platform", "system", "interface"],
            "hardware": ["hardware", "device", "machine", "equipment", "component",
                        "part", "accessory", "peripheral"],
            "training": ["training", "education", "learning", "instruction",
                        "coaching", "mentoring", "workshop", "course"],
            "health": ["health", "wellness", "fitness", "nutrition", "diet",
                      "exercise", "medical", "therapy"],
            "finance": ["finance", "money", "investment", "budgeting", "savings",
                       "retirement", "tax", "financial planning"],
            "travel": ["travel", "trip", "vacation", "journey", "adventure",
                      "destination", "itinerary", "accommodation"],
            "cooking": ["cooking", "recipe", "cuisine", "meal", "dish",
                       "ingredient", "technique", "kitchen"],
            "gardening": ["gardening", "plants", "flowers", "vegetables", "soil",
                         "watering", "pruning", "landscaping"]
        }
        
        # Compile regex patterns
        self.compiled_intent_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.intent_patterns.items()
        }
        
        self.compiled_vague_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.vague_patterns
        ]
    
    def validate_hub_title(self, title: str, niche: str, hub_theme: str) -> TaxonomyIntentResult:
        """Validate a hub title for user intent and niche-specificity."""
        analysis = self.niche_resolver.analyze(niche)
        title_lower = title.lower()
        
        violations = []
        suggestions = []
        
        # 1. Check for vague patterns (SOFTENED: conditional fail only if no concreteness markers)
        vague_found = False
        vague_pattern_matched = None
        for pattern in self.compiled_vague_patterns:
            if pattern.search(title_lower):
                vague_found = True
                vague_pattern_matched = pattern.pattern
                break
        
        # Check for concreteness markers
        has_concreteness_marker = False
        if vague_found:
            # Check if title contains any concreteness marker
            for marker in self.concreteness_markers:
                if re.search(marker, title_lower, re.IGNORECASE):
                    has_concreteness_marker = True
                    break
        
        # Only add violation if vague phrase found AND no concreteness marker
        if vague_found and not has_concreteness_marker:
            violations.append(f"Contains vague phrase: {vague_pattern_matched}")
            suggestions.append("Add concrete language like 'how to', 'step-by-step', 'checklist', etc.")
        
        # 2. DOMAIN-AWARE KEYWORD SATISFACTION
        # Get domain keywords for the niche's domain
        domain_keywords = self.domain_keywords.get(analysis.domain, set())
        
        # Get all possible keyword sets to check
        niche_words = set(niche.lower().split())
        title_words = set(re.findall(r'\w+', title_lower))
        
        # Check if title satisfies keyword requirement through any of:
        # A) Contains niche keywords
        niche_overlap = len(niche_words & title_words)
        has_niche_keywords = niche_overlap > 0
        
        # B) Contains domain synonyms
        has_domain_keywords = len(domain_keywords & title_words) > 0
        
        # C) Contains hub-theme synonyms (if hub theme is part of domain family)
        hub_theme_lower = hub_theme.lower()
        theme_synonyms = self._get_theme_synonyms(hub_theme_lower)
        has_theme_keywords = len(set(theme_synonyms) & title_words) > 0
        
        # Determine if keyword requirement is satisfied
        keyword_satisfied = has_niche_keywords or has_domain_keywords or has_theme_keywords
        
        # Only require keywords for non-broad niches
        if not analysis.is_broad and not keyword_satisfied:
            violations.append(f"Title doesn't contain niche, domain, or theme keywords for: {niche}")
            suggestions.append(f"Include '{niche}', domain terms, or theme-related terms")
        
        # 3. Check for hub theme relevance using semantic matching
        is_theme_relevant, matched_keywords = self._check_theme_relevance(title, hub_theme)
        if not is_theme_relevant:
            violations.append(f"Title not relevant to hub theme: {hub_theme}")
            theme_synonyms = self._get_theme_synonyms(hub_theme.lower())
            if len(theme_synonyms) > 5:
                suggestions.append(f"Reference '{hub_theme}' or related terms like {', '.join(theme_synonyms[:3])}...")
            else:
                suggestions.append(f"Reference '{hub_theme}' or related terms like {', '.join(theme_synonyms)}")
        
        # 4. Detect intent type
        intent_type = self._detect_intent_type(title_lower)
        
        # 5. Calculate confidence score (updated to consider domain keywords)
        confidence_score = self._calculate_confidence_score(
            title_lower, niche, hub_theme, intent_type, vague_found and not has_concreteness_marker,
            niche_overlap, has_domain_keywords, has_theme_keywords
        )
        
        # 6. Generate suggestions if needed
        if violations and not suggestions:
            suggestions = self._generate_title_suggestions(
                title, niche, hub_theme, intent_type, analysis
            )
        
        is_valid = len(violations) == 0 and confidence_score >= 0.6
        
        return TaxonomyIntentResult(
            is_valid=is_valid,
            intent_type=intent_type,
            confidence_score=confidence_score,
            violations=violations,
            suggestions=suggestions
        )
    
    def validate_cluster_title(self, title: str, niche: str, hub: str, cluster_theme: str) -> TaxonomyIntentResult:
        """Validate a cluster title (more specific than hub)."""
        # For clusters, we require even more specificity
        hub_result = self.validate_hub_title(title, niche, hub)
        
        if not hub_result.is_valid:
            return hub_result
        
        # Additional cluster-specific checks
        title_lower = title.lower()
        cluster_theme_lower = cluster_theme.lower()
        
        violations = hub_result.violations.copy()
        suggestions = hub_result.suggestions.copy()
        
        # Check for cluster theme relevance using semantic matching
        is_cluster_theme_relevant, matched_keywords = self._check_theme_relevance(title, cluster_theme)
        if not is_cluster_theme_relevant:
            violations.append(f"Cluster title not specific to theme: {cluster_theme}")
            cluster_synonyms = self._get_theme_synonyms(cluster_theme_lower)
            if len(cluster_synonyms) > 5:
                suggestions.append(f"Make title more specific to '{cluster_theme}' or related terms like {', '.join(cluster_synonyms[:3])}...")
            else:
                suggestions.append(f"Make title more specific to '{cluster_theme}' or related terms like {', '.join(cluster_synonyms)}")
        
        # Check for action-oriented language (clusters should be more actionable)
        action_verbs = {"fix", "solve", "install", "configure", "troubleshoot", 
                       "choose", "compare", "optimize", "implement", "build"}
        title_words = set(re.findall(r'\w+', title_lower))
        has_action_verb = len(action_verbs & title_words) > 0
        
        if not has_action_verb:
            violations.append("Cluster title should be more action-oriented")
            suggestions.append("Include action verbs like 'fix', 'install', 'choose', etc.")
        
        confidence_score = max(0, hub_result.confidence_score - 0.1)  # Slightly stricter
        
        is_valid = len(violations) == 0 and confidence_score >= 0.65
        
        return TaxonomyIntentResult(
            is_valid=is_valid,
            intent_type=hub_result.intent_type,
            confidence_score=confidence_score,
            violations=violations,
            suggestions=suggestions
        )
    
    def _detect_intent_type(self, title_lower: str) -> Optional[IntentType]:
        """Detect the primary intent type from title."""
        intent_scores = {}
        
        for intent_type, patterns in self.compiled_intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(title_lower):
                    score += 1
            
            if score > 0:
                intent_scores[intent_type] = score
        
        if intent_scores:
            # Return intent with highest score
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _calculate_confidence_score(self, title_lower: str, niche: str, hub_theme: str,
                                   intent_type: Optional[IntentType], vague_found: bool,
                                   niche_overlap: int, has_domain_keywords: bool = False,
                                   has_theme_keywords: bool = False) -> float:
        """Calculate confidence score (0-1) for title intentfulness."""
        score = 0.5  # Base score
        
        # Bonus for intent type
        if intent_type:
            score += 0.2
        
        # Penalty for vague patterns (reduced penalty if domain/theme keywords present)
        if vague_found:
            # Less severe penalty if title has domain or theme keywords
            if has_domain_keywords or has_theme_keywords:
                score -= 0.15  # Half penalty
            else:
                score -= 0.3
        
        # Bonus for niche overlap
        if niche_overlap > 0:
            score += 0.1 * niche_overlap
        
        # Bonus for domain keywords
        if has_domain_keywords:
            score += 0.15
        
        # Bonus for theme keywords
        if has_theme_keywords:
            score += 0.1
        
        # Bonus for hub theme inclusion
        if hub_theme.lower() in title_lower:
            score += 0.1
        
        # Bonus for action verbs
        action_verbs = {"how", "fix", "solve", "choose", "compare", "install",
                       "troubleshoot", "optimize", "build", "create"}
        title_words = set(re.findall(r'\w+', title_lower))
        if len(action_verbs & title_words) > 0:
            score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0, min(1, score))
    
    def _get_theme_synonyms(self, theme: str) -> List[str]:
        """Get synonyms for a theme to check relevance."""
        theme_lower = theme.lower()
        
        # First check if we have a theme family
        for family_name, family_words in self.theme_families.items():
            if family_name == theme_lower or theme_lower in family_words:
                return family_words
        
        # Fallback: return the theme itself and common variations
        # Try to find partial matches (e.g., "photo editing" -> check for "editing" family)
        for family_name, family_words in self.theme_families.items():
            if family_name in theme_lower or any(word in theme_lower for word in family_words):
                return family_words
        
        # Last resort: return theme and stem variations
        if theme_lower.endswith('ing'):
            return [theme_lower, theme_lower[:-3]]  # e.g., "photographing" -> "photograph"
        elif theme_lower.endswith('s'):
            return [theme_lower, theme_lower[:-1]]  # e.g., "techniques" -> "technique"
        else:
            return [theme_lower]
    
    def _check_theme_relevance(self, title: str, theme: str) -> Tuple[bool, List[str]]:
        """
        Check if title is relevant to theme using semantic matching.
        
        Returns:
            Tuple of (is_relevant, matched_keywords)
        """
        title_lower = title.lower()
        theme_lower = theme.lower()
        
        # Get theme family words
        theme_words = self._get_theme_synonyms(theme_lower)
        
        # Tokenize title (simple word extraction)
        title_tokens = set(re.findall(r'\w+', title_lower))
        
        # Check for exact matches
        matched = []
        for word in theme_words:
            if word in title_lower:  # Check substring for multi-word phrases
                matched.append(word)
        
        # Also check for word boundaries
        for word in theme_words:
            if re.search(r'\b' + re.escape(word) + r'\b', title_lower):
                if word not in matched:
                    matched.append(word)
        
        # Calculate simple Jaccard similarity
        theme_word_set = set(theme_words)
        intersection = len(title_tokens & theme_word_set)
        union = len(title_tokens | theme_word_set)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Consider relevant if we have matches OR decent similarity
        is_relevant = len(matched) > 0 or jaccard_similarity > 0.1
        
        return is_relevant, matched
    
    def _generate_title_suggestions(self, title: str, niche: str, hub_theme: str,
                                   intent_type: Optional[IntentType], analysis: NicheAnalysis) -> List[str]:
        """Generate suggested title improvements."""
        suggestions = []
        
        # Base template based on intent type
        if intent_type == IntentType.HOW_TO:
            suggestions.append(f"How to {hub_theme} for {niche}: Step-by-Step Guide")
        elif intent_type == IntentType.COMPARISON:
            suggestions.append(f"{hub_theme} Comparison: Best Options for {niche}")
        elif intent_type == IntentType.PROBLEM_SOLUTION:
            suggestions.append(f"How to Fix Common {hub_theme} Problems in {niche}")
        elif intent_type == IntentType.BUYING_GUIDE:
            suggestions.append(f"Buying Guide: Best {hub_theme} for {niche}")
        else:
            # Generic improvement
            suggestions.append(f"{title}: Practical Guide for {niche}")
            suggestions.append(f"How to Master {hub_theme} for {niche}")
        
        return suggestions[:3]  # Return top 3 suggestions


class TitleRewriteGate:
    """Validates title rewrites to prevent nonsense and domain drift."""
    
    def __init__(self):
        self.niche_resolver = NicheBreadthResolver()
        
        # Domain drift detection patterns
        self.domain_drift_patterns = {
            DomainCategory.PHOTOGRAPHY: [
                r"\b(finance|budget|loan|investment|stock|money)\b",
                r"\b(medical|health|treatment|therapy|medicine)\b",
                r"\b(legal|law|attorney|court|contract)\b",
                r"\b(real.*estate|property|mortgage|rent)\b"
            ],
            DomainCategory.FINANCE: [
                r"\b(camera|lens|photo|shoot|edit)\b",
                r"\b(hike|camp|trail|wilderness|outdoor)\b",
                r"\b(recipe|cooking|food|kitchen|meal)\b",
                r"\b(software|code|programming|app|development)\b"
            ],
            DomainCategory.HEALTH: [
                r"\b(software|code|programming|app)\b",
                r"\b(finance|budget|investment|money)\b",
                r"\b(real.*estate|property|construction)\b",
                r"\b(legal|law|attorney)\b"
            ],
            DomainCategory.SOFTWARE: [
                r"\b(camera|lens|photo)\b",
                r"\b(hike|camp|trail)\b",
                r"\b(recipe|cooking|food)\b",
                r"\b(medical|health|treatment)\b"
            ]
        }
        
        # Compile patterns
        self.compiled_drift_patterns = {
            domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for domain, patterns in self.domain_drift_patterns.items()
        }
    
    def validate_rewrite(self, original_title: str, rewritten_title: str,
                        niche: str, domain: DomainCategory, hub_theme: str = "") -> TitleRewriteResult:
        """Validate that a rewritten title preserves meaning and stays in-domain."""
        original_lower = original_title.lower()
        rewritten_lower = rewritten_title.lower()
        
        violations = []
        
        # Check if niche is broad (disable noun injection for broad niches)
        analysis = self.niche_resolver.analyze(niche)
        is_broad_niche = analysis.is_broad
        
        # 1. Check semantic similarity (ENHANCED: require at least 2 shared meaningful tokens OR similarity > threshold)
        original_words = set(re.findall(r'\w+', original_lower))
        rewritten_words = set(re.findall(r'\w+', rewritten_lower))
        
        # Filter out stopwords for meaningful token comparison
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        original_meaningful = {w for w in original_words if w not in stopwords and len(w) > 2}
        rewritten_meaningful = {w for w in rewritten_words if w not in stopwords and len(w) > 2}
        
        shared_meaningful = list(original_meaningful & rewritten_meaningful)
        semantic_similarity = len(shared_meaningful) / max(len(original_meaningful), 1)
        
        # ENHANCED CONSTRAINT: Require at least 2 shared meaningful tokens OR similarity > 0.4
        if len(shared_meaningful) < 2 and semantic_similarity < 0.4:
            violations.append(f"Low semantic similarity: only {len(shared_meaningful)} shared meaningful keywords")
        
        # 2. Check for domain drift
        domain_alignment = self._check_domain_alignment(rewritten_lower, domain)
        
        if domain_alignment < 0.7:
            violations.append(f"Domain drift detected: title contains out-of-domain terms")
        
        # 3. Check for nonsense patterns (random noun injection)
        nonsense_detected = self._detect_nonsense(original_lower, rewritten_lower)
        if nonsense_detected:
            violations.append("Rewrite contains nonsense/non-sequitur terms")
        
        # 4. ENHANCED: Check for required niche/domain term (niche keyword OR domain synonym)
        niche_words = set(niche.lower().split())
        has_niche_term = any(word in rewritten_lower for word in niche_words)
        
        # Get domain keywords from TaxonomyIntentGate (need to import or replicate)
        domain_keywords = self._get_domain_keywords(domain)
        has_domain_term = any(keyword in rewritten_lower for keyword in domain_keywords)
        
        # Also check hub theme synonyms
        has_theme_term = False
        if hub_theme:
            theme_synonyms = self._get_theme_synonyms_for_rewrite(hub_theme.lower())
            has_theme_term = any(synonym in rewritten_lower for synonym in theme_synonyms)
        
        # Title is valid if it contains ANY of: niche term, domain term, or theme term
        has_required_term = has_niche_term or has_domain_term or has_theme_term
        if not has_required_term:
            violations.append(f"Rewrite doesn't contain niche, domain, or theme terms for: {niche}")
        
        # 5. ENHANCED: Check for banned random noun injection with stricter rules
        random_nouns = self._detect_random_noun_injection(original_lower, rewritten_lower)
        if random_nouns:
            # For broad niches, completely ban random noun injection
            if is_broad_niche:
                violations.append(f"Random noun injection detected in broad niche: {random_nouns}")
            else:
                # For specific niches, only allow if nouns are from allowed vocab
                allowed_nouns = niche_words | domain_keywords
                if hub_theme:
                    allowed_nouns |= set(self._get_theme_synonyms_for_rewrite(hub_theme.lower()))
                
                # Check if injected nouns are in allowed vocabulary
                disallowed_nouns = [noun for noun in random_nouns if noun not in allowed_nouns]
                if disallowed_nouns:
                    violations.append(f"Random noun injection with disallowed terms: {disallowed_nouns}")
        
        # 6. Check anchor preservation
        preserves_anchors, preserved_anchors, missing_anchors = self._check_anchor_preservation(
            original_title, rewritten_title, niche, hub_theme
        )
        if not preserves_anchors:
            violations.append(f"Anchor preservation failed: missing {len(missing_anchors)} key terms")
            if missing_anchors:
                violations.append(f"Missing anchors: {missing_anchors[:3]}")  # Show first 3
        
        # 7. ENHANCED: Check that rewritten title matches hub theme family
        if hub_theme:
            theme_relevant, matched_theme_words = self._check_theme_relevance_for_rewrite(rewritten_title, hub_theme)
            if not theme_relevant:
                violations.append(f"Rewrite doesn't match hub theme family: {hub_theme}")
        
        # Calculate overall validity with enhanced criteria
        is_valid = (len(violations) == 0 and
                   (len(shared_meaningful) >= 2 or semantic_similarity >= 0.4) and
                   domain_alignment >= 0.7 and
                   has_required_term)
        
        # Generate suggested fix if invalid
        suggested_fix = None
        if not is_valid and violations:
            suggested_fix = self._generate_fix_suggestion(
                original_title, rewritten_title, niche, domain, violations
            )
            # If no fix generated, provide a safe fallback template
            if not suggested_fix:
                suggested_fix = self._generate_safe_fallback_rewrite(original_title, niche, domain, hub_theme)
        
        return TitleRewriteResult(
            is_valid=is_valid,
            semantic_similarity=semantic_similarity,
            shared_keywords=shared_meaningful,
            domain_alignment=domain_alignment,
            violations=violations,
            suggested_fix=suggested_fix
        )
    
    def _check_domain_alignment(self, title_lower: str, domain: DomainCategory) -> float:
        """Check if title aligns with domain (0-1 score)."""
        if domain not in self.compiled_drift_patterns:
            return 1.0  # No patterns defined for this domain
        
        patterns = self.compiled_drift_patterns[domain]
        drift_terms = []
        
        for pattern in patterns:
            if pattern.search(title_lower):
                drift_terms.append(pattern.pattern)
        
        # Score based on number of drift terms
        if len(drift_terms) == 0:
            return 1.0
        elif len(drift_terms) == 1:
            return 0.5
        else:
            return 0.0
    
    def _detect_nonsense(self, original_lower: str, rewritten_lower: str) -> bool:
        """Detect nonsense/non-sequitur rewrites."""
        # Specific nonsense patterns from observed failures
        # These are combinations that are almost always nonsense
        nonsense_patterns = [
            # Animal + financial terms (e.g., "whale measure", "horse invest")
            r"\b(whale|horse|platypus|elephant|giraffe)\b.*\b(measure|invest|monitor|cost|finance|budget|stock|market)\b",
            # First-aid kit + upgrade (domain drift)
            r"\b(first.*aid.*kit)\b.*\b(upgrade|configure|install|optimize)\b",
            # Tackle box + configure (domain drift)
            r"\b(tackle.*box)\b.*\b(configure|install|setup|program)\b",
            # Brush + install (domain drift)
            r"\b(brush)\b.*\b(install|configure|setup|program)\b",
            # Explicit random noun injection
            r"\b(random.*noun)\b.*\b(in.*2024|in.*2025)\b"
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, rewritten_lower, re.IGNORECASE):
                return True
        
        # Check for completely unrelated word pairs with context awareness
        original_words = set(re.findall(r'\w+', original_lower))
        rewritten_words = set(re.findall(r'\w+', rewritten_lower))
        new_words = rewritten_words - original_words
        
        # If no new words, definitely not nonsense
        if not new_words:
            return False
        
        # Common concreteness markers and intent patterns that are GOOD additions
        good_additions = {
            # Concreteness markers
            "how", "to", "step", "by", "guide", "tutorial", "checklist", "template",
            "example", "settings", "tips", "mistakes", "troubleshoot", "fix", "best",
            "vs", "versus", "review", "cost", "complete", "master", "learn", "improve",
            "optimize", "solve", "choose", "select", "buy", "purchase", "compare",
            "evaluate", "rate", "score", "test", "pros", "cons", "advantages", "disadvantages",
            # Common qualifiers
            "ultimate", "essential", "comprehensive", "definitive", "practical", "effective",
            "proven", "tested", "verified", "expert", "professional", "advanced", "beginner",
            "starter", "novice", "intermediate", "expert", "master", "simple", "easy", "quick",
            "fast", "efficient", "powerful", "reliable", "accurate", "precise", "detailed",
            "thorough", "complete", "full", "comprehensive", "ultimate", "definitive"
        }
        
        # Count how many new words are actually good additions
        good_new_words = [word for word in new_words if word in good_additions]
        
        # If most new words are good additions, don't flag as nonsense
        if len(good_new_words) >= len(new_words) * 0.7:  # 70% or more are good
            return False
        
        # If more than 50% of words are completely new AND not good additions, might be nonsense
        # But be more lenient - only flag if > 60% new AND < 30% good
        new_word_ratio = len(new_words) / len(rewritten_words) if rewritten_words else 0
        good_word_ratio = len(good_new_words) / len(new_words) if new_words else 1.0
        
        if new_word_ratio > 0.6 and good_word_ratio < 0.3:
            return True
        
        return False
    
    def _detect_random_noun_injection(self, original_lower: str, rewritten_lower: str) -> List[str]:
        """Detect random noun injection from global noun bank."""
        # Common random nouns that appear in failures
        random_noun_list = [
            "whale", "horse", "platypus", "elephant", "giraffe",
            "first-aid kit", "tackle box", "brush", "hammer",
            "software", "app", "device", "system"
        ]
        
        original_words = set(re.findall(r'\w+', original_lower))
        injected_nouns = []
        
        for noun in random_noun_list:
            if noun in rewritten_lower and noun not in original_lower:
                injected_nouns.append(noun)
        
        return injected_nouns
    
    def _extract_anchor_keywords(self, title: str, niche: str, hub_theme: str = "") -> List[str]:
        """Extract anchor keywords from title that must be preserved in rewrite."""
        title_lower = title.lower()
        
        # Extract nouns and important phrases (simple heuristic)
        words = re.findall(r'\w+', title_lower)
        
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                     "for", "of", "with", "by", "is", "are", "was", "were", "be",
                     "been", "being", "have", "has", "had", "do", "does", "did"}
        
        # Keep longer words (likely nouns) and words that appear in niche/hub_theme
        anchors = []
        for word in words:
            if (len(word) >= 5 and word not in stop_words) or \
               (word in niche.lower()) or \
               (hub_theme and word in hub_theme.lower()):
                anchors.append(word)
        
        # Also include niche and hub_theme words
        niche_words = niche.lower().split()
        anchors.extend([w for w in niche_words if w not in anchors])
        
        if hub_theme:
            theme_words = hub_theme.lower().split()
            anchors.extend([w for w in theme_words if w not in anchors])
        
        # Deduplicate and return
        return list(dict.fromkeys(anchors))  # Preserves order while removing duplicates
    
    def _check_anchor_preservation(self, original_title: str, rewritten_title: str,
                                 niche: str, hub_theme: str = "") -> Tuple[bool, List[str], List[str]]:
        """
        Check if rewrite preserves anchor keywords.
        
        Returns:
            Tuple of (preserves_anchors, preserved_anchors, missing_anchors)
        """
        anchors = self._extract_anchor_keywords(original_title, niche, hub_theme)
        rewritten_lower = rewritten_title.lower()
        
        preserved = []
        missing = []
        
        for anchor in anchors:
            if anchor in rewritten_lower:
                preserved.append(anchor)
            else:
                missing.append(anchor)
        
        # Need to preserve at least 2 anchors or 50% of anchors (whichever is smaller)
        min_preserved = min(2, max(1, len(anchors) // 2))
        preserves_anchors = len(preserved) >= min_preserved
        
        return preserves_anchors, preserved, missing
    
    def _generate_safe_fallback_rewrite(self, original_title: str, niche: str,
                                      hub_theme: str = "", domain: DomainCategory = None) -> str:
        """Generate a safe fallback rewrite that preserves anchors."""
        anchors = self._extract_anchor_keywords(original_title, niche, hub_theme)
        
        # Safe templates that preserve meaning
        safe_templates = [
            "{Anchor Topic}: Definitions, Examples, and Common Misconceptions",
            "How {Anchor Topic} Works: A Practical Guide",
            "{Anchor Topic} Checklist: Steps, Tools, and FAQs",
            "Understanding {Anchor Topic}: Key Concepts and Applications",
            "{Anchor Topic} Guide: Best Practices and Tips"
        ]
        
        # Use the first anchor as the main topic
        if anchors:
            main_anchor = anchors[0].title()
        else:
            main_anchor = original_title.split(":")[0] if ":" in original_title else original_title
        
        import random
        template = random.choice(safe_templates)
        return template.format(**{"Anchor Topic": main_anchor})
    
    def _get_domain_keywords(self, domain: DomainCategory) -> Set[str]:
        """Get domain keywords for validation."""
        # Domain keyword sets (replicated from TaxonomyIntentGate for consistency)
        domain_keywords = {
            DomainCategory.PHOTOGRAPHY: {
                "photography", "photo", "photograph", "photographer", "camera", "lens",
                "shoot", "exposure", "aperture", "shutter", "composition", "iso",
                "white balance", "framing", "focus", "raw", "lightroom", "photoshop",
                "editing", "retouching", "color correction", "cropping", "filter"
            },
            DomainCategory.ANIMALS: {
                "animal", "wildlife", "species", "mammal", "bird", "reptile",
                "amphibian", "fish", "insect", "habitat", "behavior", "conservation",
                "taxonomy", "pet", "care", "training", "health", "nutrition"
            },
            DomainCategory.OUTDOORS: {
                "nature", "outdoors", "hiking", "trail", "wilderness", "ecology",
                "environment", "plants", "weather", "terrain", "camping", "backpacking",
                "survival", "navigation", "gear", "equipment"
            },
            DomainCategory.HOME: {
                "home", "house", "interior", "decor", "furniture", "appliance",
                "renovation", "maintenance", "cleaning", "organization", "storage",
                "improvement", "repair", "diy", "gardening", "landscaping"
            },
            DomainCategory.SOFTWARE: {
                "software", "app", "application", "program", "code", "programming",
                "development", "coding", "debugging", "testing", "deployment",
                "framework", "library", "api", "database", "algorithm"
            }
        }
        return domain_keywords.get(domain, set())
    
    def _get_theme_synonyms_for_rewrite(self, theme: str) -> List[str]:
        """Get theme synonyms for rewrite validation."""
        # Simple theme synonyms mapping
        theme_synonyms = {
            "photography": ["photography", "photo", "photograph", "camera", "lens", "shoot"],
            "editing": ["editing", "post-processing", "retouching", "color correction", "cropping"],
            "techniques": ["techniques", "methods", "skills", "approaches", "strategies"],
            "equipment": ["equipment", "gear", "tools", "devices", "hardware"],
            "beginners": ["beginners", "novices", "starters", "newcomers", "learning"],
            "advanced": ["advanced", "expert", "professional", "master", "specialist"]
        }
        return theme_synonyms.get(theme.lower(), [theme.lower()])
    
    def _check_theme_relevance_for_rewrite(self, title: str, theme: str) -> Tuple[bool, List[str]]:
        """Check if title is relevant to theme (simplified version)."""
        title_lower = title.lower()
        theme_synonyms = self._get_theme_synonyms_for_rewrite(theme)
        
        matched = []
        for synonym in theme_synonyms:
            if synonym in title_lower:
                matched.append(synonym)
        
        return len(matched) > 0, matched
    
    def _generate_safe_fallback_rewrite(self, original_title: str, niche: str,
                                       domain: DomainCategory, hub_theme: str = "") -> str:
        """Generate a safe fallback rewrite using concrete transformation templates."""
        # Extract core topic from original title
        core_topic = self._extract_core_topic(original_title, niche)
        
        # Domain-appropriate templates
        templates = [
            f"{core_topic}: Step-by-Step Guide",
            f"{core_topic}: Common Mistakes + Fixes",
            f"{core_topic}: Checklist",
            f"How to {core_topic}: Practical Guide",
            f"{core_topic} Best Practices"
        ]
        
        # Add domain-specific templates
        if domain == DomainCategory.PHOTOGRAPHY:
            templates.append(f"{core_topic}: Camera Settings & Techniques")
            templates.append(f"How to Photograph {core_topic}")
        elif domain == DomainCategory.ANIMALS:
            templates.append(f"{core_topic}: Care & Behavior Guide")
            templates.append(f"How to Identify {core_topic}")
        elif domain == DomainCategory.OUTDOORS:
            templates.append(f"{core_topic}: Gear & Safety Guide")
            templates.append(f"How to Explore {core_topic}")
        
        # Return first template that's different from original
        for template in templates:
            if template.lower() != original_title.lower():
                return template
        
        return templates[0]
    
    def _extract_core_topic(self, title: str, niche: str) -> str:
        """Extract core topic from title by removing vague phrases."""
        title_lower = title.lower()
        
        # Remove common vague prefixes
        vague_prefixes = [
            "understanding", "exploring", "discovering", "learning about",
            "introduction to", "overview of", "basics of", "fundamentals of",
            "what is", "what are", "the importance of", "the value of"
        ]
        
        core = title
        for prefix in vague_prefixes:
            if title_lower.startswith(prefix):
                core = title[len(prefix):].strip()
                # Capitalize first letter
                if core:
                    core = core[0].upper() + core[1:]
                break
        
        # If no vague prefix found, try to extract nouns/key domain terms
        if core == title:
            # Simple extraction: take first 3-5 words
            words = title.split()
            if len(words) > 5:
                core = " ".join(words[:4])
        
        return core.strip(" :.-")
    
    def _generate_fix_suggestion(self, original_title: str, rewritten_title: str,
                               niche: str, domain: DomainCategory, violations: List[str]) -> str:
        """Generate a suggested fix for invalid rewrite."""
        # Use safe fallback rewrite as fix suggestion
        return self._generate_safe_fallback_rewrite(original_title, niche, domain)


# Convenience functions
def validate_taxonomy_intent(title: str, niche: str, level: str = "hub",
                           theme: str = "") -> Dict:
    """Convenience function for taxonomy intent validation."""
    gate = TaxonomyIntentGate()
    
    if level == "hub":
        result = gate.validate_hub_title(title, niche, theme)
    else:
        result = gate.validate_cluster_title(title, niche, "", theme)
    
    return result.to_dict()


def validate_title_rewrite(original: str, rewritten: str, niche: str,
                          domain: DomainCategory) -> Dict:
    """Convenience function for title rewrite validation."""
    gate = TitleRewriteGate()
    result = gate.validate_rewrite(original, rewritten, niche, domain)
    return result.to_dict()


if __name__ == "__main__":
    # Test the gates
    print("=== Testing Taxonomy Intent Gate ===")
    
    intent_gate = TaxonomyIntentGate()
    
    test_cases = [
        ("What Defines the Basic Characteristics of Animal Life?", "animals", "hub", "Species Identification"),
        ("How to Choose the Right Camera for Nature Photography", "nature photography", "hub", "Camera Equipment"),
        ("Common Problems with Home Gardening Tools", "home gardening", "cluster", "Tool Maintenance")
    ]
    
    for title, niche, level, theme in test_cases:
        print(f"\nTitle: '{title}'")
        print(f"Niche: {niche}, Level: {level}, Theme: {theme}")
        
        if level == "hub":
            result = intent_gate.validate_hub_title(title, niche, theme)
        else:
            result = intent_gate.validate_cluster_title(title, niche, "", theme)
        
        print(f"  Valid: {result.is_valid}")
        print(f"  Intent: {result.intent_type.value if result.intent_type else 'None'}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        if result.violations:
            print(f"  Violations: {result.violations}")
    
    print("\n=== Testing Title Rewrite Gate ===")
    
    rewrite_gate = TitleRewriteGate()
    
    # Test the problematic rewrites from user report
    problematic_rewrites = [
        ("What Defines the Basic Characteristics of Animal Life?", "whale measure Problems: Solved", "animals", DomainCategory.ANIMALS),
        ("How Do Different Animal Classifications Help Us Understand the Natural World?", "horse invest Review: Pros and Cons", "animals", DomainCategory.ANIMALS),
        ("What Are the Core Components of Animal Anatomy?", "How Much Does platypus monitor Cost? in 2024", "animals", DomainCategory.ANIMALS)
    ]
    
    for original, rewritten, niche, domain in problematic_rewrites:
        print(f"\nOriginal: '{original}'")
        print(f"Rewritten: '{rewritten}'")
        
        result = rewrite_gate.validate_rewrite(original, rewritten, niche, domain)
        print(f"  Valid: {result.is_valid}")
        print(f"  Semantic Similarity: {result.semantic_similarity:.2f}")
        print(f"  Domain Alignment: {result.domain_alignment:.2f}")
        if result.violations:
            print(f"  Violations: {result.violations}")
        if result.suggested_fix:
            print(f"  Suggested Fix: '{result.suggested_fix}'")