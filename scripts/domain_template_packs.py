#!/usr/bin/env python3
"""
domain_template_packs.py — Domain-aware template packs for title rewriting.

Replaces CIV freeform rewriting with constrained, domain-appropriate templates
that preserve semantic meaning and prevent domain drift.
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass

from niche_resolver import DomainCategory, NicheBreadthResolver
from taxonomy_intent_gate import IntentType, TaxonomyIntentGate


@dataclass
class TemplateRewrite:
    """A constrained title rewrite using domain-appropriate templates."""
    original_title: str
    rewritten_title: str
    template_id: str
    domain: DomainCategory
    intent_type: IntentType
    confidence_score: float
    validation_passed: bool
    validation_notes: List[str]


class DomainTemplatePack:
    """Domain-specific template packs for title rewriting."""
    
    def __init__(self):
        self.niche_resolver = NicheBreadthResolver()
        self.intent_gate = TaxonomyIntentGate()
        
        # Domain-specific template libraries
        self.domain_templates = {
            DomainCategory.PHOTOGRAPHY: {
                "how_to_equipment": [
                    "How to Choose the Right {noun} for {niche} Photography",
                    "{noun} Setup Guide for {niche} Photography",
                    "Best {noun} Settings for {niche} Photography"
                ],
                "technique_tutorial": [
                    "Mastering {technique} for {niche} Photography",
                    "Step-by-Step Guide to {technique} in {niche} Photography",
                    "Advanced {technique} Techniques for {niche} Photos"
                ],
                "problem_solution": [
                    "How to Fix Common {noun} Problems in {niche} Photography",
                    "Troubleshooting {noun} Issues for {niche} Photographers",
                    "Solving {problem} in {niche} Photography"
                ],
                "buying_guide": [
                    "Best {noun} for {niche} Photography: Buyer's Guide",
                    "{niche} Photography {noun} Comparison: Top Picks",
                    "How to Choose {noun} for {niche} Photography Needs"
                ]
            },
            DomainCategory.ANIMALS: {
                "identification_guide": [
                    "How to Identify {animal_type} Species in {niche}",
                    "{animal_type} Identification Guide for {niche}",
                    "Recognizing {animal_type} in Their Natural Habitat"
                ],
                "behavior_study": [
                    "Understanding {animal_type} Behavior in {niche}",
                    "How {animal_type} Adapt to {environment} Conditions",
                    "Studying {animal_type} Social Structures in {niche}"
                ],
                "conservation_guide": [
                    "How to Protect {animal_type} in {niche}",
                    "Conservation Strategies for {animal_type} in {niche}",
                    "Helping {animal_type} Thrive in {environment}"
                ],
                "care_guide": [
                    "Proper Care for {animal_type} in {niche} Settings",
                    "How to Maintain {animal_type} Health in {environment}",
                    "{animal_type} Care Guide for {niche} Enthusiasts"
                ]
            },
            DomainCategory.NATURE: {
                "ecosystem_guide": [
                    "Understanding {ecosystem_type} Ecosystems in {niche}",
                    "How {ecosystem_type} Systems Work in {niche}",
                    "Exploring {ecosystem_type} Dynamics in {environment}"
                ],
                "plant_identification": [
                    "How to Identify {plant_type} in {niche}",
                    "{plant_type} Recognition Guide for {niche}",
                    "Common {plant_type} Species in {environment}"
                ],
                "conservation_practice": [
                    "Sustainable Practices for {niche} Conservation",
                    "How to Preserve {ecosystem_type} in {niche}",
                    "Protecting {niche} Ecosystems: Practical Guide"
                ],
                "field_technique": [
                    "Field Techniques for Studying {niche}",
                    "How to Document {observation_type} in {niche}",
                    "Research Methods for {niche} Exploration"
                ]
            },
            DomainCategory.OUTDOORS: {
                "gear_guide": [
                    "Essential {gear_type} for {niche} Adventures",
                    "How to Choose {gear_type} for {niche} Conditions",
                    "Best {gear_type} for {activity} in {niche}"
                ],
                "safety_guide": [
                    "Safety Protocols for {niche} Activities",
                    "How to Stay Safe During {activity} in {niche}",
                    "{niche} Safety Guide: Essential Precautions"
                ],
                "skill_tutorial": [
                    "Mastering {skill} for {niche} Enthusiasts",
                    "How to Improve Your {skill} in {niche} Settings",
                    "Advanced {skill} Techniques for {activity}"
                ],
                "planning_guide": [
                    "How to Plan a Successful {niche} Trip",
                    "{niche} Trip Planning: Complete Checklist",
                    "Preparing for {activity} in {environment}"
                ]
            },
            DomainCategory.GENERAL: {
                "how_to": [
                    "How to {action} for {niche}: Step-by-Step Guide",
                    "Complete Guide to {action} in {niche}",
                    "Mastering {action} for {niche} Success"
                ],
                "problem_solution": [
                    "How to Solve Common {problem} in {niche}",
                    "Fixing {problem} Issues for {niche}",
                    "Troubleshooting {problem} in {niche} Context"
                ],
                "comparison": [
                    "Best {item} for {niche}: Comparison Guide",
                    "{item} Comparison: Which is Right for Your {niche} Needs?",
                    "How to Choose {item} for {niche}: Pros and Cons"
                ],
                "beginner_guide": [
                    "Getting Started with {niche}: Beginner's Guide",
                    "{niche} Basics: Everything You Need to Know",
                    "First Steps in {niche}: Complete Introduction"
                ]
            }
        }
        
        # Domain-specific vocabulary
        self.domain_vocabulary = {
            DomainCategory.PHOTOGRAPHY: {
                "nouns": ["camera", "lens", "tripod", "flash", "filter", "lighting", "composition", "editing"],
                "techniques": ["long exposure", "macro photography", "portrait lighting", "landscape composition", "HDR processing"],
                "problems": ["blurry images", "poor lighting", "composition issues", "color balance", "focus problems"]
            },
            DomainCategory.ANIMALS: {
                "animal_types": ["mammals", "birds", "reptiles", "amphibians", "insects", "marine life", "wildlife"],
                "environments": ["forest", "desert", "ocean", "grassland", "wetland", "mountain", "urban"],
                "observations": ["behavior patterns", "feeding habits", "migration routes", "social interactions", "breeding cycles"]
            },
            DomainCategory.NATURE: {
                "ecosystem_types": ["forest", "wetland", "marine", "alpine", "desert", "grassland", "coastal"],
                "plant_types": ["trees", "shrubs", "flowers", "grasses", "ferns", "mosses", "fungi"],
                "observation_types": ["plant growth", "weather patterns", "soil conditions", "water quality", "animal signs"]
            },
            DomainCategory.OUTDOORS: {
                "gear_types": ["backpacks", "tents", "sleeping bags", "boots", "navigation tools", "cooking equipment"],
                "activities": ["hiking", "camping", "backpacking", "trail running", "rock climbing", "kayaking"],
                "skills": ["navigation", "shelter building", "fire starting", "water purification", "first aid"]
            }
        }
    
    def rewrite_title(self, original_title: str, niche: str, 
                     max_attempts: int = 3) -> Optional[TemplateRewrite]:
        """
        Rewrite a title using domain-appropriate templates with validation.
        
        Returns None if no valid rewrite can be generated after max_attempts.
        """
        analysis = self.niche_resolver.analyze(niche)
        domain = analysis.domain
        
        # Get appropriate templates for domain
        templates = self.domain_templates.get(domain, self.domain_templates[DomainCategory.GENERAL])
        vocabulary = self.domain_vocabulary.get(domain, {})
        
        for attempt in range(max_attempts):
            # Select template category based on original title intent
            intent_type = self._detect_title_intent(original_title)
            template_category = self._map_intent_to_template_category(intent_type)
            
            if template_category not in templates:
                template_category = random.choice(list(templates.keys()))
            
            # Select specific template
            template = random.choice(templates[template_category])
            
            # Fill template with domain-appropriate vocabulary
            rewritten = self._fill_template(template, niche, domain, vocabulary, original_title)
            
            # Validate the rewrite
            is_valid, validation_notes = self._validate_rewrite(
                original_title, rewritten, niche, domain
            )
            
            if is_valid:
                confidence = self._calculate_confidence(original_title, rewritten, domain)
                
                return TemplateRewrite(
                    original_title=original_title,
                    rewritten_title=rewritten,
                    template_id=template_category,
                    domain=domain,
                    intent_type=intent_type,
                    confidence_score=confidence,
                    validation_passed=True,
                    validation_notes=validation_notes
                )
        
        # If all attempts failed, return None
        return None
    
    def rewrite_with_constraints(self, original_title: str, niche: str,
                               required_keywords: List[str] = None,
                               banned_keywords: List[str] = None) -> Optional[TemplateRewrite]:
        """
        Rewrite title with specific constraints (e.g., must include certain keywords).
        """
        analysis = self.niche_resolver.analyze(niche)
        domain = analysis.domain
        
        # Get templates
        templates = self.domain_templates.get(domain, self.domain_templates[DomainCategory.GENERAL])
        vocabulary = self.domain_vocabulary.get(domain, {})
        
        # Try each template category
        for template_category in templates:
            for template in templates[template_category]:
                rewritten = self._fill_template(template, niche, domain, vocabulary, original_title)
                
                # Check constraints
                constraints_met = True
                validation_notes = []
                
                if required_keywords:
                    for keyword in required_keywords:
                        if keyword.lower() not in rewritten.lower():
                            constraints_met = False
                            validation_notes.append(f"Missing required keyword: {keyword}")
                
                if banned_keywords:
                    for keyword in banned_keywords:
                        if keyword.lower() in rewritten.lower():
                            constraints_met = False
                            validation_notes.append(f"Contains banned keyword: {keyword}")
                
                if constraints_met:
                    # Validate the rewrite
                    is_valid, val_notes = self._validate_rewrite(
                        original_title, rewritten, niche, domain
                    )
                    
                    if is_valid:
                        validation_notes.extend(val_notes)
                        intent_type = self._detect_title_intent(original_title)
                        confidence = self._calculate_confidence(original_title, rewritten, domain)
                        
                        return TemplateRewrite(
                            original_title=original_title,
                            rewritten_title=rewritten,
                            template_id=template_category,
                            domain=domain,
                            intent_type=intent_type,
                            confidence_score=confidence,
                            validation_passed=True,
                            validation_notes=validation_notes
                        )
        
        return None
    
    def _detect_title_intent(self, title: str) -> IntentType:
        """Detect intent type from title."""
        title_lower = title.lower()
        
        intent_patterns = {
            IntentType.HOW_TO: [r"how to", r"step.*by.*step", r"guide to", r"tutorial"],
            IntentType.COMPARISON: [r"vs\.?", r"versus", r"comparison", r"pros.*cons"],
            IntentType.PROBLEM_SOLUTION: [r"fix", r"solve", r"troubleshoot", r"problem"],
            IntentType.BUYING_GUIDE: [r"buy", r"purchase", r"choose", r"best.*for"],
            IntentType.TROUBLESHOOTING: [r"troubleshoot", r"diagnose", r"common.*problem"],
            IntentType.BEGINNER_GUIDE: [r"beginner", r"getting started", r"basics"],
            IntentType.ADVANCED_TECHNIQUE: [r"advanced", r"expert", r"master.*technique"],
            IntentType.REVIEW: [r"review", r"tested", r"evaluation", r"rating"]
        }
        
        for intent_type, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower, re.IGNORECASE):
                    return intent_type
        
        # Default to HOW_TO for actionable titles, otherwise BEGINNER_GUIDE
        action_verbs = {"how", "fix", "solve", "choose", "install", "build", "create"}
        title_words = set(re.findall(r'\w+', title_lower))
        if len(action_verbs & title_words) > 0:
            return IntentType.HOW_TO
        else:
            return IntentType.BEGINNER_GUIDE
    
    def _map_intent_to_template_category(self, intent_type: IntentType) -> str:
        """Map intent type to template category."""
        mapping = {
            IntentType.HOW_TO: "how_to",
            IntentType.COMPARISON: "comparison",
            IntentType.PROBLEM_SOLUTION: "problem_solution",
            IntentType.BUYING_GUIDE: "buying_guide",
            IntentType.TROUBLESHOOTING: "problem_solution",
            IntentType.BEGINNER_GUIDE: "beginner_guide",
            IntentType.ADVANCED_TECHNIQUE: "technique_tutorial",
            IntentType.REVIEW: "comparison"
        }
        
        return mapping.get(intent_type, "how_to")
    
    def _fill_template(self, template: str, niche: str, domain: DomainCategory,
                      vocabulary: Dict, original_title: str) -> str:
        """Fill template with appropriate vocabulary."""
        filled = template
        
        # Replace {niche}
        filled = filled.replace("{niche}", niche)
        
        # Replace domain-specific placeholders
        if "{noun}" in filled and "nouns" in vocabulary:
            filled = filled.replace("{noun}", random.choice(vocabulary["nouns"]))
        
        if "{technique}" in filled and "techniques" in vocabulary:
            filled = filled.replace("{technique}", random.choice(vocabulary["techniques"]))
        
        if "{problem}" in filled and "problems" in vocabulary:
            filled = filled.replace("{problem}", random.choice(vocabulary["problems"]))
        
        if "{animal_type}" in filled and "animal_types" in vocabulary:
            filled = filled.replace("{animal_type}", random.choice(vocabulary["animal_types"]))
        
        if "{environment}" in filled and "environments" in vocabulary:
            filled = filled.replace("{environment}", random.choice(vocabulary["environments"]))
        
        if "{ecosystem_type}" in filled and "ecosystem_types" in vocabulary:
            filled = filled.replace("{ecosystem_type}", random.choice(vocabulary["ecosystem_types"]))
        
        if "{plant_type}" in filled and "plant_types" in vocabulary:
            filled = filled.replace("{plant_type}", random.choice(vocabulary["plant_types"]))
        
        if "{gear_type}" in filled and "gear_types" in vocabulary:
            filled = filled.replace("{gear_type}", random.choice(vocabulary["gear_types"]))
        
        if "{activity}" in filled and "activities" in vocabulary:
            filled = filled.replace("{activity}", random.choice(vocabulary["activities"]))
        
        if "{skill}" in filled and "skills" in vocabulary:
            filled = filled.replace("{skill}", random.choice(vocabulary["skills"]))
        
        if "{observation_type}" in filled and "observation_types" in vocabulary:
            filled = filled.replace("{observation_type}", random.choice(vocabulary["observation_types"]))
        
        # Generic placeholders
        if "{action}" in filled:
            actions = ["get started", "improve results", "achieve success", "make progress"]
            filled = filled.replace("{action}", random.choice(actions))
        
        if "{item}" in filled:
            items = ["tools", "equipment", "resources", "solutions"]
            filled = filled.replace("{item}", random.choice(items))
        
        # Extract key terms from original title to preserve meaning
        original_words = set(re.findall(r'\w+', original_title.lower()))
        domain_keywords = set()
        for word_list in vocabulary.values():
            if isinstance(word_list, list):
                domain_keywords.update([w.lower() for w in word_list])
        
        shared_keywords = original_words & domain_keywords
        if shared_keywords:
            # Try to preserve at least one shared keyword
            keyword = random.choice(list(shared_keywords))
            if "{preserved_keyword}" in filled:
                filled = filled.replace("{preserved_keyword}", keyword.title())
        
        return filled
    
    def _validate_rewrite(self, original: str, rewritten: str, 
                         niche: str, domain: DomainCategory) -> Tuple[bool, List[str]]:
        """Validate that rewrite preserves meaning and stays in-domain."""
        validation_notes = []
        
        # 1. Check semantic similarity
        original_words = set(re.findall(r'\w+', original.lower()))
        rewritten_words = set(re.findall(r'\w+', rewritten.lower()))
        shared_keywords = original_words & rewritten_words
        
        if len(shared_keywords) < 2:  # At least 2 shared keywords
            validation_notes.append(f"Low semantic similarity: only {len(shared_keywords)} shared keywords")
        
        # 2. Check for domain drift (using simple keyword checks)
        domain_drift = self._check_domain_drift(rewritten.lower(), domain)
        if domain_drift:
            validation_notes.append(f"Domain drift detected: {domain_drift}")
        
        # 3. Check for nonsense patterns
        if self._contains_nonsense(rewritten.lower()):
            validation_notes.append("Contains nonsense/non-sequitur terms")
        
        # 4. Check for niche term inclusion
        niche_words = set(niche.lower().split())
        has_niche_term = any(word in rewritten.lower() for word in niche_words)
        if not has_niche_term:
            validation_notes.append(f"Missing niche term: {niche}")
        
        # Determine if rewrite is valid
        is_valid = len(validation_notes) == 0
        
        return is_valid, validation_notes
    
    def _check_domain_drift(self, text_lower: str, domain: DomainCategory) -> Optional[str]:
        """Check for domain-inappropriate terms."""
        # Domain drift patterns (simplified)
        drift_patterns = {
            DomainCategory.PHOTOGRAPHY: [
                (r"\b(finance|budget|loan|investment)\b", "finance terms"),
                (r"\b(medical|health|treatment|therapy)\b", "medical terms"),
                (r"\b(legal|law|attorney|court)\b", "legal terms")
            ],
            DomainCategory.ANIMALS: [
                (r"\b(software|code|programming|app)\b", "software terms"),
                (r"\b(finance|budget|investment|money)\b", "finance terms"),
                (r"\b(real.*estate|property|mortgage)\b", "real estate terms")
            ],
            DomainCategory.NATURE: [
                (r"\b(software|code|programming)\b", "software terms"),
                (r"\b(finance|budget|investment)\b", "finance terms"),
                (r"\b(urban|city|metropolitan)\b", "urban terms")
            ]
        }
        
        if domain in drift_patterns:
            for pattern, description in drift_patterns[domain]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return description
        
        return None
    
    def _contains_nonsense(self, text_lower: str) -> bool:
        """Check for nonsense/non-sequitur patterns."""
        nonsense_patterns = [
            r"\b(whale|horse|platypus|elephant)\b.*\b(measure|invest|monitor|cost)\b",
            r"\b(first.*aid.*kit)\b.*\b(upgrade)\b",
            r"\b(tackle.*box)\b.*\b(configure)\b",
            r"\b(brush)\b.*\b(install)\b",
            r"\b(random.*noun)\b.*\b(in.*2024)\b"
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_confidence(self, original: str, rewritten: str, domain: DomainCategory) -> float:
        """Calculate confidence score for rewrite (0-1)."""
        score = 0.5  # Base score
        
        # Semantic similarity bonus
        original_words = set(re.findall(r'\w+', original.lower()))
        rewritten_words = set(re.findall(r'\w+', rewritten.lower()))
        shared_keywords = original_words & rewritten_words
        
        similarity_ratio = len(shared_keywords) / max(len(original_words), 1)
        score += similarity_ratio * 0.3
        
        # Domain alignment bonus
        domain_drift = self._check_domain_drift(rewritten.lower(), domain)
        if not domain_drift:
            score += 0.2
        
        # Nonsense penalty
        if self._contains_nonsense(rewritten.lower()):
            score -= 0.3
        
        return max(0, min(1, score))
    
    @classmethod
    def for_domain(cls, domain: DomainCategory) -> 'DomainTemplatePack':
        """
        Create a DomainTemplatePack instance for a specific domain.
        
        This is a convenience method for tests and external callers.
        """
        instance = cls()
        # Optionally pre-configure for domain if needed
        # Currently just returns a standard instance
        return instance


# Convenience functions
def rewrite_title_safely(original_title: str, niche: str,
                        max_attempts: int = 3) -> Optional[str]:
    """Safe title rewriting with domain-aware templates."""
    pack = DomainTemplatePack()
    result = pack.rewrite_title(original_title, niche, max_attempts)
    
    if result and result.validation_passed and result.confidence_score >= 0.7:
        return result.rewritten_title
    
    return None


if __name__ == "__main__":
    # Test the template packs
    print("=== Testing Domain Template Packs ===")
    
    pack = DomainTemplatePack()
    
    test_cases = [
        ("What Defines the Basic Characteristics of Animal Life?", "animals"),
        ("How Do Different Animal Classifications Help Us Understand the Natural World?", "animals"),
        ("What Are the Core Components of Animal Anatomy?", "animals"),
        ("Foundational Knowledge for Nature Photography", "nature photography"),
        ("Understanding Light in Outdoor Settings", "nature photography"),
        ("Nature Composition Techniques", "nature photography")
    ]
    
    for original, niche in test_cases:
        print(f"\nOriginal: '{original}'")
        print(f"Niche: {niche}")
        
        result = pack.rewrite_title(original, niche, max_attempts=3)
        
        if result:
            print(f"  Rewritten: '{result.rewritten_title}'")
            print(f"  Template: {result.template_id}")
            print(f"  Domain: {result.domain.value}")
            print(f"  Intent: {result.intent_type.value}")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Valid: {result.validation_passed}")
            if result.validation_notes:
                print(f"  Notes: {result.validation_notes}")
        else:
            print("  [FAILED] No valid rewrite found")