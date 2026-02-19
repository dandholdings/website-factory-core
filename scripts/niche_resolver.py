#!/usr/bin/env python3
"""
niche_resolver.py — Niche Breadth Resolver + Domain Classifier for website factory.

Analyzes niche strings to:
1. Detect if niche is broad (single word / umbrella term)
2. Classify into domain categories (photography, outdoors, DIY, health, finance, software, etc.)
3. Suggest 1-2 lenses (activity/audience/environment/goal) to narrow intent
4. Provide domain-specific guidance for hub generation and title rewriting
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class DomainCategory(Enum):
    """Domain categories for niche classification."""
    PHOTOGRAPHY = "photography"
    OUTDOORS = "outdoors"
    DIY = "diy"
    HEALTH = "health"
    FINANCE = "finance"
    SOFTWARE = "software"
    EDUCATION = "education"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    HOME = "home"
    TRAVEL = "travel"
    FOOD = "food"
    FITNESS = "fitness"
    SCIENCE = "science"
    ARTS = "arts"
    ANIMALS = "animals"
    NATURE = "nature"
    GENERAL = "general"


class NicheLens(Enum):
    """Lenses to narrow broad niches."""
    ACTIVITY = "activity"      # What people do (photograph, hike, cook, code)
    AUDIENCE = "audience"      # Who it's for (beginners, professionals, families)
    ENVIRONMENT = "environment" # Where it happens (indoors, outdoors, urban, rural)
    GOAL = "goal"              # What they want (learn, improve, solve, create)
    TOOL = "tool"              # What they use (camera, software, equipment)
    PROBLEM = "problem"        # What they're solving (budget, time, quality)


@dataclass
class NicheAnalysis:
    """Complete analysis of a niche string."""
    niche: str
    is_broad: bool
    domain: DomainCategory
    suggested_lenses: List[NicheLens]
    domain_keywords: Set[str]
    recommended_hub_themes: List[str]
    title_constraints: Dict[str, List[str]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "niche": self.niche,
            "is_broad": self.is_broad,
            "domain": self.domain.value,
            "suggested_lenses": [lens.value for lens in self.suggested_lenses],
            "domain_keywords": list(self.domain_keywords),
            "recommended_hub_themes": self.recommended_hub_themes,
            "title_constraints": self.title_constraints
        }


class NicheBreadthResolver:
    """Deterministic niche breadth analysis and domain classification."""
    
    # Broad/umbrella terms that need narrowing
    BROAD_TERMS = {
        "animals", "nature", "art", "music", "science", "history", "technology",
        "business", "health", "fitness", "food", "travel", "education", "finance",
        "home", "garden", "sports", "entertainment", "culture", "environment"
    }
    
    # Domain classification patterns
    DOMAIN_PATTERNS = {
        DomainCategory.PHOTOGRAPHY: [
            r"photography", r"camera", r"lens", r"photo", r"shoot", r"image",
            r"digital.*photo", r"nature.*photo", r"portrait", r"landscape"
        ],
        DomainCategory.OUTDOORS: [
            r"outdoor", r"hiking", r"camping", r"nature", r"wilderness", r"trail",
            r"mountain", r"forest", r"park", r"adventure", r"exploration"
        ],
        DomainCategory.DIY: [
            r"diy", r"do.*it.*yourself", r"home.*improvement", r"craft", r"build",
            r"make", r"create", r"woodwork", r"repair", r"renovation"
        ],
        DomainCategory.HEALTH: [
            r"health", r"medical", r"wellness", r"fitness", r"nutrition", r"diet",
            r"exercise", r"therapy", r"treatment", r"medicine", r"care"
        ],
        DomainCategory.FINANCE: [
            r"finance", r"money", r"investment", r"budget", r"saving", r"debt",
            r"credit", r"loan", r"bank", r"financial", r"economy"
        ],
        DomainCategory.SOFTWARE: [
            r"software", r"programming", r"coding", r"app", r"application", r"web",
            r"mobile", r"development", r"code", r"computer", r"tech"
        ],
        DomainCategory.EDUCATION: [
            r"education", r"learning", r"teaching", r"school", r"course", r"study",
            r"tutorial", r"guide", r"instruction", r"training"
        ],
        DomainCategory.BUSINESS: [
            r"business", r"marketing", r"sales", r"management", r"strategy",
            r"entrepreneur", r"startup", r"company", r"corporate"
        ],
        DomainCategory.TECHNOLOGY: [
            r"technology", r"tech", r"digital", r"electronic", r"gadget",
            r"device", r"innovation", r"ai", r"machine", r"robot"
        ],
        DomainCategory.HOME: [
            r"home", r"house", r"residential", r"living", r"family", r"domestic",
            r"household", r"interior", r"decor", r"furniture"
        ],
        DomainCategory.TRAVEL: [
            r"travel", r"tourism", r"vacation", r"trip", r"journey", r"destination",
            r"adventure", r"explore", r"backpack", r"tour"
        ],
        DomainCategory.FOOD: [
            r"food", r"cooking", r"recipe", r"cuisine", r"meal", r"dining",
            r"restaurant", r"chef", r"baking", r"kitchen"
        ],
        DomainCategory.FITNESS: [
            r"fitness", r"exercise", r"workout", r"gym", r"training", r"sport",
            r"athletic", r"physical", r"strength", r"cardio"
        ],
        DomainCategory.SCIENCE: [
            r"science", r"research", r"study", r"experiment", r"laboratory",
            r"physics", r"chemistry", r"biology", r"geology", r"astronomy"
        ],
        DomainCategory.ARTS: [
            r"art", r"painting", r"drawing", r"sculpture", r"design", r"creative",
            r"craft", r"artist", r"visual", r"graphic"
        ],
        DomainCategory.ANIMALS: [
            r"animal", r"pet", r"wildlife", r"species", r"mammal", r"bird",
            r"reptile", r"insect", r"fauna", r"creature", r"zoo", r"veterinary"
        ],
        DomainCategory.NATURE: [
            r"nature", r"environment", r"ecology", r"ecosystem", r"wilderness",
            r"conservation", r"sustainability", r"natural", r"outdoor", r"wild"
        ]
    }
    
    # Domain-specific hub themes (to replace generic "fundamentals/techniques/etc.")
    DOMAIN_HUB_THEMES = {
        DomainCategory.PHOTOGRAPHY: [
            "Camera Equipment", "Lighting Techniques", "Composition Rules",
            "Editing Software", "Genre Specialties", "Business of Photography",
            "Workflow Optimization", "Client Management"
        ],
        DomainCategory.OUTDOORS: [
            "Gear & Equipment", "Navigation Skills", "Safety Protocols",
            "Environmental Ethics", "Trip Planning", "Survival Skills",
            "Wildlife Knowledge", "Weather Preparedness"
        ],
        DomainCategory.ANIMALS: [
            "Species Identification", "Habitat Management", "Behavior Studies",
            "Conservation Efforts", "Animal Care", "Research Methods",
            "Ethical Guidelines", "Field Techniques"
        ],
        DomainCategory.NATURE: [
            "Ecosystem Basics", "Plant Identification", "Geological Features",
            "Conservation Practices", "Observation Techniques", "Field Guides",
            "Environmental Impact", "Sustainable Practices"
        ],
        DomainCategory.GENERAL: [
            "Core Concepts", "Practical Techniques", "Strategic Approaches",
            "Essential Tools", "Key Benefits", "Common Challenges",
            "Learning Resources", "Expert Community"
        ]
    }
    
    # Domain-specific title constraints
    DOMAIN_TITLE_CONSTRAINTS = {
        DomainCategory.PHOTOGRAPHY: {
            "required_keywords": ["camera", "lens", "photo", "shoot", "edit"],
            "banned_keywords": ["finance", "budget", "medical", "legal"],
            "preferred_verbs": ["shoot", "edit", "compose", "light", "process"]
        },
        DomainCategory.OUTDOORS: {
            "required_keywords": ["gear", "trail", "safety", "navigation", "wilderness"],
            "banned_keywords": ["software", "digital", "office", "indoor"],
            "preferred_verbs": ["hike", "camp", "navigate", "prepare", "survive"]
        },
        DomainCategory.ANIMALS: {
            "required_keywords": ["species", "habitat", "behavior", "conservation", "wildlife"],
            "banned_keywords": ["software", "finance", "real estate"],
            "preferred_verbs": ["identify", "observe", "protect", "study", "care"]
        },
        DomainCategory.NATURE: {
            "required_keywords": ["ecosystem", "environment", "conservation", "wildlife", "sustainability"],
            "banned_keywords": ["technology", "finance", "urban"],
            "preferred_verbs": ["explore", "conserve", "identify", "protect", "study"]
        }
    }
    
    def __init__(self):
        self.compiled_patterns = {
            domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }
    
    def analyze(self, niche: str) -> NicheAnalysis:
        """Analyze a niche string and return comprehensive analysis."""
        niche_lower = niche.lower().strip()
        
        # 1. Determine if niche is broad
        is_broad = self._is_broad_niche(niche_lower)
        
        # 2. Classify domain
        domain = self._classify_domain(niche_lower)
        
        # 3. Suggest lenses for narrowing
        suggested_lenses = self._suggest_lenses(niche_lower, is_broad, domain)
        
        # 4. Get domain-specific keywords
        domain_keywords = self._get_domain_keywords(domain)
        
        # 5. Get recommended hub themes
        recommended_hub_themes = self._get_hub_themes(domain, niche_lower)
        
        # 6. Get title constraints
        title_constraints = self._get_title_constraints(domain)
        
        return NicheAnalysis(
            niche=niche,
            is_broad=is_broad,
            domain=domain,
            suggested_lenses=suggested_lenses,
            domain_keywords=domain_keywords,
            recommended_hub_themes=recommended_hub_themes,
            title_constraints=title_constraints
        )
    
    def _is_broad_niche(self, niche_lower: str) -> bool:
        """Determine if niche is broad/umbrella term."""
        # Single word niches are often broad
        words = niche_lower.split()
        if len(words) == 1:
            return words[0] in self.BROAD_TERMS
        
        # Check if any broad term appears
        for term in self.BROAD_TERMS:
            if term in niche_lower:
                return True
        
        return False
    
    def _classify_domain(self, niche_lower: str) -> DomainCategory:
        """Classify niche into domain category."""
        domain_scores = {}
        
        for domain, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(niche_lower):
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Return domain with highest score
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # Default to general domain
        return DomainCategory.GENERAL
    
    def _suggest_lenses(self, niche_lower: str, is_broad: bool, domain: DomainCategory) -> List[NicheLens]:
        """Suggest 1-2 lenses to narrow broad niches."""
        lenses = []
        
        if not is_broad:
            # For specific niches, suggest complementary lenses
            if domain in [DomainCategory.PHOTOGRAPHY, DomainCategory.DIY, DomainCategory.SOFTWARE]:
                lenses.extend([NicheLens.TOOL, NicheLens.ACTIVITY])
            elif domain in [DomainCategory.HEALTH, DomainCategory.FITNESS, DomainCategory.EDUCATION]:
                lenses.extend([NicheLens.AUDIENCE, NicheLens.GOAL])
            elif domain in [DomainCategory.OUTDOORS, DomainCategory.TRAVEL]:
                lenses.extend([NicheLens.ENVIRONMENT, NicheLens.ACTIVITY])
        else:
            # For broad niches, suggest narrowing lenses
            lenses.extend([NicheLens.ACTIVITY, NicheLens.AUDIENCE])
            
            # Add domain-specific lenses
            if domain == DomainCategory.PHOTOGRAPHY:
                lenses.append(NicheLens.TOOL)
            elif domain == DomainCategory.HEALTH:
                lenses.append(NicheLens.PROBLEM)
            elif domain == DomainCategory.FINANCE:
                lenses.append(NicheLens.GOAL)
        
        # Return unique lenses, limit to 2
        unique_lenses = []
        for lens in lenses:
            if lens not in unique_lenses:
                unique_lenses.append(lens)
        
        return unique_lenses[:2]
    
    def _get_domain_keywords(self, domain: DomainCategory) -> Set[str]:
        """Get domain-specific keywords for noun bank generation."""
        # Map domains to keyword sets
        domain_keyword_map = {
            DomainCategory.PHOTOGRAPHY: {
                "camera", "lens", "tripod", "flash", "aperture", "shutter",
                "iso", "composition", "lighting", "editing", "software"
            },
            DomainCategory.OUTDOORS: {
                "gear", "backpack", "tent", "sleeping bag", "boots", "compass",
                "map", "first aid", "water filter", "stove", "headlamp"
            },
            DomainCategory.ANIMALS: {
                "species", "habitat", "behavior", "conservation", "wildlife",
                "observation", "research", "protection", "ecosystem", "biodiversity"
            },
            DomainCategory.NATURE: {
                "ecosystem", "environment", "conservation", "sustainability",
                "wildlife", "plants", "geology", "climate", "preservation"
            },
            DomainCategory.GENERAL: {
                "guide", "manual", "tutorial", "checklist", "tool", "system",
                "process", "method", "technique", "solution", "resource"
            }
        }
        
        return domain_keyword_map.get(domain, domain_keyword_map[DomainCategory.GENERAL])
    
    def _get_hub_themes(self, domain: DomainCategory, niche_lower: str) -> List[str]:
        """Get domain-specific hub themes to replace generic framework."""
        # Check for special cases
        if "animal" in niche_lower:
            return self.DOMAIN_HUB_THEMES.get(DomainCategory.ANIMALS, [])
        elif "nature" in niche_lower and "photo" not in niche_lower:
            return self.DOMAIN_HUB_THEMES.get(DomainCategory.NATURE, [])
        
        # Return domain-specific themes or general fallback
        return self.DOMAIN_HUB_THEMES.get(domain, self.DOMAIN_HUB_THEMES[DomainCategory.GENERAL])
    
    def _get_title_constraints(self, domain: DomainCategory) -> Dict[str, List[str]]:
        """Get domain-specific title constraints for rewriting."""
        return self.DOMAIN_TITLE_CONSTRAINTS.get(domain, {
            "required_keywords": [],
            "banned_keywords": [],
            "preferred_verbs": []
        })


# Convenience functions
def analyze_niche(niche: str) -> Dict:
    """Convenience function to analyze niche and return dict."""
    resolver = NicheBreadthResolver()
    analysis = resolver.analyze(niche)
    return analysis.to_dict()


def get_niche_analysis(niche: str) -> NicheAnalysis:
    """Convenience function to get NicheAnalysis object."""
    resolver = NicheBreadthResolver()
    return resolver.analyze(niche)


if __name__ == "__main__":
    # Test the resolver
    test_niches = ["animals", "nature", "nature photography", "home gardening", "personal finance"]
    
    resolver = NicheBreadthResolver()
    for niche in test_niches:
        print(f"\n=== Analyzing: '{niche}' ===")
        analysis = resolver.analyze(niche)
        print(f"  Is broad: {analysis.is_broad}")
        print(f"  Domain: {analysis.domain.value}")
        print(f"  Suggested lenses: {[lens.value for lens in analysis.suggested_lenses]}")
        print(f"  Hub themes: {analysis.recommended_hub_themes[:4]}...")
        print(f"  Title constraints: {analysis.title_constraints}")