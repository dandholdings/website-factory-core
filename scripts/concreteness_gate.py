#!/usr/bin/env python3
"""
ConcretenessGate validator and auto-rewriter.

Validates titles for concrete, high-intent content and automatically
rewrites vague titles using LLM with fallback templates.
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import llm_json
from concrete_intent_validator import ConcreteIntentValidator, NounBankGenerator

logger = logging.getLogger(__name__)


@dataclass
class ConcretenessGateResult:
    """Result of concreteness validation and rewriting."""
    title: str
    is_concrete: bool
    score: float
    issues: List[str]
    rewritten_title: Optional[str] = None
    rewrite_reason: Optional[str] = None
    attempts: int = 0
    intent_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "is_concrete": self.is_concrete,
            "score": self.score,
            "issues": self.issues,
            "rewritten_title": self.rewritten_title,
            "rewrite_reason": self.rewrite_reason,
            "attempts": self.attempts,
            "intent_type": self.intent_type
        }


class ConcretenessGate:
    """Validates and rewrites titles for concrete, high-intent content."""
    
    def __init__(self, site_root: Optional[Path] = None):
        self.site_root = site_root or Path.cwd()
        self.validator = ConcreteIntentValidator()
        self.noun_bank_generator = NounBankGenerator(self.site_root)
        
        # Vague patterns that trigger auto-rewrite
        self.vague_patterns = [
            r'\b(?:complete|ultimate|comprehensive|definitive|essential)\s+(?:guide|handbook|manual|resource)\b',
            r'\b(?:everything|all)\s+(?:you|one)\s+(?:need|should)\s+(?:to\s+)?know\b',
            r'\b(?:beginners|starters|newbies)\s+(?:guide|introduction|tutorial)\b',
            r'\b(?:master|learn|understand)\s+(?:the\s+)?(?:art|science|basics|fundamentals)\b',
            r'\b(?:step-by-step|how-to)\s+(?:tutorial|guide)\b',
            r'\b(?:tips|tricks|hacks|secrets)\s+(?:for|to|of)\b',
            r'\b(?:improve|boost|enhance|maximize|optimize)\s+(?:your|the)\b',
            r'\b(?:best|top|greatest|awesome|amazing)\s+(?:ways|methods|techniques|practices)\b',
            r'\b(?:ultimate|complete)\s+list\b',
            r'\b(?:what|why|how)\s+(?:is|are|does|do|can)\b',
        ]
        
        # Generic category labels that should be avoided
        self.generic_category_labels = [
            "guide", "tutorial", "resource", "overview", "introduction",
            "handbook", "manual", "reference", "primer", "basics",
            "fundamentals", "essentials", "101", "crash course", "quick start",
            "getting started", "beginner's guide", "starter kit", "walkthrough",
            "explanation", "description", "summary", "review", "analysis"
        ]
        
        # Intent types for mix validation
        self.intent_types = ["howto", "troubleshoot", "cost_decision", "checklist_safety", "local_seasonal"]
        
        # Intent mix quotas (minimum percentage of each type)
        self.intent_mix_quotas = {
            "howto": 0.25,          # At least 25% how-to content
            "troubleshoot": 0.20,   # At least 20% troubleshooting
            "cost_decision": 0.15,  # At least 15% cost/decision
            "checklist_safety": 0.15, # At least 15% checklist/safety
            "local_seasonal": 0.10   # At least 10% local/seasonal
        }
        
        # High-intent templates for fallback rewriting - improved to be more relevant
        self.high_intent_templates = [
            # How-to templates (most common intent)
            "How to {action} {noun}: Step-by-Step Guide",
            "How to Choose the Right {noun} for {context}",
            "How to Install {noun}: Complete Installation Guide",
            "How to Maintain {noun} for Optimal Performance",
            "How to Troubleshoot {noun}: Fix Common Problems",
            
            # Problem-solving templates
            "{noun} Not Working? Here's How to Fix It",
            "Common {noun} Problems and Solutions",
            "How to Diagnose {noun} Issues",
            
            # Cost/decision templates
            "{noun} Cost Analysis: Budget & ROI Calculator",
            "How Much Does {noun} Cost? Complete Pricing Guide",
            "{noun} vs {alternative}: Which is Better for You?",
            
            # Safety/checklist templates
            "{noun} Safety Checklist: Essential Precautions",
            "How to Use {noun} Safely: Best Practices",
            "{noun} Maintenance Checklist: Keep It Running Smoothly",
            
            # Local/seasonal templates
            "Best {noun} for {context} in Your Region",
            "Seasonal {noun} Guide: What to Use When",
            "{noun} for Different Climates and Conditions",
            
            # Benefit-focused templates
            "How to {action} {noun} and Save Money",
            "{noun} Efficiency Tips: Maximize Performance",
            "How to Optimize {noun} for Better Results",
            
            # Selection/decision templates
            "How to Select the Best {noun} for Your Needs",
            "{noun} Buying Guide: What to Look For",
            "Top {number} {noun} Options Compared",
        ]
    
    def validate_title(self, title: str, niche: str, hub: str = "", cluster: str = "") -> ConcretenessGateResult:
        """
        Validate a title for concreteness and high-intent.
        
        Args:
            title: The title to validate
            niche: The niche context
            hub: Optional hub context
            cluster: Optional cluster context
            
        Returns:
            ConcretenessGateResult with validation details
        """
        logger.debug(f"Validating title: '{title}' for niche: {niche}")
        
        # Get noun bank for context
        noun_bank = self.noun_bank_generator.get_noun_bank(niche, hub, cluster)
        
        # Validate with CIV
        is_valid, issues = self.validator.validate_title(title, noun_bank)
        score_result = self.validator.score_title(title, noun_bank)
        score = score_result.get("score", 0.0)
        
        # Check for vague patterns
        vague_detected, vague_phrase = self._detect_vague_patterns(title)
        if vague_detected:
            issues.append(f"Contains vague phrase: '{vague_phrase}'")
            is_valid = False
        
        # Check for generic category labels
        generic_detected, generic_label = self._detect_generic_category_label(title)
        if generic_detected:
            issues.append(f"Contains generic category label: '{generic_label}'")
            is_valid = False
        
        # Detect intent type
        intent_type = self._detect_intent_type(title)
        if intent_type:
            issues.append(f"Intent type detected: {intent_type}")
        
        # Check title length
        if len(title) < 15:
            issues.append("Title too short (less than 15 characters)")
            is_valid = False
        elif len(title) > 80:
            issues.append("Title too long (more than 80 characters)")
            # Not necessarily invalid, just a warning
        
        result = ConcretenessGateResult(
            title=title,
            is_concrete=is_valid,
            score=score,
            issues=issues
        )
        
        # Add intent type to result metadata
        result.intent_type = intent_type
        
        return result
    
    def validate_and_rewrite(self, title: str, niche: str, hub: str = "", cluster: str = "", 
                           max_attempts: int = 3) -> ConcretenessGateResult:
        """
        Validate title and automatically rewrite if not concrete.
        
        Args:
            title: The title to validate and potentially rewrite
            niche: The niche context
            hub: Optional hub context
            cluster: Optional cluster context
            max_attempts: Maximum rewrite attempts
            
        Returns:
            ConcretenessGateResult with rewritten title if needed
        """
        # Initial validation
        result = self.validate_title(title, niche, hub, cluster)
        
        if result.is_concrete:
            logger.debug(f"Title '{title}' is already concrete (score: {result.score:.2f})")
            return result
        
        logger.info(f"Title '{title}' needs rewriting. Issues: {result.issues}")
        
        # Try to rewrite with LLM
        rewritten_title = None
        rewrite_reason = None
        attempts = 0
        
        for attempt in range(max_attempts):
            attempts = attempt + 1
            try:
                rewritten_title = self._rewrite_with_llm(title, niche, hub, cluster, attempt)
                if rewritten_title and rewritten_title != title:
                    # Validate the rewritten title
                    rewritten_result = self.validate_title(rewritten_title, niche, hub, cluster)
                    if rewritten_result.is_concrete:
                        rewrite_reason = f"LLM rewrite (attempt {attempts})"
                        break
                    else:
                        logger.debug(f"Rewrite attempt {attempts} still not concrete. Score: {rewritten_result.score:.2f}")
                        # Continue to next attempt
                else:
                    logger.debug(f"LLM returned same title or None on attempt {attempts}")
            
            except Exception as e:
                logger.warning(f"LLM rewrite failed on attempt {attempts}: {e}")
        
        # If LLM failed or still not concrete, use template-based fallback
        if not rewritten_title or not self._is_title_improved(rewritten_title, result.score, niche, hub, cluster):
            logger.info("Using template-based fallback rewrite")
            rewritten_title = self._rewrite_with_template(title, niche, hub, cluster)
            rewrite_reason = "Template fallback"
            attempts = max_attempts + 1  # Indicates fallback was used
        
        if rewritten_title and rewritten_title != title:
            # Final validation of rewritten title
            final_result = self.validate_title(rewritten_title, niche, hub, cluster)
            
            return ConcretenessGateResult(
                title=title,
                is_concrete=final_result.is_concrete,
                score=final_result.score,
                issues=result.issues,
                rewritten_title=rewritten_title,
                rewrite_reason=rewrite_reason,
                attempts=attempts
            )
        
        # Could not rewrite successfully
        return result
    
    def _detect_vague_patterns(self, title: str) -> Tuple[bool, str]:
        """Detect vague patterns in title."""
        title_lower = title.lower()
        
        for pattern in self.vague_patterns:
            match = re.search(pattern, title_lower)
            if match:
                return True, match.group(0)
        
        return False, ""
    
    def _detect_generic_category_label(self, title: str) -> Tuple[bool, str]:
        """Detect generic category labels in title."""
        title_lower = title.lower()
        
        for label in self.generic_category_labels:
            # Check for exact word match (with word boundaries)
            pattern = r'\b' + re.escape(label) + r'\b'
            match = re.search(pattern, title_lower)
            if match:
                return True, label
        
        return False, ""
    
    def _detect_intent_type(self, title: str) -> Optional[str]:
        """Detect intent type from title patterns."""
        title_lower = title.lower()
        
        # How-to intent patterns
        howto_patterns = [
            r'how to\b',
            r'step-by-step',
            r'step by step',
            r'tutorial',
            r'guide to',
            r'learn how',
            r'get started with'
        ]
        
        # Troubleshoot intent patterns
        troubleshoot_patterns = [
            r'troubleshoot',
            r'fix\b',
            r'repair\b',
            r'debug\b',
            r'solve\b',
            r'problem',
            r'issue',
            r'error',
            r'won\'t work',
            r'not working'
        ]
        
        # Cost decision intent patterns
        cost_patterns = [
            r'cost\b',
            r'price\b',
            r'budget',
            r'affordable',
            r'expensive',
            r'cheap',
            r'worth it',
            r'roi',
            r'return on investment',
            r'how much'
        ]
        
        # Checklist/safety intent patterns
        checklist_patterns = [
            r'checklist',
            r'safety',
            r'precautions',
            r'risks',
            r'dangers',
            r'warnings',
            r'mistakes to avoid',
            r'common mistakes',
            r'do\'s and don\'ts'
        ]
        
        # Local/seasonal intent patterns
        local_patterns = [
            r'local\b',
            r'seasonal',
            r'regional',
            r'climate',
            r'weather',
            r'temperature',
            r'humidity',
            r'zone',
            r'area specific',
            r'location'
        ]
        
        # Check patterns in order of specificity
        for pattern in howto_patterns:
            if re.search(pattern, title_lower):
                return "howto"
        
        for pattern in troubleshoot_patterns:
            if re.search(pattern, title_lower):
                return "troubleshoot"
        
        for pattern in cost_patterns:
            if re.search(pattern, title_lower):
                return "cost_decision"
        
        for pattern in checklist_patterns:
            if re.search(pattern, title_lower):
                return "checklist_safety"
        
        for pattern in local_patterns:
            if re.search(pattern, title_lower):
                return "local_seasonal"
        
        return None
    
    def validate_intent_mix(self, titles: List[str]) -> Dict[str, Any]:
        """
        Validate that a list of titles meets intent mix quotas.
        
        Args:
            titles: List of titles to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        if not titles:
            return {"valid": True, "message": "No titles to validate"}
        
        # Count intent types
        intent_counts = {intent_type: 0 for intent_type in self.intent_types}
        total_titles = len(titles)
        
        for title in titles:
            intent_type = self._detect_intent_type(title)
            if intent_type and intent_type in intent_counts:
                intent_counts[intent_type] += 1
        
        # Calculate percentages
        intent_percentages = {}
        for intent_type, count in intent_counts.items():
            intent_percentages[intent_type] = count / total_titles if total_titles > 0 else 0
        
        # Check quotas
        quota_violations = []
        recommendations = []
        
        for intent_type, min_percentage in self.intent_mix_quotas.items():
            actual_percentage = intent_percentages.get(intent_type, 0)
            if actual_percentage < min_percentage:
                quota_violations.append(intent_type)
                needed_count = int(min_percentage * total_titles) - intent_counts[intent_type]
                if needed_count > 0:
                    recommendations.append(
                        f"Add {needed_count} more {intent_type.replace('_', ' ')} titles"
                    )
        
        valid = len(quota_violations) == 0
        
        return {
            "valid": valid,
            "total_titles": total_titles,
            "intent_counts": intent_counts,
            "intent_percentages": intent_percentages,
            "quota_violations": quota_violations,
            "recommendations": recommendations,
            "quotas": self.intent_mix_quotas
        }
    
    def _rewrite_with_llm(self, title: str, niche: str, hub: str = "", cluster: str = "", attempt: int = 0) -> Optional[str]:
        """Rewrite title using LLM to make it more concrete and high-intent."""
        system_prompt = """You are a content title expert specializing in concrete, high-intent titles for SEO and user engagement.

CRITICAL REQUIREMENTS:
1. Output MUST be a single JSON object with "rewritten_title" field
2. NO markdown, NO code fences, NO explanatory text
3. The rewritten title MUST be:
   - Concrete (specific nouns, actionable verbs)
   - High-intent (solves a problem, answers a question)
   - 15-80 characters
   - No vague phrases like "complete guide", "everything you need to know"
   - Include specific benefits or outcomes

EXAMPLE INPUT/OUTPUT:
Input: "Complete guide to home insulation"
Output: {"rewritten_title": "How to Install Home Insulation: Save 30% on Energy Bills"}

Input: "Everything about solar panels"
Output: {"rewritten_title": "Solar Panel Installation: Cost, Benefits, and ROI Calculator"}

Input: "Beginners tutorial for gardening"
Output: {"rewritten_title": "How to Start a Vegetable Garden: Step-by-Step for First-Timers"}"""

        context_info = f"Niche: {niche}"
        if hub:
            context_info += f", Hub: {hub}"
        if cluster:
            context_info += f", Cluster: {cluster}"
        
        user_prompt = f"""Rewrite this vague title to be concrete and high-intent:

Original title: "{title}"
Context: {context_info}

IMPORTANT:
- Keep the core topic but make it specific and actionable
- Include concrete nouns and action verbs
- Add benefit or outcome if possible
- Avoid generic phrases
- Target length: 15-80 characters

Return ONLY the JSON object with "rewritten_title" field."""

        # Adjust temperature based on attempt
        temperature = 0.7 if attempt == 0 else 0.9
        
        try:
            response = llm_json(
                system=system_prompt,
                user=user_prompt,
                temperature=temperature,
                max_tokens=500
            )
            
            if isinstance(response, dict) and "rewritten_title" in response:
                rewritten = response["rewritten_title"].strip()
                if rewritten and len(rewritten) > 10:
                    return rewritten
            
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")
        
        return None
    
    def _rewrite_with_template(self, title: str, niche: str, hub: str = "", cluster: str = "") -> str:
        """Rewrite title using template-based fallback."""
        # Extract nouns from title
        words = title.lower().split()
        nouns = [w for w in words if len(w) > 3 and w not in self.validator.stopwords]
        
        # Get noun bank for better nouns
        noun_bank = self.noun_bank_generator.get_noun_bank(niche, hub, cluster)
        if noun_bank and len(noun_bank) > 0:
            # Use nouns from noun bank that appear in title or are related
            bank_nouns = list(noun_bank)
            if bank_nouns:
                # Add some bank nouns to our list
                nouns.extend(bank_nouns[:3])
        
        # Deduplicate and clean
        nouns = list(dict.fromkeys(nouns))[:5]  # Keep first 5 unique
        
        # Action verbs
        action_verbs = ["install", "choose", "fix", "build", "optimize", "maintain", 
                       "troubleshoot", "select", "compare", "calculate", "save", "reduce"]
        
        # Benefits/contexts
        benefits = ["save money", "improve performance", "increase efficiency", 
                   "reduce costs", "enhance safety", "boost productivity", 
                   "maximize results", "solve problems", "avoid mistakes"]
        
        # Determine intent type from title content
        intent_type = self._detect_intent_type(title)
        if not intent_type:
            # Fallback based on title keywords
            title_lower = title.lower()
            if any(word in title_lower for word in ["how to", "guide", "tutorial", "step"]):
                intent_type = "howto"
            elif any(word in title_lower for word in ["fix", "problem", "issue", "troubleshoot"]):
                intent_type = "troubleshoot"
            elif any(word in title_lower for word in ["cost", "price", "budget", "money"]):
                intent_type = "cost_decision"
            elif any(word in title_lower for word in ["safety", "checklist", "precaution", "secure"]):
                intent_type = "checklist_safety"
            elif any(word in title_lower for word in ["local", "regional", "seasonal", "climate"]):
                intent_type = "local_seasonal"
            else:
                intent_type = "howto"  # Default
        
        # Select template based on intent type
        import random
        
        # Filter templates by intent type
        intent_templates = {
            "howto": [t for t in self.high_intent_templates if "How to" in t or "Step-by-Step" in t],
            "troubleshoot": [t for t in self.high_intent_templates if "Fix" in t or "Problem" in t or "Troubleshoot" in t],
            "cost_decision": [t for t in self.high_intent_templates if "Cost" in t or "Price" in t or "Budget" in t],
            "checklist_safety": [t for t in self.high_intent_templates if "Safety" in t or "Checklist" in t],
            "local_seasonal": [t for t in self.high_intent_templates if "Region" in t or "Seasonal" in t or "Climate" in t],
        }
        
        # Get templates for this intent type, fallback to all templates
        available_templates = intent_templates.get(intent_type, [])
        if not available_templates:
            available_templates = self.high_intent_templates
        
        # Choose template
        template = random.choice(available_templates)
        
        # Prepare template variables
        template_vars = {}
        
        # Noun (required for most templates)
        if nouns:
            template_vars["{noun}"] = random.choice(nouns).title()
        else:
            # Use niche as fallback noun
            template_vars["{noun}"] = niche.title()
        
        # Action verb
        if action_verbs:
            template_vars["{action}"] = random.choice(action_verbs).title()
        
        # Benefit
        if benefits:
            template_vars["{benefit}"] = random.choice(benefits)
        
        # Context (niche)
        template_vars["{context}"] = niche
        
        # Problem (extract from title or use generic)
        if "?" in title:
            template_vars["{problem}"] = title
        else:
            # Extract problem from title or use generic
            problem_words = [w for w in words if len(w) > 4]
            if problem_words:
                template_vars["{problem}"] = f"How to deal with {random.choice(problem_words)}"
            else:
                template_vars["{problem}"] = f"How to solve {niche.lower()} problems"
        
        # Solution (action verb)
        if action_verbs:
            template_vars["{solution}"] = random.choice(action_verbs)
        
        # Number
        template_vars["{number}"] = random.choice(["5", "7", "10", "3"])
        
        # Year
        from datetime import datetime
        template_vars["{year}"] = str(datetime.now().year)
        
        # Alternative (for comparison templates)
        if len(nouns) > 1:
            template_vars["{alternative}"] = random.choice(nouns[1:]).title()
        
        # Fill template
        filled = template
        for placeholder, value in template_vars.items():
            if placeholder in filled:
                filled = filled.replace(placeholder, value)
        
        # Clean up
        filled = filled.replace("  ", " ").strip()
        
        # Ensure reasonable length
        if len(filled) > 80:
            filled = filled[:77] + "..."
        
        return filled
    
    def _is_title_improved(self, new_title: str, old_score: float, niche: str, hub: str = "", cluster: str = "") -> bool:
        """Check if new title is an improvement over old one."""
        if not new_title:
            return False
        
        # Validate new title
        new_result = self.validate_title(new_title, niche, hub, cluster)
        
        # Improvement if: concrete when old wasn't, or score increased significantly
        if not new_result.is_concrete:
            return False
        
        score_improvement = new_result.score - old_score
        return score_improvement > 0.1  # At least 10% improvement
    
    def batch_validate_and_rewrite(self, titles: List[str], niche: str, hub: str = "", cluster: str = "") -> List[ConcretenessGateResult]:
        """Validate and rewrite multiple titles."""
        results = []
        
        for title in titles:
            result = self.validate_and_rewrite(title, niche, hub, cluster)
            results.append(result)
        
        return results


def enforce_concreteness_gate(titles: List[str], niche: str, site_root: Path) -> List[str]:
    """
    Convenience function to enforce concreteness gate on titles.
    
    Args:
        titles: List of titles to process
        niche: The niche context
        site_root: Site root path for cache
        
    Returns:
        List of concrete titles (rewritten if necessary)
    """
    gate = ConcretenessGate(site_root)
    results = gate.batch_validate_and_rewrite(titles, niche)
    
    concrete_titles = []
    rewrite_stats = {"total": len(titles), "rewritten": 0, "failed": 0}
    
    for result in results:
        if result.rewritten_title:
            concrete_titles.append(result.rewritten_title)
            rewrite_stats["rewritten"] += 1
            logger.info(f"Rewritten: '{result.title}' -> '{result.rewritten_title}'")
        elif result.is_concrete:
            concrete_titles.append(result.title)
        else:
            # Use fallback: template rewrite
            fallback = gate._rewrite_with_template(result.title, niche)
            concrete_titles.append(fallback)
            rewrite_stats["failed"] += 1
            logger.warning(f"Failed to rewrite, using template: '{result.title}' -> '{fallback}'")
    
    logger.info(f"ConcretenessGate stats: {rewrite_stats['rewritten']}/{rewrite_stats['total']} rewritten, "
                f"{rewrite_stats['failed']} used template fallback")
    
    return concrete_titles


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ConcretenessGate validator")
    parser.add_argument("title", help="Title to validate and rewrite")
    parser.add_argument("--niche", default="test niche", help="Niche context")
    parser.add_argument("--hub", default="", help="Hub context (optional)")
    parser.add_argument("--cluster", default="", help="Cluster context (optional)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    gate = ConcretenessGate()
    result = gate.validate_and_rewrite(args.title, args.niche, args.hub, args.cluster)
    
    print(f"Original title: {result.title}")
    print(f"Is concrete: {result.is_concrete}")
    print(f"Score: {result.score:.2f}")
    print(f"Issues: {result.issues}")
    
    if result.rewritten_title:
        print(f"Rewritten title: {result.rewritten_title}")
        print(f"Rewrite reason: {result.rewrite_reason}")
        print(f"Attempts: {result.attempts}")
    else:
        print("No rewrite needed or possible")
    
    print(f"\nValidation details:")
    print(f"  Has concrete noun: {result.score > 0.5}")
    print(f"  Has action verb: {'action' in str(result.issues).lower()}")
    print(f"  Vague detected: {'vague' in str(result.issues).lower()}")
