#!/usr/bin/env python3
"""
ToneSpec: Deterministic tone preset + risk tier system for site-wide consistency.

This module provides:
1. Tone Preset Library (6-10 presets with specific attributes)
2. Risk Tier Classifier (deterministic rules)
3. Site-Level ToneSpec computation and persistence
4. ToneGate validation with rewrite loop
"""

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tiers controlling disclaimers and claims language."""
    TIER_0 = 0  # No risk - factual, educational, non-actionable
    TIER_1 = 1  # Low risk - recommendations, opinions, mild claims
    TIER_2 = 2  # Medium risk - actionable advice, product recommendations
    TIER_3 = 3  # High risk - health, safety, financial, legal implications


@dataclass
class TonePreset:
    """A tone preset defining voice, style, and safety rules."""
    id: str  # Unique identifier (e.g., "practical_guide")
    label: str  # Human-readable label (e.g., "Practical Guide")
    voice_description: str  # Detailed voice description for LLM prompts
    reading_level: str  # "8th_grade", "high_school", "college", "technical"
    style_rules: List[str]  # Specific style rules to enforce
    banned_words: List[str]  # Words to avoid in this tone
    required_phrases: List[str]  # Phrases that should appear (if applicable)
    disclaimer_template: Optional[str] = None  # Template for disclaimers
    claims_safety_level: str = "moderate"  # "strict", "moderate", "permissive"
    risk_tier_range: Tuple[int, int] = (0, 2)  # Min and max risk tiers allowed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_prompt_context(self) -> str:
        """Get the prompt context string for LLM instructions."""
        return f"""Tone: {self.label}
Voice: {self.voice_description}
Reading Level: {self.reading_level.replace('_', ' ').title()}
Style Rules:
{chr(10).join(f'- {rule}' for rule in self.style_rules)}
Avoid: {', '.join(self.banned_words) if self.banned_words else 'None'}
Required Phrases: {', '.join(self.required_phrases) if self.required_phrases else 'None'}"""


class TonePresetLibrary:
    """Library of tone presets (6-10 presets as specified)."""
    
    @staticmethod
    def get_all_presets() -> Dict[str, TonePreset]:
        """Return all available tone presets."""
        return {
            preset.id: preset for preset in [
                # 1. Practical Guide (default) - Most common, balanced
                TonePreset(
                    id="practical_guide",
                    label="Practical Guide",
                    voice_description="Clear, step-by-step, actionable advice. Direct but friendly. Focus on practical solutions and real-world application. Avoid theoretical digressions.",
                    reading_level="8th_grade",
                    style_rules=[
                        "Use second-person 'you' when giving instructions",
                        "Break complex tasks into numbered steps",
                        "Include concrete examples for each point",
                        "Use active voice for clarity",
                        "End with a clear call-to-action or summary"
                    ],
                    banned_words=[
                        "theoretically", "arguably", "perhaps", "maybe", 
                        "somewhat", "kind of", "sort of", "very", "really"
                    ],
                    required_phrases=[],
                    disclaimer_template="This guide provides practical advice based on common practices. Individual results may vary.",
                    claims_safety_level="moderate",
                    risk_tier_range=(0, 2)
                ),
                
                # 2. Technical Explainer - For complex topics
                TonePreset(
                    id="technical_explainer",
                    label="Technical Explainer",
                    voice_description="Precise, detailed, systematic. Explain concepts with appropriate technical depth. Assume reader has basic domain knowledge but needs clarification on specifics.",
                    reading_level="college",
                    style_rules=[
                        "Define technical terms on first use",
                        "Use diagrams or analogies where helpful",
                        "Compare and contrast different approaches",
                        "Cite specifications or standards when relevant",
                        "Structure explanations from general to specific"
                    ],
                    banned_words=[
                        "easy", "simple", "just", "merely", "obviously",
                        "as you know", "of course", "clearly"
                    ],
                    required_phrases=[],
                    disclaimer_template="This explanation covers technical concepts at a general level. Consult official documentation for implementation details.",
                    claims_safety_level="strict",
                    risk_tier_range=(0, 1)
                ),
                
                # 3. Supportive Coach - Encouraging, motivational
                TonePreset(
                    id="supportive_coach",
                    label="Supportive Coach",
                    voice_description="Encouraging, empathetic, patient. Acknowledge challenges while providing gentle guidance. Use positive reinforcement and celebrate small wins.",
                    reading_level="8th_grade",
                    style_rules=[
                        "Use 'we' language to create partnership",
                        "Acknowledge common frustrations or obstacles",
                        "Provide encouragement after difficult steps",
                        "Use positive framing (what to do vs what not to do)",
                        "Include progress checkpoints"
                    ],
                    banned_words=[
                        "failure", "wrong", "bad", "stupid", "idiot",
                        "can't", "won't", "impossible", "never"
                    ],
                    required_phrases=[],
                    disclaimer_template="This coaching approach is meant to be supportive and encouraging. Progress happens at different paces for everyone.",
                    claims_safety_level="moderate",
                    risk_tier_range=(0, 2)
                ),
                
                # 4. Outdoors/Field Manual - Rugged, direct, no-nonsense
                TonePreset(
                    id="field_manual",
                    label="Field Manual",
                    voice_description="Concise, direct, no-fluff. Write like a military field manual or survival guide. Prioritize essential information and clear procedures.",
                    reading_level="high_school",
                    style_rules=[
                        "Use imperative mood for instructions",
                        "Prioritize information by importance",
                        "Include warning/caution notes where needed",
                        "Use bullet points for equipment lists",
                        "Keep sentences short and declarative"
                    ],
                    banned_words=[
                        "perhaps", "maybe", "possibly", "sometimes",
                        "usually", "often", "frequently", "beautiful",
                        "wonderful", "amazing"
                    ],
                    required_phrases=[],
                    disclaimer_template="Field procedures carry inherent risks. Always assess local conditions and use proper safety equipment.",
                    claims_safety_level="strict",
                    risk_tier_range=(1, 3)
                ),
                
                # 5. Budget/Consumer - Value-focused, comparison-driven
                TonePreset(
                    id="budget_consumer",
                    label="Budget Consumer Guide",
                    voice_description="Value-conscious, comparison-focused. Help readers make smart purchasing decisions. Emphasize features vs cost, long-term value, and alternatives.",
                    reading_level="8th_grade",
                    style_rules=[
                        "Compare options using clear criteria",
                        "Highlight cost-benefit tradeoffs",
                        "Mention cheaper alternatives where applicable",
                        "Discuss long-term ownership costs",
                        "Include 'what to look for' checklists"
                    ],
                    banned_words=[
                        "luxury", "premium", "exclusive", "elite",
                        "expensive", "costly", "overpriced"
                    ],
                    required_phrases=[],
                    disclaimer_template="Prices and availability change frequently. Always verify current information before making purchases.",
                    claims_safety_level="moderate",
                    risk_tier_range=(0, 2)
                ),
                
                # 6. Creative/Hobby - Inspirational, exploratory
                TonePreset(
                    id="creative_hobby",
                    label="Creative Hobby Guide",
                    voice_description="Inspirational, exploratory, open-ended. Encourage creativity and personal expression. Focus on possibilities rather than rigid procedures.",
                    reading_level="high_school",
                    style_rules=[
                        "Use open-ended questions to spark ideas",
                        "Show multiple approaches or variations",
                        "Include 'try this' experimentation prompts",
                        "Celebrate unique results and personal style",
                        "Avoid rigid 'right way/wrong way' framing"
                    ],
                    banned_words=[
                        "must", "should", "have to", "required",
                        "correct", "incorrect", "proper", "improper"
                    ],
                    required_phrases=[],
                    disclaimer_template="Creative expression is subjective. These suggestions are starting points for your own exploration.",
                    claims_safety_level="permissive",
                    risk_tier_range=(0, 1)
                ),
                
                # 7. Medical/Health Advisory - Cautious, evidence-based
                TonePreset(
                    id="medical_advisory",
                    label="Medical/Health Advisory",
                    voice_description="Cautious, evidence-based, clear about limitations. Distinguish between established facts and emerging research. Emphasize professional consultation.",
                    reading_level="high_school",
                    style_rules=[
                        "Cite sources or evidence levels when possible",
                        "Use qualifiers like 'research suggests' or 'studies indicate'",
                        "Clearly separate facts from opinions",
                        "Include 'when to see a professional' guidance",
                        "Avoid absolute statements about health outcomes"
                    ],
                    banned_words=[
                        "cure", "guarantee", "promise", "miracle",
                        "breakthrough", "revolutionary", "100%", "always",
                        "never"
                    ],
                    required_phrases=["Consult a healthcare professional"],
                    disclaimer_template="This information is for educational purposes only and not medical advice. Always consult qualified healthcare providers.",
                    claims_safety_level="strict",
                    risk_tier_range=(0, 3)
                ),
                
                # 8. Business/Professional - Formal, strategic
                TonePreset(
                    id="business_professional",
                    label="Business/Professional",
                    voice_description="Formal, strategic, results-oriented. Focus on ROI, efficiency, and professional standards. Use business terminology appropriately.",
                    reading_level="college",
                    style_rules=[
                        "Use industry-standard terminology",
                        "Include ROI or business case considerations",
                        "Reference best practices or standards",
                        "Structure content by business objectives",
                        "Include implementation timelines or phases"
                    ],
                    banned_words=[
                        "just", "simple", "easy", "quick fix",
                        "hack", "trick", "secret", "guru"
                    ],
                    required_phrases=[],
                    disclaimer_template="Business strategies depend on specific circumstances. Conduct due diligence before implementation.",
                    claims_safety_level="moderate",
                    risk_tier_range=(0, 2)
                )
            ]
        }
    
    @staticmethod
    def get_preset(preset_id: str) -> Optional[TonePreset]:
        """Get a specific tone preset by ID."""
        return TonePresetLibrary.get_all_presets().get(preset_id)
    
    @staticmethod
    def get_default_preset() -> TonePreset:
        """Get the default tone preset (Practical Guide)."""
        return TonePresetLibrary.get_preset("practical_guide")


class RiskTierClassifier:
    """Deterministic risk tier classification based on niche keywords."""
    
    # Keyword mappings to risk tiers
    RISK_KEYWORDS = {
        RiskTier.TIER_3: {  # High risk - health, safety, financial, legal
            "health", "medical", "safety", "financial", "legal", "law", 
            "investment", "therapy", "treatment", "diagnosis", "prescription",
            "surgery", "emergency", "danger", "risk", "hazard", "toxic",
            "legal advice", "medical advice", "financial advice", "insurance",
            "retirement", "investment", "stock", "crypto", "lawsuit"
        },
        RiskTier.TIER_2: {  # Medium risk - actionable advice, products
            "how to", "tutorial", "guide", "DIY", "repair", "install",
            "build", "construct", "cook", "prepare", "recipe", "exercise",
            "workout", "training", "driving", "operate", "use", "apply",
            "product review", "best", "top", "recommend", "buy", "purchase"
        },
        RiskTier.TIER_1: {  # Low risk - recommendations, opinions
            "review", "compare", "versus", "vs", "opinion", "thoughts",
            "experience", "story", "perspective", "analysis", "trend",
            "news", "update", "what's new", "changes", "improvements"
        },
        RiskTier.TIER_0: {  # No risk - factual, educational
            "facts", "information", "education", "learn", "study",
            "history", "background", "overview", "introduction",
            "explanation", "definition", "what is", "basics"
        }
    }
    
    @staticmethod
    def classify_from_niche(niche: str) -> RiskTier:
        """
        Deterministically classify risk tier from niche string.
        Uses keyword matching with fallback to TIER_1.
        """
        niche_lower = niche.lower()
        
        # Check for highest risk first (TIER_3)
        for keyword in RiskTierClassifier.RISK_KEYWORDS[RiskTier.TIER_3]:
            if keyword in niche_lower:
                return RiskTier.TIER_3
        
        # Check for TIER_2
        for keyword in RiskTierClassifier.RISK_KEYWORDS[RiskTier.TIER_2]:
            if keyword in niche_lower:
                return RiskTier.TIER_2
        
        # Check for TIER_0
        for keyword in RiskTierClassifier.RISK_KEYWORDS[RiskTier.TIER_0]:
            if keyword in niche_lower:
                return RiskTier.TIER_0
        
        # Default to TIER_1 (low risk)
        return RiskTier.TIER_1
    
    @staticmethod
    def get_disclaimer_for_tier(risk_tier: RiskTier, tone_preset: TonePreset) -> str:
        """Get appropriate disclaimer text for risk tier and tone preset."""
        base_disclaimer = tone_preset.disclaimer_template or ""
        
        if risk_tier == RiskTier.TIER_3:
            return f"IMPORTANT SAFETY NOTICE: {base_disclaimer} This content discusses topics with significant health, safety, financial, or legal implications. Always consult qualified professionals before taking any action."
        elif risk_tier == RiskTier.TIER_2:
            return f"Note: {base_disclaimer} This guide provides actionable advice. Use appropriate safety precautions and verify information fits your specific situation."
        elif risk_tier == RiskTier.TIER_1:
            return f"Disclaimer: {base_disclaimer} These are recommendations based on general knowledge. Your experience may vary."
        else:  # TIER_0
            return f"Educational Note: {base_disclaimer} This content is for informational purposes only."


@dataclass
class ToneSpec:
    """Complete tone specification for a site."""
    niche: str
    tone_preset: TonePreset
    risk_tier: RiskTier
    computed_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    # Generated fields
    disclaimer_text: str = ""
    prompt_context: str = ""
    
    def __post_init__(self):
        """Generate derived fields after initialization."""
        self.disclaimer_text = RiskTierClassifier.get_disclaimer_for_tier(
            self.risk_tier, self.tone_preset
        )
        self.prompt_context = self.tone_preset.get_prompt_context()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "niche": self.niche,
            "tone_preset_id": self.tone_preset.id,
            "risk_tier": self.risk_tier.value,
            "computed_at": self.computed_at.isoformat(),
            "version": self.version,
            "disclaimer_text": self.disclaimer_text,
            "prompt_context": self.prompt_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToneSpec':
        """Create ToneSpec from dictionary."""
        preset = TonePresetLibrary.get_preset(data["tone_preset_id"])
        if not preset:
            preset = TonePresetLibrary.get_default_preset()
        
        risk_tier = RiskTier(data["risk_tier"])
        
        # Parse datetime
        computed_at = datetime.fromisoformat(data["computed_at"])
        
        spec = cls(
            niche=data["niche"],
            tone_preset=preset,
            risk_tier=risk_tier,
            computed_at=computed_at,
            version=data.get("version", "1.0")
        )
        
        # Set generated fields if present
        if "disclaimer_text" in data:
            spec.disclaimer_text = data["disclaimer_text"]
        if "prompt_context" in data:
            spec.prompt_context = data["prompt_context"]
        
        return spec
    
    def get_llm_instruction(self) -> str:
        """Get complete LLM instruction incorporating tone and risk."""
        return f"""WRITE IN THIS TONE AND STYLE:

{self.prompt_context}

RISK LEVEL: {self.risk_tier.name.replace('_', ' ').title()}

IMPORTANT CONSTRAINTS:
1. {self.disclaimer_text}
2. Claims must be appropriate for {self.risk_tier.name.replace('_', ' ').lower()} risk content
3. Follow all style rules listed above"""


class ToneSpecGenerator:
    """Generates and persists site-level ToneSpec."""
    
    def __init__(self, site_root: Optional[Path] = None):
        self.site_root = Path(site_root) if site_root else Path.cwd()
        self.cache_dir = self.site_root / "data" / "tone_specs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_for_niche(self, niche: str, preset_id: Optional[str] = None) -> ToneSpec:
        """
        Compute ToneSpec for a niche.
        
        Args:
            niche: The niche string
            preset_id: Optional preset ID, if None uses deterministic selection
        
        Returns:
            ToneSpec object
        """
        # Determine tone preset
        if preset_id:
            tone_preset = TonePresetLibrary.get_preset(preset_id)
            if not tone_preset:
                logger.warning(f"Preset {preset_id} not found, using default")
                tone_preset = self._determine_preset_from_niche(niche)
        else:
            tone_preset = self._determine_preset_from_niche(niche)
        
        # Determine risk tier
        risk_tier = RiskTierClassifier.classify_from_niche(niche)
        
        # Ensure preset allows this risk tier
        min_risk, max_risk = tone_preset.risk_tier_range
        if risk_tier.value < min_risk or risk_tier.value > max_risk:
            logger.warning(f"Risk tier {risk_tier.value} outside preset range {min_risk}-{max_risk}, adjusting")
            # Adjust to nearest allowed value
            if risk_tier.value < min_risk:
                risk_tier = RiskTier(min_risk)
            else:
                risk_tier = RiskTier(max_risk)
        
        # Create ToneSpec
        tone_spec = ToneSpec(
            niche=niche,
            tone_preset=tone_preset,
            risk_tier=risk_tier
        )
        
        return tone_spec
    
    def _determine_preset_from_niche(self, niche: str) -> TonePreset:
        """Deterministically select tone preset based on niche keywords."""
        niche_lower = niche.lower()
        
        # Medical/health niches -> Medical Advisory
        medical_keywords = {"health", "medical", "therapy", "treatment", "medicine", "wellness"}
        if any(keyword in niche_lower for keyword in medical_keywords):
            return TonePresetLibrary.get_preset("medical_advisory")
        
        # Business/professional niches
        business_keywords = {"business", "professional", "enterprise", "corporate", "strategy", "management"}
        if any(keyword in niche_lower for keyword in business_keywords):
            return TonePresetLibrary.get_preset("business_professional")
        
        # Technical/engineering niches
        technical_keywords = {"technical", "engineering", "software", "code", "programming", "technology"}
        if any(keyword in niche_lower for keyword in technical_keywords):
            return TonePresetLibrary.get_preset("technical_explainer")
        
        # Creative/hobby niches
        creative_keywords = {"creative", "hobby", "art", "craft", "diy", "handmade", "design"}
        if any(keyword in niche_lower for keyword in creative_keywords):
            return TonePresetLibrary.get_preset("creative_hobby")
        
        # Outdoor/field niches
        outdoor_keywords = {"outdoor", "field", "survival", "camping", "hiking", "fishing", "hunting"}
        if any(keyword in niche_lower for keyword in outdoor_keywords):
            return TonePresetLibrary.get_preset("field_manual")
        
        # Budget/consumer niches
        budget_keywords = {"budget", "cheap", "affordable", "value", "consumer", "shopping", "buying"}
        if any(keyword in niche_lower for keyword in budget_keywords):
            return TonePresetLibrary.get_preset("budget_consumer")
        
        # Default to Practical Guide
        return TonePresetLibrary.get_default_preset()
    
    def save_tone_spec(self, tone_spec: ToneSpec) -> Path:
        """Save ToneSpec to cache file."""
        cache_file = self.cache_dir / f"{self._slugify(tone_spec.niche)}.json"
        
        data = tone_spec.to_dict()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved ToneSpec to {cache_file}")
        return cache_file
    
    def load_tone_spec(self, niche: str) -> Optional[ToneSpec]:
        """Load ToneSpec from cache if exists and recent (< 30 days)."""
        cache_file = self.cache_dir / f"{self._slugify(niche)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is recent (less than 30 days)
            computed_at = datetime.fromisoformat(data["computed_at"])
            age_days = (datetime.now() - computed_at).days
            if age_days > 30:
                logger.info(f"ToneSpec cache for {niche} is {age_days} days old, regenerating")
                return None
            
            return ToneSpec.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load ToneSpec from {cache_file}: {e}")
            return None
    
    def get_or_compute(self, niche: str, preset_id: Optional[str] = None) -> ToneSpec:
        """Get cached ToneSpec or compute new one."""
        # Try to load from cache
        cached = self.load_tone_spec(niche)
        if cached:
            return cached
        
        # Compute new
        tone_spec = self.compute_for_niche(niche, preset_id)
        
        # Save to cache
        self.save_tone_spec(tone_spec)
        
        return tone_spec
    
    def _slugify(self, text: str) -> str:
        """Create filesystem-safe slug from text."""
        # Convert to lowercase
        text = text.lower()
        # Replace spaces and special characters with hyphens
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        # Remove leading/trailing hyphens
        return text.strip('-')


def get_tone_spec_for_niche(niche: str, site_root: Optional[Path] = None) -> ToneSpec:
    """Convenience function to get ToneSpec for a niche."""
    generator = ToneSpecGenerator(site_root)
    return generator.get_or_compute(niche)


def ensure_tone_spec_in_prompt(system_prompt: str, tone_spec: ToneSpec) -> str:
    """Ensure ToneSpec instructions are included in system prompt."""
    tone_instruction = tone_spec.get_llm_instruction()
    
    # Check if tone instructions are already in prompt
    if "WRITE IN THIS TONE AND STYLE" in system_prompt:
        # Already has tone instructions, return as-is
        return system_prompt
    
    # Add tone instructions at the beginning
    return f"{tone_instruction}\n\n{system_prompt}"


# --- ToneGate Validator -------------------------------------------------

@dataclass
class ToneGateResult:
    """Result of tone validation and rewriting."""
    passed: bool
    rewritten_text: Optional[str] = None
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "rewritten_text": self.rewritten_text,
            "failures": self.failures,
            "warnings": self.warnings
        }


class ToneGate:
    """Validates content against ToneSpec with rewrite loop."""
    
    def __init__(self, tone_spec: ToneSpec, site_root: Optional[Path] = None):
        self.tone_spec = tone_spec
        self.site_root = Path(site_root) if site_root else Path.cwd()
        
    def validate_content(self, content: str, content_type: str = "article") -> ToneGateResult:
        """
        Validate content against ToneSpec.
        
        Args:
            content: Text content to validate
            content_type: Type of content ("article", "hub", "cluster", "page")
        
        Returns:
            ToneGateResult with validation results
        """
        failures = []
        warnings = []
        
        # 1. Structural validation (for articles)
        if content_type == "article":
            if not self._has_required_blocks(content):
                failures.append("Missing required article blocks (introduction, body, conclusion)")
        
        # 2. Anti-fluff validation
        fluff_score = self._detect_fluff(content)
        if fluff_score > 0.3:  # More than 30% fluff
            failures.append(f"Excessive fluff detected ({fluff_score:.0%})")
        elif fluff_score > 0.15:  # 15-30% fluff
            warnings.append(f"Moderate fluff detected ({fluff_score:.0%})")
        
        # 3. Claims safety validation
        unsafe_claims = self._detect_unsafe_claims(content)
        if unsafe_claims:
            risk_level = self.tone_spec.risk_tier.value
            if risk_level >= 2:  # Medium or high risk
                failures.append(f"Unsafe claims detected: {', '.join(unsafe_claims[:3])}")
            else:
                warnings.append(f"Potentially unsafe claims: {', '.join(unsafe_claims[:2])}")
        
        # 4. Tone consistency validation
        tone_violations = self._check_tone_consistency(content)
        if tone_violations:
            warnings.extend(tone_violations)
        
        # 5. Banned words validation
        banned_words_found = self._check_banned_words(content)
        if banned_words_found:
            failures.append(f"Banned words used: {', '.join(banned_words_found)}")
        
        # 6. Required phrases check (if any)
        missing_required = self._check_required_phrases(content)
        if missing_required:
            warnings.append(f"Missing recommended phrases: {', '.join(missing_required)}")
        
        passed = len(failures) == 0
        
        return ToneGateResult(
            passed=passed,
            failures=failures,
            warnings=warnings
        )
    
    def validate_and_rewrite(self, content: str, content_type: str = "article",
                           max_attempts: int = 2) -> ToneGateResult:
        """
        Validate content and rewrite if needed (max 2 attempts as specified).
        
        Args:
            content: Original content
            content_type: Type of content
            max_attempts: Maximum rewrite attempts (default: 2)
        
        Returns:
            ToneGateResult with rewritten text if applicable
        """
        original_result = self.validate_content(content, content_type)
        
        if original_result.passed:
            return original_result
        
        # Attempt rewrite
        for attempt in range(max_attempts):
            logger.info(f"ToneGate rewrite attempt {attempt + 1}/{max_attempts}")
            
            rewritten = self._rewrite_with_llm(content, content_type, attempt)
            if not rewritten:
                continue
            
            # Validate rewritten content
            new_result = self.validate_content(rewritten, content_type)
            
            if new_result.passed:
                return ToneGateResult(
                    passed=True,
                    rewritten_text=rewritten,
                    failures=[],
                    warnings=new_result.warnings
                )
        
        # All attempts failed
        return original_result
    
    def _has_required_blocks(self, content: str) -> bool:
        """Check if article has required structural blocks."""
        # Simple check for common section markers
        has_intro = any(marker in content.lower() for marker in
                       ["introduction", "overview", "what is", "in this article"])
        has_body = len(content.split("\n\n")) >= 3  # At least 3 paragraphs
        has_conclusion = any(marker in content.lower() for marker in
                           ["conclusion", "summary", "key takeaways", "final thoughts"])
        
        return has_intro and has_body and has_conclusion
    
    def _detect_fluff(self, content: str) -> float:
        """Detect fluff content ratio (0.0 to 1.0)."""
        fluff_patterns = [
            r"\b(very|really|quite|extremely|incredibly|absolutely)\b",
            r"\b(in fact|as a matter of fact|the fact is)\b",
            r"\b(it is important to note|it should be noted|it is worth mentioning)\b",
            r"\b(in today's world|in this day and age|nowadays)\b",
            r"\b(at the end of the day|when all is said and done)\b",
            r"\b(as you know|as we all know|obviously|clearly)\b",
            r"\b(needless to say|it goes without saying)\b",
        ]
        
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return 0.0
        
        fluff_sentences = 0
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue
            
            # Check for fluff patterns
            for pattern in fluff_patterns:
                if re.search(pattern, sentence_lower):
                    fluff_sentences += 1
                    break
            
            # Check for very short, empty sentences
            if len(sentence_lower.split()) < 3:
                fluff_sentences += 1
        
        return fluff_sentences / len(sentences) if sentences else 0.0
    
    def _detect_unsafe_claims(self, content: str) -> List[str]:
        """Detect unsafe claims based on risk tier."""
        risk_tier = self.tone_spec.risk_tier
        
        # Claims that are unsafe at different risk levels
        unsafe_patterns = []
        
        if risk_tier.value >= 3:  # High risk
            unsafe_patterns.extend([
                r"\b(cure|heal|treat|diagnose|prescribe)\b",
                r"\b(guarantee|promise|100%|always|never)\b",
                r"\b(invest|buy|sell|trade)\s+(stock|share|currency|crypto)\b",
                r"\b(legal advice|legal opinion|you should sue)\b",
            ])
        
        if risk_tier.value >= 2:  # Medium risk
            unsafe_patterns.extend([
                r"\b(best|top|perfect|ideal)\b",
                r"\b(easy|simple|quick|fast)\s+(fix|solution|answer)\b",
                r"\b(you must|you have to|you need to)\b",
                r"\b(proven|scientifically proven|clinically proven)\b",
            ])
        
        if risk_tier.value >= 1:  # Low risk
            unsafe_patterns.extend([
                r"\b(everyone|nobody|always|never)\b",
                r"\b(the only|the best|the worst)\b",
            ])
        
        # Find matches
        unsafe_claims = []
        for pattern in unsafe_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            unsafe_claims.extend(matches)
        
        return list(set(unsafe_claims))[:10]  # Return unique, limit to 10
    
    def _check_tone_consistency(self, content: str) -> List[str]:
        """Check for tone consistency violations."""
        violations = []
        preset = self.tone_spec.tone_preset
        
        # Check reading level (simple heuristic)
        avg_word_length = sum(len(word) for word in content.split()) / max(1, len(content.split()))
        if preset.reading_level == "8th_grade" and avg_word_length > 6:
            violations.append("Vocabulary may be too advanced for 8th grade reading level")
        elif preset.reading_level == "college" and avg_word_length < 4:
            violations.append("Vocabulary may be too simple for college reading level")
        
        # Check style rule compliance
        for rule in preset.style_rules[:3]:  # Check first 3 rules
            rule_lower = rule.lower()
            if "active voice" in rule_lower and self._has_passive_voice(content):
                violations.append("Uses passive voice contrary to style rules")
            if "second-person" in rule_lower and not self._has_second_person(content):
                violations.append("Missing second-person address as required")
            if "bullet points" in rule_lower and not self._has_bullet_points(content):
                violations.append("Missing bullet points as recommended")
        
        return violations
    
    def _check_banned_words(self, content: str) -> List[str]:
        """Check for banned words from tone preset."""
        content_lower = content.lower()
        banned_words_found = []
        
        for word in self.tone_spec.tone_preset.banned_words:
            if re.search(r'\b' + re.escape(word.lower()) + r'\b', content_lower):
                banned_words_found.append(word)
        
        return banned_words_found
    
    def _check_required_phrases(self, content: str) -> List[str]:
        """Check for missing required phrases."""
        content_lower = content.lower()
        missing = []
        
        for phrase in self.tone_spec.tone_preset.required_phrases:
            if phrase.lower() not in content_lower:
                missing.append(phrase)
        
        return missing
    
    def _has_passive_voice(self, content: str) -> bool:
        """Simple passive voice detection."""
        passive_patterns = [
            r"\bis\s+[a-z]+\s+by\b",
            r"\bare\s+[a-z]+\s+by\b",
            r"\bwas\s+[a-z]+\s+by\b",
            r"\bwere\s+[a-z]+\s+by\b",
            r"\bbe\s+[a-z]+\s+by\b",
        ]
        
        for pattern in passive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _has_second_person(self, content: str) -> bool:
        """Check for second-person pronouns."""
        second_person = ["you", "your", "yours", "yourself", "yourselves"]
        content_lower = content.lower()
        
        for pronoun in second_person:
            if re.search(r'\b' + pronoun + r'\b', content_lower):
                return True
        
        return False
    
    def _has_bullet_points(self, content: str) -> bool:
        """Check for bullet points or numbered lists."""
        bullet_patterns = [r'^\s*[\-\*•]\s', r'^\s*\d+\.\s']
        
        for line in content.split('\n'):
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    return True
        
        return False
    
    def _rewrite_with_llm(self, content: str, content_type: str, attempt: int) -> Optional[str]:
        """Rewrite content using LLM to fix tone violations."""
        try:
            from llm_client import llm_json
        except ImportError:
            logger.error("Cannot import llm_client for rewrite")
            return None
        
        preset = self.tone_spec.tone_preset
        risk_tier = self.tone_spec.risk_tier
        
        system_prompt = f"""You are a tone editor. Rewrite the following {content_type} content to match the required tone and fix violations.

TONE REQUIREMENTS:
{preset.get_prompt_context()}

RISK LEVEL: {risk_tier.name.replace('_', ' ').title()}

REWRITE INSTRUCTIONS:
1. Maintain all factual information and key points
2. Fix tone violations while preserving meaning
3. Adjust vocabulary to match {preset.reading_level.replace('_', ' ')} reading level
4. Remove fluff and unnecessary phrases
5. Ensure claims are appropriate for {risk_tier.name.replace('_', ' ').lower()} risk content
6. Follow all style rules: {chr(10).join(f'- {rule}' for rule in preset.style_rules[:3])}

Return ONLY the rewritten content, no explanations."""
        
        user_prompt = f"Original {content_type} content to rewrite:\n\n{content}"
        
        try:
            response = llm_json(
                system=system_prompt,
                user=user_prompt,
                temperature=0.7 + (attempt * 0.1)  # Increase temperature with attempts
            )
            
            # Extract rewritten content
            if isinstance(response, dict) and "rewritten_content" in response:
                return response["rewritten_content"]
            elif isinstance(response, str):
                return response
            else:
                # Try to find text in response
                response_str = str(response)
                # Extract content between markers or use whole response
                return response_str[:5000]  # Limit length
            
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")
            return None


def apply_tone_gate(content: str, niche: str, content_type: str = "article",
                   site_root: Optional[Path] = None) -> Tuple[str, ToneGateResult]:
    """
    Apply ToneGate to content for a given niche.
    
    Returns:
        Tuple of (final_content, tone_gate_result)
    """
    # Get ToneSpec for niche
    tone_spec = get_tone_spec_for_niche(niche, site_root)
    
    # Create ToneGate
    tone_gate = ToneGate(tone_spec, site_root)
    
    # Validate and rewrite
    result = tone_gate.validate_and_rewrite(content, content_type)
    
    # Return appropriate content
    final_content = result.rewritten_text if result.rewritten_text else content
    
    return final_content, result
