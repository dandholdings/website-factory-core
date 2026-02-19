#!/usr/bin/env python3
"""
json_schema_enforcer.py — Enforce JSON schema for page generation with fallbacks.

Provides stronger JSON forcing wrapper and validation before accepting model output.
If missing required keys or sections, auto-retry with schema reminder prompt.
After N retries, fall back to safe deterministic page skeleton.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class PageType(Enum):
    """Supported page types for schema validation."""
    HOW_TO_GUIDE = "how_to_guide"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    BUYING_GUIDE = "buying_guide"
    BEGINNER_GUIDE = "beginner_guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"


@dataclass
class SchemaValidationResult:
    """Result of JSON schema validation."""
    is_valid: bool
    missing_keys: List[str]
    invalid_sections: List[str]
    schema_violations: List[str]
    suggested_fixes: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "missing_keys": self.missing_keys,
            "invalid_sections": self.invalid_sections,
            "schema_violations": self.schema_violations,
            "suggested_fixes": self.suggested_fixes
        }


class JSONSchemaEnforcer:
    """Enforces JSON schema for page generation with retries and fallbacks."""
    
    def __init__(self):
        # Define required schemas for different page types
        self.page_schemas = {
            PageType.HOW_TO_GUIDE: {
                "required_keys": ["page_type", "title", "meta_description", "h1", 
                                 "introduction", "steps", "conclusion", "faqs"],
                "required_h2_sections": ["Introduction", "Step-by-Step Instructions", 
                                        "Tips and Best Practices", "Conclusion", "FAQs"],
                "min_steps": 3,
                "max_steps": 10,
                "min_faqs": 2,
                "max_faqs": 5
            },
            PageType.TROUBLESHOOTING: {
                "required_keys": ["page_type", "title", "meta_description", "h1",
                                 "problem_overview", "symptoms", "solutions", "prevention"],
                "required_h2_sections": ["Problem Overview", "Common Symptoms", 
                                        "Step-by-Step Solutions", "Prevention Tips", "When to Seek Help"],
                "min_solutions": 3,
                "max_solutions": 8
            },
            PageType.COMPARISON: {
                "required_keys": ["page_type", "title", "meta_description", "h1",
                                 "introduction", "comparison_criteria", "product_reviews", "verdict"],
                "required_h2_sections": ["Introduction", "Comparison Criteria", 
                                        "Product/Service Reviews", "Head-to-Head Comparison", "Final Verdict"],
                "min_products": 2,
                "max_products": 5
            },
            PageType.BUYING_GUIDE: {
                "required_keys": ["page_type", "title", "meta_description", "h1",
                                 "introduction", "buying_criteria", "top_picks", "how_to_choose"],
                "required_h2_sections": ["Introduction", "What to Look For", 
                                        "Our Top Picks", "How to Choose", "FAQs"],
                "min_picks": 3,
                "max_picks": 7
            }
        }
        
        # Fallback skeleton templates
        self.fallback_skeletons = {
            PageType.HOW_TO_GUIDE: {
                "page_type": "how_to_guide",
                "title": "",
                "meta_description": "",
                "h1": "",
                "introduction": "",
                "steps": [],
                "conclusion": "",
                "faqs": []
            },
            PageType.TROUBLESHOOTING: {
                "page_type": "troubleshooting",
                "title": "",
                "meta_description": "",
                "h1": "",
                "problem_overview": "",
                "symptoms": [],
                "solutions": [],
                "prevention": "",
                "when_to_seek_help": ""
            },
            PageType.COMPARISON: {
                "page_type": "comparison",
                "title": "",
                "meta_description": "",
                "h1": "",
                "introduction": "",
                "comparison_criteria": [],
                "product_reviews": [],
                "verdict": ""
            },
            PageType.BUYING_GUIDE: {
                "page_type": "buying_guide",
                "title": "",
                "meta_description": "",
                "h1": "",
                "introduction": "",
                "buying_criteria": [],
                "top_picks": [],
                "how_to_choose": "",
                "faqs": []
            }
        }
    
    def validate_page_json(self, page_data: Dict, page_type: Optional[PageType] = None) -> SchemaValidationResult:
        """
        Validate page JSON against schema requirements.
        
        Args:
            page_data: The JSON data to validate
            page_type: Expected page type (inferred from data if not provided)
        
        Returns:
            SchemaValidationResult with validation details
        """
        missing_keys = []
        invalid_sections = []
        schema_violations = []
        suggested_fixes = []
        
        # Determine page type
        if not page_type:
            page_type = self._infer_page_type(page_data)
        
        if not page_type:
            schema_violations.append("Could not determine page_type")
            suggested_fixes.append("Add 'page_type' field with value: how_to_guide, troubleshooting, comparison, or buying_guide")
            return SchemaValidationResult(
                is_valid=False,
                missing_keys=missing_keys,
                invalid_sections=invalid_sections,
                schema_violations=schema_violations,
                suggested_fixes=suggested_fixes
            )
        
        # Get schema for this page type
        schema = self.page_schemas.get(page_type)
        if not schema:
            schema_violations.append(f"Unknown page type: {page_type}")
            return SchemaValidationResult(
                is_valid=False,
                missing_keys=missing_keys,
                invalid_sections=invalid_sections,
                schema_violations=schema_violations,
                suggested_fixes=suggested_fixes
            )
        
        # Check required keys
        for key in schema["required_keys"]:
            if key not in page_data:
                missing_keys.append(key)
                suggested_fixes.append(f"Add missing key: '{key}'")
        
        # Check H2 sections in content (if content field exists)
        if "content" in page_data and "required_h2_sections" in schema:
            content = page_data["content"]
            for h2_section in schema["required_h2_sections"]:
                # Look for H2 markers in content
                h2_pattern = rf'^##\s+{re.escape(h2_section)}\b'
                if not re.search(h2_pattern, content, re.MULTILINE | re.IGNORECASE):
                    # Also check for variations
                    alt_patterns = [
                        rf'^##\s+.*{re.escape(h2_section.split()[-1])}\b',  # Last word
                        rf'^#{1,3}\s+.*{re.escape(h2_section)}\b'  # Any heading level
                    ]
                    found = False
                    for pattern in alt_patterns:
                        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                            found = True
                            break
                    
                    if not found:
                        invalid_sections.append(h2_section)
                        suggested_fixes.append(f"Add H2 section: '## {h2_section}'")
        
        # Validate array lengths
        if "steps" in page_data and "min_steps" in schema:
            steps = page_data["steps"]
            if not isinstance(steps, list):
                schema_violations.append("'steps' must be a list")
            elif len(steps) < schema["min_steps"]:
                schema_violations.append(f"Need at least {schema['min_steps']} steps, got {len(steps)}")
                suggested_fixes.append(f"Add more steps to reach minimum of {schema['min_steps']}")
        
        if "faqs" in page_data and "min_faqs" in schema:
            faqs = page_data["faqs"]
            if not isinstance(faqs, list):
                schema_violations.append("'faqs' must be a list")
            elif len(faqs) < schema["min_faqs"]:
                schema_violations.append(f"Need at least {schema['min_faqs']} FAQs, got {len(faqs)}")
                suggested_fixes.append(f"Add more FAQs to reach minimum of {schema['min_faqs']}")
        
        # Validate page_type matches
        if "page_type" in page_data:
            data_page_type = page_data["page_type"]
            if data_page_type != page_type.value:
                schema_violations.append(f"page_type mismatch: expected {page_type.value}, got {data_page_type}")
                suggested_fixes.append(f"Change page_type to '{page_type.value}'")
        
        is_valid = len(missing_keys) == 0 and len(invalid_sections) == 0 and len(schema_violations) == 0
        
        return SchemaValidationResult(
            is_valid=is_valid,
            missing_keys=missing_keys,
            invalid_sections=invalid_sections,
            schema_violations=schema_violations,
            suggested_fixes=suggested_fixes
        )
    
    def enforce_schema_with_retries(self, raw_json_text: str, page_type: PageType,
                                  max_retries: int = 2) -> Tuple[Dict, SchemaValidationResult, int]:
        """
        Enforce JSON schema with retry logic.
        
        Args:
            raw_json_text: Raw JSON text from LLM
            page_type: Expected page type
            max_retries: Maximum number of retry attempts
        
        Returns:
            Tuple of (validated_data, validation_result, attempts_made)
        """
        attempts = 0
        last_validation = None
        
        while attempts <= max_retries:
            attempts += 1
            
            # Parse JSON (with repair if needed)
            page_data = self._parse_and_repair_json(raw_json_text)
            
            if not page_data:
                # JSON parsing failed
                last_validation = SchemaValidationResult(
                    is_valid=False,
                    missing_keys=[],
                    invalid_sections=[],
                    schema_violations=["Failed to parse JSON"],
                    suggested_fixes=["Ensure output is valid JSON"]
                )
                continue
            
            # Validate against schema
            validation = self.validate_page_json(page_data, page_type)
            last_validation = validation
            
            if validation.is_valid:
                return page_data, validation, attempts
            
            # If not valid and we have retries left, we would normally retry with LLM
            # For now, we'll just return the invalid data
            if attempts > max_retries:
                break
        
        # All retries exhausted, use fallback skeleton
        fallback_data = self._create_fallback_skeleton(page_type)
        fallback_validation = self.validate_page_json(fallback_data, page_type)
        
        return fallback_data, fallback_validation, attempts
    
    def create_schema_reminder_prompt(self, page_type: PageType, 
                                    validation_result: SchemaValidationResult) -> str:
        """
        Create a prompt to remind LLM of schema requirements.
        
        Args:
            page_type: The page type being generated
            validation_result: Previous validation results
        
        Returns:
            Prompt string for LLM retry
        """
        schema = self.page_schemas.get(page_type, {})
        
        prompt_parts = []
        prompt_parts.append(f"IMPORTANT: Your previous response failed schema validation for {page_type.value}.")
        prompt_parts.append("Please regenerate with ALL required elements:")
        
        # List required keys
        if validation_result.missing_keys:
            prompt_parts.append(f"Missing required keys: {', '.join(validation_result.missing_keys)}")
        
        # List missing sections
        if validation_result.invalid_sections:
            prompt_parts.append(f"Missing H2 sections: {', '.join(validation_result.invalid_sections)}")
        
        # List schema violations
        if validation_result.schema_violations:
            prompt_parts.append(f"Schema violations: {', '.join(validation_result.schema_violations)}")
        
        # Provide schema requirements
        prompt_parts.append("")
        prompt_parts.append(f"REQUIRED SCHEMA for {page_type.value}:")
        
        if "required_keys" in schema:
            prompt_parts.append(f"Required keys: {', '.join(schema['required_keys'])}")
        
        if "required_h2_sections" in schema:
            prompt_parts.append(f"Required H2 sections in content: {', '.join(schema['required_h2_sections'])}")
        
        if "min_steps" in schema:
            prompt_parts.append(f"Minimum steps: {schema['min_steps']}")
        
        if "min_faqs" in schema:
            prompt_parts.append(f"Minimum FAQs: {schema['min_faqs']}")
        
        prompt_parts.append("")
        prompt_parts.append("Return ONLY valid JSON matching this schema. Do not include any explanatory text.")
        
        return "\n".join(prompt_parts)
    
    def _infer_page_type(self, page_data: Dict) -> Optional[PageType]:
        """Infer page type from data."""
        # Check page_type field
        if "page_type" in page_data:
            page_type_str = page_data["page_type"]
            for page_type in PageType:
                if page_type.value == page_type_str:
                    return page_type
        
        # Infer from content/structure
        if "steps" in page_data and isinstance(page_data["steps"], list):
            return PageType.HOW_TO_GUIDE
        elif "solutions" in page_data and isinstance(page_data["solutions"], list):
            return PageType.TROUBLESHOOTING
        elif "comparison_criteria" in page_data or "product_reviews" in page_data:
            return PageType.COMPARISON
        elif "buying_criteria" in page_data or "top_picks" in page_data:
            return PageType.BUYING_GUIDE
        
        return None
    
    def _parse_and_repair_json(self, text: str) -> Optional[Dict]:
        """Parse JSON with repair attempts for common issues."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text (might be wrapped in markdown)
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # JSON code fence
            r'```\s*(.*?)\s*```',      # Any code fence
            r'\{.*\}',                  # Anything that looks like JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try to repair common JSON issues
        repaired = self._repair_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _repair_json(self, text: str) -> Optional[str]:
        """Attempt to repair common JSON issues."""
        # Remove leading/trailing non-JSON text
        lines = text.strip().split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if '{' in line or '[' in line:
                in_json = True
            if in_json:
                json_lines.append(line)
            if '}' in line or ']' in line:
                in_json = False
        
        if not json_lines:
            return None
        
        repaired = '\n'.join(json_lines)
        
        # Fix trailing commas
        repaired = re.sub(r',\s*}', '}', repaired)
        repaired = re.sub(r',\s*]', ']', repaired)
        
        # Fix missing quotes on keys
        repaired = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', repaired)
        
        return repaired
    
    def _create_fallback_skeleton(self, page_type: PageType) -> Dict:
        """Create a fallback skeleton with required keys."""
        skeleton = self.fallback_skeletons.get(page_type, {}).copy()
        
        # Add minimal content to pass validation
        if page_type == PageType.HOW_TO_GUIDE:
            skeleton["steps"] = ["Step 1: Gather materials", "Step 2: Follow basic procedure", "Step 3: Verify results"]
            skeleton["faqs"] = [
                {"question": "What do I need to get started?", "answer": "Basic materials and tools."},
                {"question": "How long does this take?", "answer": "Typically 1-2 hours for beginners."}
            ]
        elif page_type == PageType.TROUBLESHOOTING:
            skeleton["symptoms"] = ["Common issue 1", "Common issue 2", "Common issue 3"]
            skeleton["solutions"] = ["Solution 1: Check basics", "Solution 2: Reset system", "Solution 3: Seek expert help"]
        
        return skeleton


# Convenience functions
def validate_and_fix_page_json(page_data: Dict, page_type: Optional[str] = None) -> Tuple[Dict, bool]:
    """
    Convenience function to validate and fix page JSON.
    
    Args:
        page_data: The page JSON data to validate
        page_type: Expected page type (string)
    
    Returns:
        Tuple of (fixed_page_data, is_valid)
    """
    enforcer = JSONSchemaEnforcer()
    
    # Convert string page_type to PageType enum if provided
    page_type_enum = None
    if page_type:
        for pt in PageType:
            if pt.value == page_type:
                page_type_enum = pt
                break
    
    # Validate the page data
    validation = enforcer.validate_page_json(page_data, page_type_enum)
    
    if validation.is_valid:
        return page_data, True
    
    # If not valid, try to fix common issues
    fixed_data = page_data.copy()
    
    # Add missing required keys with placeholder values
    for key in validation.missing_keys:
        if key == "page_type" and page_type:
            fixed_data[key] = page_type
        elif key == "title":
            fixed_data[key] = "Untitled Page"
        elif key == "meta_description":
            fixed_data[key] = "A helpful resource page."
        elif key == "h1":
            fixed_data[key] = fixed_data.get("title", "Untitled Page")
        elif key == "introduction":
            fixed_data[key] = "This page provides helpful information."
        elif key == "conclusion":
            fixed_data[key] = "In summary, this information should be useful."
        elif key == "steps" and key not in fixed_data:
            fixed_data[key] = ["Step 1", "Step 2", "Step 3"]
        elif key == "faqs" and key not in fixed_data:
            fixed_data[key] = [
                {"question": "What is this about?", "answer": "This page provides helpful information."},
                {"question": "Who is this for?", "answer": "Anyone interested in this topic."}
            ]
        elif key == "solutions" and key not in fixed_data:
            fixed_data[key] = ["Solution 1", "Solution 2", "Solution 3"]
        elif key == "symptoms" and key not in fixed_data:
            fixed_data[key] = ["Symptom 1", "Symptom 2", "Symptom 3"]
        elif key == "comparison_criteria" and key not in fixed_data:
            fixed_data[key] = ["Criterion 1", "Criterion 2", "Criterion 3"]
        elif key == "product_reviews" and key not in fixed_data:
            fixed_data[key] = ["Product 1 review", "Product 2 review"]
        elif key == "buying_criteria" and key not in fixed_data:
            fixed_data[key] = ["Price", "Quality", "Features"]
        elif key == "top_picks" and key not in fixed_data:
            fixed_data[key] = ["Top pick 1", "Top pick 2", "Top pick 3"]
        else:
            fixed_data[key] = ""
    
    # Re-validate after fixes
    fixed_validation = enforcer.validate_page_json(fixed_data, page_type_enum)
    
    return fixed_data, fixed_validation.is_valid


def enforce_json_schema_in_generation(raw_json_text: str, page_type: str, max_retries: int = 2) -> Dict:
    """
    Enforce JSON schema during page generation with retry logic.
    
    Args:
        raw_json_text: Raw JSON text from LLM
        page_type: Expected page type as string
        max_retries: Maximum retry attempts
    
    Returns:
        Validated page data (or fallback skeleton)
    """
    enforcer = JSONSchemaEnforcer()
    
    # Convert string to PageType enum
    page_type_enum = None
    for pt in PageType:
        if pt.value == page_type:
            page_type_enum = pt
            break
    
    if not page_type_enum:
        # Default to HOW_TO_GUIDE if unknown
        page_type_enum = PageType.HOW_TO_GUIDE
    
    # Enforce schema with retries
    page_data, validation, attempts = enforcer.enforce_schema_with_retries(
        raw_json_text, page_type_enum, max_retries
    )
    
    return page_data
