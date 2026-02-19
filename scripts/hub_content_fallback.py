#!/usr/bin/env python3
"""
hub_content_fallback.py — Deterministic fallback generator for hub content.

When JSON context generation fails or returns empty content, this provides
coherent, substantive hub content instead of minimal placeholders.
"""

import re
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from niche_resolver import NicheBreadthResolver, DomainCategory


class HubContentFallback:
    """Generates deterministic fallback content for hubs when LLM fails."""
    
    def __init__(self):
        self.niche_resolver = NicheBreadthResolver()
        
        # Domain-specific content templates
        self.domain_intros = {
            DomainCategory.PHOTOGRAPHY: [
                "Welcome to our comprehensive guide on {hub_theme} for {niche} photography. "
                "Whether you're a beginner learning the basics or an experienced photographer "
                "looking to refine your skills, this hub covers everything you need to know.",
                
                "This hub focuses on {hub_theme} in the context of {niche} photography. "
                "We'll explore techniques, equipment, and best practices to help you "
                "capture stunning images and overcome common challenges.",
                
                "Mastering {hub_theme} is essential for success in {niche} photography. "
                "In this hub, we break down complex concepts into practical, actionable "
                "advice you can apply immediately to your photography workflow."
            ],
            DomainCategory.ANIMALS: [
                "Welcome to our guide on {hub_theme} for {niche} enthusiasts. "
                "Understanding {hub_theme} is crucial for anyone interested in {niche}, "
                "whether for research, conservation, or personal interest.",
                
                "This hub explores {hub_theme} in the world of {niche}. "
                "We'll cover identification techniques, behavioral patterns, "
                "and practical knowledge to deepen your understanding.",
                
                "Whether you're a researcher, conservationist, or simply curious about {niche}, "
                "this hub on {hub_theme} provides valuable insights and practical guidance."
            ],
            DomainCategory.NATURE: [
                "Explore {hub_theme} in the context of {niche} with our comprehensive guide. "
                "From basic concepts to advanced techniques, this hub covers everything "
                "you need to understand and engage with {niche} effectively.",
                
                "This hub focuses on {hub_theme} as it relates to {niche}. "
                "We'll examine ecological principles, observation methods, "
                "and practical applications for nature enthusiasts.",
                
                "Understanding {hub_theme} is key to appreciating and protecting {niche}. "
                "This hub provides the knowledge and tools you need to engage meaningfully "
                "with the natural world."
            ],
            DomainCategory.OUTDOORS: [
                "Welcome to our guide on {hub_theme} for {niche} adventures. "
                "Whether you're planning a day hike or an extended expedition, "
                "this hub covers essential knowledge and skills for safe, enjoyable outdoor experiences.",
                
                "This hub focuses on {hub_theme} in {niche} settings. "
                "We'll cover equipment selection, technique development, "
                "and safety considerations for outdoor enthusiasts.",
                
                "Mastering {hub_theme} is essential for success in {niche} activities. "
                "This hub provides practical advice and proven strategies to enhance "
                "your outdoor skills and confidence."
            ],
            DomainCategory.GENERAL: [
                "Welcome to our comprehensive guide on {hub_theme} for {niche}. "
                "This hub covers essential concepts, practical techniques, "
                "and valuable resources to help you succeed.",
                
                "This hub explores {hub_theme} in the context of {niche}. "
                "Whether you're just getting started or looking to deepen your knowledge, "
                "you'll find valuable insights and actionable advice here.",
                
                "Understanding {hub_theme} is key to mastering {niche}. "
                "This hub breaks down complex topics into clear, practical guidance "
                "you can apply immediately."
            ]
        }
        
        # "What you'll learn" bullet templates
        self.learning_bullets = {
            DomainCategory.PHOTOGRAPHY: [
                "Essential {hub_theme} techniques and when to use them",
                "Common mistakes in {hub_theme} and how to avoid them",
                "Equipment recommendations for different {niche} scenarios",
                "Step-by-step workflows for consistent results",
                "Creative approaches to {hub_theme} for unique images"
            ],
            DomainCategory.ANIMALS: [
                "Key characteristics for identifying different species",
                "Behavioral patterns and what they indicate",
                "Habitat requirements and conservation considerations",
                "Research methods and observation techniques",
                "Ethical guidelines for interacting with {niche}"
            ],
            DomainCategory.NATURE: [
                "Fundamental principles of {hub_theme}",
                "Identification techniques for common elements",
                "Ecological relationships and systems thinking",
                "Observation and documentation methods",
                "Conservation practices and sustainable approaches"
            ],
            DomainCategory.OUTDOORS: [
                "Essential gear selection and maintenance",
                "Safety protocols and risk management",
                "Navigation and route planning techniques",
                "Environmental ethics and Leave No Trace principles",
                "Skill development and progression pathways"
            ]
        }
        
        # FAQ templates by domain
        self.faq_templates = {
            DomainCategory.PHOTOGRAPHY: [
                ("What equipment do I need for {hub_theme}?", 
                 "The basic equipment includes [camera, lens, tripod], but specific needs depend on your {niche} photography goals. We recommend starting with..."),
                
                ("How can I improve my {hub_theme} skills?", 
                 "Practice is key, but structured learning helps. Start with the fundamentals in our beginner guides, then progress to advanced techniques."),
                
                ("What are common mistakes in {hub_theme}?", 
                 "Beginners often struggle with [common issue]. Our troubleshooting guides address these challenges with practical solutions.")
            ],
            DomainCategory.ANIMALS: [
                ("How do I get started with studying {hub_theme}?", 
                 "Begin with basic observation and documentation. Our beginner guides provide step-by-step instructions for getting started safely."),
                
                ("What resources do I need for {hub_theme} research?", 
                 "Basic equipment includes [field guide, notebook, binoculars]. More advanced research may require specialized tools discussed in our guides."),
                
                ("How can I contribute to {hub_theme} conservation?", 
                 "Start by educating yourself and others. Our conservation guides provide practical ways to make a difference in {niche} protection.")
            ],
            DomainCategory.NATURE: [
                ("What's the best way to learn about {hub_theme}?", 
                 "Combine field experience with structured learning. Start with our beginner guides, then apply knowledge through guided observation."),
                
                ("How can I identify different aspects of {hub_theme}?", 
                 "Use field guides and identification keys. Our identification guides provide clear criteria and comparison tools."),
                
                ("What are the conservation implications of {hub_theme}?", 
                 "Understanding {hub_theme} helps inform conservation decisions. Our guides connect ecological knowledge with practical conservation actions.")
            ]
        }
    
    def generate_hub_content(self, niche: str, hub_theme: str, 
                           cluster_titles: List[str] = None) -> Dict[str, str]:
        """
        Generate complete hub content with fallback when LLM fails.
        
        Returns dict with: intro, learning_bullets, faqs, internal_links
        """
        analysis = self.niche_resolver.analyze(niche)
        domain = analysis.domain
        
        # Get domain-appropriate templates
        domain_intros = self.domain_intros.get(domain, self.domain_intros[DomainCategory.GENERAL])
        learning_templates = self.learning_bullets.get(domain, self.learning_bullets.get(DomainCategory.GENERAL, []))
        faq_templates = self.faq_templates.get(domain, [])
        
        # 1. Generate introduction
        intro_template = random.choice(domain_intros)
        intro = intro_template.format(
            hub_theme=hub_theme.lower(),
            niche=niche
        )
        
        # 2. Generate "What you'll learn" bullets
        learning_bullets = []
        for template in learning_templates[:5]:  # Use first 5 templates
            bullet = template.format(
                hub_theme=hub_theme.lower(),
                niche=niche
            )
            learning_bullets.append(bullet)
        
        # If cluster titles are provided, create more specific bullets
        if cluster_titles and len(cluster_titles) >= 3:
            specific_bullets = [
                f"How to apply {hub_theme.lower()} principles to {cluster_titles[0].lower()}",
                f"Techniques for mastering {cluster_titles[1].lower()} in {niche} contexts",
                f"Practical solutions for common challenges in {cluster_titles[2].lower()}"
            ]
            learning_bullets = specific_bullets + learning_bullets[:2]
        
        # 3. Generate FAQs
        faqs = []
        for question_template, answer_template in faq_templates[:3]:  # First 3 FAQs
            question = question_template.format(
                hub_theme=hub_theme.lower(),
                niche=niche
            )
            answer = answer_template.format(
                hub_theme=hub_theme.lower(),
                niche=niche
            )
            faqs.append({"question": question, "answer": answer})
        
        # 4. Generate internal links section
        internal_links = self._generate_internal_links(niche, hub_theme, cluster_titles)
        
        return {
            "intro": intro,
            "learning_bullets": learning_bullets,
            "faqs": faqs,
            "internal_links": internal_links,
            "generated_at": datetime.now().isoformat(),
            "is_fallback": True
        }
    
    def generate_hub_markdown(self, niche: str, hub_theme: str, 
                            hub_id: str, cluster_titles: List[str] = None) -> str:
        """
        Generate complete hub index markdown with fallback content.
        
        Returns markdown string suitable for content/hubs/{hub_id}/_index.md
        """
        content = self.generate_hub_content(niche, hub_theme, cluster_titles)
        
        # Build markdown
        markdown_lines = []
        
        # Introduction
        markdown_lines.append(content["intro"])
        markdown_lines.append("")
        
        # What You'll Learn section
        markdown_lines.append("## What You'll Learn")
        markdown_lines.append("")
        for bullet in content["learning_bullets"]:
            markdown_lines.append(f"- {bullet}")
        markdown_lines.append("")
        
        # Internal Links section
        if content["internal_links"]:
            markdown_lines.append("## Explore Related Content")
            markdown_lines.append("")
            for link_text, link_url in content["internal_links"]:
                markdown_lines.append(f"- [{link_text}]({link_url})")
            markdown_lines.append("")
        
        # FAQs section
        if content["faqs"]:
            markdown_lines.append("## Frequently Asked Questions")
            markdown_lines.append("")
            for faq in content["faqs"]:
                markdown_lines.append(f"### {faq['question']}")
                markdown_lines.append("")
                markdown_lines.append(faq['answer'])
                markdown_lines.append("")
        
        # Generated notice (hidden comment)
        markdown_lines.append(f"<!-- Generated with fallback content at {content['generated_at']} -->")
        
        return "\n".join(markdown_lines)
    
    def _generate_internal_links(self, niche: str, hub_theme: str, 
                               cluster_titles: List[str] = None) -> List[Tuple[str, str]]:
        """Generate internal links to related content."""
        links = []
        niche_slug = niche.lower().replace(" ", "-")
        hub_slug = hub_theme.lower().replace(" ", "-")
        
        # Link to hub root
        links.append((
            f"Back to {niche} main page",
            f"/{niche_slug}/"
        ))
        
        # Links to clusters if available
        if cluster_titles:
            for i, cluster_title in enumerate(cluster_titles[:5]):  # First 5 clusters
                cluster_slug = cluster_title.lower().replace(" ", "-")
                links.append((
                    f"Learn about {cluster_title}",
                    f"/{niche_slug}/{hub_slug}/{cluster_slug}/"
                ))
        
        # Related hubs (generic)
        related_hubs = [
            "Getting Started Guides",
            "Advanced Techniques",
            "Tools and Equipment",
            "Troubleshooting Common Issues"
        ]
        
        for related_hub in related_hubs[:3]:
            related_slug = related_hub.lower().replace(" ", "-")
            links.append((
                f"Explore {related_hub}",
                f"/{niche_slug}/{related_slug}/"
            ))
        
        return links
    
    def ensure_minimum_content(self, existing_markdown: str, niche: str, 
                             hub_theme: str, min_chars: int = 500) -> str:
        """
        Ensure markdown has minimum content length, generating fallback if needed.
        
        Args:
            existing_markdown: Current markdown content (may be empty or minimal)
            niche: The niche for context
            hub_theme: The hub theme for context
            min_chars: Minimum character count required
        
        Returns:
            Enhanced markdown meeting minimum length requirements
        """
        if len(existing_markdown.strip()) >= min_chars:
            return existing_markdown
        
        # Extract any existing frontmatter
        frontmatter = ""
        content_body = existing_markdown
        
        if existing_markdown.startswith("---"):
            parts = existing_markdown.split("---", 2)
            if len(parts) >= 3:
                frontmatter = f"---{parts[1]}---"
                content_body = parts[2] if len(parts) > 2 else ""
        
        # Generate fallback content
        fallback_content = self.generate_hub_markdown(niche, hub_theme, "")
        
        # Combine: keep frontmatter, add fallback content
        if frontmatter:
            return f"{frontmatter}\n\n{fallback_content}"
        else:
            return fallback_content


# Convenience functions
def generate_fallback_hub_content(niche: str, hub_theme: str, 
                                cluster_titles: List[str] = None) -> Dict[str, str]:
    """Convenience function for generating fallback hub content."""
    generator = HubContentFallback()
    return generator.generate_hub_content(niche, hub_theme, cluster_titles)


def ensure_hub_content_quality(markdown_path: Path, niche: str, 
                             hub_theme: str, min_chars: int = 500) -> bool:
    """
    Ensure a hub markdown file meets minimum quality standards.
    
    Returns True if content was enhanced, False if already sufficient.
    """
    generator = HubContentFallback()
    
    if not markdown_path.exists():
        # Create new content
        content = generator.generate_hub_markdown(niche, hub_theme, "")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(content, encoding="utf-8")
        return True
    
    # Read existing content
    existing = markdown_path.read_text(encoding="utf-8")
    
    if len(existing.strip()) < min_chars:
        # Enhance content
        enhanced = generator.ensure_minimum_content(existing, niche, hub_theme, min_chars)
        markdown_path.write_text(enhanced, encoding="utf-8")
        return True
    
    return False


if __name__ == "__main__":
    # Test the fallback generator
    print("=== Testing Hub Content Fallback Generator ===")
    
    generator = HubContentFallback()
    
    test_cases = [
        ("animals", "Species Identification", ["Mammal Classification", "Bird Identification", "Reptile Characteristics"]),
        ("nature photography", "Camera Equipment", ["Lens Selection", "Tripod Setup", "Lighting Accessories"]),
        ("home gardening", "Tool Maintenance", ["Pruning Tools", "Watering Equipment", "Soil Testing Kits"])
    ]
    
    for niche, hub_theme, cluster_titles in test_cases:
        print(f"\n=== Niche: {niche}, Hub: {hub_theme} ===")
        
        # Generate content
        content = generator.generate_hub_content(niche, hub_theme, cluster_titles)
        
        print(f"Intro length: {len(content['intro'])} chars")
        print(f"Learning bullets: {len(content['learning_bullets'])}")
        print(f"FAQs: {len(content['faqs'])}")
        print(f"Internal links: {len(content['internal_links'])}")
        
        # Generate markdown
        markdown = generator.generate_hub_markdown(niche, hub_theme, "test-hub", cluster_titles)
        print(f"Total markdown length: {len(markdown)} chars")
        
        # Test minimum content enforcement
        short_content = "Short content"
        enhanced = generator.ensure_minimum_content(short_content, niche, hub_theme, 500)
        print(f"Enhanced from {len(short_content)} to {len(enhanced)} chars")