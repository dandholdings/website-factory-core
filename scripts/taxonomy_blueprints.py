"""
taxonomy_blueprints.py — Deterministic blueprint library for the website factory.

Each blueprint defines a niche family with:
- 8 hubs (each with label, description, related_hubs)
- 5 clusters per hub (each with label)
- keyword triggers for family selection

Usage:
    from taxonomy_blueprints import select_blueprint, BLUEPRINTS
    family_id = select_blueprint("home energy efficiency")
    bp = BLUEPRINTS[family_id]
"""

import re

# ---------------------------------------------------------------------------
# Family → keyword mapping (deterministic selection)
# ---------------------------------------------------------------------------
FAMILY_KEYWORDS = {
    "energy-efficiency": [
        "energy", "electricity", "heating", "cooling", "insulation",
        "bills", "solar", "power", "thermostat", "hvac", "heat pump",
        "kilowatt", "kwh", "utility", "efficiency", "renewable",
        "appliance energy", "draft", "sealing", "hot water",
    ],
    "home-systems": [
        "plumbing", "wiring", "electrical systems", "hvac", "ventilation",
        "structure", "detectors", "pipes", "circuits", "breakers",
        "drainage", "smoke alarm", "foundation", "maintenance",
        "water supply", "airflow", "filter",
    ],
    "decision-science": [
        "cognitive bias", "decision", "risk", "heuristics", "uncertainty",
        "behavioral economics", "judgment", "thinking", "rationality",
        "bias", "mental model", "probability", "procrastination",
        "self-control", "attention", "choice",
    ],
    "digital-habits": [
        "productivity", "screen time", "notifications", "focus",
        "distraction", "social media", "sleep tech", "digital",
        "phone", "apps", "scrolling", "deep work", "task management",
        "digital wellbeing", "blue light", "workflow",
    ],
}


def select_blueprint(niche: str, min_overlap_threshold: int = 4) -> str:
    """Select the best-matching family for a niche string with relevance guard.
    
    Enhanced with theme detection to prevent mismatches like "oceanliving → energy-efficiency".
    
    Args:
        niche: The niche/topic string
        min_overlap_threshold: Minimum keyword overlap required to select a family (increased to 4)
        
    Returns:
        family_id or "new-family" if no family meets the threshold
    """
    niche_lower = niche.lower().strip()
    niche_words = set(re.findall(r'\w+', niche_lower))
    
    # Detect niche theme
    niche_theme = _detect_niche_theme(niche_lower)
    
    # Calculate overlap scores
    scores = {}
    overlaps = {}
    
    for family_id, keywords in FAMILY_KEYWORDS.items():
        score = 0
        family_keyword_set = set()
        
        # Build set of all keyword words for this family
        for kw in keywords:
            if kw in niche_lower:
                score += len(kw)  # longer keyword matches score higher
            # Add individual words from keyword
            family_keyword_set.update(re.findall(r'\w+', kw.lower()))
        
        # Calculate word overlap
        word_overlap = len(niche_words & family_keyword_set)
        overlaps[family_id] = word_overlap
        scores[family_id] = score + (word_overlap * 10)  # Weight word overlap higher
    
    # Find best match
    best = max(scores, key=scores.get)
    best_overlap = overlaps.get(best, 0)
    
    # Apply relevance guard with higher threshold
    if best_overlap < min_overlap_threshold:
        # Not enough overlap - create new family instead of forcing wrong match
        return "new-family"
    
    # Also check if the best score is too low
    if scores[best] < 5:  # Very low score threshold
        return "new-family"
    
    # Theme mismatch check - prevent "ocean" niches from selecting "energy" families
    if niche_theme:
        family_theme = _get_family_theme(best)
        if family_theme and niche_theme != family_theme:
            # Themes don't match - create new family
            return "new-family"
    
    return best


def _detect_niche_theme(niche_lower: str) -> str:
    """Detect the broad theme of a niche to prevent category mismatches.
    
    Returns:
        Theme string like "energy", "home", "digital", "decision", or empty string
    """
    # Theme detection patterns
    theme_patterns = {
        "energy": ["energy", "electric", "power", "solar", "wind", "battery", "grid", "watt", "volt"],
        "home": ["home", "house", "residential", "property", "real estate", "building", "construction"],
        "digital": ["digital", "tech", "software", "app", "mobile", "computer", "internet", "online"],
        "decision": ["decision", "choice", "judgment", "thinking", "cognitive", "bias", "heuristic"],
        "ocean": ["ocean", "sea", "marine", "water", "fishing", "sailing", "coastal", "beach"],
        "health": ["health", "medical", "fitness", "wellness", "diet", "nutrition", "exercise"],
    }
    
    for theme, keywords in theme_patterns.items():
        for kw in keywords:
            if kw in niche_lower:
                return theme
    
    return ""


def _get_family_theme(family_id: str) -> str:
    """Get the theme of a blueprint family."""
    family_themes = {
        "energy-efficiency": "energy",
        "home-systems": "home",
        "digital-habits": "digital",
        "decision-science": "decision",
    }
    return family_themes.get(family_id, "")


# ---------------------------------------------------------------------------
# Blueprint definitions
# ---------------------------------------------------------------------------

BLUEPRINTS = {
    # ==================================================================
    # ENERGY EFFICIENCY
    # ==================================================================
    "energy-efficiency": {
        "family_id": "energy-efficiency",
        "family_label": "Energy Efficiency",
        "hubs": [
            {
                "id": "electricity-basics",
                "label": "Electricity Fundamentals",
                "description": "How electricity works in a home, from power generation to your outlets.",
                "related_hubs": ["energy-costs-billing", "appliance-energy", "home-envelope"],
                "clusters": [
                    {"id": "how-electricity-works", "label": "How Electricity Works"},
                    {"id": "power-vs-energy", "label": "Power vs Energy"},
                    {"id": "load-profiles", "label": "Load Profiles"},
                    {"id": "safety-basics", "label": "Safety Basics"},
                    {"id": "measurement-tools", "label": "Measurement Tools"},
                ],
            },
            {
                "id": "energy-costs-billing",
                "label": "Energy Costs & Bills",
                "description": "Understanding energy bills, tariffs, and what drives costs up or down.",
                "related_hubs": ["electricity-basics", "appliance-energy", "efficiency-upgrades"],
                "clusters": [
                    {"id": "billing-components", "label": "Billing Components"},
                    {"id": "tariffs-time-of-use", "label": "Tariffs & Time-of-Use"},
                    {"id": "bill-estimation", "label": "Bill Estimation"},
                    {"id": "cost-drivers", "label": "Cost Drivers"},
                    {"id": "planning-budgeting", "label": "Planning & Budgeting"},
                ],
            },
            {
                "id": "heating-cooling",
                "label": "Heating & Cooling Efficiency",
                "description": "How heating and cooling systems use energy, and what affects their efficiency.",
                "related_hubs": ["home-envelope", "efficiency-upgrades", "electricity-basics"],
                "clusters": [
                    {"id": "hvac-fundamentals", "label": "HVAC Fundamentals"},
                    {"id": "efficiency-ratings", "label": "Efficiency Ratings"},
                    {"id": "thermostat-control", "label": "Thermostat Control"},
                    {"id": "heat-loss-gain", "label": "Heat Loss & Gain"},
                    {"id": "comfort-vs-cost", "label": "Comfort vs Cost"},
                ],
            },
            {
                "id": "appliance-energy",
                "label": "Appliance Energy Use",
                "description": "Which appliances use the most energy and how usage patterns affect consumption.",
                "related_hubs": ["electricity-basics", "energy-costs-billing", "efficiency-upgrades"],
                "clusters": [
                    {"id": "high-draw-appliances", "label": "High-Draw Appliances"},
                    {"id": "always-on-loads", "label": "Always-On Loads"},
                    {"id": "usage-patterns", "label": "Usage Patterns"},
                    {"id": "efficiency-labels", "label": "Efficiency Labels"},
                    {"id": "measuring-appliances", "label": "Measuring Appliances"},
                ],
            },
            {
                "id": "home-envelope",
                "label": "Insulation, Airflow & Sealing",
                "description": "How the physical structure of a home affects energy retention and loss.",
                "related_hubs": ["heating-cooling", "efficiency-upgrades", "water-heating"],
                "clusters": [
                    {"id": "insulation-types", "label": "Insulation Types"},
                    {"id": "air-leaks-drafts", "label": "Air Leaks & Drafts"},
                    {"id": "windows-doors", "label": "Windows & Doors"},
                    {"id": "ventilation-balance", "label": "Ventilation Balance"},
                    {"id": "seasonal-effects", "label": "Seasonal Effects"},
                ],
            },
            {
                "id": "water-heating",
                "label": "Water Heating & Hot Water Use",
                "description": "How water heating works, what affects efficiency, and common patterns of use.",
                "related_hubs": ["home-envelope", "efficiency-upgrades", "energy-costs-billing"],
                "clusters": [
                    {"id": "water-heater-types", "label": "Water Heater Types"},
                    {"id": "hot-water-demand", "label": "Hot Water Demand"},
                    {"id": "heat-loss-pipes", "label": "Heat Loss in Pipes"},
                    {"id": "temperature-settings", "label": "Temperature Settings"},
                    {"id": "efficiency-options", "label": "Efficiency Options"},
                ],
            },
            {
                "id": "efficiency-upgrades",
                "label": "Efficiency Upgrades & Tradeoffs",
                "description": "Common efficiency improvements, their costs, payback periods, and realistic expectations.",
                "related_hubs": ["appliance-energy", "heating-cooling", "renewables-home"],
                "clusters": [
                    {"id": "upgrade-priorities", "label": "Upgrade Priorities"},
                    {"id": "payback-tradeoffs", "label": "Payback & Tradeoffs"},
                    {"id": "common-upgrades", "label": "Common Upgrades"},
                    {"id": "behavior-vs-hardware", "label": "Behavior vs Hardware"},
                    {"id": "mistakes-myths", "label": "Mistakes & Myths"},
                ],
            },
            {
                "id": "renewables-home",
                "label": "Home Renewables & Storage",
                "description": "Solar panels, battery storage, and how renewable energy integrates with a home.",
                "related_hubs": ["efficiency-upgrades", "electricity-basics", "energy-costs-billing"],
                "clusters": [
                    {"id": "solar-basics", "label": "Solar Basics"},
                    {"id": "battery-storage", "label": "Battery Storage"},
                    {"id": "self-consumption", "label": "Self-Consumption"},
                    {"id": "grid-interaction", "label": "Grid Interaction"},
                    {"id": "limitations-tradeoffs", "label": "Limitations & Tradeoffs"},
                ],
            },
        ],
    },

    # ==================================================================
    # HOME SYSTEMS
    # ==================================================================
    "home-systems": {
        "family_id": "home-systems",
        "family_label": "Home Systems",
        "hubs": [
            {
                "id": "plumbing",
                "label": "Plumbing Basics",
                "description": "How water flows through a home, from supply pipes to drains.",
                "related_hubs": ["water-systems", "maintenance-basics", "structure"],
                "clusters": [
                    {"id": "flow-pressure", "label": "Flow & Pressure"},
                    {"id": "fixtures-drains", "label": "Fixtures & Drains"},
                    {"id": "hot-vs-cold", "label": "Hot vs Cold"},
                    {"id": "common-issues-explained", "label": "Common Issues Explained"},
                    {"id": "materials-components", "label": "Materials & Components"},
                ],
            },
            {
                "id": "electrical",
                "label": "Home Electrical Systems",
                "description": "How home wiring, circuits, and electrical components work together.",
                "related_hubs": ["safety-systems", "maintenance-basics", "hvac"],
                "clusters": [
                    {"id": "circuits-breakers", "label": "Circuits & Breakers"},
                    {"id": "power-capacity", "label": "Power Capacity"},
                    {"id": "wiring-basics", "label": "Wiring Basics"},
                    {"id": "outlets-switches", "label": "Outlets & Switches"},
                    {"id": "safety-basics", "label": "Safety Basics"},
                ],
            },
            {
                "id": "hvac",
                "label": "Heating & Cooling Systems",
                "description": "How HVAC systems regulate temperature and air quality in a home.",
                "related_hubs": ["ventilation", "electrical", "maintenance-basics"],
                "clusters": [
                    {"id": "system-types", "label": "System Types"},
                    {"id": "heat-transfer", "label": "Heat Transfer"},
                    {"id": "controls-thermostats", "label": "Controls & Thermostats"},
                    {"id": "efficiency-concepts", "label": "Efficiency Concepts"},
                    {"id": "comfort-humidity", "label": "Comfort & Humidity"},
                ],
            },
            {
                "id": "ventilation",
                "label": "Ventilation & Indoor Air",
                "description": "How fresh air enters, moves through, and exits a home.",
                "related_hubs": ["hvac", "structure", "safety-systems"],
                "clusters": [
                    {"id": "fresh-air-basics", "label": "Fresh Air Basics"},
                    {"id": "exhaust-systems", "label": "Exhaust Systems"},
                    {"id": "airflow-pressure", "label": "Airflow & Pressure"},
                    {"id": "filters-filtration", "label": "Filters & Filtration"},
                    {"id": "indoor-pollutants", "label": "Indoor Pollutants"},
                ],
            },
            {
                "id": "structure",
                "label": "Structural Basics",
                "description": "The physical framework of a home: foundations, walls, and load paths.",
                "related_hubs": ["plumbing", "ventilation", "maintenance-basics"],
                "clusters": [
                    {"id": "load-bearing-basics", "label": "Load-Bearing Basics"},
                    {"id": "materials", "label": "Materials"},
                    {"id": "moisture-mold-risk", "label": "Moisture & Mold Risk"},
                    {"id": "foundations", "label": "Foundations"},
                    {"id": "building-envelope", "label": "Building Envelope"},
                ],
            },
            {
                "id": "water-systems",
                "label": "Water Supply & Drainage",
                "description": "How water enters a home, gets distributed, and leaves through drainage.",
                "related_hubs": ["plumbing", "maintenance-basics", "safety-systems"],
                "clusters": [
                    {"id": "supply-lines", "label": "Supply Lines"},
                    {"id": "drainage-sewage", "label": "Drainage & Sewage"},
                    {"id": "water-quality-basics", "label": "Water Quality Basics"},
                    {"id": "hot-water-distribution", "label": "Hot Water Distribution"},
                    {"id": "pressure-regulation", "label": "Pressure Regulation"},
                ],
            },
            {
                "id": "safety-systems",
                "label": "Safety Systems & Detectors",
                "description": "Smoke alarms, CO detectors, and other safety devices in a home.",
                "related_hubs": ["electrical", "ventilation", "maintenance-basics"],
                "clusters": [
                    {"id": "smoke-alarms", "label": "Smoke Alarms"},
                    {"id": "co-alarms", "label": "CO Alarms"},
                    {"id": "electrical-safety-devices", "label": "Electrical Safety Devices"},
                    {"id": "fire-safety-basics", "label": "Fire Safety Basics"},
                    {"id": "emergency-preparedness", "label": "Emergency Preparedness"},
                ],
            },
            {
                "id": "maintenance-basics",
                "label": "Maintenance Fundamentals",
                "description": "Understanding how home systems age, what to inspect, and when help is needed.",
                "related_hubs": ["plumbing", "electrical", "hvac"],
                "clusters": [
                    {"id": "inspection-mindset", "label": "Inspection Mindset"},
                    {"id": "seasonal-checks", "label": "Seasonal Checks"},
                    {"id": "lifespan-expectations", "label": "Lifespan Expectations"},
                    {"id": "when-to-call-help", "label": "When to Call Help"},
                    {"id": "myths-misconceptions", "label": "Myths & Misconceptions"},
                ],
            },
        ],
    },

    # ==================================================================
    # DECISION SCIENCE
    # ==================================================================
    "decision-science": {
        "family_id": "decision-science",
        "family_label": "Decision Science",
        "hubs": [
            {
                "id": "decision-foundations",
                "label": "Foundations of Decisions",
                "description": "How decisions form, what shapes them, and the basic mechanics of choosing.",
                "related_hubs": ["cognitive-biases", "judgment-heuristics", "habits-self-control"],
                "clusters": [
                    {"id": "how-decisions-form", "label": "How Decisions Form"},
                    {"id": "goals-tradeoffs", "label": "Goals & Tradeoffs"},
                    {"id": "emotion-reason", "label": "Emotion & Reason"},
                    {"id": "feedback-learning", "label": "Feedback & Learning"},
                    {"id": "choice-architecture", "label": "Choice Architecture"},
                ],
            },
            {
                "id": "cognitive-biases",
                "label": "Cognitive Biases",
                "description": "Systematic patterns in thinking that affect judgment and decision-making.",
                "related_hubs": ["decision-foundations", "judgment-heuristics", "social-influence"],
                "clusters": [
                    {"id": "biases-of-belief", "label": "Biases of Belief"},
                    {"id": "biases-of-memory", "label": "Biases of Memory"},
                    {"id": "biases-of-perception", "label": "Biases of Perception"},
                    {"id": "biases-of-evaluation", "label": "Biases of Evaluation"},
                    {"id": "debiasing-concepts", "label": "Debiasing Concepts"},
                ],
            },
            {
                "id": "risk-uncertainty",
                "label": "Risk & Uncertainty",
                "description": "How people perceive and respond to risk, probability, and the unknown.",
                "related_hubs": ["decision-foundations", "cognitive-biases", "long-term-thinking"],
                "clusters": [
                    {"id": "risk-perception", "label": "Risk Perception"},
                    {"id": "probability-intuition", "label": "Probability & Intuition"},
                    {"id": "loss-aversion", "label": "Loss Aversion"},
                    {"id": "uncertainty-tolerance", "label": "Uncertainty Tolerance"},
                    {"id": "expected-value-basics", "label": "Expected Value Basics"},
                ],
            },
            {
                "id": "judgment-heuristics",
                "label": "Heuristics & Shortcuts",
                "description": "Mental shortcuts people use to make quick judgments and their effects.",
                "related_hubs": ["cognitive-biases", "decision-foundations", "attention-information"],
                "clusters": [
                    {"id": "availability-representativeness", "label": "Availability & Representativeness"},
                    {"id": "anchoring-adjustment", "label": "Anchoring & Adjustment"},
                    {"id": "rule-of-thumb-thinking", "label": "Rule-of-Thumb Thinking"},
                    {"id": "mental-models", "label": "Mental Models"},
                    {"id": "confidence-calibration", "label": "Confidence & Calibration"},
                ],
            },
            {
                "id": "social-influence",
                "label": "Social Influence & Group Thinking",
                "description": "How other people, groups, and social norms shape individual decisions.",
                "related_hubs": ["cognitive-biases", "attention-information", "habits-self-control"],
                "clusters": [
                    {"id": "conformity-norms", "label": "Conformity & Norms"},
                    {"id": "authority-influence", "label": "Authority & Influence"},
                    {"id": "social-proof", "label": "Social Proof"},
                    {"id": "group-polarization", "label": "Group Polarization"},
                    {"id": "identity-beliefs", "label": "Identity & Beliefs"},
                ],
            },
            {
                "id": "attention-information",
                "label": "Attention & Information Overload",
                "description": "How limited attention and information volume affect decision quality.",
                "related_hubs": ["judgment-heuristics", "social-influence", "long-term-thinking"],
                "clusters": [
                    {"id": "limited-attention", "label": "Limited Attention"},
                    {"id": "information-overload", "label": "Information Overload"},
                    {"id": "signal-vs-noise", "label": "Signal vs Noise"},
                    {"id": "misinformation-patterns", "label": "Misinformation Patterns"},
                    {"id": "media-diets", "label": "Media Diets"},
                ],
            },
            {
                "id": "habits-self-control",
                "label": "Habits, Motivation & Self-Control",
                "description": "How habits form, what drives motivation, and how self-control works.",
                "related_hubs": ["decision-foundations", "long-term-thinking", "social-influence"],
                "clusters": [
                    {"id": "habit-loops", "label": "Habit Loops"},
                    {"id": "motivation-reward", "label": "Motivation & Reward"},
                    {"id": "procrastination-patterns", "label": "Procrastination Patterns"},
                    {"id": "impulses-willpower", "label": "Impulses & Willpower"},
                    {"id": "environment-design", "label": "Environment Design"},
                ],
            },
            {
                "id": "long-term-thinking",
                "label": "Long-Term vs Short-Term Thinking",
                "description": "How time horizons, planning, and future thinking affect decisions.",
                "related_hubs": ["risk-uncertainty", "habits-self-control", "decision-foundations"],
                "clusters": [
                    {"id": "time-discounting", "label": "Time Discounting"},
                    {"id": "planning-fallacy", "label": "Planning Fallacy"},
                    {"id": "goal-setting", "label": "Goal Setting"},
                    {"id": "regret-anticipation", "label": "Regret & Anticipation"},
                    {"id": "systems-vs-goals", "label": "Systems vs Goals"},
                ],
            },
        ],
    },

    # ==================================================================
    # DIGITAL HABITS
    # ==================================================================
    "digital-habits": {
        "family_id": "digital-habits",
        "family_label": "Digital Habits",
        "hubs": [
            {
                "id": "attention-focus",
                "label": "Attention & Focus",
                "description": "How attention works, what disrupts it, and how focus is sustained or lost.",
                "related_hubs": ["digital-distraction", "workflows-systems", "digital-wellbeing"],
                "clusters": [
                    {"id": "focus-basics", "label": "Focus Basics"},
                    {"id": "context-switching", "label": "Context Switching"},
                    {"id": "deep-work-mechanics", "label": "Deep Work Mechanics"},
                    {"id": "cognitive-load", "label": "Cognitive Load"},
                    {"id": "attention-training", "label": "Attention Training"},
                ],
            },
            {
                "id": "digital-distraction",
                "label": "Distraction & Interruptions",
                "description": "What causes digital distractions, how they affect focus, and common patterns.",
                "related_hubs": ["attention-focus", "notification-loops", "social-platforms"],
                "clusters": [
                    {"id": "interruptions", "label": "Interruptions"},
                    {"id": "multitasking-myths", "label": "Multitasking Myths"},
                    {"id": "device-friction", "label": "Device Friction"},
                    {"id": "environment-design", "label": "Environment Design"},
                    {"id": "workplace-distraction", "label": "Workplace Distraction"},
                ],
            },
            {
                "id": "screen-time",
                "label": "Screen Time Patterns",
                "description": "How screen time is measured, what patterns look like, and what the data shows.",
                "related_hubs": ["digital-wellbeing", "sleep-recovery", "social-platforms"],
                "clusters": [
                    {"id": "measurement-tracking", "label": "Measurement & Tracking"},
                    {"id": "patterns-triggers", "label": "Patterns & Triggers"},
                    {"id": "passive-vs-active-use", "label": "Passive vs Active Use"},
                    {"id": "age-context-differences", "label": "Age & Context Differences"},
                    {"id": "tradeoffs-balance", "label": "Tradeoffs & Balance"},
                ],
            },
            {
                "id": "notification-loops",
                "label": "Notifications & Reward Loops",
                "description": "How notifications are designed, how reward mechanics work, and their effects.",
                "related_hubs": ["digital-distraction", "attention-focus", "social-platforms"],
                "clusters": [
                    {"id": "reward-mechanics", "label": "Reward Mechanics"},
                    {"id": "notification-design", "label": "Notification Design"},
                    {"id": "alerts-vs-importance", "label": "Alerts vs Importance"},
                    {"id": "anxiety-urgency", "label": "Anxiety & Urgency"},
                    {"id": "habit-formation", "label": "Habit Formation"},
                ],
            },
            {
                "id": "workflows-systems",
                "label": "Productivity Systems",
                "description": "How task management, planning, and workflow systems are structured.",
                "related_hubs": ["attention-focus", "digital-wellbeing", "digital-distraction"],
                "clusters": [
                    {"id": "task-management", "label": "Task Management"},
                    {"id": "planning-review", "label": "Planning & Review"},
                    {"id": "prioritization", "label": "Prioritization"},
                    {"id": "tools-frameworks", "label": "Tools & Frameworks"},
                    {"id": "automation-basics", "label": "Automation Basics"},
                ],
            },
            {
                "id": "social-platforms",
                "label": "Social Media Behavior",
                "description": "How social platforms shape behavior, consumption patterns, and identity.",
                "related_hubs": ["notification-loops", "digital-distraction", "digital-wellbeing"],
                "clusters": [
                    {"id": "feeds-algorithms", "label": "Feeds & Algorithms"},
                    {"id": "comparison-status", "label": "Comparison & Status"},
                    {"id": "community-identity", "label": "Community & Identity"},
                    {"id": "sharing-validation", "label": "Sharing & Validation"},
                    {"id": "content-consumption", "label": "Content Consumption"},
                ],
            },
            {
                "id": "digital-wellbeing",
                "label": "Digital Wellbeing Concepts",
                "description": "Broad concepts around living with technology in a sustainable way.",
                "related_hubs": ["attention-focus", "screen-time", "sleep-recovery"],
                "clusters": [
                    {"id": "boundaries", "label": "Boundaries"},
                    {"id": "digital-minimalism", "label": "Digital Minimalism"},
                    {"id": "values-alignment", "label": "Values Alignment"},
                    {"id": "sustainable-habits", "label": "Sustainable Habits"},
                    {"id": "misconceptions", "label": "Misconceptions"},
                ],
            },
            {
                "id": "sleep-recovery",
                "label": "Sleep, Recovery & Technology",
                "description": "How technology intersects with sleep patterns, rest, and recovery.",
                "related_hubs": ["digital-wellbeing", "screen-time", "notification-loops"],
                "clusters": [
                    {"id": "blue-light-context", "label": "Blue Light in Context"},
                    {"id": "bedtime-routines", "label": "Bedtime Routines"},
                    {"id": "late-night-scrolling", "label": "Late-Night Scrolling"},
                    {"id": "recovery-time", "label": "Recovery Time"},
                    {"id": "device-in-bedroom", "label": "Device in Bedroom"},
                ],
            },
        ],
    },
}
