# Clinic FAQ logic

import json
import logging
from typing import Optional

FAQ_FILE = "data/faq.json"

logger = logging.getLogger(__name__)


def get_faq_answer(question: str) -> Optional[str]:
    """
    Return answer to a clinic FAQ if found using flexible keyword matching.
    """
    logger.info(f"[FAQ] Looking up answer for question: {question}")
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)["faqs"]

    question_lower = question.lower()

    # First try exact substring matching (original logic)
    for faq in faqs:
        if question_lower in faq["question"].lower() or faq["question"].lower() in question_lower:
            logger.info(f"[FAQ] Exact match found: {faq['question']} -> {faq['answer']}")
            return faq["answer"]

    # Then try keyword-based matching for better flexibility
    question_keywords = set(question_lower.split())

    # Special keyword mappings for common variations with priority scoring
    priority_matches = [
        # High priority - specific combinations
        (['pay', 'bill', 'online'], ['pay', 'bill', 'online']),
        (['parking', 'facility'], ['parking']),
        (['health', 'education', 'class'], ['health', 'education']),
        (['wheelchair', 'access'], ['wheelchair', 'access']),
        (['clinic', 'hours'], ['hours']),
        (['services', 'offer'], ['services']),
    ]

    # Check high-priority specific matches first
    for question_keywords_needed, faq_keywords_needed in priority_matches:
        if all(keyword in question_lower for keyword in question_keywords_needed):
            for faq in faqs:
                if all(keyword in faq["question"].lower() for keyword in faq_keywords_needed):
                    logger.info(f"[FAQ] Priority match found: {faq['question']} -> {faq['answer']}")
                    return faq["answer"]

    # Continue with general matching
    for faq in faqs:
        faq_keywords = set(faq["question"].lower().split())

        # Check for significant keyword overlap
        common_keywords = question_keywords.intersection(faq_keywords)

        # General keyword mappings for broader matches
        keyword_mappings = {
            'parking': ['parking', 'park'],
            'payment': ['pay', 'bill', 'payment', 'billing'],
            'online': ['online', 'portal', 'website'],
            'education': ['education', 'class', 'classes', 'workshop', 'workshops'],
            'health': ['health', 'medical'],
            'hours': ['hours', 'time', 'open', 'closed'],
            'location': ['location', 'address', 'where'],
            'insurance': ['insurance', 'coverage', 'plan'],
            'appointment': ['appointment', 'schedule', 'booking'],
            'wheelchair': ['wheelchair', 'accessible', 'access', 'disability']
        }

        # Check for semantic matches
        for concept, variations in keyword_mappings.items():
            if any(var in question_lower for var in variations) and concept in faq["question"].lower():
                logger.info(f"[FAQ] Semantic match found: {faq['question']} -> {faq['answer']}")
                return faq["answer"]

        # If we have 2+ common meaningful keywords, consider it a match
        meaningful_words = common_keywords - {'do', 'you', 'have', 'is', 'are', 'can', 'i', 'the', 'a', 'an', 'for', 'to', 'of', 'in', 'on', 'at', 'with'}
        if len(meaningful_words) >= 2:
            logger.info(f"[FAQ] Keyword match found: {faq['question']} -> {faq['answer']}")
            return faq["answer"]

    logger.info("[FAQ] No match found for question.")
    return None
