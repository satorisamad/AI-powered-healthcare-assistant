import json
import logging
import os
from typing import Dict, List, Optional

from app.ai.llm_client import LLMClient
from app.backend.faq import get_faq_answer
from app.backend.insurance import verify_insurance
from app.backend.scheduler import (
    confirm_schedule,
    find_or_suggest_time,
    validate_appointment_fields,
)
from app.utils.date_utils import (
    get_current_date_info,
    get_date_context_for_llm,
    parse_relative_date,
)
from app.voice.response_cache import get_response_cache

# Import calendar tools with error handling
try:
    from app.backend.calendar_toolkit import calendar_tools
except Exception as e:
    logging.warning(f"Could not import calendar_tools: {e}")
    calendar_tools = []

# Tool schema definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_or_suggest_time",
            "description": "MANDATORY: Check appointment availability and book it. Call this when you have patient name, date, and time, and you are ready to actually schedule the appointment. If you say you're 'scheduling', 'booking', or 'getting you set up', you MUST call this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_date": {"type": "string", "description": "Requested date in YYYY-MM-DD format"},
                    "requested_time": {"type": "string", "description": "Requested time in HH:MM format"},
                    "patient_name": {"type": "string", "description": "Patient's name for booking"}
                },
                "required": ["requested_date", "requested_time", "patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date_info",
            "description": "Get current date and time information. Use this when you need to know today's date, tomorrow's date, or calculate relative dates.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_insurance",
            "description": "Check if the provided insurance provider is accepted by the clinic. Only call this after gathering patient name, insurance provider, and what specifically needs to be verified through natural conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Insurance provider name"}
                },
                "required": ["provider"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_faq_answer",
            "description": "MANDATORY tool for answering ANY clinic-related questions including: hours, location, services, parking, payments, billing, insurance, accessibility, policies, staff info, patient portal, health classes, or any operational questions. Always call this for clinic information rather than guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question about clinic information, services, or policies"}
                },
                "required": ["question"]
            }
        }
    }
]

class HealthAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.history: List[Dict[str, str]] = []
        self.calendar_tools = calendar_tools
        self.response_cache = get_response_cache()

        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), 'healthcare_system_prompt.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except Exception as e:
            logging.error(f"Failed to load system prompt: {e}")
            self.system_prompt = "You are a helpful healthcare assistant."

    def set_history(self, history: List[Dict[str, str]]):
        """Set conversation history for this agent."""
        self.history = history

    def is_noisy_or_unclear(self, prompt: str) -> bool:
        """Check if the input is noisy or unclear."""
        if not prompt or len(prompt.strip()) < 3:
            return True

        # Check for common unclear patterns
        unclear_patterns = [
            "uh", "um", "er", "ah", "hmm", "...", "???",
            "what", "huh", "sorry", "pardon", "repeat"
        ]

        words = prompt.lower().split()
        if len(words) <= 2 and any(pattern in prompt.lower() for pattern in unclear_patterns):
            return True

        return False

    def is_emergency_situation(self, prompt: str) -> bool:
        """Check if the input indicates an emergency situation."""
        if not prompt:
            return False

        prompt_lower = prompt.lower()

        # Emergency keywords that require immediate 911 response
        emergency_keywords = [
            "emergency", "911", "can't breathe", "cant breathe", "chest pain", "heart attack", "stroke",
            "bleeding heavily", "unconscious", "overdose", "poisoning", "severe injury",
            "choking", "severe allergic reaction", "can't move", "cant move", "severe pain",
            "accident", "fell", "hit my head", "broken bone", "dying"
        ]

        # Mental health emergency keywords
        mental_health_keywords = [
            "suicidal", "want to die", "hurt myself", "kill myself", "end it all",
            "suicide", "want to hurt myself", "going to hurt myself"
        ]

        # Medical emergency phrases that need context
        contextual_emergency_phrases = [
            ("help me", ["can't breathe", "chest pain", "bleeding", "choking", "unconscious", "dying", "emergency"]),
            ("need help", ["can't breathe", "chest pain", "bleeding", "choking", "unconscious", "dying", "emergency"])
        ]

        # Check for direct emergency keywords
        for keyword in emergency_keywords + mental_health_keywords:
            if keyword in prompt_lower:
                return True

        # Check for contextual emergency phrases
        for phrase, context_words in contextual_emergency_phrases:
            if phrase in prompt_lower:
                # Only consider it an emergency if it has medical context
                if any(context in prompt_lower for context in context_words):
                    return True

        return False

    def process(self, prompt: str, voice_mode: bool = False) -> str:
        """Process user input and return response using OpenAI function calling."""
        logging.info(f"[Agent] Received user prompt: {prompt} (voice_mode={voice_mode})")

        try:
            first_interaction = len(self.history) == 0

            # Log the current conversation transcript
            logging.info(f"[Agent] Conversation history: {self.history}")

            # If this is the first user message and it's empty/unclear, just greet
            if first_interaction and (prompt.strip() == "" or self.is_noisy_or_unclear(prompt)):
                if voice_mode:
                    response = "Hello! You've reached Harmony Health Clinic. I'm Sarah, your AI health assistant - how can I help you today?"
                else:
                    response = "Hello! You've reached Harmony Health Clinic. I'm Sarah, your AI health assistant - how can I help you today?"
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": response})
                logging.info(f"[Agent] Assistant reply: {response}")
                return response

            # CRITICAL: Check for emergency situations first
            if self.is_emergency_situation(prompt):
                # Check for mental health emergency keywords
                mental_health_keywords = ["suicidal", "want to die", "hurt myself", "kill myself", "end it all", "suicide"]
                is_mental_health_emergency = any(keyword in prompt.lower() for keyword in mental_health_keywords)

                if is_mental_health_emergency:
                    response = "I'm very concerned about what you're telling me. Please contact emergency services at 911 immediately, or call the National Suicide Prevention Lifeline at 988. You deserve immediate professional help, and these services are available 24/7. Please reach out to them right now."
                else:
                    response = "This sounds like a medical emergency that requires immediate attention. Please hang up and call 911 right away, or have someone call for you. If you're having trouble breathing, chest pain, or any life-threatening symptoms, emergency services can help you much faster than I can. Please get emergency help immediately."

                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": response})
                logging.warning(f"[Agent] EMERGENCY DETECTED: {prompt[:100]}... - Provided emergency response")
                return response

            # Handle unclear input for ongoing conversations
            if not first_interaction and self.is_noisy_or_unclear(prompt):
                if voice_mode:
                    response = "Sorry, could you repeat that?"
                else:
                    response = "I'm sorry, I didn't catch that. Could you please rephrase or provide more details?"
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": response})
                return response

            # Check cache first for common responses (voice mode gets shorter cache time)
            cache_max_age = 1 if voice_mode else 24  # 1 hour for voice, 24 hours for text
            context = self.history[-2]['content'] if len(self.history) >= 2 else None
            cached_response = self.response_cache.get(prompt, context, cache_max_age)

            if cached_response:
                logging.info(f"[Agent] Using cached response for: {prompt[:50]}...")
                response = cached_response
            else:
                # Use OpenAI function calling for all other cases, with voice optimization
                response = self.llm.query_llm(prompt, self.history, tools=TOOLS, system_prompt=self.system_prompt, voice_mode=voice_mode)

                # Cache the response for future use
                self.response_cache.put(prompt, response, context)

            # The system prompt should handle greeting, so we don't need to add it here
            # The LLM will include greeting based on the system prompt instructions
            
            # Add to conversation history
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})
            
            logging.info(f"[Agent] Assistant reply: {response}")
            return response
            
        except Exception as e:
            logging.error(f"HealthAgent error: {str(e)}", exc_info=True)
            return "I'm sorry, I'm having trouble processing your request right now."
