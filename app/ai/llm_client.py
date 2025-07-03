import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from environment variable or .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables or .env file.")

# Load system prompt from file (same as HealthAgent)
def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), 'healthcare_system_prompt.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load system prompt: {e}")
        return "You are a helpful, friendly, and professional AI front-desk assistant for a healthcare clinic. Your job is to help patients with appointment scheduling, insurance verification, and answering common clinic questions. Always ask one question at a time, clarify uncertainties, and keep responses concise and polite. Never provide medical advice or discuss sensitive topics."

SYSTEM_PROMPT = load_system_prompt()

class LLMClient:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using gpt-4o-mini for function calling support
        # Optimize for voice interactions
        self.voice_optimized_params = {
            "temperature": 0.2,  # More consistent for voice
            "max_tokens": 40,    # Shorter responses for voice
            "top_p": 0.8,       # More focused responses
        }
        # Ultra-fast mode for voice calls
        self.ultra_fast_params = {
            "temperature": 0.1,  # Maximum consistency
            "max_tokens": 20,    # Ultra-short responses
            "top_p": 0.6,       # Very focused
        }
        # Speed-optimized parameters for minimum latency
        self.speed_optimized_params = {
            "temperature": 0.05,  # Maximum consistency for speed
            "max_tokens": 25,     # Very short for speed
            "top_p": 0.6,        # Very focused
            "frequency_penalty": 0.2,  # Reduce repetition aggressively
            "presence_penalty": 0.2    # Encourage extreme conciseness
        }
        # Ultra-speed parameters for critical voice responses
        self.ultra_speed_params = {
            "temperature": 0.01,  # Minimal randomness
            "max_tokens": 20,     # Extremely short
            "top_p": 0.5,        # Very focused
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        }

    def query_llm(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict]] = None, system_prompt: Optional[str] = None, voice_mode: bool = False) -> str:
        import time
        llm_start_time = time.time()
        logging.info(f"[LLM] Called query_llm with prompt: {prompt} (voice_mode={voice_mode})")
        logging.info(f"[LATENCY] LLM query started at {llm_start_time:.3f}")

        try:
            # Use provided system prompt or fall back to default
            active_system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
            messages = [{"role": "system", "content": active_system_prompt}]
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            # Choose optimization parameters based on voice mode and tool usage
            if tools and voice_mode:
                # Use speed-optimized parameters for voice mode with tools
                optimization_params = {
                    "max_tokens": 80,   # Reduced for faster response
                    "temperature": 0.1, # Maximum consistency
                    "top_p": 0.7,      # Focused responses
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1
                }
                logging.info("[LLM] Using SPEED-OPTIMIZED voice parameters with tools: max_tokens=80")
            elif tools and not voice_mode:
                # Use adequate parameters for non-voice mode with tools
                optimization_params = {
                    "max_tokens": 120,  # Reduced from 150
                    "temperature": 0.2, # Faster consistency
                    "top_p": 0.8
                }
                logging.info("[LLM] Using optimized non-voice parameters with tools: max_tokens=120")
            elif voice_mode:
                # Use ultra-speed for voice mode without tools
                optimization_params = self.ultra_speed_params
                logging.info(f"[LLM] Using ULTRA-SPEED parameters: max_tokens={optimization_params['max_tokens']}")
            else:
                # Use standard voice-optimized for non-voice mode without tools
                optimization_params = self.voice_optimized_params
                logging.info(f"[LLM] Using voice-optimized parameters: max_tokens={optimization_params['max_tokens']}")

            # Prepare API call parameters with appropriate optimization
            api_params = {
                "model": self.model,
                "messages": messages,
                **optimization_params
            }

            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # First API call (may trigger function/tool call)
            api_call_start = time.time()
            logging.info(f"[LATENCY] Starting OpenAI API call at {api_call_start:.3f}")

            response = self.client.chat.completions.create(**api_params)

            api_call_end = time.time()
            api_call_duration = api_call_end - api_call_start
            logging.info(f"[LATENCY] OpenAI API call completed in {api_call_duration:.3f}s")

            message = response.choices[0].message
            if message.tool_calls:
                logging.info(f"🛠 Tool(s) requested by LLM: {[tc.function.name for tc in message.tool_calls]}")
                tool_start = time.time()
                result = self._handle_tool_calls(message, messages, tools, voice_mode)
                tool_end = time.time()
                logging.info(f"[LATENCY] Tool handling completed in {tool_end - tool_start:.3f}s")

                total_llm_time = time.time() - llm_start_time
                logging.info(f"[LATENCY] TOTAL LLM TIME: {total_llm_time:.3f}s (API: {api_call_duration:.3f}s, Tools: {tool_end - tool_start:.3f}s)")
                return result
            else:
                logging.info("📩 No tool calls made — using direct LLM completion.")
                ai_response = message.content.strip()
                logging.info(f"[LLM] Response: {ai_response}")

                total_llm_time = time.time() - llm_start_time
                logging.info(f"[LATENCY] TOTAL LLM TIME: {total_llm_time:.3f}s (API: {api_call_duration:.3f}s)")
                return ai_response

        except Exception as e:
            logging.error(f"[LLM] Error in query_llm: {e}")
            raise

    def _handle_tool_calls(self, message, messages: List[Dict], tools: List[Dict], voice_mode: bool = False) -> str:
        """Handle tool calls and return the final response."""
        import json
        from app.backend.scheduler import find_or_suggest_time, confirm_schedule
        from app.backend.insurance import verify_insurance
        from app.backend.faq import get_faq_answer
        from app.utils.date_utils import get_current_date_info

        function_messages = [message]

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name

            # Parse function arguments with error handling
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.error(f"[LLM] JSON parsing error for {function_name}: {e}")
                logging.error(f"[LLM] Raw arguments: {tool_call.function.arguments}")

                # Try to extract partial arguments or use defaults
                if function_name == "find_or_suggest_time":
                    # Extract time information from the raw string if possible
                    raw_args = tool_call.function.arguments
                    function_args = {"requested_time": raw_args.strip('"').strip("'")}
                else:
                    # Use empty args as fallback
                    function_args = {}

            logging.info(f"[LLM] Calling {function_name} with args: {function_args}")

            # Execute the appropriate function
            if function_name == "find_or_suggest_time":
                function_response = find_or_suggest_time(**function_args)
            elif function_name == "confirm_schedule":
                function_response = confirm_schedule(**function_args)
            elif function_name == "verify_insurance":
                function_response = verify_insurance(**function_args)
            elif function_name == "get_faq_answer":
                function_response = get_faq_answer(**function_args)
            elif function_name == "get_current_date_info":
                function_response = get_current_date_info(**function_args)
            else:
                function_response = f"Error: unknown function {function_name}"

            function_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })

        # Second API call with tool response, using appropriate parameters
        if voice_mode:
            # Use moderate parameters for voice mode (need coherent response)
            optimization_params = {
                "max_tokens": 150,  # Enough for a complete response
                "temperature": 0.3,
                "top_p": 0.8
            }
        else:
            optimization_params = self.voice_optimized_params

        second_response = self.client.chat.completions.create(
            model=self.model,
            messages=[*messages, *function_messages],
            **optimization_params
        )

        final_output = second_response.choices[0].message.content.strip()
        logging.info(f"[LLM] Response after tool calls: {final_output}")
        return final_output
