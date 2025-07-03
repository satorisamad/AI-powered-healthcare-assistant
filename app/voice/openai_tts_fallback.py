#!/usr/bin/env python3
"""
OpenAI TTS Fallback - Reliable TTS for phone calls
"""

import os
import logging
import time
from openai import OpenAI

logger = logging.getLogger(__name__)

def synthesize_speech_openai(text: str, output_path: str) -> str:
    """
    Generate speech using OpenAI TTS - reliable fallback
    """
    logger.info(f"[OpenAI TTS] Starting synthesis: {text[:50]}...")
    start_time = time.time()
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",  # Fast model
            voice="alloy",  # Clear female voice
            input=text,
            response_format="mp3",
            speed=1.0
        )
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        elapsed = time.time() - start_time
        logger.info(f"[OpenAI TTS] Generated in {elapsed:.2f}s: {output_path}")
        
        return output_path
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[OpenAI TTS] Failed after {elapsed:.2f}s: {e}")
        raise

def synthesize_speech_simple_fallback(text: str, output_path: str) -> str:
    """
    Fallback TTS function that tries OpenAI first, then Cartesia
    """
    try:
        # Try OpenAI TTS first
        return synthesize_speech_openai(text, output_path)
    except Exception as openai_error:
        logger.warning(f"[TTS] OpenAI failed: {openai_error}")
        
        try:
            # Fallback to Cartesia
            from app.voice.cartesia_tts_simple import synthesize_speech_simple
            return synthesize_speech_simple(text, output_path)
        except Exception as cartesia_error:
            logger.error(f"[TTS] Both TTS services failed: OpenAI={openai_error}, Cartesia={cartesia_error}")
            raise Exception(f"All TTS services failed: {cartesia_error}")
