#!/usr/bin/env python3
"""
Simple HTTP-based Cartesia TTS fallback
This avoids WebSocket complexity and uses simple HTTP requests
"""

import os
import requests
import logging
import time
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Cartesia API configuration
CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

# Simple cache for TTS responses
_simple_tts_cache = {}

async def synthesize_speech_simple_http(text: str, output_path: str) -> str:
    """
    Simple HTTP-based TTS using Cartesia API
    Fallback when WebSocket version fails
    """
    if not CARTESIA_API_KEY:
        raise Exception("CARTESIA_API_KEY not found in environment variables")
    
    start_time = time.time()
    
    # Create cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first
    if text_hash in _simple_tts_cache:
        cached_path = _simple_tts_cache[text_hash]
        if os.path.exists(cached_path):
            try:
                # Copy cached file to output path
                with open(cached_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                elapsed = time.time() - start_time
                logger.info(f"[Cartesia TTS] Cache hit in {elapsed:.3f}s")
                return output_path
            except Exception as cache_error:
                logger.warning(f"[Cartesia TTS] Cache read failed: {cache_error}")
    
    try:
        # Prepare request
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Content-Type": "application/json",
            "Cartesia-Version": "2025-04-16"
        }
        
        # Use Sonic-2 model with proven voice ID
        payload = {
            "model_id": "sonic-2",
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": "b7d50908-b17c-442d-ad8d-810c63997ed9"  # Proven voice from original system
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000
            },
            "language": "en"
        }
        
        logger.info(f"[Cartesia TTS] Making HTTP request for TTS...")
        
        # Make HTTP request with timeout
        response = requests.post(
            CARTESIA_API_URL,
            json=payload,
            headers=headers,
            timeout=10  # 10 second timeout
        )
        
        if response.status_code == 200:
            # Save audio data
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Cache the result
            _simple_tts_cache[text_hash] = output_path
            
            elapsed = time.time() - start_time
            logger.info(f"[Cartesia TTS] Generated in {elapsed:.2f}s: {output_path}")
            return output_path
        else:
            logger.error(f"[Cartesia TTS] HTTP error {response.status_code}: {response.text}")
            raise Exception(f"Cartesia API error: {response.status_code}")

    except requests.exceptions.Timeout:
        logger.error("[Cartesia TTS] Request timeout")
        raise Exception("Cartesia TTS timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"[Cartesia TTS] Request error: {e}")
        raise Exception(f"Cartesia TTS request failed: {e}")
    except Exception as e:
        logger.error(f"[Cartesia TTS] Unexpected error: {e}")
        raise

# Wrapper function to match the expected interface
async def synthesize_speech_fallback(text: str, output_path: str) -> str:
    """Async wrapper for the simple HTTP TTS"""
    return await synthesize_speech_simple_http(text, output_path)
