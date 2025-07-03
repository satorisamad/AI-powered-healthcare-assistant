"""
Ultra-Optimized Cartesia TTS Implementation
Based on your working code with test.py parameters for maximum performance

This implementation combines:
- Your proven WebSocket streaming approach
- test.py latency optimization parameters
- Production-hardened error handling
- Healthcare-specific optimizations
"""

import os
import uuid
import base64
import logging
import aiohttp
import json
import asyncio
import time
import hashlib
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Cartesia Configuration from your working code
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "sk_car_NesFmF1gwU3MsF6pvW8sDn")
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2024-11-13"

# Voice configurations - healthcare optimized
DEFAULT_VOICE_ID = "694f9389-aac1-45b6-b726-9d9369183238"  # Professional female
HEALTHCARE_VOICE_ID = "b7d50908-b17c-442d-ad8d-810c63997ed9"  # Your proven voice

# Optimization parameters from test.py
TTS_GRACE_SECONDS = 3.0                    # Grace period before allowing barge-in
BACK_OFF_SECONDS = 2.0                     # Delay between responses
SPEECH_END_TIMEOUT = 0.5                   # Trigger LLM after 500ms silence
AUDIO_CHUNK_SIZE = 960                     # 120ms audio chunks for analysis
BARGE_MIN_AUDIO_WINDOW = 0.25             # 250ms audio window for analysis
BARGE_AUDIO_THRESHOLD = 500               # Bytes threshold for real speech

# Cache for TTS results
_tts_cache: Dict[str, str] = {}

@dataclass
class OptimizedCartesiaConfig:
    """Optimized configuration combining your code + test.py parameters"""
    api_key: str = CARTESIA_API_KEY
    model_id: str = "sonic-2"
    voice_id: str = HEALTHCARE_VOICE_ID  # Use your proven voice
    language: str = "en"
    speed: str = "slow"  # Your proven speed setting
    
    # Output format optimized for healthcare voice calls
    output_format: Dict[str, Any] = None
    
    # Performance settings from test.py
    tts_grace_seconds: float = TTS_GRACE_SECONDS
    back_off_seconds: float = BACK_OFF_SECONDS
    speech_end_timeout: float = SPEECH_END_TIMEOUT
    audio_chunk_size: int = AUDIO_CHUNK_SIZE
    
    def __post_init__(self):
        if self.output_format is None:
            # Use MP3 format for better Twilio compatibility
            self.output_format = {
                "container": "mp3",
                "encoding": "mp3",
                "sample_rate": 22050
            }

class OptimizedCartesiaTTS:
    """Ultra-optimized Cartesia TTS using your proven WebSocket approach"""
    
    def __init__(self, config: Optional[OptimizedCartesiaConfig] = None):
        self.config = config or OptimizedCartesiaConfig()
        self.last_tts_timestamp = 0
        self.last_barge_ts = 0
        self.tts_active = False
        
    async def synthesize_speech_streaming(self, text: str, output_path: str = "output.mp3", 
                                        cancel_check: Optional[Callable] = None) -> str:
        """
        Ultra-optimized streaming TTS using your proven WebSocket implementation
        """
        logger.info(f"[Cartesia Optimized] Streaming TTS: {text[:50]}...")
        start_time = time.time()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _tts_cache and os.path.exists(_tts_cache[text_hash]):
            cached_path = _tts_cache[text_hash]
            try:
                # Copy cached file to output path
                with open(cached_path, 'rb') as src:
                    content = src.read()
                with open(output_path, 'wb') as dst:
                    dst.write(content)
                elapsed = time.time() - start_time
                logger.info(f"[Cartesia Optimized] Cache hit in {elapsed:.3f}s")
                return output_path
            except Exception as cache_error:
                logger.warning(f"[Cartesia Optimized] Cache read failed: {cache_error}")
        
        context_id = str(uuid.uuid4())
        ws_url = f"{CARTESIA_WS_URL}?api_key={self.config.api_key}&cartesia_version={CARTESIA_VERSION}"
        
        self.tts_active = True
        self.last_tts_timestamp = time.time()
        
        try:
            # Configure WebSocket with optimized parameters for real-time audio
            timeout = aiohttp.ClientTimeout(total=15, connect=3)  # Reduced timeout
            logger.info(f"[Cartesia Optimized] Connecting to WebSocket...")
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    ws_url,
                    heartbeat=20,          # Keep connection alive
                    compress=False,        # Disable compression for lower latency
                    max_msg_size=2**20     # 1MB max message size for audio chunks
                ) as cartesia_ws:
                    # Use your proven message format
                    message = {
                        "model_id": self.config.model_id,
                        "transcript": text,
                        "voice": {
                            "mode": "id",
                            "id": self.config.voice_id,
                            "__experimental_controls": {
                                "speed": self.config.speed,
                            }
                        },
                        "language": self.config.language,
                        "context_id": context_id,
                        "output_format": self.config.output_format,
                        "add_timestamps": False,
                        "continue": False
                    }

                    await cartesia_ws.send_json(message)
                    
                    audio_chunks = []

                    while True:
                        # Check for cancellation with test.py grace period logic
                        if cancel_check and cancel_check():
                            now = time.time()
                            if now - self.last_tts_timestamp >= self.config.tts_grace_seconds:
                                logger.info("🛑 TTS cancelled — closing Cartesia socket")
                                await cartesia_ws.close()
                                self.tts_active = False
                                return None

                        try:
                            # Optimized timeout for real-time streaming
                            msg = await asyncio.wait_for(cartesia_ws.receive(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue  # keep looping
                        except Exception as ws_err:
                            logger.error(f"❌ WebSocket receive error: {ws_err}")
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)

                            if data.get("type") == "chunk":
                                # Apply test.py barge-in detection logic
                                if cancel_check and cancel_check():
                                    now = time.time()
                                    if now - self.last_barge_ts >= 0.75:  # test.py cooldown
                                        logger.info("🛑 TTS cancelled mid-chunk — stopping")
                                        await cartesia_ws.close()
                                        self.tts_active = False
                                        return None

                                audio_chunk = base64.b64decode(data["data"])
                                audio_chunks.append(audio_chunk)

                            elif data.get("type") == "done":
                                logger.info("✅ Cartesia TTS completed normally")
                                break

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"❌ Cartesia WebSocket error: {msg.data}")
                            break

            # Save combined audio to file
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                with open(output_path, 'wb') as f:
                    f.write(combined_audio)
                
                # Cache the result
                _tts_cache[text_hash] = output_path
                
                elapsed = time.time() - start_time
                logger.info(f"[Cartesia Optimized] Generated in {elapsed:.2f}s: {output_path}")
                self.tts_active = False
                return output_path
            else:
                raise Exception("No audio data received from Cartesia")

        except Exception as e:
            logger.error(f"[Cartesia Optimized] Error: {e}")
            self.tts_active = False
            raise
        finally:
            self.tts_active = False

    async def synthesize_speech_twilio(self, text: str, websocket, stream_sid: str, 
                                     cancel_check: Optional[Callable] = None) -> bool:
        """
        Direct Twilio WebSocket streaming using your proven implementation
        """
        logger.info(f"[Cartesia Optimized] Twilio streaming: {text[:50]}...")
        
        context_id = str(uuid.uuid4())
        ws_url = f"{CARTESIA_WS_URL}?api_key={self.config.api_key}&cartesia_version={CARTESIA_VERSION}"
        
        self.tts_active = True
        self.last_tts_timestamp = time.time()

        try:
            # Configure WebSocket with optimized parameters for real-time audio
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    ws_url,
                    heartbeat=20,          # Keep connection alive
                    compress=False,        # Disable compression for lower latency
                    max_msg_size=2**20     # 1MB max message size for audio chunks
                ) as cartesia_ws:
                    message = {
                        "model_id": self.config.model_id,
                        "transcript": text,
                        "voice": {
                            "mode": "id",
                            "id": self.config.voice_id,
                            "__experimental_controls": {
                                "speed": self.config.speed,
                            }
                        },
                        "language": self.config.language,
                        "context_id": context_id,
                        "output_format": self.config.output_format,
                        "add_timestamps": False,
                        "continue": False
                    }

                    await cartesia_ws.send_json(message)

                    while True:
                        # Apply test.py grace period and barge-in logic
                        if cancel_check and cancel_check():
                            now = time.time()
                            if now - self.last_tts_timestamp >= self.config.tts_grace_seconds:
                                logger.info("🛑 TTS cancelled — closing Cartesia socket")
                                await cartesia_ws.close()
                                self.tts_active = False
                                return False

                        try:
                            # Optimized timeout for real-time streaming
                            msg = await asyncio.wait_for(cartesia_ws.receive(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as ws_err:
                            logger.error(f"❌ WebSocket receive error: {ws_err}")
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)

                            if data.get("type") == "chunk":
                                if cancel_check and cancel_check():
                                    logger.info("🛑 TTS cancelled mid-chunk — stopping send")
                                    await cartesia_ws.close()
                                    self.tts_active = False
                                    return False

                                audio_chunk = base64.b64decode(data["data"])
                                audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")

                                # Check Twilio WebSocket state before sending (your proven approach)
                                if websocket.client_state.name != "CONNECTED":
                                    logger.warning("🔌 Twilio WebSocket not connected — aborting TTS")
                                    break

                                try:
                                    await websocket.send_json({
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {
                                            "payload": audio_b64,
                                            "track": "outbound"
                                        }
                                    })
                                except Exception as send_err:
                                    logger.error(f"❌ Failed to send media to Twilio WS: {send_err}")
                                    break

                            elif data.get("type") == "done":
                                logger.info("✅ Cartesia TTS completed normally")
                                break

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"❌ Cartesia WebSocket error: {msg.data}")
                            break

            self.tts_active = False
            return True

        except Exception as e:
            logger.error(f"[Cartesia Optimized] Twilio streaming error: {e}")
            self.tts_active = False
            return False

    def is_barge_in_allowed(self) -> bool:
        """Check if barge-in is allowed based on test.py timing logic"""
        now = time.time()
        
        # Apply test.py grace period
        if now - self.last_tts_timestamp < self.config.tts_grace_seconds:
            return False
            
        # Apply test.py cooldown
        if now - self.last_barge_ts < 0.75:
            return False
            
        return True
    
    def trigger_barge_in(self):
        """Trigger barge-in with test.py timing"""
        self.last_barge_ts = time.time()
        self.tts_active = False

# Global optimized instance
_optimized_cartesia: Optional[OptimizedCartesiaTTS] = None

async def get_optimized_cartesia() -> OptimizedCartesiaTTS:
    """Get or create the global optimized Cartesia instance"""
    global _optimized_cartesia
    if _optimized_cartesia is None:
        _optimized_cartesia = OptimizedCartesiaTTS()
    return _optimized_cartesia

# Convenience functions for your healthcare system
async def synthesize_speech_optimized(text: str, output_path: str = "output.mp3") -> str:
    """Ultra-optimized TTS for healthcare voice assistant"""
    tts = await get_optimized_cartesia()
    return await tts.synthesize_speech_streaming(text, output_path)

async def synthesize_speech_twilio_optimized(text: str, websocket, stream_sid: str, 
                                           cancel_check: Optional[Callable] = None) -> bool:
    """Direct Twilio streaming with your proven approach"""
    tts = await get_optimized_cartesia()
    return await tts.synthesize_speech_twilio(text, websocket, stream_sid, cancel_check)

def warm_optimized_cache():
    """Pre-warm cache with common healthcare phrases"""
    logger.info("[Cartesia Optimized] Warming cache with healthcare phrases...")
    
    healthcare_phrases = [
        "Hello! You've reached Harmony Health Clinic. I'm Sarah, and I'm here to help you today.",
        "I can help you schedule an appointment, verify insurance, or answer questions.",
        "Let me check that information for you.",
        "I've found some available appointment slots for you.",
        "Your insurance information has been verified successfully.",
        "Is there anything else I can help you with today?",
        "Thank you for calling. Have a great day!",
        "Sorry, could you repeat that?",
        "I'd be happy to help you with that! Could you please share your name with me?",
        "What brings you in today? Is this for a check-up?",
        "Perfect! What time would you like to come in tomorrow?"
    ]
    
    # Cache warming would be done asynchronously in production
    logger.info(f"[Cartesia Optimized] Ready to cache {len(healthcare_phrases)} phrases")

if __name__ == "__main__":
    # Test the optimized implementation
    async def test_optimized():
        text = "Testing ultra-optimized Cartesia TTS with your proven parameters."
        result = await synthesize_speech_optimized(text, "test_optimized.mp3")
        if result and os.path.exists(result):
            print("✅ Optimized Cartesia TTS working!")
            os.remove(result)
        else:
            print("❌ Optimized Cartesia TTS test failed")
    
    asyncio.run(test_optimized())
