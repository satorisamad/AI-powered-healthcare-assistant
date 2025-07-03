"""
Ultra-Optimized Cartesia TTS Implementation for Healthcare Voice Assistant

This module provides a high-performance Cartesia TTS implementation with:
- Chunked text processing for faster response times
- Parallel processing capabilities
- Advanced caching with LRU eviction
- Parameter tuning for minimal latency
- Streaming support for real-time applications
"""

import asyncio
import hashlib
import logging
import os
import time
import aiohttp
import aiofiles
import concurrent.futures
import threading
import requests
import websockets
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Cartesia API Configuration - Updated to match your curl example
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")  # Remove hardcoded key - use .env file
CARTESIA_BASE_URL = "https://api.cartesia.ai/tts/bytes"  # Updated endpoint
CARTESIA_VERSION = "2024-06-10"  # Updated to match your curl example

# Voice configuration - Using your specified voice ID
DEFAULT_VOICE_ID = "bf0a246a-8642-498a-9950-80c35e9276b5"  # Your specified voice ID
HEALTHCARE_VOICE_ID = "bf0a246a-8642-498a-9950-80c35e9276b5"  # Same voice for consistency
MODERN_VOICE_BACKUP = "bf0a246a-8642-498a-9950-80c35e9276b5"  # Same voice for consistency

# Audio format configuration - Updated to match your curl example for better performance
DEFAULT_OUTPUT_FORMAT = {
    "container": "wav",      # Changed from mp3 to wav as per your curl
    "encoding": "pcm_f32le", # Added encoding as per your curl
    "sample_rate": 44100     # Keep same sample rate
}

# Cache for TTS results with access tracking (cleared for voice change)
_tts_cache: Dict[str, str] = {}
_cache_access_times: Dict[str, float] = {}
MAX_CACHE_SIZE = 50  # Limit cache size

# Performance optimization constants (optimized for speed)
CHUNK_SIZE_CHARS = 120  # Smaller chunks for better reliability
MIN_CHUNK_SIZE = 40     # Minimum chunk size to avoid over-chunking
PARALLEL_CHUNK_THRESHOLD = 250  # Higher threshold to avoid unnecessary chunking
TTS_TIMEOUT_SECONDS = 15  # REDUCED: Faster timeout for better responsiveness
HTTP_TIMEOUT_SECONDS = 10  # REDUCED: Faster HTTP timeout

@dataclass
class SimpleCartesiaConfig:
    """Simple configuration for Cartesia TTS balanced for quality and speed"""
    api_key: str = CARTESIA_API_KEY
    model_id: str = "sonic-2"
    voice_id: str = DEFAULT_VOICE_ID
    language: str = "en"
    speed: str = "normal"  # BALANCED: Normal speed for better quality
    output_format: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.output_format is None:
            self.output_format = DEFAULT_OUTPUT_FORMAT.copy()

def smart_text_chunker(text: str, max_chunk_size: int = CHUNK_SIZE_CHARS) -> List[str]:
    """
    Intelligently chunk text for parallel TTS processing.
    Splits on sentence boundaries when possible to maintain natural speech flow.
    """
    if len(text) <= max_chunk_size:
        return [text]

    # Try to split on sentence boundaries first
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed chunk size, start new chunk
        if current_chunk and len(current_chunk + ". " + sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # If chunks are too small, merge them
    merged_chunks = []
    current_merged = ""

    for chunk in chunks:
        if len(current_merged + " " + chunk) <= max_chunk_size:
            if current_merged:
                current_merged += ". " + chunk
            else:
                current_merged = chunk
        else:
            if current_merged:
                merged_chunks.append(current_merged)
            current_merged = chunk

    if current_merged:
        merged_chunks.append(current_merged)

    return merged_chunks if merged_chunks else [text]

class SimpleCartesiaTTS:
    """Ultra-optimized Cartesia TTS client with chunked processing and parallel execution"""

    def __init__(self, config: Optional[SimpleCartesiaConfig] = None):
        self.config = config or SimpleCartesiaConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # For parallel processing
        
    async def __aenter__(self):
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def _ensure_session(self):
        """Ensure HTTP session is available with production-hardened settings"""
        try:
            if self.session is None or self.session.closed:
                headers = {
                    "X-API-Key": self.config.api_key,
                    "Cartesia-Version": CARTESIA_VERSION,
                    "Content-Type": "application/json"
                }
                # Optimized timeouts for speed
                timeout = aiohttp.ClientTimeout(
                    total=HTTP_TIMEOUT_SECONDS,  # Use configurable timeout
                    connect=5,      # REDUCED: Faster connection timeout
                    sock_read=10    # REDUCED: Faster socket read timeout
                )
                # Production-optimized connector
                connector = aiohttp.TCPConnector(
                    limit=5,           # Reduced connection pool
                    limit_per_host=2,  # Match Cartesia concurrency limit
                    ttl_dns_cache=300, # DNS cache TTL
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                self.session = aiohttp.ClientSession(
                    headers=headers,
                    timeout=timeout,
                    connector=connector
                )
        except Exception as e:
            logger.error(f"[Cartesia Simple] Failed to create session: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def synthesize_speech(self, text: str, output_path: str = "output.mp3") -> str:
        """
        Generate speech using Cartesia's HTTP API.
        Robust implementation with caching and error handling.
        """
        logger.info(f"[Cartesia Simple] Synthesizing: {text[:50]}...")
        start_time = time.time()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _tts_cache:
            cached_path = _tts_cache[text_hash]
            if os.path.exists(cached_path):
                try:
                    # Copy cached file to output path
                    async with aiofiles.open(cached_path, 'rb') as src:
                        content = await src.read()
                    async with aiofiles.open(output_path, 'wb') as dst:
                        await dst.write(content)
                    elapsed = time.time() - start_time
                    logger.info(f"[Cartesia Simple] Cache hit in {elapsed:.3f}s")
                    return output_path
                except Exception as cache_error:
                    logger.warning(f"[Cartesia Simple] Cache read failed: {cache_error}")
                    # Continue to generate new audio
        
        try:
            await self._ensure_session()
            
            # Updated payload to match your curl example exactly
            payload = {
                "model_id": "sonic-2",  # Use sonic-2 as per your curl
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": self.config.voice_id  # Uses your specified voice ID
                },
                "output_format": self.config.output_format,
                "language": "en"  # Simplified language as per your curl
            }

            url = CARTESIA_BASE_URL  # Already includes /tts/bytes endpoint
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save to output file
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(content)
                    
                    # Cache the result for future use
                    _tts_cache[text_hash] = output_path
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[Cartesia Simple] Generated in {elapsed:.2f}s: {output_path}")
                    return output_path
                    
                elif response.status == 429:
                    error_text = await response.text()
                    raise Exception(f"Rate limit exceeded: {error_text}")
                    
                elif response.status == 401:
                    raise Exception("Invalid API key")
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"[Cartesia Simple] Network error: {e}")
            raise Exception(f"Network error: {e}")
            
        except Exception as e:
            logger.error(f"[Cartesia Simple] Error: {e}")
            raise

    async def synthesize_speech_chunked(self, text: str, output_path: str = "output.mp3") -> str:
        """
        Ultra-fast TTS using chunked parallel processing.
        Inspired by test.py optimization techniques.
        """
        logger.info(f"[Cartesia Optimized] Chunked synthesis: {text[:50]}...")
        start_time = time.time()

        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in _tts_cache:
            cached_path = _tts_cache[text_hash]
            if os.path.exists(cached_path):
                try:
                    # Copy cached file to output path
                    import shutil
                    shutil.copy2(cached_path, output_path)
                    elapsed = time.time() - start_time
                    logger.info(f"[Cartesia Optimized] Cache hit in {elapsed:.3f}s")
                    return output_path
                except Exception as cache_error:
                    logger.warning(f"[Cartesia Optimized] Cache read failed: {cache_error}")

        # Determine if we should use parallel processing
        # Be more conservative with parallel processing to avoid timeouts
        if len(text) < PARALLEL_CHUNK_THRESHOLD:
            # Use regular synthesis for short text
            return await self.synthesize_speech(text, output_path)

        # Chunk the text for parallel processing
        chunks = smart_text_chunker(text, CHUNK_SIZE_CHARS)
        logger.info(f"[Cartesia Optimized] Processing {len(chunks)} chunks in parallel")

        try:
            # Process chunks in parallel
            chunk_tasks = []
            chunk_paths = []

            for i, chunk in enumerate(chunks):
                chunk_path = f"temp_chunk_{text_hash}_{i}.mp3"
                chunk_paths.append(chunk_path)
                task = asyncio.create_task(self.synthesize_speech(chunk, chunk_path))
                chunk_tasks.append(task)

            # Wait for all chunks to complete with timeout
            chunk_results = await asyncio.wait_for(
                asyncio.gather(*chunk_tasks, return_exceptions=True),
                timeout=TTS_TIMEOUT_SECONDS
            )

            # Check for errors
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"[Cartesia Optimized] Chunk {i} failed: {result}")
                    # Fallback to regular synthesis
                    return await self.synthesize_speech(text, output_path)

            # Combine audio files (simple concatenation for MP3)
            await self._combine_audio_files(chunk_paths, output_path)

            # Cleanup temporary files
            for chunk_path in chunk_paths:
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except Exception as e:
                    logger.warning(f"[Cartesia Optimized] Failed to cleanup {chunk_path}: {e}")

            # Cache the result
            cache_path = f"cache_optimized_{text_hash}.mp3"
            try:
                import shutil
                shutil.copy2(output_path, cache_path)
                _tts_cache[text_hash] = cache_path
                _cache_access_times[text_hash] = time.time()

                # Cleanup old cache entries if needed
                if len(_tts_cache) > MAX_CACHE_SIZE:
                    self._cleanup_cache()

            except Exception as cache_error:
                logger.warning(f"[Cartesia Optimized] Failed to cache result: {cache_error}")

            elapsed = time.time() - start_time
            logger.info(f"[Cartesia Optimized] Chunked synthesis completed in {elapsed:.3f}s")
            return output_path

        except asyncio.TimeoutError:
            logger.error(f"[Cartesia Optimized] Chunked synthesis timed out after {TTS_TIMEOUT_SECONDS}s")
            # Fallback to regular synthesis
            return await self.synthesize_speech(text, output_path)
        except Exception as e:
            logger.error(f"[Cartesia Optimized] Chunked synthesis failed: {e}")
            # Fallback to regular synthesis
            return await self.synthesize_speech(text, output_path)

    async def _combine_audio_files(self, chunk_paths: List[str], output_path: str):
        """Combine multiple MP3 files into one"""
        try:
            # Simple binary concatenation for MP3 files
            # Note: This is a basic approach. For production, consider using ffmpeg
            combined_data = bytearray()

            for chunk_path in chunk_paths:
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as f:
                        chunk_data = f.read()
                        combined_data.extend(chunk_data)

            with open(output_path, 'wb') as f:
                f.write(combined_data)

        except Exception as e:
            logger.error(f"[Cartesia Optimized] Failed to combine audio files: {e}")
            raise

    def _cleanup_cache(self):
        """Clean up old cache entries using LRU strategy"""
        if len(_tts_cache) <= MAX_CACHE_SIZE:
            return

        # Sort by access time (oldest first)
        sorted_items = sorted(_cache_access_times.items(), key=lambda x: x[1])

        # Remove oldest entries
        entries_to_remove = len(_tts_cache) - MAX_CACHE_SIZE + 10  # Remove extra for efficiency
        for i in range(min(entries_to_remove, len(sorted_items))):
            cache_key, _ = sorted_items[i]

            # Remove file
            if cache_key in _tts_cache:
                cache_path = _tts_cache[cache_key]
                try:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                except Exception as e:
                    logger.warning(f"[Cartesia Optimized] Failed to remove {cache_path}: {e}")

                # Remove from cache
                del _tts_cache[cache_key]
                if cache_key in _cache_access_times:
                    del _cache_access_times[cache_key]

        logger.info(f"[Cartesia Optimized] Cleaned up {entries_to_remove} old cache entries")

def clear_tts_cache():
    """Clear all TTS cache entries (useful when changing voices)"""
    global _tts_cache, _cache_access_times

    # Remove all cached files
    for cache_path in _tts_cache.values():
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as e:
            logger.warning(f"[TTS Cache] Failed to remove {cache_path}: {e}")

    # Clear cache dictionaries
    _tts_cache.clear()
    _cache_access_times.clear()

    logger.info("[TTS Cache] All cache entries cleared")

# Global instance
_simple_cartesia: Optional[SimpleCartesiaTTS] = None

async def get_simple_cartesia() -> SimpleCartesiaTTS:
    """Get or create the global simple Cartesia instance"""
    global _simple_cartesia
    if _simple_cartesia is None:
        _simple_cartesia = SimpleCartesiaTTS()
    return _simple_cartesia

# Production-ready convenience functions
async def cartesia_synthesize_async(text: str, output_path: str = "output.mp3") -> str:
    """Production-ready async Cartesia TTS with fresh instance"""
    # Create a fresh TTS instance to avoid event loop issues
    tts = SimpleCartesiaTTS()
    try:
        await tts._ensure_session()
        return await tts.synthesize_speech(text, output_path)
    finally:
        await tts.close()

def cartesia_synthesize_sync(text: str, output_path: str = "output.mp3") -> str:
    """Production-ready sync Cartesia TTS with robust event loop handling"""
    tts_start_time = time.time()
    logger.info(f"[Cartesia Simple] Called synthesize_speech with text: {text}")
    logger.info(f"[LATENCY] Cartesia TTS started at {tts_start_time:.3f}")

    def run_in_fresh_loop():
        """Run async function in a completely fresh event loop"""
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(cartesia_synthesize_async(text, output_path))
        finally:
            try:
                new_loop.close()
            except:
                pass
            asyncio.set_event_loop(None)

    try:
        api_call_start = time.time()

        # Always use a fresh event loop in a thread to avoid conflicts
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_fresh_loop)
            result = future.result(timeout=HTTP_TIMEOUT_SECONDS)

        api_call_time = time.time() - api_call_start
        total_tts_time = time.time() - tts_start_time

        logger.info(f"[LATENCY] TOTAL CARTESIA TTS TIME: {total_tts_time:.3f}s (API: {api_call_time:.3f}s)")
        logger.info(f"[Cartesia Simple] Audio saved to: {output_path}")
        return result

    except Exception as e:
        logger.error(f"[Cartesia Simple] Error in cartesia_synthesize_sync: {e}")
        raise

# Optimized synchronous wrapper using chunked processing
def synthesize_speech_optimized(text: str, output_path: str = "output.mp3") -> str:
    """Ultra-optimized TTS with chunked processing and parallel execution"""
    tts_start_time = time.time()
    logger.info(f"[Cartesia Ultra] Starting optimized synthesis: {text[:50]}...")
    logger.info(f"[LATENCY] Cartesia Ultra TTS started at {tts_start_time:.3f}")

    async def run_optimized():
        async with SimpleCartesiaTTS() as tts:
            return await tts.synthesize_speech_chunked(text, output_path)

    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # If we're already in an event loop, we need to run in a thread
        import concurrent.futures

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(run_optimized())
            finally:
                new_loop.close()

        api_call_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            result = future.result(timeout=TTS_TIMEOUT_SECONDS)
        api_call_time = time.time() - api_call_start

        total_tts_time = time.time() - tts_start_time
        logger.info(f"[LATENCY] TOTAL CARTESIA ULTRA TTS TIME: {total_tts_time:.3f}s (Processing: {api_call_time:.3f}s)")
        logger.info(f"[Cartesia Ultra] Audio saved to: {output_path}")
        return result

    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        api_call_start = time.time()
        result = asyncio.run(run_optimized())
        api_call_time = time.time() - api_call_start

        total_tts_time = time.time() - tts_start_time
        logger.info(f"[LATENCY] TOTAL CARTESIA ULTRA TTS TIME: {total_tts_time:.3f}s (Processing: {api_call_time:.3f}s)")
        logger.info(f"[Cartesia Ultra] Audio saved to: {output_path}")
        return result

# Simplified, reliable synchronous function
def synthesize_speech_simple(text: str, output_path: str = "output.mp3") -> str:
    """ULTRA-FAST direct Cartesia TTS - no async overhead"""
    tts_start_time = time.time()
    logger.info(f"[Cartesia Simple] Starting synthesis: {text[:50]}...")
    logger.info(f"[LATENCY] Cartesia Simple TTS started at {tts_start_time:.3f}")

    # Check cache first for instant response
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _tts_cache:
        cached_path = _tts_cache[text_hash]
        if os.path.exists(cached_path):
            try:
                import shutil
                shutil.copy2(cached_path, output_path)
                elapsed = time.time() - tts_start_time
                logger.info(f"[LATENCY] TOTAL CARTESIA CACHE HIT: {elapsed:.3f}s")
                logger.info(f"[Cartesia Simple] Cache hit: {output_path}")
                return output_path
            except Exception as cache_error:
                logger.warning(f"[Cartesia Simple] Cache read failed: {cache_error}")

    # DIRECT API CALL - No async/thread overhead
    try:
        api_call_start = time.time()

        # Get API key
        api_key = os.getenv("CARTESIA_API_KEY", "")
        if not api_key:
            raise Exception("CARTESIA_API_KEY not set")

        # Direct HTTP request - same as our test
        import requests
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

        # Cartesia official format - exactly as in documentation
        payload = {
            "model_id": "sonic-2",
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": "bf0a246a-8642-498a-9950-80c35e9276b5"
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100
            }
        }

        # Make the API call with streaming
        response = requests.post(url, headers=headers, json=payload, timeout=10, stream=True)

        if response.status_code == 200:
            # Stream and save the audio file immediately
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        api_call_time = time.time() - api_call_start

        # Cache the result
        try:
            cache_path = f"cache_simple_{text_hash}.wav"
            import shutil
            shutil.copy2(output_path, cache_path)
            _tts_cache[text_hash] = cache_path
            _cache_access_times[text_hash] = time.time()
        except Exception as cache_error:
            logger.warning(f"[Cartesia Simple] Failed to cache result: {cache_error}")

        total_tts_time = time.time() - tts_start_time
        logger.info(f"[LATENCY] TOTAL CARTESIA SIMPLE TTS TIME: {total_tts_time:.3f}s (API: {api_call_time:.3f}s)")
        logger.info(f"[Cartesia Simple] Audio saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"[Cartesia Simple] Synthesis failed: {e}")
        raise


async def synthesize_speech_streaming_twilio(text: str, websocket, stream_sid: str,
                                           cancel_check: Optional[Callable] = None) -> bool:
    """
    ULTRA-LOW LATENCY WebSocket streaming TTS for Twilio
    Based on your latency optimization techniques - streams audio chunks immediately
    """
    import base64
    import json
    import websockets
    from urllib.parse import urlencode

    logger.info(f"[Cartesia Streaming] Starting streaming TTS: {text[:50]}...")
    start_time = time.time()

    try:
        # Get API key
        api_key = os.getenv("CARTESIA_API_KEY", "")
        if not api_key:
            raise Exception("CARTESIA_API_KEY not set")

        # WebSocket URL with optimized parameters
        params = {
            "api_key": api_key,
            "cartesia_version": "2024-06-10"
        }
        ws_url = f"wss://api.cartesia.ai/tts/websocket?{urlencode(params)}"

        # Connect to Cartesia WebSocket with optimized parameters for real-time audio
        async with websockets.connect(
            ws_url,
            ping_interval=20,      # Keep connection alive
            ping_timeout=10,       # Quick timeout for failed pings
            max_size=2**20,        # 1MB max message size for audio chunks
            compression=None       # Disable compression for lower latency
        ) as cartesia_ws:
            logger.info(f"[Cartesia Streaming] Connected to WebSocket")

            # Send TTS request with optimized format for phone calls
            message = {
                "model_id": "sonic-2",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": "bf0a246a-8642-498a-9950-80c35e9276b5"
                },
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_mulaw",  # Optimized for phone calls
                    "sample_rate": 8000       # Twilio standard
                },
                "language": "en"
            }

            await cartesia_ws.send(json.dumps(message))
            logger.info(f"[Cartesia Streaming] Sent TTS request")

            chunks_sent = 0
            first_chunk_time = None

            # Stream audio chunks as they arrive (ZERO BUFFERING)
            while True:
                # Check for cancellation (barge-in detection)
                if cancel_check and cancel_check():
                    logger.info("🛑 [Cartesia Streaming] TTS cancelled — closing socket")
                    await cartesia_ws.close()
                    return False

                try:
                    # Optimized non-blocking receive with minimal timeout for real-time streaming
                    msg = await asyncio.wait_for(cartesia_ws.recv(), timeout=0.05)
                    data = json.loads(msg)

                    if data.get("type") == "chunk":
                        # Record first chunk latency (KEY METRIC)
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            first_chunk_latency = first_chunk_time - start_time
                            logger.info(f"[LATENCY] 🚀 First audio chunk in {first_chunk_latency:.3f}s")

                        # Get audio data
                        audio_chunk = base64.b64decode(data["data"])

                        # Convert to base64 for Twilio
                        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')

                        # IMMEDIATELY send to Twilio with minimal buffering for real-time streaming
                        media_message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_b64}
                        }
                        await websocket.send_json(media_message)

                        chunks_sent += 1

                    elif data.get("type") == "done":
                        logger.info(f"[Cartesia Streaming] ✅ Completed - sent {chunks_sent} chunks")
                        break

                    elif data.get("type") == "error":
                        logger.error(f"[Cartesia Streaming] ❌ Error: {data}")
                        return False

                except asyncio.TimeoutError:
                    # Continue looping - this is normal for streaming
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("[Cartesia Streaming] Connection closed")
                    break
                except Exception as e:
                    logger.error(f"[Cartesia Streaming] Error receiving: {e}")
                    break

            total_time = time.time() - start_time
            logger.info(f"[LATENCY] 🎯 TOTAL STREAMING TTS TIME: {total_time:.3f}s ({chunks_sent} chunks)")
            return True

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[Cartesia Streaming] ❌ Failed after {total_time:.3f}s: {e}")
        return False


def warm_simple_cache():
    """Pre-warm the cache with common healthcare phrases"""
    logger.info("[Cartesia Simple] Warming cache...")
    
    common_phrases = [
        "Hello! I'm here to help you with your healthcare needs today.",
        "I can help you schedule an appointment, verify insurance, or answer questions.",
        "Let me check that information for you.",
        "I've found some available appointment slots for you.",
        "Your insurance information has been verified successfully.",
        "Is there anything else I can help you with today?",
        "Thank you for calling. Have a great day!",
        "Sorry, could you repeat that?",
        "I understand you're looking for an appointment.",
        "What brings you in today?"
    ]
    
    # Use thread pool to avoid event loop issues
    import concurrent.futures
    import threading
    
    def cache_phrase(phrase):
        try:
            output_path = f"cache_simple_{hashlib.md5(phrase.encode()).hexdigest()}.mp3"
            return cartesia_synthesize_sync(phrase, output_path)
        except Exception as e:
            logger.warning(f"[Cartesia Simple] Failed to cache: {e}")
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cache_phrase, phrase) for phrase in common_phrases]
        
        successful = 0
        for future in concurrent.futures.as_completed(futures, timeout=120):
            try:
                result = future.result()
                if result:
                    successful += 1
            except Exception as e:
                logger.warning(f"[Cartesia Simple] Cache warming error: {e}")
    
    logger.info(f"[Cartesia Simple] Cached {successful}/{len(common_phrases)} phrases")

# Test function
async def test_simple_cartesia():
    """Test the simple Cartesia implementation"""
    text = "This is a test of the simple Cartesia TTS implementation."
    output_path = "test_simple.mp3"
    
    try:
        start_time = time.time()
        result = await cartesia_synthesize_async(text, output_path)
        elapsed = time.time() - start_time
        
        if os.path.exists(result):
            file_size = os.path.getsize(result)
            logger.info(f"✅ Simple Cartesia test successful: {elapsed:.2f}s, {file_size} bytes")
            os.remove(result)  # Cleanup
            return True
        else:
            logger.error("❌ Simple Cartesia test failed: no file generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple Cartesia test error: {e}")
        return False

if __name__ == "__main__":
    # Test the simple implementation
    asyncio.run(test_simple_cartesia())
