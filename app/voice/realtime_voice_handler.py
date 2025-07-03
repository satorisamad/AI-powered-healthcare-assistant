"""
Real-time WebSocket voice handler for ultra-low latency voice interactions.
Based on optimizations from test.py for healthcare assistant.
"""

import os
import time
import base64
import logging
import asyncio
import json
import audioop
import numpy as np
from urllib.parse import urlencode
from fastapi import WebSocket
import websockets
from typing import Optional, Callable, Dict, List
from app.ai.session_manager import get_session_manager
from app.voice.tts_config import get_tts_manager
from app.voice.audio_manager import get_audio_manager
from app.voice.barge_in_detector import get_barge_in_detector

logger = logging.getLogger(__name__)

def is_real_speech(audio: bytes, threshold_db: float = -25.0) -> bool:
    """
    Optimized speech detection from test.py using RMS energy analysis.
    Estimate if the audio contains real speech using RMS energy from µ-law samples.
    Returns True if audio energy is above a given dB threshold.
    """
    if not audio:
        return False

    try:
        # Convert µ-law to 16-bit PCM
        pcm_audio = audioop.ulaw2lin(audio, 2)  # 2 bytes = 16-bit PCM
        rms = audioop.rms(pcm_audio, 2)  # width=2 for 16-bit
        db = 20 * np.log10(rms / 32768) if rms > 0 else -float("inf")
        return db > threshold_db
    except Exception as e:
        logger.warning(f"RMS energy check failed: {e}")
        return False

# Voice optimization constants from test.py - OPTIMIZED FOR ULTRA-LOW LATENCY
TTS_GRACE_SECONDS = 3                    # Grace period before allowing barge-in
MIN_BARGE_TRANSCRIPT_LENGTH = 10         # Minimum chars to trigger barge-in
BARGE_MIN_AUDIO_WINDOW = 0.25           # 250ms audio window for analysis
BARGE_AUDIO_THRESHOLD = 500             # Bytes threshold for real speech
BACK_OFF_SECONDS = 2.0                  # Delay between responses
AUDIO_CHUNK_SIZE = 960                  # 120ms audio chunks for analysis (from test.py)
SPEECH_END_TIMEOUT = 0.5                # Trigger LLM after 500ms silence (from test.py)
SILENCE_BYTES = b'\100' * 160           # 20ms silence in mulaw @8kHz (from test.py)
SILENCE_BYTES = b'\100' * 160  # ~20ms silence in mulaw @8kHz
KEEP_ALIVE_INTERVAL = 4
AUDIO_CHUNK_SIZE = 960  # 120ms of audio to analyze
SPEECH_END_TIMEOUT = 0.5

class RealtimeVoiceHandler:
    """Real-time voice handler with ultra-low latency optimizations."""
    
    def __init__(self):
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.session_manager = get_session_manager()
        self.audio_manager = get_audio_manager()
        self.barge_in_detector = get_barge_in_detector()
        
        # Voice state tracking
        self.stream_sid = None
        self.call_sid = None
        self.caller_id = None
        self.call_start_time = None
        
        # Audio processing state
        self.is_speaking = False
        self.last_tts_timestamp = 0
        self.tts_active = False
        self.tts_cancelled = False
        self.last_audio_ts = time.time()
        self.last_barge_ts = 0
        
        # Transcript processing
        self.partial_transcript = ""
        self.final_transcript = ""
        self.last_speech_time = 0
        self.utterance_finalized = False
        
        # Conversation state
        self.conversation = []
        
        # WebSocket connections
        self.twilio_ws = None
        self.deepgram_ws = None
        self.tts_task = None
        
        # Audio buffer for real-time analysis
        self.audio_buffer = bytearray()
    
    async def setup_deepgram_stt(self) -> websockets.WebSocketClientProtocol:
        """Setup Deepgram STT WebSocket with optimized parameters."""
        params = {
            "model": "nova-3",  # Fastest, most accurate model
            "language": "en-US",
            "smart_format": "true",
            "encoding": "mulaw",
            "sample_rate": "8000",
            "channels": "1",
            "interim_results": "true",  # Enable interim results for faster response
            "endpointing": "300",       # Faster endpointing (300ms)
            "interim_results": "true",
            "endpointing": "300",
            "utterance_end_ms": "1000",
            "filler_words": "false",
            "no_delay": "true"  # Critical for low latency
        }
        url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
        headers = {"Authorization": f"Token {self.deepgram_api_key}"}
        return await websockets.connect(url, additional_headers=headers)
    
    def is_real_speech(self, audio: bytes, threshold_db: float = -25.0) -> bool:
        """
        Detect real speech using RMS energy analysis.
        Critical for barge-in detection.
        """
        if not audio:
            return False
        
        try:
            # Convert µ-law to 16-bit PCM
            pcm_audio = audioop.ulaw2lin(audio, 2)
            rms = audioop.rms(pcm_audio, 2)
            db = 20 * np.log10(rms / 32768) if rms > 0 else -float("inf")
            return db > threshold_db
        except Exception as e:
            logger.warning(f"RMS energy check failed: {e}")
            return False
    
    async def send_mark_message(self, ws: WebSocket, sid: str, text: str):
        """Send mark message to Twilio WebSocket."""
        await ws.send_json({
            "event": "mark",
            "streamSid": sid,
            "mark": {"name": "intro", "value": text}
        })
    
    async def send_clear_event(self, ws: WebSocket, sid: str):
        """Send clear event to stop current audio playback."""
        await ws.send_json({
            "event": "clear",
            "streamSid": sid
        })
    
    async def speak_response_optimized(self, text: str, websocket: WebSocket, 
                                    stream_sid: str, should_cancel: Callable[[], bool]) -> bool:
        """
        Optimized TTS response with cancellation support.
        """
        try:
            # Check for duplicate content first
            existing_path = self.audio_manager.deduplicate_content(text)
            if existing_path:
                logger.info(f"[RealtimeVoice] Using cached audio for: {text[:50]}...")
                audio_filename = os.path.basename(existing_path)
            else:
                # Generate new audio with unified TTS manager
                output_path = self.audio_manager.generate_audio_path(self.call_sid, text)
                tts_manager = await get_tts_manager()
                result_path = await tts_manager.synthesize_speech(text, output_path)
                self.audio_manager.register_file(result_path, self.call_sid, text, ttl_minutes=15)
                audio_filename = os.path.basename(result_path)
            
            # Send audio to Twilio
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "contentType": "audio/mp3",
                    "payload": audio_filename
                }
            })
            
            # Mark file as accessed
            self.audio_manager.access_file(existing_path if existing_path else result_path)
            
            return True
            
        except Exception as e:
            logger.error(f"[RealtimeVoice] TTS error: {e}")
            return False
    
    async def process_transcript_with_llm(self, transcript: str, websocket: WebSocket, 
                                        stream_sid: str) -> bool:
        """
        Process transcript through LLM with voice optimizations.
        """
        try:
            cleaned_transcript = transcript.strip()
            if not cleaned_transcript:
                logger.warning("[RealtimeVoice] Empty transcript, skipping LLM")
                return False

            logger.warning(f"[RealtimeVoice] 🎯 UTTERANCE RECEIVED: '{cleaned_transcript}' "
                          f"(length: {len(cleaned_transcript)} chars)")

            # Process through session manager with voice mode
            start_time = time.time()
            logger.info(f"[RealtimeVoice] 🧠 Sending to LLM (voice_mode=True)...")

            result = self.session_manager.process_message(
                self.call_sid, cleaned_transcript, voice_mode=True
            )
            llm_time = time.time() - start_time

            agent_reply = result['reply']
            logger.warning(f"[RealtimeVoice] 🤖 LLM RESPONSE ({llm_time:.2f}s): '{agent_reply}' "
                          f"(length: {len(agent_reply)} chars)")
            
            # Add to conversation history
            self.conversation.append({"role": "user", "content": cleaned_transcript})
            self.conversation.append({
                "role": "assistant", 
                "content": agent_reply,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Apply back-off if needed
            now = time.time()
            if now - self.last_barge_ts < BACK_OFF_SECONDS:
                sleep_duration = BACK_OFF_SECONDS - (now - self.last_barge_ts)
                logger.info(f"[RealtimeVoice] Backing off for {sleep_duration:.2f}s")
                await asyncio.sleep(sleep_duration)
            
            # Wait for any active TTS to finish
            while self.tts_active:
                logger.warning("[RealtimeVoice] Waiting for previous TTS...")
                await asyncio.sleep(0.1)
            
            # Start TTS
            self.tts_cancelled = False
            self.tts_active = True
            self.last_tts_timestamp = time.time()
            
            self.tts_task = asyncio.create_task(
                self.speak_response_optimized(
                    agent_reply, websocket, stream_sid, 
                    lambda: self.tts_cancelled
                )
            )
            
            try:
                await self.tts_task
            except asyncio.CancelledError:
                logger.info("[RealtimeVoice] TTS task cancelled cleanly")
                return False
            
            # Small buffer and flush Deepgram
            await asyncio.sleep(0.05)
            if self.deepgram_ws and not self.deepgram_ws.closed:
                await self.deepgram_ws.send(b'\x00' * 320)
            
            return True
            
        except Exception as e:
            logger.error(f"[RealtimeVoice] LLM processing error: {e}")
            return False
        finally:
            self.tts_active = False
    
    async def deepgram_keepalive(self):
        """Keep Deepgram connection alive with periodic silence."""
        try:
            consecutive_failures = 0
            MAX_FAILURES = 3
            
            while True:
                await asyncio.sleep(KEEP_ALIVE_INTERVAL)
                
                if (getattr(self.deepgram_ws, "closed", True) or 
                    self.deepgram_ws.close_code not in (None, 1000)):
                    logger.info("[RealtimeVoice] Deepgram WS closed, ending keepalive")
                    break
                
                current_time = time.time()
                time_since_last_audio = current_time - self.last_audio_ts
                
                if time_since_last_audio > KEEP_ALIVE_INTERVAL:
                    logger.debug(f"[RealtimeVoice] Sending keepalive silence")
                    try:
                        await asyncio.wait_for(
                            self.deepgram_ws.send(SILENCE_BYTES), timeout=2.0
                        )
                        consecutive_failures = 0
                    except Exception as e:
                        consecutive_failures += 1
                        logger.warning(f"[RealtimeVoice] Keepalive failed ({consecutive_failures}/{MAX_FAILURES}): {e}")
                        
                        if consecutive_failures >= MAX_FAILURES:
                            logger.error("[RealtimeVoice] Max failures, reconnecting Deepgram...")
                            try:
                                await self.deepgram_ws.close()
                                self.deepgram_ws = await self.setup_deepgram_stt()
                                consecutive_failures = 0
                            except Exception as reconnect_err:
                                logger.error(f"[RealtimeVoice] Reconnect failed: {reconnect_err}")
                                break
                
        except Exception as e:
            logger.error(f"[RealtimeVoice] Keepalive loop crashed: {e}")
            if self.deepgram_ws:
                try:
                    await self.deepgram_ws.close()
                except:
                    pass

    async def handle_twilio_events(self, websocket: WebSocket):
        """Handle Twilio WebSocket events with barge-in detection."""
        try:
            while True:
                msg = await websocket.receive_json()
                event = msg.get("event")

                if event == "start":
                    # Initialize call state
                    self.call_start_time = time.time()
                    self.stream_sid = msg["start"]["streamSid"]
                    self.call_sid = msg["start"]["callSid"]

                    # Reset all state
                    self.conversation = []
                    self.is_speaking = False
                    self.last_tts_timestamp = 0
                    self.tts_active = False
                    self.tts_cancelled = False
                    self.last_audio_ts = time.time()
                    self.last_barge_ts = 0
                    self.partial_transcript = ""
                    self.final_transcript = ""
                    self.utterance_finalized = False
                    self.audio_buffer = bytearray()

                    logger.info(f"[RealtimeVoice] Call started: {self.call_sid}")

                    # Send initial greeting
                    await self.send_mark_message(websocket, self.stream_sid, "intro")
                    greeting = "Hello! You've reached Harmony Health Clinic. I'm Sarah, your AI health assistant - how can I help you today?"

                    self.tts_active = True
                    self.last_tts_timestamp = time.time()
                    await self.speak_response_optimized(
                        greeting, websocket, self.stream_sid, lambda: self.tts_cancelled
                    )
                    self.tts_active = False

                    # Send silence to keep Deepgram alive
                    if self.deepgram_ws and not self.deepgram_ws.closed:
                        await self.deepgram_ws.send(SILENCE_BYTES)
                    self.last_audio_ts = time.time()

                elif event == "media":
                    now = time.time()
                    audio = base64.b64decode(msg["media"]["payload"])
                    self.last_audio_ts = now

                    # Buffer audio for analysis (120ms chunks)
                    self.audio_buffer.extend(audio)
                    if len(self.audio_buffer) >= AUDIO_CHUNK_SIZE:
                        audio_chunk = bytes(self.audio_buffer[:AUDIO_CHUNK_SIZE])
                        self.audio_buffer = self.audio_buffer[AUDIO_CHUNK_SIZE:]

                        # Barge-in detection during TTS
                        if self.tts_active:
                            # Check grace period
                            if now - self.last_tts_timestamp < TTS_GRACE_SECONDS:
                                logger.debug("[RealtimeVoice] Within TTS grace period")
                                if self.deepgram_ws and not self.deepgram_ws.closed:
                                    await self.deepgram_ws.send(audio)
                                continue

                            # Check barge-in cooldown
                            if now - self.last_barge_ts < 0.75:
                                logger.debug("[RealtimeVoice] Barge-in cooldown active")
                                if self.deepgram_ws and not self.deepgram_ws.closed:
                                    await self.deepgram_ws.send(audio)
                                continue

                            # Advanced barge-in detection
                            grace_period_active = now - self.last_tts_timestamp < TTS_GRACE_SECONDS
                            time_since_tts = now - self.last_tts_timestamp

                            logger.debug(f"[RealtimeVoice] 🔍 Barge-in check: "
                                        f"TTS_active={self.tts_active}, "
                                        f"grace_period={grace_period_active}, "
                                        f"time_since_tts={time_since_tts:.2f}s")

                            should_barge_in, confidence, debug_info = self.barge_in_detector.detect_barge_in(
                                audio_chunk,
                                tts_active=self.tts_active,
                                grace_period_active=grace_period_active
                            )

                            if should_barge_in:
                                logger.warning(f"[RealtimeVoice] 🛑 ADVANCED BARGE-IN DETECTED! "
                                              f"Confidence: {confidence:.2f}, "
                                              f"Time since TTS: {time_since_tts:.2f}s")
                                logger.warning(f"[RealtimeVoice] 📊 Barge-in details: {debug_info}")

                                self.tts_cancelled = True
                                self.last_barge_ts = now

                                # Cancel active TTS
                                if self.tts_task and not self.tts_task.done():
                                    try:
                                        self.tts_task.cancel()
                                        await asyncio.wait_for(self.tts_task, timeout=1.0)
                                    except asyncio.CancelledError:
                                        logger.info("[RealtimeVoice] TTS cancelled cleanly")
                                    except Exception as e:
                                        logger.warning(f"[RealtimeVoice] TTS cancel error: {e}")

                                # Clear Twilio audio queue
                                await self.send_clear_event(websocket, self.stream_sid)
                                self.tts_active = False
                                self.is_speaking = False

                                await asyncio.sleep(0.05)
                                continue

                    # Forward audio to Deepgram
                    if self.deepgram_ws and not self.deepgram_ws.closed:
                        await self.deepgram_ws.send(audio)

                elif event == "stop":
                    logger.info("[RealtimeVoice] Call ended")
                    # Flush and close connections
                    if self.deepgram_ws and not self.deepgram_ws.closed:
                        await self.deepgram_ws.send(b"")
                        await self.deepgram_ws.close()
                    await websocket.close()
                    break

        except Exception as e:
            logger.error(f"[RealtimeVoice] Twilio handler error: {e}")

    async def handle_deepgram_events(self):
        """Handle Deepgram STT events with utterance detection."""
        try:
            async for msg in self.deepgram_ws:
                try:
                    if not isinstance(msg, str) or not msg.strip().startswith("{"):
                        continue

                    data = json.loads(msg)

                    # Process channel transcripts
                    if isinstance(data.get("channel"), dict):
                        alternatives = data["channel"].get("alternatives", [])
                        if not alternatives:
                            continue

                        alt = alternatives[0]
                        transcript = alt.get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)

                        if transcript:
                            if self.utterance_finalized:
                                # Reset for new utterance
                                self.utterance_finalized = False
                                self.final_transcript = ""
                                self.partial_transcript = ""

                            self.last_speech_time = time.time()
                            self.is_speaking = True

                            if is_final:
                                self.final_transcript += " " + transcript
                                self.partial_transcript = transcript
                            else:
                                self.partial_transcript = transcript

                        # Process complete utterance
                        if speech_final and not self.utterance_finalized:
                            cleaned_final = self.final_transcript.strip()
                            if cleaned_final and not self.tts_active:
                                logger.info(f"[RealtimeVoice] Speech final: {cleaned_final}")
                                await self.process_transcript_with_llm(
                                    cleaned_final, self.twilio_ws, self.stream_sid
                                )
                            self.utterance_finalized = True

                    # Handle UtteranceEnd as fallback
                    if (data.get("type") == "UtteranceEnd" and
                        not self.utterance_finalized):
                        cleaned_final = self.final_transcript.strip()
                        if cleaned_final and not self.tts_active:
                            logger.info(f"[RealtimeVoice] Utterance end: {cleaned_final}")
                            await self.process_transcript_with_llm(
                                cleaned_final, self.twilio_ws, self.stream_sid
                            )
                        self.utterance_finalized = True

                except Exception as e:
                    logger.error(f"[RealtimeVoice] Deepgram message error: {e}")

        except Exception as e:
            logger.error(f"[RealtimeVoice] Deepgram handler error: {e}")

    async def handle_voice_call(self, websocket: WebSocket, greeting: str = None):
        """
        Main entry point for handling real-time voice calls.
        """
        self.twilio_ws = websocket

        try:
            await websocket.accept()
            logger.info("[RealtimeVoice] WebSocket accepted")

            # Setup Deepgram connection
            self.deepgram_ws = await self.setup_deepgram_stt()
            await self.deepgram_ws.send(SILENCE_BYTES)  # Initial keepalive

            # Run all handlers concurrently
            await asyncio.gather(
                self.handle_twilio_events(websocket),
                self.handle_deepgram_events(),
                self.deepgram_keepalive()
            )

        except Exception as e:
            logger.error(f"[RealtimeVoice] Handler error: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources and save conversation."""
        logger.info("[RealtimeVoice] Starting cleanup...")

        # Cancel active TTS
        if self.tts_task and not self.tts_task.done():
            self.tts_cancelled = True
            self.tts_task.cancel()
            try:
                await asyncio.wait_for(self.tts_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self.tts_active = False

        # Close WebSocket connections
        if self.deepgram_ws and not self.deepgram_ws.closed:
            try:
                await self.deepgram_ws.close()
            except:
                pass

        if self.twilio_ws and not self.twilio_ws.client_state.DISCONNECTED:
            try:
                await self.twilio_ws.close()
            except:
                pass

        # Save conversation if exists
        if self.conversation and self.call_sid:
            call_end_time = time.time()
            call_duration = call_end_time - self.call_start_time if self.call_start_time else None

            conversation_data = {
                'call_sid': self.call_sid,
                'stream_sid': self.stream_sid,
                'caller_id': self.caller_id,
                'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                'conversation': self.conversation,
                'call_duration': call_duration,
                'optimization_used': 'realtime_websocket'
            }

            # Save to session manager
            try:
                session_info = self.session_manager.get_session_info(self.call_sid)
                if session_info:
                    session_info['conversation_data'] = conversation_data
                    logger.info(f"[RealtimeVoice] Conversation saved for {self.call_sid}")
            except Exception as e:
                logger.error(f"[RealtimeVoice] Error saving conversation: {e}")

        logger.info("[RealtimeVoice] Cleanup completed")
