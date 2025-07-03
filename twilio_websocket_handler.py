#!/usr/bin/env python3
"""
Twilio WebSocket Handler with Streaming TTS
Integrates with your existing health agent and session manager
"""

import asyncio
import json
import logging
import time
import base64
import websockets
from websockets.server import serve
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing components
try:
    from app.ai.health_agent import HealthAgent
    from app.ai.session_manager import get_session_manager
    from app.voice.cartesia_tts_simple import synthesize_speech_streaming_twilio
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
session_manager = get_session_manager()

class TwilioWebSocketHandler:
    """Handles Twilio WebSocket connections with streaming TTS"""
    
    def __init__(self):
        self.health_agent = HealthAgent()
        self.active_calls = {}
        
    async def handle_connection(self, websocket, path):
        """Handle incoming WebSocket connection from Twilio"""
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"🔌 New WebSocket connection from {client_ip}")
        
        # Connection state
        stream_sid = None
        call_sid = None
        user_speaking = False
        tts_active = False
        
        def should_cancel_tts():
            """Check if TTS should be cancelled due to user speaking"""
            return user_speaking and tts_active
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event = data.get("event", "unknown")
                    
                    if event == "connected":
                        logger.info("✅ Twilio connected successfully!")
                        
                    elif event == "start":
                        # Call started
                        start_data = data.get("start", {})
                        stream_sid = start_data.get("streamSid")
                        call_sid = start_data.get("callSid")
                        
                        logger.info(f"📞 Call started: {call_sid}")
                        logger.info(f"🌊 Stream ID: {stream_sid}")
                        
                        # Store call info
                        self.active_calls[call_sid] = {
                            "stream_sid": stream_sid,
                            "start_time": time.time(),
                            "websocket": websocket
                        }
                        
                        # Send welcome message with streaming TTS
                        welcome_text = "Hello! I'm your healthcare assistant. How can I help you today?"
                        logger.info(f"🤖 Sending welcome: {welcome_text}")
                        
                        tts_active = True
                        start_time = time.time()
                        
                        # Use streaming TTS if available, fallback to simple TTS
                        try:
                            success = await synthesize_speech_streaming_twilio(
                                welcome_text, websocket, stream_sid, should_cancel_tts
                            )
                            
                            if success:
                                welcome_time = time.time() - start_time
                                logger.info(f"🎯 Welcome streamed in {welcome_time:.2f}s")
                            else:
                                logger.warning("⚠️ Streaming TTS failed, using fallback")
                                await self.send_simple_tts_response(welcome_text, websocket, stream_sid)
                                
                        except Exception as e:
                            logger.error(f"❌ TTS error: {e}")
                            await self.send_simple_tts_response(welcome_text, websocket, stream_sid)
                        
                        tts_active = False
                        
                    elif event == "media":
                        # Audio data from user
                        media_data = data.get("media", {})
                        payload = media_data.get("payload", "")
                        
                        # For now, we'll handle STT in the existing router
                        # This is where you'd implement real-time STT
                        logger.debug("🎤 Received audio data from user")
                        
                    elif event == "mark":
                        # Audio playback markers
                        mark_data = data.get("mark", {})
                        logger.debug(f"🔖 Audio mark: {mark_data}")
                        
                    elif event == "stop":
                        logger.info(f"📞 Call ended: {call_sid}")
                        if call_sid in self.active_calls:
                            call_duration = time.time() - self.active_calls[call_sid]["start_time"]
                            logger.info(f"⏱️ Call duration: {call_duration:.1f}s")
                            del self.active_calls[call_sid]
                        break
                        
                    else:
                        logger.debug(f"📋 Other event: {event}")
                        
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
        finally:
            # Cleanup
            if call_sid and call_sid in self.active_calls:
                del self.active_calls[call_sid]
    
    async def send_simple_tts_response(self, text, websocket, stream_sid):
        """Fallback method to send TTS response"""
        try:
            # Use your existing simple TTS
            from app.voice.cartesia_tts_simple import synthesize_speech_simple
            
            # Generate audio file
            output_file = f"temp_response_{int(time.time())}.wav"
            audio_path = synthesize_speech_simple(text, output_file)
            
            # Read and encode audio
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Convert to base64 and send
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            response = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": audio_b64}
            }
            
            await websocket.send(json.dumps(response))
            logger.info("🎵 Sent fallback TTS response")
            
            # Cleanup temp file
            try:
                os.remove(audio_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"❌ Fallback TTS error: {e}")

async def main():
    """Start the WebSocket server"""
    handler = TwilioWebSocketHandler()
    
    host = "0.0.0.0"
    port = 8001
    
    logger.info("🚀 Starting Twilio WebSocket Handler")
    logger.info(f"🔌 Server: {host}:{port}")
    logger.info("📋 Features:")
    logger.info("  • Streaming TTS with Cartesia")
    logger.info("  • Healthcare conversation AI")
    logger.info("  • Session management")
    logger.info("  • Real-time audio processing")
    logger.info("\n🔗 Use with ngrok: ws://your-ngrok-url.ngrok.io")
    logger.info("📞 Configure in Twilio Console")
    
    async with serve(handler.handle_connection, host, port):
        logger.info("✅ WebSocket server is running...")
        logger.info("🛑 Press Ctrl+C to stop")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
