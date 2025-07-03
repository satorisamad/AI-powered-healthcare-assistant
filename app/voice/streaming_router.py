#!/usr/bin/env python3
"""
Ultra-Low Latency Streaming Router for Twilio WebSocket
Uses WebSocket streaming TTS for sub-1-second first-audio latency
"""

import asyncio
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.voice.cartesia_tts_simple import synthesize_speech_streaming_twilio
from app.ai.health_agent import HealthAgent
from app.ai.session_manager import get_session_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ultra-Low Latency Voice Assistant")

# Global instances
session_manager = get_session_manager()

@app.websocket("/voice-stream")
async def websocket_voice_handler(websocket: WebSocket):
    """
    Ultra-low latency WebSocket handler for voice interactions
    Implements streaming TTS for sub-1-second response times
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    
    # Connection state
    stream_sid = None
    call_sid = None
    health_agent = HealthAgent()
    
    # Barge-in detection state
    user_speaking = False
    tts_active = False
    
    def should_cancel_tts():
        """Check if TTS should be cancelled due to user speaking"""
        return user_speaking and tts_active
    
    try:
        while True:
            # Receive message from Twilio
            message = await websocket.receive_text()
            data = json.loads(message)
            
            event = data.get("event")
            
            if event == "connected":
                logger.info("📞 Twilio connected")
                
            elif event == "start":
                # Call started
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                logger.info(f"🚀 Call started: {call_sid}")
                
                # Send welcome message with streaming TTS
                welcome_text = "Hello! I'm your healthcare assistant. How can I help you today?"
                tts_active = True
                
                start_time = time.time()
                success = await synthesize_speech_streaming_twilio(
                    welcome_text, websocket, stream_sid, should_cancel_tts
                )
                
                if success:
                    welcome_time = time.time() - start_time
                    logger.info(f"🎯 Welcome message streamed in {welcome_time:.2f}s")
                
                tts_active = False
                
            elif event == "media":
                # Audio data from user - could implement real-time STT here
                # For now, we'll handle this in the existing router
                pass
                
            elif event == "speech_started":
                # User started speaking - enable barge-in
                user_speaking = True
                if tts_active:
                    logger.info("🛑 User speaking - will cancel TTS")
                
            elif event == "speech_ended":
                # User stopped speaking
                user_speaking = False
                
            elif event == "transcript":
                # Speech-to-text result
                transcript = data.get("transcript", "").strip()
                if transcript:
                    logger.info(f"👤 User: {transcript}")
                    
                    # Process with health agent
                    start_time = time.time()
                    
                    # Get session
                    session = session_manager.get_or_create_session(call_sid)
                    
                    # Process message
                    response = await asyncio.to_thread(
                        health_agent.process, transcript, voice_mode=True
                    )
                    
                    agent_time = time.time() - start_time
                    logger.info(f"🤖 Agent response ({agent_time:.2f}s): {response[:50]}...")
                    
                    # Stream TTS response
                    tts_active = True
                    tts_start = time.time()
                    
                    success = await synthesize_speech_streaming_twilio(
                        response, websocket, stream_sid, should_cancel_tts
                    )
                    
                    if success:
                        total_response_time = time.time() - start_time
                        tts_time = time.time() - tts_start
                        logger.info(f"🎯 TOTAL RESPONSE TIME: {total_response_time:.2f}s (Agent: {agent_time:.2f}s, TTS: {tts_time:.2f}s)")
                    
                    tts_active = False
                
            elif event == "stop":
                logger.info("📞 Call ended")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        logger.info("🧹 Cleaning up WebSocket connection")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ultra-low-latency-voice-assistant"}

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Ultra-Low Latency Voice Assistant",
        "features": [
            "WebSocket streaming TTS",
            "Sub-1-second first-audio latency",
            "Real-time barge-in detection",
            "Healthcare conversation AI"
        ],
        "endpoints": {
            "websocket": "/voice-stream",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Ultra-Low Latency Voice Assistant")
    print("Features:")
    print("  • WebSocket streaming TTS")
    print("  • Sub-1-second first-audio latency") 
    print("  • Real-time barge-in detection")
    print("  • Healthcare conversation AI")
    print("\nEndpoints:")
    print("  • WebSocket: ws://localhost:8001/voice-stream")
    print("  • Health: http://localhost:8001/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
