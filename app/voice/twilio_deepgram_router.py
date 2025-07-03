import os
import logging
import uuid
import time
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Body, HTTPException

# Load environment variables from .env file
load_dotenv()
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from app.voice.cartesia_tts_simple_http import synthesize_speech_simple_http as synthesize_speech
from app.ai.health_agent import HealthAgent
from app.ai.session_manager import get_session_manager

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

# Initialize your agent
agent = HealthAgent()

# Ensure audio_files directory exists
os.makedirs("audio_files", exist_ok=True)

# Simple health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Twilio Voice Assistant"}

@app.post("/voice", response_class=PlainTextResponse)
async def voice(request: Request):
    """Simple Twilio voice endpoint - handles incoming calls"""
    logging.info("[Twilio] New call received")

    try:
        form = await request.form()
        call_sid = form.get('CallSid', str(uuid.uuid4()))

        logging.info(f"[Twilio] CallSid: {call_sid}")
        logging.info(f"[Twilio] From: {form.get('From')}")

        # Create session for this call
        session_mgr = get_session_manager()
        if not session_mgr.get_session_info(call_sid):
            session_mgr.create_session(call_sid)
            logging.info(f"[Twilio] Created new session for call: {call_sid}")

        # Create simple TwiML response
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=f"/gather?call_sid={call_sid}",
            method="POST",
            timeout=20,  # Longer timeout for initial greeting
            speechTimeout="auto",  # Let Twilio detect end of speech automatically
            language="en-US"
        )

        # Clear greeting that prompts for speech with instructions
        gather.say("Hello! You've reached Harmony Health Clinic. I'm Sarah, your AI health assistant - how can I help you today?")
        response.append(gather)

        # Better fallback - don't hang up immediately, give another chance
        fallback_gather = Gather(
            input="speech",
            action=f"/gather?call_sid={call_sid}",
            method="POST",
            timeout=15,
            speechTimeout="auto",
            language="en-US"
        )
        fallback_gather.say("I didn't hear anything. Please speak now and tell me how I can help you.")
        response.append(fallback_gather)

        # Final message before hanging up
        response.say("Thank you for calling Harmony Health Clinic. Please call back when you're ready to speak. Goodbye.")

        logging.info(f"[Twilio] Sending TwiML response for call: {call_sid}")
        logging.info(f"[Twilio] TwiML: {str(response)}")  # Log the actual TwiML being sent
        return str(response)

    except Exception as e:
        logging.error(f"[Twilio] Error in voice endpoint: {e}")
        # Simple fallback
        response = VoiceResponse()
        response.say("I'm sorry, there was a technical issue. Please try calling again.")
        return str(response)

@app.post("/gather", response_class=PlainTextResponse)
async def gather(request: Request):
    """Simple gather endpoint - processes speech input"""
    request_start_time = time.time()
    logging.info(f"[LATENCY] Request received at {request_start_time:.3f}")
    logging.info("[Twilio] /gather endpoint called")

    try:
        form_parse_start = time.time()
        form = await request.form()
        call_sid = request.query_params.get('call_sid') or form.get('CallSid', str(uuid.uuid4()))
        transcript = form.get("SpeechResult", "")
        confidence = form.get("Confidence", "Unknown")

        form_parse_time = time.time() - form_parse_start
        logging.info(f"[LATENCY] Form parsing took {form_parse_time:.3f}s")

        logging.info(f"[Twilio] CallSid: {call_sid}")
        logging.info(f"[Twilio] Speech: '{transcript}'")
        logging.info(f"[Twilio] Confidence: {confidence}")
        logging.info(f"[Twilio] All form data: {dict(form)}")  # Debug: see all data from Twilio

        session_mgr = get_session_manager()
        response = VoiceResponse()

        if transcript:
            # Process the message through the health agent - detailed timing
            agent_start_time = time.time()
            logging.info(f"[LATENCY] Starting agent processing at {agent_start_time:.3f}")

            result = session_mgr.process_message(call_sid, transcript, voice_mode=True)
            agent_reply = result['reply']

            agent_end_time = time.time()
            agent_duration = agent_end_time - agent_start_time
            logging.info(f"[LATENCY] Agent processing completed in {agent_duration:.3f}s")
            logging.info(f"[Voice] Agent reply ({agent_duration:.2f}s): {agent_reply}")

            # Generate TTS audio - detailed timing
            try:
                tts_start_time = time.time()
                logging.info(f"[LATENCY] Starting TTS generation at {tts_start_time:.3f}")

                audio_filename = f"response_{call_sid}_{int(time.time())}.wav"
                audio_path = os.path.join("audio_files", audio_filename)

                # Use optimized TTS generation with MP3 encoding for Twilio
                logging.info(f"[Voice] Starting TTS generation for: {agent_reply[:50]}...")

                # Use Cartesia HTTP TTS (reliable and working)
                logging.info(f"[Voice] Using Cartesia TTS...")
                await synthesize_speech(agent_reply, audio_path)

                tts_end_time = time.time()
                tts_duration = tts_end_time - tts_start_time
                logging.info(f"[LATENCY] TTS generation completed in {tts_duration:.3f}s")
                logging.info(f"[Voice] TTS generated ({tts_duration:.2f}s): {audio_filename}")

                # Verify audio file was created and get size
                if os.path.exists(audio_path):
                    file_size = os.path.getsize(audio_path)
                    logging.info(f"[Voice] Audio file created: {file_size} bytes")
                    logging.info(f"[Voice] Audio file path: {audio_path}")
                    logging.info(f"[Voice] Audio file extension: {os.path.splitext(audio_filename)[1]}")
                else:
                    logging.error(f"[Voice] Audio file not created: {audio_path}")
                    raise Exception("TTS audio file generation failed")

                # Calculate total latency so far
                total_processing_time = time.time() - request_start_time
                logging.info(f"[LATENCY] TOTAL PROCESSING TIME: {total_processing_time:.3f}s (Form: {form_parse_time:.3f}s, Agent: {agent_duration:.3f}s, TTS: {tts_duration:.3f}s)")

                # Create audio URL - use ngrok URL for Twilio accessibility
                base_url = str(request.base_url).rstrip('/')
                audio_url = f"{base_url}/static/{audio_filename}"
                logging.info(f"[Voice] Audio URL: {audio_url}")

                # Create next gather
                gather = Gather(
                    input="speech",
                    action=f"/gather?call_sid={call_sid}",
                    method="POST",
                    timeout=15,  # Increased timeout
                    speechTimeout="auto",
                    language="en-US"
                )
                gather.play(audio_url)
                response.append(gather)

            except Exception as tts_error:
                logging.error(f"[Voice] TTS error: {tts_error}")
                # Fallback to text-to-speech
                gather = Gather(
                    input="speech",
                    action=f"/gather?call_sid={call_sid}",
                    method="POST",
                    timeout=15,  # Increased timeout
                    speechTimeout="auto",
                    language="en-US"
                )
                gather.say(agent_reply)
                response.append(gather)

        else:
            # No speech detected - keep the call alive with reprompting
            logging.info("[Twilio] No speech detected, reprompting user")

            # Get session info to track retry attempts
            session_info = session_mgr.get_session_info(call_sid)
            retry_count = session_info.get('no_speech_count', 0) if session_info else 0
            retry_count += 1

            # Update session with retry count
            if session_info:
                session_info['no_speech_count'] = retry_count

            logging.info(f"[Twilio] No speech retry count: {retry_count} for call: {call_sid}")

            if retry_count <= 2:  # Give user 2 more chances
                # Create another gather to keep the call alive
                gather = Gather(
                    input="speech",
                    action=f"/gather?call_sid={call_sid}",
                    method="POST",
                    timeout=15,
                    speechTimeout="auto",
                    language="en-US"
                )

                if retry_count == 1:
                    gather.say("I didn't hear anything. Please speak clearly and tell me how I can help you today. You can ask about appointments, insurance, or health questions.")
                else:
                    gather.say("I still don't hear anything. Please make sure you're speaking clearly. How can I help you today?")

                response.append(gather)

                # Encouraging message
                response.say("If you're having trouble, please try speaking right after you hear this message end.")
            else:
                # After 2 retries, be more helpful but still keep call alive
                gather = Gather(
                    input="speech",
                    action=f"/gather?call_sid={call_sid}",
                    method="POST",
                    timeout=20,  # Longer timeout for final attempt
                    speechTimeout="auto",
                    language="en-US"
                )
                gather.say("I'm having trouble hearing you. This might be due to background noise or connection issues. Please speak loudly and clearly. How can I help you today?")
                response.append(gather)

                # Final helpful message
                response.say("If you continue to have issues, please try calling from a quieter location or check your phone connection. Thank you for your patience.")

        return str(response)

    except Exception as e:
        logging.error(f"[Twilio] Error in gather endpoint: {e}")
        # Simple fallback
        response = VoiceResponse()
        response.say("I'm sorry, there was a technical issue. Please try again.")
        return str(response)

@app.post("/chat", response_class=JSONResponse)
async def chat(body: dict = Body(...)):
    """
    Chat endpoint for testing the healthcare assistant via text.
    Accepts JSON: {"message": "...", "session_id": "..."}
    Returns: {"reply": "...", "conversation_history": [...], "session_id": "...", "message_count": N}
    """
    message = body.get("message", "")
    session_id = body.get("session_id")

    # Get session manager
    session_mgr = get_session_manager()

    if message:
        # Process message through session manager
        result = session_mgr.process_message(session_id or str(uuid.uuid4()), message)
        logging.info(f"[Chat] [session_id={result['session_id']}] User: {message}")
        logging.info(f"[Chat] [session_id={result['session_id']}] Agent: {result['reply']}")
        return result
    else:
        # Create session for empty message
        if not session_id:
            session_id = str(uuid.uuid4())
        session_mgr.create_session(session_id)
        return {
            "reply": "Please provide a message.",
            "session_id": session_id,
            "message_count": 0,
            "conversation_history": []
        }

@app.get("/sessions", response_class=JSONResponse)
async def get_sessions():
    """Get all active sessions."""
    session_mgr = get_session_manager()
    return {
        "active_sessions": session_mgr.get_active_sessions(),
        "stats": session_mgr.get_stats()
    }

@app.get("/sessions/{session_id}", response_class=JSONResponse)
async def get_session(session_id: str):
    """Get specific session information."""
    session_mgr = get_session_manager()
    session_info = session_mgr.get_session_info(session_id)
    if session_info:
        return session_info
    else:
        return {"error": "Session not found or expired"}

@app.delete("/sessions/{session_id}", response_class=JSONResponse)
async def delete_session(session_id: str):
    """Delete a specific session."""
    session_mgr = get_session_manager()
    success = session_mgr.cleanup_session(session_id)
    return {"success": success, "message": f"Session {session_id} {'deleted' if success else 'not found'}"}

@app.post("/sessions/cleanup", response_class=JSONResponse)
async def cleanup_sessions():
    """Clean up all expired sessions."""
    session_mgr = get_session_manager()
    cleaned_count = session_mgr.cleanup_expired_sessions()
    return {"cleaned_sessions": cleaned_count}

@app.get("/static/{filename}")
async def static_audio(filename: str, request: Request):
    """Simple static audio file serving"""
    logging.info(f"[Twilio] Serving audio file: {filename}")

    # Check in audio_files directory first
    audio_files_path = os.path.join("audio_files", filename)

    if os.path.exists(audio_files_path):
        logging.info(f"[Twilio] Found audio file: {filename}")
        # Determine media type based on file extension
        media_type = "audio/mpeg" if filename.endswith('.mp3') else "audio/wav"
        content_type = "audio/mpeg" if filename.endswith('.mp3') else "audio/wav"

        return FileResponse(
            audio_files_path,
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
                "Content-Type": content_type,
                "Accept-Ranges": "bytes"  # Enable range requests for better streaming
            }
        )

    # Check in current directory as fallback
    if os.path.exists(filename):
        logging.info(f"[Twilio] Found audio file in current dir: {filename}")
        return FileResponse(
            filename,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )

    # File not found
    logging.error(f"[Twilio] Audio file not found: {filename}")
    raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

# Simple status webhook for call completion logging
@app.post("/voice/status", response_class=PlainTextResponse)
async def voice_status(request: Request):
    """Handle Twilio call status webhooks"""
    try:
        form = await request.form()
        call_sid = form.get('CallSid')
        call_status = form.get('CallStatus')

        logging.info(f"[Twilio] Call status: {call_sid} - {call_status}")

        if call_status in ['completed', 'busy', 'no-answer', 'failed', 'canceled']:
            logging.info(f"[Twilio] Call ended: {call_sid}")

        return ""

    except Exception as e:
        logging.error(f"[Twilio] Error in status webhook: {e}")
        return ""

