#!/usr/bin/env python3
"""
Streamlit chatbot for testing the healthcare assistant conversation flow.
"""

import json
import os
import sys
import uuid
from datetime import datetime

import requests
import streamlit as st

# Add the project root to the Python path for direct testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
FASTAPI_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{FASTAPI_BASE_URL}/chat"

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "use_api" not in st.session_state:
        st.session_state.use_api = True

def send_message_to_api(message: str, session_id: str) -> dict:
    """Send message to FastAPI chat endpoint."""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "reply": f"API Error: {response.status_code} - {response.text}",
                "history": [],
                "session_id": session_id
            }
    except requests.exceptions.ConnectionError:
        return {
            "reply": "❌ Cannot connect to FastAPI server. Please start the server with: uvicorn app.voice.twilio_deepgram_router:app --reload --port 8000",
            "history": [],
            "session_id": session_id
        }
    except Exception as e:
        return {
            "reply": f"Error: {str(e)}",
            "history": [],
            "session_id": session_id
        }

def send_message_direct(message: str) -> str:
    """Send message directly to HealthAgent (for testing without API)."""
    try:
        from app.ai.health_agent import HealthAgent
        
        # Get or create agent for this session
        if "direct_agent" not in st.session_state:
            st.session_state.direct_agent = HealthAgent()
        
        agent = st.session_state.direct_agent
        response = agent.process(message)
        return response
    except Exception as e:
        return f"Direct Error: {str(e)}"

def display_message(role: str, content: str, timestamp: str = None):
    """Display a message in the chat interface."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        with st.chat_message("user"):
            st.write(f"**You** ({timestamp})")
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(f"**Healthcare Assistant** ({timestamp})")
            st.write(content)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Healthcare Assistant Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🏥 Healthcare Assistant")
        st.markdown("---")
        
        # Connection mode
        st.subheader("Connection Mode")
        use_api = st.radio(
            "Choose how to test:",
            ["FastAPI Endpoint", "Direct HealthAgent"],
            index=0 if st.session_state.use_api else 1,
            help="FastAPI tests the full server setup, Direct tests just the HealthAgent"
        )
        st.session_state.use_api = (use_api == "FastAPI Endpoint")
        
        # Session info
        st.subheader("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            if "direct_agent" in st.session_state:
                del st.session_state.direct_agent
            st.rerun()
        
        # Test scenarios
        st.subheader("🧪 Test Scenarios")
        st.markdown("Click to test specific scenarios:")
        
        test_scenarios = [
            ("👋 First Greeting", "Hello"),
            ("🏥 Clinic Hours", "What are your clinic hours?"),
            ("♿ Wheelchair Access", "Do you have wheelchair access?"),
            ("💳 Insurance Check", "Do you accept Aetna insurance?"),
            ("📅 Book Appointment", "I need to schedule an appointment for July 10th at 2 PM"),
            ("🤒 Health Question", "I have a headache and fever. What should I do?"),
            ("📍 Location", "Where is the clinic located?"),
            ("🚗 Parking", "Is parking available?"),
        ]
        
        for label, message in test_scenarios:
            if st.button(label, key=f"test_{label}"):
                # Add user message
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "user",
                    "content": message,
                    "timestamp": timestamp
                })
                
                # Get response
                if st.session_state.use_api:
                    result = send_message_to_api(message, st.session_state.session_id)
                    reply = result.get("reply", "No response")
                else:
                    reply = send_message_direct(message)
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.rerun()
    
    # Main chat interface
    st.title("🏥 Healthcare Assistant Chatbot")
    
    # Connection status
    if st.session_state.use_api:
        st.info("🌐 Using FastAPI endpoint - Make sure server is running on http://localhost:8000")
    else:
        st.info("🔧 Using direct HealthAgent - Testing without server")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"], 
                message.get("timestamp")
            )
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message immediately
        with chat_container:
            display_message("user", prompt, timestamp)
        
        # Get response
        with st.spinner("Healthcare Assistant is thinking..."):
            if st.session_state.use_api:
                result = send_message_to_api(prompt, st.session_state.session_id)
                reply = result.get("reply", "No response")
                
                # Show additional info in expander
                if "history" in result and len(result["history"]) > 0:
                    with st.expander("📋 Conversation History (from API)"):
                        st.json(result["history"])
            else:
                reply = send_message_direct(prompt)
        
        # Add and display assistant response
        assistant_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "timestamp": assistant_timestamp
        })
        
        with chat_container:
            display_message("assistant", reply, assistant_timestamp)
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Try asking about clinic hours, booking appointments, insurance verification, or general health questions. "
        "The assistant will greet you on first interaction and maintain conversation history."
    )

if __name__ == "__main__":
    main()
