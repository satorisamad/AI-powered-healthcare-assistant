#!/usr/bin/env python3
"""
Session Manager for handling multiple user conversations with the healthcare assistant.
"""

import logging
import uuid
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.ai.health_agent import HealthAgent

class SessionManager:
    """Manages multiple user sessions and their conversation histories."""
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize the session manager.
        
        Args:
            session_timeout_minutes: How long to keep inactive sessions (default: 30 minutes)
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        logging.info(f"[SessionManager] Initialized with {session_timeout_minutes} minute timeout")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session or return existing session ID.
        
        Args:
            session_id: Optional existing session ID
            
        Returns:
            Session ID (new or existing)
        """
        if session_id and session_id in self.sessions:
            # Update last activity time for existing session
            self.sessions[session_id]['last_activity'] = datetime.now()
            logging.info(f"[SessionManager] Reusing existing session: {session_id}")
            return session_id
        
        # Create new session
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'agent': HealthAgent(),
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0,
            'user_info': {}  # Store user details like name, etc.
        }
        
        logging.info(f"[SessionManager] Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found/expired
        """
        if session_id not in self.sessions:
            logging.warning(f"[SessionManager] Session not found: {session_id}")
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() - session['last_activity'] > self.session_timeout:
            logging.info(f"[SessionManager] Session expired: {session_id}")
            self.cleanup_session(session_id)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return session
    
    def process_message(self, session_id: str, message: str, voice_mode: bool = False) -> Dict:
        """
        Process a message for a specific session.

        Args:
            session_id: Session identifier
            message: User message
            voice_mode: Whether this is a voice interaction (enables optimizations)

        Returns:
            Response dictionary with reply, session info, etc.
        """
        import time
        session_start_time = time.time()
        logging.info(f"[LATENCY] SessionManager.process_message started at {session_start_time:.3f}")

        # Get or create session
        session_lookup_start = time.time()
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session(session_id)
            session = self.sessions[session_id]
        session_lookup_time = time.time() - session_lookup_start
        logging.info(f"[LATENCY] Session lookup/creation took {session_lookup_time:.3f}s")

        try:
            # Process message with the session's agent, passing voice_mode
            agent_process_start = time.time()
            logging.info(f"[LATENCY] Starting agent.process at {agent_process_start:.3f}")

            agent = session['agent']
            response = agent.process(message, voice_mode=voice_mode)

            agent_process_end = time.time()
            agent_process_time = agent_process_end - agent_process_start
            logging.info(f"[LATENCY] Agent.process completed in {agent_process_time:.3f}s")

            # Update session stats
            session['message_count'] += 1
            session['last_activity'] = datetime.now()

            # Extract user info from conversation if available
            self._extract_user_info(session, message, response)

            total_session_time = time.time() - session_start_time
            logging.info(f"[LATENCY] TOTAL SESSION PROCESSING TIME: {total_session_time:.3f}s (Lookup: {session_lookup_time:.3f}s, Agent: {agent_process_time:.3f}s)")
            logging.info(f"[SessionManager] Processed message for session {session_id}: {len(message)} chars -> {len(response)} chars")

            return {
                'session_id': session_id,
                'reply': response,
                'message_count': session['message_count'],
                'conversation_history': agent.history,
                'user_info': session['user_info'],
                'session_duration': str(datetime.now() - session['created_at'])
            }

        except Exception as e:
            logging.error(f"[SessionManager] Error processing message for session {session_id}: {e}")
            return {
                'session_id': session_id,
                'reply': "I'm sorry, I'm having trouble processing your request right now. Please try again.",
                'error': str(e)
            }
    
    def _extract_user_info(self, session: Dict, message: str, response: str):
        """Extract user information from conversation."""
        message_lower = message.lower()
        
        # Extract name if mentioned
        if 'my name is' in message_lower:
            try:
                name_part = message_lower.split('my name is')[1].strip()
                # Take first few words as name
                name = ' '.join(name_part.split()[:3])
                session['user_info']['name'] = name.title()
            except:
                pass
        
        # Extract other info as needed
        if 'insurance' in message_lower:
            session['user_info']['discussed_insurance'] = True
        
        if 'appointment' in message_lower:
            session['user_info']['discussed_appointment'] = True
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session information without processing a message.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session info or None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'message_count': session['message_count'],
            'user_info': session['user_info'],
            'conversation_length': len(session['agent'].history),
            'session_duration': str(datetime.now() - session['created_at'])
        }
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Remove a session from memory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logging.info(f"[SessionManager] Cleaned up session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if now - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        if expired_sessions:
            logging.info(f"[SessionManager] Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_active_sessions(self) -> List[Dict]:
        """
        Get list of all active sessions.
        
        Returns:
            List of session info dictionaries
        """
        active_sessions = []
        for session_id in list(self.sessions.keys()):
            session_info = self.get_session_info(session_id)
            if session_info:
                active_sessions.append(session_info)
        
        return active_sessions
    
    def get_stats(self) -> Dict:
        """
        Get overall session manager statistics.
        
        Returns:
            Statistics dictionary
        """
        active_sessions = self.get_active_sessions()
        total_messages = sum(session['message_count'] for session in active_sessions)
        
        return {
            'active_sessions': len(active_sessions),
            'total_messages_processed': total_messages,
            'session_timeout_minutes': self.session_timeout.total_seconds() / 60,
            'oldest_session': min([s['created_at'] for s in active_sessions]) if active_sessions else None,
            'most_active_session': max(active_sessions, key=lambda x: x['message_count']) if active_sessions else None
        }

# Global session manager instance
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return session_manager
