"""
Audio file management system for voice optimization.
Handles audio file cleanup, compression, and efficient serving.
"""

import os
import logging
import time
import threading
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AudioFileManager:
    """Manages audio files for voice interactions with optimization features."""
    
    def __init__(self, audio_dir: str = "audio_files", max_files: int = 100, cleanup_interval: int = 300):
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)
        self.max_files = max_files
        self.cleanup_interval = cleanup_interval  # seconds
        
        # Track active files
        self.active_files: Dict[str, Dict] = {}
        self.file_lock = threading.Lock()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.should_cleanup = True
        self.start_cleanup_thread()
        
        # Compression executor
        self.compression_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio-compress")
        
        logger.info(f"[AudioManager] Initialized with directory: {self.audio_dir}")
    
    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("[AudioManager] Started cleanup thread")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.should_cleanup:
            try:
                self.cleanup_old_files()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"[AudioManager] Cleanup error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def generate_audio_path(self, call_sid: str, content: Optional[str] = None) -> str:
        """Generate optimized audio file path."""
        timestamp = int(time.time())

        if content:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            filename = f"audio_{content_hash[:8]}_{timestamp}.mp3"
        else:
            # Use call_sid for unique files (clean the call_sid for filename safety)
            clean_call_sid = call_sid.replace('CA', '').replace('-', '')[:10]
            filename = f"agent_reply_{clean_call_sid}_{timestamp}.mp3"

        return str(self.audio_dir / filename)
    
    def register_file(self, file_path: str, call_sid: str, content: str = "", ttl_minutes: int = 30) -> str:
        """Register an audio file for tracking and cleanup."""
        with self.file_lock:
            file_info = {
                'path': file_path,
                'call_sid': call_sid,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=ttl_minutes),
                'content_hash': hashlib.md5(content.encode()).hexdigest() if content else None,
                'size_bytes': 0,
                'access_count': 0,
                'last_accessed': datetime.now()
            }
            
            # Update file size if file exists
            if os.path.exists(file_path):
                file_info['size_bytes'] = os.path.getsize(file_path)
            
            self.active_files[file_path] = file_info
            logger.info(f"[AudioManager] Registered file: {os.path.basename(file_path)}")
            
            return file_path
    
    def access_file(self, file_path: str) -> bool:
        """Mark file as accessed and update stats."""
        with self.file_lock:
            if file_path in self.active_files:
                self.active_files[file_path]['access_count'] += 1
                self.active_files[file_path]['last_accessed'] = datetime.now()
                return True
            return False
    
    def cleanup_old_files(self, force_cleanup: bool = False) -> int:
        """Clean up expired and old audio files."""
        cleaned_count = 0
        current_time = datetime.now()
        
        with self.file_lock:
            expired_files = []
            
            # Find expired files
            for file_path, info in self.active_files.items():
                if current_time > info['expires_at'] or force_cleanup:
                    expired_files.append(file_path)
            
            # If we have too many files, remove oldest ones
            if len(self.active_files) > self.max_files:
                sorted_files = sorted(
                    self.active_files.items(),
                    key=lambda x: (x[1]['access_count'], x[1]['last_accessed'])
                )
                
                # Remove least accessed, oldest files
                excess_count = len(self.active_files) - self.max_files
                for file_path, _ in sorted_files[:excess_count]:
                    if file_path not in expired_files:
                        expired_files.append(file_path)
            
            # Remove expired files
            for file_path in expired_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"[AudioManager] Deleted file: {os.path.basename(file_path)}")
                    
                    del self.active_files[file_path]
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"[AudioManager] Error deleting {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"[AudioManager] Cleaned up {cleaned_count} audio files")
        
        return cleaned_count
    
    def get_file_stats(self) -> Dict:
        """Get audio file statistics."""
        with self.file_lock:
            total_size = sum(info['size_bytes'] for info in self.active_files.values())
            total_files = len(self.active_files)
            
            # Calculate average access count
            avg_access = sum(info['access_count'] for info in self.active_files.values()) / total_files if total_files > 0 else 0
            
            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'average_access_count': round(avg_access, 2),
                'max_files': self.max_files,
                'cleanup_interval_seconds': self.cleanup_interval
            }
    
    def compress_audio_async(self, file_path: str, target_bitrate: int = 64) -> str:
        """Compress audio file asynchronously (placeholder for future implementation)."""
        # This would use ffmpeg or similar tool to compress audio
        # For now, just return the original path
        logger.info(f"[AudioManager] Audio compression requested for: {os.path.basename(file_path)}")
        return file_path
    
    def deduplicate_content(self, content: str) -> Optional[str]:
        """Check if we already have audio for this content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with self.file_lock:
            for file_path, info in self.active_files.items():
                if info['content_hash'] == content_hash and os.path.exists(file_path):
                    # Update access info
                    info['access_count'] += 1
                    info['last_accessed'] = datetime.now()
                    logger.info(f"[AudioManager] Found duplicate content, reusing: {os.path.basename(file_path)}")
                    return file_path
        
        return None
    
    def shutdown(self):
        """Shutdown the audio manager and cleanup resources."""
        logger.info("[AudioManager] Shutting down...")
        self.should_cleanup = False
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        self.compression_executor.shutdown(wait=True)
        
        # Final cleanup
        self.cleanup_old_files(force_cleanup=True)
        
        logger.info("[AudioManager] Shutdown complete")

# Global audio manager instance
_audio_manager = None

def get_audio_manager() -> AudioFileManager:
    """Get the global audio manager instance."""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioFileManager()
    return _audio_manager

def cleanup_audio_manager():
    """Cleanup the global audio manager."""
    global _audio_manager
    if _audio_manager:
        _audio_manager.shutdown()
        _audio_manager = None
