"""
Intelligent response caching system for healthcare assistant.
Caches common responses to reduce LLM calls and improve voice response times.
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ResponseCache:
    """Intelligent caching system for healthcare responses."""
    
    def __init__(self, cache_file: str = "response_cache.json", max_cache_size: int = 1000):
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.load_cache()
        
        # Common healthcare patterns that should be cached
        self.cacheable_patterns = [
            "hours", "location", "address", "phone", "insurance", "appointment",
            "schedule", "cancel", "reschedule", "doctor", "clinic", "parking",
            "directions", "cost", "price", "payment", "emergency", "urgent",
            "walk-in", "same day", "availability", "book", "confirm"
        ]
        
        # Responses that should never be cached (personalized)
        self.non_cacheable_patterns = [
            "name", "birthday", "birth date", "ssn", "social security",
            "address", "phone number", "email", "personal", "private"
        ]
    
    def _generate_cache_key(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a cache key for the prompt and context."""
        # Normalize the prompt
        normalized = prompt.lower().strip()
        
        # Include context if provided (for conversation-aware caching)
        if context:
            normalized += f"|context:{context.lower().strip()}"
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _is_cacheable(self, prompt: str) -> bool:
        """Determine if a prompt should be cached."""
        prompt_lower = prompt.lower()
        
        # Don't cache if it contains personal information patterns
        if any(pattern in prompt_lower for pattern in self.non_cacheable_patterns):
            return False
        
        # Cache if it contains common healthcare patterns
        if any(pattern in prompt_lower for pattern in self.cacheable_patterns):
            return True
        
        # Cache short, common questions
        if len(prompt.split()) <= 10 and any(word in prompt_lower for word in ["what", "when", "where", "how", "can", "do", "is", "are"]):
            return True
        
        return False
    
    def get(self, prompt: str, context: Optional[str] = None, max_age_hours: int = 24) -> Optional[str]:
        """Get cached response if available and not expired."""
        if not self._is_cacheable(prompt):
            return None
        
        cache_key = self._generate_cache_key(prompt, context)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if entry is expired
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                # Remove expired entry
                del self.cache[cache_key]
                self.miss_count += 1
                return None
            
            # Update access info
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.now().isoformat()
            
            self.hit_count += 1
            logger.info(f"[ResponseCache] Cache HIT for: {prompt[:50]}...")
            return entry['response']
        
        self.miss_count += 1
        return None
    
    def put(self, prompt: str, response: str, context: Optional[str] = None) -> bool:
        """Cache a response if it's cacheable."""
        if not self._is_cacheable(prompt):
            return False
        
        cache_key = self._generate_cache_key(prompt, context)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # Store the response
        self.cache[cache_key] = {
            'prompt': prompt,
            'response': response,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'access_count': 1,
            'last_accessed': datetime.now().isoformat()
        }
        
        logger.info(f"[ResponseCache] Cached response for: {prompt[:50]}...")
        
        # Periodically save to disk
        if len(self.cache) % 10 == 0:
            self.save_cache()
        
        return True
    
    def _evict_oldest(self):
        """Remove the oldest or least accessed entries."""
        if not self.cache:
            return
        
        # Sort by access count (ascending) then by timestamp (ascending)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1]['access_count'], x[1]['timestamp'])
        )
        
        # Remove the least accessed, oldest entry
        oldest_key = sorted_entries[0][0]
        del self.cache[oldest_key]
        logger.info("[ResponseCache] Evicted oldest cache entry")
    
    def load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.hit_count = data.get('hit_count', 0)
                    self.miss_count = data.get('miss_count', 0)
                logger.info(f"[ResponseCache] Loaded {len(self.cache)} cached responses")
        except Exception as e:
            logger.error(f"[ResponseCache] Error loading cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                'cache': self.cache,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'last_saved': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"[ResponseCache] Saved {len(self.cache)} cached responses")
        except Exception as e:
            logger.error(f"[ResponseCache] Error saving cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def clear_expired(self, max_age_hours: int = 24):
        """Clear expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if current_time - cached_time > timedelta(hours=max_age_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"[ResponseCache] Cleared {len(expired_keys)} expired entries")
        
        return len(expired_keys)

# Global cache instance
_response_cache = None

def get_response_cache() -> ResponseCache:
    """Get the global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
