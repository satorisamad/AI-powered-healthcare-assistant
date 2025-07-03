"""
TTS Configuration and Provider Management

This module provides a unified interface for switching between different TTS providers
(Deepgram, Cartesia) and managing their configurations.
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optimized output format for phone calls using μ-law encoding
PHONE_OPTIMIZED_FORMAT = {
    "container": "raw",
    "encoding": "pcm_mulaw",  # μ-law encoding optimized for telephony
    "sample_rate": 8000       # Standard telephony sample rate
}

class TTSProvider(Enum):
    """Available TTS providers - Cartesia only"""
    CARTESIA = "cartesia"

@dataclass
class TTSConfig:
    """Unified TTS configuration with test.py optimizations"""
    provider: TTSProvider = TTSProvider.CARTESIA  # Default to Cartesia for better performance

    # Cartesia-specific settings (only TTS provider)
    cartesia_api_key: str = ""  # Use environment variable
    cartesia_model: str = "sonic-2"
    cartesia_voice_id: str = "bf0a246a-8642-498a-9950-80c35e9276b5"  # Your preferred voice ID
    cartesia_speed: str = "fast"  # OPTIMIZED: Use fast speed for lower latency

    # Common settings
    language: str = "en"
    cache_enabled: bool = True
    use_streaming: bool = False  # OPTIMIZED: HTTP more reliable than WebSocket in production
    fallback_provider: Optional[TTSProvider] = None  # REMOVED: No fallback, Cartesia only

    # Performance settings - OPTIMIZED from test.py
    max_concurrent_requests: int = 2  # Match Cartesia plan limit
    request_timeout: int = 15  # Shorter timeout for faster fallback

    # Voice interaction timing - from test.py optimizations
    tts_grace_seconds: float = 3.0      # Grace period before allowing barge-in
    back_off_seconds: float = 2.0       # Delay between responses
    speech_end_timeout: float = 0.5     # Trigger LLM after 500ms silence
    audio_chunk_size: int = 960         # 120ms audio chunks for analysis
    
    @classmethod
    def from_env(cls) -> 'TTSConfig':
        """Create configuration from environment variables - Cartesia only"""
        return cls(
            provider=TTSProvider.CARTESIA,  # Force Cartesia only
            cartesia_api_key=os.getenv("CARTESIA_API_KEY", cls.cartesia_api_key),
            cartesia_model=os.getenv("CARTESIA_MODEL", cls.cartesia_model),
            cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", cls.cartesia_voice_id),
            cartesia_speed=os.getenv("CARTESIA_SPEED", cls.cartesia_speed),
            language=os.getenv("TTS_LANGUAGE", cls.language),
            cache_enabled=os.getenv("TTS_CACHE_ENABLED", "true").lower() == "true",
            use_streaming=os.getenv("TTS_USE_STREAMING", "true").lower() == "true",
        )

class TTSManager:
    """Unified TTS manager that can switch between providers"""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._providers = {}
        self._current_provider = None
        
    async def _get_provider(self, provider_type: TTSProvider):
        """Get or create a TTS provider instance - Cartesia only"""
        if provider_type not in self._providers:
            if provider_type == TTSProvider.CARTESIA:
                # Use the ultra-optimized Cartesia implementation with your proven WebSocket approach
                from app.voice.cartesia_tts_optimized import OptimizedCartesiaTTS, OptimizedCartesiaConfig
                cartesia_config = OptimizedCartesiaConfig(
                    api_key=self.config.cartesia_api_key,
                    model_id=self.config.cartesia_model,
                    voice_id=self.config.cartesia_voice_id,
                    language=self.config.language,
                    speed="slow",  # Use your proven speed setting
                    output_format=PHONE_OPTIMIZED_FORMAT,  # Use μ-law encoding
                    # Apply test.py optimization parameters
                    tts_grace_seconds=3.0,
                    back_off_seconds=2.0,
                    speech_end_timeout=0.5,
                    audio_chunk_size=960
                )
                self._providers[provider_type] = OptimizedCartesiaTTS(cartesia_config)
            else:
                raise ValueError(f"Unsupported TTS provider: {provider_type}. Only Cartesia is supported.")

        return self._providers[provider_type]
    
    async def synthesize_speech(self, text: str, output_path: str = "output.mp3") -> str:
        """
        Synthesize speech using Cartesia TTS only (no fallback)
        """
        primary_provider = self.config.provider

        # Only use Cartesia - no fallback logic
        return await self._synthesize_with_provider(primary_provider, text, output_path)
    
    async def _synthesize_with_provider(self, provider_type: TTSProvider, text: str, output_path: str) -> str:
        """Synthesize speech with a specific provider"""
        try:
            if provider_type == TTSProvider.CARTESIA:
                provider = await self._get_provider(provider_type)
                # Use the ultra-optimized WebSocket streaming implementation
                return await provider.synthesize_speech_streaming(text, output_path)

            elif provider_type == TTSProvider.DEEPGRAM:
                provider = await self._get_provider(provider_type)

                # Use the fastest Deepgram function available
                if hasattr(provider, 'synthesize_speech_lightning_async'):
                    return await provider.synthesize_speech_lightning_async(text, output_path)
                elif hasattr(provider, 'synthesize_speech_async'):
                    return await provider.synthesize_speech_async(text, output_path)
                else:
                    # Fallback to sync version
                    import asyncio
                    return await asyncio.to_thread(provider.synthesize_speech, text, output_path)

            else:
                raise ValueError(f"Unsupported TTS provider: {provider_type}")

        except Exception as e:
            logger.error(f"[TTSManager] Provider {provider_type.value} failed: {e}")
            raise
    
    async def close(self):
        """Clean up provider resources"""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        self._providers.clear()

# Global TTS manager instance
_tts_manager: Optional[TTSManager] = None

async def get_tts_manager() -> TTSManager:
    """Get or create the global TTS manager"""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager()
    return _tts_manager

# Convenience functions for backward compatibility
async def synthesize_speech_unified(text: str, output_path: str = "output.mp3") -> str:
    """Unified TTS function that uses the configured provider"""
    manager = await get_tts_manager()
    return await manager.synthesize_speech(text, output_path)

def synthesize_speech_unified_sync(text: str, output_path: str = "output.mp3") -> str:
    """Synchronous version of unified TTS"""
    import asyncio
    return asyncio.run(synthesize_speech_unified(text, output_path))

# Provider-specific convenience functions
async def synthesize_speech_cartesia(text: str, output_path: str = "output.mp3") -> str:
    """Use Cartesia TTS directly"""
    from app.voice.cartesia_tts_optimized import synthesize_speech_optimized
    return await synthesize_speech_optimized(text, output_path)

def get_available_providers() -> Dict[str, bool]:
    """Check which TTS providers are available - Cartesia only"""
    providers = {}

    # Check Cartesia only
    try:
        cartesia_key = os.getenv("CARTESIA_API_KEY", "")
        providers["cartesia"] = bool(cartesia_key)
    except Exception:
        providers["cartesia"] = False

    return providers

def print_tts_status():
    """Print current TTS configuration and provider status"""
    config = TTSConfig.from_env()
    providers = get_available_providers()
    
    print("🎤 TTS Configuration Status")
    print("=" * 40)
    print(f"Primary Provider: {config.provider.value}")
    print(f"Fallback Provider: {config.fallback_provider.value if config.fallback_provider else 'None'}")
    print(f"Language: {config.language}")
    print(f"Caching: {'Enabled' if config.cache_enabled else 'Disabled'}")
    print(f"Streaming: {'Enabled' if config.use_streaming else 'Disabled'}")
    print()
    
    print("📡 Provider Availability")
    print("=" * 40)
    for provider, available in providers.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{provider.capitalize()}: {status}")
    
    if config.provider.value not in providers or not providers[config.provider.value]:
        print(f"\n⚠️ WARNING: Primary provider '{config.provider.value}' is not available!")
    
    print()

if __name__ == "__main__":
    print_tts_status()
