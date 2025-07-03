"""
Advanced barge-in detection system for natural conversation flow.
Implements intelligent audio analysis to detect when users want to interrupt.
"""

import time
import logging
import audioop
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioMetrics:
    """Audio analysis metrics for barge-in detection."""
    rms_energy: float
    db_level: float
    zero_crossing_rate: float
    spectral_centroid: float
    is_speech: bool
    confidence: float

class BargeInDetector:
    """
    Advanced barge-in detection with multiple audio analysis techniques.
    """
    
    def __init__(self, 
                 speech_threshold_db: float = -25.0,
                 min_speech_duration: float = 0.25,
                 max_silence_gap: float = 0.5,
                 confidence_threshold: float = 0.7):
        """
        Initialize barge-in detector with configurable parameters.
        
        Args:
            speech_threshold_db: Minimum dB level to consider as speech
            min_speech_duration: Minimum duration to confirm speech intent
            max_silence_gap: Maximum silence gap before resetting detection
            confidence_threshold: Minimum confidence to trigger barge-in
        """
        self.speech_threshold_db = speech_threshold_db
        self.min_speech_duration = min_speech_duration
        self.max_silence_gap = max_silence_gap
        self.confidence_threshold = confidence_threshold
        
        # Audio analysis state
        self.audio_buffer = deque(maxlen=20)  # Keep last 20 chunks (~2.4s at 120ms chunks)
        self.speech_start_time = None
        self.last_speech_time = None
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        
        # Adaptive thresholds
        self.background_noise_level = -40.0  # dB
        self.adaptive_threshold = speech_threshold_db
        
        # Statistics
        self.total_detections = 0
        self.false_positives = 0
        self.true_positives = 0
        
        logger.info(f"[BargeInDetector] Initialized with threshold: {speech_threshold_db}dB")
    
    def analyze_audio_chunk(self, audio: bytes) -> AudioMetrics:
        """
        Analyze audio chunk and extract speech detection metrics.
        """
        try:
            if not audio:
                logger.debug(f"[BargeInDetector] Empty audio chunk received")
                return AudioMetrics(0, -float('inf'), 0, 0, False, 0)

            # Convert µ-law to 16-bit PCM
            pcm_audio = audioop.ulaw2lin(audio, 2)

            # Calculate RMS energy
            rms = audioop.rms(pcm_audio, 2)
            db_level = 20 * np.log10(rms / 32768) if rms > 0 else -float('inf')

            # Convert to numpy array for advanced analysis
            audio_array = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Zero crossing rate (indicates speech vs noise)
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array) if len(audio_array) > 0 else 0

            # Simple spectral centroid approximation
            # (In a full implementation, you'd use FFT)
            spectral_centroid = np.mean(np.abs(np.diff(audio_array))) * 1000

            # Speech detection logic
            is_speech = self._is_speech_like(db_level, zcr, spectral_centroid)

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(db_level, zcr, spectral_centroid, is_speech)

            # Enhanced logging for audio analysis
            logger.debug(f"[BargeInDetector] Audio Analysis: "
                        f"dB={db_level:.1f}, ZCR={zcr:.3f}, SC={spectral_centroid:.1f}, "
                        f"Speech={is_speech}, Confidence={confidence:.2f}, "
                        f"Threshold={self.adaptive_threshold:.1f}, "
                        f"Background={self.background_noise_level:.1f}")

            return AudioMetrics(
                rms_energy=rms,
                db_level=db_level,
                zero_crossing_rate=zcr,
                spectral_centroid=spectral_centroid,
                is_speech=is_speech,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"[BargeInDetector] Audio analysis error: {e}")
            return AudioMetrics(0, -float('inf'), 0, 0, False, 0)
    
    def _is_speech_like(self, db_level: float, zcr: float, spectral_centroid: float) -> bool:
        """
        Determine if audio characteristics indicate speech.
        """
        # Basic energy threshold
        if db_level < self.adaptive_threshold:
            return False
        
        # Speech typically has moderate zero crossing rate (not too high like noise)
        if zcr > 0.3:  # Too much noise
            return False
        
        # Speech has certain spectral characteristics
        if spectral_centroid < 10 or spectral_centroid > 1000:  # Outside typical speech range
            return False
        
        return True
    
    def _calculate_confidence(self, db_level: float, zcr: float, 
                            spectral_centroid: float, is_speech: bool) -> float:
        """
        Calculate confidence score for speech detection.
        """
        if not is_speech:
            return 0.0
        
        confidence = 0.0
        
        # Energy confidence (higher energy = higher confidence)
        energy_conf = min(1.0, max(0.0, (db_level - self.adaptive_threshold) / 10.0))
        confidence += energy_conf * 0.4
        
        # ZCR confidence (moderate ZCR is good for speech)
        zcr_conf = 1.0 - abs(zcr - 0.1) / 0.2  # Optimal around 0.1
        zcr_conf = max(0.0, min(1.0, zcr_conf))
        confidence += zcr_conf * 0.3
        
        # Spectral confidence
        spectral_conf = 1.0 - abs(spectral_centroid - 200) / 300  # Optimal around 200
        spectral_conf = max(0.0, min(1.0, spectral_conf))
        confidence += spectral_conf * 0.3
        
        return min(1.0, confidence)
    
    def update_adaptive_threshold(self, metrics: AudioMetrics):
        """
        Update adaptive threshold based on background noise.
        """
        if not metrics.is_speech and metrics.db_level > -50:
            old_noise_level = self.background_noise_level
            old_threshold = self.adaptive_threshold

            # Update background noise estimate
            self.background_noise_level = (self.background_noise_level * 0.95 +
                                         metrics.db_level * 0.05)

            # Adjust adaptive threshold
            self.adaptive_threshold = self.background_noise_level + 15  # 15dB above noise
            self.adaptive_threshold = max(self.adaptive_threshold, self.speech_threshold_db)

            # Log significant changes in noise levels
            if abs(self.background_noise_level - old_noise_level) > 2.0:
                logger.info(f"[BargeInDetector] 🔧 NOISE ADAPTATION: "
                           f"Background noise: {old_noise_level:.1f} → {self.background_noise_level:.1f}dB, "
                           f"Threshold: {old_threshold:.1f} → {self.adaptive_threshold:.1f}dB")
    
    def detect_barge_in(self, audio: bytes, tts_active: bool = False, 
                       grace_period_active: bool = False) -> Tuple[bool, float, Dict]:
        """
        Main barge-in detection method.
        
        Args:
            audio: Raw audio bytes (µ-law encoded)
            tts_active: Whether TTS is currently playing
            grace_period_active: Whether we're in TTS grace period
            
        Returns:
            Tuple of (should_barge_in, confidence, debug_info)
        """
        current_time = time.time()
        
        # Analyze current audio chunk
        metrics = self.analyze_audio_chunk(audio)
        self.audio_buffer.append((current_time, metrics))
        
        # Update adaptive threshold
        self.update_adaptive_threshold(metrics)
        
        # Don't detect during grace period
        if grace_period_active:
            logger.debug(f"[BargeInDetector] Grace period active - blocking detection")
            return False, 0.0, {"reason": "grace_period_active"}

        # Track speech continuity
        if metrics.is_speech and metrics.confidence > self.confidence_threshold:
            if self.speech_start_time is None:
                self.speech_start_time = current_time
                logger.info(f"[BargeInDetector] 🎤 SPEECH START detected! "
                           f"dB={metrics.db_level:.1f}, confidence={metrics.confidence:.2f}")
            self.last_speech_time = current_time
            self.consecutive_speech_chunks += 1
            self.consecutive_silence_chunks = 0

            logger.debug(f"[BargeInDetector] Speech chunk #{self.consecutive_speech_chunks}, "
                        f"duration={current_time - self.speech_start_time:.2f}s")
        else:
            self.consecutive_silence_chunks += 1

            if metrics.db_level > -40:  # Only log if not complete silence
                logger.debug(f"[BargeInDetector] Non-speech: dB={metrics.db_level:.1f}, "
                            f"ZCR={metrics.zero_crossing_rate:.3f}, "
                            f"reason={'low_confidence' if metrics.is_speech else 'not_speech'}")

            # Reset if too much silence
            if (self.last_speech_time and
                current_time - self.last_speech_time > self.max_silence_gap):
                logger.debug(f"[BargeInDetector] Resetting due to silence gap")
                self._reset_detection_state()
        
        # Determine if we should trigger barge-in
        should_barge_in = False
        confidence = 0.0
        debug_info = {}
        
        if (self.speech_start_time and
            current_time - self.speech_start_time >= self.min_speech_duration and
            self.consecutive_speech_chunks >= 3):  # At least 3 consecutive speech chunks

            # Calculate overall confidence from recent chunks
            recent_confidences = [m.confidence for _, m in list(self.audio_buffer)[-5:]
                                if m.is_speech]

            if recent_confidences:
                confidence = np.mean(recent_confidences)
                speech_duration = current_time - self.speech_start_time

                logger.info(f"[BargeInDetector] 🔍 EVALUATING BARGE-IN: "
                           f"duration={speech_duration:.2f}s, "
                           f"chunks={self.consecutive_speech_chunks}, "
                           f"confidence={confidence:.2f}, "
                           f"threshold={self.confidence_threshold}")

                if confidence > self.confidence_threshold:
                    should_barge_in = True
                    self.total_detections += 1

                    logger.warning(f"[BargeInDetector] 🛑 BARGE-IN TRIGGERED! "
                                  f"Speech duration: {speech_duration:.2f}s, "
                                  f"Avg confidence: {confidence:.2f}, "
                                  f"Chunks: {self.consecutive_speech_chunks}")

                    debug_info = {
                        "speech_duration": speech_duration,
                        "consecutive_chunks": self.consecutive_speech_chunks,
                        "avg_confidence": confidence,
                        "adaptive_threshold": self.adaptive_threshold,
                        "background_noise": self.background_noise_level
                    }

                    # Reset state after detection
                    self._reset_detection_state()
                else:
                    logger.debug(f"[BargeInDetector] Confidence too low: {confidence:.2f} < {self.confidence_threshold}")
            else:
                logger.debug(f"[BargeInDetector] No recent speech confidences available")
        
        debug_info.update({
            "current_db": metrics.db_level,
            "is_speech": metrics.is_speech,
            "confidence": metrics.confidence,
            "zcr": metrics.zero_crossing_rate
        })
        
        return should_barge_in, confidence, debug_info
    
    def _reset_detection_state(self):
        """Reset detection state for new utterance."""
        self.speech_start_time = None
        self.last_speech_time = None
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        accuracy = (self.true_positives / max(1, self.total_detections)) * 100
        
        return {
            "total_detections": self.total_detections,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "accuracy_percent": round(accuracy, 2),
            "adaptive_threshold_db": round(self.adaptive_threshold, 2),
            "background_noise_db": round(self.background_noise_level, 2),
            "current_buffer_size": len(self.audio_buffer)
        }
    
    def mark_detection_result(self, was_correct: bool):
        """Mark whether the last detection was correct (for learning)."""
        if was_correct:
            self.true_positives += 1
        else:
            self.false_positives += 1
    
    def adjust_sensitivity(self, increase: bool = True):
        """Adjust detection sensitivity based on feedback."""
        adjustment = -2.0 if increase else 2.0
        self.speech_threshold_db += adjustment
        self.adaptive_threshold += adjustment
        
        logger.info(f"[BargeInDetector] Sensitivity adjusted: {self.speech_threshold_db}dB")

# Global detector instance
_barge_in_detector = None

def get_barge_in_detector() -> BargeInDetector:
    """Get the global barge-in detector instance."""
    global _barge_in_detector
    if _barge_in_detector is None:
        _barge_in_detector = BargeInDetector()
    return _barge_in_detector
