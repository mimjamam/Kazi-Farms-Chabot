"""
Dynamic Threshold Calibrator - Self-adjusting thresholds based on performance
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class CalibrationResult:
    """Result from threshold calibration"""
    threshold: float
    confidence: float
    sample_count: int
    success_rate: float
    adjustment_made: bool

class ThresholdCalibrator:
    """Dynamic threshold calibrator that adjusts based on performance"""
    
    def __init__(self, 
                 window: int = 500, 
                 init_threshold: float = 0.42, 
                 min_threshold: float = 0.30, 
                 max_threshold: float = 0.60,
                 min_samples: int = 50,
                 adjustment_rate: float = 0.1):
        self.window = window
        self.data = deque(maxlen=window)
        self.threshold = init_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_samples = min_samples
        self.adjustment_rate = adjustment_rate
        
        print(f"[THRESHOLD CALIBRATOR] Initialized with threshold={init_threshold}, window={window}")
    
    def add_sample(self, score: float, success: bool) -> CalibrationResult:
        """Add a new sample and potentially adjust threshold"""
        self.data.append((score, success))
        
        # Calculate current metrics
        total_samples = len(self.data)
        success_samples = sum(1 for _, success in self.data if success)
        success_rate = success_samples / total_samples if total_samples > 0 else 0.0
        
        # Adjust threshold if we have enough samples
        adjustment_made = False
        if total_samples >= self.min_samples:
            old_threshold = self.threshold
            self._recompute_threshold()
            adjustment_made = abs(self.threshold - old_threshold) > 0.001
        
        return CalibrationResult(
            threshold=self.threshold,
            confidence=min(1.0, total_samples / self.min_samples),
            sample_count=total_samples,
            success_rate=success_rate,
            adjustment_made=adjustment_made
        )
    
    def _recompute_threshold(self) -> None:
        """Recompute threshold based on successful samples"""
        if len(self.data) < self.min_samples:
            return
        
        # Get successful samples
        successful_scores = [score for score, success in self.data if success]
        
        if not successful_scores:
            # No successful samples, lower threshold
            self.threshold = max(self.min_threshold, self.threshold - self.adjustment_rate)
            return
        
        # Calculate percentile-based threshold
        try:
            # Use 25th percentile of successful scores as new threshold
            new_threshold = float(np.percentile(successful_scores, 25))
            
            # Apply adjustment rate for gradual changes
            adjusted_threshold = (1 - self.adjustment_rate) * self.threshold + self.adjustment_rate * new_threshold
            
            # Ensure within bounds
            self.threshold = max(self.min_threshold, min(self.max_threshold, adjusted_threshold))
            
            print(f"[THRESHOLD CALIBRATOR] Adjusted threshold to {self.threshold:.3f}")
            
        except Exception as e:
            print(f"[THRESHOLD CALIBRATOR ERROR] Failed to recompute: {str(e)}")
    
    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold
    
    def get_stats(self) -> dict:
        """Get calibration statistics"""
        if not self.data:
            return {
                "threshold": self.threshold,
                "sample_count": 0,
                "success_rate": 0.0,
                "confidence": 0.0
            }
        
        total_samples = len(self.data)
        success_samples = sum(1 for _, success in self.data if success)
        success_rate = success_samples / total_samples
        
        return {
            "threshold": self.threshold,
            "sample_count": total_samples,
            "success_rate": success_rate,
            "confidence": min(1.0, total_samples / self.min_samples),
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold
        }
    
    def reset(self, new_threshold: Optional[float] = None) -> None:
        """Reset calibrator with optional new threshold"""
        self.data.clear()
        if new_threshold is not None:
            self.threshold = new_threshold
        print(f"[THRESHOLD CALIBRATOR] Reset with threshold={self.threshold}")

class DynamicThresholdManager:
    """Manages multiple threshold calibrators for different components"""
    
    def __init__(self):
        self.calibrators = {
            "domain_guard": ThresholdCalibrator(init_threshold=0.42),
            "relevance": ThresholdCalibrator(init_threshold=0.60),
            "grounding": ThresholdCalibrator(init_threshold=0.45),
            "retrieval": ThresholdCalibrator(init_threshold=0.50)
        }
        print("[DYNAMIC THRESHOLD MANAGER] Initialized with multiple calibrators")
    
    def add_sample(self, component: str, score: float, success: bool) -> CalibrationResult:
        """Add sample to specific component calibrator"""
        if component not in self.calibrators:
            print(f"[DYNAMIC THRESHOLD MANAGER WARNING] Unknown component: {component}")
            return CalibrationResult(threshold=0.5, confidence=0.0, sample_count=0, success_rate=0.0, adjustment_made=False)
        
        return self.calibrators[component].add_sample(score, success)
    
    def get_threshold(self, component: str) -> float:
        """Get threshold for specific component"""
        if component not in self.calibrators:
            return 0.5  # Default threshold
        
        return self.calibrators[component].get_threshold()
    
    def get_all_stats(self) -> dict:
        """Get statistics for all calibrators"""
        return {component: calibrator.get_stats() for component, calibrator in self.calibrators.items()}
    
    def reset_component(self, component: str, new_threshold: Optional[float] = None) -> None:
        """Reset specific component calibrator"""
        if component in self.calibrators:
            self.calibrators[component].reset(new_threshold)
        else:
            print(f"[DYNAMIC THRESHOLD MANAGER WARNING] Unknown component: {component}")
    
    def reset_all(self) -> None:
        """Reset all calibrators"""
        for calibrator in self.calibrators.values():
            calibrator.reset()
        print("[DYNAMIC THRESHOLD MANAGER] Reset all calibrators")
