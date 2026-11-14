"""
Seasonality strength of a time series from a Seasonal and Trend decomposition
using Loess (STL).
"""

import numpy as np
from statsmodels.tsa.seasonal import STL
import warnings


class SeasonalityCalculator:
    """Computes seasonality strength for batched time series.
    
    Accepts only 3D input arrays of shape (batch_size, time_steps, features).
    """

    def __init__(self, ts, period=None, detrend=False):
        """
        Args:
            ts: 3D array (batch_size, time_steps, features)
            period: Seasonal period, for example 7 for daily data with a weekly cycle
                   or 24 for hourly data with a daily cycle. None auto-detects it.
            detrend: Stored but not used. STL separates the trend itself.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Input ts must be a numpy array.")
        
        if ts.ndim != 3:
            raise ValueError(
                f"Input ts must be a 3D array with shape (batch_size, time_steps, features). "
                f"Got shape {ts.shape} with {ts.ndim} dimensions."
            )
        
        self.ts = ts
        self.period = period
        self.detrend = detrend

    def _auto_detect_period(self, ts_1d, max_period=None):
        """
        Detect the seasonal period from the peak of the FFT power spectrum.
        
        Args:
            ts_1d: 1D array (time_steps,)
            max_period: Maximum period to consider (default: len(ts)//2)
            
        Returns:
            period: The detected period, or 7 if none is found
        """
        n = len(ts_1d)
        
        # Quick checks for edge cases
        if n < 10:
            return 7  # Default for very short series
        
        if np.std(ts_1d) < 1e-8:
            return 7  # Constant series has no periodicity
        
        # Set max_period if not provided
        if max_period is None:
            max_period = n // 2
        
        min_period = 2
        
        # Normalize series (remove mean)
        ts_normalized = ts_1d - np.mean(ts_1d)
        
        # Compute FFT
        fft = np.fft.fft(ts_normalized)
        freqs = np.fft.fftfreq(n)
        power = np.abs(fft)**2
        
        # Only consider positive frequencies (negative are redundant)
        pos_mask = (freqs > 0)
        
        if not np.any(pos_mask):
            return 7  # No positive frequencies found
        
        # Find peak frequency (dominant periodicity)
        peak_freq = freqs[pos_mask][np.argmax(power[pos_mask])]
        
        if peak_freq == 0:
            return 7  # No dominant frequency
        
        # Convert frequency to period
        period = int(round(1 / peak_freq))
        
        # Validate period is within reasonable bounds
        if min_period <= period <= max_period:
            return period
        else:
            return 7  # Default if period is out of bounds

    def _compute_seasonality_strength(self, ts_1d, period):
        """
        Computes seasonality strength for a single time series using STL.
        
        Seasonality strength is defined as:
        1 - Var(Remainder) / Var(Detrended)
        
        where Detrended = Seasonal + Remainder
        
        Args:
            ts_1d: 1D array (time_steps,)
            period: Seasonal period
            
        Returns:
            seasonality_strength: float in [0, 1], where 1 means strong seasonality
        """
        # Handle edge cases
        if len(ts_1d) < 2 * period:
            return 0.0  # Series too short for meaningful seasonality
        
        if np.std(ts_1d) < 1e-8:
            return 0.0  # Constant series has no seasonality
        
        try:
            # Suppress warnings from STL
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Perform STL decomposition
                seasonal_length = max(7, period if period % 2 == 1 else period + 1)
                stl = STL(ts_1d, period=period, seasonal=seasonal_length, robust=True)

                result = stl.fit()
                
                # Extract components
                seasonal = result.seasonal
                remainder = result.resid
                
                # Compute variances
                var_remainder = np.var(remainder)
                var_detrended = np.var(seasonal + remainder)
                
                # Compute seasonality strength
                if var_detrended < 1e-8:
                    return 0.0
                
                strength = max(0.0, 1.0 - var_remainder / var_detrended)
                return min(1.0, strength)  # Clip to [0, 1]
                
        except Exception as e:
            # If STL fails, return 0
            warnings.warn(f"STL decomposition failed: {e}. Returning 0 seasonality.")
            return 0.0

    def compute_seasonality_strength(self, mode='per_feature'):
        """
        Computes seasonality strength for the entire series.
        
        Args:
            mode: 
                - 'per_feature': Compute seasonality for each feature separately.
                - 'combined': Average features and compute seasonality on combined signal.
                
        Returns:
            seasonality_strength:
                - If mode='per_feature': 2D array (batch_size, features)
                - If mode='combined': 1D array (batch_size,)
        """
        batch_size, time_steps, features = self.ts.shape
        
        if mode == 'per_feature':
            result = np.zeros((batch_size, features))
            for b in range(batch_size):
                for f in range(features):
                    ts_1d = self.ts[b, :, f]
                    period = self.period if self.period else self._auto_detect_period(ts_1d)
                    result[b, f] = self._compute_seasonality_strength(ts_1d, period)
            return result
            
        elif mode == 'combined':
            result = np.zeros(batch_size)
            for b in range(batch_size):
                combined_signal = np.mean(self.ts[b, :, :], axis=1)
                period = self.period if self.period else self._auto_detect_period(combined_signal)
                result[b] = self._compute_seasonality_strength(combined_signal, period)
            return result
            
        else:
            raise ValueError("Mode must be 'per_feature' or 'combined'.")
