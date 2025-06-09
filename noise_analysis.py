"""Module for extracting and analyzing image noise patterns."""

import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from scipy.stats import entropy
from scipy.signal import find_peaks
import logging
from typing import Dict, Tuple, Optional
import torch

class NoiseAnalyzer:
    def __init__(self, config):
        self.config = config.image_config
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 config.gpu_config['use_gpu'] else 'cpu')
        
    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """Extract noise residual from image by subtracting Gaussian blur.
        
        Args:
            image: Input image as numpy array (grayscale)
            
        Returns:
            Noise residual array
        """
        try:
            # Convert to float32 for better precision
            image = image.astype(np.float32) / 255.0
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(
                image, 
                ksize=(0, 0),  # Compute from sigma
                sigmaX=self.config['gaussian_blur_sigma']
            )
            
            # Extract noise residual
            residual = image - blurred
            
            return residual
            
        except Exception as e:
            logging.error(f"Error extracting noise residual: {e}")
            return None
            
    def compute_fft_features(self, residual: np.ndarray) -> Dict[str, float]:
        """Compute FFT-based features from noise residual.
        
        Args:
            residual: Noise residual array
            
        Returns:
            Dictionary of computed features
        """
        try:
            # Compute 2D FFT
            fft = fftshift(fft2(residual))
            magnitude = np.abs(fft)
            
            # Compute variance of residual
            var_residual = np.var(residual)
            
            # Compute entropy of FFT magnitude
            hist, _ = np.histogram(magnitude, bins=self.config['entropy_bins'])
            hist = hist / hist.sum()
            fft_entropy = entropy(hist)
            
            # Find prominent peaks in magnitude spectrum
            peak_rows, _ = find_peaks(
                magnitude.mean(axis=1),
                prominence=self.config['peak_prominence'],
                distance=self.config['min_peak_distance']
            )
            peak_cols, _ = find_peaks(
                magnitude.mean(axis=0),
                prominence=self.config['peak_prominence'],
                distance=self.config['min_peak_distance']
            )
            num_peaks = len(peak_rows) + len(peak_cols)
            
            return {
                'variance': float(var_residual),
                'entropy': float(fft_entropy),
                'num_peaks': num_peaks,
                'peak_intensity': float(magnitude[peak_rows, peak_cols].mean()) if num_peaks > 0 else 0.0,
                'high_freq_energy': float(magnitude[magnitude.shape[0]//2:, :].sum() / magnitude.sum())
            }
            
        except Exception as e:
            logging.error(f"Error computing FFT features: {e}")
            return None
            
    def analyze_local_patterns(self, residual: np.ndarray) -> Dict[str, float]:
        """Analyze noise patterns in local image blocks.
        
        Args:
            residual: Noise residual array
            
        Returns:
            Dictionary of local pattern features
        """
        try:
            block_size = self.config['block_size']
            h, w = residual.shape
            
            # Split image into blocks
            blocks_h = h // block_size
            blocks_w = w // block_size
            
            block_vars = []
            block_entropies = []
            
            for i in range(blocks_h):
                for j in range(blocks_w):
                    block = residual[i*block_size:(i+1)*block_size, 
                                   j*block_size:(j+1)*block_size]
                    
                    # Compute block statistics
                    block_vars.append(np.var(block))
                    
                    hist, _ = np.histogram(block, bins=self.config['entropy_bins'])
                    hist = hist / hist.sum()
                    block_entropies.append(entropy(hist))
            
            return {
                'block_var_mean': float(np.mean(block_vars)),
                'block_var_std': float(np.std(block_vars)),
                'block_entropy_mean': float(np.mean(block_entropies)),
                'block_entropy_std': float(np.std(block_entropies))
            }
            
        except Exception as e:
            logging.error(f"Error analyzing local patterns: {e}")
            return None
            
    def extract_features(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract all features from an input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of all computed features or None if error
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Extract noise residual
            residual = self.extract_noise_residual(image)
            if residual is None:
                return None
                
            # Compute all features
            fft_features = self.compute_fft_features(residual)
            local_features = self.analyze_local_patterns(residual)
            
            if fft_features is None or local_features is None:
                return None
                
            # Combine all features
            features = {**fft_features, **local_features}
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
