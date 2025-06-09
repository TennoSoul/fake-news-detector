"""Video frame extraction module."""

import cv2
import numpy as np
import logging
from typing import Generator, Optional, Tuple
from pathlib import Path
import torch

class FrameExtractor:
    def __init__(self, config):
        """Initialize frame extractor with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 config.gpu_config['use_gpu'] else 'cpu')
                                 
    def extract_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Extract frames from video at specified interval.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Numpy array containing frame image
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            frame_interval = int(fps * self.config.image_config['frame_interval'])
            frame_interval = max(1, frame_interval)  # Ensure at least 1
            
            frames_extracted = 0
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract frame at interval
                if frame_count % frame_interval == 0:
                    yield frame
                    frames_extracted += 1
                    
                    # Check if we've hit the maximum
                    if frames_extracted >= self.config.image_config['max_frames']:
                        break
                        
                frame_count += 1
                
            cap.release()
            
            # Warn if we didn't get minimum frames
            if frames_extracted < self.config.image_config['min_frames']:
                logging.warning(
                    f"Only extracted {frames_extracted} frames, "
                    f"less than minimum {self.config.image_config['min_frames']}"
                )
                
        except Exception as e:
            logging.error(f"Error extracting frames: {e}")
            yield None
            
    def analyze_video(self, video_path: str, detector) -> dict:
        """Analyze frames from video using provided detector.
        
        Args:
            video_path: Path to video file
            detector: ImageDetector instance
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            frame_results = []
            total_confidence = 0
            num_ai_frames = 0
            
            # Extract and analyze frames
            for i, frame in enumerate(self.extract_frames(video_path)):
                if frame is None:
                    continue
                    
                # Analyze frame
                result = detector.predict(frame)
                if result['prediction'] != 'error':
                    frame_results.append(result)
                    total_confidence += result['confidence']
                    
                    if result['prediction'] == 'AI-generated':
                        num_ai_frames += 1
                        
            if not frame_results:
                return {
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': 'No valid frames analyzed'
                }
                
            # Calculate overall metrics
            num_frames = len(frame_results)
            ai_ratio = num_ai_frames / num_frames
            avg_confidence = total_confidence / num_frames
            
            # Make final prediction
            if ai_ratio > 0.5:
                prediction = 'AI-generated'
            else:
                prediction = 'real'
                
            return {
                'prediction': prediction,
                'confidence': avg_confidence,
                'ai_frame_ratio': ai_ratio,
                'frames_analyzed': num_frames,
                'frame_results': frame_results
            }
            
        except Exception as e:
            logging.error(f"Error analyzing video: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
