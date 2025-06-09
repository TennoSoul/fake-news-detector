"""Configuration settings for media detection."""

class MediaConfig:
    def __init__(self):
        # Available detection tools
        self.available_tools = {
            'noise_analysis': {
                'name': 'Noise Pattern Analysis',
                'description': 'Detects AI-generated images using noise pattern analysis',
                'supported_types': ['image'],
                'enabled': True
            },
            'deepfake': {
                'name': 'DeepFake Detection',
                'description': 'Detects manipulated faces in images and videos',
                'supported_types': ['image', 'video'],
                'enabled': True,
                'config': {
                    'model_path': './saved_model/media/deepfake_detector.pth',
                    'face_detection_confidence': 0.5,
                    'manipulation_threshold': 0.7,
                    'batch_size': 4
                }
            },
            'metadata': {
                'name': 'Metadata Analysis',
                'description': 'Analyzes image/video metadata for signs of manipulation',
                'supported_types': ['image', 'video'],
                'enabled': True
            }
        }
        
        # Image detection settings
        self.image_config = {
            # Noise analysis settings
            'gaussian_blur_sigma': 2,       # Sigma for Gaussian blur
            'noise_threshold': 0.1,         # Threshold for noise detection
            'fft_peak_threshold': 0.75,     # Threshold for peak detection in FFT
            'min_peak_distance': 10,        # Minimum distance between peaks
            'block_size': 64,              # Block size for local analysis
            'model_path': './saved_model/media/image_classifier.pkl',
            
            # Feature extraction settings
            'entropy_bins': 50,            # Number of bins for entropy calculation
            'peak_prominence': 0.5,        # Required prominence for peak detection
            
            # Video settings
            'frame_interval': 1.0,         # Seconds between frame extractions
            'min_frames': 5,               # Minimum frames to analyze
            'max_frames': 30,              # Maximum frames to analyze
            
            # Model settings
            'classifier_type': 'random_forest',  # or 'logistic_regression'
            'random_forest_config': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            },
            'logistic_regression_config': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
        # GPU settings for media processing
        self.gpu_config = {
            'use_gpu': True,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True
        }
        
        # Output settings
        self.output_config = {
            'save_analysis': True,
            'plot_features': False,
            'confidence_threshold': 0.8,
            'save_dir': './data/media_analysis'
        }
