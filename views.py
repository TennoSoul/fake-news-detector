from django.shortcuts import render
from fake_news_detection.system import FakeNewsDetectionSystem
from fake_news_detection.monitoring.feed_monitor import NewsFeedMonitor
from fake_news_detection.scraping.news_scraper import NewsScraper
from fake_news_detection.config.model_config import ModelConfig
from fake_news_detection.config.media_config import MediaConfig
from fake_news_detection.media_detection.image_detection import ImageDetector
from fake_news_detection.media_detection.frame_extractor import FrameExtractor
import logging
import pandas as pd
import os
from django.core.files.storage import FileSystemStorage
import mimetypes

# Initialize components
config = ModelConfig()
media_config = MediaConfig()
system = FakeNewsDetectionSystem(config=config)
scraper = NewsScraper()
monitor = NewsFeedMonitor(scraper=scraper, detector=system.detector)
image_detector = ImageDetector(media_config)
frame_extractor = FrameExtractor(media_config)

# Log system status
logging.info(f"Using device: {system.device}")
if system.detector.bert_model:
    logging.info(f"BERT Model Status: {system.detector.bert_model.check_model_health()}")
else:
    logging.info("BERT model not enabled")

def analyze_metadata(file_path: str) -> dict:
    """Analyze metadata of media files for signs of manipulation.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        dict: Metadata analysis results
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import os
        
        results = {
            'file_size': os.path.getsize(file_path),
            'last_modified': os.path.getmtime(file_path),
            'creation_time': os.path.getctime(file_path),
            'exif_data': {}
        }
        
        # Try to extract EXIF data for images
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    for tag_id in exif:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exif.get(tag_id)
                        # Decode bytes if necessary
                        if isinstance(data, bytes):
                            try:
                                data = data.decode()
                            except:
                                data = str(data)
                        results['exif_data'][tag] = str(data)
                    
                results['dimensions'] = img.size
                results['format'] = img.format
                results['mode'] = img.mode
        except Exception as e:
            results['image_metadata_error'] = str(e)
            
        return results
        
    except Exception as e:
        return {'error': f'Error analyzing metadata: {str(e)}'}

def analyze_media(file_path: str, selected_tools: list) -> dict:
    """Analyze media file (image or video) using selected tools.
    
    Args:
        file_path: Path to the media file
        selected_tools: List of tool IDs to use for analysis
        
    Returns:
        dict: Results from each selected tool
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        results = {}
        
        if not mime_type:
            return {'error': 'Could not determine file type'}
            
        media_type = 'image' if mime_type.startswith('image/') else 'video' if mime_type.startswith('video/') else None
        
        if not media_type:
            return {'error': f'Unsupported media type: {mime_type}'}
            
        for tool in selected_tools:
            # Skip tools that don't support this media type
            if media_type not in media_config.available_tools[tool]['supported_types']:
                results[tool] = {'error': f'Tool does not support {media_type} analysis'}
                continue
                
            try:
                if tool == 'noise_analysis':
                    if media_type == 'image':
                        results[tool] = image_detector.predict(file_path)
                    else:  # video
                        results[tool] = frame_extractor.analyze_video(file_path, image_detector)
                        
                elif tool == 'deepfake':
                    if hasattr(system, 'deepfake_detector'):
                        results[tool] = system.deepfake_detector.detect(file_path)
                    else:
                        results[tool] = {'error': 'DeepFake detection not configured'}
                        
                elif tool == 'metadata':
                    results[tool] = analyze_metadata(file_path)
                    
            except Exception as e:
                logging.error(f"Error in {tool} analysis: {e}")
                results[tool] = {'error': str(e)}
                
        return results
            
    except Exception as e:
        logging.error(f"Error in media analysis: {e}")
        return {'error': str(e)}

def home(request):
    """Handle home page requests with text and media analysis."""
    generated_html = None
    news_text = ''
    feature_type = ''
    media_result = None

    if request.method == 'POST':
        news_text = request.POST.get('news_text', '')
        feature_type = request.POST.get('feature', '')
        
        try:
            results = {}
            
            # Handle text analysis if provided
            if news_text:
                article = scraper.scrape_article(news_text)
                if article:
                    text_result = system.predict(article['content'])
                    results['text_analysis'] = text_result
            
            # Handle media upload if provided
            if request.FILES.get('media'):
                media_file = request.FILES['media']
                fs = FileSystemStorage()
                filename = fs.save(media_file.name, media_file)
                file_path = fs.path(filename)
                
                media_result = analyze_media(file_path)
                results['media_analysis'] = media_result
                
                # Clean up uploaded file
                fs.delete(filename)
            
            # Generate combined report
            data = pd.DataFrame([results])
            generated_html = monitor._generate_html_report(data, "Content Analysis Results")
                
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            generated_html = f"<div class='error'>Error analyzing content: {str(e)}</div>"
    
    return render(request, 'newsdetector/home.html', {
        'result_html': generated_html,
        'news_text': news_text,
        'feature_type': feature_type,
        'media_result': media_result
    })

def analyze_media_view(request):
    """Handle media analysis with tool selection."""
    if request.method == 'POST':
        if 'media_file' not in request.FILES:
            return render(request, 'tool_selection.html', {
                'error': 'No file uploaded',
                'available_tools': media_config.available_tools
            })
            
        media_file = request.FILES['media_file']
        selected_tools = request.POST.getlist('selected_tools')
        
        if not selected_tools:
            return render(request, 'tool_selection.html', {
                'error': 'No analysis tools selected',
                'available_tools': media_config.available_tools
            })
            
        # Save the uploaded file
        fs = FileSystemStorage()
        filename = fs.save(media_file.name, media_file)
        file_path = fs.path(filename)
        
        try:
            # Run the analysis
            results = analyze_media(file_path, selected_tools)
            
            # Clean up the file
            fs.delete(filename)
            
            return render(request, 'tool_selection.html', {
                'results': results,
                'available_tools': media_config.available_tools
            })
            
        except Exception as e:
            # Clean up on error
            fs.delete(filename)
            return render(request, 'tool_selection.html', {
                'error': str(e),
                'available_tools': media_config.available_tools
            })
            
    # GET request
    return render(request, 'tool_selection.html', {
        'available_tools': media_config.available_tools
    })


