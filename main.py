import logging
from fake_news_detection.config.model_config import ModelConfig
from fake_news_detection.models.bert_model import BERTModel
from fake_news_detection.scraping.news_scraper import NewsScraper
from fake_news_detection.monitoring.feed_monitor import NewsFeedMonitor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    config = ModelConfig()
    
    # Initialize components with health checks
    try:
        scraper = NewsScraper()
        detector = BERTModel(config)
        
        # Verify model health
        if not detector.check_model_health():
            logging.error("Model health check failed")
            return
            
        monitor = NewsFeedMonitor(scraper, detector)

        # Test URLs
        test_urls = [
            'https://edition.cnn.com/2025/03/20/europe/london-heathrow-airport-shut-intl-hnk/index.html',
            'https://www.twincities.com/2016/01/04/wisconsin-notified-of-more-than-10000-layoffs-in-2015/',
            'https://edition.cnn.com/2025/04/27/politics/wisconsin-judge-arrest-trump-administration/index.html',
            'https://babylonbee.com/news/report-congress-split-between-those-who-want-to-spend-a-ridiculous-amount-and-those-who-want-to-spend-an-even-more-ridiculous-amount',
            'https://www.foxnews.com/politics/judge-temporarily-pauses-trump-move-cancel-harvard-student-visa-policy-after-lawsuit'
        ]

        logging.info("Starting fake news detection system...")
        results = monitor.process_url_list(test_urls)
        
        if results is not None:
            # Check if we have any valid predictions
            valid_results = results[results['analysis'].apply(
                lambda x: x.get('prediction') not in ['Error', 'Uncertain']
            )]
            
            if not valid_results.empty:
                monitor.generate_report(valid_results)
                logging.info("Analysis complete. Check the report in the data directory.")
            else:
                logging.error("No valid predictions generated")
        else:
            logging.error("No results generated")
            
    except Exception as e:
        logging.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
