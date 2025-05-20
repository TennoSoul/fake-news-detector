import requests
from bs4 import BeautifulSoup
import pandas as pd
import tweepy
import re
import time
import os
import pickle
from datetime import datetime
from urllib.parse import urlparse
import twest  # Import the fake news detection module you've shared

class NewsScraper:
    """Scrapes content from various news sources and social media platforms."""
    
    def __init__(self, twitter_credentials=None):
        """
        Initialize the scraper with optional Twitter API credentials.
        
        Args:
            twitter_credentials (dict): Twitter API credentials with keys 
                                       'consumer_key', 'consumer_secret', 
                                       'access_token', 'access_token_secret'
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Initialize Twitter API if credentials provided
        self.twitter_api = None
        if twitter_credentials:
            auth = tweepy.OAuthHandler(
                twitter_credentials['consumer_key'], 
                twitter_credentials['consumer_secret']
            )
            auth.set_access_token(
                twitter_credentials['access_token'], 
                twitter_credentials['access_token_secret']
            )
            self.twitter_api = tweepy.API(auth)
    
    def scrape_news_article(self, url):
        """
        Scrape content from a news article URL.
        
        Args:
            url (str): URL of the news article
            
        Returns:
            dict: Article data including title, content, source, etc.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article title
            title = soup.find('h1')
            title_text = title.get_text().strip() if title else "No title found"
            
            # Extract article content - different sites have different structures
            # This is a basic approach that needs refinement for specific sites
            article_content = ""
            
            # Try to find content in article tags
            article_tag = soup.find('article')
            if article_tag:
                paragraphs = article_tag.find_all('p')
                article_content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # If no article tag or content is empty, try with main tag
            if not article_content:
                main_tag = soup.find('main')
                if main_tag:
                    paragraphs = main_tag.find_all('p')
                    article_content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # If still no content, try with common content div classes
            if not article_content:
                for div_class in ['content', 'article-content', 'story-content', 'entry-content']:
                    content_div = soup.find('div', class_=re.compile(div_class, re.I))
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        article_content = ' '.join([p.get_text().strip() for p in paragraphs])
                        break
            
            # Extract publication date if available
            pub_date = None
            date_meta = soup.find('meta', property='article:published_time')
            if date_meta:
                pub_date = date_meta.get('content')
            else:
                time_tag = soup.find('time')
                if time_tag and time_tag.has_attr('datetime'):
                    pub_date = time_tag['datetime']
            
            # Get domain as source
            domain = urlparse(url).netloc
            
            return {
                'title': title_text,
                'content': article_content,
                'url': url,
                'source': domain,
                'date_scraped': datetime.now().isoformat(),
                'publication_date': pub_date,
                'type': 'news_article'
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_multiple_articles(self, urls):
        """
        Scrape multiple news article URLs.
        
        Args:
            urls (list): List of article URLs to scrape
            
        Returns:
            list: List of article data dictionaries
        """
        articles = []
        for url in urls:
            article = self.scrape_news_article(url)
            if article:
                articles.append(article)
            time.sleep(1)  # Be nice to the servers
        
        return articles
    
    def search_twitter(self, query, count=100):
        """
        Search for tweets matching a query.
        
        Args:
            query (str): Search query
            count (int): Maximum number of tweets to return
            
        Returns:
            list: List of tweet data dictionaries
        """
        if not self.twitter_api:
            raise ValueError("Twitter API credentials not provided")
        
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query, tweet_mode='extended').items(count):
                tweet_data = {
                    'id': tweet.id_str,
                    'content': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'created_at': tweet.created_at.isoformat(),
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                    'date_scraped': datetime.now().isoformat(),
                    'type': 'tweet'
                }
                tweets.append(tweet_data)
        except Exception as e:
            print(f"Error searching Twitter: {e}")
        
        return tweets
    
    def get_user_timeline(self, username, count=100):
        """
        Get tweets from a user's timeline.
        
        Args:
            username (str): Twitter username
            count (int): Maximum number of tweets to return
            
        Returns:
            list: List of tweet data dictionaries
        """
        if not self.twitter_api:
            raise ValueError("Twitter API credentials not provided")
        
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.twitter_api.user_timeline, screen_name=username, tweet_mode='extended').items(count):
                # Skip retweets
                if hasattr(tweet, 'retweeted_status'):
                    continue
                    
                tweet_data = {
                    'id': tweet.id_str,
                    'content': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'created_at': tweet.created_at.isoformat(),
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                    'date_scraped': datetime.now().isoformat(),
                    'type': 'tweet'
                }
                tweets.append(tweet_data)
        except Exception as e:
            print(f"Error getting timeline for {username}: {e}")
        
        return tweets
    
    def save_data_to_csv(self, data, filename):
        """
        Save scraped data to a CSV file.
        
        Args:
            data (list): List of dictionaries containing scraped data
            filename (str): Output filename
        """
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

class FakeNewsDetectionSystem:
    """Detects fake news in scraped content using pre-trained models."""
    
    def __init__(self, detector=None):
        """
        Initialize with a detector model.
        
        Args:
            detector: Pre-trained detector model (optional)
        """
        # Use provided detector or create a new one
        if detector:
            self.detector = detector
        else:
            # Load the detector from the twest module
            try:
                self.detector = twest.FakeNewsDetectionSystem()
                print("Using the pre-trained fake news detection system")
            except Exception as e:
                print(f"Error loading pre-trained detection system: {e}")
                print("Initializing a new detection system")
                self.detector = twest.ComprehensiveFakeNewsDetector(use_saved_model=True)
    
    def analyze_content(self, content, method="weighted"):
        """Analyze content to determine if it might be fake news."""
        try:
            prediction = self.detector.predict(content, method=method)
            
            # Adjust prediction based on confidence
            if isinstance(prediction, dict):
                confidence = prediction.get('confidence', 0)
                # Mark as fake news if confidence is below threshold
                if confidence < 0.8:
                    prediction['prediction'] = 'Fake News'
                    prediction['confidence'] = confidence  # Keep original confidence
            
            return prediction
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"error": str(e)}
    
    def analyze_multiple_items(self, items, content_key="content", method="weighted"):
        """
        Analyze multiple content items.
        
        Args:
            items (list): List of dictionaries containing content
            content_key (str): Dictionary key that contains the text content
            method (str): Detection method
            
        Returns:
            list: Original items with added analysis results
        """
        results = []
        for item in items:
            if content_key in item and item[content_key]:
                analysis = self.analyze_content(item[content_key], method)
                item['analysis'] = analysis
                results.append(item)
            else:
                item['analysis'] = {"error": f"No {content_key} found or empty content"}
                results.append(item)
        
        return results
    
    def analyze_and_save(self, items, content_key="content", method="weighted", output_file=None):
        """
        Analyze content items and save results to a CSV file.
        
        Args:
            items (list): List of dictionaries containing content
            content_key (str): Dictionary key that contains the text content
            method (str): Detection method
            output_file (str): Output filename (optional)
            
        Returns:
            DataFrame: DataFrame with analysis results
        """
        analyzed_items = self.analyze_multiple_items(items, content_key, method)
        
        # Create a DataFrame
        df = pd.DataFrame(analyzed_items)
        
        # Extract prediction and confidence
        if 'analysis' in df.columns:
            df['is_fake_news'] = df['analysis'].apply(
                lambda x: x.get('prediction', 'Unknown') == 'Fake News' if isinstance(x, dict) else 'Error'
            )
            df['confidence'] = df['analysis'].apply(
                lambda x: x.get('confidence', 0) if isinstance(x, dict) else 0
            )
        
        # Save to file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Analysis results saved to {output_file}")
        
        return df

class NewsFeedMonitor:
    """Monitors news feeds and social media for new content to analyze."""
    
    def __init__(self, scraper, detector, feeds=None, save_dir="data"):
        """
        Initialize the news feed monitor.
        
        Args:
            scraper (NewsScraper): Scraper object
            detector (FakeNewsDetectionSystem): Detector object
            feeds (list): List of RSS feeds to monitor
            save_dir (str): Directory to save data
        """
        self.scraper = scraper
        self.detector = detector
        self.feeds = feeds or []
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def add_feed(self, feed_url):
        """Add an RSS feed to monitor."""
        if feed_url not in self.feeds:
            self.feeds.append(feed_url)
    
    def process_url_list(self, urls, detection_method="weighted", save=True):
        """Process a list of news article URLs."""
        try:
            # Scrape articles
            articles = self.scraper.scrape_multiple_articles(urls)
            
            if not articles:
                print("No articles were successfully scraped")
                return None
            
            # Analyze articles
            df = self.detector.analyze_and_save(
                articles, 
                content_key="content", 
                method=detection_method,
                output_file=os.path.join(self.save_dir, 
                    f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv") if save else None
            )
            
            print(f"Processed {len(df)} articles")
            return df
            
        except Exception as e:
            print(f"Error processing URLs: {e}")
            return None
    
    def process_twitter_search(self, query, count=100, detection_method="weighted", save=True):
        """
        Search for tweets and analyze them.
        
        Args:
            query (str): Twitter search query
            count (int): Maximum number of tweets
            detection_method (str): Fake news detection method
            save (bool): Whether to save results
            
        Returns:
            DataFrame: Analysis results
        """
        # Search tweets
        tweets = self.scraper.search_twitter(query, count)
        
        if not tweets:
            print("No tweets were found or an error occurred")
            return None
        
        # Analyze tweets
        df = self.detector.analyze_and_save(
            tweets, 
            content_key="content", 
            method=detection_method,
            output_file=os.path.join(self.save_dir, f"twitter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv") if save else None
        )
        
        return df
    
    def generate_report(self, data, title="Fake News Analysis Report", confidence_threshold=0.8):
        """Generate a report from analysis results."""
        if not isinstance(data, pd.DataFrame):
            return "No data available for report"
        
        # Process the data
        data = self._process_analysis_data(data, confidence_threshold)
        
        # Generate HTML report
        html = self._generate_html_report(data, title, confidence_threshold)
        
        # Save the report
        report_path = self._save_report(html)
        
        return html

    def _process_analysis_data(self, data, confidence_threshold):
        """Process and prepare analysis data."""
        # Handle is_fake_news column
        if 'is_fake_news' not in data.columns:
            if 'analysis' in data.columns:
                data['is_fake_news'] = data['analysis'].apply(
                    lambda x: x.get('prediction', 'Unknown') == 'Fake News' if isinstance(x, dict) else False
                )
                data['confidence'] = data['analysis'].apply(
                    lambda x: x.get('confidence', 0) if isinstance(x, dict) else 0
                )
            else:
                data['is_fake_news'] = False
                data['confidence'] = 0
        
        # Mark as fake news if confidence is below threshold
        data['adjusted_is_fake'] = data['confidence'] < confidence_threshold
        
        return data
    
    def _generate_html_report(self, data, title, confidence_threshold):
        """Generate HTML report from processed data."""
        # Count fake vs. real using adjusted values
        fake_count = sum(data['adjusted_is_fake'] == True)
        real_count = sum(data['adjusted_is_fake'] == False)
        error_count = len(data) - fake_count - real_count
        
        # Calculate percentage
        total = len(data)
        fake_pct = (fake_count / total) * 100 if total > 0 else 0
        real_pct = (real_count / total) * 100 if total > 0 else 0
        
        # Get low confidence and already classified fake news items
        low_confidence_items = data[data['adjusted_is_fake'] == True]
        
        # Generate HTML report
        html = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .fake {{ color: #d9534f; }}
                .real {{ color: #5cb85c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .high-confidence {{ background-color: #ffdddd; }}
                .low-confidence {{ background-color: #fff0f0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total items analyzed: {total}</p>
                <p><span class="fake">Fake news (includes confidence < {confidence_threshold*100}%): {fake_count} ({fake_pct:.1f}%)</span></p>
                <p><span class="real">Real news (confidence >= {confidence_threshold*100}%): {real_count} ({real_pct:.1f}%)</span></p>
                <p>Analysis errors: {error_count}</p>
                <p><strong>Note:</strong> Items with confidence level below {confidence_threshold*100}% are classified as fake news.</p>
            </div>
            
            <h2>Fake News Items ({len(low_confidence_items)} items)</h2>
            <table>
                <tr>
                    <th>Source</th>
                    <th>Title/Content</th>
                    <th>Reason</th>
                    <th>Confidence</th>
                </tr>
        """
        
        # Add fake news items
        for _, row in low_confidence_items.iterrows():
            # Safe access of columns with default values
            title = row['title'] if 'title' in row.index and pd.notna(row['title']) else 'N/A'
            content = row['content'] if 'content' in row.index and pd.notna(row['content']) else 'N/A'
            
            # Create display text
            if title != 'N/A':
                display_text = title
            elif content != 'N/A':
                display_text = content[:100] + ('...' if len(content) > 100 else '')
            else:
                display_text = 'No content available'
            
            # Get source
            if 'source' in row.index and pd.notna(row['source']):
                source = row['source']
            elif 'user' in row.index and pd.notna(row['user']):
                source = row['user']
            else:
                source = 'Unknown'
            
            # Get confidence
            confidence = row['confidence'] * 100 if 'confidence' in row.index and pd.notna(row['confidence']) else 0
            
            # Determine reason
            if row['is_fake_news'] == True:
                reason = "Detected as fake news"
                row_class = "high-confidence"
            else:
                reason = f"Low confidence (< {confidence_threshold*100}%)"
                row_class = "low-confidence"
            
            html += f"""
                <tr class="{row_class}">
                    <td>{source}</td>
                    <td>{display_text}</td>
                    <td>{reason}</td>
                    <td>{confidence:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>All Analyzed Items</h2>
            <table>
                <tr>
                    <th>Source</th>
                    <th>Title/Content</th>
                    <th>Original Prediction</th>
                    <th>Final Classification</th>
                    <th>Confidence</th>
                </tr>
        """
        
        # Add all items
        for _, row in data.iterrows():
            # Safe access of columns with default values
            title = row['title'] if 'title' in row.index and pd.notna(row['title']) else 'N/A'
            content = row['content'] if 'content' in row.index and pd.notna(row['content']) else 'N/A'
            
            # Create display text
            if title != 'N/A':
                display_text = title
            elif content != 'N/A':
                display_text = content[:100] + ('...' if len(content) > 100 else '')
            else:
                display_text = 'No content available'
            
            # Get source
            if 'source' in row.index and pd.notna(row['source']):
                source = row['source']
            elif 'user' in row.index and pd.notna(row['user']):
                source = row['user']
            else:
                source = 'Unknown'
            
            # Determine original prediction
            is_fake = bool(row['is_fake_news']) if 'is_fake_news' in row.index and pd.notna(row['is_fake_news']) else False
            original_prediction = "Fake News" if is_fake else "Real News"
            
            # Determine final classification
            is_adjusted_fake = bool(row['adjusted_is_fake']) if 'adjusted_is_fake' in row.index else False
            final_classification = "Fake News" if is_adjusted_fake else "Real News"
            
            # Get confidence
            confidence = row['confidence'] * 100 if 'confidence' in row.index and pd.notna(row['confidence']) else 0
            
            # Set row class for highlighting
            if is_fake:
                row_class = "high-confidence"
            elif is_adjusted_fake:
                row_class = "low-confidence"
            else:
                row_class = ""
            
            html += f"""
                <tr class="{row_class}">
                    <td>{source}</td>
                    <td>{display_text}</td>
                    <td>{original_prediction}</td>
                    <td>{final_classification}</td>
                    <td>{confidence:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _save_report(self, html):
        """Save the HTML report to a file."""
        report_path = os.path.join(self.save_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Report saved to {report_path}")
        
        return report_path

# Example usage
def example_usage():
    try:
        # Initialize components
        scraper = NewsScraper()
        detector = FakeNewsDetectionSystem()
        monitor = NewsFeedMonitor(scraper, detector)
        
        # Test URLs
        news_urls = [
            'https://edition.cnn.com/2025/03/20/europe/london-heathrow-airport-shut-intl-hnk/index.html',
            'https://www.twincities.com/2016/01/04/wisconsin-notified-of-more-than-10000-layoffs-in-2015/',
            'https://edition.cnn.com/2025/04/27/politics/wisconsin-judge-arrest-trump-administration/index.html',
            'https://www.foxnews.com/media/adam-schiff-reveals-blunt-comment-san-francisco-cashier-told-him-about-dems-warns-party-has-major-problem',
            'https://babylonbee.com/news/a-day-that-will-live-in-infamy-donald-trump-declares-war-on-mexico-after-attack-on-brooklyn-bridge'
        ]
        
        print(f"\nProcessing {len(news_urls)} URLs:")
        articles = []
        
        # Process each URL individually with progress tracking
        for idx, url in enumerate(news_urls, 1):
            print(f"\nProcessing URL {idx}/{len(news_urls)}: {url}")
            try:
                article = scraper.scrape_news_article(url)
                if article:
                    # Analyze single article
                    analysis = detector.analyze_content(article['content'])
                    article['analysis'] = analysis
                    articles.append(article)
                    print(f"Successfully processed URL {idx} with confidence: {analysis.get('confidence', 0):.2%}")
                else:
                    print(f"Failed to scrape URL {idx}")
            except Exception as e:
                print(f"Error processing URL {idx}: {str(e)}")
        
        if articles:
            # Create DataFrame from collected articles
            results = pd.DataFrame(articles)
            # Generate report
            monitor.generate_report(results)
            print(f"\nAnalysis complete. Successfully processed {len(articles)} out of {len(news_urls)} URLs.")
            print("Check the report in the data directory.")
        else:
            print("\nNo articles were successfully processed.")
        
    except Exception as e:
        print(f"Error in example usage: {e}")

if __name__ == "__main__":
    example_usage()