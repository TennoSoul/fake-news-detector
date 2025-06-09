import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
import logging
import re
from fake_news_detection.utils.validators import validate_url, validate_twitter_credentials
import tweepy
import aiohttp
import asyncio

class NewsScraper:
    def __init__(self, twitter_credentials=None):
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.twitter_api = self._setup_twitter(twitter_credentials)
    
    def _setup_twitter(self, credentials):
        if credentials and validate_twitter_credentials(credentials):
            try:
                auth = tweepy.OAuthHandler(
                    credentials['consumer_key'],
                    credentials['consumer_secret']
                )
                auth.set_access_token(
                    credentials['access_token'],
                    credentials['access_token_secret']
                )
                return tweepy.API(auth)
            except Exception as e:
                self.logger.error(f"Twitter API initialization failed: {e}")
        return None

    def scrape_article(self, url: str) -> dict:
        try:
            if not validate_url(url):
                self.logger.error(f"Invalid URL: {url}")
                return None
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            pub_date = self._extract_date(soup)
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'source': urlparse(url).netloc,
                'date_scraped': datetime.now().isoformat(),
                'publication_date': pub_date
            }
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return None

    async def scrape_async(self, session, url: str) -> dict:
        """Asynchronously scrape article content."""
        if not validate_url(url):
            self.logger.error(f"Invalid URL: {url}")
            return None
            
        try:
            async with session.get(url, headers=self.headers, timeout=30) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                return {
                    'title': self._extract_title(soup),
                    'content': self._extract_content(soup),
                    'url': url,
                    'source': urlparse(url).netloc,
                    'date_scraped': datetime.now().isoformat(),
                    'publication_date': self._extract_date(soup)
                }
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title = soup.find('h1')
        return title.get_text().strip() if title else "No title found"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content using multiple strategies."""
        content = ""
        
        # Strategy 1: Look for article tag
        if article := soup.find('article'):
            content = self._extract_from_element(article)
        
        # Strategy 2: Look for main content div
        if not content and (main := soup.find('main')):
            content = self._extract_from_element(main)
        
        # Strategy 3: Look for common content classes
        if not content:
            content_classes = ['article-content', 'story-content', 'post-content']
            for class_ in content_classes:
                if div := soup.find(class_=re.compile(class_, re.I)):
                    content = self._extract_from_element(div)
                    if content:
                        break
        
        return content.strip()

    def _extract_from_element(self, element: BeautifulSoup) -> str:
        """Extract text from element removing ads and navigation."""
        # Remove unwanted elements
        for unwanted in element.find_all(['nav', 'script', 'style', 'iframe']):
            unwanted.decompose()
            
        # Get all paragraphs
        paragraphs = element.find_all('p')
        
        # Filter out short/navigational paragraphs
        valid_paragraphs = [p.get_text().strip() for p in paragraphs 
                           if len(p.get_text().strip()) > 40]
        
        return ' '.join(valid_paragraphs)

    def _extract_date(self, soup: BeautifulSoup) -> str:
        # Add existing date extraction logic here
        # ...existing code from original scraper...
        pass

    def scrape_multiple(self, urls: list) -> list:
        articles = []
        for url in urls:
            if article := self.scrape_article(url):
                articles.append(article)
        return articles

    async def scrape_multiple_async(self, urls: list) -> list:
        articles = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.scrape_async(session, url) for url in urls]
            articles = await asyncio.gather(*tasks)
        return [article for article in articles if article]
