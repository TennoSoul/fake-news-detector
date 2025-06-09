import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from fake_news_detection.utils.validators import validate_url, validate_content
from fake_news_detection.utils.data_processor import DataProcessor
import asyncio
import aiohttp
from typing import List, Dict, Any

class NewsFeedMonitor:
    def __init__(self, scraper, detector, save_dir="data"):
        self.logger = logging.getLogger(__name__)
        self.scraper = scraper
        self.detector = detector
        self.save_dir = Path(save_dir)
        self.processor = DataProcessor()
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories."""
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "reports").mkdir(exist_ok=True)
            (self.save_dir / "analysis").mkdir(exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")

    async def process_urls_async(self, urls: List[str], batch_size: int = 5) -> Dict[str, Any]:
        """Process URLs asynchronously in batches."""
        results = []
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.scraper.scrape_async(session, url) 
                    for url in batch if validate_url(url)
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        
        return self.processor.format_results(results)

    def process_url_list(self, urls, detection_method="weighted", save=True):
        """Process a list of news article URLs with validation."""
        valid_urls = [url for url in urls if validate_url(url)]
        if not valid_urls:
            self.logger.error("No valid URLs provided")
            return None
            
        try:
            articles = self.scraper.scrape_multiple(valid_urls)
            if not articles:
                self.logger.warning("No articles were successfully scraped")
                return None

            results = []
            for article in articles:
                prediction = self.detector.predict(article['content'])
                article['analysis'] = prediction
                results.append(article)

            df = pd.DataFrame(results)
            
            if save:
                output_path = self.save_dir / "analysis" / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(output_path, index=False)
                self.logger.info(f"Analysis saved to {output_path}")

            return df

        except Exception as e:
            self.logger.error(f"Error processing URLs: {e}")
            return None

    def generate_report(self, data, title="Fake News Analysis Report"):
        """Generate analysis report with improved formatting."""
        if data.empty:
            self.logger.warning("No data available for report")
            return None
            
        # Enhanced report generation code
        report_path = self._generate_enhanced_report(data, title)
        self.logger.info(f"Report generated: {report_path}")
        return report_path
    
    def _generate_enhanced_report(self, data, title):
        """Generate an enhanced HTML report with visualizations."""
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.save_dir / "reports" / f"report_{report_time}.html"
        
        report_content = self._generate_html_report(data, title)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return report_path

    def _generate_html_report(self, data, title):
        """Generate an enhanced HTML report with better styling."""
        # Calculate statistics
        total = len(data)
        fake_news = sum(data['analysis'].apply(
            lambda x: x.get('prediction') == 'Fake News' if isinstance(x, dict) else False
        ))
        real_news = total - fake_news
        fake_pct = (fake_news / total * 100) if total > 0 else 0
        real_pct = (real_news / total * 100) if total > 0 else 0

        html = f"""
        <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f8f9fa;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    h1, h2 {{
                        color: #2c3e50;
                        border-bottom: 2px solid #eee;
                        padding-bottom: 10px;
                    }}
                    .stats {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                        padding: 20px;
                        background: #f8f9fa;
                        border-radius: 8px;
                    }}
                    .stat-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        text-align: center;
                    }}
                    .stat-card h3 {{
                        margin: 0;
                        color: #666;
                        font-size: 1em;
                    }}
                    .stat-card .number {{
                        font-size: 2em;
                        font-weight: bold;
                        margin: 10px 0;
                    }}
                    .fake {{ color: #dc3545; }}
                    .real {{ color: #28a745; }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        background: white;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    th, td {{
                        padding: 12px 15px;
                        text-align: left;
                        border-bottom: 1px solid #e9ecef;
                    }}
                    th {{
                        background-color: #f8f9fa;
                        font-weight: 600;
                        color: #495057;
                    }}
                    tr:hover {{
                        background-color: #f8f9fa;
                    }}
                    .high-confidence {{
                        background-color: #fde8e8;
                    }}
                    .low-confidence {{
                        background-color: #fff3e6;
                    }}
                    .source-tag {{
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 4px;
                        background: #e9ecef;
                        font-size: 0.9em;
                    }}
                    .prediction {{
                        font-weight: 500;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }}
                    .prediction.fake {{
                        background: #fde8e8;
                    }}
                    .prediction.real {{
                        background: #d4edda;
                    }}
                    .confidence-bar {{
                        width: 100%;
                        background: #e9ecef;
                        border-radius: 4px;
                        height: 8px;
                        margin-top: 5px;
                    }}
                    .confidence-level {{
                        height: 100%;
                        border-radius: 4px;
                        background: linear-gradient(90deg, #28a745, #dc3545);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    
                    <div class="stats">
                        <div class="stat-card">
                            <h3>Total Articles</h3>
                            <div class="number">{total}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Fake News</h3>
                            <div class="number fake">{fake_news} ({fake_pct:.1f}%)</div>
                        </div>
                        <div class="stat-card">
                            <h3>Real News</h3>
                            <div class="number real">{real_news} ({real_pct:.1f}%)</div>
                        </div>
                    </div>

                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Source</th>
                            <th>Title</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                        </tr>
        """

        # Add rows for each article
        for _, row in data.iterrows():
            analysis = row['analysis']
            prediction = analysis.get('prediction', 'Unknown')
            confidence = analysis.get('confidence', 0) * 100
            confidence_class = 'high-confidence' if confidence > 80 else 'low-confidence'
            prediction_class = 'fake' if prediction == 'Fake News' else 'real'

            html += f"""
                <tr class="{confidence_class}">
                    <td><div class="source-tag">{row.get('source', 'Unknown')}</div></td>
                    <td>{row.get('title', 'No title')}</td>
                    <td><span class="prediction {prediction_class}">{prediction}</span></td>
                    <td>
                        {confidence:.1f}%
                        <div class="confidence-bar">
                            <div class="confidence-level" style="width: {confidence}%"></div>
                        </div>
                    </td>
                </tr>
            """

        html += """
                    </table>
                </div>
            </body>
        </html>
        """

        return html
