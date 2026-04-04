"""
Enhanced News Engine V2 - Standalone
Multi-source news analysis with improved NLP
"""

import os
import requests
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


class NewsAnalyzer:
    """Advanced news analysis for trading decisions"""
    
    def __init__(self):
        self.api_key = NEWS_API_KEY
        
        # Event risk keywords (expanded)
        self.high_risk_keywords = [
            'earnings', 'result', 'results', 'quarterly',
            'rbi', 'sebi', 'policy', 'rate', 'interest',
            'merger', 'acquisition', 'takeover',
            'probe', 'investigation', 'scam', 'fraud',
            'lawsuit', 'penalty', 'fine',
            'ban', 'suspension', 'halt',
            'bankruptcy', 'default', 'debt'
        ]
        
        self.moderate_risk_keywords = [
            'guidance', 'forecast', 'outlook',
            'dividend', 'split', 'buyback',
            'expansion', 'launch', 'project',
            'competition', 'rival'
        ]
    
    def fetch_news(self, symbol, days_back=1, max_articles=20):
        """Fetch news from NewsAPI"""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        company_name = symbol.replace('.NS', '').replace('.BO', '')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': company_name,
            'from': from_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'pageSize': max_articles
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def detect_risk_level(self, headlines):
        """Detect event risk level from headlines"""
        if not headlines:
            return 'NORMAL'
        
        combined = ' '.join(headlines).lower()
        high_count = sum(1 for kw in self.high_risk_keywords if kw in combined)
        moderate_count = sum(1 for kw in self.moderate_risk_keywords if kw in combined)
        
        if high_count >= 2:
            return 'HIGH'
        if moderate_count >= 2 or high_count == 1:
            return 'MODERATE'
        return 'NORMAL'
    
    def analyze_stock_news(self, symbol):
        """Comprehensive news analysis for a stock"""
        articles = self.fetch_news(symbol, days_back=2)
        
        if not articles:
            return {
                'symbol': symbol,
                'sentiment': 0.0,
                'risk_level': 'NORMAL',
                'confidence': 0.0,
                'article_count': 0,
                'headlines': [],
                'summary': 'No recent news available'
            }
        
        headlines = [a.get('title', '') for a in articles[:10] if a.get('title')]
        all_text = ' '.join(headlines)
        
        sentiment = self.analyze_sentiment(all_text)['polarity']
        risk_level = self.detect_risk_level(headlines)
        
        return {
            'symbol': symbol,
            'sentiment': round(sentiment, 3),
            'risk_level': risk_level,
            'confidence': min(len(articles) / 10, 1.0),
            'article_count': len(articles),
            'headlines': headlines[:5],
            'summary': f"Sentiment: {'Positive' if sentiment > 0 else 'Negative'}, Risk: {risk_level}"
        }


# Global analyzer instance
_analyzer = NewsAnalyzer()


def analyze_news(symbol):
    """
    Convenience function for news analysis
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    
    Returns:
    --------
    dict : News analysis result
    """
    return _analyzer.analyze_stock_news(symbol)


def explain_news_decision(news_result):
    """
    Generate explanation of news-based decision
    
    Parameters:
    -----------
    news_result : dict
        News analysis result
    
    Returns:
    --------
    str : Explanation text
    """
    risk_level = news_result.get('risk_level', 'NORMAL')
    
    if risk_level == 'HIGH':
        return "⚠️  HIGH RISK: Major events detected. Avoid trading until conditions stabilize."
    elif risk_level == 'MODERATE':
        return "⚡ MODERATE RISK: Proceed with caution. Use tighter stops."
    else:
        return "✅ NORMAL: No major event risk detected.\nMarket conditions are considered normal for intraday trading."


if __name__ == "__main__":
    # Test news engine
    result = analyze_news("RELIANCE.NS")
    print("News Analysis Result:")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Headlines: {len(result['headlines'])}")
    print(f"\n{explain_news_decision(result)}")