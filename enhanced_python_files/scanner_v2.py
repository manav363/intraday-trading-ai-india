"""
Enhanced Market Scanner - Standalone Version
Scans multiple stocks for trading opportunities
"""

import logging
from data_engine_v2 import fetch_intraday_data, NIFTY_50
from feature_engine_v2 import add_all_features, get_feature_list
from model_engine_v2 import train_intraday_model
from decision_engine_v2 import generate_trade_plan
from news_engine_v2 import analyze_news

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_market(symbols=None, capital=100000, max_stocks=10):
    """
    Scan multiple stocks for trading opportunities
    
    Parameters:
    -----------
    symbols : list, optional
        List of symbols to scan. If None, uses top NIFTY_50 stocks
    capital : float
        Capital available for trading
    max_stocks : int
        Maximum number of stocks to scan
    
    Returns:
    --------
    list : List of trading opportunities sorted by confidence
    """
    if symbols is None:
        symbols = NIFTY_50[:max_stocks]
    
    logger.info(f"Scanning {len(symbols)} stocks...")
    
    opportunities = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{len(symbols)}] Analyzing {symbol}...")
            
            # Fetch data
            df = fetch_intraday_data(symbol, lookback_days=5)
            
            # Add features
            df = add_all_features(df)
            features = get_feature_list()
            
            # Train model
            df_pred, model = train_intraday_model(
                df, 
                features=features, 
                model_type='xgboost',
                horizon=3
            )
            
            # Analyze news
            news = analyze_news(symbol)
            
            # Generate trade plan
            plan = generate_trade_plan(df_pred, capital, news)
            
            # Add to opportunities if valid trade
            if plan['Action'] != 'NO TRADE':
                plan['Symbol'] = symbol
                plan['News_Sentiment'] = news.get('sentiment', 0)
                plan['News_Risk'] = news.get('risk_level', 'UNKNOWN')
                opportunities.append(plan)
                logger.info(f"✓ Opportunity found: {symbol}")
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue
    
    # Sort by confidence
    opportunities.sort(key=lambda x: x.get('Confidence', 0), reverse=True)
    
    logger.info(f"Found {len(opportunities)} opportunities")
    
    return opportunities


if __name__ == "__main__":
    # Test the scanner
    print("Testing Enhanced Market Scanner\n")
    
    opportunities = scan_market(max_stocks=5, capital=100000)
    
    print(f"\n{'='*70}")
    print(f"FOUND {len(opportunities)} OPPORTUNITIES")
    print('='*70)
    
    if opportunities:
        print(f"\n{'Rank':<6} {'Symbol':<15} {'Action':<8} {'Confidence':<12} {'R:R':<8}")
        print('-'*70)
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i:<6} {opp['Symbol']:<15} {opp['Action']:<8} "
                  f"{opp['Confidence']:<12.2f} {opp.get('Risk_Reward', 0):<8.2f}")
    else:
        print("\nNo opportunities found.")