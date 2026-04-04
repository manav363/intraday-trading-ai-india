"""
Enhanced Data Engine v2.0 - With Caching & Retry Logic
Author: Manav Garg
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expanded NIFTY 50 universe
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "LT.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "WIPRO.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
]

NIFTY_UNIVERSE = NIFTY_50

# Cache directory
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(symbol, interval, lookback_days):
    """Generate cache file path"""
    cache_name = f"{symbol}_{interval}_{lookback_days}d.pkl"
    return CACHE_DIR / cache_name


def load_from_cache(symbol, interval, lookback_days, max_age_hours=1):
    """Load data from cache if available and fresh"""
    cache_path = get_cache_path(symbol, interval, lookback_days)
    
    if not cache_path.exists():
        return None
    
    # Check cache age
    cache_age = time.time() - cache_path.stat().st_mtime
    if cache_age > max_age_hours * 3600:
        logger.info(f"Cache expired for {symbol}")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            logger.info(f"Loaded {symbol} from cache")
            return data
    except Exception as e:
        logger.warning(f"Cache load failed for {symbol}: {e}")
        return None


def save_to_cache(symbol, interval, lookback_days, df):
    """Save data to cache"""
    cache_path = get_cache_path(symbol, interval, lookback_days)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Cached data for {symbol}")
    except Exception as e:
        logger.warning(f"Cache save failed for {symbol}: {e}")


def fetch_intraday_data(symbol, interval="15m", lookback_days=120, use_cache=True):
    """
    Fetch intraday OHLCV data with caching and retry logic
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'RELIANCE.NS')
    interval : str
        Data interval ('1m', '5m', '15m', '30m', '1h')
    lookback_days : int
        Number of days to look back
    use_cache : bool
        Whether to use cached data
    
    Returns:
    --------
    pd.DataFrame : OHLCV data with datetime index
    """
    
    # Try cache first
    if use_cache:
        cached_data = load_from_cache(symbol, interval, lookback_days)
        if cached_data is not None:
            return cached_data
    
    logger.info(f"Fetching fresh data for {symbol}...")
    
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if df is not None and not df.empty:
                break
                
            logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}/{max_retries}")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise ValueError(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    
    if df is None or df.empty:
        raise ValueError(f"No intraday data available for {symbol}")
    
    # Clean columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Select required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols].copy()
    
    # Remove rows with missing data
    df.dropna(inplace=True)
    
    # Validate data quality
    if len(df) < 50:
        raise ValueError(f"Insufficient data for {symbol}: only {len(df)} rows")
    
    # Save to cache
    if use_cache:
        save_to_cache(symbol, interval, lookback_days, df)
    
    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df


def fetch_multiple_symbols(symbols, interval="5m", lookback_days=5):
    """
    Fetch data for multiple symbols with progress tracking
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols
    interval : str
        Data interval
    lookback_days : int
        Number of days to look back
    
    Returns:
    --------
    dict : {symbol: DataFrame} mapping
    """
    data_dict = {}
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
            df = fetch_intraday_data(symbol, interval, lookback_days)
            data_dict[symbol] = df
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            failed.append(symbol)
    
    if failed:
        logger.warning(f"Failed symbols: {failed}")
    
    logger.info(f"Successfully fetched {len(data_dict)}/{len(symbols)} symbols")
    return data_dict


def clear_cache():
    """Clear all cached data"""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        logger.info("Cache cleared")


if __name__ == "__main__":
    # Test the data engine
    print("Testing Enhanced Data Engine v2.0\n")
    
    # Test single symbol
    symbol = "RELIANCE.NS"
    df = fetch_intraday_data(symbol)
    print(f"\n{symbol} Data:")
    print(df.tail())
    print(f"Shape: {df.shape}")
    
    # Test caching
    print("\nTesting cache...")
    df2 = fetch_intraday_data(symbol)  # Should load from cache
    print(f"Cache test passed: {df.equals(df2)}")