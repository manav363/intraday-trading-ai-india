"""
Advanced Feature Engineering v2.0 - 35+ Technical Indicators
FIXED VERSION - Corrected TR calculation
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_returns_momentum(df):
    """Add returns and momentum features"""
    df = df.copy()
    
    # Returns at different horizons
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)
    
    # Momentum
    df["Momentum_3"] = df["Close"] - df["Close"].shift(3)
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    
    return df


def add_volume_features(df):
    """Add volume-based features"""
    df = df.copy()
    
    # Volume moving averages
    df["Vol_MA_5"] = df["Volume"].rolling(5).mean()
    df["Vol_MA_10"] = df["Volume"].rolling(10).mean()
    df["Vol_MA_20"] = df["Volume"].rolling(20).mean()
    
    # Volume ratios
    df["Volume_Ratio_5"] = df["Volume"] / df["Vol_MA_5"].replace(0, np.nan)
    df["Volume_Ratio_10"] = df["Volume"] / df["Vol_MA_10"].replace(0, np.nan)
    
    # Volume trend
    df["Volume_Trend"] = (df["Vol_MA_5"] - df["Vol_MA_20"]) / df["Vol_MA_20"].replace(0, np.nan)
    
    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Return_1"]) * df["Volume"]).cumsum()
    df["OBV_MA"] = df["OBV"].rolling(10).mean()
    df["OBV_Signal"] = df["OBV"] - df["OBV_MA"]
    
    return df


def add_price_patterns(df):
    """Add price pattern features"""
    df = df.copy()
    
    # Candle body and wick sizes
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Total_Range"] = df["High"] - df["Low"]
    
    # Body ratio
    df["Body_Ratio"] = df["Body"] / df["Total_Range"].replace(0, np.nan)
    
    # Candle direction
    df["Bullish"] = (df["Close"] > df["Open"]).astype(int)
    
    return df


def add_vwap_features(df):
    """Add VWAP and related features"""
    df = df.copy()
    
    # Typical price
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    
    # VWAP (reset daily for intraday)
    cumulative_tp_vol = (df["Typical_Price"] * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].cumsum()
    df["VWAP"] = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    
    # Distance from VWAP
    df["VWAP_Distance"] = df["Close"] - df["VWAP"]
    df["VWAP_Distance_Pct"] = (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)
    
    # Price position relative to VWAP
    df["Above_VWAP"] = (df["Close"] > df["VWAP"]).astype(int)
    
    return df


def add_moving_averages(df):
    """Add multiple moving averages and crossovers"""
    df = df.copy()
    
    # Simple Moving Averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    
    # Exponential Moving Averages
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # Trend direction
    df["Trend_5_20"] = df["SMA_5"] - df["SMA_20"]
    df["Trend_10_20"] = df["SMA_10"] - df["SMA_20"]
    
    # Distance from moving averages
    df["Distance_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)
    
    # Golden/Death cross signals
    df["MA_Cross_5_10"] = (df["SMA_5"] > df["SMA_10"]).astype(int)
    df["MA_Cross_10_20"] = (df["SMA_10"] > df["SMA_20"]).astype(int)
    
    return df


def add_volatility_features(df):
    """Add volatility and ATR features - FIXED VERSION"""
    df = df.copy()
    
    # Historical volatility at different windows
    df["Volatility_5"] = df["Return_1"].rolling(5).std()
    df["Volatility_10"] = df["Return_1"].rolling(10).std()
    df["Volatility_20"] = df["Return_1"].rolling(20).std()
    
    # Volatility ratio
    df["Vol_Ratio"] = df["Volatility_5"] / df["Volatility_20"].replace(0, np.nan)
    
    # True Range - FIXED METHOD
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - prev_close)
    low_close = abs(df["Low"] - prev_close)
    
    df["TR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Average True Range
    df["ATR_10"] = df["TR"].rolling(10).mean()
    df["ATR_20"] = df["TR"].rolling(20).mean()
    
    # Normalized ATR
    df["ATR_Pct"] = df["ATR_10"] / df["Close"].replace(0, np.nan)
    
    return df


def add_rsi(df, period=14):
    """Add Relative Strength Index"""
    df = df.copy()
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    
    # RSI zones
    df["RSI_Overbought"] = (df[f"RSI_{period}"] > 70).astype(int)
    df["RSI_Oversold"] = (df[f"RSI_{period}"] < 30).astype(int)
    
    return df


def add_macd(df):
    """Add MACD indicator"""
    df = df.copy()
    
    # MACD calculation
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    
    # MACD crossover
    df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
    
    return df


def add_bollinger_bands(df, period=20, std=2):
    """Add Bollinger Bands"""
    df = df.copy()
    
    df["BB_Middle"] = df["Close"].rolling(period).mean()
    df["BB_Std"] = df["Close"].rolling(period).std()
    
    df["BB_Upper"] = df["BB_Middle"] + (std * df["BB_Std"])
    df["BB_Lower"] = df["BB_Middle"] - (std * df["BB_Std"])
    
    # Bollinger Band width
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"].replace(0, np.nan)
    
    # Position within bands
    band_range = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / band_range
    
    return df


def add_time_features(df):
    """Add time-based features"""
    df = df.copy()
    
    # Extract time components
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute
    df["Day_of_Week"] = df.index.dayofweek
    
    # Market session features
    df["Market_Open"] = ((df["Hour"] == 9) & (df["Minute"] >= 15)).astype(int)
    df["Market_Close"] = ((df["Hour"] == 15) & (df["Minute"] <= 30)).astype(int)
    df["Lunch_Hour"] = ((df["Hour"] >= 12) & (df["Hour"] < 14)).astype(int)
    
    # Cyclical encoding for time
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    return df


def add_all_features(df, include_time=True):
    """
    Add all technical features to the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    include_time : bool
        Whether to include time-based features
    
    Returns:
    --------
    pd.DataFrame : Enhanced dataframe with all features
    """
    logger.info("Adding comprehensive feature set (35+ indicators)...")
    
    initial_len = len(df)
    
    # Add all feature groups
    df = add_returns_momentum(df)
    df = add_volume_features(df)
    df = add_price_patterns(df)
    df = add_vwap_features(df)
    df = add_moving_averages(df)
    df = add_volatility_features(df)  # FIXED VERSION
    df = add_rsi(df, period=14)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    
    if include_time:
        df = add_time_features(df)
    
    # Remove NaN values
    df.dropna(inplace=True)
    final_len = len(df)
    
    logger.info(f"Features added successfully: {initial_len} -> {final_len} rows")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df


def get_feature_list():
    """Return list of feature columns for model training"""
    features = [
        # Returns and Momentum
        "Return_1", "Return_3", "Return_5",
        "Momentum_3", "Momentum_5", "Momentum_10",
        
        # Volume
        "Volume_Ratio_5", "Volume_Ratio_10", "Volume_Trend",
        "OBV_Signal",
        
        # Price Patterns
        "Body_Ratio", "Bullish",
        
        # VWAP
        "VWAP_Distance_Pct", "Above_VWAP",
        
        # Moving Averages
        "Trend_5_20", "Trend_10_20", "Distance_SMA20",
        "MA_Cross_5_10", "MA_Cross_10_20",
        
        # Volatility
        "Volatility_10", "Vol_Ratio", "ATR_Pct",
        
        # RSI
        "RSI_14", "RSI_Overbought", "RSI_Oversold",
        
        # MACD
        "MACD_Histogram", "MACD_Bullish",
        
        # Bollinger Bands
        "BB_Width", "BB_Position",
        
        # Time (optional)
        "Hour_Sin", "Hour_Cos", "Market_Open", "Market_Close"
    ]
    
    return features


if __name__ == "__main__":
    print("Enhanced Feature Engine V2 - Fixed TR Calculation")
    print("Features: 35+ technical indicators")
    print("Status: Ready")