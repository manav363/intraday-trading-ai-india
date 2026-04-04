"""
Enhanced Decision Engine V2 - Standalone
Advanced risk management and trade planning
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Advanced decision making for trade execution"""
    
    def __init__(self, capital=100000, max_risk_per_trade=0.01):
        """
        Initialize decision engine
        
        Parameters:
        -----------
        capital : float
            Total capital available
        max_risk_per_trade : float
            Maximum risk per trade as fraction of capital (default 1%)
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
    
    def calculate_position_size(self, price, volatility, confidence):
        """
        Calculate optimal position size using volatility-based approach
        
        Parameters:
        -----------
        price : float
            Current price
        volatility : float
            Price volatility (std of returns)
        confidence : float
            Model confidence (0-1)
        
        Returns:
        --------
        dict : Position sizing details
        """
        # Risk amount per trade
        risk_amount = self.capital * self.max_risk_per_trade
        
        # Stop loss based on volatility (2x ATR equivalent)
        stop_loss_pct = max(volatility * 2, 0.01)  # Minimum 1%
        stop_loss_pct = min(stop_loss_pct, 0.05)   # Maximum 5%
        
        # Take profit (Risk:Reward = 1:2.5)
        take_profit_pct = stop_loss_pct * 2.5
        
        # Shares calculation
        shares = int(risk_amount / (price * stop_loss_pct))
        shares = max(shares, 1)  # Minimum 1 share
        
        # Adjust for confidence (lower confidence = smaller position)
        if confidence < 0.65:
            shares = int(shares * 0.5)
        elif confidence < 0.75:
            shares = int(shares * 0.75)
        
        # Calculate expected outcomes
        investment = price * shares
        expected_profit = shares * price * take_profit_pct
        expected_loss = shares * price * stop_loss_pct
        risk_reward = expected_profit / expected_loss if expected_loss > 0 else 0
        
        return {
            'shares': shares,
            'investment': investment,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'expected_profit': expected_profit,
            'expected_loss': expected_loss,
            'risk_reward': risk_reward
        }
    
    def generate_trade_plan(self, df, news=None):
        """
        Generate comprehensive trade plan
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with predictions and features
        news : dict
            News analysis results
        
        Returns:
        --------
        dict : Complete trade plan
        """
        # Get latest data point
        latest = df.iloc[-1]
        
        # Extract key values
        prediction = int(latest.get("Prediction", 0))
        confidence = float(latest.get("Confidence", 0.5))
        price = float(latest["Close"])
        
        # Get volatility (try multiple column names)
        volatility = 0.02  # Default
        for vol_col in ["Volatility", "Volatility_10", "Volatility_5"]:
            if vol_col in latest:
                volatility = float(latest[vol_col])
                break
        
        # News-based adjustments
        news_adjustment = 0.0
        news_reason = ""
        
        if news:
            # High risk = no trade
            if news.get("risk_level") == "HIGH":
                return {
                    "Action": "NO TRADE",
                    "Reason": "High news event risk detected",
                    "Price": round(price, 2),
                    "Confidence": round(confidence, 3),
                    "News_Sentiment": news.get("sentiment", 0),
                    "News_Risk": news.get("risk_level", "UNKNOWN")
                }
            
            # Moderate risk = reduce confidence
            if news.get("risk_level") == "MODERATE":
                news_adjustment -= 0.1
                news_reason = "Moderate news risk"
            
            # Sentiment boost
            sentiment = news.get("sentiment", 0)
            if abs(sentiment) > 0.2:
                news_adjustment += sentiment * 0.15
        
        # Adjust confidence
        final_confidence = confidence + news_adjustment
        
        # Confidence threshold
        if final_confidence < 0.55:
            return {
                "Action": "NO TRADE",
                "Reason": f"Low confidence ({final_confidence:.2f}) after news adjustment",
                "Price": round(price, 2),
                "Base_Confidence": round(confidence, 3),
                "Final_Confidence": round(final_confidence, 3),
                "News_Adjustment": round(news_adjustment, 3)
            }
        
        # Determine action
        action = "BUY" if prediction == 1 else "SELL"
        
        # Calculate position sizing
        position = self.calculate_position_size(price, volatility, final_confidence)
        
        # Build trade plan
        plan = {
            "Action": action,
            "Price": round(price, 2),
            "Confidence": round(final_confidence, 3),
            "Quantity": position['shares'],
            "Investment": round(position['investment'], 2),
            "Stop_Loss_Pct": round(position['stop_loss_pct'] * 100, 2),
            "Take_Profit_Pct": round(position['take_profit_pct'] * 100, 2),
            "Stop_Loss_Price": round(price * (1 - position['stop_loss_pct']) if action == "BUY" 
                                     else price * (1 + position['stop_loss_pct']), 2),
            "Take_Profit_Price": round(price * (1 + position['take_profit_pct']) if action == "BUY"
                                       else price * (1 - position['take_profit_pct']), 2),
            "Expected_Profit": round(position['expected_profit'], 2),
            "Expected_Loss": round(position['expected_loss'], 2),
            "Risk_Reward": round(position['risk_reward'], 2),
            "Volatility": round(volatility * 100, 2),
        }
        
        # Add news info if available
        if news:
            plan["News_Sentiment"] = round(news.get("sentiment", 0), 3)
            plan["News_Risk"] = news.get("risk_level", "UNKNOWN")
            plan["News_Adjustment"] = round(news_adjustment, 3)
        
        # Add technical signals if available
        if "RSI_14" in latest:
            plan["RSI"] = round(latest["RSI_14"], 1)
        
        if "MACD_Histogram" in latest:
            plan["MACD_Signal"] = "Bullish" if latest["MACD_Histogram"] > 0 else "Bearish"
        
        return plan


def generate_trade_plan(df, capital=100000, news=None):
    """
    Convenience function to generate trade plan
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with predictions
    capital : float
        Available capital
    news : dict
        News analysis
    
    Returns:
    --------
    dict : Trade plan
    """
    engine = DecisionEngine(capital=capital)
    return engine.generate_trade_plan(df, news)


if __name__ == "__main__":
    print("Enhanced Decision Engine V2 - Ready")