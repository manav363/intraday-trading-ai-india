"""
Enhanced Intraday Trading AI - Main Console V2
Fully standalone - uses only V2 enhanced modules
"""

from data_engine_v2 import fetch_intraday_data, NIFTY_50
from feature_engine_v2 import add_all_features, get_feature_list
from model_engine_v2 import train_intraday_model
from decision_engine_v2 import generate_trade_plan
from news_engine_v2 import analyze_news, explain_news_decision
from scanner_v2 import scan_market
from backtester import BacktestEngine


def main():
    print("\n" + "="*70)
    print("🚀 ENHANCED INTRADAY TRADING AI V2 (INDIA)")
    print("="*70 + "\n")

    # Ask user what they want to do
    print("Choose mode:")
    print("1. Quick Trade Analysis (single stock)")
    print("2. Market Scanner (find opportunities)")
    print("3. Backtest Strategy (performance analysis)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        run_quick_analysis()
    elif choice == "2":
        run_market_scanner()
    elif choice == "3":
        run_backtest()
    else:
        print("Invalid choice. Running quick analysis...")
        run_quick_analysis()


def run_quick_analysis():
    """Quick single stock analysis"""
    print("\n" + "="*70)
    print("📊 QUICK TRADE ANALYSIS")
    print("="*70 + "\n")
    
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").upper()
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    
    capital = float(input("Enter capital (₹): ") or "100000")
    
    print(f"\n📊 Analyzing {symbol}...\n")
    
    try:
        # Fetch data
        print("1/5 Fetching market data...")
        df = fetch_intraday_data(symbol, lookback_days=10)
        
        # Add features
        print("2/5 Engineering features (35+ indicators)...")
        df = add_all_features(df)
        features = get_feature_list()
        
        # Train model
        print("3/5 Training ensemble model (XGBoost + LightGBM + RF)...")
        df_pred, model = train_intraday_model(df, features, model_type='ensemble')
        
        # Analyze news
        print("4/5 Analyzing news sentiment...")
        news = analyze_news(symbol)
        
        # Generate plan
        print("5/5 Generating trade plan...\n")
        plan = generate_trade_plan(df_pred, capital, news)
        
        # Display results
        print("\n" + "="*70)
        print("📈 TRADE ANALYSIS RESULTS")
        print("="*70)
        print(f"Stock: {symbol}")
        print("-"*70)
        
        for key, value in plan.items():
            print(f"{key:.<30} {value}")
        
        print("\n" + "="*70)
        print("📰 NEWS ANALYSIS")
        print("="*70)
        print(explain_news_decision(news))
        
        if news.get("headlines"):
            print("\nRecent Headlines:")
            for h in news["headlines"][:3]:
                print(f"  • {h}")
        
        # Show model performance
        print("\n" + "="*70)
        print("🤖 MODEL PERFORMANCE")
        print("="*70)
        print(model.get_performance_report())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check if the symbol is correct and try again.")


def run_market_scanner():
    """Scan multiple stocks for opportunities"""
    print("\n" + "="*70)
    print("📊 MARKET SCANNER")
    print("="*70 + "\n")
    
    capital = float(input("Enter capital (₹) [default: 100000]: ") or "100000")
    max_stocks = int(input("How many stocks to scan? [default: 10]: ") or "10")
    
    print(f"\n📊 Scanning top {max_stocks} NIFTY stocks...\n")
    
    opportunities = scan_market(capital=capital, max_stocks=max_stocks)
    
    # Display opportunities
    print("\n" + "="*70)
    print(f"📊 FOUND {len(opportunities)} OPPORTUNITIES")
    print("="*70 + "\n")
    
    if opportunities:
        print(f"{'Rank':<6} {'Symbol':<15} {'Action':<8} {'Confidence':<12} {'R:R':<8}")
        print("-"*70)
        
        for i, opp in enumerate(opportunities[:10], 1):
            print(f"{i:<6} {opp['Symbol']:<15} {opp['Action']:<8} "
                  f"{opp['Confidence']:<12.2f} {opp.get('Risk_Reward', 0):<8.2f}")
        
        # Ask if user wants details on specific stock
        print()
        detail = input("Enter rank number for details (or press Enter to skip): ").strip()
        
        if detail.isdigit() and 1 <= int(detail) <= len(opportunities):
            idx = int(detail) - 1
            selected = opportunities[idx]
            
            print("\n" + "="*70)
            print(f"DETAILED ANALYSIS - {selected['Symbol']}")
            print("="*70)
            
            for key, value in selected.items():
                print(f"{key:.<30} {value}")
    else:
        print("No high-quality opportunities found at this time.")
        print("Try again later or scan more stocks.")


def run_backtest():
    """Run comprehensive backtest"""
    print("\n" + "="*70)
    print("📊 BACKTEST STRATEGY")
    print("="*70 + "\n")
    
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").upper()
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    
    lookback = int(input("Enter lookback days [default: 30]: ") or "30")
    capital = float(input("Enter initial capital (₹) [default: 100000]: ") or "100000")
    
    print(f"\n📊 Running backtest on {symbol} for {lookback} days...\n")
    
    try:
        # Fetch data
        print("1/4 Fetching historical data...")
        df = fetch_intraday_data(symbol, lookback_days=lookback)
        
        # Add features
        print("2/4 Engineering features...")
        df = add_all_features(df)
        features = get_feature_list()
        
        # Train model
        print("3/4 Training model...")
        df_pred, model = train_intraday_model(df, features, model_type='ensemble')
        
        # Backtest
        print("4/4 Running backtest...\n")
        backtest = BacktestEngine(initial_capital=capital)
        trades_df = backtest.backtest_signals(df_pred)
        
        # Display results
        print(backtest.get_performance_report())
        
        # Ask to save plot
        save = input("\nSave equity curve plot? (y/n): ").lower()
        if save == 'y':
            filename = f"backtest_{symbol.replace('.NS', '')}_{lookback}d.png"
            backtest.plot_results(save_path=filename)
            print(f"✅ Plot saved as {filename}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your inputs and try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")