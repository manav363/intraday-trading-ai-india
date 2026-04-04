# 🧠 Intraday Trading AI — Indian Markets

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600?style=flat)
![License](https://img.shields.io/badge/license-MIT-22c55e?style=flat)
![Markets](https://img.shields.io/badge/Market-NSE%20%7C%20NIFTY-blue?style=flat)

> A professional-grade AI-powered intraday trading intelligence system for Indian stock markets. Combines machine learning, live news sentiment, and volatility-adjusted risk management to generate rupee-based trade decisions — with full walk-forward validation to ensure honest, out-of-sample results.

---

## 🚀 Quick Start

```bash
git clone https://github.com/manav363/intraday-trading-ai-india.git
cd intraday-trading-ai-india
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora

# Add your NewsAPI key
echo "NEWS_API_KEY=your_key_here" > .env

# Run
python3 enhanced_python_files/main_v2.py
```

---

## 🎯 What This System Does

This system acts as a **trader's decision-support assistant**, not a blind execution bot. It:

- Scans the NIFTY universe for live intraday opportunities
- Engineers **72 technical features** from raw OHLCV data
- Trains an **ensemble ML model** (XGBoost + Random Forest) with walk-forward validation
- Integrates **live news sentiment** and event-risk detection
- Filters out low-confidence and high-risk conditions automatically
- Generates clear, ₹-denominated trade plans with stop-loss and take-profit levels
- Explains every decision in plain English

---

## 🧩 System Architecture

```
Market Data (yFinance · NSE)
        ↓
Feature Engine (72 indicators)
        ↓
Walk-Forward ML Training
(XGBoost + RF Ensemble · 5-fold time-series validation)
        ↓
News Engine (NewsAPI + TextBlob sentiment)
        ↓
Decision Engine
(Confidence filter · Volatility sizing · Risk:Reward 1:2.5)
        ↓
Market Scanner → Trader Console
```

---

## 🧠 What Makes This System Different

| Feature | Why It Matters |
|---|---|
| **Walk-forward validation** | Predictions are strictly out-of-sample — no look-ahead bias |
| **72-feature engineering** | VWAP, ATR, RSI, MACD, Bollinger Bands, OBV, time encodings |
| **Ensemble model** | XGBoost + Random Forest with soft voting |
| **Live news integration** | Detects earnings, RBI events, M&A, fraud — avoids high-risk setups |
| **₹-based risk control** | Volatility-adjusted position sizing, max 1% risk per trade |
| **Confidence filtering** | System refuses to trade when model certainty is low |
| **Data caching layer** | Avoids redundant API calls with pickle-based cache |
| **Human-readable output** | No black-box decisions — every trade plan is explained |

---

## 📊 Model Validation

The model uses **walk-forward cross-validation** — the same approach used by professional quant funds to prevent overfitting.

Example (ICICIBANK.NS — 5 folds):

| Fold | OOS Accuracy |
|------|-------------|
| 1    | 0.375       |
| 2    | 0.250       |
| 3    | 0.500       |
| 4    | 0.875       |
| 5    | 0.778       |
| **Mean** | **0.556** |

> OOS Accuracy: **0.554** · OOS AUC-ROC: **0.570**
> Realistic market predictability. The system doesn't claim to beat the market — it claims to manage risk better.

---

## 🖥 Example Output

```
======================================================================
📈 TRADE ANALYSIS RESULTS
======================================================================
Stock: ICICIBANK.NS
----------------------------------------------------------------------
Action........................ BUY
Price......................... 1216.50
Confidence.................... 0.72
Quantity...................... 12
Investment.................... ₹14,598.00
Stop_Loss_Price............... ₹1,192.17
Take_Profit_Price............. ₹1,277.33
Expected_Profit............... ₹729.90
Expected_Loss................. ₹291.96
Risk_Reward................... 2.50
RSI........................... 44.2
MACD_Signal................... Bullish
Volatility.................... 1.84%

======================================================================
📰 NEWS ANALYSIS
======================================================================
✅ NORMAL: No major event risk detected.
Market conditions are considered normal for intraday trading.
```

---

## 🔍 Market Scanner

Scans multiple NIFTY stocks simultaneously and ranks by confidence:

```
Rank   Symbol          Action   Confidence   R:R
----------------------------------------------------------------------
1      RELIANCE.NS     BUY      0.79         2.50
2      HDFCBANK.NS     BUY      0.73         2.50
3      TCS.NS          BUY      0.67         2.50
```

---

## 📂 Project Structure

```
intraday_trading_ai_india/
│
├── enhanced_python_files/
│   ├── main_v2.py            # Trading console (3 modes)
│   ├── data_engine_v2.py     # NSE data fetcher + caching
│   ├── feature_engine_v2.py  # 72-feature engineering pipeline
│   ├── model_engine_v2.py    # Walk-forward ensemble training
│   ├── news_engine_v2.py     # Live news + sentiment analysis
│   ├── decision_engine_v2.py # Trade plan + position sizing
│   ├── scanner_v2.py         # Multi-stock opportunity scanner
│   └── backtester.py         # Backtesting framework
│
├── requirements.txt
├── requirements_enhanced.txt
├── .env.example
└── README.md
```

---

## ⚙️ Modes

```
Choose mode:
1. Quick Trade Analysis   →  Single stock deep analysis
2. Market Scanner         →  Scan full NIFTY universe
3. Backtest Strategy      →  Historical performance analysis
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| ML | XGBoost, Scikit-learn (Random Forest, VotingClassifier) |
| Data | yFinance, Pandas, NumPy |
| NLP | TextBlob, NewsAPI |
| Visualization | Matplotlib, Seaborn |
| Config | python-dotenv |

---

## 📌 Known Limitations

- Uses daily/intraday OHLCV only — no order book or tick data
- 5-minute bars are inherently noisy; 15m bars recommended for production use
- News sentiment from NewsAPI may not cover all NSE-relevant sources
- No live broker integration (paper trading / research only)

These are intentional research simplifications, not oversights.

---

## 🔮 Roadmap

- [ ] Switch to 15-minute bars for cleaner signals
- [ ] Add regime-aware intraday trading (integrating market_regime project)
- [ ] Zerodha Kite API integration for paper trading
- [ ] Telegram alerts for scanner opportunities
- [ ] Web dashboard (FastAPI + React)

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only.**
It does not constitute financial advice. Always use proper risk management and personal judgment when trading.

---

## 👤 Author

**Manav Garg**
Quantitative Research · AI Systems · Indian Markets
