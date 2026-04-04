"""
Comprehensive Backtesting Framework v2.0
Author: Manav Garg
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for intraday trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.0003, slippage=0.0005):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in ₹
        commission : float
            Commission rate (0.03% default)
        slippage : float
            Slippage rate (0.05% default)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
        
    def calculate_position_size(self, capital, price, risk_per_trade=0.01, stop_loss_pct=0.02):
        """Calculate position size based on risk management"""
        risk_amount = capital * risk_per_trade
        shares = int(risk_amount / (price * stop_loss_pct))
        return max(shares, 1)
    
    def execute_trade(self, entry_price, exit_price, direction, shares):
        """Execute a single trade and calculate P&L"""
        # Apply slippage
        if direction == 'BUY':
            actual_entry = entry_price * (1 + self.slippage)
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_entry = entry_price * (1 - self.slippage)
            actual_exit = exit_price * (1 + self.slippage)
        
        # Calculate gross P&L
        if direction == 'BUY':
            gross_pnl = (actual_exit - actual_entry) * shares
        else:
            gross_pnl = (actual_entry - actual_exit) * shares
        
        # Calculate commission
        total_commission = (actual_entry + actual_exit) * shares * self.commission
        
        # Net P&L
        net_pnl = gross_pnl - total_commission
        
        # ROI
        investment = actual_entry * shares
        roi = (net_pnl / investment) * 100 if investment > 0 else 0
        
        return {
            'direction': direction,
            'entry_price': actual_entry,
            'exit_price': actual_exit,
            'shares': shares,
            'gross_pnl': gross_pnl,
            'commission': total_commission,
            'net_pnl': net_pnl,
            'roi': roi,
            'investment': investment
        }
    
    def backtest_signals(self, df, capital_per_trade=10000):
        """Backtest trading signals from dataframe"""
        logger.info("Running backtest...")
        
        self.trades = []
        current_capital = self.initial_capital
        position = None
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            
            if 'Prediction' not in row or pd.isna(row['Prediction']):
                continue
            
            if row.get('Confidence', 0) < 0.55:
                continue
            
            if position is None:
                # Entry
                direction = 'BUY' if row['Prediction'] == 1 else 'SELL'
                entry_price = row['Close']
                volatility = row.get('Volatility_10', 0.02)
                stop_loss_pct = max(volatility * 2, 0.01)
                
                shares = self.calculate_position_size(
                    capital_per_trade, entry_price, 0.01, stop_loss_pct
                )
                
                position = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_index': i,
                    'shares': shares,
                    'stop_loss': entry_price * (1 - stop_loss_pct) if direction == 'BUY' 
                                 else entry_price * (1 + stop_loss_pct),
                    'take_profit': entry_price * (1 + stop_loss_pct * 2.5) if direction == 'BUY'
                                   else entry_price * (1 - stop_loss_pct * 2.5)
                }
            else:
                # Check exit conditions
                current_price = row['Close']
                exit_trade = False
                
                # Stop loss / Take profit
                if position['direction'] == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        exit_trade = True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        exit_trade = True
                
                # Signal reverse
                new_signal = 'BUY' if row['Prediction'] == 1 else 'SELL'
                if new_signal != position['direction']:
                    exit_trade = True
                
                # Max hold period
                if i - position['entry_index'] >= 50:
                    exit_trade = True
                
                if exit_trade:
                    trade = self.execute_trade(
                        position['entry_price'], current_price,
                        position['direction'], position['shares']
                    )
                    self.trades.append(trade)
                    current_capital += trade['net_pnl']
                    position = None
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return pd.DataFrame(self.trades)
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = trades_df['net_pnl'].sum()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Returns
        total_return = ((self.initial_capital + total_pnl) / self.initial_capital - 1) * 100
        
        # Sharpe ratio
        returns = trades_df['roi'].values
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        # Drawdown
        equity = self.initial_capital
        equity_curve = [equity]
        for pnl in trades_df['net_pnl']:
            equity += pnl
            equity_curve.append(equity)
        
        self.equity_curve = equity_curve
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.initial_capital + total_pnl
        }
        
        self.metrics = metrics
        return metrics
    
    def get_performance_report(self):
        """Generate detailed performance report"""
        if not self.metrics:
            self.calculate_metrics()
        
        m = self.metrics
        
        report = f"""
{'='*70}
BACKTEST PERFORMANCE REPORT
{'='*70}

CAPITAL OVERVIEW:
  Initial Capital:        ₹{self.initial_capital:,.2f}
  Final Capital:          ₹{m['final_capital']:,.2f}
  Total P&L:              ₹{m['total_pnl']:,.2f}
  Total Return:           {m['total_return_pct']:.2f}%

TRADE STATISTICS:
  Total Trades:           {m['total_trades']}
  Winning Trades:         {m['winning_trades']} ({m['win_rate']:.1f}%)
  Losing Trades:          {m['losing_trades']} ({100-m['win_rate']:.1f}%)
  
PROFITABILITY:
  Average Win:            ₹{m['avg_win']:,.2f}
  Average Loss:           ₹{m['avg_loss']:,.2f}
  Profit Factor:          {m['profit_factor']:.2f}
  
RISK METRICS:
  Max Drawdown:           {m['max_drawdown_pct']:.2f}%
  Sharpe Ratio:           {m['sharpe_ratio']:.2f}

{'='*70}
"""
        return report
    
    def plot_results(self, save_path=None):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.trades:
                logger.warning("No trades to plot")
                return
            
            trades_df = pd.DataFrame(self.trades)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Equity Curve
            ax1 = axes[0, 0]
            ax1.plot(self.equity_curve, linewidth=2, color='#2E86AB')
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('Equity Curve', fontweight='bold')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Capital (₹)')
            ax1.grid(True, alpha=0.3)
            ax1.fill_between(range(len(self.equity_curve)), self.initial_capital, 
                              self.equity_curve, alpha=0.3, color='#2E86AB')
            
            # 2. P&L Distribution
            ax2 = axes[0, 1]
            ax2.hist(trades_df['net_pnl'], bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title('P&L Distribution', fontweight='bold')
            ax2.set_xlabel('Net P&L (₹)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # 3. Win/Loss Pie Chart
            ax3 = axes[1, 0]
            if self.metrics:
                win_loss = [self.metrics['winning_trades'], self.metrics['losing_trades']]
                colors = ['#06A77D', '#D62828']
                ax3.pie(win_loss, labels=['Wins', 'Losses'], autopct='%1.1f%%',
                        colors=colors, startangle=90)
                ax3.set_title(f"Win Rate: {self.metrics['win_rate']:.1f}%", fontweight='bold')
            
            # 4. Cumulative P&L
            ax4 = axes[1, 1]
            cumulative_pnl = trades_df['net_pnl'].cumsum()
            ax4.plot(cumulative_pnl.values, linewidth=2, color='#F77F00')
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_title('Cumulative P&L', fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative P&L (₹)')
            ax4.grid(True, alpha=0.3)
            ax4.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                              alpha=0.3, color='#F77F00')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
                print(f"✅ Plot saved as {save_path}")
            else:
                plt.show()
            
            plt.close()
            return fig
            
        except ImportError:
            print("⚠️  Matplotlib not installed. Install with: pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            print(f"⚠️  Could not create plot: {e}")


if __name__ == "__main__":
    print("Backtesting Framework v2.0 - Ready!")
    print("Usage: backtest = BacktestEngine(initial_capital=100000)")