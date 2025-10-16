# advanced_analysis.py - Advanced Strategy Analysis Suite
# All 5 Tests: Market Regime, Win/Loss Streaks, Correlation, Sharpe Ratio, Trade Distribution

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class AdvancedStrategyAnalysis:
    def __init__(self, trades_file='trades.json', data_file='data.csv', initial_capital=30000):
        """
        Advanced Strategy Analysis Suite
        
        Tests:
        1. Market Regime Analysis (Bull/Bear/Range)
        2. Win/Loss Streak Analysis
        3. Correlation Analysis
        4. Sharpe Ratio Calculation
        5. Trade Distribution Analysis
        """
        self.trades_file = trades_file
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.trades = []
        self.price_data = None
        
    def load_data(self):
        """Load trades and price data"""
        print(f"\n{'='*80}")
        print(f"🔬 ADVANCED STRATEGY ANALYSIS SUITE")
        print(f"{'='*80}\n")
        
        # Load trades
        with open(self.trades_file, 'r') as f:
            self.trades = json.load(f)
        print(f"✓ Loaded {len(self.trades)} trades from {self.trades_file}")
        
        # Load price data
        self.price_data = pd.read_csv(self.data_file)
        self.price_data['Datetime'] = pd.to_datetime(self.price_data['Datetime'])
        print(f"✓ Loaded {len(self.price_data):,} price bars from {self.data_file}\n")
        
    # ========================================================================
    # TEST #1: MARKET REGIME ANALYSIS
    # ========================================================================
    
    def analyze_market_regime(self):
        """Analyze strategy performance in different market regimes"""
        print(f"{'='*80}")
        print(f"📊 TEST #1: MARKET REGIME ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Analyzing strategy performance in Bull/Bear/Range markets...\n")
        
        # Calculate market regime for each trade
        df_trades = pd.DataFrame(self.trades)
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        
        # Merge with price data to get market context
        regime_results = []
        
        for _, trade in df_trades.iterrows():
            entry_time = trade['entry_date']
            
            # Get 20-day window before entry
            window_start = entry_time - pd.Timedelta(days=20)
            window_data = self.price_data[
                (self.price_data['Datetime'] >= window_start) & 
                (self.price_data['Datetime'] <= entry_time)
            ]
            
            if len(window_data) < 10:
                regime = 'Unknown'
            else:
                # Calculate trend and volatility
                returns = window_data['close'].pct_change()
                trend = returns.mean() * 100
                volatility = returns.std() * 100
                
                # Classify regime
                if trend > 0.1 and volatility < 2:
                    regime = 'Bull'
                elif trend < -0.1 and volatility < 2:
                    regime = 'Bear'
                elif volatility > 3:
                    regime = 'High Vol'
                else:
                    regime = 'Range'
            
            regime_results.append({
                'trade_number': trade['trade_number'],
                'type': trade['type'],
                'regime': regime,
                'total_pnl': trade['total_pnl']
            })
        
        df_regime = pd.DataFrame(regime_results)
        
        # Group by regime
        print(f"{'='*80}")
        print(f"PERFORMANCE BY MARKET REGIME")
        print(f"{'='*80}\n")
        
        regime_summary = df_regime.groupby('regime').agg({
            'total_pnl': ['count', 'sum', 'mean'],
            'trade_number': 'count'
        }).round(2)
        
        print(f"{'Regime':<15} {'Trades':<10} {'Total PnL':<15} {'Avg PnL':<12} {'Win Rate':<10}")
        print(f"{'-'*80}")
        
        for regime in df_regime['regime'].unique():
            regime_trades = df_regime[df_regime['regime'] == regime]
            wins = len(regime_trades[regime_trades['total_pnl'] > 0])
            win_rate = (wins / len(regime_trades)) * 100
            
            print(f"{regime:<15} {len(regime_trades):<10} "
                  f"₹{regime_trades['total_pnl'].sum():>13,.2f} "
                  f"₹{regime_trades['total_pnl'].mean():>10,.2f} "
                  f"{win_rate:>8.1f}%")
        
        print(f"\n✅ REGIME INSIGHTS:")
        best_regime = df_regime.groupby('regime')['total_pnl'].mean().idxmax()
        worst_regime = df_regime.groupby('regime')['total_pnl'].mean().idxmin()
        
        print(f"  Best Regime:     {best_regime}")
        print(f"  Worst Regime:    {worst_regime}")
        print(f"  Strategy Type:   {'Trend-following' if best_regime == 'Bull' else 'Mean reversion'}")
        
        print(f"\n{'='*80}\n")
        return df_regime
    
    # ========================================================================
    # TEST #2: WIN/LOSS STREAK ANALYSIS
    # ========================================================================
    
    def analyze_streaks(self):
        """Analyze consecutive win/loss streaks"""
        print(f"{'='*80}")
        print(f"📊 TEST #2: WIN/LOSS STREAK ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Analyzing consecutive wins and losses for psychological preparation...\n")
        
        df_trades = pd.DataFrame(self.trades)
        
        # Create win/loss flag
        df_trades['is_win'] = df_trades['pnl_before_expense'] > 0
        
        # Calculate streaks
        streaks = []
        current_streak = 1
        current_type = df_trades.iloc[0]['is_win']
        
        for i in range(1, len(df_trades)):
            if df_trades.iloc[i]['is_win'] == current_type:
                current_streak += 1
            else:
                streaks.append({
                    'type': 'Win' if current_type else 'Loss',
                    'length': current_streak
                })
                current_streak = 1
                current_type = df_trades.iloc[i]['is_win']
        
        # Add last streak
        streaks.append({
            'type': 'Win' if current_type else 'Loss',
            'length': current_streak
        })
        
        df_streaks = pd.DataFrame(streaks)
        
        # Statistics
        win_streaks = df_streaks[df_streaks['type'] == 'Win']['length']
        loss_streaks = df_streaks[df_streaks['type'] == 'Loss']['length']
        
        print(f"{'='*80}")
        print(f"STREAK ANALYSIS")
        print(f"{'='*80}\n")
        
        print(f"WIN STREAKS:")
        print(f"  Longest Win Streak:        {win_streaks.max()} trades")
        print(f"  Average Win Streak:        {win_streaks.mean():.1f} trades")
        print(f"  Median Win Streak:         {win_streaks.median():.0f} trades")
        print(f"  Total Win Streaks:         {len(win_streaks)}\n")
        
        print(f"LOSS STREAKS:")
        print(f"  Longest Loss Streak:       {loss_streaks.max()} trades 🚨")
        print(f"  Average Loss Streak:       {loss_streaks.mean():.1f} trades")
        print(f"  Median Loss Streak:        {loss_streaks.median():.0f} trades")
        print(f"  Total Loss Streaks:        {len(loss_streaks)}\n")
        
        # Probability analysis
        prob_5_losses = (len(loss_streaks[loss_streaks >= 5]) / len(loss_streaks)) * 100
        prob_10_losses = (len(loss_streaks[loss_streaks >= 10]) / len(loss_streaks)) * 100
        
        print(f"📊 PROBABILITY OF LOSING STREAKS:")
        print(f"  Probability of 5+ losses:  {prob_5_losses:.1f}%")
        print(f"  Probability of 10+ losses: {prob_10_losses:.1f}%")
        
        # Psychological preparation
        print(f"\n🧠 PSYCHOLOGICAL PREPARATION:")
        if loss_streaks.max() <= 5:
            print(f"  EXCELLENT: Max loss streak is manageable (<5 trades)")
        elif loss_streaks.max() <= 10:
            print(f"  GOOD: Max loss streak is acceptable (5-10 trades)")
        else:
            print(f"  WARNING: Long loss streaks detected (>10 trades)")
            print(f"  Be prepared for {loss_streaks.max()} consecutive losses!")
        
        print(f"\n  Recommendation: Set aside funds for {int(loss_streaks.max() * 500)} in consecutive losses")
        
        print(f"\n{'='*80}\n")
        return df_streaks
    
    # ========================================================================
    # TEST #3: CORRELATION ANALYSIS
    # ========================================================================
    
    def analyze_correlation(self):
        """Analyze correlation between strategy returns and market"""
        print(f"{'='*80}")
        print(f"📊 TEST #3: CORRELATION ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Analyzing market correlation and beta...\n")
        
        # Calculate daily returns for both strategy and market
        df_trades = pd.DataFrame(self.trades)
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades['date'] = df_trades['exit_date'].dt.date
        
        # Aggregate trades by day
        daily_pnl = df_trades.groupby('date')['total_pnl'].sum().reset_index()
        daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
        
        # Calculate market daily returns
        self.price_data['date'] = self.price_data['Datetime'].dt.date
        daily_market = self.price_data.groupby('date').agg({
            'close': 'last'
        }).reset_index()
        daily_market['date'] = pd.to_datetime(daily_market['date'])
        daily_market['market_return'] = daily_market['close'].pct_change() * 100
        
        # Merge
        merged = pd.merge(daily_pnl, daily_market, on='date', how='inner')
        merged['strategy_return'] = (merged['total_pnl'] / self.initial_capital) * 100
        
        # Calculate correlation and beta
        correlation = merged['strategy_return'].corr(merged['market_return'])
        
        # Beta calculation
        covariance = np.cov(merged['strategy_return'], merged['market_return'])[0][1]
        market_variance = np.var(merged['market_return'])
        beta = covariance / market_variance if market_variance > 0 else 0
        
        print(f"{'='*80}")
        print(f"CORRELATION & BETA ANALYSIS")
        print(f"{'='*80}\n")
        
        print(f"CORRELATION METRICS:")
        print(f"  Correlation to Market:     {correlation:.3f}")
        print(f"  Beta:                      {beta:.3f}\n")
        
        print(f"INTERPRETATION:")
        if abs(correlation) < 0.3:
            print(f"  ✅ LOW CORRELATION: Strategy is market-neutral")
            print(f"     Your returns are independent of market direction")
        elif abs(correlation) < 0.6:
            print(f"  ⚠️  MODERATE CORRELATION: Some market dependency")
            print(f"     Strategy has some directional bias")
        else:
            print(f"  🚨 HIGH CORRELATION: Strong market dependency")
            print(f"     Strategy is heavily influenced by market direction")
        
        print(f"\n  Beta Interpretation:")
        if abs(beta) < 0.3:
            print(f"  ✅ Market-neutral strategy (low beta)")
        elif abs(beta) < 0.7:
            print(f"  ⚠️  Moderate market exposure")
        else:
            print(f"  🚨 High market exposure (beta > 0.7)")
        
        print(f"\n{'='*80}\n")
        return {'correlation': correlation, 'beta': beta}
    
    # ========================================================================
    # TEST #4: SHARPE RATIO
    # ========================================================================
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.06):
        """Calculate Sharpe Ratio for risk-adjusted returns"""
        print(f"{'='*80}")
        print(f"📊 TEST #4: SHARPE RATIO (Risk-Adjusted Returns)")
        print(f"{'='*80}\n")
        print(f"Calculating risk-adjusted performance metrics...\n")
        
        df_trades = pd.DataFrame(self.trades)
        
        # Calculate returns
        total_return = df_trades['total_pnl'].sum()
        num_years = 5  # Your backtest period
        annual_return = ((self.initial_capital + total_return) / self.initial_capital) ** (1/num_years) - 1
        
        # Calculate volatility (standard deviation of returns)
        df_trades['return_pct'] = (df_trades['total_pnl'] / self.initial_capital) * 100
        daily_volatility = df_trades['return_pct'].std()
        annual_volatility = daily_volatility * np.sqrt(252)  # Annualize
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / (annual_volatility / 100)
        
        # Sortino Ratio (downside deviation only)
        downside_returns = df_trades[df_trades['total_pnl'] < 0]['return_pct']
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - risk_free_rate) / (downside_deviation / 100) if downside_deviation > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        max_dd = 31.59  # From your backtest
        calmar_ratio = annual_return / (max_dd / 100)
        
        print(f"{'='*80}")
        print(f"RISK-ADJUSTED PERFORMANCE METRICS")
        print(f"{'='*80}\n")
        
        print(f"RETURN METRICS:")
        print(f"  Total Return (5 years):    {(total_return / self.initial_capital * 100):.1f}%")
        print(f"  Annual Return (CAGR):      {annual_return * 100:.1f}%")
        print(f"  Annual Volatility:         {annual_volatility:.1f}%\n")
        
        print(f"RISK-ADJUSTED RATIOS:")
        print(f"  Sharpe Ratio:              {sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:             {sortino_ratio:.2f}")
        print(f"  Calmar Ratio:              {calmar_ratio:.2f}\n")
        
        print(f"INTERPRETATION:")
        print(f"\n  Sharpe Ratio Assessment:")
        if sharpe_ratio > 2.0:
            print(f"  🏆 EXCELLENT: Sharpe > 2.0 (exceptional risk-adjusted returns)")
        elif sharpe_ratio > 1.5:
            print(f"  ✅ VERY GOOD: Sharpe > 1.5 (great risk-adjusted returns)")
        elif sharpe_ratio > 1.0:
            print(f"  ✅ GOOD: Sharpe > 1.0 (acceptable risk-adjusted returns)")
        else:
            print(f"  ⚠️  MODERATE: Sharpe < 1.0 (below industry standard)")
        
        print(f"\n  Industry Standards:")
        print(f"    Sharpe > 1.0:  Acceptable")
        print(f"    Sharpe > 1.5:  Good")
        print(f"    Sharpe > 2.0:  Excellent")
        print(f"    Sharpe > 3.0:  Outstanding (rare)")
        
        print(f"\n{'='*80}\n")
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility
        }
    
    # ========================================================================
    # TEST #5: TRADE DISTRIBUTION ANALYSIS
    # ========================================================================
    
    def analyze_trade_distribution(self):
        """Analyze when and how you make money"""
        print(f"{'='*80}")
        print(f"📊 TEST #5: TRADE DISTRIBUTION ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Analyzing when and how you make money...\n")
        
        df_trades = pd.DataFrame(self.trades)
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        
        # Time-based analysis
        df_trades['hour'] = df_trades['entry_date'].dt.hour
        df_trades['day_of_week'] = df_trades['entry_date'].dt.dayofweek
        df_trades['month'] = df_trades['entry_date'].dt.month
        
        print(f"{'='*80}")
        print(f"TRADE DISTRIBUTION BY TIME")
        print(f"{'='*80}\n")
        
        # By strategy type
        print(f"BY STRATEGY TYPE:")
        strategy_summary = df_trades.groupby('type').agg({
            'total_pnl': ['count', 'sum', 'mean']
        }).round(2)
        
        for strategy in df_trades['type'].unique():
            strat_trades = df_trades[df_trades['type'] == strategy]
            wins = len(strat_trades[strat_trades['total_pnl'] > 0])
            win_rate = (wins / len(strat_trades)) * 100
            
            print(f"  {strategy:<12} Trades: {len(strat_trades):>4}  "
                  f"Win Rate: {win_rate:>5.1f}%  "
                  f"Total PnL: ₹{strat_trades['total_pnl'].sum():>10,.2f}  "
                  f"Avg: ₹{strat_trades['total_pnl'].mean():>7,.2f}")
        
        # By hour of day
        print(f"\nBY HOUR OF DAY (Entry Time):")
        hourly = df_trades.groupby('hour')['total_pnl'].agg(['count', 'sum', 'mean']).round(2)
        best_hour = hourly['sum'].idxmax()
        worst_hour = hourly['sum'].idxmin()
        
        print(f"  Best Hour:     {best_hour:02d}:00 (₹{hourly.loc[best_hour, 'sum']:,.2f} total PnL)")
        print(f"  Worst Hour:    {worst_hour:02d}:00 (₹{hourly.loc[worst_hour, 'sum']:,.2f} total PnL)")
        
        # By day of week
        print(f"\nBY DAY OF WEEK:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for i, day in enumerate(days):
            day_trades = df_trades[df_trades['day_of_week'] == i]
            if len(day_trades) > 0:
                wins = len(day_trades[day_trades['total_pnl'] > 0])
                win_rate = (wins / len(day_trades)) * 100
                print(f"  {day:<10} Trades: {len(day_trades):>4}  "
                      f"Win Rate: {win_rate:>5.1f}%  "
                      f"Total PnL: ₹{day_trades['total_pnl'].sum():>10,.2f}")
        
        # By month
        print(f"\nBY MONTH:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pnl = df_trades.groupby('month')['total_pnl'].sum().round(2)
        best_month = monthly_pnl.idxmax()
        worst_month = monthly_pnl.idxmin()
        
        print(f"  Best Month:    {months[best_month-1]} (₹{monthly_pnl[best_month]:,.2f})")
        print(f"  Worst Month:   {months[worst_month-1]} (₹{monthly_pnl[worst_month]:,.2f})")
        
        # PnL distribution
        print(f"\nPnL DISTRIBUTION:")
        print(f"  Wins > ₹1000:     {len(df_trades[df_trades['total_pnl'] > 1000])} trades")
        print(f"  Wins ₹500-1000:   {len(df_trades[(df_trades['total_pnl'] > 500) & (df_trades['total_pnl'] <= 1000)])} trades")
        print(f"  Small wins <₹500: {len(df_trades[(df_trades['total_pnl'] > 0) & (df_trades['total_pnl'] <= 500)])} trades")
        print(f"  Small losses:     {len(df_trades[(df_trades['total_pnl'] < 0) & (df_trades['total_pnl'] >= -500)])} trades")
        print(f"  Losses > ₹500:    {len(df_trades[df_trades['total_pnl'] < -500])} trades")
        
        print(f"\n{'='*80}\n")
        return df_trades
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self):
        """Run all advanced tests"""
        self.load_data()
        
        # Test 1: Market Regime
        print(f"Running Test 1/5...\n")
        self.analyze_market_regime()
        
        # Test 2: Win/Loss Streaks
        print(f"Running Test 2/5...\n")
        self.analyze_streaks()
        
        # Test 3: Correlation
        print(f"Running Test 3/5...\n")
        self.analyze_correlation()
        
        # Test 4: Sharpe Ratio
        print(f"Running Test 4/5...\n")
        self.calculate_sharpe_ratio()
        
        # Test 5: Trade Distribution
        print(f"Running Test 5/5...\n")
        self.analyze_trade_distribution()
        
        print(f"{'='*80}")
        print(f"✅ ALL ADVANCED TESTS COMPLETE!")
        print(f"{'='*80}\n")
        
        print(f"🎯 NEXT STEPS:")
        print(f"  1. Review all test results above")
        print(f"  2. Identify strengths and weaknesses")
        print(f"  3. Prepare for paper trading")
        print(f"  4. Deploy with confidence!\n")


if __name__ == '__main__':
    # Run all advanced tests
    analyzer = AdvancedStrategyAnalysis(
        trades_file='trades.json',
        data_file='data.csv',
        initial_capital=30000
    )
    analyzer.run_all_tests()
