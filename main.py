# main.py - V2 IMPROVED (BEST STRATEGY) - ALGO ACE BY TRADENITI

import pandas as pd
import json
from class_file import TradingStrategy
from report_generator import ReportGenerator
from datetime import datetime


def calculate_roi_on_invested_capital(trades):
    """Calculate ROI based on actual margin used"""
    total_margin_used = 0
    total_pnl = 0
    
    for trade in trades:
        total_margin_used += trade['margin_used']
        total_pnl += trade['total_pnl']
    
    if total_margin_used == 0:
        return 0
    
    roi = (total_pnl / total_margin_used) * 100
    return roi


def main():
    print("=" * 80)
    print("SILVERMIC V2 IMPROVED STRATEGY BACKTEST - ALGO ACE BY TRADENITI")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data from data.csv...")
    df = pd.read_csv('data.csv')
    print(f"✓ Loaded {len(df)} rows of data")
    print(f"  Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
    print()
    
    # Initialize and run strategy - CHANGED TO ₹30,000
    print("Running backtest with ₹30,000 initial capital...")
    strategy = TradingStrategy(initial_capital=50000, expense_per_trade=10)
    trades = strategy.run(df)
    
    # Save trades to JSON
    print("\nSaving trade data...")
    trades_json_path = 'trades.json'
    with open(trades_json_path, 'w') as f:
        json.dump(trades, f, indent=4, default=str)
    print(f"✓ Trades saved to {trades_json_path}")
    
    # Save price history to JSON for chart generation
    print("Saving price history for charts...")
    price_history_path = 'price_history.json'
    with open(price_history_path, 'w') as f:
        json.dump(strategy.price_history, f, indent=4, default=str)
    print(f"✓ Price history saved to {price_history_path}")
    
    # Calculate additional metrics
    if trades:
        roi_invested = calculate_roi_on_invested_capital(trades)
        wins = len([t for t in trades if t['pnl_before_expense'] > 0])
        losses = len(trades) - wins
        win_rate = (wins / len(trades) * 100) if trades else 0
        
        avg_win = sum([t['total_pnl'] for t in trades if t['total_pnl'] > 0]) / wins if wins > 0 else 0
        avg_loss = sum([t['total_pnl'] for t in trades if t['total_pnl'] < 0]) / losses if losses > 0 else 0
        
        profit_factor = abs(sum([t['total_pnl'] for t in trades if t['total_pnl'] > 0]) / 
                           sum([t['total_pnl'] for t in trades if t['total_pnl'] < 0])) if losses > 0 else float('inf')
        
        # Find best and worst trades
        best_trade = max(trades, key=lambda x: x['total_pnl'])
        worst_trade = min(trades, key=lambda x: x['total_pnl'])
        
        print("\n" + "=" * 80)
        print("DETAILED METRICS")
        print("=" * 80)
        print(f"ROI on Invested Capital: {roi_invested:.2f}%")
        print(f"Average Win: ₹{avg_win:,.2f}")
        print(f"Average Loss: ₹{avg_loss:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"\nBest Trade: #{best_trade['trade_number']} - {best_trade['type']} - ₹{best_trade['total_pnl']:,.2f}")
        print(f"Worst Trade: #{worst_trade['trade_number']} - {worst_trade['type']} - ₹{worst_trade['total_pnl']:,.2f}")
        
        # Generate HTML report
        print("\n" + "=" * 80)
        print("GENERATING HTML REPORT")
        print("=" * 80)
        
        report_gen = ReportGenerator(
            trades=trades,
            initial_capital=strategy.initial_capital,
            final_capital=strategy.available_capital,
            max_drawdown=strategy.max_drawdown,
            price_history=strategy.price_history
        )
        
        report_path = report_gen.generate_report()
        print(f"✓ HTML report generated: {report_path}")
        print(f"✓ Charts embedded in report")
        
        # Generate TradingView Lightweight Charts HTML
        print("\nGenerating interactive chart viewer...")
        chart_viewer_path = report_gen.generate_tradingview_chart()
        print(f"✓ TradingView chart generated: {chart_viewer_path}")
        
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE!")
        print("=" * 80)
        print(f"📊 View detailed report: {report_path}")
        print(f"📈 View interactive chart: {chart_viewer_path}")
        print("=" * 80)
    else:
        print("\n⚠ No trades executed")


if __name__ == "__main__":
    main()
