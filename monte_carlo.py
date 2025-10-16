# monte_carlo.py - Monte Carlo Simulation for Trade Robustness Testing

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class MonteCarloSimulation:
    def __init__(self, trades_file='trades.json', initial_capital=30000, num_simulations=10000):
        """
        Monte Carlo Simulation for strategy robustness testing
        
        Args:
            trades_file: Path to trades JSON file
            initial_capital: Starting capital
            num_simulations: Number of random shuffles (default 10,000)
        """
        self.trades_file = trades_file
        self.initial_capital = initial_capital
        self.num_simulations = num_simulations
        self.trades = []
        self.simulation_results = []
        
    def load_trades(self):
        """Load trades from JSON file"""
        print(f"\n{'='*80}")
        print(f"🎲 MONTE CARLO SIMULATION - LOADING TRADES")
        print(f"{'='*80}\n")
        
        with open(self.trades_file, 'r') as f:
            self.trades = json.load(f)
        
        print(f"✓ Loaded {len(self.trades)} trades from {self.trades_file}")
        
        # Extract just the PnL values (what we need for simulation)
        self.trade_pnls = [trade['total_pnl'] for trade in self.trades]
        
        print(f"✓ Original ROI: {((sum(self.trade_pnls) + self.initial_capital) / self.initial_capital - 1) * 100:.2f}%")
        print(f"✓ Original Max Drawdown: {self._calculate_max_drawdown(self.trade_pnls, self.initial_capital):.2f}%\n")
        
    def _calculate_max_drawdown(self, pnls, starting_capital):
        """Calculate maximum drawdown from PnL sequence"""
        capital = starting_capital
        peak = starting_capital
        max_dd = 0
        
        for pnl in pnls:
            capital += pnl
            peak = max(peak, capital)
            drawdown = (peak - capital) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def run_simulation(self):
        """Run Monte Carlo simulation by shuffling trades"""
        print(f"🎲 Running {self.num_simulations:,} simulations...\n")
        
        self.simulation_results = []
        
        for i in range(self.num_simulations):
            # Shuffle the trade PnLs randomly
            shuffled_pnls = random.sample(self.trade_pnls, len(self.trade_pnls))
            
            # Calculate metrics for this shuffled sequence
            final_capital = self.initial_capital + sum(shuffled_pnls)
            roi = (final_capital / self.initial_capital - 1) * 100
            max_dd = self._calculate_max_drawdown(shuffled_pnls, self.initial_capital)
            
            self.simulation_results.append({
                'simulation_number': i + 1,
                'final_capital': final_capital,
                'roi': roi,
                'max_drawdown': max_dd
            })
            
            # Progress update every 1000 simulations
            if (i + 1) % 1000 == 0:
                print(f"  Completed: {i + 1:,} / {self.num_simulations:,} simulations")
        
        print(f"\n✓ All {self.num_simulations:,} simulations complete!\n")
        
    def analyze_results(self):
        """Analyze Monte Carlo simulation results"""
        df = pd.DataFrame(self.simulation_results)
        
        print(f"{'='*80}")
        print(f"📊 MONTE CARLO SIMULATION RESULTS")
        print(f"{'='*80}\n")
        
        # Original backtest results
        original_roi = ((self.initial_capital + sum(self.trade_pnls)) / self.initial_capital - 1) * 100
        original_dd = self._calculate_max_drawdown(self.trade_pnls, self.initial_capital)
        
        print(f"🎯 ORIGINAL BACKTEST (Sequential Order):")
        print(f"  Final Capital: ₹{self.initial_capital + sum(self.trade_pnls):,.2f}")
        print(f"  ROI: {original_roi:.2f}%")
        print(f"  Max Drawdown: {original_dd:.2f}%\n")
        
        # Monte Carlo statistics
        print(f"📈 MONTE CARLO STATISTICS ({self.num_simulations:,} simulations):\n")
        
        print(f"ROI Distribution:")
        print(f"  Best Case (95th percentile):  {df['roi'].quantile(0.95):.2f}%")
        print(f"  75th percentile:               {df['roi'].quantile(0.75):.2f}%")
        print(f"  Median (50th percentile):      {df['roi'].quantile(0.50):.2f}%")
        print(f"  Mean (Average):                {df['roi'].mean():.2f}%")
        print(f"  25th percentile:               {df['roi'].quantile(0.25):.2f}%")
        print(f"  Worst Case (5th percentile):   {df['roi'].quantile(0.05):.2f}%\n")
        
        print(f"Max Drawdown Distribution:")
        print(f"  Best Case (5th percentile):    {df['max_drawdown'].quantile(0.05):.2f}%")
        print(f"  25th percentile:               {df['max_drawdown'].quantile(0.25):.2f}%")
        print(f"  Median (50th percentile):      {df['max_drawdown'].quantile(0.50):.2f}%")
        print(f"  Mean (Average):                {df['max_drawdown'].mean():.2f}%")
        print(f"  75th percentile:               {df['max_drawdown'].quantile(0.75):.2f}%")
        print(f"  Worst Case (95th percentile):  {df['max_drawdown'].quantile(0.95):.2f}%\n")
        
        # Probability analysis
        positive_sims = len(df[df['roi'] > 0])
        prob_profit = (positive_sims / self.num_simulations) * 100
        
        roi_above_original = len(df[df['roi'] >= original_roi])
        prob_beat_original = (roi_above_original / self.num_simulations) * 100
        
        print(f"🎲 PROBABILITY ANALYSIS:")
        print(f"  Probability of Profit:              {prob_profit:.2f}%")
        print(f"  Probability of ROI ≥ Original:      {prob_beat_original:.2f}%")
        print(f"  Probability of ROI ≥ 200%:          {(len(df[df['roi'] >= 200]) / self.num_simulations * 100):.2f}%")
        print(f"  Probability of Max DD < 25%:        {(len(df[df['max_drawdown'] < 25]) / self.num_simulations * 100):.2f}%")
        print(f"  Probability of Max DD < 40%:        {(len(df[df['max_drawdown'] < 40]) / self.num_simulations * 100):.2f}%\n")
        
        # Risk assessment
        print(f"⚠️ RISK ASSESSMENT:")
        risk_of_loss = 100 - prob_profit
        print(f"  Risk of Overall Loss:               {risk_of_loss:.2f}%")
        
        worst_roi = df['roi'].min()
        print(f"  Worst Possible ROI:                 {worst_roi:.2f}%")
        
        worst_dd = df['max_drawdown'].max()
        print(f"  Worst Possible Drawdown:            {worst_dd:.2f}%\n")
        
        # Confidence intervals
        print(f"📊 CONFIDENCE INTERVALS:")
        print(f"  90% Confidence ROI Range:           {df['roi'].quantile(0.05):.2f}% to {df['roi'].quantile(0.95):.2f}%")
        print(f"  90% Confidence Drawdown Range:      {df['max_drawdown'].quantile(0.05):.2f}% to {df['max_drawdown'].quantile(0.95):.2f}%\n")
        
        return df
    
    def generate_charts(self, df):
        """Generate visualization charts"""
        print(f"📊 Generating visualization charts...\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Monte Carlo Simulation Results - Trade Robustness Analysis', fontsize=16, fontweight='bold')
        
        # 1. ROI Distribution (Histogram)
        ax1 = axes[0, 0]
        ax1.hist(df['roi'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(df['roi'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["roi"].mean():.1f}%')
        ax1.axvline(df['roi'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["roi"].median():.1f}%')
        original_roi = ((self.initial_capital + sum(self.trade_pnls)) / self.initial_capital - 1) * 100
        ax1.axvline(original_roi, color='orange', linestyle='-', linewidth=2, label=f'Original: {original_roi:.1f}%')
        ax1.set_xlabel('ROI (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('ROI Distribution (10,000 Simulations)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Max Drawdown Distribution (Histogram)
        ax2 = axes[0, 1]
        ax2.hist(df['max_drawdown'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
        ax2.axvline(df['max_drawdown'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["max_drawdown"].mean():.1f}%')
        ax2.axvline(df['max_drawdown'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["max_drawdown"].median():.1f}%')
        original_dd = self._calculate_max_drawdown(self.trade_pnls, self.initial_capital)
        ax2.axvline(original_dd, color='orange', linestyle='-', linewidth=2, label=f'Original: {original_dd:.1f}%')
        ax2.set_xlabel('Max Drawdown (%)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Max Drawdown Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ROI vs Max Drawdown Scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['max_drawdown'], df['roi'], alpha=0.3, s=10, c=df['roi'], cmap='RdYlGn')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.scatter([original_dd], [original_roi], color='orange', s=200, marker='*', 
                   edgecolors='black', linewidths=2, label='Original Backtest', zorder=5)
        ax3.set_xlabel('Max Drawdown (%)', fontweight='bold')
        ax3.set_ylabel('ROI (%)', fontweight='bold')
        ax3.set_title('ROI vs Max Drawdown Relationship', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='ROI (%)')
        
        # 4. Cumulative Probability Distribution
        ax4 = axes[1, 1]
        sorted_roi = np.sort(df['roi'])
        cumulative_prob = np.arange(1, len(sorted_roi) + 1) / len(sorted_roi) * 100
        ax4.plot(sorted_roi, cumulative_prob, color='blue', linewidth=2)
        ax4.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax4.axvline(original_roi, color='orange', linestyle='-', linewidth=2, label=f'Original: {original_roi:.1f}%')
        ax4.axhline(50, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Median (50%)')
        ax4.set_xlabel('ROI (%)', fontweight='bold')
        ax4.set_ylabel('Cumulative Probability (%)', fontweight='bold')
        ax4.set_title('Cumulative Probability of ROI', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = 'monte_carlo_analysis.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Chart saved: {chart_filename}\n")
        
        # Save results to CSV
        csv_filename = 'monte_carlo_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"✓ Results saved: {csv_filename}")
        
        print(f"{'='*80}\n")
        
    def run(self):
        """Run complete Monte Carlo simulation"""
        self.load_trades()
        self.run_simulation()
        df = self.analyze_results()
        self.generate_charts(df)
        
        print(f"✅ MONTE CARLO SIMULATION COMPLETE!")
        print(f"   View results: monte_carlo_analysis.png")
        print(f"   View data: monte_carlo_results.csv\n")


if __name__ == '__main__':
    # Run Monte Carlo simulation with 10,000 iterations
    mc = MonteCarloSimulation(
        trades_file='trades.json',
        initial_capital=30000,
        num_simulations=10000
    )
    mc.run()
