# report_generator.py - ALGO ACE BY TRADENITI - NO SAMPLING = ULTRA RESPONSIVE

import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    def __init__(self, trades, initial_capital, final_capital, max_drawdown, price_history):
        self.trades = trades
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.max_drawdown = max_drawdown
        self.price_history = price_history
        
    def generate_interactive_price_chart_html(self):
        """Generate Plotly interactive chart embedded as HTML"""
        if not self.price_history or not self.trades:
            return '<p style="text-align:center; color:#999;">⚠️ No data available for chart</p>'
        
        try:
            df = pd.DataFrame(self.price_history)
            df['time'] = pd.to_datetime(df['time'])
            
            # Sample if too large (keep for Plotly report chart)
            if len(df) > 10000:
                sample_rate = len(df) // 10000
                df = df.iloc[::sample_rate].copy()
                df = df.reset_index(drop=True)
            
            # Prepare trade annotations
            annotations = []
            shapes = []
            
            for trade in self.trades:
                try:
                    entry_date_str = str(trade.get('entry_date', ''))
                    exit_date_str = str(trade.get('exit_date', ''))
                    
                    if (not entry_date_str or not exit_date_str or 
                        entry_date_str in ['', 'nan', 'None'] or 
                        exit_date_str in ['', 'nan', 'None']):
                        continue
                    
                    entry_time = pd.to_datetime(entry_date_str)
                    exit_time = pd.to_datetime(exit_date_str)
                    entry_price = trade['entry']
                    exit_price = trade['exit']
                    is_profit = trade['total_pnl'] > 0
                    
                    entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M:%S')
                    exit_time_str = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if trade['type'] == 'SHORT':
                        entry_color = '#FF6B6B'
                    else:
                        entry_color = '#4CAF50'
                    
                    exit_color = '#4CAF50' if is_profit else '#FF6B6B'
                    
                    annotations.append({
                        'x': entry_time_str,
                        'y': entry_price,
                        'text': f"{trade['type']}<br>#{trade['trade_number']}",
                        'showarrow': True,
                        'arrowhead': 2,
                        'arrowsize': 1,
                        'arrowwidth': 2,
                        'arrowcolor': entry_color,
                        'ax': 0,
                        'ay': -40 if trade['type'] != 'SHORT' else 40,
                        'font': {'size': 9, 'color': entry_color},
                        'bgcolor': 'rgba(255,255,255,0.9)',
                        'bordercolor': entry_color,
                        'borderwidth': 2,
                        'borderpad': 2
                    })
                    
                    annotations.append({
                        'x': exit_time_str,
                        'y': exit_price,
                        'text': f"EXIT<br>₹{trade['total_pnl']:,.0f}",
                        'showarrow': True,
                        'arrowhead': 2,
                        'arrowsize': 1,
                        'arrowwidth': 2,
                        'arrowcolor': exit_color,
                        'ax': 0,
                        'ay': 40 if trade['type'] != 'SHORT' else -40,
                        'font': {'size': 9, 'color': exit_color},
                        'bgcolor': 'rgba(255,255,255,0.9)',
                        'bordercolor': exit_color,
                        'borderwidth': 2,
                        'borderpad': 2
                    })
                    
                    shapes.append({
                        'type': 'line',
                        'x0': entry_time_str,
                        'y0': entry_price,
                        'x1': exit_time_str,
                        'y1': exit_price,
                        'line': {
                            'color': exit_color,
                            'width': 2,
                            'dash': 'dot'
                        }
                    })
                    
                except:
                    continue
            
            time_strings = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            
            plotly_html = f'''
            <div id="plotly-chart" style="width:100%; height:700px;"></div>
            <script>
                var trace = {{
                    x: {json.dumps(time_strings)},
                    close: {df['close'].tolist()},
                    high: {df['high'].tolist()},
                    low: {df['low'].tolist()},
                    open: {df['open'].tolist()},
                    type: 'candlestick',
                    name: 'SILVERMIC',
                    increasing: {{line: {{color: '#26a69a'}}}},
                    decreasing: {{line: {{color: '#ef5350'}}}}
                }};
                
                var layout = {{
                    title: {{
                        text: 'SILVERMIC - Interactive Price Chart',
                        font: {{size: 18, color: '#333'}}
                    }},
                    xaxis: {{
                        title: 'Date',
                        rangeslider: {{visible: false}},
                        type: 'date'
                    }},
                    yaxis: {{
                        title: 'Price (₹)',
                        side: 'right'
                    }},
                    annotations: {json.dumps(annotations)},
                    shapes: {json.dumps(shapes)},
                    hovermode: 'x unified',
                    dragmode: 'zoom',
                    plot_bgcolor: '#f9f9f9',
                    paper_bgcolor: 'white'
                }};
                
                var config = {{
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                }};
                
                Plotly.newPlot('plotly-chart', [trace], layout, config);
            </script>
            '''
            
            return plotly_html
        
        except Exception as e:
            print(f"  ✗ Error generating Plotly chart: {e}")
            return '<p style="text-align:center; color:#999;">⚠️ Chart generation failed</p>'
    
    def generate_equity_curve_html(self):
        """Generate interactive equity curve with Plotly"""
        if not self.trades:
            return '<p style="text-align:center; color:#999;">⚠️ No trades to display</p>'
        
        try:
            equity = [self.initial_capital]
            trade_numbers = [0]
            
            for trade in self.trades:
                equity.append(trade['capital_after'])
                trade_numbers.append(trade['trade_number'])
            
            hover_texts = [f'Start<br>₹{self.initial_capital:,.0f}']
            for i, trade in enumerate(self.trades):
                hover_texts.append(f"Trade {trade['trade_number']}<br>₹{trade['capital_after']:,.0f}")
            
            plotly_html = f'''
            <div id="equity-chart" style="width:100%; height:500px;"></div>
            <script>
                var equityTrace = {{
                    x: {json.dumps(trade_numbers)},
                    y: {json.dumps(equity)},
                    mode: 'lines+markers',
                    name: 'Capital',
                    line: {{color: '#2962FF', width: 3}},
                    marker: {{size: 6, color: '#2962FF'}},
                    fill: 'tonexty',
                    fillcolor: 'rgba(41, 98, 255, 0.1)',
                    text: {json.dumps(hover_texts)},
                    hovertemplate: '%{{text}}<extra></extra>'
                }};
                
                var baselineTrace = {{
                    x: [0, {len(self.trades)}],
                    y: [{self.initial_capital}, {self.initial_capital}],
                    mode: 'lines',
                    name: 'Initial Capital',
                    line: {{color: 'gray', width: 2, dash: 'dash'}},
                    hoverinfo: 'skip'
                }};
                
                var equityLayout = {{
                    title: {{
                        text: 'Equity Curve - Capital Growth',
                        font: {{size: 18, color: '#333'}}
                    }},
                    xaxis: {{
                        title: 'Trade Number',
                        gridcolor: '#e0e0e0'
                    }},
                    yaxis: {{
                        title: 'Capital (₹)',
                        tickformat: ',.0f',
                        gridcolor: '#e0e0e0'
                    }},
                    hovermode: 'x unified',
                    plot_bgcolor: '#f9f9f9',
                    paper_bgcolor: 'white',
                    showlegend: true
                }};
                
                var equityConfig = {{
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                }};
                
                Plotly.newPlot('equity-chart', [equityTrace, baselineTrace], equityLayout, equityConfig);
            </script>
            '''
            
            print("  ✓ Equity curve generated successfully")
            return plotly_html
        
        except Exception as e:
            print(f"  ✗ Error generating equity curve: {e}")
            import traceback
            traceback.print_exc()
            return '<p style="text-align:center; color:#999;">⚠️ Equity curve failed</p>'
    
    def generate_report(self):
        """Generate HTML report"""
        with open('template.html', 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Calculate metrics
        total_trades = len(self.trades)
        wins = len([t for t in self.trades if t['pnl_before_expense'] > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum([t['total_pnl'] for t in self.trades])
        total_expenses = sum([t['expense'] for t in self.trades])
        
        avg_win = sum([t['total_pnl'] for t in self.trades if t['total_pnl'] > 0]) / wins if wins > 0 else 0
        avg_loss = sum([t['total_pnl'] for t in self.trades if t['total_pnl'] < 0]) / losses if losses > 0 else 0
        
        profit_factor = abs(sum([t['total_pnl'] for t in self.trades if t['total_pnl'] > 0]) / 
                           sum([t['total_pnl'] for t in self.trades if t['total_pnl'] < 0])) if losses > 0 else float('inf')
        
        roi = ((self.final_capital - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        best_trade = max(self.trades, key=lambda x: x['total_pnl']) if self.trades else None
        worst_trade = min(self.trades, key=lambda x: x['total_pnl']) if self.trades else None
        
        # Generate charts
        print("  Generating Plotly price chart...")
        price_chart_html = self.generate_interactive_price_chart_html()
        print("  Generating equity curve...")
        equity_chart_html = self.generate_equity_curve_html()
        
        # Strategy breakdown
        strategy_stats = {}
        for strategy in ['PRIMARY', 'SECONDARY', 'REENTRY-1', 'SHORT']:
            strat_trades = [t for t in self.trades if t['type'] == strategy]
            if strat_trades:
                strat_wins = len([t for t in strat_trades if t['pnl_before_expense'] > 0])
                strat_pnl = sum([t['total_pnl'] for t in strat_trades])
                strat_win_rate = (strat_wins / len(strat_trades) * 100)
                strategy_stats[strategy] = {
                    'count': len(strat_trades),
                    'wins': strat_wins,
                    'win_rate': strat_win_rate,
                    'pnl': strat_pnl
                }
        
        # Build trades table
        trades_html = ""
        for trade in self.trades:
            pnl_class = "profit" if trade['total_pnl'] > 0 else "loss"
            row_class = ""
            if best_trade and trade['trade_number'] == best_trade['trade_number']:
                row_class = 'class="best-trade"'
            elif worst_trade and trade['trade_number'] == worst_trade['trade_number']:
                row_class = 'class="worst-trade"'
            
            trades_html += f"""
            <tr {row_class}>
                <td>{trade['trade_number']}</td>
                <td>{trade['entry_date_display']}</td>
                <td>{trade['exit_date_display']}</td>
                <td><span class="badge badge-{trade['type'].lower().replace('-', '')}">{trade['type']}</span></td>
                <td>₹{trade['entry']:,.2f}</td>
                <td>₹{trade['exit']:,.2f}</td>
                <td class="{pnl_class}">₹{trade['total_pnl']:,.2f}</td>
                <td class="{pnl_class}">₹{trade['cumulative_pnl']:,.2f}</td>
                <td>{trade['roi']:.2f}%</td>
                <td><span class="badge badge-exit">{trade['exit_reason']}</span></td>
            </tr>
            """
        
        # Build strategy breakdown
        strategy_html = ""
        for strategy, stats in strategy_stats.items():
            pnl_class = "profit" if stats['pnl'] > 0 else "loss"
            strategy_html += f"""
            <tr>
                <td><span class="badge badge-{strategy.lower().replace('-', '')}">{strategy}</span></td>
                <td>{stats['count']}</td>
                <td>{stats['wins']}</td>
                <td>{stats['win_rate']:.1f}%</td>
                <td class="{pnl_class}">₹{stats['pnl']:,.2f}</td>
            </tr>
            """
        
        # Replace placeholders
        html = template.replace('{{INITIAL_CAPITAL}}', f'₹{self.initial_capital:,.2f}')
        html = html.replace('{{FINAL_CAPITAL}}', f'₹{self.final_capital:,.2f}')
        html = html.replace('{{TOTAL_TRADES}}', str(total_trades))
        html = html.replace('{{WINS}}', str(wins))
        html = html.replace('{{LOSSES}}', str(losses))
        html = html.replace('{{WIN_RATE}}', f'{win_rate:.2f}%')
        html = html.replace('{{TOTAL_PNL}}', f'₹{total_pnl:,.2f}')
        html = html.replace('{{TOTAL_EXPENSES}}', f'₹{total_expenses:,.2f}')
        html = html.replace('{{ROI}}', f'{roi:.2f}%')
        html = html.replace('{{MAX_DRAWDOWN}}', f'{self.max_drawdown:.2f}%')
        html = html.replace('{{AVG_WIN}}', f'₹{avg_win:,.2f}')
        html = html.replace('{{AVG_LOSS}}', f'₹{avg_loss:,.2f}')
        html = html.replace('{{PROFIT_FACTOR}}', f'{profit_factor:.2f}')
        html = html.replace('{{TRADES_TABLE}}', trades_html)
        html = html.replace('{{STRATEGY_BREAKDOWN}}', strategy_html)
        
        if best_trade:
            html = html.replace('{{BEST_TRADE}}', 
                f"Trade #{best_trade['trade_number']} - {best_trade['type']} - ₹{best_trade['total_pnl']:,.2f}")
        else:
            html = html.replace('{{BEST_TRADE}}', 'N/A')
        
        if worst_trade:
            html = html.replace('{{WORST_TRADE}}', 
                f"Trade #{worst_trade['trade_number']} - {worst_trade['type']} - ₹{worst_trade['total_pnl']:,.2f}")
        else:
            html = html.replace('{{WORST_TRADE}}', 'N/A')
        
        html = html.replace('{{PRICE_CHART}}', price_chart_html)
        html = html.replace('{{EQUITY_CHART}}', equity_chart_html)
        html = html.replace('{{TIMESTAMP}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        report_path = 'backtest_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return report_path
    
    def generate_tradingview_chart(self):
        """Generate TradingView chart - ZERO SAMPLING = ULTRA RESPONSIVE LINES"""
        if not self.price_history or not self.trades:
            print("  Warning: No data for chart")
            return None
        
        try:
            df = pd.DataFrame(self.price_history)
            df['time'] = pd.to_datetime(df['time'])
            
            # ✅ NO SAMPLING - USE ALL DATA POINTS
            df_sampled = df.copy()
            
            print(f"  ✓ Using ALL {len(df_sampled)} data points (NO SAMPLING) for ultra-responsive lines")
            
            # Calculate indicators on full dataset
            fib_length = min(264, len(df_sampled))
            
            # Manual calculation for Top/Bottom (same as class_file.py)
            top_line_values = []
            bottom_line_values = []
            
            for i in range(len(df_sampled)):
                start_idx = max(0, i - fib_length + 1)
                window_data = df_sampled['close'].iloc[start_idx:i+1]
                top_line_values.append(window_data.max())
                bottom_line_values.append(window_data.min())
            
            df_sampled['Top_Line'] = top_line_values
            df_sampled['Bottom_Line'] = bottom_line_values
            df_sampled['Pivot_Line'] = df_sampled['Top_Line'] - 0.50 * (df_sampled['Top_Line'] - df_sampled['Bottom_Line'])
            df_sampled['SMA2'] = df_sampled['close'].rolling(window=2).mean()
            
            # Tolerance zones
            tolerance_amount = df_sampled['Bottom_Line'] * 0.004
            df_sampled['Upper_Tolerance'] = df_sampled['Bottom_Line'] + tolerance_amount
            df_sampled['Lower_Tolerance'] = df_sampled['Bottom_Line'] - tolerance_amount
            
            # Prepare JSON data
            candle_data = []
            top_line_data = []
            bottom_line_data = []
            pivot_line_data = []
            sma2_data = []
            upper_tolerance_data = []
            lower_tolerance_data = []
            
            for _, row in df_sampled.iterrows():
                try:
                    timestamp = int(row['time'].timestamp())
                    candle_data.append({
                        'time': timestamp,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close'])
                    })
                    
                    if not pd.isna(row['Top_Line']):
                        top_line_data.append({'time': timestamp, 'value': float(row['Top_Line'])})
                    if not pd.isna(row['Bottom_Line']):
                        bottom_line_data.append({'time': timestamp, 'value': float(row['Bottom_Line'])})
                    if not pd.isna(row['Pivot_Line']):
                        pivot_line_data.append({'time': timestamp, 'value': float(row['Pivot_Line'])})
                    if not pd.isna(row['SMA2']):
                        sma2_data.append({'time': timestamp, 'value': float(row['SMA2'])})
                    if not pd.isna(row['Upper_Tolerance']):
                        upper_tolerance_data.append({'time': timestamp, 'value': float(row['Upper_Tolerance'])})
                    if not pd.isna(row['Lower_Tolerance']):
                        lower_tolerance_data.append({'time': timestamp, 'value': float(row['Lower_Tolerance'])})
                except:
                    continue
            
            # Trade markers
            all_markers = []
            for trade in self.trades:
                try:
                    entry_date_str = str(trade.get('entry_date', ''))
                    exit_date_str = str(trade.get('exit_date', ''))
                    
                    if (not entry_date_str or not exit_date_str or 
                        entry_date_str in ['', 'nan', 'None'] or 
                        exit_date_str in ['', 'nan', 'None']):
                        continue
                    
                    entry_time = int(pd.to_datetime(entry_date_str).timestamp())
                    exit_time = int(pd.to_datetime(exit_date_str).timestamp())
                    is_profit = trade['total_pnl'] > 0
                    
                    if trade['type'] == 'SHORT':
                        all_markers.append({
                            'time': entry_time,
                            'position': 'aboveBar',
                            'color': '#e91e63',
                            'shape': 'arrowDown',
                            'text': f"SHORT #{trade['trade_number']}"
                        })
                        all_markers.append({
                            'time': exit_time,
                            'position': 'belowBar',
                            'color': '#26a69a' if is_profit else '#ef5350',
                            'shape': 'arrowUp',
                            'text': f"₹{trade['total_pnl']:,.0f}"
                        })
                    else:
                        all_markers.append({
                            'time': entry_time,
                            'position': 'belowBar',
                            'color': '#26a69a',
                            'shape': 'arrowUp',
                            'text': f"{trade['type'][:3]} #{trade['trade_number']}"
                        })
                        all_markers.append({
                            'time': exit_time,
                            'position': 'aboveBar',
                            'color': '#26a69a' if is_profit else '#ef5350',
                            'shape': 'arrowDown',
                            'text': f"₹{trade['total_pnl']:,.0f}"
                        })
                except:
                    continue
            
            print(f"  ✓ Generated: {len(candle_data)} candles, {len(all_markers)} trade markers, {len(top_line_data)} indicator points")
            
            # Generate HTML
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algo Ace by Tradeniti - Ultra Responsive Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
            overflow: hidden;
        }}
        .container {{ display: flex; flex-direction: column; height: 100vh; }}
        
        .header {{
            background-color: #1e222d;
            border-bottom: 1px solid #2a2e39;
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-left {{ display: flex; align-items: center; gap: 20px; }}
        .symbol {{ font-size: 24px; font-weight: 700; color: #ffffff; }}
        .brand {{ font-size: 11px; color: #667eea; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
        .header-right {{ display: flex; gap: 12px; }}
        .btn {{
            padding: 9px 16px;
            border: 1px solid #2a2e39;
            background-color: #1e222d;
            color: #d1d4dc;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{ background-color: #2a2e39; border-color: #454d5e; }}
        .btn.active {{ background-color: #2962ff; border-color: #2962ff; color: #ffffff; }}
        
        .toolbar {{
            background-color: #1e222d;
            border-bottom: 1px solid #2a2e39;
            padding: 12px 24px;
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .toolbar-section {{
            display: flex;
            gap: 6px;
            border-right: 1px solid #2a2e39;
            padding-right: 16px;
            align-items: center;
        }}
        .toolbar-section:last-child {{ border-right: none; }}
        .toolbar-label {{ color: #a6acb7; font-size: 12px; margin-right: 8px; font-weight: 500; }}
        .tool-btn {{
            padding: 7px 12px;
            border: none;
            background-color: transparent;
            color: #d1d4dc;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .tool-btn:hover {{ background-color: #2a2e39; }}
        .tool-btn.active {{ background-color: #2962ff; color: #ffffff; }}
        
        .chart-container {{ flex: 1; background-color: #131722; position: relative; }}
        #chart {{ width: 100%; height: 100%; }}
        
        .legend-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: rgba(30, 34, 45, 0.95);
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 16px;
            font-size: 12px;
            color: #d1d4dc;
            z-index: 100;
            min-width: 220px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .legend-title {{ font-weight: 600; color: #ffffff; margin-bottom: 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .legend-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
        .legend-color {{ width: 30px; height: 3px; border-radius: 2px; }}
        
        .status-badge {{
            position: absolute;
            top: 20px;
            right: 20px;
            background-color: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-left">
                <div>
                    <div class="symbol">SILVERMIC</div>
                    <div class="brand">Algo Ace by Tradeniti</div>
                </div>
            </div>
            <div class="header-right">
                <button class="btn" onclick="downloadChart()">⬇️ Screenshot</button>
                <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            </div>
        </div>

        <div class="toolbar">
            <div class="toolbar-section">
                <span class="toolbar-label">Indicators:</span>
                <button class="tool-btn active" onclick="toggleIndicator('top')" id="btn-top">Top Line</button>
                <button class="tool-btn active" onclick="toggleIndicator('bottom')" id="btn-bottom">Bottom Line</button>
                <button class="tool-btn active" onclick="toggleIndicator('pivot')" id="btn-pivot">Pivot Line</button>
                <button class="tool-btn active" onclick="toggleIndicator('sma2')" id="btn-sma2">SMA2</button>
            </div>
            <div class="toolbar-section">
                <span class="toolbar-label">Tolerance:</span>
                <button class="tool-btn active" onclick="toggleIndicator('tolerance')" id="btn-tolerance">Upper/Lower</button>
            </div>
            <div class="toolbar-section">
                <button class="tool-btn" onclick="resetChart()">🔄 Reset</button>
                <button class="tool-btn" onclick="fitContent()">📐 Fit</button>
            </div>
        </div>

        <div class="chart-container">
            <div id="chart"></div>
            
            <div class="status-badge">
                ✓ ULTRA RESPONSIVE - ALL DATA POINTS
            </div>
            
            <div class="legend-overlay">
                <div class="legend-title">Fibonacci Lines</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
                    <span>Top Line (264H)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF5722;"></div>
                    <span>Bottom Line (264L)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0;"></div>
                    <span>Pivot (50% Fib)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800;"></div>
                    <span>SMA2</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(76, 175, 80, 0.4); height: 8px;"></div>
                    <span>Upper Tolerance</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(244, 67, 54, 0.4); height: 8px;"></div>
                    <span>Lower Tolerance</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chartData = {json.dumps(candle_data)};
        const markers = {json.dumps(all_markers)};
        const topLineData = {json.dumps(top_line_data)};
        const bottomLineData = {json.dumps(bottom_line_data)};
        const pivotLineData = {json.dumps(pivot_line_data)};
        const sma2Data = {json.dumps(sma2_data)};
        const upperToleranceData = {json.dumps(upper_tolerance_data)};
        const lowerToleranceData = {json.dumps(lower_tolerance_data)};

        const chartDiv = document.getElementById('chart');
        const chart = LightweightCharts.createChart(chartDiv, {{
            layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
            width: chartDiv.clientWidth,
            height: chartDiv.clientHeight,
            timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#2a2e39' }},
            grid: {{ horzLines: {{ color: '#2a2e39' }}, vertLines: {{ color: '#2a2e39' }} }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{ color: '#667eea', width: 1, style: LightweightCharts.LineStyle.Dashed }},
                horzLine: {{ color: '#667eea', width: 1, style: LightweightCharts.LineStyle.Dashed }}
            }},
            rightPriceScale: {{ borderColor: '#2a2e39' }}
        }});

        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a', downColor: '#ef5350',
            borderUpColor: '#26a69a', borderDownColor: '#ef5350',
            wickUpColor: '#26a69a', wickDownColor: '#ef5350'
        }});
        candlestickSeries.setData(chartData);
        candlestickSeries.setMarkers(markers);

        const topLineSeries = chart.addLineSeries({{ color: '#2196F3', lineWidth: 2 }});
        topLineSeries.setData(topLineData);

        const bottomLineSeries = chart.addLineSeries({{ color: '#FF5722', lineWidth: 2 }});
        bottomLineSeries.setData(bottomLineData);

        const pivotLineSeries = chart.addLineSeries({{ color: '#9C27B0', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed }});
        pivotLineSeries.setData(pivotLineData);

        const sma2Series = chart.addLineSeries({{ color: '#FF9800', lineWidth: 1.5 }});
        sma2Series.setData(sma2Data);

        const upperToleranceSeries = chart.addLineSeries({{ color: 'rgba(76, 175, 80, 0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted }});
        upperToleranceSeries.setData(upperToleranceData);

        const lowerToleranceSeries = chart.addLineSeries({{ color: 'rgba(244, 67, 54, 0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted }});
        lowerToleranceSeries.setData(lowerToleranceData);

        chart.timeScale().fitContent();

        let visibility = {{ top: true, bottom: true, pivot: true, sma2: true, tolerance: true }};

        function toggleIndicator(type) {{
            visibility[type] = !visibility[type];
            document.getElementById('btn-' + type).classList.toggle('active');
            if (type === 'top') topLineSeries.applyOptions({{ visible: visibility.top }});
            else if (type === 'bottom') bottomLineSeries.applyOptions({{ visible: visibility.bottom }});
            else if (type === 'pivot') pivotLineSeries.applyOptions({{ visible: visibility.pivot }});
            else if (type === 'sma2') sma2Series.applyOptions({{ visible: visibility.sma2 }});
            else if (type === 'tolerance') {{
                upperToleranceSeries.applyOptions({{ visible: visibility.tolerance }});
                lowerToleranceSeries.applyOptions({{ visible: visibility.tolerance }});
            }}
        }}

        function resetChart() {{ chart.timeScale().resetTimeScale(); }}
        function fitContent() {{ chart.timeScale().fitContent(); }}

        function downloadChart() {{
            html2canvas(document.querySelector('.container'), {{ backgroundColor: '#131722', scale: 2 }}).then(canvas => {{
                const link = document.createElement('a');
                link.download = 'silvermic-ultra-responsive-' + new Date().getTime() + '.png';
                link.href = canvas.toDataURL();
                link.click();
            }});
        }}

        let isFullscreen = false;
        function toggleFullscreen() {{
            if (!isFullscreen) {{
                document.documentElement.requestFullscreen();
                isFullscreen = true;
            }} else {{
                document.exitFullscreen();
                isFullscreen = false;
            }}
        }}

        window.addEventListener('resize', () => {{
            chart.applyOptions({{ width: chartDiv.clientWidth, height: chartDiv.clientHeight }});
        }});

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'f' || e.key === 'F') toggleFullscreen();
            if (e.key === 's' || e.key === 'S') downloadChart();
        }});
    </script>
</body>
</html>'''
            
            chart_path = 'chart_viewer.html'
            with open(chart_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  ✓ Ultra-responsive chart saved: {chart_path}")
            return chart_path
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
