import sys
from pathlib import Path

# Load environment variables from .env file - ALWAYS from project root
# This ensures it works regardless of which directory you run from
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env", override=True)
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.cli.input import (
    parse_cli_inputs,
)

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json


def format_sentiment_insights(reasoning: dict) -> str:
    """Convert sentiment analyst's nested metrics into human-readable Chinese insights."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    category_names = {
        "insider_trading": "👤 内幕交易",
        "news_sentiment": "📰 新闻情绪",
        "social_media": "💬 社交媒体",
        "analyst_ratings": "📋 分析师评级"
    }
    
    insights = []
    
    for category_key, category_data in reasoning.items():
        if category_key == "combined_analysis":
            continue  # Skip the combined section for now
            
        if category_key not in category_names:
            continue
            
        name = category_names[category_key]
        signal = category_data.get("signal", "neutral").upper()
        signal_cn = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}.get(signal, signal)
        confidence = category_data.get("confidence", 0)
        metrics = category_data.get("metrics", {})
        
        # Generate metric explanations
        metric_notes = []
        if category_key == "insider_trading":
            total = metrics.get("total_trades", 0)
            bullish = metrics.get("bullish_trades", 0)
            bearish = metrics.get("bearish_trades", 0)
            if total > 0:
                metric_notes.append(f"共{total}笔交易")
                if bullish > 0:
                    metric_notes.append(f"买入{bullish}")
                if bearish > 0:
                    metric_notes.append(f"卖出{bearish}")
        
        elif category_key == "news_sentiment":
            total = metrics.get("total_articles", 0)
            bullish = metrics.get("bullish_articles", 0)
            bearish = metrics.get("bearish_articles", 0)
            if total > 0:
                metric_notes.append(f"共{total}篇")
                metric_notes.append(f"正面{bullish}")
                metric_notes.append(f"负面{bearish}")
            else:
                metric_notes.append("无新闻数据")
        
        # Signal color class
        signal_class = f"signal-{signal}"
        
        # Build HTML for this category
        metrics_html = " · ".join(metric_notes)
        insights.append(f"""
            <div style="margin-bottom: 8px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-weight: 600; font-size: 0.8rem;">{name}</span>
                    <span style="font-weight: 700; font-size: 0.75rem;" class="{signal_class}">{signal_cn} {confidence}%</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">{metrics_html}</div>
            </div>
        """)
    
    # Add combined analysis summary
    if "combined_analysis" in reasoning:
        combined = reasoning["combined_analysis"]
        determination = combined.get("signal_determination", "")
        # Translate to Chinese
        if "Bullish" in determination:
            determination = "综合信号：看涨"
        elif "Bearish" in determination:
            determination = "综合信号：看跌"
        insights.append(f"""
            <div style="margin-top: 8px; padding: 8px; background: rgba(124, 58, 237, 0.15); border-radius: 8px; border-left: 3px solid #7c3aed;">
                <div style="font-size: 0.75rem; color: #a78bfa; font-weight: 600;">🎯 {determination}</div>
            </div>
        """)
    
    return "\n".join(insights)


def format_technical_insights(reasoning: dict) -> str:
    """Convert technical analyst's nested metrics into human-readable Chinese insights."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    # Check if this is the technical analyst nested structure
    strategy_names = {
        "trend_following": "📈 趋势跟踪",
        "mean_reversion": "🔄 均值回归",
        "momentum": "⚡ 动量分析",
        "volatility": "📊 波动率",
        "statistical_arbitrage": "📐 统计套利"
    }
    
    insights = []
    
    for strategy_key, strategy_data in reasoning.items():
        if strategy_key not in strategy_names:
            continue
            
        name = strategy_names[strategy_key]
        signal = strategy_data.get("signal", "neutral").upper()
        signal_cn = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}.get(signal, signal)
        confidence = strategy_data.get("confidence", 0)
        metrics = strategy_data.get("metrics", {})
        
        # Generate metric explanations
        metric_notes = []
        if strategy_key == "trend_following":
            adx = metrics.get("adx", 0)
            if adx > 25:
                metric_notes.append(f"ADX {adx:.1f} (趋势强)")
            elif adx > 20:
                metric_notes.append(f"ADX {adx:.1f} (有趋势)")
            else:
                metric_notes.append(f"ADX {adx:.1f} (震荡)")
        
        elif strategy_key == "mean_reversion":
            rsi = metrics.get("rsi_14", 50)
            z_score = metrics.get("z_score", 0)
            if rsi > 70:
                metric_notes.append(f"RSI {rsi:.1f} (超买)")
            elif rsi < 30:
                metric_notes.append(f"RSI {rsi:.1f} (超卖)")
            else:
                metric_notes.append(f"RSI {rsi:.1f} (中性)")
            if abs(z_score) > 1.5:
                metric_notes.append(f"Z {z_score:.2f}σ (偏离均值)")
        
        elif strategy_key == "momentum":
            mom_1m = metrics.get("momentum_1m", 0)
            vol_mom = metrics.get("volume_momentum", 0)
            if mom_1m > 0.1:
                metric_notes.append(f"1月动量 +{mom_1m:.1%}")
            elif mom_1m < -0.1:
                metric_notes.append(f"1月动量 {mom_1m:.1%}")
            if vol_mom > 1.2:
                metric_notes.append(f"放量 x{vol_mom:.1f}")
        
        elif strategy_key == "volatility":
            hv = metrics.get("historical_volatility", 0)
            if hv > 0.5:
                metric_notes.append(f"波动率 {hv:.1%} (高波动)")
            elif hv < 0.2:
                metric_notes.append(f"波动率 {hv:.1%} (低波动)")
        
        # Signal color class
        signal_class = f"signal-{signal}"
        
        # Build HTML for this strategy
        metrics_html = " · ".join(metric_notes)
        insights.append(f"""
            <div style="margin-bottom: 8px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-weight: 600; font-size: 0.8rem;">{name}</span>
                    <span style="font-weight: 700; font-size: 0.75rem;" class="{signal_class}">{signal_cn} {confidence}%</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">{metrics_html}</div>
            </div>
        """)
    
    return "\n".join(insights)


def generate_html_report(analyst_signals, decisions, timestamp):
    """Generate a beautiful HTML report in Chinese."""
    # Chinese translation helper for signals
    def cn(text):
        translations = {
            'BULLISH': '看涨', 'BEARISH': '看跌', 'NEUTRAL': '中性',
            'LONG': '做多', 'SHORT': '做空', 'HOLD': '持有'
        }
        return translations.get(text, text)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI对冲基金分析报告 - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #888;
            margin-bottom: 40px;
            font-size: 0.9rem;
        }}
        .ticker-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .ticker-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }}
        .ticker-name {{
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
        }}
        .decision-tag {{
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.1rem;
            text-transform: uppercase;
        }}
        .decision-LONG {{ background: linear-gradient(135deg, #10b981, #059669); }}
        .decision-SHORT {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
        .decision-HOLD {{ background: linear-gradient(135deg, #6366f1, #4f46e5); }}
        .analyst-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .analyst-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .analyst-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(124, 58, 237, 0.5);
        }}
        .analyst-name {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; color: #fff; }}
        .analyst-signal {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .signal-BULLISH {{ color: #10b981; font-weight: 700; }}
        .signal-BEARISH {{ color: #ef4444; font-weight: 700; }}
        .signal-NEUTRAL {{ color: #6366f1; font-weight: 700; }}
        .confidence-bar {{
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        .confidence-fill {{ height: 100%; border-radius: 3px; }}
        .confidence-BULLISH .confidence-fill {{ background: linear-gradient(90deg, #10b981, #34d399); }}
        .confidence-BEARISH .confidence-fill {{ background: linear-gradient(90deg, #ef4444, #f87171); }}
        .confidence-NEUTRAL .confidence-fill {{ background: linear-gradient(90deg, #6366f1, #818cf8); }}
        .analyst-reasoning {{ font-size: 0.85rem; color: #a0a0a0; line-height: 1.5; }}
        .decision-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .decision-item {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .decision-label {{ font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
        .decision-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 5px; color: #fff; }}
        .portfolio-section {{
            background: rgba(124, 58, 237, 0.1);
            border-radius: 16px;
            padding: 30px;
            border: 1px solid rgba(124, 58, 237, 0.3);
        }}
        .portfolio-title {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #a78bfa;
        }}
        .portfolio-table {{ width: 100%; border-collapse: collapse; }}
        .portfolio-table th {{
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid rgba(124, 58, 237, 0.5);
            color: #a78bfa;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }}
        .portfolio-table td {{
            padding: 15px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .portfolio-strategy {{
            margin-top: 25px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            font-style: italic;
            color: #bbb;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI对冲基金分析</h1>
        <div class="timestamp">生成时间: {timestamp}</div>
"""

    # Add each ticker section
    for ticker, ticker_decision in decisions.items():
        decision = ticker_decision.get("action", "HOLD").upper()
        html_content += f"""
        <div class="ticker-section">
            <div class="ticker-header">
                <div class="ticker-name">{ticker}</div>
                <div class="decision-tag decision-{decision}">{cn(decision)}</div>
            </div>
            
            <div class="analyst-grid">
"""

        # Add each analyst's signal for this ticker
        for agent, signals in analyst_signals.items():
            if ticker not in signals:
                continue
            if agent == "risk_management_agent":
                continue

            signal_data = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            
            # Handle different signal formats - could be string or dict
            if isinstance(signal_data, str):
                # If it's just a string signal, use defaults
                signal = signal_data.upper()
                confidence = 50
                reasoning_html = ""
            elif isinstance(signal_data, dict):
                signal = signal_data.get("signal", "NEUTRAL").upper()
                confidence = signal_data.get("confidence", 0)
                reasoning = signal_data.get("reasoning", "")
                
                # Check if this is technical or sentiment analysis (has nested strategies)
                if agent == "technical_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_technical_insights(reasoning)
                elif agent == "sentiment_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_sentiment_insights(reasoning)
                else:
                    reasoning_html = str(reasoning) if reasoning else ""
            else:
                signal = "NEUTRAL"
                confidence = 0
                reasoning_html = ""
            
            # Ensure confidence is a number, not a string
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0

            # Make technical and sentiment analyst cards wider to show all strategies
            card_style = ""
            if agent == "technical_analyst_agent" or agent == "sentiment_analyst_agent":
                card_style = "grid-column: span 2;"
            
            html_content += f"""
                <div class="analyst-card" style="{card_style}">
                    <div class="analyst-name">{agent_name}</div>
                    <div class="analyst-signal">
                        <span class="signal-{signal}">{cn(signal)}</span>
                        <span>{int(confidence)}%</span>
                    </div>
                    <div class="confidence-bar confidence-{signal}">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    <div class="analyst-reasoning">{reasoning_html}</div>
                </div>
"""

        # Decision panel
        html_content += f"""
            </div>
            
            <div class="decision-panel">
                <div class="decision-item">
                    <div class="decision-label">操作</div>
                    <div class="decision-value">{cn(decision)}</div>
                </div>
                <div class="decision-item">
                    <div class="decision-label">数量</div>
                    <div class="decision-value">{ticker_decision.get('quantity', 0)}</div>
                </div>
                <div class="decision-item">
                    <div class="decision-label">置信度</div>
                    <div class="decision-value">{ticker_decision.get('confidence', 0):.1f}%</div>
                </div>
            </div>
        </div>
"""

    # Portfolio summary
    html_content += f"""
        <div class="portfolio-section">
            <div class="portfolio-title">📊 投资组合摘要</div>
            <table class="portfolio-table">
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>操作</th>
                        <th>数量</th>
                        <th>置信度</th>
                        <th>看涨</th>
                        <th>看跌</th>
                        <th>中性</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Count bullish/bearish/neutral signals per ticker
    for ticker, ticker_decision in decisions.items():
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for agent, signals in analyst_signals.items():
            if ticker not in signals:
                continue
            if agent == "risk_management_agent":
                continue
                
            signal_data = signals[ticker]
            if isinstance(signal_data, dict):
                signal = signal_data.get("signal", "NEUTRAL").upper()
            elif isinstance(signal_data, str):
                signal = signal_data.upper()
            else:
                signal = "NEUTRAL"
                
            if "BULLISH" in signal or "LONG" in signal:
                bullish_count += 1
            elif "BEARISH" in signal or "SHORT" in signal:
                bearish_count += 1
            else:
                neutral_count += 1

        action = ticker_decision.get("action", "HOLD").upper()
        html_content += f"""
                    <tr>
                        <td style="font-weight: 700; color: #fff;">{ticker}</td>
                        <td class="signal-{action}">{cn(action)}</td>
                        <td>{ticker_decision.get('quantity', 0)}</td>
                        <td>{ticker_decision.get('confidence', 0):.1f}%</td>
                        <td style="color: #10b981;">{bullish_count}</td>
                        <td style="color: #ef4444;">{bearish_count}</td>
                        <td style="color: #6366f1;">{neutral_count}</td>
                    </tr>
"""

    # Find the strategy text from decisions
    strategy_text = ""
    if decisions:
        for ticker, ticker_decision in decisions.items():
            if ticker_decision.get('reasoning'):
                strategy_text = ticker_decision.get('reasoning', '')
                break

    html_content += f"""
                </tbody>
            </table>
            <div class="portfolio-strategy">
                💡 {strategy_text}
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html_content

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "ark-code-latest",
    model_provider: str = "OpenAI",
    language: str = "en",
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "language": language,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        language=inputs.language,
    )
    # Do NOT print trading output to console - too verbose
    # print_trading_output(result)
    
    # Save result to file - ALWAYS use project's output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"hedge_fund_analysis_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✅ JSON analysis saved to: {json_path}")
    
    # Generate beautiful HTML report (ONLY HTML output)
    html_report = generate_html_report(
        result.get("analyst_signals", {}),
        result.get("decisions", {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    html_path = output_dir / f"hedge_fund_analysis_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"✅ HTML report saved to: {html_path}")
    print(f"   → Open in your browser for a beautiful, readable view!")
