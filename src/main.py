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


def format_sentiment_insights(reasoning: dict, language: str = "zh") -> str:
    """Convert sentiment analyst's nested metrics into human-readable insights with language support."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    if language == "en":
        category_names = {
            "insider_trading": "👤 Insider Trading",
            "news_sentiment": "📰 News Sentiment",
            "social_media": "💬 Social Media",
            "analyst_ratings": "📋 Analyst Ratings"
        }
        signal_map = {"BULLISH": "Bullish", "BEARISH": "Bearish", "NEUTRAL": "Neutral"}
    else:  # zh
        category_names = {
            "insider_trading": "👤 内幕交易",
            "news_sentiment": "📰 新闻情绪",
            "social_media": "💬 社交媒体",
            "analyst_ratings": "📋 分析师评级"
        }
        signal_map = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}
    
    insights = []
    
    for category_key, category_data in reasoning.items():
        if category_key == "combined_analysis":
            continue  # Skip the combined section for now
            
        if category_key not in category_names:
            continue
            
        name = category_names[category_key]
        signal = category_data.get("signal", "neutral").upper()
        signal_cn = signal_map.get(signal, signal)
        confidence = category_data.get("confidence", 0)
        metrics = category_data.get("metrics", {})
        
        # Generate metric explanations based on language
        metric_notes = []
        if category_key == "insider_trading":
            total = metrics.get("total_trades", 0)
            bullish = metrics.get("bullish_trades", 0)
            bearish = metrics.get("bearish_trades", 0)
            if total > 0:
                if language == "en":
                    metric_notes.append(f"{total} trades")
                    if bullish > 0:
                        metric_notes.append(f"{bullish} buys")
                    if bearish > 0:
                        metric_notes.append(f"{bearish} sells")
                else:  # zh
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
                if language == "en":
                    metric_notes.append(f"{total} articles")
                    metric_notes.append(f"{bullish} positive")
                    metric_notes.append(f"{bearish} negative")
                else:  # zh
                    metric_notes.append(f"共{total}篇")
                    metric_notes.append(f"正面{bullish}")
                    metric_notes.append(f"负面{bearish}")
            else:
                metric_notes.append("No news data" if language == "en" else "无新闻数据")
        
        # Signal color class (lowercase for CSS consistency)
        signal_class = f"signal-{signal.lower()}"

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


def format_growth_insights(reasoning: dict, language: str = "zh") -> str:
    """Convert growth analyst's metrics into human-readable insights with language support."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    if language == "en":
        category_names = {
            "historical_growth": "📊 Historical Growth",
            "growth_valuation": "💰 Growth Valuation",
            "margin_expansion": "📈 Margin Trends",
            "insider_conviction": "👤 Insider Conviction",
            "financial_health": "🏦 Financial Health"
        }
        signal_map = {"BULLISH": "Bullish", "BEARISH": "Bearish", "NEUTRAL": "Neutral"}
    else:  # zh
        category_names = {
            "historical_growth": "📊 历史成长",
            "growth_valuation": "💰 成长估值",
            "margin_expansion": "📈 利润率趋势",
            "insider_conviction": "👤 内部人士信心",
            "financial_health": "🏦 财务健康"
        }
        signal_map = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}
    
    insights = []
    
    for category_key, category_data in reasoning.items():
        if category_key not in category_names:
            continue
        if not isinstance(category_data, dict):
            continue
            
        name = category_names[category_key]
        score = category_data.get("score") or 0
        
        # Determine signal based on score
        if score >= 0.7:
            signal = "BULLISH"
        elif score <= 0.3:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        signal_cn = signal_map.get(signal, signal)
        
        # Build details string based on language
        details_parts = []
        if category_key == "historical_growth":
            rev_trend = category_data.get("revenue_trend") or 0
            eps_trend = category_data.get("eps_trend") or 0
            if language == "en":
                if rev_trend > 0.1:
                    details_parts.append(f"Rev Trend +{rev_trend:.1%}")
                elif rev_trend < -0.1:
                    details_parts.append(f"Rev Trend {rev_trend:.1%}")
                if eps_trend > 0.1:
                    details_parts.append(f"EPS Trend +{eps_trend:.1%}")
                elif eps_trend < -0.1:
                    details_parts.append(f"EPS Trend {eps_trend:.1%}")
            else:  # zh
                if rev_trend > 0.1:
                    details_parts.append(f"营收趋势 +{rev_trend:.1%}")
                elif rev_trend < -0.1:
                    details_parts.append(f"营收趋势 {rev_trend:.1%}")
                if eps_trend > 0.1:
                    details_parts.append(f"EPS趋势 +{eps_trend:.1%}")
                elif eps_trend < -0.1:
                    details_parts.append(f"EPS趋势 {eps_trend:.1%}")
        
        elif category_key == "growth_valuation":
            peg = category_data.get("peg_ratio") or 0
            ps = category_data.get("price_to_sales_ratio") or 0
            details_parts.append(f"PEG: {peg:.2f}")
            details_parts.append(f"P/S: {ps:.2f}")
        
        elif category_key == "margin_expansion":
            gm = category_data.get("gross_margin") or 0
            om = category_data.get("operating_margin") or 0
            if gm or om:
                if language == "en":
                    if gm:
                        details_parts.append(f"Gross Margin: {gm:.1%}")
                    if om:
                        details_parts.append(f"Operating Margin: {om:.1%}")
                else:  # zh
                    if gm:
                        details_parts.append(f"毛利率: {gm:.1%}")
                    if om:
                        details_parts.append(f"经营利润率: {om:.1%}")
        
        elif category_key == "insider_conviction":
            net_flow = category_data.get("net_flow_ratio") or 0
            buys = category_data.get("buys") or 0
            sells = category_data.get("sells") or 0
            if abs(net_flow) > 0.5:
                if language == "en":
                    direction = "Net Buying" if net_flow > 0 else "Net Selling"
                    details_parts.append(f"Insider {direction}")
                else:  # zh
                    direction = "净买入" if net_flow > 0 else "净卖出"
                    details_parts.append(f"内部人士{direction}")
        
        elif category_key == "financial_health":
            dte = category_data.get("debt_to_equity") or 0
            cr = category_data.get("current_ratio") or 0
            if dte:
                details_parts.append(f"D/E: {dte:.2f}")
            if cr:
                if language == "en":
                    details_parts.append(f"Current Ratio: {cr:.2f}")
                else:  # zh
                    details_parts.append(f"流动比率: {cr:.2f}")
        
        if language == "en":
            details = " · ".join(details_parts) if details_parts else f"Score: {score:.2f}"
        else:  # zh
            details = " · ".join(details_parts) if details_parts else f"评分: {score:.2f}"
        
        signal_class = f"signal-{signal.lower()}"
        insights.append(f"""
            <div style="margin-bottom: 8px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-weight: 600; font-size: 0.8rem;">{name}</span>
                    <span style="font-weight: 700; font-size: 0.75rem;" class="{signal_class}">{signal_cn}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">{details}</div>
            </div>
        """)
    
    return "\n".join(insights)


def format_fundamentals_insights(reasoning: dict, language: str = "zh") -> str:
    """Convert fundamentals analyst's metrics into human-readable insights with language support."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    if language == "en":
        category_names = {
            "profitability_signal": "💰 Profitability",
            "growth_signal": "📈 Growth Potential",
            "financial_health_signal": "🏦 Financial Health",
            "price_ratios_signal": "📊 Valuation Ratios"
        }
        signal_map = {"BULLISH": "Bullish", "BEARISH": "Bearish", "NEUTRAL": "Neutral"}
    else:  # zh
        category_names = {
            "profitability_signal": "💰 盈利能力",
            "growth_signal": "📈 成长能力",
            "financial_health_signal": "🏦 财务健康",
            "price_ratios_signal": "📊 估值比率"
        }
        signal_map = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}
    
    insights = []
    
    for category_key, category_data in reasoning.items():
        if category_key not in category_names:
            continue
            
        name = category_names[category_key]
        signal = category_data.get("signal", "neutral").upper()
        signal_cn = signal_map.get(signal, signal)
        details = category_data.get("details", "")
        
        signal_class = f"signal-{signal.lower()}"
        insights.append(f"""
            <div style="margin-bottom: 8px; padding: 6px 8px; background: rgba(255,255,255,0.03); border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-weight: 600; font-size: 0.8rem;">{name}</span>
                    <span style="font-weight: 700; font-size: 0.75rem;" class="{signal_class}">{signal_cn}</span>
                </div>
                <div style="font-size: 0.7rem; color: #888;">{details}</div>
            </div>
        """)
    
    return "\n".join(insights)


def format_technical_insights(reasoning: dict, language: str = "zh") -> str:
    """Convert technical analyst's nested metrics into human-readable insights with language support."""
    if not reasoning or not isinstance(reasoning, dict):
        return ""
    
    if language == "en":
        strategy_names = {
            "trend_following": "📈 Trend Following",
            "mean_reversion": "🔄 Mean Reversion",
            "momentum": "⚡ Momentum",
            "volatility": "📊 Volatility",
            "statistical_arbitrage": "📐 Statistical Arbitrage"
        }
        signal_map = {"BULLISH": "Bullish", "BEARISH": "Bearish", "NEUTRAL": "Neutral"}
    else:  # zh
        strategy_names = {
            "trend_following": "📈 趋势跟踪",
            "mean_reversion": "🔄 均值回归",
            "momentum": "⚡ 动量分析",
            "volatility": "📊 波动率",
            "statistical_arbitrage": "📐 统计套利"
        }
        signal_map = {"BULLISH": "看涨", "BEARISH": "看跌", "NEUTRAL": "中性"}
    
    insights = []
    
    for strategy_key, strategy_data in reasoning.items():
        if strategy_key not in strategy_names:
            continue
            
        name = strategy_names[strategy_key]
        signal = strategy_data.get("signal", "neutral").upper()
        signal_cn = signal_map.get(signal, signal)
        confidence = strategy_data.get("confidence", 0)
        metrics = strategy_data.get("metrics", {})
        
        # Generate metric explanations based on language
        metric_notes = []
        if strategy_key == "trend_following":
            adx = metrics.get("adx") or 0
            if language == "en":
                if adx > 25:
                    metric_notes.append(f"ADX {adx:.1f} (Strong Trend)")
                elif adx > 20:
                    metric_notes.append(f"ADX {adx:.1f} (Trending)")
                else:
                    metric_notes.append(f"ADX {adx:.1f} (Ranging)")
            else:  # zh
                if adx > 25:
                    metric_notes.append(f"ADX {adx:.1f} (趋势强)")
                elif adx > 20:
                    metric_notes.append(f"ADX {adx:.1f} (有趋势)")
                else:
                    metric_notes.append(f"ADX {adx:.1f} (震荡)")
        
        elif strategy_key == "mean_reversion":
            rsi = metrics.get("rsi_14") or 50
            z_score = metrics.get("z_score") or 0
            if language == "en":
                if rsi > 70:
                    metric_notes.append(f"RSI {rsi:.1f} (Overbought)")
                elif rsi < 30:
                    metric_notes.append(f"RSI {rsi:.1f} (Oversold)")
                else:
                    metric_notes.append(f"RSI {rsi:.1f} (Neutral)")
                if abs(z_score) > 1.5:
                    metric_notes.append(f"Z {z_score:.2f}σ (Deviated)")
            else:  # zh
                if rsi > 70:
                    metric_notes.append(f"RSI {rsi:.1f} (超买)")
                elif rsi < 30:
                    metric_notes.append(f"RSI {rsi:.1f} (超卖)")
                else:
                    metric_notes.append(f"RSI {rsi:.1f} (中性)")
                if abs(z_score) > 1.5:
                    metric_notes.append(f"Z {z_score:.2f}σ (偏离均值)")
        
        elif strategy_key == "momentum":
            mom_1m = metrics.get("momentum_1m") or 0
            vol_mom = metrics.get("volume_momentum") or 0
            if language == "en":
                if mom_1m > 0.1:
                    metric_notes.append(f"1M Mom +{mom_1m:.1%}")
                elif mom_1m < -0.1:
                    metric_notes.append(f"1M Mom {mom_1m:.1%}")
                if vol_mom > 1.2:
                    metric_notes.append(f"High Vol x{vol_mom:.1f}")
            else:  # zh
                if mom_1m > 0.1:
                    metric_notes.append(f"1月动量 +{mom_1m:.1%}")
                elif mom_1m < -0.1:
                    metric_notes.append(f"1月动量 {mom_1m:.1%}")
                if vol_mom > 1.2:
                    metric_notes.append(f"放量 x{vol_mom:.1f}")
        
        elif strategy_key == "volatility":
            hv = metrics.get("historical_volatility") or 0
            if language == "en":
                if hv > 0.5:
                    metric_notes.append(f"Volatility {hv:.1%} (High)")
                elif hv < 0.2:
                    metric_notes.append(f"Volatility {hv:.1%} (Low)")
            else:  # zh
                if hv > 0.5:
                    metric_notes.append(f"波动率 {hv:.1%} (高波动)")
                elif hv < 0.2:
                    metric_notes.append(f"波动率 {hv:.1%} (低波动)")
        
        elif strategy_key == "statistical_arbitrage":
            hurst = metrics.get("hurst_exponent") or 0.5
            skew = metrics.get("skewness") or 0
            kurt = metrics.get("kurtosis") or 3
            
            if language == "en":
                if hurst < 0.4:
                    metric_notes.append(f"Hurst {hurst:.2f} (Mean Reverting)")
                elif hurst > 0.6:
                    metric_notes.append(f"Hurst {hurst:.2f} (Trending)")
                else:
                    metric_notes.append(f"Hurst {hurst:.2f} (Random Walk)")
            else:  # zh
                if hurst < 0.4:
                    metric_notes.append(f"Hurst {hurst:.2f} (均值回归)")
                elif hurst > 0.6:
                    metric_notes.append(f"Hurst {hurst:.2f} (强趋势)")
                else:
                    metric_notes.append(f"Hurst {hurst:.2f} (随机游走)")
            
            # Skewness
            if abs(skew) > 1:
                metric_notes.append(f"偏度 {skew:.2f}")
            
            # Kurtosis (fat tails - interesting for stat arb)
            if kurt > 4:
                metric_notes.append(f"峰度 {kurt:.1f} (肥尾)")
        
        # Signal color class (lowercase for CSS consistency)
        signal_class = f"signal-{signal.lower()}"

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


def generate_html_report(analyst_signals, decisions, timestamp, language="zh"):
    """Generate a beautiful HTML report with language support (en/zh)."""
    # Chinese translation helper for signals
    def cn(text):
        if language == "en":
            return text
        translations = {
            'BULLISH': '看涨', 'BEARISH': '看跌', 'NEUTRAL': '中性',
            'LONG': '做多', 'SHORT': '做空', 'HOLD': '持有'
        }
        return translations.get(text, text)
    
    # Analyst name translations
    def translate_analyst(agent_key):
        if language == "en":
            return agent_key.replace("_agent", "").replace("_", " ").title()
        
        analyst_names = {
            # Value investors
            "warren_buffett_agent": "沃伦·巴菲特",
            "charlie_munger_agent": "查理·芒格",
            "ben_graham_agent": "本杰明·格雷厄姆",
            "mohnish_pabrai_agent": "莫尼什·帕伯莱",
            
            # Growth investors
            "cathie_wood_agent": "凯西·伍德",
            "phil_fisher_agent": "菲利普·费雪",
            "peter_lynch_agent": "彼得·林奇",
            
            # Contrarian / Macro
            "michael_burry_agent": "迈克尔·伯里",
            "nassim_taleb_agent": "纳西姆·塔勒布",
            "stanley_druckenmiller_agent": "斯坦利·德鲁肯米勒",
            "bill_ackman_agent": "比尔·阿克曼",
            
            # Other
            "rakesh_jhunjhunwala_agent": "拉克什·金君瓦拉",
            "aswath_damodaran_agent": "阿斯沃斯·达摩达兰",
            
            # Data analysts
            "technical_analyst_agent": "技术分析",
            "fundamentals_analyst_agent": "基本面分析",
            "valuation_agent": "估值分析",
            "sentiment_analyst_agent": "情绪分析",
            "news_sentiment_agent": "新闻情绪",
            "growth_analyst_agent": "成长性分析",
        }
        return analyst_names.get(agent_key, agent_key.replace("_agent", "").replace("_", " ").title())
    
    # Intro text based on language
    if language == "en":
        intro_title = "📊 About This Report"
        intro_text = """This report is automatically generated by the AI Hedge Fund multi-agent system. The system brings together the analytical frameworks of the world's top investment masters, including Warren Buffett (Value Investing), Charlie Munger (Multidisciplinary Thinking), Michael Burry (Contrarian Investing), Cathie Wood (Growth Investing), and 10 other investment legends.<br><br>
The system generates comprehensive trading decisions and position recommendations for each stock through multiple dimensions: technical analysis, fundamental analysis, valuation calculation, sentiment analysis, and risk management."""
        disclaimer_text = """⚠️ <strong>DISCLAIMER</strong>: This report is for educational and research purposes only and does not constitute any investment advice. Past performance does not indicate future results. Please invest at your own risk."""
        page_title = "AI Investment Insights Report"
        main_title = "🤖 AI Investment Insights"
        time_label = "Generated: "
    else:  # zh
        intro_title = "📊 关于本报告"
        intro_text = """本报告由AI对冲基金多智能体系统自动生成。系统汇集了全球顶尖投资大师的分析框架，包括沃伦·巴菲特（价值投资）、查理·芒格（多学科思维）、迈克尔·伯里（逆向投资）、凯西·伍德（成长投资）等13位投资大师的交易策略。<br><br>
系统通过技术分析、基本面分析、估值计算、情绪分析、风险管理等多个维度，为每支股票生成综合交易决策和持仓建议。"""
        disclaimer_text = """⚠️ <strong>免责声明</strong>：本报告仅供教育和研究目的，不构成任何投资建议。过去表现不代表未来结果，请自行承担投资风险。"""
        page_title = "AI 投资洞察报告"
        main_title = "🤖 AI 投资洞察报告"
        time_label = "生成时间: "
    
    html_content = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title} - {timestamp}</title>
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
        .intro-section {{
            background: rgba(124, 58, 237, 0.1);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(124, 58, 237, 0.3);
            text-align: center;
        }}
        .intro-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #a78bfa;
            margin-bottom: 15px;
        }}
        .intro-text {{
            font-size: 0.9rem;
            line-height: 1.8;
            color: #a0a0a0;
        }}
        .disclaimer {{
            margin-top: 15px;
            padding: 12px;
            background: rgba(239, 68, 68, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            font-size: 0.8rem;
            color: #fca5a5;
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
        .group-header {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #a78bfa;
            margin: 30px 0 18px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(167, 139, 250, 0.3);
        }}
        /* Special layout for technical analysis - 2-column grid inside to avoid being too long */
        .technical-insights {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
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
        <h1>{main_title}</h1>
        <div class="intro-section">
            <div class="intro-title">{intro_title}</div>
            <div class="intro-text">
                {intro_text}
            </div>
            <div class="disclaimer">
                {disclaimer_text}
            </div>
        </div>
        <div class="timestamp">{time_label}{timestamp}</div>
"""

    # Add each ticker section
    for ticker, ticker_decision in decisions.items():
        decision = ticker_decision.get("action", "HOLD").upper()
        
        # Define analyst order and grouping (logical flow: Value → Growth → Contrarian → Data Analysis)
        analyst_order = [
            # Value Investors
            "warren_buffett_agent", "charlie_munger_agent", "ben_graham_agent", "mohnish_pabrai_agent",
            # Growth Investors
            "cathie_wood_agent", "phil_fisher_agent", "peter_lynch_agent", "growth_analyst_agent",
            # Contrarian / Macro Investors
            "michael_burry_agent", "nassim_taleb_agent", "stanley_druckenmiller_agent", "bill_ackman_agent",
            "rakesh_jhunjhunwala_agent", "aswath_damodaran_agent",
            # Data Analysis (last, most detailed)
            "fundamentals_analyst_agent", "valuation_agent", "sentiment_analyst_agent", "news_sentiment_agent",
            "technical_analyst_agent",
        ]
        
        # Group titles for section headers
        group_titles = {
            "value": {"zh": "🏛️ 价值投资大师", "en": "🏛️ Value Investing Masters"},
            "growth": {"zh": "📈 成长投资大师", "en": "📈 Growth Investing Masters"},
            "contrarian": {"zh": "⚡ 逆向与宏观投资", "en": "⚡ Contrarian & Macro"},
            "data": {"zh": "📊 数据分析", "en": "📊 Data Analysis"},
        }
        
        # Map agents to groups
        agent_groups = {
            "warren_buffett_agent": "value",
            "charlie_munger_agent": "value",
            "ben_graham_agent": "value",
            "mohnish_pabrai_agent": "value",
            "cathie_wood_agent": "growth",
            "phil_fisher_agent": "growth",
            "peter_lynch_agent": "growth",
            "growth_analyst_agent": "growth",
            "michael_burry_agent": "contrarian",
            "nassim_taleb_agent": "contrarian",
            "stanley_druckenmiller_agent": "contrarian",
            "bill_ackman_agent": "contrarian",
            "rakesh_jhunjhunwala_agent": "contrarian",
            "aswath_damodaran_agent": "contrarian",
            "fundamentals_analyst_agent": "data",
            "valuation_agent": "data",
            "sentiment_analyst_agent": "data",
            "news_sentiment_agent": "data",
            "technical_analyst_agent": "data",
        }
        
        html_content += f"""
        <div class="ticker-section">
            <div class="ticker-header">
                <div class="ticker-name">{ticker}</div>
                <div class="decision-tag decision-{decision}">{cn(decision)}</div>
            </div>
"""

        # Add each analyst's signal for this ticker, by groups
        current_group = None
        group_content = []
        
        for agent in analyst_order:
            if agent not in analyst_signals:
                continue
            signals = analyst_signals[agent]
            if ticker not in signals:
                continue
            if agent == "risk_management_agent":
                continue
                
            # Check if we need to start a new group
            group = agent_groups.get(agent, "data")
            if group != current_group:
                # Flush previous group content
                if group_content:
                    html_content += f"""
                    <div class="analyst-grid">
                        {''.join(group_content)}
                    </div>
                    """
                    group_content = []
                
                # Add group header
                current_group = group
                html_content += f"""
                    <div class="group-header">{group_titles[group][language]}</div>
                    """
            
            signal_data = signals[ticker]
            agent_name = translate_analyst(agent)
            
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
                
                # Check if this is technical, sentiment, news_sentiment, fundamentals, or growth analysis
                if agent == "technical_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = f'<div class="technical-insights">{format_technical_insights(reasoning, language)}</div>'
                elif agent == "sentiment_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_sentiment_insights(reasoning, language)
                elif agent == "news_sentiment_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_sentiment_insights(reasoning, language)
                elif agent == "fundamentals_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_fundamentals_insights(reasoning, language)
                elif agent == "growth_analyst_agent" and isinstance(reasoning, dict):
                    reasoning_html = format_growth_insights(reasoning, language)
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

            # Make technical, sentiment, and news_sentiment analyst cards wider to show all strategies
            card_style = ""
            if agent in ["technical_analyst_agent", "sentiment_analyst_agent", "news_sentiment_agent"]:
                card_style = "grid-column: span 2;"
            
            group_content.append(f"""
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
""")
        
        # Flush remaining group content
        if group_content:
            html_content += f"""
            <div class="analyst-grid">
                {''.join(group_content)}
            </div>
            """
        
        # Decision panel
        html_content += f"""
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
                    <div class="decision-value">{(ticker_decision.get('confidence') or 0):.1f}%</div>
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
                        <td>{(ticker_decision.get('confidence') or 0):.1f}%</td>
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


def update_index_html(output_dir: Path, language: str = "zh"):
    """Update index.html to list all historical HTML reports."""
    index_path = output_dir / "index.html"
    
    # Get all HTML files (excluding index.html itself)
    html_files = []
    for f in output_dir.glob("*.html"):
        if f.name != "index.html":
            html_files.append(f)
    
    # Sort by modification time (newest first)
    html_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Build file list
    file_items = []
    for f in html_files:
        # Parse filename: tickers_timestamp.html or hedge_fund_analysis_timestamp.html
        name_without_ext = f.stem
        file_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract tickers from filename
        if name_without_ext.startswith("hedge_fund_analysis_"):
            tickers_display = name_without_ext.replace("hedge_fund_analysis_", "")
        else:
            # Format: tickers_timestamp (e.g., AAPL,MSFT_20260503_174903)
            parts = name_without_ext.split("_")
            if len(parts) >= 3:
                tickers_display = parts[0]
            else:
                tickers_display = name_without_ext
        
        file_items.append(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; font-size: 1rem; margin-bottom: 4px;">📊 {tickers_display}</div>
                        <div style="font-size: 0.75rem; color: #888;">{file_time}</div>
                    </div>
                    <a href="{f.name}" style="background: linear-gradient(135deg, #7c3aed, #6366f1); color: white; padding: 8px 16px; border-radius: 8px; text-decoration: none; font-size: 0.85rem; font-weight: 600;">
                        {"查看报告" if language == "zh" else "View Report"} →
                    </a>
                </div>
            </div>
        """)
    
    title = "AI 投资洞察报告 - 索引" if language == "zh" else "AI Hedge Fund Reports - Index"
    header = "📊 AI 投资洞察报告" if language == "zh" else "📊 AI Hedge Fund Reports"
    subtitle = "所有历史报告索引" if language == "zh" else "Index of all historical reports"
    no_files = "暂无报告文件" if language == "zh" else "No report files yet"
    
    files_html = "\n".join(file_items) if file_items else f'<div style="text-align: center; padding: 40px; color: #888;">{no_files}</div>'
    
    index_html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        .header p {{
            color: #888;
            font-size: 0.9rem;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px 24px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #a78bfa;
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: #888;
            margin-top: 4px;
        }}
        .file-list {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{header}</h1>
            <p>{subtitle}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(file_items)}</div>
                <div class="stat-label">{"报告总数" if language == "zh" else "Total Reports"}</div>
            </div>
        </div>
        
        <div class="file-list">
            {files_html}
        </div>
    </div>
</body>
</html>"""
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)


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
    tickers_str = ",".join(tickers)
    json_path = output_dir / f"{tickers_str}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✅ JSON analysis saved to: {json_path}")
    
    # Generate beautiful HTML report (ONLY HTML output)
    html_report = generate_html_report(
        result.get("analyst_signals", {}),
        result.get("decisions", {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        language=inputs.language
    )
    html_path = output_dir / f"{tickers_str}_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"✅ HTML report saved to: {html_path}")
    
    # Update index.html with all historical reports
    update_index_html(output_dir, language=inputs.language)
    print(f"✅ Index updated: {output_dir / 'index.html'}")
    print(f"   → Open in your browser for a beautiful, readable view!")
