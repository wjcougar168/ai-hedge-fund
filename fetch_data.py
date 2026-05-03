#!/usr/bin/env python3
"""
Data pre-fetcher for the AI Hedge Fund.
Fetch all required data for tickers one at a time with delays to avoid rate limits.
Data is saved to persistent cache at ~/.cache/ai-hedge-fund/
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv()

from src.tools.api import (
    get_prices,
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
)
from src.data.cache import get_cache

cache = get_cache()

# Default date range (3 months)
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

def get_api_key() -> str:
    """Get API key from environment."""
    return os.environ.get("FINANCIAL_DATASETS_API_KEY", "")

def fetch_ticker_data(ticker: str, start_date: str, end_date: str) -> bool:
    """Fetch all data types for a single ticker and cache them."""
    api_key = get_api_key()
    ticker = ticker.upper()
    
    print(f"\n{'='*60}")
    print(f"Fetching data for: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    success_count = 0
    total_types = 5
    
    # 1. Prices
    print(f"\n[1/5] Fetching prices...")
    try:
        if cache.has_sufficient_prices(ticker, start_date, end_date):
            cached = cache.get_prices(ticker, start_date, end_date)
            print(f"  ✅ Using {len(cached)} cached price points")
            success_count += 1
        else:
            prices = get_prices(ticker, start_date, end_date, api_key)
            if prices:
                print(f"  ✅ Fetched {len(prices)} price points")
                success_count += 1
            else:
                print(f"  ⚠️  No price data returned")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(2)
    
    # 2. Financial Metrics
    print(f"\n[2/5] Fetching financial metrics...")
    try:
        if cache.has_sufficient_financial_metrics(ticker):
            cached = cache.get_financial_metrics(ticker)
            print(f"  ✅ Using {len(cached)} cached financial metric periods")
            success_count += 1
        else:
            metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
            if metrics:
                print(f"  ✅ Fetched {len(metrics)} periods of financial metrics")
                success_count += 1
            else:
                print(f"  ⚠️  No financial metrics returned")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(2)
    
    # 3. Line Items (financial statement data)
    print(f"\n[3/5] Fetching financial line items...")
    try:
        if cache.has_sufficient_line_items(ticker):
            cached = cache.get_line_items(ticker)
            print(f"  ✅ Using {len(cached)} cached line item periods")
            success_count += 1
        else:
            line_items = search_line_items(
                ticker,
                ["revenue", "net_income", "ebitda", "gross_profit", "operating_income",
                 "total_assets", "total_liabilities", "shareholders_equity"],
                end_date,
                period="ttm",
                limit=10
            )
            if line_items:
                print(f"  ✅ Fetched {len(line_items)} periods of line items")
                success_count += 1
            else:
                print(f"  ⚠️  No line items returned")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(2)
    
    # 4. Market Cap
    print(f"\n[4/5] Fetching market cap...")
    try:
        mcap = get_market_cap(ticker, end_date)
        if mcap:
            print(f"  ✅ Market cap: ${mcap / 1e9:.2f}B")
            success_count += 1
        else:
            print(f"  ⚠️  No market cap returned")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(2)
    
    # 5. News (fetch enough for ALL analysts - Burry needs 250!)
    print(f"\n[5/5] Fetching company news...")
    try:
        if cache.has_sufficient_company_news(ticker, start_date, end_date, min_count=250):
            cached = cache.get_company_news(ticker, start_date, end_date)
            print(f"  ✅ Using {len(cached)} cached news articles")
            success_count += 1
        else:
            news = get_company_news(ticker, end_date, start_date, limit=250)
            if news:
                print(f"  ✅ Fetched {len(news)} news articles")
                success_count += 1
            else:
                print(f"  ⚠️  No news returned")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Summary
    print(f"\n--- {ticker} Summary ---")
    print(f"Success: {success_count}/{total_types} data types")
    
    if success_count >= 3:
        print(f"✅ {ticker} is ready for analysis!")
        return True
    else:
        print(f"⚠️  {ticker} may have incomplete data")
        return False

def show_cache_status(tickers: list) -> None:
    """Show what's already cached with detailed info."""
    from datetime import datetime
    import json
    
    print(f"\n{'='*70}")
    print(f"Current Cache Status (at: ~/.cache/ai-hedge-fund/)")
    print(f"{'='*70}")
    
    cache_dir = Path.home() / ".cache" / "ai-hedge-fund"
    
    for ticker in sorted(tickers):
        ticker = ticker.upper()
        details = []
        
        # Check prices with date range
        prices_file = cache_dir / f"{ticker}_prices.json"
        if prices_file.exists():
            try:
                with open(prices_file) as f:
                    prices = json.load(f)
                if prices:
                    dates = [p.get("time", "").split("T")[0] for p in prices if p.get("time")]
                    if dates:
                        min_date = min(dates)
                        max_date = max(dates)
                        details.append(f"prices({len(prices)}pt, {min_date}→{max_date})")
                    else:
                        details.append(f"prices({len(prices)}pt)")
            except:
                details.append("prices(?)")
        
        # Check financial metrics
        metrics_file = cache_dir / f"{ticker}_financial_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                details.append(f"metrics({len(metrics)}qtr)")
            except:
                details.append("metrics")
        
        # Check line items
        line_file = cache_dir / f"{ticker}_line_items.json"
        if line_file.exists():
            try:
                with open(line_file) as f:
                    lines = json.load(f)
                details.append(f"line_items({len(lines)}qtr)")
            except:
                details.append("line_items")
        
        # Check insider trades with freshness
        insider_file = cache_dir / f"{ticker}_insider_trades.json"
        if insider_file.exists():
            try:
                mtime = datetime.fromtimestamp(insider_file.stat().st_mtime)
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                freshness = f"{age_hours:.1f}h old"
                with open(insider_file) as f:
                    insiders = json.load(f)
                details.append(f"insiders({len(insiders)}, {freshness})")
            except:
                details.append("insiders")
        
        # Check news with date range and freshness
        news_file = cache_dir / f"{ticker}_company_news.json"
        if news_file.exists():
            try:
                mtime = datetime.fromtimestamp(news_file.stat().st_mtime)
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                freshness = f"{age_hours:.1f}h old"
                with open(news_file) as f:
                    news = json.load(f)
                if news:
                    dates = [n.get("date", "").split("T")[0] for n in news if n.get("date")]
                    if dates:
                        min_date = min(dates)
                        max_date = max(dates)
                        details.append(f"news({len(news)}, {min_date}→{max_date}, {freshness})")
                    else:
                        details.append(f"news({len(news)}, {freshness})")
            except:
                details.append("news")
        
        if details:
            print(f"  {ticker}:")
            for d in details:
                print(f"    • {d}")
        else:
            print(f"  {ticker}: no cached data")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-fetch and cache stock data for analysis")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers (e.g., AAPL,MSFT)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD), default: 3 months ago")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--delay", type=int, default=10, help="Delay between tickers in seconds (default: 10)")
    parser.add_argument("--status", action="store_true", help="Just show cache status, don't fetch")
    
    args = parser.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    
    # Just show status if requested
    if args.status:
        show_cache_status(tickers)
        return
    
    start_date = args.start_date or START_DATE
    end_date = args.end_date or END_DATE
    
    print(f"\nData Pre-fetcher for AI Hedge Fund")
    print(f"==================================")
    print(f"Tickers to fetch: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Delay between tickers: {args.delay} seconds")
    print(f"Cache location: ~/.cache/ai-hedge-fund/")
    
    # Show existing cache first
    show_cache_status(tickers)
    
    # Fetch each ticker
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_ticker_data(ticker, start_date, end_date)
        
        # Delay except after last one
        if i < len(tickers) - 1:
            print(f"\nWaiting {args.delay}s before next ticker...")
            time.sleep(args.delay)
    
    # Final summary
    print(f"\n\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    for ticker, success in results.items():
        status = "✅ READY" if success else "⚠️  INCOMPLETE"
        print(f"  {ticker}: {status}")
    
    print(f"\nAll cached data saved to: ~/.cache/ai-hedge-fund/")
    print(f"You can now run the full analysis without rate limit issues!")

if __name__ == "__main__":
    main()
