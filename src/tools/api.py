import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)


def _normalize_ticker_for_yfinance(ticker: str) -> str:
    """Convert ticker format for yfinance compatibility.
    
    yfinance uses hyphen format for share classes (e.g., BRK-B, BRK-A, BF-B)
    while many other APIs and users use dot format (e.g., BRK.B, BRK.A, BF.B).
    This function normalizes to yfinance's expected format.
    """
    # Common multi-class stock tickers that need dot-to-hyphen conversion
    dot_tickers = {
        'BRK.B': 'BRK-B',
        'BRK.A': 'BRK-A',
        'BF.B': 'BF-B',
        'BF.A': 'BF-A',
        'MOG.A': 'MOG-A',
        'MOG.B': 'MOG-B',
        'HEI.A': 'HEI-A',
        'CWEN.A': 'CWEN-A',
        'LILK.A': 'LILK-A',
        'LG.A': 'LG-A',
        'LG.B': 'LG-B',
    }
    upper = ticker.upper()
    if upper in dot_tickers:
        return dot_tickers[upper]
    # General fallback: any ticker with .A or .B suffix
    if upper.endswith('.A') or upper.endswith('.B'):
        return upper.replace('.', '-')
    return ticker


def _normalize_ticker_for_cache(ticker: str) -> str:
    """Normalize ticker for consistent cache keys.
    Uses hyphen format so BRK.B and BRK-B map to the same cache entry.
    """
    return _normalize_ticker_for_yfinance(ticker).upper()


# Set up VPN proxy for yfinance (works in mainland China)
# This is a free workaround for rate limits and access restrictions
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache first, API only if insufficient cached data."""
    # Normalize ticker for cache consistency (BRK.B -> BRK-B)
    cache_ticker = _normalize_ticker_for_cache(ticker)
    yf_ticker_name = _normalize_ticker_for_yfinance(ticker)
    
    # Step 1: Check if we already have ANY cached data (even partial)
    # This prevents rate limit issues - prefer cached data over hitting APIs
    cached_data = _cache.get_prices(cache_ticker, start_date, end_date)
    if cached_data and len(cached_data) >= 20:  # Just need enough for technical analysis
        logger.info(f"Using {len(cached_data)} cached price points for {ticker}")
        return [Price(**price) for price in cached_data]

    # Step 2: Try yfinance FIRST (free, unlimited, works with VPN)
    try:
        import yfinance as yf
        from requests.exceptions import RequestException

        yf_ticker = yf.Ticker(yf_ticker_name)
        hist = yf_ticker.history(start=start_date, end=end_date)
        if not hist.empty:
            prices = []
            for idx, row in hist.iterrows():
                prices.append(Price(
                    time=idx.strftime("%Y-%m-%d"),
                    ticker=ticker,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                ))
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            logger.info(f"Successfully retrieved {len(prices)} price points for {ticker} via yfinance")
            return prices
    except Exception as e:
        logger.info(f"yfinance price fetch failed for {ticker}: {e}, trying fallbacks")

    # Step 3: Try financialdatasets.ai API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"

    # Wrap API call with exception handling to enable fallback gracefully
    try:
        response = _make_api_request(url, headers)
        api_success = response.status_code == 200 and not response.json().get("error")
    except Exception as e:
        logger.info(f"Financial Datasets API request failed for %s: %s", ticker, e)
        api_success = False

    # Fallback if API call fails
    if not api_success:
        # First try Alpha Vantage (proper API, less rate limited)
        alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if alpha_vantage_key:
            try:
                import requests
                from datetime import datetime

                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={alpha_vantage_key}"
                response = requests.get(url, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    # Check for rate limit
                    if "Note" in data and "rate limit" in data["Note"].lower():
                        logger.info(f"Alpha Vantage rate limit reached for {ticker} - using cached data")
                        # Return any cached data we have
                        if cached_data:
                            return [Price(**price) for price in cached_data]
                    else:
                        time_series = data.get("Time Series (Daily)", {})

                        if time_series:
                            prices = []
                            sorted_dates = sorted(time_series.keys(), reverse=True)

                            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

                            for date_str in sorted_dates:
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                                if start_dt <= date_obj <= end_dt:
                                    day_data = time_series[date_str]
                                    prices.append(Price(
                                        time=date_str,
                                        ticker=ticker,
                                        open=float(day_data["1. open"]),
                                        high=float(day_data["2. high"]),
                                        low=float(day_data["3. low"]),
                                        close=float(day_data["4. close"]),
                                        volume=int(day_data["5. volume"]),
                                    ))

                            if prices and len(prices) >= 20:
                                _cache.set_prices(ticker, [p.model_dump() for p in prices])
                                logger.info(f"Successfully retrieved {len(prices)} price points for {ticker} via Alpha Vantage")
                                return prices
            except Exception as e:
                logger.info(f"Alpha Vantage fallback failed for {ticker}: {e}")

        # Before yfinance - LAST RESORT - easily rate limited
        try:
            import yfinance as yf
            import time
            from requests.exceptions import RequestException

            # Only try ONCE - if rate limited, use cache immediately
            try:
                yf_ticker = yf.Ticker(yf_ticker_name)
                hist = yf_ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    prices = []
                    for idx, row in hist.iterrows():
                        prices.append(Price(
                            time=idx.strftime("%Y-%m-%d"),
                            ticker=ticker,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                        ))
                    _cache.set_prices(ticker, [p.model_dump() for p in prices])
                    logger.info(f"Successfully retrieved {len(prices)} price points for {ticker} via yfinance")
                    return prices
            except (yf.exceptions.YFRateLimitError, RequestException):
                # Rate limited - use any cached data we have
                if cached_data:
                    logger.info(f"yfinance rate limited for {ticker} - using {len(cached_data)} cached points")
                    return [Price(**price) for price in cached_data]
                raise  # Re-raise if no cache
        except Exception as e:
            # If we have ANY cached data, use it instead of falling through to demo
            if cached_data and len(cached_data) >= 10:
                logger.info(f"Using {len(cached_data)} existing cached price points for {ticker}")
                return [Price(**price) for price in cached_data]
            
            logger.info(f"yfinance fallback failed for {ticker}: {e}")

        # Alternative fallback: direct Yahoo Finance v10 API (more reliable)
        try:
            import requests
            from datetime import datetime

            # Convert dates to timestamps
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            period1 = int(start_dt.timestamp())
            period2 = int(end_dt.timestamp())

            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                chart = data.get("chart", {}).get("result", [])
                if chart:
                    timestamp = chart[0].get("timestamp", [])
                    indicators = chart[0].get("indicators", {}).get("quote", [{}])[0]
                    opens = indicators.get("open", [])
                    highs = indicators.get("high", [])
                    lows = indicators.get("low", [])
                    closes = indicators.get("close", [])
                    volumes = indicators.get("volume", [])

                    prices = []
                    for i in range(len(timestamp)):
                        if i < len(closes) and closes[i] is not None:
                            prices.append(Price(
                                time=datetime.fromtimestamp(timestamp[i]).strftime("%Y-%m-%d"),
                                ticker=ticker,
                                open=float(opens[i]) if opens[i] else 0.0,
                                high=float(highs[i]) if highs[i] else 0.0,
                                low=float(lows[i]) if lows[i] else 0.0,
                                close=float(closes[i]),
                                volume=int(volumes[i]) if volumes[i] else 0,
                            ))
                    if prices:
                        _cache.set_prices(ticker, [p.model_dump() for p in prices])
                        logger.info(f"Successfully retrieved {len(prices)} price points for {ticker} via Yahoo Finance API")
                        return prices
        except Exception as e:
            logger.info(f"Yahoo Finance direct API fallback failed for {ticker}: {e}")

        # NO DEMO DATA - return empty list when no real data found
        return []

    # Parse response with Pydantic model
    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except Exception as e:
        logger.info("Failed to parse price response for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    # Cache the results for future use (persisted to disk)
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def _supplement_with_finnhub(ticker: str, metric_obj: FinancialMetrics):
    """Supplement missing financial metrics with data from Finnhub (free, 60 calls/min).
    
    Finnhub's stock/metric endpoint provides 100+ financial metrics for free.
    This function fills in any None fields using Finnhub data, preserving
    existing values from primary sources.
    """
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if not finnhub_key:
        return
    
    try:
        import requests
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={finnhub_key}"
        fh_response = requests.get(url, timeout=15)
        if fh_response.status_code != 200:
            return
        
        fh_data = fh_response.json()
        if not fh_data or "metric" not in fh_data:
            return
        
        fh = fh_data["metric"]
        m = metric_obj
        supplemented = []
        
        # --- Valuation Ratios ---
        if m.price_to_earnings_ratio is None:
            val = fh.get("peTTM") or fh.get("peNormalizedAnnual") or fh.get("peExclExtraTTM")
            if val is not None:
                m.price_to_earnings_ratio = float(val)
                supplemented.append("PE")
        
        if m.price_to_book_ratio is None:
            val = fh.get("pbQuarterly") or fh.get("pbAnnual") or fh.get("pb")
            if val is not None:
                m.price_to_book_ratio = float(val)
                supplemented.append("P/B")
        
        if m.price_to_sales_ratio is None:
            val = fh.get("psTTM") or fh.get("psAnnual")
            if val is not None:
                m.price_to_sales_ratio = float(val)
                supplemented.append("P/S")
        
        if m.peg_ratio is None:
            val = fh.get("pegTTM")
            if val is not None:
                m.peg_ratio = float(val)
                supplemented.append("PEG")
        
        if m.enterprise_value is None:
            val = fh.get("enterpriseValue")
            if val is not None:
                # Finnhub returns EV in millions, convert to actual
                m.enterprise_value = float(val) * 1_000_000
                supplemented.append("EV")
        
        if m.enterprise_value_to_ebitda_ratio is None:
            val = fh.get("evEbitdaTTM")
            if val is not None:
                ev_ebitda = float(val)
                # Negative EV/EBITDA is almost always a data error
                if ev_ebitda > 0:
                    m.enterprise_value_to_ebitda_ratio = ev_ebitda
                    supplemented.append("EV/EBITDA")
        
        if m.market_cap is None:
            val = fh.get("marketCapitalization")
            if val is not None:
                m.market_cap = float(val) * 1_000_000
                supplemented.append("market_cap")
        
        # --- Profitability Ratios ---
        if m.return_on_equity is None:
            val = fh.get("roeTTM") or fh.get("roeRfy") or fh.get("roe5Y")
            if val is not None:
                # Finnhub returns percentages (16.32 means 16.32%)
                m.return_on_equity = float(val) / 100.0
                supplemented.append("ROE")
        
        if m.return_on_assets is None:
            val = fh.get("roaTTM") or fh.get("roaRfy") or fh.get("roa5Y")
            if val is not None:
                m.return_on_assets = float(val) / 100.0
                supplemented.append("ROA")
        
        if m.gross_margin is None:
            # Banks often don't have gross margin, skip if not available
            pass  # Intentionally not supplementing - gross margin N/A for financials
        
        if m.operating_margin is None:
            val = fh.get("operatingMarginTTM") or fh.get("operatingMarginAnnual")
            if val is not None:
                m.operating_margin = float(val) / 100.0
                supplemented.append("operating_margin")
        
        if m.net_margin is None or (m.net_margin is not None and m.net_margin == 0):
            val = fh.get("netProfitMarginTTM") or fh.get("netProfitMarginAnnual")
            if val is not None:
                m.net_margin = float(val) / 100.0
                supplemented.append("net_margin")
        
        # --- Leverage & Liquidity ---
        if m.debt_to_equity is None or (m.debt_to_equity is not None and m.debt_to_equity == 0):
            val = fh.get("totalDebt/totalEquityQuarterly") or fh.get("longTermDebt/equityQuarterly") or fh.get("totalDebt/totalEquityAnnual")
            if val is not None:
                m.debt_to_equity = float(val)
                supplemented.append("D/E")
        
        if m.current_ratio is None or (m.current_ratio is not None and m.current_ratio == 0):
            val = fh.get("currentRatioTTM") or fh.get("currentRatioQuarterly")
            if val is not None:
                m.current_ratio = float(val)
                supplemented.append("current_ratio")
        
        # --- Growth ---
        if m.revenue_growth is None:
            # TTM growth can be extreme/anomalous for financial stocks
            # Prefer more stable 5Y/3Y averages when TTM looks unreasonable
            ttm_val = fh.get("revenueGrowthTTMYoy")
            y5_val = fh.get("revenueGrowth5Y")
            y3_val = fh.get("revenueGrowth3Y")
            # Use TTM only if reasonable (<100%), otherwise fall back to longer-term
            if ttm_val is not None and abs(float(ttm_val)) < 100:
                m.revenue_growth = float(ttm_val) / 100.0
                supplemented.append("revenue_growth")
            elif y5_val is not None:
                m.revenue_growth = float(y5_val) / 100.0
                supplemented.append("revenue_growth(5Y)")
            elif y3_val is not None:
                m.revenue_growth = float(y3_val) / 100.0
                supplemented.append("revenue_growth(3Y)")
        
        if m.earnings_growth is None:
            val = fh.get("epsGrowthTTMYoy") or fh.get("epsGrowthQuarterlyYoy")
            if val is not None:
                m.earnings_growth = float(val) / 100.0
                supplemented.append("earnings_growth")
        
        if m.book_value_growth is None:
            val = fh.get("bookValueShareGrowth5Y")
            if val is not None:
                m.book_value_growth = float(val) / 100.0
                supplemented.append("book_value_growth")
        
        # --- Per Share ---
        if m.earnings_per_share is None:
            val = fh.get("epsTTM") or fh.get("epsAnnual")
            if val is not None:
                m.earnings_per_share = float(val)
                supplemented.append("EPS")
        
        if m.book_value_per_share is None:
            val = fh.get("bookValuePerShareQuarterly") or fh.get("bookValuePerShareAnnual")
            if val is not None:
                m.book_value_per_share = float(val)
                supplemented.append("BVPS")
        
        # --- Dividend ---
        if m.payout_ratio is None:
            val = fh.get("payoutRatioTTM") or fh.get("payoutRatioAnnual")
            if val is not None:
                m.payout_ratio = float(val) / 100.0
                supplemented.append("payout_ratio")
        
        if supplemented:
            logger.info(f"Supplemented {ticker} with Finnhub: {', '.join(supplemented)}")
    
    except Exception as e:
        logger.info(f"Finnhub supplement failed for {ticker}: {e}")


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API with multiple sources."""
    # Normalize ticker for yfinance compatibility
    yf_ticker_name = _normalize_ticker_for_yfinance(ticker)
    
    # Financial metrics are immutable - store permanently by ticker

    # Check cache first - simple exact match (cache handles normalization)
    if cached_data := _cache.get_financial_metrics(ticker):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # PRIMARY: financialdatasets.ai (best quality, if API key has credits)
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            headers["X-API-KEY"] = financial_api_key

            url = f"https://api.financialdatasets.ai/financial-metrics?ticker={ticker.upper()}&period={period}&limit={limit}"
            response = _make_api_request(url, headers)

            if response.status_code == 200:
                data = response.json()
                metrics = data.get("financial_metrics", [])
                if metrics and len(metrics) > 0:
                    # Check that metrics have actual non-zero values (not empty placeholders)
                    first = metrics[0]
                    has_real_data = (
                        first.get("return_on_equity") is not None
                        or first.get("debt_to_equity") is not None
                        or first.get("operating_margin") is not None
                    )
                    if has_real_data:
                        result = [FinancialMetrics(**metric) for metric in metrics]
                        # Supplement missing metrics from Finnhub
                        _supplement_with_finnhub(ticker, result[0])
                        _cache.set_financial_metrics(ticker, [m.model_dump() for m in result])
                        logger.info(f"Successfully retrieved {len(result)} financial metrics for {ticker} via financialdatasets.ai")
                        return result
        except Exception as e:
            logger.info(f"financialdatasets.ai financial metrics failed for {ticker}: {e}, trying fallbacks")

    # FALLBACK 1: yfinance (free, unlimited)
    try:
        import yfinance as yf
        from datetime import datetime

        # Get stock info from yfinance (use normalized ticker for compatibility)
        yf_ticker = yf.Ticker(yf_ticker_name)

        # Get HISTORICAL quarterly financials (for Growth Analyst)
        quarterly_financials = yf_ticker.quarterly_financials
        info = yf_ticker.info

        if quarterly_financials is not None and len(quarterly_financials.columns) >= 1:
            financial_metrics = []
            all_quarters = list(quarterly_financials.columns)[:limit]  # Get up to N quarters

            def safe_float(data_dict, key, default=0):
                val = data_dict.get(key, default)
                try:
                    import math
                    if val is None or val == "" or val == "None" or (isinstance(val, float) and math.isnan(val)):
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # First pass: collect only valid quarters with revenue data
            valid_quarters = []
            for quarter_date in all_quarters:
                report = quarterly_financials[quarter_date]
                revenue = safe_float(report, "Total Revenue")
                if revenue > 0:
                    valid_quarters.append((quarter_date, report))

            for i, (quarter_date, report) in enumerate(valid_quarters):
                revenue = safe_float(report, "Total Revenue")
                gross_profit = safe_float(report, "Gross Profit")
                operating_income = safe_float(report, "Operating Income")
                net_income = safe_float(report, "Net Income")

                # Calculate margins
                # NOTE: For holding companies like BRK, yfinance may have revenue
                # but no gross_profit or operating_income. Setting margin to 0 is
                # misleading (implies zero margin, not missing data). Use None instead.
                gross_margin = gross_profit / revenue if (revenue > 0 and gross_profit > 0) else None
                operating_margin = operating_income / revenue if (revenue > 0 and operating_income > 0) else None
                net_margin = net_income / revenue if revenue > 0 else 0
                
                # Fallback: if gross_margin is None from statement data, try yfinance info
                # (e.g., BRK has grossMargins=0.2778 in info but no Gross Profit in income_stmt)
                if gross_margin is None and info:
                    info_gm = safe_float(info, "grossMargins")
                    if info_gm is not None and info_gm > 0:
                        gross_margin = info_gm
                
                # Fallback: if operating_margin is None from statement data, try yfinance info
                if operating_margin is None and info:
                    info_om = safe_float(info, "operatingMargins")
                    if info_om is not None and info_om > 0:
                        operating_margin = info_om

                # Get growth: first try yfinance info for TTM growth rate
                # Fallback to sequential quarter-over-quarter calculation
                rev_growth = safe_float(info, "revenueGrowth") if info else None
                if rev_growth is None and i > 0 and i < len(valid_quarters):
                    prev_report = valid_quarters[i-1][1]
                    prev_revenue = safe_float(prev_report, "Total Revenue")
                    if prev_revenue > 0:
                        rev_growth = (revenue - prev_revenue) / prev_revenue

                # Try to get balance sheet data for calculated ratios
                # (more reliable than info dict which often is None)
                # Try multiple field name variations since yfinance is inconsistent
                debt_fields = ['Total Debt', 'Debt', 'TotalDebt', 'Long Term Debt + Short Term Debt']
                equity_fields = ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']
                ca_fields = ['Current Assets', 'Total Current Assets', 'CurrentAssets']
                cl_fields = ['Current Liabilities', 'Total Current Liabilities', 'CurrentLiabilities']
                
                total_debt = None
                for field in debt_fields:
                    total_debt = safe_float(report, field)
                    if total_debt is not None:
                        break
                
                total_equity = None
                for field in equity_fields:
                    total_equity = safe_float(report, field)
                    if total_equity is not None:
                        break
                
                total_current_assets = None
                for field in ca_fields:
                    total_current_assets = safe_float(report, field)
                    if total_current_assets is not None:
                        break
                
                total_current_liabilities = None
                for field in cl_fields:
                    total_current_liabilities = safe_float(report, field)
                    if total_current_liabilities is not None:
                        break

                # Calculate ratios from balance sheet data (more reliable)
                calculated_debt_to_equity = None
                if total_debt is not None and total_equity is not None and total_equity > 0:
                    calculated_debt_to_equity = total_debt / total_equity

                calculated_current_ratio = None
                if total_current_assets is not None and total_current_liabilities is not None and total_current_liabilities > 0:
                    calculated_current_ratio = total_current_assets / total_current_liabilities

                # Use info if available, otherwise calculated
                info_debt_eq = safe_float(info, "debtToEquity") if info else None
                info_current_ratio = safe_float(info, "currentRatio") if info else None
                
                # yfinance safe_float returns 0 for missing fields, but D/E=0 and CR=0
                # are almost always data errors (not real values). Convert 0 to None.
                if info_debt_eq == 0:
                    info_debt_eq = None
                if info_current_ratio == 0:
                    info_current_ratio = None

                final_debt_to_equity = info_debt_eq if info_debt_eq is not None else calculated_debt_to_equity
                final_current_ratio = info_current_ratio if info_current_ratio is not None else calculated_current_ratio

                # Normalize debtToEquity from yfinance (it's often returned as percentage)
                if final_debt_to_equity is not None and final_debt_to_equity > 10:
                    final_debt_to_equity = final_debt_to_equity / 100.0

                # Also normalize if it's clearly a percentage value for ROE/ROA
                info_roe = safe_float(info, "returnOnEquity") if info else None
                info_roa = safe_float(info, "returnOnAssets") if info else None
                
                final_roe = info_roe / 100.0 if info_roe is not None and info_roe > 5 else info_roe
                final_roa = info_roa / 100.0 if info_roa is not None and info_roa > 5 else info_roa

                # Sanity checks on yfinance info values
                # yfinance sometimes returns wildly incorrect values for certain stocks
                info_pb = safe_float(info, "priceToBook") if info else None
                # P/B ratio below 0.01 is almost certainly a data error (normal range: 0.1-100)
                if info_pb is not None and info_pb < 0.01:
                    info_pb = None
                    logger.info(f"Discarded suspicious priceToBook={safe_float(info, 'priceToBook')} for {ticker}")
                
                info_ev = safe_float(info, "enterpriseValue") if info else None
                # Negative EV with positive market cap is usually a data error
                info_mcap = safe_float(info, "marketCap") if info else None
                if info_ev is not None and info_mcap is not None and info_ev < 0 and info_mcap > 0:
                    info_ev = None
                    logger.info(f"Discarded suspicious enterpriseValue={safe_float(info, 'enterpriseValue')} for {ticker}")

                metrics = FinancialMetrics(
                    ticker=ticker,
                    report_period=quarter_date.strftime("%Y-%m-%d"),
                    period=period,
                    currency="USD",
                    market_cap=info_mcap,
                    enterprise_value=info_ev,
                    price_to_earnings_ratio=safe_float(info, "trailingPE") if info else None,
                    price_to_book_ratio=info_pb,
                    price_to_sales_ratio=safe_float(info, "priceToSalesTrailing12Months") if info else None,
                    enterprise_value_to_revenue_ratio=safe_float(info, "enterpriseToRevenue") if info else None,
                    enterprise_value_to_ebitda_ratio=(lambda v: v if v and v > 0 else None)(safe_float(info, "enterpriseToEbitda")) if info else None,
                    free_cash_flow_yield=None,
                    peg_ratio=safe_float(info, "pegRatio") if info else None,
                    gross_margin=gross_margin,
                    operating_margin=operating_margin,
                    net_margin=net_margin,
                    return_on_equity=final_roe,
                    return_on_assets=final_roa,
                    return_on_invested_capital=None,
                    asset_turnover=None,
                    inventory_turnover=None,
                    receivables_turnover=None,
                    days_sales_outstanding=None,
                    operating_cycle=None,
                    working_capital_turnover=None,
                    current_ratio=final_current_ratio,
                    quick_ratio=safe_float(info, "quickRatio") if info else None,
                    cash_ratio=None,
                    operating_cash_flow_ratio=None,
                    debt_to_equity=final_debt_to_equity,
                    debt_to_assets=None,
                    interest_coverage=None,
                    revenue_growth=rev_growth,
                    earnings_growth=safe_float(info, "earningsGrowth") if info else None,
                    book_value_growth=None,
                    earnings_per_share_growth=None,
                    free_cash_flow_growth=None,
                    operating_income_growth=None,
                    ebitda_growth=None,
                    payout_ratio=None,
                    earnings_per_share=safe_float(info, "trailingEps") if info else None,
                    book_value_per_share=safe_float(info, "bookValue") if info else None,
                    free_cash_flow_per_share=None,
                )
                financial_metrics.append(metrics)

            if len(financial_metrics) >= 1:
                # Supplement with comprehensive metrics from Finnhub (free, 60 calls/min)
                _supplement_with_finnhub(ticker, financial_metrics[0])
                
                _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
                logger.info(f"Successfully retrieved {len(financial_metrics)} historical financial metrics for {ticker} via yfinance")
                return financial_metrics
    except Exception as e:
        logger.info(f"yfinance financial metrics failed for {ticker}: {e}, trying fallbacks")

    api_success = False  # yfinance already tried and failed, now try next fallback

    # Fallback to Finnhub first (best free tier: 60 calls/minute)
    # NOTE: Finnhub stock/financials is PAID ONLY on free tier (403 error)
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if not api_success and finnhub_key:
        try:
            import requests
            from datetime import datetime

            url = f"https://finnhub.io/api/v1/stock/financials?symbol={ticker}&frequency=quarterly&token={finnhub_key}"
            fh_response = requests.get(url, timeout=15)

            if fh_response.status_code == 403:
                logger.info(f"Finnhub stock/financials is PAID-ONLY (403), skipping - {ticker}")
            elif fh_response.status_code == 200:
                data = fh_response.json()
                # Check for rate limit
                if "error" in data and "limit" in data["error"].lower():
                    logger.info(f"Finnhub rate limit reached for financial metrics - {ticker}")
                elif data and "financials" in data and len(data["financials"]) >= 1:
                    # Got historical financial statements - build multiple FinancialMetrics records
                    reports = data["financials"][:limit]
                    financial_metrics = []

                    def safe_float(data_dict, key, default=0):
                        val = data_dict.get(key, default)
                        try:
                            return float(val) if val and val != "None" and val != "" else default
                        except (ValueError, TypeError):
                            return default

                    for i, report in enumerate(reports):
                        report_date = report.get("period", datetime.now().strftime("%Y-%m-%d"))
                        
                        revenue = safe_float(report, "Revenue")
                        gross_profit = safe_float(report, "GrossProfit")
                        operating_income = safe_float(report, "OperatingIncome")
                        net_income = safe_float(report, "NetIncome")
                        
                        # Calculate margins
                        gross_margin = gross_profit / revenue if revenue > 0 else 0
                        operating_margin = operating_income / revenue if revenue > 0 else 0
                        net_margin = net_income / revenue if revenue > 0 else 0

                        # Get growth from previous quarter (if available)
                        rev_growth = None
                        if i > 0 and i < len(reports):
                            prev_rev = safe_float(reports[i], "Revenue")
                            if prev_rev > 0:
                                rev_growth = (revenue - prev_rev) / prev_rev

                        metrics = FinancialMetrics(
                            ticker=ticker,
                            report_period=report_date,
                            period=period,
                            currency="USD",
                            market_cap=None,
                            enterprise_value=None,
                            price_to_earnings_ratio=None,
                            price_to_book_ratio=None,
                            price_to_sales_ratio=None,
                            enterprise_value_to_ebitda_ratio=None,
                            free_cash_flow_yield=None,
                            peg_ratio=None,
                            gross_margin=gross_margin,
                            operating_margin=operating_margin,
                            net_margin=net_margin,
                            return_on_equity=safe_float(report, "ReturnOnEquityTTM"),
                            return_on_assets=safe_float(report, "ReturnOnAssetsTTM"),
                            return_on_invested_capital=None,
                            asset_turnover=None,
                            inventory_turnover=None,
                            receivables_turnover=None,
                            days_sales_outstanding=None,
                            operating_cycle=None,
                            working_capital_turnover=None,
                            current_ratio=safe_float(report, "CurrentRatio"),
                            quick_ratio=safe_float(report, "QuickRatio"),
                            cash_ratio=None,
                            operating_cash_flow_ratio=None,
                            debt_to_equity=safe_float(report, "DebtToEquity"),
                            debt_to_assets=None,
                            interest_coverage=None,
                            revenue_growth=rev_growth,
                            earnings_growth=None,
                            book_value_growth=None,
                            earnings_per_share_growth=None,
                            free_cash_flow_growth=None,
                            operating_income_growth=None,
                            ebitda_growth=None,
                            payout_ratio=None,
                            earnings_per_share=safe_float(report, "EPS"),
                            book_value_per_share=None,
                            free_cash_flow_per_share=None,
                        )
                        financial_metrics.append(metrics)

                    if len(financial_metrics) >= 1:
                        # Supplement missing metrics from Finnhub
                        _supplement_with_finnhub(ticker, financial_metrics[0])
                        
                        _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
                        logger.info(f"Successfully retrieved {len(financial_metrics)} historical financial metrics for {ticker} via Finnhub")
                        api_success = True
        except Exception as e:
            logger.info(f"Finnhub financial metrics fallback failed for {ticker}: {e}")

    # Fallback to Alpha Vantage next (proper API, 25 calls/day free limit)
    alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_success and alpha_vantage_key:
        try:
            import requests
            from datetime import datetime

            # First try to get historical financial data from INCOME_STATEMENT
            url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={alpha_vantage_key}"
            av_response = requests.get(url, timeout=15)

            if av_response.status_code == 200:
                data = av_response.json()
                # Check for rate limit
                if "Information" in data and "rate limit" in data["Information"].lower():
                    logger.info(f"Alpha Vantage rate limit reached for financial metrics - {ticker}")
                elif data and "quarterlyReports" in data and len(data["quarterlyReports"]) >= 1:
                    # Got historical income statements - build multiple FinancialMetrics records
                    reports = data["quarterlyReports"][:limit]  # Use first N quarterly reports
                    financial_metrics = []

                    def safe_float(data_dict, key, default=0):
                        val = data_dict.get(key, default)
                        try:
                            return float(val) if val and val != "None" and val != "" else default
                        except (ValueError, TypeError):
                            return default

                    for i, report in enumerate(reports):
                        report_date = report.get("fiscalDateEnding", datetime.now().strftime("%Y-%m-%d"))
                        
                        revenue = safe_float(report, "totalRevenue")
                        gross_profit = safe_float(report, "grossProfit")
                        operating_income = safe_float(report, "operatingIncome")
                        net_income = safe_float(report, "netIncome")
                        
                        # Calculate margins
                        gross_margin = gross_profit / revenue if revenue > 0 else 0
                        operating_margin = operating_income / revenue if revenue > 0 else 0
                        net_margin = net_income / revenue if revenue > 0 else 0

                        # Get growth from previous quarter (if available)
                        rev_growth = None
                        if i > 0 and i < len(reports):
                            prev_rev = safe_float(reports[i], "totalRevenue")
                            if prev_rev > 0:
                                rev_growth = (revenue - prev_rev) / prev_rev

                        metrics = FinancialMetrics(
                            ticker=ticker,
                            report_period=report_date,
                            period=period,
                            currency="USD",
                            market_cap=None,  # Not in income statement
                            enterprise_value=None,
                            price_to_earnings_ratio=None,
                            price_to_book_ratio=None,
                            price_to_sales_ratio=None,
                            enterprise_value_to_ebitda_ratio=None,
                            enterprise_value_to_revenue_ratio=None,
                            free_cash_flow_yield=None,
                            peg_ratio=None,
                            gross_margin=gross_margin,
                            operating_margin=operating_margin,
                            net_margin=net_margin,
                            return_on_equity=None,
                            return_on_assets=None,
                            return_on_invested_capital=None,
                            asset_turnover=None,
                            inventory_turnover=None,
                            receivables_turnover=None,
                            days_sales_outstanding=None,
                            operating_cycle=None,
                            working_capital_turnover=None,
                            current_ratio=None,
                            quick_ratio=None,
                            cash_ratio=None,
                            operating_cash_flow_ratio=None,
                            debt_to_equity=None,
                            debt_to_assets=None,
                            interest_coverage=None,
                            revenue_growth=rev_growth,
                            earnings_growth=None,
                            book_value_growth=None,
                            earnings_per_share_growth=None,
                            free_cash_flow_growth=None,
                            operating_income_growth=None,
                            ebitda_growth=None,
                            payout_ratio=None,
                            earnings_per_share=safe_float(report, "reportedEPS"),
                            book_value_per_share=None,
                            free_cash_flow_per_share=None,
                        )
                        financial_metrics.append(metrics)

                    if len(financial_metrics) >= 1:
                        # Supplement missing metrics from Finnhub
                        _supplement_with_finnhub(ticker, financial_metrics[0])
                        
                        _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
                        logger.info(f"Successfully retrieved {len(financial_metrics)} historical financial metrics for {ticker} via Alpha Vantage")
                        return financial_metrics

        except Exception as e:
            logger.info(f"Alpha Vantage financial metrics fallback failed for {ticker}: {e}")

    # NO DEMO DATA - Return empty list if no real data found
    logger.info(f"No real financial metrics data available for {ticker}")
    return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API with multiple sources."""
    # Normalize ticker for yfinance compatibility
    yf_ticker_name = _normalize_ticker_for_yfinance(ticker)
    
    # Check cache FIRST - avoid redundant API calls and rate limits
    # But only return cache if it contains ALL requested fields (different agents
    # request different fields, so partial cache from a previous agent is insufficient)
    if _cache.has_sufficient_line_items(ticker):
        cached_data = _cache.get_line_items(ticker)
        if cached_data:
            # Check if cached data has all requested fields (at least in one record)
            # Core fields like ticker, report_period, period, currency are always present
            data_fields = set()
            for item in cached_data:
                data_fields.update(item.keys())
            # Check if ALL requested line_item fields exist in the cached data
            missing_fields = [f for f in line_items if f not in data_fields]
            if not missing_fields:
                logger.info(f"Using {len(cached_data)} cached line items for {ticker} (all {len(line_items)} fields present)")
                return [LineItem(**item) for item in cached_data[:limit]]
            else:
                logger.info(f"Cache hit for {ticker} but missing {len(missing_fields)} fields: {missing_fields[:5]}... fetching from API")
    
    # If not in cache or insufficient data, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    api_success = response.status_code == 200 and not response.json().get("error")

    if not api_success:
        # NO DEMO DATA - try yfinance fallback
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(yf_ticker_name)
            
            # Try to get income statement and balance sheet
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            if income_stmt is None or income_stmt.empty:
                # For some tickers (e.g., banks like JPM), income_stmt may be empty
                # but cashflow or balance_sheet may have data. Use whichever has columns.
                if cash_flow is not None and not cash_flow.empty:
                    income_stmt = cash_flow  # Use cash_flow as column source
                elif balance_sheet is not None and not balance_sheet.empty:
                    income_stmt = balance_sheet  # Use balance_sheet as column source
                else:
                    # yfinance returned empty data (rate-limited or unsupported ticker)
                    # Don't return [] here - fall through to Alpha Vantage fallback
                    raise ValueError("yfinance returned empty financial statements")
            
            # Build line items from yfinance data
            # IMPORTANT: Group all fields into ONE LineItem per period
            # (matching financialdatasets.ai format), not one LineItem per field
            result_items = []
            
            # Field mapping: requested field name -> yfinance index name
            field_mapping = {
                'net_income': ['Net Income', 'Net Income Common Stockholders', 'NetIncome'],
                'total_assets': ['Total Assets'],
                'total_liabilities': ['Total Liabilities Net Minority Interest', 'Total Liabilities'],
                'shareholders_equity': ['Stockholders Equity', 'Total Equity Gross Minority Interest'],
                'gross_profit': ['Gross Profit'],
                'revenue': ['Total Revenue', 'Revenue'],
                'free_cash_flow': ['Free Cash Flow'],
                'capital_expenditure': ['Capital Expenditure', 'CapitalExpenditures'],
                'depreciation_and_amortization': ['Depreciation And Amortization', 'Depreciation'],
                'outstanding_shares': ['Share Issued', 'Ordinary Shares Number'],  # NOTE: 'Common Stock' is par value, NOT share count
                'dividends_and_other_cash_distributions': ['Cash Dividends Paid', 'Dividends Paid', 'Common Stock Dividend Paid'],
                'issuance_or_purchase_of_equity_shares': ['Issuance Of Stock', 'Repurchase Of Stock', 'Net Issuance Of Stock', 'Sale And Purchase Of Stock'],
                'operating_income': ['Operating Income'],  # NOTE: 'Operating Revenue' is NOT operating income - it's a synonym for Total Revenue
                'cost_of_revenue': ['Cost Of Revenue', 'Total Expenses'],
                'research_and_development': ['Research And Development', 'Research Development', 'R&D'],
                'operating_expense': ['Operating Expense', 'Total Operating Expenses', 'Selling General Administrative'],
                # Fields needed by Charlie Munger and other agents
                'cash_and_equivalents': ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash Equivalents'],
                'total_debt': ['Total Debt', 'Long Term Debt And Capital Lease Obligation'],
                'goodwill_and_intangible_assets': ['Goodwill And Other Intangible Assets', 'Goodwill'],
                # Fields needed by valuation, druckenmiller, fisher, damodaran, etc.
                'ebit': ['EBIT'],
                'ebitda': ['EBITDA', 'Normalized EBITDA'],
                'interest_expense': ['Interest Expense', 'Interest Expense Non Operating'],
                # Fields needed by graham, pabrai, lynch, rakesh, valuation, etc.
                'current_assets': ['Current Assets'],
                'current_liabilities': ['Current Liabilities'],
                'earnings_per_share': ['Basic EPS', 'Diluted EPS'],
                'working_capital': ['Working Capital'],
            }
            
            # We need at least 3 periods for reliable valuation
            # Get up to 5 most recent reporting periods
            periods_to_process = min(5, len(income_stmt.columns))
            
            for period_idx in range(periods_to_process):
                report_date = income_stmt.columns[period_idx]
                
                # Build one item_data dict with ALL requested fields for this period
                item_data = {
                    'ticker': ticker,
                    'report_period': report_date.strftime("%Y-%m-%d") if hasattr(report_date, 'strftime') else str(report_date).split()[0],
                    'period': period,
                    'currency': 'USD',
                }
                has_any_field = False
                
                # Helper to look up a yfinance field value for this period
                # Returns (found: bool, value: float | None) to distinguish 0 from missing
                def _lookup_yf_field(yf_field_name):
                    """Look up a single yfinance field across all 3 financial statements.
                    Returns (True, value) if found (including 0), (False, None) if not found.
                    Filters out NaN values which yfinance returns for missing data.
                    """
                    for df_candidate in [income_stmt, balance_sheet, cash_flow]:
                        if df_candidate is not None and yf_field_name in df_candidate.index and report_date in df_candidate.columns:
                            try:
                                val = float(df_candidate.loc[yf_field_name, report_date])
                                # Filter out NaN - yfinance returns NaN for missing data
                                if val != val:  # NaN != NaN is True
                                    continue
                                return True, val
                            except (KeyError, ValueError, TypeError):
                                continue
                    return False, None
                
                for requested_field in line_items:
                    if requested_field in field_mapping:
                        found, value = False, None
                        for yf_field in field_mapping[requested_field]:
                            found, value = _lookup_yf_field(yf_field)
                            if found:
                                break
                        
                        if found:
                            item_data[requested_field] = value
                            has_any_field = True
                
                # Calculate ratio fields from raw yfinance data when requested
                # These are not direct line items in yfinance but can be computed from raw values.
                # Use _lookup_yf_field which returns (found, value) to handle 0 correctly
                # (Python's `or` treats 0 as falsy, which would skip legitimate zero values).
                if 'gross_margin' in line_items and 'gross_margin' not in item_data:
                    gp_found, gross_profit = item_data.get('gross_profit') is not None, item_data.get('gross_profit')
                    if not gp_found:
                        gp_found, gross_profit = _lookup_yf_field('Gross Profit')
                    rev_found, revenue = item_data.get('revenue') is not None, item_data.get('revenue')
                    if not rev_found:
                        rev_found, revenue = _lookup_yf_field('Total Revenue')
                    if not rev_found:
                        rev_found, revenue = _lookup_yf_field('Revenue')
                    if gp_found and rev_found and revenue and revenue != 0:
                        item_data['gross_margin'] = gross_profit / revenue
                        has_any_field = True
                
                if 'operating_margin' in line_items and 'operating_margin' not in item_data:
                    oi_found, operating_income = _lookup_yf_field('Operating Income')
                    # Do NOT fall back to 'Operating Revenue' - it equals Total Revenue, not Operating Income
                    rev_found, revenue = item_data.get('revenue') is not None, item_data.get('revenue')
                    if not rev_found:
                        rev_found, revenue = _lookup_yf_field('Total Revenue')
                    if not rev_found:
                        rev_found, revenue = _lookup_yf_field('Revenue')
                    if oi_found and rev_found and revenue and revenue != 0:
                        item_data['operating_margin'] = operating_income / revenue
                        has_any_field = True
                
                if 'debt_to_equity' in line_items and 'debt_to_equity' not in item_data:
                    tl_found, total_liabilities = item_data.get('total_liabilities') is not None, item_data.get('total_liabilities')
                    if not tl_found:
                        tl_found, total_liabilities = _lookup_yf_field('Total Liabilities Net Minority Interest')
                    if not tl_found:
                        tl_found, total_liabilities = _lookup_yf_field('Total Liabilities')
                    se_found, shareholders_equity = _lookup_yf_field('Stockholders Equity')
                    if not se_found:
                        se_found, shareholders_equity = _lookup_yf_field('Total Equity Gross Minority Interest')
                    if tl_found and se_found and shareholders_equity and shareholders_equity != 0:
                        item_data['debt_to_equity'] = total_liabilities / shareholders_equity
                        has_any_field = True
                
                # ROIC = Operating Income / Invested Capital
                # Invested Capital can be computed as: Stockholders Equity + Total Debt - Cash
                # Or yfinance may provide "Invested Capital" directly
                if 'return_on_invested_capital' in line_items and 'return_on_invested_capital' not in item_data:
                    oi_found, operating_income = _lookup_yf_field('Operating Income')
                    # Do NOT fall back to 'Operating Revenue' - it equals Total Revenue, not Operating Income
                    # Try direct "Invested Capital" field first
                    ic_found, invested_capital = _lookup_yf_field('Invested Capital')
                    if not ic_found:
                        # Compute: Invested Capital = Equity + Total Debt - Cash
                        eq_found, equity = _lookup_yf_field('Stockholders Equity')
                        if not eq_found:
                            eq_found, equity = _lookup_yf_field('Total Equity Gross Minority Interest')
                        td_found, total_debt = _lookup_yf_field('Total Debt')
                        if not td_found:
                            td_found, total_debt = _lookup_yf_field('Long Term Debt And Capital Lease Obligation')
                        ca_found, cash = _lookup_yf_field('Cash And Cash Equivalents')
                        if not ca_found:
                            ca_found, cash = _lookup_yf_field('Cash Cash Equivalents And Short Term Investments')
                        if eq_found and td_found and ca_found:
                            invested_capital = equity + total_debt - cash
                            ic_found = True
                    if oi_found and ic_found and invested_capital and invested_capital != 0:
                        item_data['return_on_invested_capital'] = operating_income / invested_capital
                        has_any_field = True
                
                # book_value_per_share = shareholders_equity / outstanding_shares
                if 'book_value_per_share' in line_items and 'book_value_per_share' not in item_data:
                    se_found, equity = _lookup_yf_field('Stockholders Equity')
                    if not se_found:
                        se_found, equity = _lookup_yf_field('Total Equity Gross Minority Interest')
                    os_found, shares = item_data.get('outstanding_shares') is not None, item_data.get('outstanding_shares')
                    if not os_found:
                        os_found, shares = _lookup_yf_field('Share Issued')
                    if not os_found:
                        os_found, shares = _lookup_yf_field('Ordinary Shares Number')
                    if se_found and os_found and shares and shares != 0:
                        item_data['book_value_per_share'] = equity / shares
                        has_any_field = True
                
                if has_any_field:
                    result_items.append(LineItem(**item_data))
            
            # Also try to get shares outstanding from info (only for latest period)
            # This is the most reliable source for actual share count
            info_shares_outstanding = None
            try:
                info = ticker_obj.info
                if info:
                    info_shares_outstanding = info.get('sharesOutstanding')
                    if info_shares_outstanding:
                        info_shares_outstanding = float(info_shares_outstanding)
            except:
                pass
            
            # Fix outstanding_shares for all periods:
            # 1. yfinance 'Share Issued' / 'Ordinary Shares Number' may be in thousands
            #    for some stocks (e.g., BRK-B: 1,438,223 vs actual 1,438,223,000)
            # 2. Use info.sharesOutstanding as reference to detect and correct unit
            if 'outstanding_shares' in line_items and result_items:
                if info_shares_outstanding and info_shares_outstanding > 0:
                    # Use info.sharesOutstanding for the latest period
                    result_items[0].outstanding_shares = info_shares_outstanding
                    
                    # For historical periods, check if shares are in thousands
                    # by comparing latest period's raw value with info.sharesOutstanding
                    if len(result_items) > 1:
                        for item in result_items[1:]:
                            raw_shares = getattr(item, 'outstanding_shares', None)
                            if raw_shares and raw_shares > 0:
                                # If raw shares is < 1% of info shares, it's likely in thousands
                                if info_shares_outstanding / raw_shares > 100:
                                    item.outstanding_shares = raw_shares * 1000
                elif result_items and len(result_items) > 0:
                    # No info shares available - use raw value but warn if suspiciously small
                    # (most public companies have > 1M shares outstanding)
                    pass
            
            # Filter out NaN values from LineItem fields
            # yfinance sometimes returns NaN for missing data points
            import math
            for item in result_items:
                for attr in list(item.__dict__.keys()) if hasattr(item, '__dict__') else []:
                    val = getattr(item, attr, None)
                    if isinstance(val, float) and math.isnan(val):
                        # Remove the NaN field entirely (set to None for pydantic extra fields)
                        try:
                            delattr(item, attr)
                        except (AttributeError, TypeError):
                            pass
            
            if result_items:
                logger.info(f"Retrieved {len(result_items)} line items ({len(set(li.report_period for li in result_items))} periods) for {ticker} via yfinance")
                # Cache yfinance results to avoid redundant API calls
                _cache.set_line_items(ticker, [item.model_dump() for item in result_items])
                return result_items
        except Exception as e:
            logger.info(f"yfinance line items fallback failed for {ticker}: {e}")
        
        # Alpha Vantage fallback for line items
        alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if alpha_vantage_key:
            try:
                import requests as _requests
                from datetime import datetime as _dt
                
                # Alpha Vantage field mapping: requested field -> (API function, response key)
                # We'll fetch income statement, balance sheet, and cash flow separately
                av_income_fields = {
                    'revenue': 'totalRevenue',
                    'gross_profit': 'grossProfit',
                    'operating_income': 'operatingIncome',
                    'net_income': 'netIncome',
                    'research_and_development': 'researchAndDevelopment',
                    'operating_expense': 'operatingExpenses',
                    'interest_expense': 'interestExpense',
                    'depreciation_and_amortization': 'depreciationAndAmortization',
                    'ebit': 'ebitda',  # AV doesn't have EBIT directly, approximate from EBITDA
                    'ebitda': 'ebitda',
                    'earnings_per_share': 'reportedEPS',
                    'cost_of_revenue': 'costofGoodsAndServicesSold',
                }
                av_balance_fields = {
                    'total_assets': 'totalAssets',
                    'total_liabilities': 'totalLiabilities',
                    'shareholders_equity': 'totalShareholderEquity',
                    'cash_and_equivalents': 'cashAndShortTermInvestments',
                    'total_debt': 'shortLongTermDebtTotal',
                    'goodwill_and_intangible_assets': 'goodwillAndIntangibleAssetsTotal',
                    'current_assets': 'totalCurrentAssets',
                    'current_liabilities': 'totalCurrentLiabilities',
                    'inventory': 'inventory',
                    'outstanding_shares': 'commonStockSharesOutstanding',
                }
                av_cash_fields = {
                    'capital_expenditure': 'capitalExpenditures',
                    'free_cash_flow': 'operatingCashflow',  # Will compute FCF properly below
                    'operating_cash_flow': 'operatingCashflow',
                    'dividends_and_other_cash_distributions': 'dividendPayout',
                    'issuance_or_purchase_of_equity_shares': 'issuanceOfStock',
                }
                
                # Helper to safely parse float from AV data
                def _av_safe_float(data_dict, key, default=None):
                    val = data_dict.get(key)
                    if val is None or val == "None" or val == "":
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                # Fetch all 3 statements
                freq = "quarterly" if period == "quarterly" else "annual"
                reports_by_period = {}  # date_str -> dict of all fields
                
                # Income Statement
                is_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={alpha_vantage_key}"
                is_resp = _requests.get(is_url, timeout=15)
                if is_resp.status_code == 200:
                    is_data = is_resp.json()
                    reports_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
                    if reports_key in is_data:
                        for report in is_data[reports_key][:limit]:
                            date_str = report.get("fiscalDateEnding", "")
                            if not date_str:
                                continue
                            if date_str not in reports_by_period:
                                reports_by_period[date_str] = {}
                            for req_field, av_key in av_income_fields.items():
                                if req_field in line_items:
                                    val = _av_safe_float(report, av_key)
                                    if val is not None:
                                        reports_by_period[date_str][req_field] = val
                            # Also store ebitda for FCF calculation
                            ebitda_val = _av_safe_float(report, 'ebitda')
                            if ebitda_val is not None:
                                reports_by_period[date_str]['_ebitda'] = ebitda_val
                
                # Balance Sheet
                bs_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={alpha_vantage_key}"
                bs_resp = _requests.get(bs_url, timeout=15)
                if bs_resp.status_code == 200:
                    bs_data = bs_resp.json()
                    reports_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
                    if reports_key in bs_data:
                        for report in bs_data[reports_key][:limit]:
                            date_str = report.get("fiscalDateEnding", "")
                            if not date_str:
                                continue
                            if date_str not in reports_by_period:
                                reports_by_period[date_str] = {}
                            for req_field, av_key in av_balance_fields.items():
                                if req_field in line_items:
                                    val = _av_safe_float(report, av_key)
                                    if val is not None:
                                        reports_by_period[date_str][req_field] = val
                
                # Cash Flow
                cf_url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={alpha_vantage_key}"
                cf_resp = _requests.get(cf_url, timeout=15)
                if cf_resp.status_code == 200:
                    cf_data = cf_resp.json()
                    reports_key = "quarterlyReports" if freq == "quarterly" else "annualReports"
                    if reports_key in cf_data:
                        for report in cf_data[reports_key][:limit]:
                            date_str = report.get("fiscalDateEnding", "")
                            if not date_str:
                                continue
                            if date_str not in reports_by_period:
                                reports_by_period[date_str] = {}
                            for req_field, av_key in av_cash_fields.items():
                                if req_field in line_items:
                                    val = _av_safe_float(report, av_key)
                                    if val is not None:
                                        reports_by_period[date_str][req_field] = val
                            # Store capex for FCF calculation
                            capex = _av_safe_float(report, 'capitalExpenditures')
                            if capex is not None:
                                reports_by_period[date_str]['_capex'] = capex
                
                # Build LineItem objects from collected data
                av_items = []
                for date_str in sorted(reports_by_period.keys(), reverse=True):
                    fields = reports_by_period[date_str]
                    if not fields:
                        continue
                    
                    # Calculate FCF = Operating Cash Flow - Capex
                    if 'free_cash_flow' in line_items and 'free_cash_flow' not in fields:
                        ocf = fields.get('operating_cash_flow')
                        capex = fields.get('_capex')
                        if ocf is not None and capex is not None:
                            fields['free_cash_flow'] = ocf - abs(capex)
                    
                    # Calculate gross_margin
                    if 'gross_margin' in line_items and 'gross_margin' not in fields:
                        gp = fields.get('gross_profit')
                        rev = fields.get('revenue')
                        if gp is not None and rev is not None and rev != 0:
                            fields['gross_margin'] = gp / rev
                    
                    # Calculate operating_margin
                    if 'operating_margin' in line_items and 'operating_margin' not in fields:
                        oi = fields.get('operating_income')
                        rev = fields.get('revenue')
                        if oi is not None and rev is not None and rev != 0:
                            fields['operating_margin'] = oi / rev
                    
                    # Calculate debt_to_equity
                    if 'debt_to_equity' in line_items and 'debt_to_equity' not in fields:
                        td = fields.get('total_debt')
                        se = fields.get('shareholders_equity')
                        if td is not None and se is not None and se != 0:
                            fields['debt_to_equity'] = td / se
                    
                    # Calculate return_on_invested_capital
                    if 'return_on_invested_capital' in line_items and 'return_on_invested_capital' not in fields:
                        oi = fields.get('operating_income')
                        se = fields.get('shareholders_equity')
                        td = fields.get('total_debt')
                        cash = fields.get('cash_and_equivalents')
                        if oi is not None and se is not None and td is not None and cash is not None:
                            invested = se + td - cash
                            if invested != 0:
                                fields['return_on_invested_capital'] = oi / invested
                    
                    # Calculate book_value_per_share
                    if 'book_value_per_share' in line_items and 'book_value_per_share' not in fields:
                        se = fields.get('shareholders_equity')
                        shares = fields.get('outstanding_shares')
                        if se is not None and shares is not None and shares != 0:
                            fields['book_value_per_share'] = se / shares
                    
                    # Calculate working_capital from current_assets - current_liabilities
                    if 'working_capital' in line_items and 'working_capital' not in fields:
                        ca = fields.get('current_assets')
                        cl = fields.get('current_liabilities')
                        if ca is not None and cl is not None:
                            fields['working_capital'] = ca - cl
                    
                    # Remove internal helper keys
                    item_data = {k: v for k, v in fields.items() if not k.startswith('_')}
                    
                    if item_data:
                        item_data['ticker'] = ticker
                        item_data['report_period'] = date_str
                        item_data['period'] = period
                        item_data['currency'] = 'USD'
                        av_items.append(LineItem(**item_data))
                
                if av_items:
                    logger.info(f"Retrieved {len(av_items)} line items for {ticker} via Alpha Vantage")
                    # Cache the results to avoid wasting AV API quota
                    _cache.set_line_items(ticker, [item.model_dump() for item in av_items])
                    return av_items
                    
            except Exception as e:
                logger.info(f"Alpha Vantage line items fallback failed for {ticker}: {e}")
        
        return []

    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except Exception as e:
        logger.info("Failed to parse line items response for %s: %s", ticker, e)
        return []
    if not search_results:
        return []

    # Cache the results to avoid redundant API calls
    _cache.set_line_items(ticker, [item.model_dump() for item in search_results[:limit]])
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Step 1: Check if we have sufficient AND fresh cached insider trades
    # Insider trades expire after 24 hours
    if _cache.has_sufficient_insider_trades(ticker):
        cached_data = _cache.get_insider_trades(ticker)
        logger.info(f"Using {len(cached_data)} fresh cached insider trades for {ticker}")
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200 or response.json().get("error"):
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.info("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    # NO DEMO DATA - return empty list when no real data found
    if not all_trades:
        return []
    
    # Only cache REAL insider trades data
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 5,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache first, API only if insufficient cached data."""
    # Normalize ticker for yfinance compatibility
    yf_ticker_name = _normalize_ticker_for_yfinance(ticker)
    # Step 1: Check if we have sufficient AND fresh cached news
    # News expires after 12 hours - always refresh stale cache
    if _cache.has_sufficient_company_news(ticker, start_date or "2020-01-01", end_date, min_count=limit):
        cached_data = _cache.get_company_news(ticker, start_date or "2020-01-01", end_date)
        logger.info(f"Using {len(cached_data)} fresh cached news articles for {ticker}")
        return [CompanyNews(**news) for news in cached_data[:limit]]
    
    # Step 2: Not enough cached data - fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200 or response.json().get("error"):
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.info("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit or len(all_news) >= limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    # Try Alpha Vantage for news - MORE RELIABLE than yfinance!
    # Alpha Vantage NEWS_SENTIMENT API provides pre-analyzed sentiment
    alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not all_news and alpha_vantage_key:
        try:
            import requests

            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_key}&limit={min(limit, 50)}"
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                if "Information" in data and "rate limit" in data["Information"].lower():
                    logger.info(f"Alpha Vantage rate limit reached for news - will use cached data")
                else:
                    feed = data.get("feed", [])
                    if feed and len(feed) > 0:
                        for item in feed:
                            # Map Alpha Vantage sentiment to our format
                            av_tickers = item.get("ticker_sentiment", [])
                            relevant_sentiment = None
                            for t in av_tickers:
                                if t.get("ticker", "").upper() == ticker.upper():
                                    relevant_sentiment = t
                                    break

                            sentiment = "neutral"
                            if relevant_sentiment:
                                score = float(relevant_sentiment.get("ticker_sentiment_score", 0))
                                if score > 0.15:
                                    sentiment = "positive"
                                elif score < -0.15:
                                    sentiment = "negative"

                            all_news.append(CompanyNews(
                                ticker=ticker,
                                title=item.get("title", ""),
                                source=item.get("source", "Alpha Vantage"),
                                date=item.get("time_published", "").replace(" ", "T") + "Z",
                                url=item.get("url", ""),
                                tickers=[t.get("ticker", "") for t in av_tickers],
                                sentiment=sentiment,  # Pre-analyzed by Alpha Vantage!
                            ))
                        logger.info(f"Successfully retrieved {len(feed)} news articles for {ticker} via Alpha Vantage")
        except Exception as e:
            logger.info(f"Alpha Vantage news fallback failed for {ticker}: {e}")

    # Fallback: try Finnhub - 60 requests/minute free tier!
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if not all_news and finnhub_key:
        try:
            import requests
            from datetime import datetime, timedelta
            
            # Calculate start date (14 days ago)
            start_dt = datetime.now() - timedelta(days=14)
            finnhub_start = start_dt.strftime("%Y-%m-%d")
            finnhub_end = end_date
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={finnhub_start}&to={finnhub_end}&token={finnhub_key}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    all_news = []
                    for item in data[:limit]:
                        all_news.append(CompanyNews(
                            ticker=ticker,
                            title=item.get("headline", ""),
                            source=item.get("source", "Finnhub"),
                            date=datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            url=item.get("url", ""),
                            tickers=[ticker],
                        ))
                    logger.info(f"Successfully retrieved {len(all_news)} news articles for {ticker} via Finnhub")
        except Exception as e:
            logger.info(f"Finnhub news fallback failed for {ticker}: {e}")

    # Fallback: try yfinance for real news if API failed (last resort - easily rate limited)
    if not all_news:
        try:
            import yfinance as yf
            from datetime import datetime
            
            yf_ticker = yf.Ticker(yf_ticker_name)
            yf_news = yf_ticker.news
            
            if yf_news and len(yf_news) > 0:
                all_news = []
                for item in yf_news[:limit]:
                    all_news.append(CompanyNews(
                        ticker=ticker,
                        title=item.get("title", ""),
                        source=item.get("publisher", "Yahoo Finance"),
                        date=datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        url=item.get("link", ""),
                        tickers=[ticker],
                    ))
                if all_news:
                    logger.info(f"Successfully retrieved {len(all_news)} real news articles for {ticker} via yfinance")
        except Exception as e:
            logger.info(f"yfinance news fallback failed for {ticker}: {e}")

    # Add simple keyword-based sentiment analysis for news without pre-analyzed sentiment
    # This ensures news from Finnhub/yfinance is not dropped by .dropna() in sentiment_analyst
    if all_news:
        positive_keywords = {"growth", "profit", "beat", "exceed", "success", "upgrade", "bullish", "gain", "increase", "surge", "rally", "record", "strong", "boom", "positive", "surpass", "raise"}
        negative_keywords = {"loss", "miss", "downgrade", "bearish", "drop", "fall", "weak", "crash", "slump", "lawsuit", "fraud", "decline", "investigation", "recall", "negative", "cut", "lower"}
        
        for news in all_news:
            if news.sentiment is None:
                title_lower = (news.title or "").lower()
                positive_count = sum(1 for word in positive_keywords if word in title_lower)
                negative_count = sum(1 for word in negative_keywords if word in title_lower)
                
                if positive_count > negative_count:
                    news.sentiment = "positive"
                elif negative_count > positive_count:
                    news.sentiment = "negative"
                else:
                    news.sentiment = "neutral"
    
    # Before falling back - check if we have ANY cached news
    # Even if not "sufficient", real cached news is better than nothing
    if not all_news:
        existing_cached = _cache.get_company_news(ticker, start_date, end_date)
        if existing_cached and len(existing_cached) > 0:
            logger.info(f"Rate limited, using {len(existing_cached)} existing cached news articles for {ticker}")
            return [CompanyNews(**news) for news in existing_cached]
    
    # NO DEMO DATA - return empty list when no real data found
    if not all_news:
        return []
    
    # Only cache REAL news (persisted to disk)
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from cache or API."""
    # Normalize ticker for yfinance compatibility
    yf_ticker_name = _normalize_ticker_for_yfinance(ticker)
    
    # Check cache first - market cap doesn't change frequently
    if cached_mc := _cache.get_market_cap(ticker):
        return cached_mc

    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code == 200 and not response.json().get("error"):
            try:
                data = response.json()
                response_model = CompanyFactsResponse(**data)
                if response_model.company_facts.market_cap:
                    _cache.set_market_cap(ticker, response_model.company_facts.market_cap)
                    return response_model.company_facts.market_cap
            except Exception as e:
                logger.info(f"Failed to parse company facts for {ticker}: {e}")

        # Fallback to yfinance info for market cap
        try:
            import yfinance as yf
            yf_ticker = yf.Ticker(yf_ticker_name)
            info = yf_ticker.info
            if info and info.get("marketCap"):
                logger.info(f"Retrieved market cap for {ticker} via yfinance")
                result = float(info.get("marketCap"))
                _cache.set_market_cap(ticker, result)
                return result
        except Exception as e:
            # Don't print warning - just continue
            pass

    # Fallback: try to get market cap from financial metrics
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if financial_metrics:
        market_cap = financial_metrics[0].market_cap
        if market_cap:
            _cache.set_market_cap(ticker, market_cap)
            return market_cap

    # NO DEMO DATA - return None when no real data found
    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
