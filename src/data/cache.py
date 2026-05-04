import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class Cache:
    """Persistent disk cache for API responses with in-memory fallback."""
    
    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Normalize ticker for consistent cache keys.
        Converts dot format to hyphen format (BRK.B -> BRK-B) so that
        different ticker formats from different APIs map to the same cache entry.
        """
        upper = ticker.upper()
        # Common multi-class stock tickers
        dot_to_hyphen = {
            'BRK.B': 'BRK-B', 'BRK.A': 'BRK-A',
            'BF.B': 'BF-B', 'BF.A': 'BF-A',
            'MOG.A': 'MOG-A', 'MOG.B': 'MOG-B',
            'HEI.A': 'HEI-A', 'CWEN.A': 'CWEN-A',
        }
        if upper in dot_to_hyphen:
            return dot_to_hyphen[upper]
        # General fallback: any ticker with .A or .B suffix
        if upper.endswith('.A') or upper.endswith('.B'):
            return upper.replace('.', '-')
        return upper
    
    def __init__(self, cache_dir: str = None):
        # Set up cache directory - default to ~/.cache/ai-hedge-fund
        if cache_dir is None:
            home = Path.home()
            cache_dir = home / ".cache" / "ai-hedge-fund"
        else:
            cache_dir = Path(cache_dir)
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # TTL settings for time-sensitive data (hours)
        self.news_ttl_hours = 12           # News expires after 12 hours (new news every day)
        self.insider_ttl_hours = 24        # Insider trades expire after 24 hours
        self.prices_ttl_hours = 24         # Price data updates daily after market close
        self.financials_ttl_days = 7       # Financial metrics update quarterly, check weekly
        self.market_cap_ttl_hours = 24     # Market cap changes daily
        
        # In-memory cache for fast lookups
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}
        self._market_cap_cache: dict[str, float] = {}
        
        # Load existing cached data from disk
        self._load_from_disk()
    
    def _get_cache_file(self, data_type: str, ticker: str) -> Path:
        """Get the cache file path for a specific data type and ticker."""
        return self.cache_dir / f"{ticker.upper()}_{data_type}.json"
    
    def _load_from_disk(self):
        """Load all cached data from disk on startup."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    # Parse filename like "AAPL_prices.json"
                    parts = cache_file.stem.split("_")
                    if len(parts) >= 2:
                        ticker = parts[0]
                        data_type = "_".join(parts[1:])
                        
                        with open(cache_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        # Load into appropriate in-memory cache
                        if data_type == "prices":
                            self._prices_cache[ticker] = data
                        elif data_type == "financial_metrics":
                            self._financial_metrics_cache[ticker] = data
                        elif data_type == "line_items":
                            self._line_items_cache[ticker] = data
                        elif data_type == "insider_trades":
                            self._insider_trades_cache[ticker] = data
                        elif data_type == "company_news":
                            self._company_news_cache[ticker] = data
                        elif data_type == "market_cap":
                            # Market cap is stored as {"value": float}
                            self._market_cap_cache[ticker] = data.get("value", 0)
                except Exception as e:
                    print(f"Warning: Could not load cache file {cache_file}: {e}")
        except Exception as e:
            print(f"Warning: Could not load cache from disk: {e}")
    
    def _save_to_disk(self, data_type: str, ticker: str, data: list[dict] | dict):
        """Save data to persistent disk cache."""
        try:
            cache_file = self._get_cache_file(data_type, ticker)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save cache to disk: {e}")
    
    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, combining fields for matching keys.
        
        When multiple agents request different line_item fields for the same
        ticker and report_period, we need to MERGE fields into the same record
        (not discard the second agent's data). For example:
        - Agent 1 caches: {report_period: "2025-12-31", revenue: 5B, net_income: -1B}
        - Agent 2 caches: {report_period: "2025-12-31", total_assets: 49B, current_assets: 7B}
        - Merged result:  {report_period: "2025-12-31", revenue: 5B, net_income: -1B, total_assets: 49B, current_assets: 7B}
        """
        if not existing:
            return new_data

        # Build a lookup dict for existing items by key_field
        existing_by_key = {item[key_field]: i for i, item in enumerate(existing)}
        merged = existing.copy()

        for new_item in new_data:
            key_val = new_item[key_field]
            if key_val in existing_by_key:
                # MERGE: add new fields to existing record (don't overwrite existing non-None values)
                idx = existing_by_key[key_val]
                for field, value in new_item.items():
                    if field == key_field:
                        continue
                    # Only add field if it doesn't exist in existing record OR existing is None
                    if field not in merged[idx] or merged[idx][field] is None:
                        merged[idx][field] = value
            else:
                # New record - add it
                merged.append(new_item)

        return merged
    
    def _filter_by_date_range(self, data: list[dict], start_date: str, end_date: str, date_field: str) -> list[dict]:
        """Filter cached data to only include entries within the requested date range."""
        if not data:
            return []
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        filtered = []
        for item in data:
            item_date_str = item.get(date_field, "").split("T")[0]
            if not item_date_str:
                continue
            try:
                item_date = datetime.strptime(item_date_str, "%Y-%m-%d").date()
                if start_dt <= item_date <= end_dt:
                    filtered.append(item)
            except ValueError:
                continue
        
        return filtered
    
    def _has_sufficient_data(self, cached: list[dict], start_date: str, end_date: str, date_field: str) -> bool:
        """Check if cached data fully covers the requested date range."""
        if not cached:
            return False
        
        cached_dates = []
        for item in cached:
            date_str = item.get(date_field, "").split("T")[0]
            if date_str:
                try:
                    cached_dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                except ValueError:
                    continue
        
        if not cached_dates:
            return False
        
        min_cached = min(cached_dates)
        max_cached = max(cached_dates)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Consider sufficient if cached range covers 90% of requested range
        return min_cached <= start_dt and max_cached >= end_dt

    def get_prices(self, ticker: str, start_date: str = None, end_date: str = None) -> list[dict[str, any]] | None:
        """Get cached price data, optionally filtered by date range."""
        key = self._normalize_ticker(ticker)
        cached = self._prices_cache.get(key)
        if cached and start_date and end_date:
            return self._filter_by_date_range(cached, start_date, end_date, "time")
        return cached
    
    def has_sufficient_prices(self, ticker: str, start_date: str, end_date: str) -> bool:
        """Check if we have enough cached price data AND cache is fresh."""
        key = self._normalize_ticker(ticker)
        # Check if cache is fresh enough (24 hours)
        if not self._is_cache_fresh(key, "prices", self.prices_ttl_hours):
            return False
        
        cached = self._prices_cache.get(key)
        return self._has_sufficient_data(cached, start_date, end_date, "time")

    def set_prices(self, ticker: str, data: list[dict[str, any]]):
        """Append new price data to cache (both in-memory and disk)."""
        key = self._normalize_ticker(ticker)
        merged = self._merge_data(self._prices_cache.get(key), data, key_field="time")
        self._prices_cache[key] = merged
        self._save_to_disk("prices", key, merged)

    def get_financial_metrics(self, ticker: str) -> list[dict[str, any]]:
        """Get cached financial metrics if available."""
        return self._financial_metrics_cache.get(self._normalize_ticker(ticker))
    
    def has_sufficient_financial_metrics(self, ticker: str) -> bool:
        """Check if we have cached financial metrics AND cache is fresh."""
        key = self._normalize_ticker(ticker)
        # Check if cache is fresh enough (7 days)
        if not self._is_cache_fresh(key, "financial_metrics", self.financials_ttl_days * 24):
            return False
        
        cached = self._financial_metrics_cache.get(key)
        return cached is not None and len(cached) > 0

    def set_financial_metrics(self, ticker: str, data: list[dict[str, any]]):
        """Append new financial metrics to cache (both in-memory and disk)."""
        key = self._normalize_ticker(ticker)
        merged = self._merge_data(self._financial_metrics_cache.get(key), data, key_field="report_period")
        self._financial_metrics_cache[key] = merged
        self._save_to_disk("financial_metrics", key, merged)

    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached line items if available."""
        return self._line_items_cache.get(self._normalize_ticker(ticker))
    
    def has_sufficient_line_items(self, ticker: str) -> bool:
        """Check if we have cached line items AND cache is fresh."""
        key = self._normalize_ticker(ticker)
        # Check if cache is fresh enough (7 days)
        if not self._is_cache_fresh(key, "line_items", self.financials_ttl_days * 24):
            return False
        
        cached = self._line_items_cache.get(key)
        return cached is not None and len(cached) > 0

    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        """Append new line items to cache (both in-memory and disk)."""
        key = self._normalize_ticker(ticker)
        merged = self._merge_data(self._line_items_cache.get(key), data, key_field="report_period")
        self._line_items_cache[key] = merged
        self._save_to_disk("line_items", key, merged)

    def get_insider_trades(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached insider trades if available."""
        return self._insider_trades_cache.get(self._normalize_ticker(ticker))
    
    def has_sufficient_insider_trades(self, ticker: str, days: int = 90) -> bool:
        """Check if we have recent enough insider trades AND cache is fresh."""
        key = self._normalize_ticker(ticker)
        # First check if cache is fresh enough
        if not self._is_cache_fresh(key, "insider_trades", self.insider_ttl_hours):
            return False
        
        cached = self._insider_trades_cache.get(key)
        if not cached or len(cached) == 0:
            return False
        
        # Check if most recent trade is within the last N days
        try:
            recent_dates = []
            for item in cached:
                date_str = item.get("filing_date", "").split("T")[0]
                if date_str:
                    recent_dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
            
            if recent_dates:
                most_recent = max(recent_dates)
                cutoff = datetime.now().date() - timedelta(days=days)
                return most_recent >= cutoff
        except Exception:
            pass
        
        return False

    def set_insider_trades(self, ticker: str, data: list[dict[str, any]]):
        """Append new insider trades to cache (both in-memory and disk)."""
        key = self._normalize_ticker(ticker)
        merged = self._merge_data(self._insider_trades_cache.get(key), data, key_field="filing_date")
        self._insider_trades_cache[key] = merged
        self._save_to_disk("insider_trades", key, merged)

    def get_company_news(self, ticker: str, start_date: str = None, end_date: str = None) -> list[dict[str, any]] | None:
        """Get cached company news, optionally filtered by date range."""
        key = self._normalize_ticker(ticker)
        cached = self._company_news_cache.get(key)
        if cached and start_date and end_date:
            return self._filter_by_date_range(cached, start_date, end_date, "date")
        return cached
    
    def _is_cache_fresh(self, ticker: str, data_type: str, ttl_hours: int) -> bool:
        """Check if cache file was modified recently enough."""
        cache_file = self._get_cache_file(data_type, ticker)
        if not cache_file.exists():
            return False
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            return age_hours < ttl_hours
        except:
            return False

    def has_sufficient_company_news(self, ticker: str, start_date: str, end_date: str, min_count: int = 5) -> bool:
        """Check if we have enough cached news for the date range AND it's fresh."""
        key = self._normalize_ticker(ticker)
        # First check if cache is fresh enough
        if not self._is_cache_fresh(key, "company_news", self.news_ttl_hours):
            return False
        cached = self.get_company_news(key, start_date, end_date)
        return cached is not None and len(cached) >= min_count

    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        """Append new company news to cache (both in-memory and disk)."""
        key = self._normalize_ticker(ticker)
        merged = self._merge_data(self._company_news_cache.get(key), data, key_field="date")
        self._company_news_cache[key] = merged
        self._save_to_disk("company_news", key, merged)

    def get_market_cap(self, ticker: str) -> float | None:
        """Get cached market cap value - returns None if cache is stale (over 24 hours)."""
        key = self._normalize_ticker(ticker)
        # Check if cache is fresh enough (24 hours)
        if not self._is_cache_fresh(key, "market_cap", self.market_cap_ttl_hours):
            return None
        return self._market_cap_cache.get(key)

    def set_market_cap(self, ticker: str, value: float):
        """Set cached market cap value (persisted permanently)."""
        key = self._normalize_ticker(ticker)
        self._market_cap_cache[key] = value
        # Save as dict for consistency with other cache files
        self._save_to_disk("market_cap", key, {"value": value})


# Global cache instance
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
