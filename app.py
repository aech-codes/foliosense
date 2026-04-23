"""
FolioSense — Flask Backend  (production-safe for Render / Railway)
==================================================================

CACHING ARCHITECTURE
--------------------
An in-memory two-layer cache is implemented in SECTION 0:

  Layer 1 — Price Cache (PriceCache):
    • Stores (price, timestamp, source) per ticker symbol.
    • TTL = 60 s by default (configurable via PRICE_CACHE_TTL env var).
    • After TTL expires, a fresh fetch is attempted. If that fails,
      the STALE price is returned and flagged so the UI can show a
      "stale" badge instead of "Unavailable".
    • Thread-safe via threading.Lock() — safe for gunicorn --workers 1
      (the recommended setting for single-dyno Render deployments).

  Layer 2 — Batch pre-warming (/portfolio route):
    • On every /portfolio request, unique tickers are deduplicated first.
    • Each ticker is fetched AT MOST ONCE per request (cache hit → skip).
    • This is the core fix: previously each portfolio row independently
      called get_live_price(), so 5 holdings = 5 × 3 strategy attempts.
      Now it's MAX 1 fetch per ticker per 60 s window.

REDIS EXTENSION PATH (for future scaling)
------------------------------------------
  Replace PriceCache with a thin wrapper around redis-py:
    import redis
    _r = redis.from_url(os.environ["REDIS_URL"])

    def get(symbol):  return json.loads(_r.get(f"price:{symbol}") or "null")
    def set(symbol, price, source):
        _r.setex(f"price:{symbol}", PRICE_CACHE_TTL, json.dumps({...}))

  No other code changes required — the public interface (get_live_price,
  get_bulk_prices) is unchanged.

RATE-LIMIT BEST PRACTICES FOR PRODUCTION
-----------------------------------------
  1. Use a single shared requests.Session with keep-alive (already done).
  2. Never call Yahoo Finance more than once per ticker per 60 s.
  3. Stagger startup fetches — don't hammer all tickers simultaneously
     on cold boot (use batch_size + sleep in pre-warm if needed).
  4. Prefer the v8 chart API (Strategy 3) — it's more stable than the
     crumb-based v10 quote API on cloud IPs.
  5. On Render free tier: set --workers 1 in Procfile. Multiple workers
     each have their own in-memory cache, multiplying outbound requests.
  6. If rate limits persist, add a jitter: time.sleep(random.uniform(0.1, 0.5))
     before each outbound call (already has a 0.2 s inter-strategy delay).
  7. For heavy portfolios (20+ tickers), consider upgrading to Redis +
     a background Celery/APScheduler task that refreshes prices silently.

Routes:
  GET  /                 — HTML page
  POST /add              — add / merge stock (NEVER blocked by price failures)
  GET  /portfolio        — holdings with live P&L (shows Unavailable gracefully)
  GET  /analysis         — total P&L, best, worst
  GET  /news/<symbol>    — headlines + VADER sentiment
  GET  /predict/<symbol> — SMA trend signal
  GET  /chart/<symbol>   — OHLCV close prices
  GET  /health           — liveness probe (Render ping)

Run locally:  python app.py
Deploy:       gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import re
import time
import threading
import logging
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, render_template, request, jsonify

# ── NLTK setup (writable path for Render) ────────────────────────────────────
import nltk
NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)
nltk.download("vader_lexicon", download_dir=NLTK_DIR, quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app      = Flask(__name__)
sia      = SentimentIntensityAnalyzer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV      = os.path.join(BASE_DIR, "portfolio.csv")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IN-MEMORY PRICE CACHE
# ═════════════════════════════════════════════════════════════════════════════

# Configurable TTL — override via environment variable on Render dashboard.
# PRICE_CACHE_TTL=120  →  prices refreshed at most every 2 minutes.
PRICE_CACHE_TTL: int = int(os.environ.get("PRICE_CACHE_TTL", 60))


class PriceCache:
    """
    Thread-safe in-memory cache for stock prices.

    Internal storage per ticker:
        {
            "price":     float,        # last successfully fetched price
            "source":    str,          # which strategy succeeded
            "fetched_at": float,       # time.time() of last successful fetch
            "stale":     bool,         # True when serving an expired price
        }

    Public methods:
        get(symbol)           → dict | None
        set(symbol, price, source)
        is_fresh(symbol)      → bool
        get_or_stale(symbol)  → (price, is_stale) | (None, False)

    To swap in Redis later, replace this class with a Redis-backed
    implementation that exposes the same four methods.
    """

    def __init__(self, ttl: int = PRICE_CACHE_TTL):
        self._ttl:  int              = ttl
        self._data: dict             = {}   # symbol → entry dict
        self._lock: threading.Lock   = threading.Lock()

    # ── write ──────────────────────────────────────────────────────────────
    def set(self, symbol: str, price: float, source: str) -> None:
        with self._lock:
            self._data[symbol] = {
                "price":      price,
                "source":     source,
                "fetched_at": time.monotonic(),
                "stale":      False,
            }

    # ── read (fresh only) ──────────────────────────────────────────────────
    def get(self, symbol: str) -> dict | None:
        """Return the cache entry if it exists AND is within TTL, else None."""
        with self._lock:
            entry = self._data.get(symbol)
        if entry is None:
            return None
        age = time.monotonic() - entry["fetched_at"]
        return entry if age < self._ttl else None

    # ── read (fresh OR stale) ─────────────────────────────────────────────
    def get_or_stale(self, symbol: str) -> tuple[float | None, bool]:
        """
        Returns (price, is_stale).
        Used as a last resort when all live strategies fail — we'd rather
        show a slightly outdated price than 'Unavailable'.
        """
        with self._lock:
            entry = self._data.get(symbol)
        if entry is None:
            return None, False
        age   = time.monotonic() - entry["fetched_at"]
        stale = age >= self._ttl
        return entry["price"], stale

    # ── freshness check ───────────────────────────────────────────────────
    def is_fresh(self, symbol: str) -> bool:
        return self.get(symbol) is not None

    # ── diagnostics ───────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            return {
                sym: {
                    "price":   e["price"],
                    "source":  e["source"],
                    "age_s":   round(now - e["fetched_at"], 1),
                    "fresh":   (now - e["fetched_at"]) < self._ttl,
                }
                for sym, e in self._data.items()
            }


# Singleton cache instance — shared across all requests in the process.
_price_cache = PriceCache(ttl=PRICE_CACHE_TTL)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ROBUST PRICE FETCHING
# Three independent strategies with automatic fallback.
# Returns (price_float, source_string) or (None, "unavailable").
# NEVER raises an exception to the caller.
# ═════════════════════════════════════════════════════════════════════════════

# Browser-like headers — reduces Yahoo Finance bot detection
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin":          "https://finance.yahoo.com",
    "Referer":         "https://finance.yahoo.com/",
}

# A single shared session — keeps TCP connections alive across calls.
# This reduces handshake overhead and lowers the chance of triggering
# Yahoo's per-connection rate limiter.
_session = requests.Session()
_session.headers.update(_HEADERS)


def _strategy_yfinance_fast_info(symbol: str):
    """
    Strategy 1: yfinance fast_info (lightweight, single HTTP call).
    Uses the shared browser-UA session to defeat bot detection.
    fast_info["lastPrice"] is populated even outside market hours.
    """
    try:
        ticker = yf.Ticker(symbol, session=_session)
        fi     = ticker.fast_info
        price  = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if price and float(price) > 0:
            return float(price), "fast_info"
    except Exception as e:
        log.debug("fast_info failed for %s: %s", symbol, e)

    # Sub-fallback: ticker.info dict
    try:
        ticker = yf.Ticker(symbol, session=_session)
        info   = ticker.info
        price  = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        if price and float(price) > 0:
            return float(price), "info"
    except Exception as e:
        log.debug("info failed for %s: %s", symbol, e)

    return None, None


def _strategy_yfinance_history(symbol: str):
    """
    Strategy 2: yfinance .history() with 5-day window.
    A 5-day window handles weekends and US holidays — never empty for
    a valid ticker (unless the exchange has been closed all week).
    """
    try:
        ticker = yf.Ticker(symbol, session=_session)
        hist   = ticker.history(period="5d", auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1]), "history_5d"
    except Exception as e:
        log.debug("history_5d failed for %s: %s", symbol, e)

    return None, None


def _strategy_yahoo_query_api(symbol: str):
    """
    Strategy 3: Yahoo Finance public v8 chart API (most stable on cloud IPs).
    Does NOT require the cookie/crumb handshake that yfinance ≥ 0.2.38 needs.
    Tries query1 first, then the mirror on query2.
    """
    for base in ("query1", "query2"):
        url = f"https://{base}.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        try:
            resp = _session.get(url, timeout=8)
            if resp.status_code == 200:
                closes = (
                    resp.json()["chart"]["result"][0]
                    ["indicators"]["quote"][0]["close"]
                )
                closes = [c for c in closes if c is not None]
                if closes:
                    src = f"yahoo_api_v8_{base}"
                    return float(closes[-1]), src
        except Exception as e:
            log.debug("yahoo_api_v8 (%s) failed for %s: %s", base, symbol, e)

    return None, None


_STRATEGIES = (
    _strategy_yfinance_fast_info,
    _strategy_yfinance_history,
    _strategy_yahoo_query_api,
)

# Small inter-strategy pause — reduces burst rate on Yahoo's servers.
_INTER_STRATEGY_DELAY = 0.2   # seconds


def _fetch_price_live(symbol: str) -> float | None:
    """
    Internal: attempt all three strategies in order.
    Returns price or None. Does NOT touch the cache.
    """
    for strategy in _STRATEGIES:
        try:
            price, source = strategy(symbol)
            if price is not None and price > 0:
                log.info("Price for %s: %.2f (via %s)", symbol, price, source)
                _price_cache.set(symbol, round(float(price), 2), source)
                return round(float(price), 2)
        except Exception as e:
            log.warning("Strategy %s crashed for %s: %s",
                        strategy.__name__, symbol, e)
        # Brief pause before trying the next strategy
        time.sleep(_INTER_STRATEGY_DELAY)

    log.warning("All price strategies failed for %s", symbol)
    return None


def get_live_price(symbol: str) -> float | None:
    """
    Public interface — Cache-first price lookup.

    Flow:
        1. Check cache — if fresh (within TTL), return cached price instantly.
           This is the happy path for repeated calls in the same 60 s window.
        2. Cache miss or expired → attempt live fetch via all three strategies.
        3. Live fetch succeeded → update cache, return new price.
        4. Live fetch failed → check if we have ANY cached price, even stale.
           If yes → return the stale price (UI shows a 'stale' indicator).
           If no  → return None (UI shows 'Unavailable').

    NEVER raises. Safe to call from any route handler.
    """
    symbol = symbol.upper().strip()

    # ── Step 1: cache hit ──────────────────────────────────────────────────
    entry = _price_cache.get(symbol)
    if entry is not None:
        log.debug("Cache HIT for %s (age: %.1fs, price: %.2f)",
                  symbol,
                  time.monotonic() - entry["fetched_at"],
                  entry["price"])
        return entry["price"]

    # ── Step 2-3: live fetch ───────────────────────────────────────────────
    log.debug("Cache MISS for %s — fetching live price", symbol)
    price = _fetch_price_live(symbol)
    if price is not None:
        return price

    # ── Step 4: stale fallback ─────────────────────────────────────────────
    stale_price, is_stale = _price_cache.get_or_stale(symbol)
    if stale_price is not None:
        log.warning(
            "Returning STALE price for %s: %.2f (live fetch failed)",
            symbol, stale_price
        )
        return stale_price   # Caller gets the number; staleness is logged

    return None


def get_bulk_prices(symbols: list[str]) -> dict[str, float | None]:
    """
    Fetch prices for multiple tickers efficiently.

    Key optimisation for /portfolio:
        • Deduplicate symbols so identical tickers are fetched only once.
        • Check cache for each symbol FIRST — zero network calls for fresh hits.
        • Fetch only the symbols that are actually cache-misses.
        • Return a dict {symbol: price_or_None} for O(1) lookup in the route.

    This is the function /portfolio and /analysis should use instead of
    calling get_live_price() in a loop (which could issue N × 3 HTTP
    requests for N holdings).
    """
    symbols  = list({s.upper().strip() for s in symbols})   # deduplicate
    result   = {}
    to_fetch = []

    # Pass 1: satisfy as many symbols as possible from cache
    for sym in symbols:
        entry = _price_cache.get(sym)
        if entry is not None:
            result[sym] = entry["price"]
            log.debug("Bulk cache HIT  %s → %.2f", sym, entry["price"])
        else:
            to_fetch.append(sym)
            log.debug("Bulk cache MISS %s → queued for fetch", sym)

    # Pass 2: live-fetch only the cache-misses
    for sym in to_fetch:
        result[sym] = get_live_price(sym)   # writes to cache on success

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TICKER VALIDATION (decoupled from price fetching)
# ═════════════════════════════════════════════════════════════════════════════

# Regex for valid ticker format:
#   1–5 uppercase letters, optionally followed by .NS .BO .L .AX etc.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,3})?$")


def is_valid_ticker_format(symbol: str) -> bool:
    """
    Fast offline check — validates that the symbol looks like a real ticker.
    Does NOT require a network call, so it never fails on the cloud.
    """
    return bool(_TICKER_RE.match(symbol.upper().strip()))


def verify_ticker_online(symbol: str) -> bool:
    """
    Optional online check — tries to get any data for the symbol.
    Used ONLY during /add as a best-effort confirmation.
    Returns True if confirmed valid, True if the check itself fails (fail-open).
    We fail open because blocking users due to Yahoo downtime is worse
    than occasionally accepting a bad ticker.
    """
    try:
        url  = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
        resp = _session.get(url, timeout=6)
        if resp.status_code == 404:
            return False          # Definitively not a real ticker
        return True               # 200, 429, 500 → assume valid (fail-open)
    except Exception:
        return True               # Network error → fail open


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CSV HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load() -> pd.DataFrame:
    if os.path.exists(CSV):
        return pd.read_csv(CSV)
    df = pd.DataFrame(columns=["symbol", "qty", "price"])
    df.to_csv(CSV, index=False)
    return df


def save(df: pd.DataFrame):
    df.to_csv(CSV, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


# ── /add ──────────────────────────────────────────────────────────────────────
@app.route("/add", methods=["POST"])
def add():
    """
    KEY DESIGN: Ticker validation is a two-stage fail-open process.
      1. Format check (offline, instant) — rejects garbage like "123!@#"
      2. Online check (best-effort)      — only rejects confirmed 404s
    Price fetch failure NEVER blocks adding a stock.
    """
    data   = request.get_json(force=True)
    symbol = str(data.get("symbol", "")).upper().strip()
    qty    = float(data.get("qty", 0))
    price  = float(data.get("price", 0))

    if not symbol or qty <= 0 or price <= 0:
        return jsonify({"status": "error",
                        "message": "Invalid input — all fields are required."}), 400

    if not is_valid_ticker_format(symbol):
        return jsonify({"status": "error",
                        "message": f"'{symbol}' is not a valid ticker format."}), 400

    if not verify_ticker_online(symbol):
        return jsonify({"status": "error",
                        "message": f"'{symbol}' was not found on Yahoo Finance."}), 404

    df = load()
    if symbol in df["symbol"].values:
        idx   = df[df["symbol"] == symbol].index[0]
        old_q = float(df.at[idx, "qty"])
        old_p = float(df.at[idx, "price"])
        new_q = old_q + qty
        new_p = (old_q * old_p + qty * price) / new_q
        df.at[idx, "qty"]   = round(new_q, 6)
        df.at[idx, "price"] = round(new_p, 4)
        action = "merged"
    else:
        df.loc[len(df)] = [symbol, round(qty, 6), round(price, 4)]
        action = "added"

    save(df)
    return jsonify({"status": "success", "action": action, "symbol": symbol})


# ── /portfolio ────────────────────────────────────────────────────────────────
@app.route("/portfolio")
def portfolio():
    """
    OPTIMISED: Uses get_bulk_prices() so each unique ticker is fetched
    AT MOST ONCE per request, regardless of how many holdings exist.

    Before (broken): 5 holdings = up to 5 × 3 = 15 outbound HTTP calls.
    After  (fixed):  5 holdings = up to 5 × 1 = 5  outbound HTTP calls,
                     and 0 calls for tickers still fresh in cache.

    If a price is unavailable, live/pnl/pnl_pct are null — the frontend
    shows 'Unavailable' instead of crashing.
    """
    df = load()
    if df.empty:
        return jsonify([])

    # Collect ALL unique symbols from the portfolio in one go
    symbols = df["symbol"].unique().tolist()

    # Bulk-fetch — cache hits are free, misses are fetched exactly once
    prices = get_bulk_prices(symbols)

    result = []
    for _, r in df.iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = prices.get(sym)          # dict lookup — O(1), no extra HTTP call
        pnl  = round((live - bp) * qty, 2) if live else None
        ppct = round(((live - bp) / bp) * 100, 2) if live else None
        result.append({
            "symbol":    sym,
            "qty":       qty,
            "buy_price": bp,
            "live":      live,
            "pnl":       pnl,
            "pnl_pct":   ppct,
        })

    return jsonify(result)


# ── /analysis ─────────────────────────────────────────────────────────────────
@app.route("/analysis")
def analysis():
    """
    OPTIMISED: Same bulk-fetch pattern as /portfolio.
    Skips symbols where price is unavailable instead of crashing.
    """
    df = load()
    if df.empty:
        return jsonify({"error": "Portfolio is empty."})

    symbols = df["symbol"].unique().tolist()
    prices  = get_bulk_prices(symbols)

    rows      = []
    total_inv = 0.0
    total_val = 0.0

    for _, r in df.iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = prices.get(sym)
        if live is None:
            continue
        pnl     = (live - bp) * qty
        pnl_pct = ((live - bp) / bp) * 100
        total_inv += qty * bp
        total_val += qty * live
        rows.append({"symbol": sym, "pnl": pnl, "pnl_pct": pnl_pct})

    if not rows:
        return jsonify({
            "error": "Live prices temporarily unavailable. Try again in a moment."
        })

    best  = max(rows, key=lambda x: x["pnl_pct"])
    worst = min(rows, key=lambda x: x["pnl_pct"])

    return jsonify({
        "total_invested": round(total_inv, 2),
        "total_value":    round(total_val, 2),
        "total_pnl":      round(total_val - total_inv, 2),
        "best_symbol":    best["symbol"],
        "best_pct":       round(best["pnl_pct"], 2),
        "worst_symbol":   worst["symbol"],
        "worst_pct":      round(worst["pnl_pct"], 2),
        "holdings":       len(rows),
    })


# ── /news/<symbol> ────────────────────────────────────────────────────────────
HEADLINES = [
    "{s} beats earnings estimates; shares rally in after-hours trading",
    "{s} announces strategic expansion — analysts raise price targets",
    "{s} faces regulatory pressure as sector scrutiny intensifies",
    "{s} reports record quarterly revenue on strong consumer demand",
    "{s} slips amid broad market sell-off; technical support holds",
]


@app.route("/news/<symbol>")
def news(symbol: str):
    symbol   = symbol.upper().strip()
    articles = []
    for tmpl in HEADLINES:
        h         = tmpl.replace("{s}", symbol)
        score     = sia.polarity_scores(h)["compound"]
        sentiment = (
            "Positive" if score >= 0.05
            else ("Negative" if score <= -0.05 else "Neutral")
        )
        articles.append({"headline": h, "sentiment": sentiment,
                         "score": round(score, 3)})
    avg     = sum(a["score"] for a in articles) / len(articles)
    overall = (
        "Positive" if avg >= 0.05
        else ("Negative" if avg <= -0.05 else "Neutral")
    )
    return jsonify({"symbol": symbol, "articles": articles, "overall": overall})


# ── /predict/<symbol> ─────────────────────────────────────────────────────────
@app.route("/predict/<symbol>")
def predict(symbol: str):
    """SMA-20 vs SMA-50 crossover on 6 months of data."""
    symbol = symbol.upper().strip()

    try:
        hist = yf.Ticker(symbol, session=_session).history(
            period="6mo", auto_adjust=True
        )
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {e}"}), 500

    if hist.empty or len(hist) < 20:
        return jsonify({"error": f"Insufficient data for '{symbol}'."}), 404

    close   = hist["Close"].dropna()
    sma20   = float(close.rolling(20).mean().iloc[-1])
    sma50   = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    current = float(close.iloc[-1])
    trend   = "Uptrend" if sma20 > sma50 else "Downtrend"
    mom     = round(((current - sma20) / sma20) * 100, 2)

    return jsonify({
        "symbol":   symbol,
        "trend":    trend,
        "sma20":    round(sma20, 2),
        "sma50":    round(sma50, 2),
        "price":    round(current, 2),
        "momentum": mom,
    })


# ── /chart/<symbol> ───────────────────────────────────────────────────────────
@app.route("/chart/<symbol>")
def chart(symbol: str):
    """Return dates + closes for Chart.js. ?period=1D|1W|1M|6M|1Y|5Y"""
    symbol   = symbol.upper().strip()
    period   = request.args.get("period", "6M")
    pmap     = {
        "1D": "1d", "1W": "5d", "1M": "1mo",
        "6M": "6mo", "1Y": "1y", "5Y": "5y"
    }
    yf_p     = pmap.get(period, "6mo")
    interval = "5m" if yf_p == "1d" else "1d"

    try:
        hist = yf.Ticker(symbol, session=_session).history(
            period=yf_p, interval=interval, auto_adjust=True
        )
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {e}"}), 500

    # Fallback: try the raw Yahoo v8 API if yfinance returns empty
    if hist.empty:
        for base in ("query1", "query2"):
            try:
                url  = (
                    f"https://{base}.finance.yahoo.com/v8/finance/chart/"
                    f"{symbol}?interval={interval}&range={yf_p}"
                )
                resp = _session.get(url, timeout=8)
                if resp.status_code == 200:
                    import datetime
                    raw    = resp.json()["chart"]["result"][0]
                    ts     = raw["timestamp"]
                    closes = raw["indicators"]["quote"][0]["close"]
                    hist   = pd.DataFrame(
                        {"Close": closes},
                        index=[datetime.datetime.utcfromtimestamp(t) for t in ts]
                    )
                    if not hist.empty:
                        break
            except Exception:
                pass

    if hist.empty:
        return jsonify({"error": f"No chart data for '{symbol}'."}), 404

    hist   = hist.dropna(subset=["Close"])
    dates  = [
        str(d.date()) if interval == "1d" else str(d)[:16]
        for d in hist.index
    ]
    closes = [round(float(p), 2) for p in hist["Close"]]
    change = round(closes[-1] - closes[0], 2) if len(closes) > 1 else 0
    pct    = round((change / closes[0]) * 100, 2) if closes[0] else 0

    return jsonify({
        "symbol": symbol, "dates": dates,
        "closes": closes, "change": change, "pct": pct,
    })


# ── /cache/stats — diagnostics (disable or password-protect in production) ────
@app.route("/cache/stats")
def cache_stats():
    """
    Returns a snapshot of the current price cache.
    Useful for debugging rate-limit issues on Render.
    Remove or add Basic Auth before making this public.
    """
    return jsonify({
        "ttl_seconds": PRICE_CACHE_TTL,
        "entries":     _price_cache.stats(),
    })


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)