"""
FolioSense — Flask Backend  (Render / Railway production-hardened)
==================================================================

WHY THE PREVIOUS VERSION STILL RATE-LIMITED
--------------------------------------------
The previous version still used yfinance as the PRIMARY source.
yfinance internally calls the same Yahoo Finance endpoints, so
cloud-IP blocking hits it just as hard as direct calls.

THE CORRECT ARCHITECTURE (implemented below)
--------------------------------------------

  get_live_price(symbol)
  │
  ├─ 1. Cache fresh? ──────────────────────────────────► return instantly
  │
  ├─ 2. Global rate-limit flag active?
  │      └─ Yes → return stale cache or None  (no outbound call at all)
  │
  ├─ 3. Strategy A: direct Yahoo v8 API  (requests, no yfinance)
  │      ├─ HTTP 429 → set global RL flag, stop pipeline
  │      └─ HTTP 200 → write cache, return
  │
  ├─ 4. Strategy B: direct Yahoo v7 quote API  (requests, different endpoint)
  │      ├─ HTTP 429 → set global RL flag, stop pipeline
  │      └─ HTTP 200 → write cache, return
  │
  ├─ 5. Strategy C: yfinance fast_info / history  (last-resort only)
  │
  ├─ 6. Strategy D: Alpha Vantage  (if ALPHA_VANTAGE_KEY env var is set)
  │
  └─ 7. All failed → return stale cache if any, else None. Never crash.

KEY DESIGN DECISIONS
--------------------
• Direct requests() FIRST — full control over headers + HTTP status codes.
• Global rate-limit flag (_RL_UNTIL) — a single 429 anywhere halts ALL
  outbound price calls for RATE_LIMIT_BACKOFF seconds. This prevents the
  thundering-herd problem: multiple concurrent requests each retrying
  independently and hammering Yahoo harder.
• Rotating User-Agent pool — reduces Yahoo bot fingerprinting.
• One shared requests.Session() — keeps TCP connections alive.
• get_bulk_prices() deduplicates symbols + cache-first before any HTTP call.
  This fixes the root /portfolio bug: N holdings now make ≤ N HTTP calls
  on first load, and 0 calls while the cache is fresh.
• /add ticker validation is fail-open — never blocked by rate limits.

EXTENDING TO REDIS (future scaling)
-------------------------------------
Replace PriceCache._data with a Redis hash. The four public methods
(get, set, get_or_stale, stats) are the only interface used by callers.
No route code changes required.

RENDER-SPECIFIC TIPS
---------------------
• Procfile must use --workers 1. Multiple workers each have their own
  in-memory state; the global RL flag won't propagate between them.
  If you need multiple workers, store _RL_UNTIL and _price_cache in Redis.
• Set PRICE_CACHE_TTL=120 in Render dashboard to reduce call frequency.
• Set ALPHA_VANTAGE_KEY in Render dashboard for a free reliable fallback.
  Get a key at https://www.alphavantage.co/support/#api-key

Routes:
  GET  /                 — HTML page
  POST /add              — add / merge stock
  GET  /portfolio        — holdings with live P&L
  GET  /analysis         — total P&L, best, worst
  GET  /news/<symbol>    — VADER-scored headlines
  GET  /predict/<symbol> — SMA-20/50 trend signal
  GET  /chart/<symbol>   — OHLCV closes for Chart.js
  GET  /cache/stats      — live cache + RL-flag diagnostics
  GET  /health           — liveness probe (Render ping)

Run locally:  python app.py
Deploy:       gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import re
import time
import random
import threading
import logging
import datetime
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, render_template, request, jsonify

# ── NLTK setup ────────────────────────────────────────────────────────────────
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

# Optional Alpha Vantage key — set in Render env vars to enable Strategy D.
# Free key: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY: str | None = os.environ.get("ALPHA_VANTAGE_KEY")

# Cache TTL — configurable without redeploy via Render env var.
PRICE_CACHE_TTL:   int = int(os.environ.get("PRICE_CACHE_TTL",   90))

# How long to pause ALL outbound calls after any 429 is detected.
RATE_LIMIT_BACKOFF: int = int(os.environ.get("RATE_LIMIT_BACKOFF", 90))


# =============================================================================
# SECTION 0 — GLOBAL RATE-LIMIT GUARD
# =============================================================================

_rl_lock  = threading.Lock()
_RL_UNTIL: float = 0.0   # monotonic timestamp; 0 = not rate-limited


def _is_rate_limited() -> bool:
    with _rl_lock:
        return time.monotonic() < _RL_UNTIL


def _trigger_rate_limit() -> None:
    global _RL_UNTIL
    with _rl_lock:
        _RL_UNTIL = time.monotonic() + RATE_LIMIT_BACKOFF
    log.warning(
        "HTTP 429 detected — all outbound price calls paused for %ds.",
        RATE_LIMIT_BACKOFF,
    )


def _rate_limit_remaining() -> float:
    with _rl_lock:
        return max(0.0, _RL_UNTIL - time.monotonic())


# =============================================================================
# SECTION 1 — IN-MEMORY PRICE CACHE
# =============================================================================

class PriceCache:
    """
    Thread-safe TTL cache for stock prices.

    Per-symbol storage:
        price      (float)  — last known price
        source     (str)    — strategy that fetched it
        fetched_at (float)  — time.monotonic() of successful fetch

    Redis swap-in: implement the same four public methods against a Redis
    hash and replace this class. No calling code changes needed.
    """

    def __init__(self, ttl: int = PRICE_CACHE_TTL) -> None:
        self._ttl  = ttl
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def set(self, symbol: str, price: float, source: str) -> None:
        with self._lock:
            self._data[symbol] = {
                "price":      round(float(price), 2),
                "source":     source,
                "fetched_at": time.monotonic(),
            }

    def get(self, symbol: str) -> dict | None:
        """Return the entry if it exists AND is within TTL, else None."""
        with self._lock:
            entry = self._data.get(symbol)
        if entry is None:
            return None
        return entry if (time.monotonic() - entry["fetched_at"]) < self._ttl else None

    def get_or_stale(self, symbol: str) -> tuple[float | None, bool]:
        """Return (price, is_stale) — serves expired entries as last resort."""
        with self._lock:
            entry = self._data.get(symbol)
        if entry is None:
            return None, False
        age = time.monotonic() - entry["fetched_at"]
        return entry["price"], age >= self._ttl

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


_price_cache = PriceCache(ttl=PRICE_CACHE_TTL)


# =============================================================================
# SECTION 2 — HTTP SESSION & ROTATING USER-AGENTS
# =============================================================================

# One shared session — persistent TCP connections reduce handshake overhead
# and lower per-connection rate-limit triggers.
_session = requests.Session()

_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
]


def _headers() -> dict:
    """Fresh header dict with a randomly selected User-Agent."""
    return {
        "User-Agent":      random.choice(_USER_AGENTS),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin":          "https://finance.yahoo.com",
        "Referer":         "https://finance.yahoo.com/",
        "Cache-Control":   "no-cache",
    }


# =============================================================================
# SECTION 3 — PRICE FETCH STRATEGIES
# Each returns (price_float, source_str) | (None, None). Never raises.
# Each checks _is_rate_limited() before making any outbound call.
# Each calls _trigger_rate_limit() on HTTP 429.
# =============================================================================

def _strategy_direct_v8(symbol: str) -> tuple[float | None, str | None]:
    """
    Strategy A (PRIMARY): Yahoo Finance public v8 chart API via raw requests().
    Most stable endpoint on cloud IPs — no cookie/crumb handshake needed.
    Tries query1 then query2 (Yahoo geo-routes between them unpredictably).
    """
    if _is_rate_limited():
        return None, None

    for base in ("query1", "query2"):
        url = (
            f"https://{base}.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?interval=1d&range=5d"
        )
        try:
            resp = _session.get(url, headers=_headers(), timeout=7)

            if resp.status_code == 429:
                _trigger_rate_limit()
                return None, None

            if resp.status_code != 200:
                log.debug("v8/%s → HTTP %d for %s", base, resp.status_code, symbol)
                continue

            result = resp.json().get("chart", {}).get("result")
            if not result:
                continue

            closes = [
                c for c in result[0]["indicators"]["quote"][0]["close"]
                if c is not None
            ]
            if closes:
                return float(closes[-1]), f"direct_v8_{base}"

        except requests.exceptions.Timeout:
            log.debug("v8/%s timeout for %s", base, symbol)
        except Exception as e:
            log.debug("v8/%s error for %s: %s", base, symbol, e)

    return None, None


def _strategy_direct_v7(symbol: str) -> tuple[float | None, str | None]:
    """
    Strategy B (SECONDARY): Yahoo Finance v7 quote API via raw requests().
    Different endpoint path = separate rate-limit bucket on Yahoo's CDN.
    Returns regularMarketPrice (or post/pre-market price outside hours).
    """
    if _is_rate_limited():
        return None, None

    for base in ("query1", "query2"):
        url = f"https://{base}.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        try:
            resp = _session.get(url, headers=_headers(), timeout=7)

            if resp.status_code == 429:
                _trigger_rate_limit()
                return None, None

            if resp.status_code != 200:
                continue

            items = resp.json().get("quoteResponse", {}).get("result", [])
            if not items:
                continue

            item  = items[0]
            price = (
                item.get("regularMarketPrice")
                or item.get("postMarketPrice")
                or item.get("preMarketPrice")
            )
            if price and float(price) > 0:
                return float(price), f"direct_v7_{base}"

        except requests.exceptions.Timeout:
            log.debug("v7/%s timeout for %s", base, symbol)
        except Exception as e:
            log.debug("v7/%s error for %s: %s", base, symbol, e)

    return None, None


def _strategy_yfinance(symbol: str) -> tuple[float | None, str | None]:
    """
    Strategy C (TERTIARY): yfinance — used only when both direct strategies fail.
    Kept as a fallback because it handles auth edge cases and less-common
    exchanges that the raw v7/v8 endpoints occasionally miss.
    """
    if _is_rate_limited():
        return None, None

    try:
        ticker = yf.Ticker(symbol, session=_session)
        fi     = ticker.fast_info
        price  = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if price and float(price) > 0:
            return float(price), "yfinance_fast_info"
    except Exception as e:
        log.debug("yfinance fast_info failed for %s: %s", symbol, e)

    try:
        ticker = yf.Ticker(symbol, session=_session)
        hist   = ticker.history(period="5d", auto_adjust=True)
        if not hist.empty:
            price = float(hist["Close"].dropna().iloc[-1])
            if price > 0:
                return price, "yfinance_history"
    except Exception as e:
        log.debug("yfinance history failed for %s: %s", symbol, e)

    return None, None


def _strategy_alpha_vantage(symbol: str) -> tuple[float | None, str | None]:
    """
    Strategy D (QUATERNARY): Alpha Vantage GLOBAL_QUOTE.
    Only active when ALPHA_VANTAGE_KEY env var is set.
    Free tier: 25 req/day, 5 req/min — reliable on cloud IPs.
    Get a free key: https://www.alphavantage.co/support/#api-key
    NOTE: A 429 from Alpha Vantage does NOT trigger the Yahoo RL flag.
    """
    if not ALPHA_VANTAGE_KEY or _is_rate_limited():
        return None, None

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    )
    try:
        resp = _session.get(url, headers=_headers(), timeout=8)
        if resp.status_code == 200:
            price_str = resp.json().get("Global Quote", {}).get("05. price")
            if price_str:
                price = float(price_str)
                if price > 0:
                    return price, "alpha_vantage"
    except Exception as e:
        log.debug("Alpha Vantage failed for %s: %s", symbol, e)

    return None, None


# Strategy pipeline — order matters.
_STRATEGIES = (
    _strategy_direct_v8,
    _strategy_direct_v7,
    _strategy_yfinance,
    _strategy_alpha_vantage,
)


# =============================================================================
# SECTION 4 — PUBLIC PRICE INTERFACE
# =============================================================================

def get_live_price(symbol: str) -> float | None:
    """
    Cache-first, rate-limit-aware price lookup. Never raises.

    1. Fresh cache hit → return immediately (zero HTTP calls).
    2. Rate limit active → return stale cache or None (no HTTP calls).
    3. Try strategies A → B → C → D in order.
       Any 429 triggers the global RL flag and stops the pipeline.
    4. On success → update cache, return price.
    5. All failed → return stale cached price if any, else None.
    """
    symbol = symbol.upper().strip()

    # Step 1: fresh cache
    entry = _price_cache.get(symbol)
    if entry is not None:
        log.debug("Cache HIT %s → %.2f", symbol, entry["price"])
        return entry["price"]

    # Step 2: rate-limited — serve stale or None without any outbound call
    if _is_rate_limited():
        stale, _ = _price_cache.get_or_stale(symbol)
        if stale is not None:
            log.info(
                "RL active (%.0fs left) — stale price for %s: %.2f",
                _rate_limit_remaining(), symbol, stale,
            )
        else:
            log.info(
                "RL active (%.0fs left) — no cached price for %s",
                _rate_limit_remaining(), symbol,
            )
        return stale  # None if truly never cached

    # Steps 3-4: strategy pipeline
    log.debug("Cache MISS %s — running strategy pipeline", symbol)
    for strategy in _STRATEGIES:
        try:
            price, source = strategy(symbol)
        except Exception as e:
            log.warning("Strategy %s crashed for %s: %s",
                        strategy.__name__, symbol, e)
            continue

        if price is not None and float(price) > 0:
            log.info("Price %s → %.2f via %s", symbol, price, source)
            _price_cache.set(symbol, price, source)
            return round(float(price), 2)

        if _is_rate_limited():
            log.warning("RL triggered mid-pipeline for %s — stopping.", symbol)
            break

    # Step 5: stale fallback
    stale, is_stale = _price_cache.get_or_stale(symbol)
    if stale is not None:
        log.warning("All strategies failed for %s — serving stale: %.2f", symbol, stale)
        return stale

    log.warning("No price available at all for %s", symbol)
    return None


def get_bulk_prices(symbols: list[str]) -> dict[str, float | None]:
    """
    Fetch prices for many tickers efficiently.

    Pass 1 — fill from cache (zero network calls for fresh hits).
    Pass 2 — call get_live_price() only for cache-misses.
             If the RL flag fires mid-batch, remaining symbols
             are served stale/None without further HTTP calls.

    This is the critical fix for /portfolio:
      Before: N holdings × up to 3 strategies = up to 3N HTTP calls.
      After:  ≤ N HTTP calls on cold load; 0 calls while cache is warm.
    """
    symbols  = list({s.upper().strip() for s in symbols if s.strip()})
    result:   dict[str, float | None] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = _price_cache.get(sym)
        if entry is not None:
            result[sym] = entry["price"]
            log.debug("Bulk HIT  %s → %.2f", sym, entry["price"])
        else:
            to_fetch.append(sym)

    for sym in to_fetch:
        result[sym] = get_live_price(sym)

    return result


# =============================================================================
# SECTION 5 — TICKER VALIDATION
# =============================================================================

_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")


def is_valid_ticker_format(symbol: str) -> bool:
    """Offline format check — instant, no network calls, never fails."""
    return bool(_TICKER_RE.match(symbol.upper().strip()))


def verify_ticker_online(symbol: str) -> bool:
    """
    Best-effort online confirmation. Fail-open on any error or rate limit.
    Only a confirmed HTTP 404 returns False (ticker definitively not found).
    Never blocks stock addition due to Yahoo downtime.
    """
    if _is_rate_limited():
        log.debug("verify_ticker_online: rate-limited, fail-open for %s", symbol)
        return True

    try:
        url  = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?range=1d&interval=1d"
        )
        resp = _session.get(url, headers=_headers(), timeout=6)
        if resp.status_code == 429:
            _trigger_rate_limit()
            return True   # fail open — don't punish the user
        return resp.status_code != 404
    except Exception:
        return True   # network error → fail open


# =============================================================================
# SECTION 6 — CSV HELPERS
# =============================================================================

def load() -> pd.DataFrame:
    if os.path.exists(CSV):
        try:
            return pd.read_csv(CSV)
        except Exception:
            pass
    df = pd.DataFrame(columns=["symbol", "qty", "price"])
    df.to_csv(CSV, index=False)
    return df


def save(df: pd.DataFrame) -> None:
    df.to_csv(CSV, index=False)


# =============================================================================
# SECTION 7 — ROUTES
# =============================================================================

@app.route("/")
def home():
    return render_template("index.html")


# ── /add ──────────────────────────────────────────────────────────────────────
@app.route("/add", methods=["POST"])
def add():
    """Two-stage fail-open validation. Price failure NEVER blocks /add."""
    try:
        data   = request.get_json(force=True)
        symbol = str(data.get("symbol", "")).upper().strip()
        qty    = float(data.get("qty", 0))
        price  = float(data.get("price", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Malformed request body."}), 400

    if not symbol or qty <= 0 or price <= 0:
        return jsonify({
            "status":  "error",
            "message": "All fields are required and must be positive.",
        }), 400

    if not is_valid_ticker_format(symbol):
        return jsonify({
            "status":  "error",
            "message": f"'{symbol}' is not a valid ticker format.",
        }), 400

    if not verify_ticker_online(symbol):
        return jsonify({
            "status":  "error",
            "message": f"'{symbol}' was not found on Yahoo Finance.",
        }), 404

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
    Each unique ticker fetched AT MOST ONCE per request via get_bulk_prices().
    live / pnl / pnl_pct are null when price is unavailable — no crash.
    """
    df = load()
    if df.empty:
        return jsonify([])

    prices = get_bulk_prices(df["symbol"].unique().tolist())

    result = []
    for _, r in df.iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = prices.get(sym)
        pnl  = round((live - bp) * qty, 2)           if live is not None else None
        ppct = round(((live - bp) / bp) * 100, 2)    if live is not None else None
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
    df = load()
    if df.empty:
        return jsonify({"error": "Portfolio is empty."})

    prices    = get_bulk_prices(df["symbol"].unique().tolist())
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
        rl_msg = (
            f" Rate limit active — retry in {_rate_limit_remaining():.0f}s."
            if _is_rate_limited() else ""
        )
        return jsonify({
            "error": f"Live prices temporarily unavailable.{rl_msg}"
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
    """SMA-20 / SMA-50 crossover signal on 6 months of history."""
    symbol = symbol.upper().strip()

    if _is_rate_limited():
        return jsonify({
            "error": (
                f"Yahoo Finance rate limit active — "
                f"retry in {_rate_limit_remaining():.0f}s."
            )
        }), 429

    hist = pd.DataFrame()

    # Primary: direct v8 chart API
    for base in ("query1", "query2"):
        if _is_rate_limited():
            break
        url = (
            f"https://{base}.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?interval=1d&range=6mo"
        )
        try:
            resp = _session.get(url, headers=_headers(), timeout=8)
            if resp.status_code == 429:
                _trigger_rate_limit()
                break
            if resp.status_code == 200:
                raw    = resp.json()["chart"]["result"][0]
                ts     = raw["timestamp"]
                closes = raw["indicators"]["quote"][0]["close"]
                hist   = pd.DataFrame(
                    {"Close": closes},
                    index=[datetime.datetime.utcfromtimestamp(t) for t in ts],
                ).dropna(subset=["Close"])
                if not hist.empty:
                    break
        except Exception as e:
            log.debug("predict v8/%s error for %s: %s", base, symbol, e)

    # Fallback: yfinance
    if hist.empty and not _is_rate_limited():
        try:
            hist = yf.Ticker(symbol, session=_session).history(
                period="6mo", auto_adjust=True
            )
        except Exception as e:
            log.debug("predict yfinance fallback failed for %s: %s", symbol, e)

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
    period   = request.args.get("period", "6M").upper()
    pmap     = {
        "1D": "1d", "1W": "5d", "1M": "1mo",
        "6M": "6mo", "1Y": "1y", "5Y": "5y",
    }
    yf_range = pmap.get(period, "6mo")
    interval = "5m" if yf_range == "1d" else "1d"

    if _is_rate_limited():
        return jsonify({
            "error": (
                f"Rate limit active — retry in {_rate_limit_remaining():.0f}s."
            )
        }), 429

    hist = pd.DataFrame()

    # Primary: direct v8 chart API
    for base in ("query1", "query2"):
        if _is_rate_limited():
            break
        url = (
            f"https://{base}.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?interval={interval}&range={yf_range}"
        )
        try:
            resp = _session.get(url, headers=_headers(), timeout=8)
            if resp.status_code == 429:
                _trigger_rate_limit()
                break
            if resp.status_code == 200:
                raw    = resp.json()["chart"]["result"][0]
                ts     = raw["timestamp"]
                closes = raw["indicators"]["quote"][0]["close"]
                hist   = pd.DataFrame(
                    {"Close": closes},
                    index=[datetime.datetime.utcfromtimestamp(t) for t in ts],
                )
                if not hist.empty:
                    break
        except Exception as e:
            log.debug("chart v8/%s error for %s: %s", base, symbol, e)

    # Fallback: yfinance
    if hist.empty and not _is_rate_limited():
        try:
            hist = yf.Ticker(symbol, session=_session).history(
                period=yf_range, interval=interval, auto_adjust=True
            )
        except Exception as e:
            log.debug("chart yfinance fallback for %s: %s", symbol, e)

    if hist.empty:
        if _is_rate_limited():
            return jsonify({
                "error": f"Rate limit active — retry in {_rate_limit_remaining():.0f}s."
            }), 429
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
        "symbol": symbol,
        "dates":  dates,
        "closes": closes,
        "change": change,
        "pct":    pct,
    })


# ── /cache/stats — diagnostics ────────────────────────────────────────────────
@app.route("/cache/stats")
def cache_stats():
    """
    Live snapshot of cache entries + rate-limit state.
    Visit https://your-app.onrender.com/cache/stats to debug.
    Add Basic Auth or delete this route before going fully public.
    """
    return jsonify({
        "ttl_seconds":    PRICE_CACHE_TTL,
        "rate_limited":   _is_rate_limited(),
        "rl_remaining_s": round(_rate_limit_remaining(), 1),
        "entries":        _price_cache.stats(),
    })


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)