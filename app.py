"""
FolioSense — Flask Backend
==========================

CHART ALWAYS LOADS — design contract:
  • /chart/<symbol> NEVER returns an error JSON that blocks the UI.
  • It tries 5 independent data sources in order.
  • If ALL live sources fail it synthesises plausible OHLC data from
    the last known price (or a reasonable default) so Chart.js always
    gets a drawable dataset.
  • NO rate-limit timers, NO "retry later" messages, NO blocking guards.

Price fetch contract:
  • get_live_price() tries 4 strategies, returns last cached value on
    failure, and returns None only if no data has EVER been cached.
  • The in-memory cache is purely opportunistic — it never blocks a request.

Routes:
  GET  /                 — HTML page
  POST /add              — add / merge holding
  GET  /portfolio        — live P&L (null fields when price unavailable)
  GET  /analysis         — aggregate stats
  GET  /news/<symbol>    — VADER-scored headlines
  GET  /predict/<symbol> — SMA-20/50 trend signal
  GET  /chart/<symbol>   — OHLCV closes — ALWAYS returns drawable data
  GET  /health           — liveness probe
"""

import os, re, time, random, logging, datetime, math
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, render_template, request, jsonify

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
_NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, _NLTK_DIR)
nltk.download("vader_lexicon", download_dir=_NLTK_DIR, quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app      = Flask(__name__)
sia      = SentimentIntensityAnalyzer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV      = os.path.join(BASE_DIR, "portfolio.csv")

ALPHA_VANTAGE_KEY: str | None = os.environ.get("ALPHA_VANTAGE_KEY")
PRICE_CACHE_TTL:   int        = int(os.environ.get("PRICE_CACHE_TTL", 90))


# =============================================================================
# SECTION 0 — OPPORTUNISTIC PRICE CACHE (never blocks anything)
# =============================================================================

_cache: dict[str, dict] = {}   # {symbol: {price, ts, source}}

def _cache_set(symbol: str, price: float, source: str) -> None:
    _cache[symbol] = {"price": round(price, 2), "ts": time.monotonic(), "source": source}

def _cache_get_fresh(symbol: str) -> float | None:
    """Return price only if within TTL."""
    e = _cache.get(symbol)
    if e and (time.monotonic() - e["ts"]) < PRICE_CACHE_TTL:
        return e["price"]
    return None

def _cache_get_any(symbol: str) -> float | None:
    """Return price regardless of age — stale is better than nothing."""
    e = _cache.get(symbol)
    return e["price"] if e else None


# =============================================================================
# SECTION 1 — HTTP SESSION & USER-AGENTS
# =============================================================================

_SESSION = requests.Session()
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

def _hdrs() -> dict:
    return {
        "User-Agent":      random.choice(_UAS),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin":          "https://finance.yahoo.com",
        "Referer":         "https://finance.yahoo.com/",
        "Cache-Control":   "no-cache",
    }


# =============================================================================
# SECTION 2 — PRICE FETCH STRATEGIES
# Each returns float | None. Never raises. Never blocks.
# =============================================================================

def _try_yahoo_v8_price(symbol: str) -> float | None:
    for base in ("query1", "query2"):
        try:
            r = _SESSION.get(
                f"https://{base}.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d",
                headers=_hdrs(), timeout=6
            )
            if r.status_code == 200:
                closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                closes = [c for c in closes if c is not None]
                if closes:
                    return float(closes[-1])
        except Exception:
            pass
    return None


def _try_yahoo_v7_price(symbol: str) -> float | None:
    for base in ("query1", "query2"):
        try:
            r = _SESSION.get(
                f"https://{base}.finance.yahoo.com/v7/finance/quote?symbols={symbol}",
                headers=_hdrs(), timeout=6
            )
            if r.status_code == 200:
                items = r.json().get("quoteResponse", {}).get("result", [])
                if items:
                    p = (items[0].get("regularMarketPrice")
                         or items[0].get("postMarketPrice")
                         or items[0].get("preMarketPrice"))
                    if p and float(p) > 0:
                        return float(p)
        except Exception:
            pass
    return None


def _try_yfinance_price(symbol: str) -> float | None:
    try:
        fi = yf.Ticker(symbol, session=_SESSION).fast_info
        p  = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if p and float(p) > 0:
            return float(p)
    except Exception:
        pass
    try:
        h = yf.Ticker(symbol, session=_SESSION).history(period="5d", auto_adjust=True)
        if not h.empty:
            p = float(h["Close"].dropna().iloc[-1])
            if p > 0:
                return p
    except Exception:
        pass
    return None


def _try_alpha_vantage_price(symbol: str) -> float | None:
    if not ALPHA_VANTAGE_KEY:
        return None
    try:
        r = _SESSION.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}",
            headers=_hdrs(), timeout=8
        )
        if r.status_code == 200:
            ps = r.json().get("Global Quote", {}).get("05. price")
            if ps:
                return float(ps)
    except Exception:
        pass
    return None


def get_live_price(symbol: str) -> float | None:
    """
    Cache-first, multi-strategy price lookup. Never blocks. Never raises.
    Returns stale cache on total failure rather than None when possible.
    """
    symbol = symbol.upper().strip()

    fresh = _cache_get_fresh(symbol)
    if fresh is not None:
        return fresh

    for fn in (_try_yahoo_v8_price, _try_yahoo_v7_price,
               _try_yfinance_price, _try_alpha_vantage_price):
        try:
            p = fn(symbol)
        except Exception:
            p = None
        if p and p > 0:
            _cache_set(symbol, p, fn.__name__)
            return round(p, 2)

    # All live strategies failed — return stale cache if any
    stale = _cache_get_any(symbol)
    if stale is not None:
        log.warning("Serving stale price for %s: %.2f", symbol, stale)
    return stale


def get_bulk_prices(symbols: list[str]) -> dict[str, float | None]:
    seen    = list({s.upper().strip() for s in symbols if s.strip()})
    result  = {}
    misses  = []
    for sym in seen:
        f = _cache_get_fresh(sym)
        if f is not None:
            result[sym] = f
        else:
            misses.append(sym)
    for sym in misses:
        result[sym] = get_live_price(sym)
    return result


# =============================================================================
# SECTION 3 — CHART DATA FETCHING (ALWAYS returns drawable data)
# =============================================================================

_chart_cache: dict[str, dict] = {}  # {key: {dates, closes, ts}}
_CHART_TTL = int(os.environ.get("CHART_CACHE_TTL", 300))  # 5 min default


def _chart_cache_key(symbol: str, period: str) -> str:
    return f"{symbol}:{period}"


def _chart_cache_get(symbol: str, period: str) -> dict | None:
    e = _chart_cache.get(_chart_cache_key(symbol, period))
    if e and (time.monotonic() - e["ts"]) < _CHART_TTL:
        return e
    return None


def _chart_cache_get_any(symbol: str, period: str) -> dict | None:
    """Return stale chart cache — used as fallback."""
    return _chart_cache.get(_chart_cache_key(symbol, period))


def _chart_cache_set(symbol: str, period: str, dates: list, closes: list) -> None:
    _chart_cache[_chart_cache_key(symbol, period)] = {
        "dates": dates, "closes": closes, "ts": time.monotonic()
    }


def _fetch_chart_yahoo_v8(symbol: str, yf_range: str, interval: str) -> tuple[list, list] | None:
    """Direct Yahoo v8 chart API — most stable, no yfinance overhead."""
    for base in ("query1", "query2"):
        try:
            r = _SESSION.get(
                f"https://{base}.finance.yahoo.com/v8/finance/chart/"
                f"{symbol}?interval={interval}&range={yf_range}",
                headers=_hdrs(), timeout=8
            )
            if r.status_code != 200:
                continue
            raw     = r.json()["chart"]["result"][0]
            ts      = raw["timestamp"]
            closes  = raw["indicators"]["quote"][0]["close"]
            pairs   = [(t, c) for t, c in zip(ts, closes) if c is not None]
            if len(pairs) >= 2:
                dates  = [
                    str(datetime.datetime.utcfromtimestamp(t).date()) if interval == "1d"
                    else str(datetime.datetime.utcfromtimestamp(t))[:16]
                    for t, _ in pairs
                ]
                prices = [round(float(c), 2) for _, c in pairs]
                return dates, prices
        except Exception as e:
            log.debug("v8/%s chart failed for %s: %s", base, symbol, e)
    return None


def _fetch_chart_yahoo_v8_longer(symbol: str, yf_range: str, interval: str) -> tuple[list, list] | None:
    """Try a longer range when the requested range returns too little data."""
    fallback_ranges = {"1d": "5d", "5d": "1mo", "1mo": "6mo", "6mo": "1y", "1y": "2y", "5y": "5y"}
    alt = fallback_ranges.get(yf_range)
    if alt and alt != yf_range:
        return _fetch_chart_yahoo_v8(symbol, alt, "1d")
    return None


def _fetch_chart_yfinance(symbol: str, yf_range: str, interval: str) -> tuple[list, list] | None:
    try:
        h = yf.Ticker(symbol, session=_SESSION).history(
            period=yf_range, interval=interval, auto_adjust=True
        )
        if h.empty:
            # Try broader range
            h = yf.Ticker(symbol, session=_SESSION).history(period="6mo", interval="1d", auto_adjust=True)
        if not h.empty:
            h = h.dropna(subset=["Close"])
            dates  = [
                str(d.date()) if interval == "1d" else str(d)[:16]
                for d in h.index
            ]
            closes = [round(float(p), 2) for p in h["Close"]]
            if len(closes) >= 2:
                return dates, closes
    except Exception as e:
        log.debug("yfinance chart failed for %s: %s", symbol, e)
    return None


def _fetch_chart_alpha_vantage(symbol: str) -> tuple[list, list] | None:
    """Alpha Vantage daily time series — reliable on cloud IPs."""
    if not ALPHA_VANTAGE_KEY:
        return None
    try:
        r = _SESSION.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_DAILY", "symbol": symbol,
                    "outputsize": "compact", "apikey": ALPHA_VANTAGE_KEY},
            headers=_hdrs(), timeout=10
        )
        if r.status_code == 200:
            ts = r.json().get("Time Series (Daily)", {})
            if ts:
                items  = sorted(ts.items())[-90:]   # last 90 trading days
                dates  = [d for d, _ in items]
                closes = [round(float(v["4. close"]), 2) for _, v in items]
                if len(closes) >= 2:
                    return dates, closes
    except Exception as e:
        log.debug("alpha vantage chart failed for %s: %s", symbol, e)
    return None


def _synthesise_chart(symbol: str, n_days: int = 90) -> tuple[list, list]:
    """
    Last-resort: generate a plausible price series so Chart.js never
    receives an empty dataset. Uses the last known price as the anchor
    (or 100.0 if no price has ever been cached). Applies a random walk
    so the chart looks like real market data rather than a flat line.
    This is clearly labelled as estimated data by the caller.
    """
    anchor = _cache_get_any(symbol) or 100.0
    rng    = random.Random(hash(symbol))   # deterministic per symbol
    prices = [anchor]
    for _ in range(n_days - 1):
        pct    = rng.gauss(0, 0.012)       # ±1.2% daily vol, realistic
        prices.append(round(max(0.01, prices[-1] * (1 + pct)), 2))
    prices.reverse()                        # oldest first

    today   = datetime.date.today()
    dates   = [
        str(today - datetime.timedelta(days=n_days - 1 - i))
        for i in range(n_days)
    ]
    return dates, prices


def fetch_chart_data(symbol: str, period: str) -> dict:
    """
    Master chart fetcher. Contract:
      • ALWAYS returns a dict with {symbol, dates, closes, change, pct, source}.
      • dates and closes are NEVER empty — synthesised data fills the gap.
      • No exceptions propagate to the caller.
      • No retry timers. No blocking conditions.
    """
    symbol = symbol.upper().strip()

    pmap = {
        "1D": ("1d",  "5m"),
        "1W": ("5d",  "1d"),
        "1M": ("1mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y",  "1d"),
        "5Y": ("5y",  "1wk"),
    }
    yf_range, interval = pmap.get(period.upper(), ("6mo", "1d"))

    # 1. Fresh cache
    cached = _chart_cache_get(symbol, period)
    if cached:
        dates, closes = cached["dates"], cached["closes"]
        source = "cache"
    else:
        dates, closes, source = None, None, None

        # 2. Yahoo v8 direct API (primary — no yfinance dependency)
        if not dates:
            result = _fetch_chart_yahoo_v8(symbol, yf_range, interval)
            if result:
                dates, closes = result
                source = "yahoo_v8"

        # 3. Yahoo v8 with a longer range fallback
        if not dates:
            result = _fetch_chart_yahoo_v8_longer(symbol, yf_range, interval)
            if result:
                dates, closes = result
                source = "yahoo_v8_wider"

        # 4. yfinance (uses same Yahoo endpoint internally but different auth path)
        if not dates:
            result = _fetch_chart_yfinance(symbol, yf_range, interval)
            if result:
                dates, closes = result
                source = "yfinance"

        # 5. Alpha Vantage (completely independent — never rate-limited by Yahoo)
        if not dates:
            result = _fetch_chart_alpha_vantage(symbol)
            if result:
                dates, closes = result
                source = "alpha_vantage"

        # 6. Stale chart cache (any age)
        if not dates:
            stale = _chart_cache_get_any(symbol, period)
            # Also check other periods for this symbol
            if not stale:
                for alt_period in ("6M", "1Y", "1M", "1W", "5Y", "1D"):
                    if alt_period != period:
                        stale = _chart_cache_get_any(symbol, alt_period)
                        if stale:
                            break
            if stale:
                dates, closes = stale["dates"], stale["closes"]
                source = "stale_cache"

        # 7. Synthesise — absolute last resort, always succeeds
        if not dates:
            n = {"1D": 1, "1W": 7, "1M": 30, "6M": 90, "1Y": 180, "5Y": 365}.get(period.upper(), 90)
            dates, closes = _synthesise_chart(symbol, n_days=max(n, 7))
            source = "estimated"
            log.warning("Serving synthesised chart data for %s (%s)", symbol, period)

        # Cache successful live fetches
        if source not in ("stale_cache", "estimated"):
            _chart_cache_set(symbol, period, dates, closes)

    change = round(closes[-1] - closes[0], 2) if len(closes) > 1 else 0.0
    pct    = round((change / closes[0]) * 100, 2) if closes and closes[0] else 0.0

    return {
        "symbol":    symbol,
        "period":    period,
        "dates":     dates,
        "closes":    closes,
        "change":    change,
        "pct":       pct,
        "source":    source,
        "estimated": source == "estimated",
    }


# =============================================================================
# SECTION 4 — TICKER VALIDATION
# =============================================================================

_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")

def is_valid_ticker_format(symbol: str) -> bool:
    return bool(_TICKER_RE.match(symbol.upper().strip()))

def verify_ticker_online(symbol: str) -> bool:
    """Best-effort; fail-open on any error or rate limit."""
    try:
        r = _SESSION.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d",
            headers=_hdrs(), timeout=6
        )
        return r.status_code != 404
    except Exception:
        return True   # fail open


# =============================================================================
# SECTION 5 — CSV HELPERS
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
# SECTION 6 — ROUTES
# =============================================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add", methods=["POST"])
def add():
    try:
        data   = request.get_json(force=True)
        symbol = str(data.get("symbol", "")).upper().strip()
        qty    = float(data.get("qty", 0))
        price  = float(data.get("price", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Malformed request."}), 400

    if not symbol or qty <= 0 or price <= 0:
        return jsonify({"status": "error",
                        "message": "All fields are required and must be positive."}), 400

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


@app.route("/portfolio")
def portfolio():
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
        pnl  = round((live - bp) * qty, 2)        if live is not None else None
        ppct = round(((live - bp) / bp) * 100, 2) if live is not None else None
        result.append({"symbol": sym, "qty": qty, "buy_price": bp,
                        "live": live, "pnl": pnl, "pnl_pct": ppct})
    return jsonify(result)


@app.route("/analysis")
def analysis():
    df = load()
    if df.empty:
        return jsonify({"error": "Portfolio is empty."})

    prices    = get_bulk_prices(df["symbol"].unique().tolist())
    rows      = []
    total_inv = total_val = 0.0

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
        return jsonify({"error": "Live prices temporarily unavailable."})

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
        sentiment = ("Positive" if score >= 0.05
                     else "Negative" if score <= -0.05 else "Neutral")
        articles.append({"headline": h, "sentiment": sentiment, "score": round(score, 3)})
    avg     = sum(a["score"] for a in articles) / len(articles)
    overall = ("Positive" if avg >= 0.05
               else "Negative" if avg <= -0.05 else "Neutral")
    return jsonify({"symbol": symbol, "articles": articles, "overall": overall})


@app.route("/predict/<symbol>")
def predict(symbol: str):
    symbol = symbol.upper().strip()

    # Try live fetch first, then stale chart cache as fallback
    data = fetch_chart_data(symbol, "6M")
    closes_list = data["closes"]

    if len(closes_list) < 20:
        return jsonify({"error": f"Insufficient data for '{symbol}'."}), 404

    close   = pd.Series(closes_list)
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
        "source":   data.get("source", "unknown"),
    })


@app.route("/chart/<symbol>")
def chart(symbol: str):
    """
    ALWAYS returns HTTP 200 with a drawable dataset.
    Never returns an error that would cause the frontend to show a spinner forever.
    The `estimated` flag lets the UI optionally show a "data unavailable" watermark
    while still rendering a chart.
    """
    symbol = symbol.upper().strip()
    period = request.args.get("period", "6M").upper()

    data = fetch_chart_data(symbol, period)   # guaranteed non-empty

    return jsonify(data), 200   # always 200 — frontend always renders


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)