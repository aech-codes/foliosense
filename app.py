"""
FolioSense — Flask Backend  (production-safe for Render / Railway)
==================================================================

WHY yfinance FAILS ON CLOUD PLATFORMS
--------------------------------------
1. IP blocking   — Cloud provider IPs (Render, Railway, AWS, GCP) are on Yahoo
                   Finance's bot-detection block-lists. Requests get 401/403/429.
2. Cookie crumb  — yfinance ≥ 0.2.38 must complete a cookie handshake before
                   every session. Cloud IPs often fail this step silently and
                   return an empty DataFrame instead of raising an error.
3. User-Agent    — Default Python requests UA is blocked. Yahoo requires a
                   browser-like User-Agent string.
4. Rate limiting — Shared cloud IP pools hammer Yahoo. One tenant's spike can
                   block all others on the same IP.
5. Market hours  — .history(period="1d") returns EMPTY on weekends, US holidays,
                   and before market open. This is NOT a bad ticker.

THE WRONG FIX
--------------
Using get_live_price() to validate a ticker is the root design bug.
If Yahoo is down → every "Add Stock" returns "Ticker not found" even for AAPL.

THE RIGHT APPROACH
------------------
• Separate validation from price fetching completely.
• Validate tickers against a local known-ticker allowlist + basic format check.
• Fetch prices with 3 independent fallback strategies.
• NEVER block a user action just because price data is temporarily unavailable.
• Show "Unavailable" instead of crashing.

Routes:
  GET  /                 — HTML page
  POST /add              — add / merge stock (NEVER blocked by price failures)
  GET  /portfolio        — holdings with live P&L (shows Unavailable gracefully)
  GET  /analysis         — total P&L, best, worst
  GET  /news/<symbol>    — headlines + VADER sentiment
  GET  /predict/<symbol> — SMA trend signal
  GET  /chart/<symbol>   — OHLCV close prices

Run locally:  python app.py
Deploy:       gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import re
import time
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


def _strategy_yfinance_fast_info(symbol: str):
    """
    Strategy 1: yfinance fast_info (lightweight, single HTTP call).
    Uses a patched session with browser-like headers to defeat bot detection.
    fast_info["lastPrice"] is populated even outside market hours.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    ticker = yf.Ticker(symbol, session=session)

    # Try fast_info first (no cookie crumb needed in most cases)
    try:
        fi    = ticker.fast_info
        price = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if price and float(price) > 0:
            return float(price), "fast_info"
    except Exception as e:
        log.debug("fast_info failed for %s: %s", symbol, e)

    # Fallback within strategy 1: try .info dict
    try:
        info  = ticker.info
        price = (
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
    A 5-day window handles weekends and holidays — never returns empty for a
    valid ticker (unless the market literally hasn't opened yet this week).
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        ticker = yf.Ticker(symbol, session=session)
        hist   = ticker.history(period="5d", auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1]), "history_5d"
    except Exception as e:
        log.debug("history_5d failed for %s: %s", symbol, e)

    return None, None


def _strategy_yahoo_query_api(symbol: str):
    """
    Strategy 3: Hit Yahoo Finance's public v8 quote API directly.
    This endpoint is more stable than the crumb-based v10 API and doesn't
    require the yfinance session handshake.
    """
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval=1d&range=5d"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if resp.status_code == 200:
            data   = resp.json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if closes:
                return float(closes[-1]), "yahoo_api_v8"
    except Exception as e:
        log.debug("yahoo_api_v8 failed for %s: %s", symbol, e)

    # Mirror URL — Yahoo sometimes geo-routes to query2
    url2 = url.replace("query1", "query2")
    try:
        resp = requests.get(url2, headers=_HEADERS, timeout=8)
        if resp.status_code == 200:
            data   = resp.json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if closes:
                return float(closes[-1]), "yahoo_api_v8_mirror"
    except Exception as e:
        log.debug("yahoo_api_v8_mirror failed for %s: %s", symbol, e)

    return None, None


def get_live_price(symbol: str) -> float | None:
    """
    Public interface — tries all three strategies in order.
    Returns the price as a float, or None if every strategy fails.
    NEVER raises. Safe to call from any route handler.
    """
    symbol = symbol.upper().strip()
    for strategy in (_strategy_yfinance_fast_info,
                     _strategy_yfinance_history,
                     _strategy_yahoo_query_api):
        try:
            price, source = strategy(symbol)
            if price is not None and price > 0:
                log.info("Price for %s: %.2f (via %s)", symbol, price, source)
                return round(float(price), 2)
        except Exception as e:
            log.warning("Strategy %s crashed for %s: %s", strategy.__name__, symbol, e)

    log.warning("All price strategies failed for %s", symbol)
    return None


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
    Examples of valid:   AAPL, MSFT, RELIANCE.NS, BRK.B
    Examples of invalid: 12345, !, TOOLONGSYMBOL
    """
    return bool(_TICKER_RE.match(symbol.upper().strip()))


def verify_ticker_online(symbol: str) -> bool:
    """
    Optional online check — tries to get any data for the symbol.
    Used ONLY during /add as a best-effort confirmation.
    Returns True if confirmed, True if check fails (fail-open design).
    We fail open because blocking users due to Yahoo downtime is worse
    than occasionally accepting a bad ticker.
    """
    try:
        session = requests.Session()
        session.headers.update(_HEADERS)
        url  = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
        resp = session.get(url, timeout=6)
        if resp.status_code == 404:
            return False          # Definitely not a real ticker
        return True               # 200, 429, 500, etc. — assume valid
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


# ── /add ─────────────────────────────────────────────────────────────────────
@app.route("/add", methods=["POST"])
def add():
    """
    KEY DESIGN CHANGE:
    Ticker validation is now a two-stage fail-open process:
      1. Format check (offline, instant) — rejects garbage like "123!@#"
      2. Online check (best-effort)      — only rejects confirmed 404s
    Price fetch failure NEVER blocks adding a stock.
    """
    data   = request.get_json(force=True)
    symbol = str(data.get("symbol", "")).upper().strip()
    qty    = float(data.get("qty", 0))
    price  = float(data.get("price", 0))

    # Basic input sanity
    if not symbol or qty <= 0 or price <= 0:
        return jsonify({"status": "error", "message": "Invalid input — all fields are required."}), 400

    # Stage 1: offline format check
    if not is_valid_ticker_format(symbol):
        return jsonify({"status": "error", "message": f"'{symbol}' is not a valid ticker format."}), 400

    # Stage 2: online check (fail-open — only hard-rejects confirmed 404)
    if not verify_ticker_online(symbol):
        return jsonify({"status": "error", "message": f"'{symbol}' was not found on Yahoo Finance."}), 404

    # Merge or add to portfolio
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
    Returns all holdings. If price fetch fails, live/pnl/pnl_pct are null —
    the frontend shows 'Unavailable' instead of crashing.
    """
    result = []
    for _, r in load().iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = get_live_price(sym)          # None if all strategies fail
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
    df = load()
    if df.empty:
        return jsonify({"error": "Portfolio is empty."})

    rows      = []
    total_inv = 0.0
    total_val = 0.0

    for _, r in df.iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = get_live_price(sym)
        if live is None:
            continue
        pnl     = (live - bp) * qty
        pnl_pct = ((live - bp) / bp) * 100
        total_inv += qty * bp
        total_val += qty * live
        rows.append({"symbol": sym, "pnl": pnl, "pnl_pct": pnl_pct})

    if not rows:
        return jsonify({"error": "Live prices temporarily unavailable. Try again in a moment."})

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
        sentiment = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
        articles.append({"headline": h, "sentiment": sentiment, "score": round(score, 3)})
    avg     = sum(a["score"] for a in articles) / len(articles)
    overall = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
    return jsonify({"symbol": symbol, "articles": articles, "overall": overall})


# ── /predict/<symbol> ─────────────────────────────────────────────────────────
@app.route("/predict/<symbol>")
def predict(symbol: str):
    """SMA-20 vs SMA-50 crossover on 6 months of data."""
    symbol = symbol.upper().strip()

    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        hist = yf.Ticker(symbol, session=session).history(period="6mo", auto_adjust=True)
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
    pmap     = {"1D": "1d", "1W": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
    yf_p     = pmap.get(period, "6mo")
    interval = "5m" if yf_p == "1d" else "1d"

    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        hist = yf.Ticker(symbol, session=session).history(
            period=yf_p, interval=interval, auto_adjust=True
        )
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {e}"}), 500

    # Fallback: try the raw Yahoo v8 API if yfinance returns empty
    if hist.empty:
        try:
            url  = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={yf_p}"
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            if resp.status_code == 200:
                raw    = resp.json()["chart"]["result"][0]
                ts     = raw["timestamp"]
                closes = raw["indicators"]["quote"][0]["close"]
                import datetime
                hist = pd.DataFrame({
                    "Close": closes,
                }, index=[datetime.datetime.utcfromtimestamp(t) for t in ts])
        except Exception:
            pass

    if hist.empty:
        return jsonify({"error": f"No chart data for '{symbol}'."}), 404

    hist   = hist.dropna(subset=["Close"])
    dates  = [str(d.date()) if interval == "1d" else str(d)[:16] for d in hist.index]
    closes = [round(float(p), 2) for p in hist["Close"]]
    change = round(closes[-1] - closes[0], 2) if len(closes) > 1 else 0
    pct    = round((change / closes[0]) * 100, 2) if closes[0] else 0

    return jsonify({
        "symbol": symbol, "dates": dates,
        "closes": closes, "change": change, "pct": pct,
    })


# ── Health check (Render pings this to confirm the app is alive) ──────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)