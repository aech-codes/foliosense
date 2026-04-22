"""
FolioSense — Flask Backend
==========================
Routes:
  GET  /                 — HTML page
  POST /add              — add / merge stock
  GET  /portfolio        — holdings with live P&L
  GET  /analysis         — total P&L, best, worst
  GET  /news/<symbol>    — headlines + VADER sentiment
  GET  /predict/<symbol> — SMA trend signal
  GET  /chart/<symbol>   — OHLCV close prices

Run locally:  python app.py
Deploy:       gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
"""

import os
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify

# ── NLTK — download vader_lexicon to a writable path on Render ──────────────
import nltk

# On Render the home directory is writable; point nltk there explicitly
NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)
nltk.download("vader_lexicon", download_dir=NLTK_DIR, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# FIX 1: Use an ABSOLUTE path so the CSV is always found regardless of
#         which directory gunicorn is invoked from on Render.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV      = os.path.join(BASE_DIR, "portfolio.csv")


# ── CSV helpers ──────────────────────────────────────────────────────────────
def load():
    """Load portfolio from CSV; create empty file if it doesn't exist."""
    if os.path.exists(CSV):
        return pd.read_csv(CSV)
    df = pd.DataFrame(columns=["symbol", "qty", "price"])
    df.to_csv(CSV, index=False)
    return df


def save(df):
    df.to_csv(CSV, index=False)


# ── Market data ──────────────────────────────────────────────────────────────
def get_live_price(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add", methods=["POST"])
def add():
    """Add or merge a stock. Validates ticker via live price fetch."""
    data   = request.get_json(force=True)
    symbol = str(data.get("symbol", "")).upper().strip()
    qty    = float(data.get("qty", 0))
    price  = float(data.get("price", 0))

    if not symbol or qty <= 0 or price <= 0:
        return jsonify({"status": "error", "message": "Invalid input."}), 400

    if get_live_price(symbol) is None:
        return jsonify({"status": "error", "message": f"Ticker '{symbol}' not found."}), 404

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
    """Return all holdings with live price and P&L."""
    result = []
    for _, r in load().iterrows():
        sym  = r["symbol"]
        qty  = float(r["qty"])
        bp   = float(r["price"])
        live = get_live_price(sym)
        pnl  = round((live - bp) * qty, 2) if live else None
        ppct = round(((live - bp) / bp) * 100, 2) if live else None
        result.append({
            "symbol":    sym,
            "qty":       qty,
            "buy_price": bp,
            "live":      round(live, 2) if live else None,
            "pnl":       pnl,
            "pnl_pct":   ppct,
        })
    return jsonify(result)


@app.route("/analysis")
def analysis():
    """Return aggregate portfolio performance."""
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
        return jsonify({"error": "No live prices available."})

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


# Headline templates — swap for a real News API in production
HEADLINES = [
    "{s} beats earnings estimates; shares rally in after-hours trading",
    "{s} announces strategic expansion — analysts raise price targets",
    "{s} faces regulatory pressure as sector scrutiny intensifies",
    "{s} reports record quarterly revenue on strong consumer demand",
    "{s} slips amid broad market sell-off; technical support holds",
]

@app.route("/news/<symbol>")
def news(symbol):
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


@app.route("/predict/<symbol>")
def predict(symbol):
    """SMA-20 vs SMA-50 crossover → Uptrend / Downtrend."""
    symbol = symbol.upper().strip()
    try:
        hist = yf.Ticker(symbol).history(period="6mo")
    except Exception:
        return jsonify({"error": "Data fetch failed."}), 500

    if hist.empty or len(hist) < 20:
        return jsonify({"error": f"Insufficient data for '{symbol}'."}), 404

    close   = hist["Close"]
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


@app.route("/chart/<symbol>")
def chart(symbol):
    """Return dates + closes for Chart.js. ?period=1D|1W|1M|6M|1Y|5Y"""
    symbol   = symbol.upper().strip()
    period   = request.args.get("period", "6M")
    pmap     = {"1D": "1d", "1W": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
    yf_p     = pmap.get(period, "6mo")
    interval = "5m" if yf_p == "1d" else "1d"

    try:
        hist = yf.Ticker(symbol).history(period=yf_p, interval=interval)
    except Exception:
        return jsonify({"error": "Data fetch failed."}), 500

    if hist.empty:
        return jsonify({"error": f"No data for '{symbol}'."}), 404

    dates  = [str(d.date()) if interval == "1d" else str(d)[:16] for d in hist.index]
    closes = [round(float(p), 2) for p in hist["Close"]]
    change = round(closes[-1] - closes[0], 2) if len(closes) > 1 else 0
    pct    = round((change / closes[0]) * 100, 2) if closes[0] else 0

    return jsonify({
        "symbol": symbol, "dates": dates,
        "closes": closes, "change": change, "pct": pct,
    })


# ── Entry point ──────────────────────────────────────────────────────────────
# FIX 2: Use the PORT environment variable that Render injects.
#         Gunicorn overrides this anyway, but it ensures `python app.py`
#         also works correctly on any platform.
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)