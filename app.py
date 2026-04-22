from flask import Flask, render_template, request, jsonify
import yfinance as yf

app = Flask(__name__)

# In-memory portfolio
portfolio = []


# ✅ SAFE PRICE FUNCTION (FIXED)
def get_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5d")

        if data is None or data.empty:
            return None

        return float(data["Close"].iloc[-1])

    except Exception as e:
        print("Error fetching price:", e)
        return None


# 🟢 HOME
@app.route("/")
def home():
    return render_template("index.html")


# 🟢 ADD STOCK
@app.route("/add", methods=["POST"])
def add_stock():
    data = request.json

    symbol = data.get("symbol", "").upper().strip()
    qty = int(data.get("qty", 0))
    bp = float(data.get("bp", 0))

    # 🔥 DO NOT BLOCK IF PRICE FAILS
    live_price = get_live_price(symbol)

    if live_price is None:
        print(f"Warning: Could not fetch live price for {symbol}")

    portfolio.append({
        "symbol": symbol,
        "qty": qty,
        "bp": bp
    })

    return jsonify({"status": "success"})


# 🟢 GET PORTFOLIO
@app.route("/portfolio")
def get_portfolio():
    result = []

    for stock in portfolio:
        symbol = stock["symbol"]
        qty = stock["qty"]
        bp = stock["bp"]

        live = get_live_price(symbol)

        if live is not None:
            pnl = round((live - bp) * qty, 2)
            pnl_pct = round(((live - bp) / bp) * 100, 2)
            live_val = round(live, 2)
        else:
            pnl = None
            pnl_pct = None
            live_val = None

        result.append({
            "symbol": symbol,
            "qty": qty,
            "bp": bp,
            "live": live_val,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })

    return jsonify(result)


# 🟢 ANALYSIS
@app.route("/analysis")
def analysis():
    total_investment = 0
    total_value = 0

    for stock in portfolio:
        symbol = stock["symbol"]
        qty = stock["qty"]
        bp = stock["bp"]

        live = get_live_price(symbol)

        if live is None:
            continue

        total_investment += bp * qty
        total_value += live * qty

    if total_value == 0:
        return jsonify({
            "error": "Live data unavailable. Try again later."
        })

    profit = round(total_value - total_investment, 2)
    profit_pct = round((profit / total_investment) * 100, 2)

    return jsonify({
        "total_investment": round(total_investment, 2),
        "total_value": round(total_value, 2),
        "profit": profit,
        "profit_pct": profit_pct
    })


# 🟢 DELETE ALL (RESET)
@app.route("/reset", methods=["POST"])
def reset():
    global portfolio
    portfolio = []
    return jsonify({"status": "cleared"})


# 🟢 RUN (for local)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)