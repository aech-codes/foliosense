/* =============================================================
   FolioSense — script.js
   All frontend logic: add stock · portfolio · chart · insights
   ============================================================= */

"use strict";

/* ─── Chart.js global config ──────────────────────────────── */
Chart.defaults.color           = "#5a5244";
Chart.defaults.font.family     = "'Roboto Mono', monospace";
Chart.defaults.font.size       = 10;

let stockChart     = null;   // Chart.js instance (re-used)
let activeFilter   = "6M";   // current time filter
let activeSymbol   = "";     // symbol shown in chart

/* ─────────────────────────────────────────────────────────────
   TOAST
   ───────────────────────────────────────────────────────────── */
function toast(msg, type = "info") {
  const icons = { success: "✦", error: "✕", info: "◈" };
  const el    = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || "◈"}</span>${msg}`;
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 420);
  }, 3400);
}

/* ─────────────────────────────────────────────────────────────
   SPINNER HELPERS
   ───────────────────────────────────────────────────────────── */
function setBtnLoading(btn, on) {
  if (on) {
    btn._txt   = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span>`;
    btn.disabled  = true;
    btn.style.opacity = "0.65";
  } else {
    btn.innerHTML = btn._txt || btn.innerHTML;
    btn.disabled  = false;
    btn.style.opacity = "1";
  }
}

/* ─────────────────────────────────────────────────────────────
   FORMATTERS
   ───────────────────────────────────────────────────────────── */
const fmt  = v => v == null ? "—" : "$" + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtP = v => v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
const cls  = v => v == null ? "neu" : v > 0 ? "pos" : v < 0 ? "neg" : "neu";
const sign = v => v == null ? "" : v >= 0 ? "+" : "−";

/* =============================================================
   1. ADD STOCK
   ============================================================= */
async function addStock() {
  const btn    = document.getElementById("btn-add");
  const symbol = document.getElementById("f-symbol").value.trim().toUpperCase();
  const qty    = parseFloat(document.getElementById("f-qty").value);
  const price  = parseFloat(document.getElementById("f-price").value);

  if (!symbol)               return toast("Enter a ticker symbol.", "error");
  if (isNaN(qty) || qty <= 0)   return toast("Enter a valid quantity.", "error");
  if (isNaN(price) || price <= 0) return toast("Enter a valid buy price.", "error");

  setBtnLoading(btn, true);
  try {
    const res  = await fetch("/add", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ symbol, qty, price }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Request failed.");

    toast(`${symbol} ${data.action} successfully.`, "success");
    document.getElementById("f-symbol").value = "";
    document.getElementById("f-qty").value    = "";
    document.getElementById("f-price").value  = "";
    loadPortfolio();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    setBtnLoading(btn, false);
  }
}

/* =============================================================
   2. PORTFOLIO — glass cards
   ============================================================= */
async function loadPortfolio() {
  const wrap = document.getElementById("portfolio-grid");
  wrap.innerHTML = `<div class="loading-inline"><span class="spinner"></span>Fetching live prices…</div>`;

  try {
    const res  = await fetch("/portfolio");
    const data = await res.json();

    if (!Array.isArray(data) || data.length === 0) {
      wrap.innerHTML = `
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
          </svg>
          <p>No positions yet — add your first stock above.</p>
        </div>`;
      return;
    }

    wrap.innerHTML = "";
    data.forEach((s, i) => {
      const pnl  = s.pnl;
      const card = document.createElement("div");
      card.className = `stock-card ${pnl > 0 ? "pos-glow" : pnl < 0 ? "neg-glow" : ""}`;
      card.style.animationDelay = `${i * 0.055}s`;
      card.style.opacity = "0";
      card.style.animation = `fadeUp 0.4s ease forwards ${i * 0.06}s`;
      card.innerHTML = `
        <div class="sc-symbol">${s.symbol}</div>
        <div class="sc-qty">${s.qty} shares</div>
        <div class="sc-price">${fmt(s.buy_price)} → ${fmt(s.live)}</div>
        <div class="sc-divider"></div>
        <div class="sc-pnl ${cls(pnl)}">
          ${pnl != null ? sign(pnl) + fmt(pnl) : "—"}
        </div>
        <div class="sc-pnl-pct ${cls(pnl)}">${fmtP(s.pnl_pct)}</div>`;

      // Clicking a card populates the chart symbol
      card.addEventListener("click", () => {
        document.getElementById("chart-sym").value = s.symbol;
        fetchChart(s.symbol, activeFilter);
        document.getElementById("chart-section").scrollIntoView({ behavior: "smooth" });
      });

      wrap.appendChild(card);
    });
  } catch {
    wrap.innerHTML = `<div class="loading-inline" style="color:var(--red)">Failed to load portfolio.</div>`;
  }
}

/* =============================================================
   3. CHART — Chart.js with gold line, Groww-style
   ============================================================= */
async function fetchChart(symbol, period) {
  symbol = symbol || document.getElementById("chart-sym").value.trim().toUpperCase();
  period = period || activeFilter;
  if (!symbol) return toast("Enter a symbol to chart.", "error");

  activeSymbol = symbol;
  activeFilter = period;

  // Highlight active filter button
  document.querySelectorAll(".time-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.period === period);
  });

  const loader = document.getElementById("chart-loader");
  loader.classList.remove("hidden");

  try {
    const res  = await fetch(`/chart/${symbol}?period=${period}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    renderChart(data);

    // Header price / change
    const isUp    = data.pct >= 0;
    const chgStr  = `${isUp ? "+" : ""}${data.change.toFixed(2)} (${isUp ? "+" : ""}${data.pct.toFixed(2)}%)`;
    document.getElementById("chart-sym-name").textContent = data.symbol;
    document.getElementById("chart-price").textContent    = data.closes.length
      ? "$" + data.closes[data.closes.length - 1].toLocaleString("en-US", { minimumFractionDigits: 2 })
      : "—";
    const chEl = document.getElementById("chart-change");
    chEl.textContent = chgStr;
    chEl.className   = `chart-change ${isUp ? "pos" : "neg"}`;
  } catch (err) {
    toast(err.message, "error");
  } finally {
    loader.classList.add("hidden");
  }
}

function renderChart(data) {
  const ctx  = document.getElementById("stockChart").getContext("2d");
  const last = data.closes[data.closes.length - 1];
  const first = data.closes[0];
  const isUp  = last >= first;
  const lineColor = isUp ? "#d4af37" : "#ff4444";

  // Gradient fill
  const gradient = ctx.createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, isUp ? "rgba(212,175,55,0.18)" : "rgba(255,68,68,0.14)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");

  // Thin date labels (every Nth point)
  const n     = data.dates.length;
  const step  = Math.max(1, Math.floor(n / 7));
  const labels = data.dates.map((d, i) => {
    if (i % step !== 0 && i !== n - 1) return "";
    const dt = new Date(d);
    return isNaN(dt) ? d.slice(0, 5) : dt.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  });

  const cfg = {
    type: "line",
    data: {
      labels,
      datasets: [{
        data:            data.closes,
        borderColor:     lineColor,
        backgroundColor: gradient,
        borderWidth:     2,
        tension:         0.38,
        fill:            true,
        pointRadius:     0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: lineColor,
        pointHoverBorderColor: "#0d0d0d",
        pointHoverBorderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(20,20,20,0.95)",
          borderColor: "rgba(212,175,55,0.25)",
          borderWidth: 1,
          titleColor: "#a09880",
          bodyColor: "#f5f0e8",
          titleFont: { family: "'Roboto Mono', monospace", size: 10 },
          bodyFont:  { family: "'Roboto Mono', monospace", size: 12 },
          padding: 12,
          callbacks: {
            label: ctx => " $" + ctx.raw.toLocaleString("en-US", { minimumFractionDigits: 2 }),
          },
        },
      },
      scales: {
        x: {
          grid:     { display: false },
          border:   { display: false },
          ticks: {
            maxRotation: 0,
            color: "#5a5244",
            font: { family: "'Roboto Mono', monospace", size: 9 },
            maxTicksLimit: 8,
            callback: (_, i) => labels[i],
          },
        },
        y: {
          position: "right",
          grid: {
            color: "rgba(255,255,255,0.035)",
            drawTicks: false,
          },
          border: { display: false, dash: [4, 4] },
          ticks: {
            color: "#5a5244",
            font: { family: "'Roboto Mono', monospace", size: 9 },
            maxTicksLimit: 6,
            callback: v => "$" + v.toLocaleString("en-US", { maximumFractionDigits: 0 }),
          },
        },
      },
      animation: { duration: 600, easing: "easeInOutQuart" },
    },
  };

  if (stockChart) {
    stockChart.destroy();
  }
  stockChart = new Chart(ctx, cfg);
}

/* =============================================================
   4. ANALYSIS
   ============================================================= */
async function runAnalysis() {
  const btn = document.getElementById("btn-analysis");
  const out = document.getElementById("insight-output");
  setBtnLoading(btn, true);
  out.innerHTML = `<div class="loading-inline"><span class="spinner"></span>Analysing portfolio…</div>`;
  out.classList.add("lit");

  try {
    const res  = await fetch("/analysis");
    const d    = await res.json();
    if (d.error) throw new Error(d.error);

    const tCls = d.total_pnl >= 0 ? "pos" : "neg";
    out.innerHTML = `
      <div class="stat-row">
        <div class="stat-block">
          <div class="stat-label">Invested</div>
          <div class="stat-value">${fmt(d.total_invested)}</div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Current Value</div>
          <div class="stat-value">${fmt(d.total_value)}</div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Total P&amp;L</div>
          <div class="stat-value ${tCls}">
            ${d.total_pnl >= 0 ? "+" : ""}${fmt(d.total_pnl)}
            <span style="font-size:.72rem;opacity:.7"> ${fmtP(
              ((d.total_value - d.total_invested) / d.total_invested) * 100
            )}</span>
          </div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Best Stock</div>
          <div class="stat-value pos">${d.best_symbol} <span style="font-size:.75rem">${fmtP(d.best_pct)}</span></div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Worst Stock</div>
          <div class="stat-value neg">${d.worst_symbol} <span style="font-size:.75rem">${fmtP(d.worst_pct)}</span></div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Holdings</div>
          <div class="stat-value">${d.holdings}</div>
        </div>
      </div>`;
  } catch (err) {
    out.innerHTML = `<span style="color:var(--red);font-family:var(--mono);font-size:.82rem">${err.message}</span>`;
  } finally {
    setBtnLoading(btn, false);
  }
}

/* =============================================================
   5. NEWS + SENTIMENT
   ============================================================= */
async function fetchNews() {
  const btn = document.getElementById("btn-news");
  const sym = document.getElementById("insight-sym").value.trim().toUpperCase();
  const out = document.getElementById("insight-output");
  if (!sym) return toast("Enter a symbol first.", "error");

  setBtnLoading(btn, true);
  out.innerHTML = `<div class="loading-inline"><span class="spinner"></span>Fetching news for ${sym}…</div>`;
  out.classList.add("lit");

  try {
    const res  = await fetch(`/news/${sym}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const badgeCls = s =>
      s === "Positive" ? "badge-pos" : s === "Negative" ? "badge-neg" : "badge-neu";

    const chipCls  = s =>
      s === "Positive" ? "badge-pos" : s === "Negative" ? "badge-neg" : "badge-neu";

    const items = data.articles.map(a => `
      <div class="news-item">
        <span class="news-headline">${a.headline}</span>
        <span class="news-badge ${badgeCls(a.sentiment)}">${a.sentiment}</span>
      </div>`).join("");

    out.innerHTML = `
      <div class="overall-chip ${chipCls(data.overall)}" style="margin-bottom:14px">
        Overall: <strong>${data.overall}</strong>
      </div>
      <div class="news-list">${items}</div>`;
  } catch (err) {
    out.innerHTML = `<span style="color:var(--red);font-family:var(--mono);font-size:.82rem">${err.message}</span>`;
  } finally {
    setBtnLoading(btn, false);
  }
}

/* =============================================================
   6. PREDICT
   ============================================================= */
async function fetchPredict() {
  const btn = document.getElementById("btn-predict");
  const sym = document.getElementById("insight-sym").value.trim().toUpperCase();
  const out = document.getElementById("insight-output");
  if (!sym) return toast("Enter a symbol first.", "error");

  setBtnLoading(btn, true);
  out.innerHTML = `<div class="loading-inline"><span class="spinner"></span>Analysing trend for ${sym}…</div>`;
  out.classList.add("lit");

  try {
    const res  = await fetch(`/predict/${sym}`);
    const d    = await res.json();
    if (d.error) throw new Error(d.error);

    const isUp    = d.trend === "Uptrend";
    const tCls    = isUp ? "trend-up" : "trend-down";
    const arrow   = isUp ? "▲" : "▼";
    const momSign = d.momentum >= 0 ? "+" : "";
    const momClr  = d.momentum >= 0 ? "var(--green)" : "var(--red)";

    out.innerHTML = `
      <div class="predict-wrap">
        <div class="trend-chip ${tCls}">${arrow} ${d.trend}</div>
        <div class="predict-stats">
          <div>Symbol&nbsp;&nbsp; <span>${d.symbol}</span></div>
          <div>Price&nbsp;&nbsp;&nbsp; <span>$${d.price.toFixed(2)}</span></div>
          <div>SMA-20&nbsp;&nbsp; <span>$${d.sma20.toFixed(2)}</span></div>
          <div>SMA-50&nbsp;&nbsp; <span>$${d.sma50.toFixed(2)}</span></div>
          <div>Momentum <span style="color:${momClr}">${momSign}${d.momentum.toFixed(2)}%</span></div>
        </div>
      </div>`;
  } catch (err) {
    out.innerHTML = `<span style="color:var(--red);font-family:var(--mono);font-size:.82rem">${err.message}</span>`;
  } finally {
    setBtnLoading(btn, false);
  }
}

/* =============================================================
   KEYBOARD SHORTCUTS
   ============================================================= */
document.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const id = document.activeElement.id;
    if (["f-symbol","f-qty","f-price"].includes(id)) addStock();
    if (id === "chart-sym") fetchChart();
    if (id === "insight-sym") runAnalysis();
  }
});

/* =============================================================
   TIME FILTER BUTTONS
   ============================================================= */
document.querySelectorAll(".time-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const period = btn.dataset.period;
    const sym    = document.getElementById("chart-sym").value.trim().toUpperCase() || activeSymbol;
    if (sym) fetchChart(sym, period);
    else toast("Enter a symbol first.", "error");
  });
});

/* =============================================================
   CHART SEARCH — load on Enter or button click
   ============================================================= */
document.getElementById("btn-chart-load")?.addEventListener("click", () => fetchChart());

/* =============================================================
   INIT
   ============================================================= */
document.addEventListener("DOMContentLoaded", () => {
  loadPortfolio();
});