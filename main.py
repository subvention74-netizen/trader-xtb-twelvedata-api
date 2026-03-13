from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import os
import requests
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="TRADER-XTB Twelve Data API",
    version="1.0.0",
    description="API simple pour fournir des données live Twelve Data à un GPT."
)

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
INTERNAL_BEARER_TOKEN = os.getenv("INTERNAL_BEARER_TOKEN", "")
BASE_URL = "https://api.twelvedata.com"


# -------------------------
# Modèles
# -------------------------

class QuoteResponse(BaseModel):
    ticker: str
    market: Optional[str] = None
    price: float
    change_percent: float
    change_value: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None
    volume: float
    currency: str
    timestamp: datetime
    source: str


class AnalysisDataResponse(BaseModel):
    ticker: str
    market: Optional[str] = None
    quote: Dict[str, Any]
    technicals: Dict[str, Any]
    fundamentals: Dict[str, Any]
    timestamp: datetime
    source: str


# -------------------------
# Sécurité simple
# -------------------------

def check_bearer(auth_header: Optional[str]) -> None:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.replace("Bearer ", "", 1).strip()
    if token != INTERNAL_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def td_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not TWELVE_DATA_API_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_DATA_API_KEY not configured")

    headers = {
        "Authorization": f"apikey {TWELVE_DATA_API_KEY}"
    }

    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, headers=headers, timeout=20)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Twelve Data HTTP error {response.status_code}")

    data = response.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise HTTPException(status_code=502, detail=data.get("message", "Twelve Data error"))

    return data


# -------------------------
# Calculs techniques
# -------------------------

def compute_sma(values: List[float], period: int):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def compute_ema(values: List[float], period: int):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for price in values[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def compute_momentum(values: List[float], period: int = 10):
    if len(values) <= period:
        return None
    base = values[-period - 1]
    if base == 0:
        return None
    return ((values[-1] / base) - 1) * 100


def compute_rsi(closes: List[float], period: int = 14):
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14):
    if len(closes) < period + 1:
        return None

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = ((atr * (period - 1)) + tr) / period

    return atr


def pct_volatility_from_atr(price: float, atr: Optional[float]):
    if not price or atr is None:
        return None
    return (atr / price) * 100


# -------------------------
# Twelve Data mapping
# -------------------------

def get_quote_from_twelve_data(symbol: str) -> Dict[str, Any]:
    # Endpoint price : dernier prix
    price_data = td_get("price", {"symbol": symbol})

    # Endpoint quote : infos complètes du jour
    quote_data = td_get("quote", {"symbol": symbol})

    now = datetime.now(timezone.utc)

    price = float(price_data["price"])
    percent_change = float(quote_data.get("percent_change", 0) or 0)
    change_value = float(quote_data.get("change", 0) or 0)
    open_price = float(quote_data.get("open", 0) or 0)
    high = float(quote_data.get("high", 0) or 0)
    low = float(quote_data.get("low", 0) or 0)
    previous_close = float(quote_data.get("previous_close", 0) or 0)
    volume = float(quote_data.get("volume", 0) or 0)

    return {
        "ticker": quote_data.get("symbol", symbol),
        "market": quote_data.get("exchange"),
        "price": price,
        "change_percent": percent_change,
        "change_value": change_value,
        "open": open_price,
        "high": high,
        "low": low,
        "previous_close": previous_close,
        "volume": volume,
        "currency": quote_data.get("currency", "USD"),
        "timestamp": now,
        "source": "twelvedata"
    }


def get_history_from_twelve_data(symbol: str) -> Dict[str, List[float]]:
    ts_data = td_get(
        "time_series",
        {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 60,
            "format": "JSON"
        }
    )

    values = ts_data.get("values", [])
    if not values:
        raise HTTPException(status_code=502, detail="No historical data returned by Twelve Data")

    # Twelve Data renvoie souvent l’ordre du plus récent au plus ancien
    values = list(reversed(values))

    closes = [float(v["close"]) for v in values]
    highs = [float(v["high"]) for v in values]
    lows = [float(v["low"]) for v in values]

    return {"closes": closes, "highs": highs, "lows": lows}


def get_fundamentals_from_twelve_data(symbol: str) -> Dict[str, Any]:
    # statistics est dispo à partir du plan Pro
    try:
        stats = td_get("statistics", {"symbol": symbol})
    except HTTPException:
        return {
            "pe_ratio": None,
            "eps": None,
            "market_cap": None,
            "revenue_growth_percent": None,
            "profit_margin_percent": None,
            "note": "Statistics non disponibles sur votre plan Twelve Data ou non renvoyées par l'API."
        }

    valuation = stats.get("valuation_metrics", {}) or {}
    profitability = stats.get("profitability_indicators", {}) or {}
    income_statement = stats.get("income_statement", {}) or {}
    highlights = stats.get("highlights", {}) or {}

    pe_ratio = valuation.get("price_to_earnings_ratio_ttm") or highlights.get("pe_ratio")
    eps = highlights.get("earnings_per_share") or income_statement.get("eps")
    market_cap = highlights.get("market_capitalization")
    profit_margin = profitability.get("net_profit_margin")
    revenue_growth = highlights.get("quarterly_revenue_growth_yoy")

    def to_float(value):
        try:
            return float(value)
        except Exception:
            return None

    return {
        "pe_ratio": to_float(pe_ratio),
        "eps": to_float(eps),
        "market_cap": to_float(market_cap),
        "revenue_growth_percent": to_float(revenue_growth),
        "profit_margin_percent": to_float(profit_margin)
    }


# -------------------------
# Pages simples
# -------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>TRADER-XTB Twelve Data API</title></head>
      <body>
        <h1>TRADER-XTB Twelve Data API</h1>
        <p>API en ligne.</p>
        <p>Privacy policy: <a href="/privacy">/privacy</a></p>
        <p>Health check: <a href="/health">/health</a></p>
      </body>
    </html>
    """


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <html>
      <head><title>Privacy Policy</title></head>
      <body>
        <h1>Privacy Policy</h1>
        <p>Cette API reçoit uniquement les données nécessaires pour répondre aux requêtes de marché envoyées par le GPT, comme un ticker financier et des paramètres de requête.</p>
        <p>Elle n'est pas conçue pour stocker des données personnelles sensibles. Les requêtes peuvent être journalisées à des fins techniques et de sécurité.</p>
        <p>Les données de marché proviennent de Twelve Data.</p>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/quote", response_model=QuoteResponse)
def get_quote(
    ticker: str = Query(..., min_length=1),
    authorization: Optional[str] = Header(None)
):
    check_bearer(authorization)
    data = get_quote_from_twelve_data(ticker)
    return QuoteResponse(**data)


@app.get("/analysis-data", response_model=AnalysisDataResponse)
def get_analysis_data(
    ticker: str = Query(..., min_length=1),
    authorization: Optional[str] = Header(None)
):
    check_bearer(authorization)

    quote = get_quote_from_twelve_data(ticker)
    history = get_history_from_twelve_data(ticker)
    fundamentals = get_fundamentals_from_twelve_data(ticker)

    closes = history["closes"]
    highs = history["highs"]
    lows = history["lows"]

    rsi_14 = compute_rsi(closes, 14)
    atr_14 = compute_atr(highs, lows, closes, 14)
    sma_20 = compute_sma(closes, 20)
    sma_50 = compute_sma(closes, 50)
    ema_21 = compute_ema(closes, 21)
    momentum_10 = compute_momentum(closes, 10)
    volatility_percent = pct_volatility_from_atr(quote["price"], atr_14)

    technicals = {
        "rsi_14": round(rsi_14, 2) if rsi_14 is not None else None,
        "atr_14": round(atr_14, 4) if atr_14 is not None else None,
        "sma_20": round(sma_20, 4) if sma_20 is not None else None,
        "sma_50": round(sma_50, 4) if sma_50 is not None else None,
        "ema_21": round(ema_21, 4) if ema_21 is not None else None,
        "momentum_10": round(momentum_10, 2) if momentum_10 is not None else None,
        "volatility_percent": round(volatility_percent, 2) if volatility_percent is not None else None,
    }

    return AnalysisDataResponse(
        ticker=quote["ticker"],
        market=quote.get("market"),
        quote={
            "price": quote["price"],
            "change_percent": quote["change_percent"],
            "volume": quote["volume"],
            "currency": quote["currency"],
        },
        technicals=technicals,
        fundamentals=fundamentals,
        timestamp=quote["timestamp"],
        source=quote["source"],
    )
