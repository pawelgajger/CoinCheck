import matplotlib
matplotlib.use('Agg')  # Ustawienie backendu Matplotlib na 'Agg'
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import requests

app = Flask(__name__)

# HTML Template
template = """
<!DOCTYPE html>
<html>
<head>
    <title>Analiza Kryptowalut</title>
</head>
<body>
    <h1>Wybierz Kryptowalutę</h1>
    <form method="post">
        <label for="symbol">Symbol:</label>
        <select name="symbol" id="symbol">
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="DOGEUSDT">DOGE/USDT</option>
            <option value="PEPEUSDT">PEPE/USDT</option>
        </select>
        <label for="interval">Interwał:</label>
        <select name="interval" id="interval">
            <option value="15m">Day Trading (15 minut)</option>
            <option value="1h">Day Trading (1 godzina)</option>
            <option value="4h">Swing Trading (4 godziny)</option>
            <option value="1d">Swing Trading (1 dzień)</option>
            <option value="1w">Long-Term (1 tydzień)</option>
        </select>
        <button type="submit">Analizuj</button>
    </form>

    {% if data %}
        <h2>{{ symbol }} - Analiza Techniczna</h2>
        <p><b>RSI:</b> {{ data.rsi }} - {{ data.rsi_desc }}</p>
        <p><b>MACD:</b> {{ data.macd_diff }} - {{ data.macd_desc }}</p>
        <p><b>ATR:</b> {{ data.atr }} - {{ data.atr_desc }}</p>
        <p><b>Stochastic Oscillator:</b> {{ data.stochastic }} - {{ data.stochastic_desc }}</p>
        <p><b>OBV:</b> {{ data.obv }} - {{ data.obv_desc }}</p>
        <p><b>Ichimoku Cloud:</b> {{ data.ichimoku }} - {{ data.ichimoku_desc }}</p>
        <p><b>Bollinger Bands:</b> Górne = {{ data.bb_upper }}, Dolne = {{ data.bb_lower }}</p>
        <p><b>EMA20:</b> {{ data.ema20 }}</p>
        <p><b>SMA50:</b> {{ data.sma50 }}</p>

        <h3>Wykres:</h3>
        <img src="/static/{{ data.plot }}" alt="Wykres">
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symbol = request.form["symbol"]
        interval = request.form["interval"]

        # Pobieranie danych z Binance API
        df = fetch_crypto_data(symbol, interval)

        # Analizy techniczne
        analysis = analyze_data(df, symbol)

        return render_template_string(template, symbol=symbol, interval=interval, data=analysis)
    return render_template_string(template)

def fetch_crypto_data(symbol, interval):
    """Fetch historical data from Binance API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    df['High-Low'] = df['high'] - df['low']
    df['High-Close'] = abs(df['high'] - df['close'].shift(1))
    df['Low-Close'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df['ATR']

def analyze_data(df, symbol):
    df["close"] = pd.to_numeric(df["close"])

    # Ensure enough data for Ichimoku
    if len(df) < 52:
        return {
            "error": "Za mało danych, aby obliczyć wszystkie wskaźniki. Upewnij się, że ramka danych zawiera co najmniej 52 okresy."
        }

    # RSI
    rsi = RSIIndicator(df["close"]).rsi()
    rsi_desc = "Wyprzedanie" if rsi.iloc[-1] < 30 else "Wykupienie" if rsi.iloc[-1] > 70 else "Neutralne"

    # MACD
    macd = MACD(df["close"])
    macd_diff = macd.macd_diff()
    macd_desc = "Byczy" if macd_diff.iloc[-1] > 0 else "Niedźwiedzi"

    # ATR (manual calculation)
    atr = calculate_atr(df)
    atr_desc = "Wysoka Zmienność" if atr.iloc[-1] > 1 else "Niska Zmienność"

    # Stochastic Oscillator
    stochastic = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
    stochastic_desc = "Wykupienie" if stochastic.iloc[-1] > 80 else "Wyprzedanie" if stochastic.iloc[-1] < 20 else "Neutralne"

    # On-Balance Volume (OBV)
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    obv_desc = "Akumulacja" if obv.iloc[-1] > 0 else "Dystrybucja"

    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df["high"], low=df["low"], window1=9, window2=26, window3=52)
    ichimoku_a = ichimoku.ichimoku_a()
    ichimoku_b = ichimoku.ichimoku_b()
    ichimoku_desc = "Byczy" if df["close"].iloc[-1] > max(ichimoku_a.iloc[-1], ichimoku_b.iloc[-1]) else "Niedźwiedzi"

    # Bollinger Bands
    bb = BollingerBands(df["close"])
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]

    # EMA & SMA
    ema20 = EMAIndicator(df["close"], window=20).ema_indicator()
    sma50 = SMAIndicator(df["close"], window=50).sma_indicator()

    # Wykres
    plot_name = f"{symbol}_analysis.png"
    plt.figure(figsize=(10, 5))
    plt.plot(df["close"], label="Cena Zamknięcia")
    plt.plot(ema20, label="EMA20", color="orange")
    plt.plot(sma50, label="SMA50", color="green")
    plt.legend()
    plt.title(f"{symbol} - Analiza Techniczna")
    plt.xlabel("Czas")
    plt.ylabel("Cena")
    plt.savefig(f"static/{plot_name}")
    plt.close()

    return {
        "rsi": rsi.iloc[-1],
        "rsi_desc": rsi_desc,
        "macd_diff": macd_diff.iloc[-1],
        "macd_desc": macd_desc,
        "atr": atr.iloc[-1],
        "atr_desc": atr_desc,
        "stochastic": stochastic.iloc[-1],
        "stochastic_desc": stochastic_desc,
        "obv": obv.iloc[-1],
        "obv_desc": obv_desc,
        "ichimoku": ichimoku_desc,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "ema20": ema20.iloc[-1],
        "sma50": sma50.iloc[-1],
        "plot": plot_name,
    }


if __name__ == "__main__":
    app.run(debug=False)
