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
import os

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
    <p><b>Bollinger Bands:</b> Górne = {{ data.bb_upper }}, Dolne = {{ data.bb_lower }} - {{ data.bb_desc }}</p>
    <p><b>EMA20 i SMA50:</b> EMA20 = {{ data.ema20 }}, SMA50 = {{ data.sma50 }} - {{ data.ema_sma_desc }}</p>

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
    # RSI
    rsi = RSIIndicator(df["close"]).rsi()
    if rsi.iloc[-1] < 30:
        rsi_desc = (
            "Rynek jest wyprzedany, co może oznaczać, że cena osiągnęła poziom, "
            "na którym popyt przewyższa podaż. Potencjalny sygnał kupna, "
            "ale warto poczekać na dodatkowe potwierdzenia, np. wzrost wolumenu."
        )
    elif rsi.iloc[-1] > 70:
        rsi_desc = (
            "Rynek jest wykupiony, co może oznaczać, że cena osiągnęła poziom, "
            "na którym podaż przewyższa popyt. Może to być potencjalny sygnał sprzedaży, "
            "zwłaszcza jeśli pojawią się dodatkowe oznaki osłabienia trendu."
        )
    else:
        rsi_desc = (
            "Neutralne. Rynek znajduje się w równowadze pomiędzy popytem a podażą. "
            "Brak wyraźnych sygnałów, wskazane jest dalsze obserwowanie rynku."
        )

    # MACD
    macd = MACD(df["close"])
    macd_diff = macd.macd_diff()
    if macd_diff.iloc[-1] > 0:
        macd_desc = (
            "Byczy sygnał. Krótkoterminowy trend wzrostowy jest widoczny, "
            "co może sugerować dalsze wzrosty. Warto obserwować, czy sygnał jest wspierany przez wolumen."
        )
    else:
        macd_desc = (
            "Niedźwiedzi sygnał. Krótkoterminowy trend spadkowy jest widoczny, "
            "co może oznaczać potencjalne dalsze spadki. Rozważ użycie dodatkowych wskaźników, "
            "aby potwierdzić kierunek ruchu ceny."
        )

    # ATR
    atr = calculate_atr(df)
    if atr.iloc[-1] > 1:
        atr_desc = (
            "Wysoka zmienność rynku. Oznacza, że cena może wykonywać większe ruchy, "
            "co stwarza szanse na większe zyski, ale zwiększa ryzyko strat. "
            "W takich warunkach wskazane jest dostosowanie wielkości pozycji."
        )
    else:
        atr_desc = (
            "Niska zmienność rynku. Cena porusza się w wąskim zakresie, "
            "co może oznaczać konsolidację. Potencjalne wybicie z tego zakresu "
            "może sygnalizować początek nowego trendu."
        )

    # Stochastic Oscillator
    stochastic = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
    if stochastic.iloc[-1] < 20:
        stochastic_desc = "Rynek jest wyprzedany. Możliwe odbicie w górę."
    elif stochastic.iloc[-1] > 80:
        stochastic_desc = "Rynek jest wykupiony. Możliwe spadki."
    else:
        stochastic_desc = "Neutralne. Brak wyraźnych sygnałów."

    # OBV
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    if obv.iloc[-1] > obv.iloc[-2]:
        obv_desc = "Dystrybucja kapitału. Wzrost obrotu przy wzrostach cen."
    else:
        obv_desc = "Akumulacja kapitału. Spadek obrotu przy spadkach cen."

    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(df["high"], df["low"])
    ichimoku_cloud = ichimoku.ichimoku_a() - ichimoku.ichimoku_b()
    if ichimoku_cloud.iloc[-1] > 0:
        ichimoku_desc = "Byczy sygnał. Cena powyżej chmury Ichimoku."
    else:
        ichimoku_desc = "Niedźwiedzi sygnał. Cena poniżej chmury Ichimoku."

    # Bollinger Bands
    bb = BollingerBands(df["close"])
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    if df["close"].iloc[-1] > bb_upper:
        bb_desc = (
            "Cena przekroczyła górne pasmo Bollingera. Możliwe odwrócenie trendu, "
            "zwłaszcza jeśli pojawiają się oznaki wykupienia rynku. Uważaj na fałszywe sygnały."
        )
    elif df["close"].iloc[-1] < bb_lower:
        bb_desc = (
            "Cena przekroczyła dolne pasmo Bollingera. Możliwe odbicie w górę, "
            "co sugeruje potencjalne zakupy, ale warto poczekać na dodatkowe potwierdzenia."
        )
    else:
        bb_desc = (
            "Cena w zakresie pasm Bollingera. Rynek porusza się w normalnym zakresie. "
            "Brak wyraźnych sygnałów, wskazane dalsze monitorowanie."
        )

    # EMA & SMA
    ema20 = EMAIndicator(df["close"], window=20).ema_indicator()
    sma50 = SMAIndicator(df["close"], window=50).sma_indicator()
    if ema20.iloc[-1] > sma50.iloc[-1]:
        ema_sma_desc = (
            "EMA20 powyżej SMA50. Byczy sygnał długoterminowy, sugerujący, że cena może nadal rosnąć. "
            "Warto monitorować, czy trend utrzymuje się przy wysokim wolumenie."
        )
    else:
        ema_sma_desc = (
            "EMA20 poniżej SMA50. Niedźwiedzi sygnał długoterminowy, sugerujący potencjalne dalsze spadki. "
            "Rozważ obserwowanie innych wskaźników dla potwierdzenia."
        )

    # Generowanie wykresu
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    plot_name = f"{symbol}_analysis.png"
    plt.figure(figsize=(10, 5))
    plt.plot(df["close"], label="Cena Zamknięcia", color="blue")
    plt.plot(ema20, label="EMA20", color="orange", linestyle="--")
    plt.plot(sma50, label="SMA50", color="green", linestyle="--")
    plt.title(f"{symbol} - Analiza Techniczna")
    plt.xlabel("Czas")
    plt.ylabel("Cena")
    plt.legend()
    plt.savefig(os.path.join(static_dir, plot_name))
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
    "ichimoku": ichimoku_cloud.iloc[-1],
    "ichimoku_desc": ichimoku_desc,
    "bb_upper": bb_upper,
    "bb_lower": bb_lower,
    "bb_desc": bb_desc,
    "ema20": ema20.iloc[-1],
    "sma50": sma50.iloc[-1],
    "ema_sma_desc": ema_sma_desc,
    "plot": plot_name  # Dodanie ścieżki do wykresu
    }  # Upewnij się, że ta klamra zamyka cały słownik

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

