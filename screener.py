import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os

tickers = ['AMD', 'NVDA', 'PLTR', 'ARM', 'SOFI', 'OXY', 'MPC', 'COP', 'FANG', 'DINO']

def check_pullback_strategy(ticker_symbol):
    # [Ο ΙΔΙΟΣ ΚΩΔΙΚΑΣ ΟΠΩΣ ΠΡΙΝ]
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        
        if df.empty or len(df) < 200:
            return None
            
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        latest = df.iloc[-1]
        
        price = latest['Close']
        sma200 = latest['SMA_200']
        sma50 = latest['SMA_50']
        ema20 = latest['EMA_20']
        rsi = latest['RSI_14']
        
        if price < sma200:
            return None
            
        near_ema20 = (ema20 * 0.98) <= price <= (ema20 * 1.02)
        near_sma50 = (sma50 * 0.98) <= price <= (sma50 * 1.02)
        between_ma = (sma50 <= price <= ema20) or (ema20 <= price <= sma50)
        
        if not (near_ema20 or near_sma50 or between_ma):
            return None
            
        if not (35 <= rsi <= 45):
            return None
            
        return {
            'Ticker': ticker_symbol,
            'Price': round(price, 2),
            'RSI': round(rsi, 2),
            'EMA20': round(ema20, 2),
            'SMA50': round(sma50, 2)
        }
    except:
        return None

def send_telegram_message(results, bot_token, chat_id):
    if results:
        message = "📊 **Daily Swing Trading Setups**\n\n"
        for r in results:
            message += f"🔹 **{r['Ticker']}**\nΤιμή: ${r['Price']} | RSI: {r['RSI']}\nEMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n\n"
    else:
        message = "📊 **Daily Swing Trading Setups**\n\nΚαμία μετοχή δεν ικανοποιεί τα κριτήρια σήμερα."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

results = []
for t in tickers:
    res = check_pullback_strategy(t)
    if res:
        results.append(res)

# Ανάγνωση μεταβλητών από το περιβάλλον του GitHub
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if BOT_TOKEN and CHAT_ID:
    send_telegram_message(results, BOT_TOKEN, CHAT_ID)