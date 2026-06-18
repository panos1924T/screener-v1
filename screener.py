import yfinance as yf
import pandas as pd
import requests
import os
import time

def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns:
                return [t.replace('.', '-') for t in table['Ticker'].tolist()]
    except Exception as e:
        print(f"Σφάλμα ανάκτησης tickers: {e}")
    return []

def send_telegram_chunks(results, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    if not results:
        message = "📊 **Daily Swing Trading Setups**\n\nΚαμία μετοχή δεν ικανοποιεί τα κριτήρια σήμερα."
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
        return

    current_message = "📊 **Daily Swing Trading Setups**\n\n"
    
    for r in results:
        msg_part = f"🔹 **{r['Ticker']}**\nΤιμή: ${r['Price']} | RSI: {r['RSI']}\nEMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n\n"
        
        if len(current_message) + len(msg_part) > 4000:
            payload = {"chat_id": chat_id, "text": current_message, "parse_mode": "Markdown"}
            requests.post(url, json=payload)
            time.sleep(1)
            current_message = "📊 **Daily Swing Trading Setups (Cont.)**\n\n"
        
        current_message += msg_part
        
    if len(current_message) > 45: 
        payload = {"chat_id": chat_id, "text": current_message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)

def main():
    tickers = get_nasdaq100_tickers()
    if not tickers:
        print("Αποτυχία λήψης tickers.")
        return

    print(f"Λήψη δεδομένων για {len(tickers)} μετοχές...")
    all_data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    
    results = []

    for ticker in tickers:
        try:
            df = all_data[ticker].copy()
            df.dropna(inplace=True)
            
            if df.empty or len(df) < 200:
                continue
                
            # --- PURE PANDAS TECHNICAL INDICATORS ---
            # SMA 200 & 50
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # EMA 20
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # RSI 14 (Wilder's Smoothing)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            # ----------------------------------------

            latest = df.iloc[-1]
            
            price = latest['Close']
            sma200 = latest['SMA_200']
            sma50 = latest['SMA_50']
            ema20 = latest['EMA_20']
            rsi = latest['RSI_14']
            
            if pd.isna(sma200):
                continue

            if price < sma200:
                continue
                
            near_ema20 = (ema20 * 0.98) <= price <= (ema20 * 1.02)
            near_sma50 = (sma50 * 0.98) <= price <= (sma50 * 1.02)
            between_ma = (sma50 <= price <= ema20) or (ema20 <= price <= sma50)
            
            if not (near_ema20 or near_sma50 or between_ma):
                continue
                
            if not (35 <= rsi <= 45):
                continue
                
            results.append({
                'Ticker': ticker,
                'Price': round(price, 2),
                'RSI': round(rsi, 2),
                'EMA20': round(ema20, 2),
                'SMA50': round(sma50, 2)
            })
        except Exception:
            continue

    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")

    if BOT_TOKEN and CHAT_ID:
        send_telegram_chunks(results, BOT_TOKEN, CHAT_ID)
    else:
        print("Σφάλμα: Λείπουν τα περιβαλλοντικά μεταβλητά.")

if __name__ == "__main__":
    main()
