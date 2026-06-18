import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os
import time

def get_nasdaq100_tickers():
    """Δυναμική άντληση των tickers του NASDAQ-100"""
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
    """Αποστολή αποτελεσμάτων στο Telegram με διαχείριση ορίου χαρακτήρων"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    if not results:
        message = "📊 **Daily Swing Trading Setups**\n\nΚαμία μετοχή δεν ικανοποιεί τα κριτήρια σήμερα."
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
        return

    current_message = "📊 **Daily Swing Trading Setups**\n\n"
    
    for r in results:
        msg_part = f"🔹 **{r['Ticker']}**\nΤιμή: ${r['Price']} | RSI: {r['RSI']}\nEMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n\n"
        
        # Αν το μήνυμα υπερβαίνει το όριο, στείλε το τρέχον block και ξεκίνα νέο
        if len(current_message) + len(msg_part) > 4000:
            payload = {"chat_id": chat_id, "text": current_message, "parse_mode": "Markdown"}
            requests.post(url, json=payload)
            time.sleep(1) # Αποφυγή rate limit από το Telegram API
            current_message = "📊 **Daily Swing Trading Setups (Cont.)**\n\n"
        
        current_message += msg_part
        
    # Αποστολή του τελευταίου τμήματος
    if len(current_message) > 45: 
        payload = {"chat_id": chat_id, "text": current_message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)

def main():
    tickers = get_nasdaq100_tickers()
    if not tickers:
        print("Αποτυχία λήψης tickers.")
        return

    print(f"Λήψη δεδομένων για {len(tickers)} μετοχές...")
    # Λήψη δεδομένων 1 έτους (απαιτείται για τον υπολογισμό του SMA 200)
    all_data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    
    results = []

    for ticker in tickers:
        try:
            # Απομόνωση των δεδομένων της μετοχής και αποφυγή SettingWithCopyWarning
            df = all_data[ticker].copy()
            df.dropna(inplace=True)
            
            if df.empty or len(df) < 200:
                continue
                
            # Υπολογισμός δεικτών μέσω pandas_ta
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
            
            # Έλεγχος αν ο SMA200 είναι NaN 
            if pd.isna(sma200):
                continue

            # Εφαρμογή κριτηρίων στρατηγικής
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
        print("Σφάλμα: Λείπουν τα περιβαλλοντικά μεταβλητά BOT_TOKEN ή CHAT_ID.")

if __name__ == "__main__":
    main()
