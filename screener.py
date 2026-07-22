import yfinance as yf
import pandas as pd
import requests
import os
import time
from io import StringIO

def get_nasdaq100_tickers():
    """Δυναμική άντληση των tickers του NASDAQ-100 από τη Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        
        for table in tables:
            # Έλεγχος για πιθανές ονομασίες της στήλης (Symbol ή Ticker)
            col = next((c for c in table.columns if c in ['Symbol', 'Ticker', 'Ticker symbol']), None)
            if col:
                return [str(t).replace('.', '-') for t in table[col].tolist()]
                
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
        msg_part = f"🔹 **{r['Ticker']}**\nΤιμή: ${r['Price']} | Low: ${r['Low']}\nRSI: {r['RSI']} | Volume: {r['Vol_Status']}\nEMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n\n"
        
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
    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")

    if not all([BOT_TOKEN, CHAT_ID]):
        print("Σφάλμα: Λείπουν περιβαλλοντικές μεταβλητές.")
        return

    tickers = get_nasdaq100_tickers()
    if not tickers:
        print("Αποτυχία λήψης tickers. Τερματισμός.")
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
                
            # Υπολογισμός Δεικτών
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Μέσος όγκος 20 ημερών
            df['Avg_Vol_20'] = df['Volume'].rolling(window=20).mean()
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['RSI_14'] = 100 - (100 / (1 + rs))

            latest = df.iloc[-1]
            price = latest['Close']
            low = latest['Low']
            volume = latest['Volume']
            avg_vol = latest['Avg_Vol_20']
            
            sma200 = latest['SMA_200']
            sma50 = latest['SMA_50']
            ema20 = latest['EMA_20']
            rsi = latest['RSI_14']
            
            if pd.isna(sma200) or pd.isna(avg_vol):
                continue

            # --- BUSINESS LOGIC ---
            
            # 1. Μακροπρόθεσμη τάση: Πρέπει να είμαστε πάνω από τον SMA200
            if price < sma200:
                continue
            
            # 2. Το RSI πρέπει να δείχνει διόρθωση, αλλά όχι ακραία υπερπώληση
            if not (35 <= rsi <= 55): # Άνοιξα λίγο το εύρος για να πιάσουμε την αναπήδηση
                continue

            # 3. Συνθήκη Pullback (Έλεγχος του Low)
            # Ελέγχουμε αν το χαμηλό της ημέρας "ακούμπησε" τον EMA20 ή τον SMA50 (ανοχή 1%)
            touched_ema20 = (ema20 * 0.99) <= low <= (ema20 * 1.01)
            touched_sma50 = (sma50 * 0.99) <= low <= (sma50 * 1.01)
            
            # Απαιτούμε το κλείσιμο να είναι Pano από τη στήριξη
            closed_above = (price >= ema20) or (price >= sma50)

            if not ((touched_ema20 or touched_sma50) and closed_above):
                continue

            # 4. Φίλτρο Όγκου (Προαιρετικό, αλλά ενισχύει το σήμα)
            vol_status = "High 🟢" if volume > avg_vol else "Avg ⚪"

            results.append({
                'Ticker': ticker,
                'Price': round(price, 2),
                'Low': round(low, 2),
                'RSI': round(rsi, 2),
                'EMA20': round(ema20, 2),
                'SMA50': round(sma50, 2),
                'Vol_Status': vol_status
            })
        except Exception:
            continue

    send_telegram_chunks(results, BOT_TOKEN, CHAT_ID)

if __name__ == "__main__":
    main()
