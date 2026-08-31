import yfinance as yf
import pandas as pd
import requests
import os
import time
from io import StringIO

# ============================================================
# STATIC FALLBACK — used only if BOTH live sources fail.
# Update this occasionally (e.g. once a quarter) by hand.
# Snapshot as of Nasdaq-100 constituents, Jan 2026.
# ============================================================

STATIC_FALLBACK_TICKERS = [
    "ADBE", "AMD", "ABNB", "ALNY", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN",
    "ADI", "AAPL", "AMAT", "APP", "ARM", "ASML", "ADSK", "ADP", "AXON",
    "BKR", "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH",
    "CMCSA", "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM",
    "FANG", "DASH", "EA", "EXC", "FAST", "FER", "FTNT", "GEHC", "GILD",
    "HON", "IDXX", "INSM", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC",
    "LRCX", "LIN", "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT",
    "MSTR", "MDLZ", "MPWR", "MNST", "NFLX", "NVDA", "NXPI", "ORLY", "ODFL",
    "PCAR", "PLTR", "PANW", "PAYX", "PYPL", "PDD", "PEP", "QCOM", "REGN",
    "ROP", "ROST", "SNDK", "STX", "SHOP", "SBUX", "SNPS", "TMUS", "TTWO",
    "TSLA", "TXN", "TRI", "VRSK", "VRTX", "WMT", "WBD", "WDC", "WDAY",
    "XEL", "ZS",
]


def get_nasdaq100_tickers():
    """
    Παίρνει δυναμικά τα tickers του NASDAQ-100.
    1η προσπάθεια: επίσημη σελίδα Nasdaq
    2η προσπάθεια: Wikipedia
    3η προσπάθεια: static fallback (bundled in repo)
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # --------------------------------------------------------
    # 1. OFFICIAL NASDAQ
    # --------------------------------------------------------

    try:
        url = "https://www.nasdaq.com/solutions/nasdaq-100/companies"

        response = requests.get(url, headers=headers, timeout=20)

        # DEBUG: always print what we actually got back, even on 200,
        # so a "no suitable table" failure is diagnosable from CI logs.
        print(
            f"Nasdaq: HTTP {response.status_code}, "
            f"{len(response.text)} bytes received."
        )

        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        print(f"Nasdaq: pandas found {len(tables)} <table> elements.")

        for i, table in enumerate(tables):
            print(f"Nasdaq: table {i} columns = {list(table.columns)}")

            symbol_col = None
            for col in table.columns:
                if str(col).strip().lower() in ["symbol", "ticker"]:
                    symbol_col = col
                    break

            if symbol_col is None:
                continue

            tickers = (
                table[symbol_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
                .tolist()
            )

            tickers = [t for t in tickers if t and len(t) <= 10]

            if len(tickers) >= 90:
                print(f"OK: Βρέθηκαν {len(tickers)} NASDAQ-100 tickers από Nasdaq.")
                return tickers

        print("Nasdaq: Δεν βρέθηκε κατάλληλος πίνακας (πιθανό bot-block/JS page).")

    except Exception as e:
        print(f"Nasdaq ticker retrieval failed: {type(e).__name__}: {e}")

    # --------------------------------------------------------
    # 2. WIKIPEDIA FALLBACK
    # --------------------------------------------------------

    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"

        response = requests.get(url, headers=headers, timeout=20)

        print(
            f"Wikipedia: HTTP {response.status_code}, "
            f"{len(response.text)} bytes received."
        )

        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        print(f"Wikipedia: pandas found {len(tables)} <table> elements.")

        for i, table in enumerate(tables):
            print(f"Wikipedia: table {i} columns = {list(table.columns)}")

            symbol_col = None
            for col in table.columns:
                col_name = str(col).strip().lower()
                if col_name in ("ticker", "symbol") or "ticker" in col_name or "symbol" in col_name:
                    symbol_col = col
                    break

            if symbol_col is None:
                continue

            tickers = (
                table[symbol_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
                .tolist()
            )

            tickers = [t for t in tickers if t and len(t) <= 10]

            if len(tickers) >= 90:
                print(f"OK: Βρέθηκαν {len(tickers)} NASDAQ-100 tickers από Wikipedia.")
                return tickers

        print("Wikipedia: Δεν βρέθηκε κατάλληλος πίνακας (πιθανό rate-limit από data-center IP).")

    except Exception as e:
        print(f"Wikipedia ticker retrieval failed: {type(e).__name__}: {e}")

    # --------------------------------------------------------
    # 3. STATIC FALLBACK — guarantees the job doesn't die entirely
    # --------------------------------------------------------

    print(
        f"WARNING: Live scraping failed. Χρήση static fallback λίστας "
        f"({len(STATIC_FALLBACK_TICKERS)} tickers, ενδέχεται να μην είναι 100% ενημερωμένη)."
    )
    return STATIC_FALLBACK_TICKERS
    
# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_chunks(results, bot_token, chat_id):

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        message = (
            "📊 **Daily Swing Trading Setups**\n\n"
            "Καμία μετοχή δεν ικανοποιεί τα κριτήρια σήμερα."
        )

        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=20
            )
            response.raise_for_status()

        except Exception as e:
            print(f"Telegram error: {e}")

        return

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    current_message = (
        "📊 **Daily Swing Trading Setups**\n\n"
    )

    for r in results:

        msg_part = (
            f"🔹 **{r['Ticker']}**\n"
            f"Τιμή: ${r['Price']} | Low: ${r['Low']}\n"
            f"RSI: {r['RSI']} | Volume: {r['Vol_Status']}\n"
            f"EMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n\n"
        )

        # Telegram limit ~4096 characters
        if len(current_message) + len(msg_part) > 4000:

            payload = {
                "chat_id": chat_id,
                "text": current_message,
                "parse_mode": "Markdown"
            }

            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=20
                )
                response.raise_for_status()

            except Exception as e:
                print(f"Telegram error: {e}")

            time.sleep(1)

            current_message = (
                "📊 **Daily Swing Trading Setups (Cont.)**\n\n"
            )

        current_message += msg_part

    # --------------------------------------------------------
    # SEND LAST MESSAGE
    # --------------------------------------------------------

    if len(current_message) > 45:

        payload = {
            "chat_id": chat_id,
            "text": current_message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=20
            )
            response.raise_for_status()

        except Exception as e:
            print(f"Telegram error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():

    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")

    # --------------------------------------------------------
    # ENVIRONMENT VARIABLES
    # --------------------------------------------------------

    if not BOT_TOKEN or not CHAT_ID:

        print(
            "ERROR: Λείπουν BOT_TOKEN ή CHAT_ID "
            "από τα GitHub Secrets."
        )

        return

    # --------------------------------------------------------
    # GET TICKERS
    # --------------------------------------------------------

    tickers = get_nasdaq100_tickers()

    if not tickers:

        print(
            "ERROR: Αποτυχία λήψης tickers. Τερματισμός."
        )

        return

    print(
        f"Λήψη δεδομένων για {len(tickers)} μετοχές..."
    )

    # --------------------------------------------------------
    # YAHOO FINANCE
    # --------------------------------------------------------

    try:

        all_data = yf.download(
            tickers,
            period="2y",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True
        )

    except Exception as e:

        print(
            f"ERROR: yfinance download failed: "
            f"{type(e).__name__}: {e}"
        )

        return

    # --------------------------------------------------------
    # SCREENING
    # --------------------------------------------------------

    results = []

    for ticker in tickers:

        try:

            # -----------------------------------------------
            # GET DATA FOR TICKER
            # -----------------------------------------------

            if ticker not in all_data.columns.get_level_values(0):

                print(
                    f"{ticker}: Δεν υπάρχουν δεδομένα."
                )

                continue

            df = all_data[ticker].copy()

            df.dropna(inplace=True)

            if df.empty or len(df) < 200:

                print(
                    f"{ticker}: Ανεπαρκή δεδομένα."
                )

                continue

            # -----------------------------------------------
            # INDICATORS
            # -----------------------------------------------

            # SMA 200
            df["SMA_200"] = (
                df["Close"]
                .rolling(window=200)
                .mean()
            )

            # SMA 50
            df["SMA_50"] = (
                df["Close"]
                .rolling(window=50)
                .mean()
            )

            # EMA 20
            df["EMA_20"] = (
                df["Close"]
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            # Average Volume 20
            df["Avg_Vol_20"] = (
                df["Volume"]
                .rolling(window=20)
                .mean()
            )

            # -----------------------------------------------
            # RSI 14
            # -----------------------------------------------

            delta = df["Close"].diff()

            gain = delta.where(
                delta > 0,
                0
            )

            loss = -delta.where(
                delta < 0,
                0
            )

            avg_gain = gain.ewm(
                alpha=1 / 14,
                adjust=False
            ).mean()

            avg_loss = loss.ewm(
                alpha=1 / 14,
                adjust=False
            ).mean()

            rs = avg_gain / avg_loss

            df["RSI_14"] = (
                100 - (100 / (1 + rs))
            )

            # -----------------------------------------------
            # LATEST DATA
            # -----------------------------------------------

            latest = df.iloc[-1]

            price = latest["Close"]
            low = latest["Low"]
            volume = latest["Volume"]
            avg_vol = latest["Avg_Vol_20"]

            sma200 = latest["SMA_200"]
            sma50 = latest["SMA_50"]
            ema20 = latest["EMA_20"]
            rsi = latest["RSI_14"]

            # -----------------------------------------------
            # CHECK NaN
            # -----------------------------------------------

            if any(
                pd.isna(x)
                for x in [
                    price,
                    low,
                    volume,
                    avg_vol,
                    sma200,
                    sma50,
                    ema20,
                    rsi
                ]
            ):
                continue

            # =================================================
            # BUSINESS LOGIC
            # =================================================

            # -----------------------------------------------
            # 1. LONG-TERM TREND
            # Price must be above SMA200
            # -----------------------------------------------

            if price < sma200:
                continue

            # -----------------------------------------------
            # 2. RSI PULLBACK
            # -----------------------------------------------

            if not (35 <= rsi <= 55):
                continue

            # -----------------------------------------------
            # 3. PULLBACK TO EMA20 OR SMA50
            # Tolerance = 1%
            # -----------------------------------------------

            touched_ema20 = (
                ema20 * 0.99
                <= low
                <= ema20 * 1.01
            )

            touched_sma50 = (
                sma50 * 0.99
                <= low
                <= sma50 * 1.01
            )

            # -----------------------------------------------
            # Price must close above support
            # -----------------------------------------------

            closed_above = (
                price >= ema20
                or price >= sma50
            )

            if not (
                (touched_ema20 or touched_sma50)
                and closed_above
            ):
                continue

            # -----------------------------------------------
            # 4. VOLUME
            # -----------------------------------------------

            if volume > avg_vol:
                vol_status = "High 🟢"
            else:
                vol_status = "Avg ⚪"

            # -----------------------------------------------
            # ADD RESULT
            # -----------------------------------------------

            results.append(
                {
                    "Ticker": ticker,
                    "Price": round(float(price), 2),
                    "Low": round(float(low), 2),
                    "RSI": round(float(rsi), 2),
                    "EMA20": round(float(ema20), 2),
                    "SMA50": round(float(sma50), 2),
                    "Vol_Status": vol_status
                }
            )

            print(
                f"FOUND: {ticker} | "
                f"Price={price:.2f} | "
                f"RSI={rsi:.2f}"
            )

        except Exception as e:

            # Δεν κρύβουμε πλέον τα errors
            print(
                f"ERROR στο {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"Screening ολοκληρώθηκε. "
        f"Βρέθηκαν {len(results)} setups."
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    send_telegram_chunks(
        results,
        BOT_TOKEN,
        CHAT_ID
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
