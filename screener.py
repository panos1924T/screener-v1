```python
import yfinance as yf
import pandas as pd
import requests
import os
import time


# ============================================================
# NASDAQ-100 TICKERS
# Snapshot as of Jan 2026
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


# ============================================================
# STRATEGY SETTINGS
# ============================================================

ENTRY_BUFFER = 0.15          # Entry = Signal High + $0.15
SWING_LOOKBACK = 3            # Bars left/right for swing detection
SL_BUFFER = 0.05              # SL placed $0.05 below swing low
MIN_RR = 2.0                  # Minimum acceptable Risk/Reward


# ============================================================
# FIND SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):
    """
    Finds the most recent confirmed swing low BEFORE the signal candle.

    A swing low is a candle whose Low is lower than the lows of
    SWING_LOOKBACK candles before and after it.

    IMPORTANT:
    We only use swings that were already confirmed BEFORE the
    signal candle. Therefore there is no look-ahead for the signal.
    """

    start = SWING_LOOKBACK
    end = signal_index - SWING_LOOKBACK

    if end <= start:
        return None

    for i in range(end - 1, start - 1, -1):

        current_low = df["Low"].iloc[i]

        left_lows = df["Low"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_lows = df["Low"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_low < left_lows.min()
            and current_low < right_lows.min()
        ):
            return float(current_low)

    return None


# ============================================================
# FIND PREVIOUS SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):
    """
    Finds the most recent confirmed swing high BEFORE the signal candle.

    A swing high is a candle whose High is higher than the highs of
    SWING_LOOKBACK candles before and after it.

    IMPORTANT:
    Only confirmed swings before the signal candle are used.
    """

    start = SWING_LOOKBACK
    end = signal_index - SWING_LOOKBACK

    if end <= start:
        return None

    for i in range(end - 1, start - 1, -1):

        current_high = df["High"].iloc[i]

        left_highs = df["High"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_highs = df["High"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_high > left_highs.max()
            and current_high > right_highs.max()
        ):
            return float(current_high)

    return None


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
    # SORT BY R:R
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["RR"],
        reverse=True
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    current_message = (
        "📊 **Daily Swing Trading Setups**\n\n"
        "🎯 Strategy: Breakout\n"
        "Minimum R:R: 2.0\n\n"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    for r in results:

        msg_part = (
            f"🔹 **{r['Ticker']}**\n"
            f"Τιμή: ${r['Price']} | High: ${r['High']} | Low: ${r['Low']}\n"
            f"RSI: {r['RSI']} | Volume: {r['Vol_Status']}\n"
            f"EMA20: ${r['EMA20']} | SMA50: ${r['SMA50']}\n"
            f"\n"
            f"🎯 Entry: ${r['Entry']}\n"
            f"🛑 SL: ${r['SL']}\n"
            f"💰 TP: ${r['TP']}\n"
            f"📉 Risk: ${r['Risk']}\n"
            f"📈 Reward: ${r['Reward']}\n"
            f"⚖️ R:R: **1:{r['RR']}**\n"
            f"\n"
            f"🟢 **SETUP VALID**\n\n"
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

    tickers = STATIC_FALLBACK_TICKERS

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
            # LATEST COMPLETED CANDLE
            # -----------------------------------------------

            signal_index = len(df) - 1

            latest = df.iloc[signal_index]

            price = float(latest["Close"])
            high = float(latest["High"])
            low = float(latest["Low"])
            volume = float(latest["Volume"])
            avg_vol = float(latest["Avg_Vol_20"])

            sma200 = float(latest["SMA_200"])
            sma50 = float(latest["SMA_50"])
            ema20 = float(latest["EMA_20"])
            rsi = float(latest["RSI_14"])

            # -----------------------------------------------
            # CHECK NaN
            # -----------------------------------------------

            if any(
                pd.isna(x)
                for x in [
                    price,
                    high,
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
            # EXISTING BUSINESS LOGIC
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

            # =================================================
            # NEW SWING TRADE LOGIC
            # =================================================

            # -----------------------------------------------
            # 5. ENTRY
            # Signal High + $0.15
            # -----------------------------------------------

            entry = high + ENTRY_BUFFER

            # -----------------------------------------------
            # 6. PREVIOUS SWING LOW
            # -----------------------------------------------

            swing_low = find_previous_swing_low(
                df,
                signal_index
            )

            if swing_low is None:
                print(
                    f"{ticker}: Δεν βρέθηκε Swing Low."
                )
                continue

            # -----------------------------------------------
            # 7. STOP LOSS
            #
            # Below the recent swing low.
            # -----------------------------------------------

            sl = swing_low - SL_BUFFER

            # -----------------------------------------------
            # Safety:
            # SL must be below Entry.
            # -----------------------------------------------

            if sl >= entry:
                print(
                    f"{ticker}: Invalid SL."
                )
                continue

            # -----------------------------------------------
            # 8. PREVIOUS SWING HIGH
            # -----------------------------------------------

            swing_high = find_previous_swing_high(
                df,
                signal_index
            )

            if swing_high is None:
                print(
                    f"{ticker}: Δεν βρέθηκε Swing High."
                )
                continue

            # -----------------------------------------------
            # 9. TAKE PROFIT
            #
            # Previous Swing High
            # -----------------------------------------------

            tp = swing_high

            # -----------------------------------------------
            # TP must be above Entry
            # -----------------------------------------------

            if tp <= entry:
                print(
                    f"{ticker}: Previous Swing High "
                    f"is below Entry."
                )
                continue

            # -----------------------------------------------
            # 10. RISK
            # -----------------------------------------------

            risk = entry - sl

            # -----------------------------------------------
            # 11. REWARD
            # -----------------------------------------------

            reward = tp - entry

            # -----------------------------------------------
            # 12. RISK / REWARD
            # -----------------------------------------------

            rr = reward / risk

            # -----------------------------------------------
            # 13. MINIMUM R:R FILTER
            # -----------------------------------------------

            if rr < MIN_RR:
                print(
                    f"{ticker}: R:R too low "
                    f"({rr:.2f})"
                )
                continue

            # -----------------------------------------------
            # 14. ADD RESULT
            # -----------------------------------------------

            results.append(
                {
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "High": round(high, 2),
                    "Low": round(low, 2),

                    "RSI": round(rsi, 2),
                    "EMA20": round(ema20, 2),
                    "SMA50": round(sma50, 2),

                    "Vol_Status": vol_status,

                    "Swing_Low": round(swing_low, 2),
                    "Swing_High": round(swing_high, 2),

                    "Entry": round(entry, 2),
                    "SL": round(sl, 2),
                    "TP": round(tp, 2),

                    "Risk": round(risk, 2),
                    "Reward": round(reward, 2),
                    "RR": round(rr, 2)
                }
            )

            print(
                f"FOUND: {ticker} | "
                f"Entry={entry:.2f} | "
                f"SL={sl:.2f} | "
                f"TP={tp:.2f} | "
                f"R:R=1:{rr:.2f}"
            )

        except Exception as e:

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
        f"Βρέθηκαν {len(results)} valid setups."
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
```
