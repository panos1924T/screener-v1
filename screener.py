import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time


# ============================================================
# NASDAQ-100 TICKERS
# Update occasionally
# ============================================================

TICKERS = [
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
# ACCOUNT SETTINGS
# ============================================================

DEFAULT_ACCOUNT_EQUITY = 1000.0
DEFAULT_AVAILABLE_CASH = 1000.0

RISK_PER_TRADE = 0.005       # 0.5%
MAX_OPEN_POSITIONS = 3

MIN_SHARE_SIZE = 0.0001


# ============================================================
# EXECUTION ASSUMPTIONS
#
# Same assumptions used in V8
# ============================================================

SLIPPAGE_PCT = 0.0005        # 0.05%
COMMISSION_PCT = 0.0002      # 0.02% simulated cost


# ============================================================
# STRATEGY SETTINGS - V8
# ============================================================

ATR_PERIOD = 14
SWING_LOOKBACK = 3

ENTRY_ATR_BUFFER = 0.10
SL_ATR_BUFFER = 0.10

MIN_STOP_ATR = 0.50

SMA50_SLOPE_LOOKBACK = 10

RSI_MIN = 35
RSI_MAX = 55

SUPPORT_TOLERANCE = 0.01


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.where(
        delta > 0,
        0.0
    )

    loss = -delta.where(
        delta < 0,
        0.0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# CONFIRMED SWING LOW
# ============================================================

def find_previous_swing_low(df):

    signal_index = len(df) - 1

    latest_candidate = (
        signal_index
        - SWING_LOOKBACK
        - 1
    )

    if latest_candidate < SWING_LOOKBACK:
        return None

    for i in range(
        latest_candidate,
        SWING_LOOKBACK - 1,
        -1
    ):

        current = float(
            df["Low"].iloc[i]
        )

        left = df["Low"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right = df["Low"].iloc[
            i + 1:
            i + SWING_LOOKBACK + 1
        ]

        if (
            current < float(left.min())
            and
            current < float(right.min())
        ):
            return current

    return None


# ============================================================
# QUALITY SCORE
#
# Same logic as V8 ranking
# ============================================================

def calculate_quality_score(
    close,
    high,
    low,
    ema20,
    sma50,
    old_sma50,
    atr
):

    slope_score = (
        sma50
        - old_sma50
    ) / atr

    ema_strength = (
        ema20
        - sma50
    ) / atr

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        close_location = 0

    else:

        close_location = (
            close - low
        ) / candle_range

    quality_score = (
        slope_score
        + ema_strength
        + close_location
    )

    return {
        "Quality_Score": quality_score,
        "Slope_Score": slope_score,
        "EMA_Strength": ema_strength,
        "Close_Location": close_location
    }


# ============================================================
# QQQ MARKET FILTER
# ============================================================

def check_qqq_bull():

    qqq = yf.download(
        "QQQ",
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        repair=True
    )

    if qqq.empty:

        raise RuntimeError(
            "Δεν υπάρχουν δεδομένα QQQ."
        )

    if isinstance(
        qqq.columns,
        pd.MultiIndex
    ):

        qqq.columns = (
            qqq.columns
            .get_level_values(0)
        )

    qqq.dropna(
        inplace=True
    )

    qqq["SMA200"] = (
        qqq["Close"]
        .rolling(200)
        .mean()
    )

    latest = qqq.iloc[-1]

    close = float(
        latest["Close"]
    )

    sma200 = float(
        latest["SMA200"]
    )

    if pd.isna(sma200):

        raise RuntimeError(
            "Δεν υπάρχουν αρκετά QQQ δεδομένα για SMA200."
        )

    return {
        "Bull": close > sma200,
        "Close": close,
        "SMA200": sma200
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(
    text,
    bot_token,
    chat_id
):

    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": text,
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

        print(
            f"Telegram error: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # SECRETS
    # ========================================================

    BOT_TOKEN = os.environ.get(
        "BOT_TOKEN"
    )

    CHAT_ID = os.environ.get(
        "CHAT_ID"
    )

    if not BOT_TOKEN or not CHAT_ID:

        print(
            "ERROR: Λείπουν BOT_TOKEN ή CHAT_ID."
        )

        return

    # ========================================================
    # ACCOUNT VALUES
    #
    # Can optionally be set in GitHub Secrets/Variables.
    # ========================================================

    account_equity = float(
        os.environ.get(
            "ACCOUNT_EQUITY",
            DEFAULT_ACCOUNT_EQUITY
        )
    )

    available_cash = float(
        os.environ.get(
            "AVAILABLE_CASH",
            DEFAULT_AVAILABLE_CASH
        )
    )

    # Example:
    # OPEN_POSITIONS=NVDA,AAPL
    open_positions_text = (
        os.environ.get(
            "OPEN_POSITIONS",
            ""
        )
    )

    open_positions = {
        x.strip().upper()
        for x in
        open_positions_text.split(",")
        if x.strip()
    }

    available_slots = max(
        0,
        MAX_OPEN_POSITIONS
        - len(open_positions)
    )

    print("=" * 70)

    print(
        "DAILY PULLBACK SCANNER V8"
    )

    print("=" * 70)

    print(
        f"Account equity: ${account_equity:.2f}"
    )

    print(
        f"Available cash: ${available_cash:.2f}"
    )

    print(
        f"Open positions: {len(open_positions)}"
    )

    print(
        f"Available slots: {available_slots}"
    )

    print()

    # ========================================================
    # NO AVAILABLE POSITION SLOTS
    # ========================================================

    if available_slots <= 0:

        send_telegram_message(
            (
                "📊 *Daily Pullback Scanner V8*\n\n"
                "Δεν υπάρχουν διαθέσιμα position slots.\n\n"
                f"Open positions: "
                f"{', '.join(sorted(open_positions))}\n"
                f"Max positions: {MAX_OPEN_POSITIONS}"
            ),
            BOT_TOKEN,
            CHAT_ID
        )

        return

    # ========================================================
    # QQQ MARKET REGIME
    # ========================================================

    try:

        qqq = check_qqq_bull()

    except Exception as e:

        print(
            f"QQQ ERROR: {e}"
        )

        return

    print(
        f"QQQ Close: ${qqq['Close']:.2f}"
    )

    print(
        f"QQQ SMA200: ${qqq['SMA200']:.2f}"
    )

    # ========================================================
    # BEAR MARKET = NO LONG TRADES
    # ========================================================

    if not qqq["Bull"]:

        message = (
            "📊 *Daily Pullback Scanner V8*\n\n"
            "🔴 *NO LONG TRADES TODAY*\n\n"
            f"QQQ: ${qqq['Close']:.2f}\n"
            f"SMA200: ${qqq['SMA200']:.2f}\n\n"
            "Το QQQ βρίσκεται κάτω από τον SMA200.\n"
            "Η V8 στρατηγική επιτρέπει long setups "
            "μόνο σε BULL regime."
        )

        send_telegram_message(
            message,
            BOT_TOKEN,
            CHAT_ID
        )

        print(
            "QQQ BEAR regime. No trades."
        )

        return

    print(
        "QQQ BULL regime ✅"
    )

    # ========================================================
    # DOWNLOAD STOCK DATA
    # ========================================================

    print()

    print(
        f"Downloading {len(TICKERS)} stocks..."
    )

    try:

        all_data = yf.download(
            TICKERS,
            period="2y",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            repair=True,
            threads=True
        )

    except Exception as e:

        print(
            f"DOWNLOAD ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    setups = []

    # ========================================================
    # SCREEN EACH TICKER
    # ========================================================

    for ticker in TICKERS:

        try:

            # Already owned/open
            if ticker in open_positions:
                continue

            if (
                ticker
                not in
                all_data.columns.get_level_values(0)
            ):

                continue

            df = (
                all_data[ticker]
                .copy()
            )

            df.dropna(
                inplace=True
            )

            if len(df) < 220:

                continue

            # =================================================
            # INDICATORS
            # =================================================

            df["SMA200"] = (
                df["Close"]
                .rolling(200)
                .mean()
            )

            df["SMA50"] = (
                df["Close"]
                .rolling(50)
                .mean()
            )

            df["EMA20"] = (
                df["Close"]
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            df["RSI14"] = (
                calculate_rsi(
                    df["Close"]
                )
            )

            df["ATR14"] = (
                calculate_atr(
                    df,
                    ATR_PERIOD
                )
            )

            # =================================================
            # LATEST COMPLETED CANDLE
            # =================================================

            latest = df.iloc[-1]

            open_price = float(
                latest["Open"]
            )

            close = float(
                latest["Close"]
            )

            high = float(
                latest["High"]
            )

            low = float(
                latest["Low"]
            )

            sma200 = float(
                latest["SMA200"]
            )

            sma50 = float(
                latest["SMA50"]
            )

            ema20 = float(
                latest["EMA20"]
            )

            rsi = float(
                latest["RSI14"]
            )

            atr = float(
                latest["ATR14"]
            )

            old_sma50 = float(
                df["SMA50"].iloc[
                    -1
                    - SMA50_SLOPE_LOOKBACK
                ]
            )

            values = [
                open_price,
                close,
                high,
                low,
                sma200,
                sma50,
                ema20,
                old_sma50,
                rsi,
                atr
            ]

            if (
                any(
                    pd.isna(x)
                    for x in values
                )
                or atr <= 0
            ):

                continue

            # =================================================
            # FILTER 1:
            # Price > SMA200
            # =================================================

            if close <= sma200:
                continue

            # =================================================
            # FILTER 2:
            # EMA20 > SMA50
            # =================================================

            if ema20 <= sma50:
                continue

            # =================================================
            # FILTER 3:
            # SMA50 rising
            # =================================================

            if sma50 <= old_sma50:
                continue

            # =================================================
            # FILTER 4:
            # RSI 35 - 55
            # =================================================

            if not (
                RSI_MIN
                <= rsi
                <= RSI_MAX
            ):

                continue

            # =================================================
            # FILTER 5:
            # Pullback to EMA20 / SMA50
            # =================================================

            touched_ema20 = (
                ema20
                * (
                    1
                    - SUPPORT_TOLERANCE
                )
                <= low
                <=
                ema20
                * (
                    1
                    + SUPPORT_TOLERANCE
                )
            )

            touched_sma50 = (
                sma50
                * (
                    1
                    - SUPPORT_TOLERANCE
                )
                <= low
                <=
                sma50
                * (
                    1
                    + SUPPORT_TOLERANCE
                )
            )

            if not (
                touched_ema20
                or touched_sma50
            ):

                continue

            # =================================================
            # FILTER 6:
            # Close above support
            # =================================================

            if not (
                close >= ema20
                or
                close >= sma50
            ):

                continue

            # =================================================
            # FILTER 7:
            # Green candle
            # =================================================

            if close <= open_price:
                continue

            # =================================================
            # FILTER 8:
            # Close in upper half
            # =================================================

            candle_range = (
                high - low
            )

            if candle_range <= 0:
                continue

            candle_midpoint = (
                low
                + 0.5
                * candle_range
            )

            if close <= candle_midpoint:
                continue

            # =================================================
            # ENTRY
            #
            # Signal High + 0.10 ATR
            # =================================================

            raw_entry = (
                high
                +
                ENTRY_ATR_BUFFER
                * atr
            )

            # =================================================
            # SWING LOW
            # =================================================

            swing_low = (
                find_previous_swing_low(
                    df
                )
            )

            if swing_low is None:
                continue

            # =================================================
            # STOP LOSS
            #
            # Swing Low - 0.10 ATR
            # =================================================

            raw_sl = (
                swing_low
                -
                SL_ATR_BUFFER
                * atr
            )

            technical_risk = (
                raw_entry
                - raw_sl
            )

            if technical_risk <= 0:
                continue

            # =================================================
            # MINIMUM STOP = 0.50 ATR
            # =================================================

            if technical_risk < (
                MIN_STOP_ATR
                * atr
            ):

                continue

            # =================================================
            # TAKE PROFIT = FIXED 2R
            # =================================================

            raw_tp = (
                raw_entry
                +
                2.0
                * technical_risk
            )

            # =================================================
            # QUALITY SCORE
            # =================================================

            score = (
                calculate_quality_score(
                    close,
                    high,
                    low,
                    ema20,
                    sma50,
                    old_sma50,
                    atr
                )
            )

            setups.append(
                {
                    "Ticker":
                        ticker,

                    "Price":
                        close,

                    "High":
                        high,

                    "Low":
                        low,

                    "ATR":
                        atr,

                    "RSI":
                        rsi,

                    "EMA20":
                        ema20,

                    "SMA50":
                        sma50,

                    "Entry":
                        raw_entry,

                    "SL":
                        raw_sl,

                    "TP":
                        raw_tp,

                    "Risk_Per_Share":
                        technical_risk,

                    "Quality_Score":
                        score[
                            "Quality_Score"
                        ],

                    "Slope_Score":
                        score[
                            "Slope_Score"
                        ],

                    "EMA_Strength":
                        score[
                            "EMA_Strength"
                        ],

                    "Close_Location":
                        score[
                            "Close_Location"
                        ]
                }
            )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # ========================================================
    # NO SETUPS
    # ========================================================

    if not setups:

        message = (
            "📊 *Daily Pullback Scanner V8*\n\n"
            "🟢 QQQ BULL regime\n\n"
            "Δεν βρέθηκε valid setup σήμερα."
        )

        send_telegram_message(
            message,
            BOT_TOKEN,
            CHAT_ID
        )

        print(
            "No valid setups."
        )

        return

    # ========================================================
    # RANK BEST SETUPS
    # ========================================================

    setups = sorted(
        setups,
        key=lambda x:
        x["Quality_Score"],
        reverse=True
    )

    # We can only take as many as free portfolio slots
    setups = setups[
        :available_slots
    ]

    # ========================================================
    # POSITION SIZING
    # ========================================================

    risk_budget = (
        account_equity
        * RISK_PER_TRADE
    )

    remaining_cash = (
        available_cash
    )

    final_setups = []

    for setup in setups:

        entry = (
            setup["Entry"]
        )

        sl = (
            setup["SL"]
        )

        # -----------------------------------------------
        # Simulated fills
        # -----------------------------------------------

        entry_fill = (
            entry
            * (
                1
                + SLIPPAGE_PCT
            )
        )

        stop_fill = (
            sl
            * (
                1
                - SLIPPAGE_PCT
            )
        )

        real_risk_per_share = (
            entry_fill
            - stop_fill
        )

        if real_risk_per_share <= 0:
            continue

        # -----------------------------------------------
        # Shares based on 0.5% account risk
        # -----------------------------------------------

        shares_by_risk = (
            risk_budget
            / real_risk_per_share
        )

        # -----------------------------------------------
        # Shares based on remaining cash
        # -----------------------------------------------

        effective_cost_per_share = (
            entry_fill
            * (
                1
                + COMMISSION_PCT
            )
        )

        shares_by_cash = (
            remaining_cash
            / effective_cost_per_share
        )

        shares = min(
            shares_by_risk,
            shares_by_cash
        )

        shares = np.floor(
            shares
            / MIN_SHARE_SIZE
        ) * MIN_SHARE_SIZE

        if shares < MIN_SHARE_SIZE:
            continue

        position_value = (
            shares
            * entry_fill
        )

        entry_commission = (
            position_value
            * COMMISSION_PCT
        )

        cash_required = (
            position_value
            + entry_commission
        )

        actual_risk = (
            shares
            * real_risk_per_share
        )

        # -----------------------------------------------
        # Reduce remaining simulated cash
        # -----------------------------------------------

        remaining_cash -= (
            cash_required
        )

        setup[
            "Shares"
        ] = (
            shares
        )

        setup[
            "Position_Value"
        ] = (
            position_value
        )

        setup[
            "Planned_Risk"
        ] = (
            actual_risk
        )

        setup[
            "Risk_Pct"
        ] = (
            actual_risk
            / account_equity
            * 100
        )

        final_setups.append(
            setup
        )

        if remaining_cash <= 0:
            break

    # ========================================================
    # NO AFFORDABLE SETUPS
    # ========================================================

    if not final_setups:

        message = (
            "📊 *Daily Pullback Scanner V8*\n\n"
            "🟢 QQQ BULL regime\n\n"
            "Υπάρχουν τεχνικά setups, "
            "αλλά δεν υπάρχει αρκετό διαθέσιμο cash "
            "για σωστό position sizing."
        )

        send_telegram_message(
            message,
            BOT_TOKEN,
            CHAT_ID
        )

        return

    # ========================================================
    # TELEGRAM OUTPUT
    # ========================================================

    message = (
        "📊 *Daily Pullback Scanner V8*\n\n"
        "🟢 *QQQ BULL REGIME*\n"
        f"QQQ: ${qqq['Close']:.2f}\n"
        f"QQQ SMA200: ${qqq['SMA200']:.2f}\n\n"
        f"💼 Account: ${account_equity:.2f}\n"
        f"💵 Available cash: ${available_cash:.2f}\n"
        f"🎯 Risk/trade: {RISK_PER_TRADE * 100:.2f}% "
        f"(≈ ${risk_budget:.2f})\n"
        f"📌 Free slots: {available_slots}/{MAX_OPEN_POSITIONS}\n\n"
    )

    for rank, setup in enumerate(
        final_setups,
        start=1
    ):

        message += (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏆 *#{rank} {setup['Ticker']}*\n\n"

            f"Score: *{setup['Quality_Score']:.2f}*\n"
            f"RSI: {setup['RSI']:.2f}\n"
            f"ATR: ${setup['ATR']:.2f}\n\n"

            f"🎯 *ENTRY:* ${setup['Entry']:.2f}\n"
            f"🛑 *SL:* ${setup['SL']:.2f}\n"
            f"💰 *TP 2R:* ${setup['TP']:.2f}\n\n"

            f"📦 Shares: *{setup['Shares']:.4f}*\n"
            f"💵 Position: ≈ ${setup['Position_Value']:.2f}\n"
            f"⚠️ Planned risk: ≈ ${setup['Planned_Risk']:.2f} "
            f"({setup['Risk_Pct']:.2f}%)\n\n"
        )

    message += (
        "━━━━━━━━━━━━━━━━━━\n\n"
        "📌 *Execution rule*\n"
        "Entry μόνο αν η τιμή φτάσει το Entry.\n"
        "Αν δεν ενεργοποιηθεί μέσα σε 3 trading days, "
        "το setup ακυρώνεται.\n\n"
        "Αν μπεις:\n"
        "• SL στο αναγραφόμενο επίπεδο\n"
        "• TP στο Fixed 2R\n"
        "• Max holding: 10 trading days\n\n"
        "⚠️ Paper trading phase — όχι ακόμη validated για live capital."
    )

    # ========================================================
    # TELEGRAM CHARACTER LIMIT
    # ========================================================

    if len(message) <= 4000:

        send_telegram_message(
            message,
            BOT_TOKEN,
            CHAT_ID
        )

    else:

        # Very unlikely with max 3 setups,
        # but included for safety.

        chunks = []

        current = ""

        for line in message.split("\n"):

            if (
                len(current)
                + len(line)
                + 1
                > 3900
            ):

                chunks.append(
                    current
                )

                current = ""

            current += (
                line
                + "\n"
            )

        if current:

            chunks.append(
                current
            )

        for chunk in chunks:

            send_telegram_message(
                chunk,
                BOT_TOKEN,
                CHAT_ID
            )

            time.sleep(1)

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print()

    print(
        f"Found {len(final_setups)} final setups."
    )

    for setup in final_setups:

        print(
            f"{setup['Ticker']} | "
            f"Score={setup['Quality_Score']:.2f} | "
            f"Entry={setup['Entry']:.2f} | "
            f"SL={setup['SL']:.2f} | "
            f"TP={setup['TP']:.2f} | "
            f"Shares={setup['Shares']:.4f} | "
            f"Risk=${setup['Planned_Risk']:.2f}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
