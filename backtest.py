import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# NASDAQ-100 TICKERS
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
# STRATEGY PARAMETERS
# ============================================================

ENTRY_BUFFER = 0.15

# Swing = χαμηλότερο/υψηλότερο από 3 candles αριστερά και δεξιά
SWING_LOOKBACK = 3

# Stop λίγο κάτω από το Swing Low
SL_BUFFER = 0.05

# Δεν δεχόμαστε setup με μικρότερο theoretical R:R
MIN_RR = 2.0

# Μετά το signal, δίνουμε 3 trading days για να γίνει breakout
ENTRY_VALID_DAYS = 3

# Μετά το πραγματικό entry, το trade μπορεί να μείνει ανοιχτό
# έως 10 trading days
MAX_HOLDING_DAYS = 10


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

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
# PREVIOUS CONFIRMED SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):

    """
    Βρίσκει το πιο πρόσφατο Swing Low που ήταν ήδη
    επιβεβαιωμένο πριν από το signal candle.

    Έτσι αποφεύγουμε look-ahead bias.
    """

    earliest = SWING_LOOKBACK

    # Το swing πρέπει να έχει ήδη SWING_LOOKBACK candles
    # στα δεξιά του πριν φτάσουμε στο signal.
    latest_candidate = signal_index - SWING_LOOKBACK - 1

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_low = float(df["Low"].iloc[i])

        left_lows = df["Low"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_lows = df["Low"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_low < float(left_lows.min())
            and current_low < float(right_lows.min())
        ):
            return {
                "price": current_low,
                "index": i,
                "date": df.index[i]
            }

    return None


# ============================================================
# PREVIOUS CONFIRMED SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):

    """
    Βρίσκει το πιο πρόσφατο επιβεβαιωμένο Swing High
    πριν από το signal candle.
    """

    earliest = SWING_LOOKBACK

    latest_candidate = signal_index - SWING_LOOKBACK - 1

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_high = float(df["High"].iloc[i])

        left_highs = df["High"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_highs = df["High"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_high > float(left_highs.max())
            and current_high > float(right_highs.max())
        ):
            return {
                "price": current_high,
                "index": i,
                "date": df.index[i]
            }

    return None


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    df,
    signal_index,
    entry,
    sl,
    tp
):

    """
    1. Το Entry μπορεί να ενεργοποιηθεί μόνο μέσα στις
       επόμενες ENTRY_VALID_DAYS.

    2. Αν δεν ενεργοποιηθεί -> NO_ENTRY.

    3. Από τη στιγμή που ενεργοποιείται, το trade κρατιέται
       maximum MAX_HOLDING_DAYS.

    4. Αν SL και TP εμφανίζονται στο ίδιο daily candle,
       θεωρούμε συντηρητικά ότι χτυπήθηκε πρώτα το SL.
    """

    risk = entry - sl

    if risk <= 0:
        return None

    # ========================================================
    # FIND ENTRY
    # ========================================================

    first_possible_entry = signal_index + 1

    last_possible_entry = min(
        signal_index + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    entry_index = None

    for i in range(
        first_possible_entry,
        last_possible_entry + 1
    ):

        day_high = float(df["High"].iloc[i])

        if day_high >= entry:
            entry_index = i
            break

    # Setup expired
    if entry_index is None:

        return {
            "Result": "NO_ENTRY",
            "R_Multiple": 0.0,
            "Entry_Index": None,
            "Exit_Index": None,
            "Exit_Price": None
        }

    # ========================================================
    # MANAGE OPEN TRADE
    # ========================================================

    last_holding_index = min(
        entry_index + MAX_HOLDING_DAYS - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        last_holding_index + 1
    ):

        day_high = float(df["High"].iloc[i])
        day_low = float(df["Low"].iloc[i])

        # ----------------------------------------------------
        # Same-day ambiguity
        # ----------------------------------------------------

        if day_low <= sl and day_high >= tp:

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": sl
            }

        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        if day_low <= sl:

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": sl
            }

        # ----------------------------------------------------
        # TAKE PROFIT
        # ----------------------------------------------------

        if day_high >= tp:

            reward = tp - entry
            r_multiple = reward / risk

            return {
                "Result": "WIN",
                "R_Multiple": r_multiple,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": tp
            }

    # ========================================================
    # TIME EXIT
    # ========================================================

    exit_price = float(
        df["Close"].iloc[last_holding_index]
    )

    r_multiple = (
        exit_price - entry
    ) / risk

    return {
        "Result": "TIME_EXIT",
        "R_Multiple": r_multiple,
        "Entry_Index": entry_index,
        "Exit_Index": last_holding_index,
        "Exit_Price": exit_price
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SWING TRADING BACKTEST - V2")
    print("=" * 70)

    print()
    print(f"Entry buffer:          ${ENTRY_BUFFER}")
    print(f"Entry validity:         {ENTRY_VALID_DAYS} trading days")
    print(f"Swing lookback:         {SWING_LOOKBACK}")
    print(f"SL buffer:             ${SL_BUFFER}")
    print(f"Minimum R:R:            {MIN_RR}")
    print(f"Max holding period:     {MAX_HOLDING_DAYS} trading days")
    print()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    tickers = STATIC_FALLBACK_TICKERS

    print(
        f"Downloading {len(tickers)} tickers..."
    )

    try:

        all_data = yf.download(
            tickers,
            period="5y",
            group_by="ticker",
            progress=False,

            # IMPORTANT:
            # adjusted OHLC for splits/corporate actions
            auto_adjust=True,

            threads=True
        )

    except Exception as e:

        print(
            f"DOWNLOAD ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    all_trades = []

    # ========================================================
    # EACH TICKER
    # ========================================================

    for ticker in tickers:

        try:

            if ticker not in all_data.columns.get_level_values(0):

                print(
                    f"{ticker}: no data"
                )

                continue

            df = all_data[ticker].copy()

            df.dropna(inplace=True)

            if len(df) < 250:

                print(
                    f"{ticker}: insufficient data"
                )

                continue

            # =================================================
            # INDICATORS
            # =================================================

            df["SMA_200"] = (
                df["Close"]
                .rolling(200)
                .mean()
            )

            df["SMA_50"] = (
                df["Close"]
                .rolling(50)
                .mean()
            )

            df["EMA_20"] = (
                df["Close"]
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            df["Avg_Vol_20"] = (
                df["Volume"]
                .rolling(20)
                .mean()
            )

            df["RSI_14"] = calculate_rsi(
                df["Close"],
                14
            )

            # =================================================
            # WALK FORWARD
            # =================================================

            signal_index = 210

            while signal_index < len(df) - 1:

                row = df.iloc[signal_index]

                price = float(row["Close"])
                high = float(row["High"])
                low = float(row["Low"])

                volume = float(row["Volume"])
                avg_vol = float(row["Avg_Vol_20"])

                sma200 = float(row["SMA_200"])
                sma50 = float(row["SMA_50"])
                ema20 = float(row["EMA_20"])
                rsi = float(row["RSI_14"])

                # ------------------------------------------------
                # NaN
                # ------------------------------------------------

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
                    signal_index += 1
                    continue

                # =================================================
                # SIGNAL RULES
                # =================================================

                # Long-term bullish trend
                if price < sma200:
                    signal_index += 1
                    continue

                # Pullback RSI
                if not (35 <= rsi <= 55):
                    signal_index += 1
                    continue

                # Touch EMA20
                touched_ema20 = (
                    ema20 * 0.99
                    <= low
                    <= ema20 * 1.01
                )

                # Touch SMA50
                touched_sma50 = (
                    sma50 * 0.99
                    <= low
                    <= sma50 * 1.01
                )

                # Close back above support
                closed_above = (
                    price >= ema20
                    or price >= sma50
                )

                if not (
                    (touched_ema20 or touched_sma50)
                    and closed_above
                ):
                    signal_index += 1
                    continue

                # =================================================
                # VOLUME
                # =================================================

                if volume > avg_vol:
                    volume_status = "HIGH"
                else:
                    volume_status = "AVERAGE"

                # =================================================
                # ENTRY
                # =================================================

                entry = high + ENTRY_BUFFER

                # =================================================
                # SWING LOW
                # =================================================

                swing_low_data = find_previous_swing_low(
                    df,
                    signal_index
                )

                if swing_low_data is None:
                    signal_index += 1
                    continue

                swing_low = swing_low_data["price"]

                # =================================================
                # STOP LOSS
                # =================================================

                sl = swing_low - SL_BUFFER

                if sl >= entry:
                    signal_index += 1
                    continue

                # =================================================
                # SWING HIGH
                # =================================================

                swing_high_data = find_previous_swing_high(
                    df,
                    signal_index
                )

                if swing_high_data is None:
                    signal_index += 1
                    continue

                swing_high = swing_high_data["price"]

                # =================================================
                # TAKE PROFIT
                # =================================================

                tp = swing_high

                if tp <= entry:
                    signal_index += 1
                    continue

                # =================================================
                # RISK / REWARD
                # =================================================

                risk = entry - sl
                reward = tp - entry

                if risk <= 0:
                    signal_index += 1
                    continue

                rr = reward / risk

                if rr < MIN_RR:
                    signal_index += 1
                    continue

                # =================================================
                # SIMULATE
                # =================================================

                trade_result = simulate_trade(
                    df=df,
                    signal_index=signal_index,
                    entry=entry,
                    sl=sl,
                    tp=tp
                )

                if trade_result is None:
                    signal_index += 1
                    continue

                # =================================================
                # SETUP EXPIRED WITHOUT ENTRY
                # =================================================

                if trade_result["Result"] == "NO_ENTRY":

                    # Δεν υπήρξε trade.
                    #
                    # Προχωράμε μετά το παράθυρο όπου
                    # περιμέναμε το breakout.

                    signal_index += ENTRY_VALID_DAYS + 1
                    continue

                # =================================================
                # ACTUAL TRADE
                # =================================================

                entry_index = trade_result["Entry_Index"]
                exit_index = trade_result["Exit_Index"]

                entry_date = df.index[entry_index]
                exit_date = df.index[exit_index]

                holding_days = (
                    exit_index
                    - entry_index
                    + 1
                )

                all_trades.append(
                    {
                        "Ticker": ticker,

                        "Signal_Date": df.index[signal_index],
                        "Entry_Date": entry_date,
                        "Exit_Date": exit_date,

                        "Signal_Close": round(price, 4),
                        "Signal_High": round(high, 4),
                        "Signal_Low": round(low, 4),

                        "Entry": round(entry, 4),
                        "SL": round(sl, 4),
                        "TP": round(tp, 4),
                        "Exit_Price": round(
                            trade_result["Exit_Price"],
                            4
                        ),

                        "Swing_Low": round(
                            swing_low,
                            4
                        ),

                        "Swing_High": round(
                            swing_high,
                            4
                        ),

                        "Risk": round(risk, 4),
                        "Reward": round(reward, 4),

                        "RR": round(rr, 4),

                        "RSI": round(rsi, 2),

                        "Volume_Status": volume_status,

                        "Holding_Days": holding_days,

                        "Result": trade_result["Result"],

                        "R": round(
                            trade_result["R_Multiple"],
                            4
                        )
                    }
                )

                # =================================================
                # NO OVERLAPPING TRADES
                # =================================================

                # Εφόσον έχουμε ανοιχτό trade έως το exit_index,
                # δεν ψάχνουμε νέα signals στην ίδια μετοχή
                # όσο το trade είναι ανοιχτό.

                signal_index = exit_index + 1

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print()
        print("NO TRADES FOUND.")

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    trades = pd.DataFrame(all_trades)

    # Sort chronologically across ALL companies
    trades = trades.sort_values(
        ["Entry_Date", "Ticker"]
    ).reset_index(drop=True)

    # ========================================================
    # RESULTS
    # ========================================================

    total_trades = len(trades)

    wins = (
        trades["Result"] == "WIN"
    ).sum()

    losses = (
        trades["Result"] == "LOSS"
    ).sum()

    time_exits = (
        trades["Result"] == "TIME_EXIT"
    ).sum()

    positive_trades = (
        trades["R"] > 0
    ).sum()

    negative_trades = (
        trades["R"] < 0
    ).sum()

    breakeven_trades = (
        trades["R"] == 0
    ).sum()

    # Traditional TP win-rate
    win_rate = (
        wins / total_trades * 100
    )

    # Any trade that actually made money,
    # including positive time exits
    profitable_trade_rate = (
        positive_trades
        / total_trades
        * 100
    )

    total_r = trades["R"].sum()

    average_r = trades["R"].mean()

    median_r = trades["R"].median()

    positive = trades[
        trades["R"] > 0
    ]

    negative = trades[
        trades["R"] < 0
    ]

    avg_positive_r = (
        positive["R"].mean()
        if not positive.empty
        else 0
    )

    avg_negative_r = (
        negative["R"].mean()
        if not negative.empty
        else 0
    )

    gross_profit = (
        positive["R"].sum()
        if not positive.empty
        else 0
    )

    gross_loss = (
        abs(negative["R"].sum())
        if not negative.empty
        else 0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # ========================================================
    # EQUITY CURVE IN R
    # ========================================================

    trades["Cumulative_R"] = (
        trades["R"].cumsum()
    )

    # Include starting equity = 0R
    running_peak = (
        trades["Cumulative_R"]
        .clip(lower=0)
        .cummax()
    )

    trades["Peak_R"] = running_peak

    trades["Drawdown_R"] = (
        trades["Cumulative_R"]
        - trades["Peak_R"]
    )

    max_drawdown_r = (
        trades["Drawdown_R"].min()
    )

    # ========================================================
    # CONSECUTIVE NEGATIVE TRADES
    # ========================================================

    max_consecutive_losses = 0
    current_losses = 0

    for r in trades["R"]:

        if r < 0:

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    # ========================================================
    # HOLDING PERIOD
    # ========================================================

    average_holding_days = (
        trades["Holding_Days"].mean()
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print("BACKTEST RESULTS - V2")
    print("=" * 70)

    print(
        f"Total trades:             {total_trades}"
    )

    print(
        f"TP wins:                  {wins}"
    )

    print(
        f"SL losses:                {losses}"
    )

    print(
        f"Time exits:               {time_exits}"
    )

    print()

    print(
        f"TP win rate:              {win_rate:.2f}%"
    )

    print(
        f"Profitable trades:        {profitable_trade_rate:.2f}%"
    )

    print()

    print(
        f"Total R:                  {total_r:.2f}R"
    )

    print(
        f"Average R/trade:          {average_r:.3f}R"
    )

    print(
        f"Median R/trade:           {median_r:.3f}R"
    )

    print(
        f"Average positive trade:   {avg_positive_r:.3f}R"
    )

    print(
        f"Average negative trade:   {avg_negative_r:.3f}R"
    )

    print(
        f"Profit factor:            {profit_factor:.2f}"
    )

    print()

    print(
        f"Max drawdown:             {max_drawdown_r:.2f}R"
    )

    print(
        f"Max consecutive losses:   {max_consecutive_losses}"
    )

    print(
        f"Average holding days:     {average_holding_days:.2f}"
    )

    # ========================================================
    # EXTREME TRADES
    # ========================================================

    print()
    print("=" * 70)
    print("EXTREME TRADES CHECK")
    print("=" * 70)

    print()
    print("Largest theoretical R:R setups:")

    largest_rr = trades.nlargest(
        10,
        "RR"
    )[
        [
            "Ticker",
            "Signal_Date",
            "Entry",
            "SL",
            "TP",
            "RR",
            "Result",
            "R"
        ]
    ]

    print(
        largest_rr.to_string(index=False)
    )

    print()
    print("Largest winning trades:")

    largest_winners = trades.nlargest(
        10,
        "R"
    )[
        [
            "Ticker",
            "Signal_Date",
            "Entry_Date",
            "Exit_Date",
            "Entry",
            "SL",
            "TP",
            "Result",
            "R"
        ]
    ]

    print(
        largest_winners.to_string(index=False)
    )

    # ========================================================
    # PER TICKER
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS BY TICKER")
    print("=" * 70)

    ticker_summary = (
        trades
        .groupby("Ticker")
        .agg(
            Trades=("R", "count"),
            Total_R=("R", "sum"),
            Avg_R=("R", "mean"),
            Median_R=("R", "median"),
            TP_Wins=(
                "Result",
                lambda x: (x == "WIN").sum()
            ),
            SL_Losses=(
                "Result",
                lambda x: (x == "LOSS").sum()
            ),
            Time_Exits=(
                "Result",
                lambda x: (x == "TIME_EXIT").sum()
            )
        )
    )

    ticker_summary["Profitable_%"] = (
        trades
        .assign(Profitable=trades["R"] > 0)
        .groupby("Ticker")["Profitable"]
        .mean()
        * 100
    )

    ticker_summary = ticker_summary.sort_values(
        "Total_R",
        ascending=False
    )

    print(
        ticker_summary.round(2)
    )

    # ========================================================
    # VOLUME COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("HIGH VOLUME VS AVERAGE VOLUME")
    print("=" * 70)

    volume_summary = (
        trades
        .groupby("Volume_Status")
        .agg(
            Trades=("R", "count"),
            Total_R=("R", "sum"),
            Avg_R=("R", "mean"),
            Median_R=("R", "median")
        )
    )

    print(
        volume_summary.round(3)
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    trades.to_csv(
        "backtest_trades_v2.csv",
        index=False
    )

    ticker_summary.to_csv(
        "backtest_by_ticker_v2.csv"
    )

    volume_summary.to_csv(
        "backtest_by_volume_v2.csv"
    )

    print()
    print("=" * 70)

    print("Saved:")
    print("  backtest_trades_v2.csv")
    print("  backtest_by_ticker_v2.csv")
    print("  backtest_by_volume_v2.csv")

    print()
    print("Backtest V2 completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
