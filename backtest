```python
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time


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

SWING_LOOKBACK = 3

SL_BUFFER = 0.05

MIN_RR = 2.0

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

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# FIND CONFIRMED SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):

    start = SWING_LOOKBACK

    end = signal_index - SWING_LOOKBACK

    if end <= start:
        return None

    for i in range(end - 1, start - 1, -1):

        current_low = float(df["Low"].iloc[i])

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
            return current_low

    return None


# ============================================================
# FIND CONFIRMED SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):

    start = SWING_LOOKBACK

    end = signal_index - SWING_LOOKBACK

    if end <= start:
        return None

    for i in range(end - 1, start - 1, -1):

        current_high = float(df["High"].iloc[i])

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
            return current_high

    return None


# ============================================================
# CHECK TRADE RESULT
# ============================================================

def simulate_trade(
    df,
    signal_index,
    entry,
    sl,
    tp
):
    """
    Simulates what happens AFTER the signal candle.

    Entry is activated only if a future day's HIGH reaches Entry.

    Once entry occurs:

        SL hit first  -> LOSS (-1R)
        TP hit first  -> WIN (+R)
        Neither       -> exit after MAX_HOLDING_DAYS

    IMPORTANT:
    If both SL and TP are touched during the same candle,
    we use the conservative assumption that SL was hit first.
    """

    entry_index = None

    # --------------------------------------------------------
    # WAIT FOR ENTRY
    # --------------------------------------------------------

    for i in range(
        signal_index + 1,
        len(df)
    ):

        high = float(df["High"].iloc[i])

        if high >= entry:

            entry_index = i

            break

    # Entry never happened
    if entry_index is None:

        return {
            "Result": "NO_ENTRY",
            "R_Multiple": 0.0,
            "Entry_Index": None,
            "Exit_Index": None
        }

    # --------------------------------------------------------
    # AFTER ENTRY
    # --------------------------------------------------------

    risk = entry - sl

    max_exit_index = min(
        entry_index + MAX_HOLDING_DAYS - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        max_exit_index + 1
    ):

        day_high = float(df["High"].iloc[i])
        day_low = float(df["Low"].iloc[i])

        # ----------------------------------------------------
        # BOTH SL AND TP HIT SAME DAY
        # Conservative assumption:
        # SL happens first.
        # ----------------------------------------------------

        if day_low <= sl and day_high >= tp:

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i
            }

        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        if day_low <= sl:

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i
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
                "Exit_Index": i
            }

    # --------------------------------------------------------
    # TIME EXIT
    # --------------------------------------------------------

    exit_price = float(
        df["Close"].iloc[max_exit_index]
    )

    r_multiple = (
        exit_price - entry
    ) / risk

    return {
        "Result": "TIME_EXIT",
        "R_Multiple": r_multiple,
        "Entry_Index": entry_index,
        "Exit_Index": max_exit_index
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SWING TRADING BACKTEST")
    print("=" * 70)

    print()
    print(f"Entry buffer:       ${ENTRY_BUFFER}")
    print(f"Swing lookback:     {SWING_LOOKBACK}")
    print(f"SL buffer:           ${SL_BUFFER}")
    print(f"Minimum R:R:         {MIN_RR}")
    print(f"Max holding days:    {MAX_HOLDING_DAYS}")
    print()

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

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
            auto_adjust=False,
            threads=True
        )

    except Exception as e:

        print(
            f"ERROR downloading data: "
            f"{type(e).__name__}: {e}"
        )

        return

    # --------------------------------------------------------
    # ALL TRADES
    # --------------------------------------------------------

    all_trades = []

    # ========================================================
    # LOOP THROUGH TICKERS
    # ========================================================

    for ticker in tickers:

        try:

            # ------------------------------------------------
            # CHECK DATA
            # ------------------------------------------------

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

            # ------------------------------------------------
            # INDICATORS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # WALK FORWARD THROUGH HISTORY
            # ------------------------------------------------

            # Start after enough data exists for indicators.
            #
            # We intentionally stop before the very end because
            # a future period is required to simulate the trade.

            first_index = 210

            last_index = (
                len(df)
                - MAX_HOLDING_DAYS
                - 1
            )

            for signal_index in range(
                first_index,
                last_index
            ):

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
                # CHECK NaN
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
                    continue

                # =================================================
                # ORIGINAL SCREENING RULES
                # =================================================

                # 1. Price above SMA200

                if price < sma200:
                    continue

                # 2. RSI pullback

                if not (35 <= rsi <= 55):
                    continue

                # 3. Pullback to EMA20 or SMA50

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

                # 4. Close above support

                closed_above = (
                    price >= ema20
                    or price >= sma50
                )

                if not (
                    (touched_ema20 or touched_sma50)
                    and closed_above
                ):
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

                swing_low = find_previous_swing_low(
                    df,
                    signal_index
                )

                if swing_low is None:
                    continue

                # =================================================
                # STOP LOSS
                # =================================================

                sl = swing_low - SL_BUFFER

                if sl >= entry:
                    continue

                # =================================================
                # SWING HIGH
                # =================================================

                swing_high = find_previous_swing_high(
                    df,
                    signal_index
                )

                if swing_high is None:
                    continue

                # =================================================
                # TAKE PROFIT
                # =================================================

                tp = swing_high

                if tp <= entry:
                    continue

                # =================================================
                # RISK / REWARD
                # =================================================

                risk = entry - sl

                reward = tp - entry

                rr = reward / risk

                if rr < MIN_RR:
                    continue

                # =================================================
                # SIMULATE TRADE
                # =================================================

                trade_result = simulate_trade(
                    df=df,
                    signal_index=signal_index,
                    entry=entry,
                    sl=sl,
                    tp=tp
                )

                # ------------------------------------------------
                # NO ENTRY = NOT A TRADE
                # ------------------------------------------------

                if trade_result["Result"] == "NO_ENTRY":
                    continue

                # ------------------------------------------------
                # STORE TRADE
                # ------------------------------------------------

                entry_date = df.index[
                    trade_result["Entry_Index"]
                ]

                exit_date = df.index[
                    trade_result["Exit_Index"]
                ]

                all_trades.append(
                    {
                        "Ticker": ticker,
                        "Signal_Date": df.index[signal_index],
                        "Entry_Date": entry_date,
                        "Exit_Date": exit_date,

                        "Signal_Close": price,
                        "Entry": entry,
                        "SL": sl,
                        "TP": tp,

                        "Risk": risk,
                        "Reward": reward,
                        "RR": rr,

                        "RSI": rsi,
                        "Volume_Status": volume_status,

                        "Result": trade_result["Result"],
                        "R": trade_result["R_Multiple"]
                    }
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    if not all_trades:

        print()
        print("NO TRADES FOUND.")

        return

    trades = pd.DataFrame(all_trades)

    # ========================================================
    # BASIC STATISTICS
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

    closed_trades = wins + losses + time_exits

    win_rate = (
        wins / closed_trades * 100
        if closed_trades > 0
        else 0
    )

    # ========================================================
    # R STATISTICS
    # ========================================================

    total_R = trades["R"].sum()

    average_R = trades["R"].mean()

    winning_trades = trades[
        trades["R"] > 0
    ]

    losing_trades = trades[
        trades["R"] < 0
    ]

    average_win_R = (
        winning_trades["R"].mean()
        if not winning_trades.empty
        else 0
    )

    average_loss_R = (
        losing_trades["R"].mean()
        if not losing_trades.empty
        else 0
    )

    gross_profit = (
        winning_trades["R"].sum()
        if not winning_trades.empty
        else 0
    )

    gross_loss = abs(
        losing_trades["R"].sum()
    ) if not losing_trades.empty else 0

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    trades["Cumulative_R"] = (
        trades["R"].cumsum()
    )

    trades["Peak_R"] = (
        trades["Cumulative_R"]
        .cummax()
    )

    trades["Drawdown_R"] = (
        trades["Cumulative_R"]
        - trades["Peak_R"]
    )

    max_drawdown_R = (
        trades["Drawdown_R"].min()
    )

    # ========================================================
    # CONSECUTIVE LOSSES
    # ========================================================

    max_consecutive_losses = 0

    current_losses = 0

    for result in trades["Result"]:

        if result == "LOSS":

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(
        f"Total trades:          {total_trades}"
    )

    print(
        f"Wins:                  {wins}"
    )

    print(
        f"Losses:                {losses}"
    )

    print(
        f"Time exits:            {time_exits}"
    )

    print(
        f"Win rate:              {win_rate:.2f}%"
    )

    print()

    print(
        f"Total R:               {total_R:.2f}R"
    )

    print(
        f"Average R/trade:       {average_R:.3f}R"
    )

    print(
        f"Average winning R:     {average_win_R:.3f}R"
    )

    print(
        f"Average losing R:      {average_loss_R:.3f}R"
    )

    print(
        f"Profit factor:         {profit_factor:.2f}"
    )

    print()

    print(
        f"Max drawdown:          {max_drawdown_R:.2f}R"
    )

    print(
        f"Max consecutive losses:{max_consecutive_losses}"
    )

    print("=" * 70)

    # ========================================================
    # PER-TICKER RESULTS
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
            Wins=("Result", lambda x: (x == "WIN").sum()),
            Losses=("Result", lambda x: (x == "LOSS").sum())
        )
        .sort_values(
            "Total_R",
            ascending=False
        )
    )

    ticker_summary["Win_Rate_%"] = (
        ticker_summary["Wins"]
        / ticker_summary["Trades"]
        * 100
    )

    print(
        ticker_summary.round(2)
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    trades.to_csv(
        "backtest_trades.csv",
        index=False
    )

    ticker_summary.to_csv(
        "backtest_by_ticker.csv"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  backtest_trades.csv"
    )

    print(
        "  backtest_by_ticker.csv"
    )

    print()
    print(
        "Backtest completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
```
