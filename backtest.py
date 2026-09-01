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

SWING_LOOKBACK = 3

MIN_RR = 2.0

ENTRY_VALID_DAYS = 3

MAX_HOLDING_DAYS = 10


# ============================================================
# ATR
# ============================================================

ATR_PERIOD = 14

ATR_SL_BUFFER = 0.10

MIN_STOP_ATR = 0.50


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

    high_low = (
        df["High"] - df["Low"]
    )

    high_prev = (
        df["High"] - previous_close
    ).abs()

    low_prev = (
        df["Low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_prev,
            low_prev
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):

    earliest = SWING_LOOKBACK

    latest_candidate = (
        signal_index
        - SWING_LOOKBACK
        - 1
    )

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_low = float(
            df["Low"].iloc[i]
        )

        left = df["Low"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right = df["Low"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_low < float(left.min())
            and
            current_low < float(right.min())
        ):

            return current_low

    return None


# ============================================================
# SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):

    earliest = SWING_LOOKBACK

    latest_candidate = (
        signal_index
        - SWING_LOOKBACK
        - 1
    )

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_high = float(
            df["High"].iloc[i]
        )

        left = df["High"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right = df["High"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current_high > float(left.max())
            and
            current_high > float(right.max())
        ):

            return current_high

    return None


# ============================================================
# FIND ACTUAL ENTRY
# ============================================================

def find_entry(df, signal_index, entry):

    first_index = signal_index + 1

    last_index = min(
        signal_index + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    for i in range(
        first_index,
        last_index + 1
    ):

        if float(df["High"].iloc[i]) >= entry:

            return i

    return None


# ============================================================
# SINGLE TARGET STRATEGY
#
# Used for:
# Swing High
# Fixed 2R
# Fixed 2.5R
# ============================================================

def simulate_single_target(
    df,
    entry_index,
    entry,
    sl,
    target
):

    risk = entry - sl

    target_r = (
        target - entry
    ) / risk

    last_index = min(
        entry_index
        + MAX_HOLDING_DAYS
        - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        last_index + 1
    ):

        high = float(
            df["High"].iloc[i]
        )

        low = float(
            df["Low"].iloc[i]
        )

        # ----------------------------------------------------
        # SL + TP same daily candle
        #
        # We cannot know intraday order.
        # Conservative assumption = SL first.
        # ----------------------------------------------------

        if (
            low <= sl
            and high >= target
        ):

            return {
                "R": -1.0,
                "Result": "SL",
                "Exit_Index": i
            }

        if low <= sl:

            return {
                "R": -1.0,
                "Result": "SL",
                "Exit_Index": i
            }

        if high >= target:

            return {
                "R": target_r,
                "Result": "TP",
                "Exit_Index": i
            }

    # --------------------------------------------------------
    # TIME EXIT
    # --------------------------------------------------------

    exit_price = float(
        df["Close"].iloc[
            last_index
        ]
    )

    r_result = (
        exit_price - entry
    ) / risk

    return {
        "R": r_result,
        "Result": "TIME",
        "Exit_Index": last_index
    }


# ============================================================
# PARTIAL STRATEGY
#
# 50% at 2R
# 50% at Swing High
#
# SL remains unchanged.
# ============================================================

def simulate_partial_strategy(
    df,
    entry_index,
    entry,
    sl,
    swing_high
):

    risk = entry - sl

    tp_2r = (
        entry
        + 2.0 * risk
    )

    swing_rr = (
        swing_high - entry
    ) / risk

    last_index = min(
        entry_index
        + MAX_HOLDING_DAYS
        - 1,
        len(df) - 1
    )

    first_half_closed = False

    for i in range(
        entry_index,
        last_index + 1
    ):

        high = float(
            df["High"].iloc[i]
        )

        low = float(
            df["Low"].iloc[i]
        )

        # ====================================================
        # BEFORE 2R HAS BEEN HIT
        # ====================================================

        if not first_half_closed:

            # SL and 2R occur in same daily candle.
            # Conservative assumption = SL happened first.
            if (
                low <= sl
                and high >= tp_2r
            ):

                return {
                    "R": -1.0,
                    "Result": "SL_BEFORE_PARTIAL",
                    "Exit_Index": i,
                    "Hit_2R": False,
                    "Hit_Swing": False
                }

            if low <= sl:

                return {
                    "R": -1.0,
                    "Result": "SL_BEFORE_PARTIAL",
                    "Exit_Index": i,
                    "Hit_2R": False,
                    "Hit_Swing": False
                }

            # ------------------------------------------------
            # Price reached 2R
            # ------------------------------------------------

            if high >= tp_2r:

                first_half_closed = True

                # 50% position closed at +2R
                #
                # Portfolio contribution:
                #
                # 0.50 × 2R = +1R

                # If the same candle also reaches Swing High,
                # and it did NOT hit our SL, price necessarily
                # passed through 2R first on the way upward.

                if high >= swing_high:

                    total_r = (
                        1.0
                        + 0.50 * swing_rr
                    )

                    return {
                        "R": total_r,
                        "Result": "FULL_TARGET",
                        "Exit_Index": i,
                        "Hit_2R": True,
                        "Hit_Swing": True
                    }

                continue

        # ====================================================
        # AFTER FIRST 50% HAS BEEN SOLD AT 2R
        # ====================================================

        else:

            # Remaining 50%:
            #
            # if SL happens:
            #
            # first half = +1R
            # remaining half loss = -0.5R
            #
            # TOTAL = +0.5R

            if (
                low <= sl
                and high >= swing_high
            ):

                # Daily candle ambiguity.
                # Conservative assumption = SL first.

                return {
                    "R": 0.5,
                    "Result": "PARTIAL_THEN_SL",
                    "Exit_Index": i,
                    "Hit_2R": True,
                    "Hit_Swing": False
                }

            if low <= sl:

                return {
                    "R": 0.5,
                    "Result": "PARTIAL_THEN_SL",
                    "Exit_Index": i,
                    "Hit_2R": True,
                    "Hit_Swing": False
                }

            if high >= swing_high:

                total_r = (
                    1.0
                    + 0.50 * swing_rr
                )

                return {
                    "R": total_r,
                    "Result": "FULL_TARGET",
                    "Exit_Index": i,
                    "Hit_2R": True,
                    "Hit_Swing": True
                }

    # ========================================================
    # TIME EXIT
    # ========================================================

    exit_price = float(
        df["Close"].iloc[
            last_index
        ]
    )

    exit_r = (
        exit_price - entry
    ) / risk

    # --------------------------------------------------------
    # Never reached 2R
    # Full position still open
    # --------------------------------------------------------

    if not first_half_closed:

        return {
            "R": exit_r,
            "Result": "TIME_BEFORE_PARTIAL",
            "Exit_Index": last_index,
            "Hit_2R": False,
            "Hit_Swing": False
        }

    # --------------------------------------------------------
    # 50% already banked at 2R.
    #
    # +1R from first half
    # + 0.50 × current R from remaining half
    # --------------------------------------------------------

    total_r = (
        1.0
        + 0.50 * exit_r
    )

    return {
        "R": total_r,
        "Result": "PARTIAL_TIME",
        "Exit_Index": last_index,
        "Hit_2R": True,
        "Hit_Swing": False
    }


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def calculate_stats(
    trades,
    column_name,
    exit_column
):

    r = trades[
        column_name
    ]

    total = len(r)

    profitable = (
        r > 0
    ).sum()

    losing = (
        r < 0
    ).sum()

    gross_profit = (
        r[r > 0].sum()
    )

    gross_loss = abs(
        r[r < 0].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    cumulative = (
        r.cumsum()
    )

    peak = (
        cumulative
        .clip(lower=0)
        .cummax()
    )

    drawdown = (
        cumulative - peak
    )

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # Consecutive losing trades
    # --------------------------------------------------------

    max_consecutive_losses = 0
    current_losses = 0

    for value in r:

        if value < 0:

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    # --------------------------------------------------------
    # Holding period
    # --------------------------------------------------------

    holding_days = (
        trades[exit_column]
        -
        trades["Entry_Index"]
        + 1
    )

    return {
        "Trades": total,

        "Profitable_%":
            profitable / total * 100,

        "Losing_%":
            losing / total * 100,

        "Total_R":
            r.sum(),

        "Avg_R":
            r.mean(),

        "Median_R":
            r.median(),

        "Profit_Factor":
            profit_factor,

        "Max_Drawdown_R":
            max_drawdown,

        "Max_Consecutive_Losses":
            max_consecutive_losses,

        "Avg_Holding_Days":
            holding_days.mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("PULLBACK BACKTEST V4 - EXIT STRATEGY COMPARISON")
    print("=" * 78)

    print()

    print(
        "Comparing:"
    )

    print(
        "A) Previous Swing High"
    )

    print(
        "B) Fixed 2R"
    )

    print(
        "C) Fixed 2.5R"
    )

    print(
        "D) 50% at 2R + 50% at Swing High"
    )

    print()

    print(
        f"Entry validity:        {ENTRY_VALID_DAYS} days"
    )

    print(
        f"Max holding:           {MAX_HOLDING_DAYS} days"
    )

    print(
        f"ATR minimum stop:      {MIN_STOP_ATR} ATR"
    )

    print(
        f"Minimum Swing R:R:     {MIN_RR}"
    )

    print()

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    tickers = STATIC_FALLBACK_TICKERS

    print(
        f"Downloading {len(tickers)} tickers..."
    )

    try:

        all_data = yf.download(
            tickers,
            period="5y",
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

    all_trades = []

    rejected_small_stop = 0
    rejected_rr = 0
    expired = 0

    # ========================================================
    # EACH TICKER
    # ========================================================

    for ticker in tickers:

        try:

            if (
                ticker
                not in
                all_data.columns.get_level_values(0)
            ):

                continue

            df = all_data[
                ticker
            ].copy()

            df.dropna(
                inplace=True
            )

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

            df["RSI_14"] = (
                calculate_rsi(
                    df["Close"]
                )
            )

            df["ATR_14"] = (
                calculate_atr(
                    df
                )
            )

            # =================================================
            # WALK FORWARD
            # =================================================

            signal_index = 210

            while (
                signal_index
                < len(df) - 1
            ):

                row = df.iloc[
                    signal_index
                ]

                price = float(
                    row["Close"]
                )

                high = float(
                    row["High"]
                )

                low = float(
                    row["Low"]
                )

                volume = float(
                    row["Volume"]
                )

                avg_vol = float(
                    row["Avg_Vol_20"]
                )

                sma200 = float(
                    row["SMA_200"]
                )

                sma50 = float(
                    row["SMA_50"]
                )

                ema20 = float(
                    row["EMA_20"]
                )

                rsi = float(
                    row["RSI_14"]
                )

                atr = float(
                    row["ATR_14"]
                )

                # =================================================
                # FILTERS
                # =================================================

                if (
                    any(
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
                            rsi,
                            atr
                        ]
                    )
                    or atr <= 0
                ):

                    signal_index += 1
                    continue

                # Price above SMA200

                if price < sma200:

                    signal_index += 1
                    continue

                # RSI

                if not (
                    35 <= rsi <= 55
                ):

                    signal_index += 1
                    continue

                # Pullback

                touched_ema = (
                    ema20 * 0.99
                    <= low
                    <= ema20 * 1.01
                )

                touched_sma = (
                    sma50 * 0.99
                    <= low
                    <= sma50 * 1.01
                )

                closed_above = (
                    price >= ema20
                    or
                    price >= sma50
                )

                if not (
                    (
                        touched_ema
                        or touched_sma
                    )
                    and
                    closed_above
                ):

                    signal_index += 1
                    continue

                # =================================================
                # ENTRY
                # =================================================

                entry = (
                    high
                    + ENTRY_BUFFER
                )

                # =================================================
                # SWING LOW
                # =================================================

                swing_low = (
                    find_previous_swing_low(
                        df,
                        signal_index
                    )
                )

                if swing_low is None:

                    signal_index += 1
                    continue

                # =================================================
                # ATR STOP
                # =================================================

                sl = (
                    swing_low
                    -
                    ATR_SL_BUFFER
                    * atr
                )

                risk = (
                    entry - sl
                )

                if risk <= 0:

                    signal_index += 1
                    continue

                if (
                    risk
                    <
                    MIN_STOP_ATR * atr
                ):

                    rejected_small_stop += 1

                    signal_index += 1
                    continue

                # =================================================
                # SWING HIGH
                # =================================================

                swing_high = (
                    find_previous_swing_high(
                        df,
                        signal_index
                    )
                )

                if swing_high is None:

                    signal_index += 1
                    continue

                if swing_high <= entry:

                    signal_index += 1
                    continue

                swing_rr = (
                    swing_high - entry
                ) / risk

                # Keep same qualifying signals as V3

                if swing_rr < MIN_RR:

                    rejected_rr += 1

                    signal_index += 1
                    continue

                # =================================================
                # ACTUAL ENTRY
                # =================================================

                entry_index = (
                    find_entry(
                        df,
                        signal_index,
                        entry
                    )
                )

                if entry_index is None:

                    expired += 1

                    signal_index += (
                        ENTRY_VALID_DAYS
                        + 1
                    )

                    continue

                # =================================================
                # TARGETS
                # =================================================

                target_2r = (
                    entry
                    + 2.0 * risk
                )

                target_25r = (
                    entry
                    + 2.5 * risk
                )

                # =================================================
                # A — SWING HIGH
                # =================================================

                swing_result = (
                    simulate_single_target(
                        df,
                        entry_index,
                        entry,
                        sl,
                        swing_high
                    )
                )

                # =================================================
                # B — FIXED 2R
                # =================================================

                r2_result = (
                    simulate_single_target(
                        df,
                        entry_index,
                        entry,
                        sl,
                        target_2r
                    )
                )

                # =================================================
                # C — FIXED 2.5R
                # =================================================

                r25_result = (
                    simulate_single_target(
                        df,
                        entry_index,
                        entry,
                        sl,
                        target_25r
                    )
                )

                # =================================================
                # D — PARTIAL
                # =================================================

                partial_result = (
                    simulate_partial_strategy(
                        df,
                        entry_index,
                        entry,
                        sl,
                        swing_high
                    )
                )

                # =================================================
                # SAVE EXACT SAME TRADE FOR ALL STRATEGIES
                # =================================================

                all_trades.append(
                    {
                        "Ticker":
                            ticker,

                        "Signal_Date":
                            df.index[
                                signal_index
                            ],

                        "Entry_Date":
                            df.index[
                                entry_index
                            ],

                        "Entry_Index":
                            entry_index,

                        "Entry":
                            entry,

                        "SL":
                            sl,

                        "Swing_High":
                            swing_high,

                        "ATR":
                            atr,

                        "Swing_RR":
                            swing_rr,

                        # A
                        "Swing_R":
                            swing_result["R"],

                        "Swing_Result":
                            swing_result["Result"],

                        "Swing_Exit":
                            swing_result["Exit_Index"],

                        # B
                        "R2_R":
                            r2_result["R"],

                        "R2_Result":
                            r2_result["Result"],

                        "R2_Exit":
                            r2_result["Exit_Index"],

                        # C
                        "R25_R":
                            r25_result["R"],

                        "R25_Result":
                            r25_result["Result"],

                        "R25_Exit":
                            r25_result["Exit_Index"],

                        # D
                        "Partial_R":
                            partial_result["R"],

                        "Partial_Result":
                            partial_result["Result"],

                        "Partial_Exit":
                            partial_result[
                                "Exit_Index"
                            ],

                        "Partial_Hit_2R":
                            partial_result[
                                "Hit_2R"
                            ],

                        "Partial_Hit_Swing":
                            partial_result[
                                "Hit_Swing"
                            ]
                    }
                )

                # =================================================
                # SAME TRADE SET FOR ALL STRATEGIES
                #
                # Do not allow another trade until ALL four
                # versions of the current trade are finished.
                # =================================================

                last_exit = max(
                    swing_result[
                        "Exit_Index"
                    ],
                    r2_result[
                        "Exit_Index"
                    ],
                    r25_result[
                        "Exit_Index"
                    ],
                    partial_result[
                        "Exit_Index"
                    ]
                )

                signal_index = (
                    last_exit + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # ========================================================
    # RESULTS
    # ========================================================

    if not all_trades:

        print(
            "NO TRADES FOUND."
        )

        return

    trades = pd.DataFrame(
        all_trades
    )

    trades = trades.sort_values(
        [
            "Entry_Date",
            "Ticker"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # CALCULATE FOUR STRATEGIES
    # ========================================================

    swing_stats = calculate_stats(
        trades,
        "Swing_R",
        "Swing_Exit"
    )

    r2_stats = calculate_stats(
        trades,
        "R2_R",
        "R2_Exit"
    )

    r25_stats = calculate_stats(
        trades,
        "R25_R",
        "R25_Exit"
    )

    partial_stats = calculate_stats(
        trades,
        "Partial_R",
        "Partial_Exit"
    )

    comparison = pd.DataFrame(
        {
            "Swing High":
                swing_stats,

            "Fixed 2R":
                r2_stats,

            "Fixed 2.5R":
                r25_stats,

            "50% 2R + 50% Swing":
                partial_stats
        }
    ).T

    # ========================================================
    # PRINT
    # ========================================================

    print()

    print("=" * 90)
    print("V4 FINAL COMPARISON")
    print("=" * 90)

    print()

    print(
        comparison.round(3)
        .to_string()
    )

    # ========================================================
    # PARTIAL STRATEGY DETAILS
    # ========================================================

    hit_2r = (
        trades[
            "Partial_Hit_2R"
        ].sum()
    )

    hit_swing = (
        trades[
            "Partial_Hit_Swing"
        ].sum()
    )

    print()

    print("=" * 90)
    print("PARTIAL STRATEGY DETAILS")
    print("=" * 90)

    print(
        f"Total trades:                 {len(trades)}"
    )

    print(
        f"Reached first 2R target:      {hit_2r}"
    )

    print(
        f"Reached final Swing High:     {hit_swing}"
    )

    print(
        f"2R hit rate:                  {hit_2r / len(trades) * 100:.2f}%"
    )

    print(
        f"Final Swing hit rate:         {hit_swing / len(trades) * 100:.2f}%"
    )

    # ========================================================
    # SHOW BEST STRATEGY
    # ========================================================

    best_avg_r = (
        comparison[
            "Avg_R"
        ].idxmax()
    )

    best_pf = (
        comparison[
            "Profit_Factor"
        ].idxmax()
    )

    smallest_dd = (
        comparison[
            "Max_Drawdown_R"
        ].idxmax()
    )

    print()

    print("=" * 90)
    print("LEADERS")
    print("=" * 90)

    print(
        f"Best Average R:       {best_avg_r}"
    )

    print(
        f"Best Profit Factor:   {best_pf}"
    )

    print(
        f"Lowest Drawdown:      {smallest_dd}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "backtest_v4_trades.csv",
        index=False
    )

    comparison.to_csv(
        "backtest_v4_comparison.csv"
    )

    print()

    print("=" * 90)

    print(
        "Saved:"
    )

    print(
        "  backtest_v4_trades.csv"
    )

    print(
        "  backtest_v4_comparison.csv"
    )

    print()

    print(
        "Pullback V4 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
