import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# NASDAQ-100 TICKERS
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
# STRATEGY PARAMETERS - V6
# ============================================================

SWING_LOOKBACK = 3

ATR_PERIOD = 14

# Entry = signal High + 10% ATR
ENTRY_ATR_BUFFER = 0.10

# SL = Swing Low - 10% ATR
SL_ATR_BUFFER = 0.10

# Entry -> SL must be at least 0.50 ATR
MIN_STOP_ATR = 0.50

# Signal remains valid for 3 sessions
ENTRY_VALID_DAYS = 3

# Trade can remain open for 10 sessions
MAX_HOLDING_DAYS = 10

# Trend slope
SMA50_SLOPE_LOOKBACK = 10

# RSI
RSI_MIN = 35
RSI_MAX = 55

# Pullback proximity
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
# PREVIOUS CONFIRMED SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):

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
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current < float(left.min())
            and
            current < float(right.min())
        ):
            return current

    return None


# ============================================================
# FIND ENTRY
# ============================================================

def find_entry(
    df,
    signal_index,
    entry
):

    last_index = min(
        signal_index + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    for i in range(
        signal_index + 1,
        last_index + 1
    ):

        day_high = float(
            df["High"].iloc[i]
        )

        if day_high >= entry:
            return i

    return None


# ============================================================
# FIXED 2R
# ============================================================

def simulate_fixed_2r(
    df,
    entry_index,
    entry,
    sl
):

    risk = entry - sl

    target = (
        entry
        + 2.0 * risk
    )

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

        # Daily ambiguity:
        # if SL and TP both touched -> assume SL first
        if low <= sl:
            return {
                "R": -1.0,
                "Exit_Index": i,
                "Result": "SL"
            }

        if high >= target:
            return {
                "R": 2.0,
                "Exit_Index": i,
                "Result": "TP"
            }

    exit_price = float(
        df["Close"].iloc[last_index]
    )

    r = (
        exit_price - entry
    ) / risk

    return {
        "R": r,
        "Exit_Index": last_index,
        "Result": "TIME"
    }


# ============================================================
# 50% @ 2R
# 50% @ 4R
# MOVE SECOND HALF TO BREAKEVEN AFTER TP1
# ============================================================

def simulate_partial_2r_4r_be(
    df,
    entry_index,
    entry,
    sl
):

    risk = entry - sl

    tp1 = (
        entry
        + 2.0 * risk
    )

    tp2 = (
        entry
        + 4.0 * risk
    )

    last_index = min(
        entry_index
        + MAX_HOLDING_DAYS
        - 1,
        len(df) - 1
    )

    tp1_hit = False

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
        # BEFORE TP1
        # ====================================================

        if not tp1_hit:

            # Conservative daily assumption
            if low <= sl:

                return {
                    "R": -1.0,
                    "Exit_Index": i,
                    "Result": "SL_BEFORE_TP1",
                    "Hit_2R": False,
                    "Hit_4R": False
                }

            if high >= tp1:

                tp1_hit = True

                # Same candle reaches 4R too
                if high >= tp2:

                    # 50% × 2R = +1R
                    # 50% × 4R = +2R
                    # Total = +3R

                    return {
                        "R": 3.0,
                        "Exit_Index": i,
                        "Result": "FULL_TARGET",
                        "Hit_2R": True,
                        "Hit_4R": True
                    }

                continue

        # ====================================================
        # AFTER TP1
        # ====================================================

        else:

            # Remaining 50% now has stop at Entry

            # Conservative:
            # if BE and TP2 occur in same candle -> BE first
            if low <= entry:

                # First half already banked +1R
                # Second half exits 0R
                # Total = +1R

                return {
                    "R": 1.0,
                    "Exit_Index": i,
                    "Result": "TP1_THEN_BE",
                    "Hit_2R": True,
                    "Hit_4R": False
                }

            if high >= tp2:

                return {
                    "R": 3.0,
                    "Exit_Index": i,
                    "Result": "FULL_TARGET",
                    "Hit_2R": True,
                    "Hit_4R": True
                }

    # ========================================================
    # TIME EXIT
    # ========================================================

    exit_price = float(
        df["Close"].iloc[last_index]
    )

    current_r = (
        exit_price - entry
    ) / risk

    if not tp1_hit:

        return {
            "R": current_r,
            "Exit_Index": last_index,
            "Result": "TIME_BEFORE_TP1",
            "Hit_2R": False,
            "Hit_4R": False
        }

    # Half already sold at +2R:
    #
    # 0.5 × 2R = +1R
    #
    # remaining 50%:
    # 0.5 × current R

    total_r = (
        1.0
        + 0.5 * current_r
    )

    return {
        "R": total_r,
        "Exit_Index": last_index,
        "Result": "TP1_THEN_TIME",
        "Hit_2R": True,
        "Hit_4R": False
    }


# ============================================================
# PERFORMANCE STATS
# ============================================================

def calculate_stats(
    df,
    r_column
):

    if df.empty:

        return {
            "Trades": 0,
            "Profitable_%": 0,
            "Total_R": 0,
            "Avg_R": 0,
            "Median_R": 0,
            "Profit_Factor": 0,
            "Max_Drawdown_R": 0,
            "Max_Consecutive_Losses": 0
        }

    r = df[r_column]

    positive = r[
        r > 0
    ]

    negative = r[
        r < 0
    ]

    gross_profit = (
        positive.sum()
    )

    gross_loss = abs(
        negative.sum()
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

    return {
        "Trades":
            len(df),

        "Profitable_%":
            (r > 0).mean() * 100,

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
            max_consecutive_losses
    }


# ============================================================
# BUILD STRATEGY COMPARISON
# ============================================================

def build_comparison(
    trades
):

    return pd.DataFrame(
        {
            "Fixed 2R":
                calculate_stats(
                    trades,
                    "Fixed2_R"
                ),

            "50% 2R + 50% 4R / BE":
                calculate_stats(
                    trades,
                    "Partial_R"
                )
        }
    ).T


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)

    print(
        "PULLBACK BACKTEST V6 - TREND QUALITY + ATR ENTRY + WALK-FORWARD"
    )

    print("=" * 100)

    print()

    print(
        f"Entry: High + {ENTRY_ATR_BUFFER:.2f} ATR"
    )

    print(
        f"SL: Swing Low - {SL_ATR_BUFFER:.2f} ATR"
    )

    print(
        f"Minimum stop distance: {MIN_STOP_ATR:.2f} ATR"
    )

    print(
        f"SMA50 slope lookback: {SMA50_SLOPE_LOOKBACK} days"
    )

    print()

    # ========================================================
    # DOWNLOAD 5 YEARS
    # ========================================================

    print(
        f"Downloading {len(TICKERS)} tickers..."
    )

    try:

        all_data = yf.download(
            TICKERS,
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

    rejected_trend = 0
    rejected_candle = 0
    rejected_stop = 0
    expired = 0

    # ========================================================
    # EACH TICKER
    # ========================================================

    for ticker in TICKERS:

        try:

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

            if len(df) < 250:

                print(
                    f"{ticker}: insufficient data"
                )

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
                    df["Close"],
                    14
                )
            )

            df["ATR14"] = (
                calculate_atr(
                    df,
                    ATR_PERIOD
                )
            )

            # =================================================
            # WALK THROUGH HISTORY
            # =================================================

            signal_index = max(
                210,
                SMA50_SLOPE_LOOKBACK
            )

            while (
                signal_index
                < len(df) - 1
            ):

                row = df.iloc[
                    signal_index
                ]

                open_price = float(
                    row["Open"]
                )

                close = float(
                    row["Close"]
                )

                high = float(
                    row["High"]
                )

                low = float(
                    row["Low"]
                )

                sma200 = float(
                    row["SMA200"]
                )

                sma50 = float(
                    row["SMA50"]
                )

                ema20 = float(
                    row["EMA20"]
                )

                rsi = float(
                    row["RSI14"]
                )

                atr = float(
                    row["ATR14"]
                )

                old_sma50 = float(
                    df["SMA50"].iloc[
                        signal_index
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

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 1
                # PRICE ABOVE SMA200
                # =================================================

                if close <= sma200:

                    rejected_trend += 1

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 2
                # EMA20 ABOVE SMA50
                # =================================================

                if ema20 <= sma50:

                    rejected_trend += 1

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 3
                # SMA50 MUST BE RISING
                # =================================================

                if sma50 <= old_sma50:

                    rejected_trend += 1

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 4
                # RSI
                # =================================================

                if not (
                    RSI_MIN
                    <= rsi
                    <= RSI_MAX
                ):

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 5
                # PULLBACK TO EMA20 OR SMA50
                # =================================================

                touched_ema20 = (
                    ema20
                    * (1 - SUPPORT_TOLERANCE)
                    <= low
                    <=
                    ema20
                    * (1 + SUPPORT_TOLERANCE)
                )

                touched_sma50 = (
                    sma50
                    * (1 - SUPPORT_TOLERANCE)
                    <= low
                    <=
                    sma50
                    * (1 + SUPPORT_TOLERANCE)
                )

                if not (
                    touched_ema20
                    or touched_sma50
                ):

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 6
                # CLOSE ABOVE SUPPORT
                # =================================================

                if not (
                    close >= ema20
                    or close >= sma50
                ):

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 7
                # GREEN CANDLE
                # =================================================

                if close <= open_price:

                    rejected_candle += 1

                    signal_index += 1
                    continue

                # =================================================
                # FILTER 8
                # CLOSE IN UPPER HALF OF DAILY RANGE
                # =================================================

                candle_range = (
                    high - low
                )

                if candle_range <= 0:

                    signal_index += 1
                    continue

                midpoint = (
                    low
                    + candle_range * 0.50
                )

                if close <= midpoint:

                    rejected_candle += 1

                    signal_index += 1
                    continue

                # =================================================
                # ATR-BASED ENTRY
                # =================================================

                entry = (
                    high
                    + ENTRY_ATR_BUFFER
                    * atr
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
                    SL_ATR_BUFFER
                    * atr
                )

                risk = (
                    entry - sl
                )

                if risk <= 0:

                    signal_index += 1
                    continue

                # =================================================
                # MINIMUM STOP DISTANCE
                # =================================================

                if risk < (
                    MIN_STOP_ATR
                    * atr
                ):

                    rejected_stop += 1

                    signal_index += 1
                    continue

                # =================================================
                # ENTRY MUST TRIGGER WITHIN 3 DAYS
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
                # FIXED 2R
                # =================================================

                fixed = simulate_fixed_2r(
                    df,
                    entry_index,
                    entry,
                    sl
                )

                # =================================================
                # PARTIAL 2R / 4R BE
                # =================================================

                partial = (
                    simulate_partial_2r_4r_be(
                        df,
                        entry_index,
                        entry,
                        sl
                    )
                )

                signal_date = pd.Timestamp(
                    df.index[
                        signal_index
                    ]
                )

                entry_date = pd.Timestamp(
                    df.index[
                        entry_index
                    ]
                )

                all_trades.append(
                    {
                        "Ticker":
                            ticker,

                        "Signal_Date":
                            signal_date,

                        "Entry_Date":
                            entry_date,

                        "Year":
                            signal_date.year,

                        "Entry":
                            entry,

                        "SL":
                            sl,

                        "ATR":
                            atr,

                        "Risk_ATR":
                            risk / atr,

                        "RSI":
                            rsi,

                        "EMA20":
                            ema20,

                        "SMA50":
                            sma50,

                        "Fixed2_R":
                            fixed["R"],

                        "Fixed2_Result":
                            fixed["Result"],

                        "Partial_R":
                            partial["R"],

                        "Partial_Result":
                            partial["Result"],

                        "Partial_Hit_2R":
                            partial["Hit_2R"],

                        "Partial_Hit_4R":
                            partial["Hit_4R"]
                    }
                )

                # =================================================
                # NO OVERLAPPING TRADE PER TICKER
                # =================================================

                last_exit = max(
                    fixed[
                        "Exit_Index"
                    ],
                    partial[
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
    # NO RESULTS
    # ========================================================

    if not all_trades:

        print()
        print(
            "NO TRADES FOUND."
        )

        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    trades = pd.DataFrame(
        all_trades
    )

    trades = (
        trades
        .sort_values(
            [
                "Signal_Date",
                "Ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # ALL PERIOD
    # ========================================================

    overall = (
        build_comparison(
            trades
        )
    )

    print()

    print("=" * 100)
    print("V6 - ALL DATA")
    print("=" * 100)

    print(
        overall.round(3)
        .to_string()
    )

    # ========================================================
    # WALK-FORWARD / YEAR BY YEAR
    # ========================================================

    years = sorted(
        trades[
            "Year"
        ].unique()
    )

    print()

    print("=" * 100)
    print("V6 - YEAR-BY-YEAR WALK-FORWARD CHECK")
    print("=" * 100)

    yearly_results = []

    for year in years:

        year_trades = trades[
            trades["Year"]
            == year
        ].copy()

        comparison = (
            build_comparison(
                year_trades
            )
        )

        for strategy in (
            comparison.index
        ):

            row = comparison.loc[
                strategy
            ]

            yearly_results.append(
                {
                    "Year":
                        year,

                    "Strategy":
                        strategy,

                    "Trades":
                        row["Trades"],

                    "Profitable_%":
                        row[
                            "Profitable_%"
                        ],

                    "Total_R":
                        row[
                            "Total_R"
                        ],

                    "Avg_R":
                        row[
                            "Avg_R"
                        ],

                    "Profit_Factor":
                        row[
                            "Profit_Factor"
                        ],

                    "Max_DD_R":
                        row[
                            "Max_Drawdown_R"
                        ]
                }
            )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    print(
        yearly_df.round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # YEAR CONSISTENCY
    # ========================================================

    print()

    print("=" * 100)
    print("V6 - CONSISTENCY CHECK")
    print("=" * 100)

    for strategy in [
        "Fixed 2R",
        "50% 2R + 50% 4R / BE"
    ]:

        subset = yearly_df[
            yearly_df[
                "Strategy"
            ]
            == strategy
        ]

        positive_years = (
            subset[
                "Total_R"
            ] > 0
        ).sum()

        negative_years = (
            subset[
                "Total_R"
            ] < 0
        ).sum()

        avg_yearly_r = (
            subset[
                "Avg_R"
            ].mean()
        )

        print()
        print(strategy)

        print(
            f"Positive years:       "
            f"{positive_years}/{len(subset)}"
        )

        print(
            f"Negative years:       "
            f"{negative_years}/{len(subset)}"
        )

        print(
            f"Average yearly Avg R: "
            f"{avg_yearly_r:.3f}R"
        )

    # ========================================================
    # RECENT PERIOD
    # ========================================================

    recent = trades[
        trades[
            "Signal_Date"
        ]
        >= pd.Timestamp(
            "2025-01-01"
        )
    ].copy()

    print()

    print("=" * 100)
    print("V6 - 2025+ CHECK")
    print("=" * 100)

    if not recent.empty:

        recent_results = (
            build_comparison(
                recent
            )
        )

        print(
            recent_results
            .round(3)
            .to_string()
        )

    # ========================================================
    # SIGNAL COUNTS
    # ========================================================

    print()

    print("=" * 100)
    print("FILTER DIAGNOSTICS")
    print("=" * 100)

    print(
        f"Final trades:                 {len(trades)}"
    )

    print(
        f"Rejected by trend filters:    {rejected_trend}"
    )

    print(
        f"Rejected by candle filters:   {rejected_candle}"
    )

    print(
        f"Rejected small stops:         {rejected_stop}"
    )

    print(
        f"Expired setups:               {expired}"
    )

    print()

    print(
        f"Partial trades reaching 2R:   "
        f"{trades['Partial_Hit_2R'].sum()}"
    )

    print(
        f"Partial trades reaching 4R:   "
        f"{trades['Partial_Hit_4R'].sum()}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "backtest_v6_trades.csv",
        index=False
    )

    overall.to_csv(
        "backtest_v6_overall.csv"
    )

    yearly_df.to_csv(
        "backtest_v6_yearly.csv",
        index=False
    )

    print()

    print("=" * 100)

    print(
        "Saved:"
    )

    print(
        "  backtest_v6_trades.csv"
    )

    print(
        "  backtest_v6_overall.csv"
    )

    print(
        "  backtest_v6_yearly.csv"
    )

    print()

    print(
        "Pullback V6 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
