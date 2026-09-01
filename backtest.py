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
# STRATEGY PARAMETERS
# ============================================================

ENTRY_BUFFER = 0.15
ENTRY_VALID_DAYS = 3
MAX_HOLDING_DAYS = 10

SWING_LOOKBACK = 3

ATR_PERIOD = 14
ATR_SL_BUFFER = 0.10
MIN_STOP_ATR = 0.50

MIN_SWING_RR = 2.0


# ============================================================
# OUT-OF-SAMPLE SPLIT
#
# Anything BEFORE 2025 = development
# 2025 onwards = out-of-sample
# ============================================================

TEST_START = pd.Timestamp("2025-01-01")


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
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    )

    true_range = ranges.max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# SWING LOW
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
# SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):

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
            df["High"].iloc[i]
        )

        left = df["High"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right = df["High"].iloc[
            i + 1:i + SWING_LOOKBACK + 1
        ]

        if (
            current > float(left.max())
            and
            current > float(right.max())
        ):
            return current

    return None


# ============================================================
# FIND ENTRY
# ============================================================

def find_entry(df, signal_index, entry):

    last_index = min(
        signal_index + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    for i in range(
        signal_index + 1,
        last_index + 1
    ):

        if float(df["High"].iloc[i]) >= entry:
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
    target = entry + 2 * risk

    last_index = min(
        entry_index + MAX_HOLDING_DAYS - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        last_index + 1
    ):

        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])

        # Conservative:
        # if TP and SL appear in same daily candle -> SL first

        if low <= sl:
            return -1.0, i, "SL"

        if high >= target:
            return 2.0, i, "TP"

    exit_price = float(
        df["Close"].iloc[last_index]
    )

    r = (
        exit_price - entry
    ) / risk

    return r, last_index, "TIME"


# ============================================================
# PARTIAL STRATEGY
#
# 50% at 2R
# Remaining 50% at RUNNER_R
#
# breakeven_after_tp1:
# False -> original SL remains
# True  -> remaining 50% SL moves to entry
# ============================================================

def simulate_partial(
    df,
    entry_index,
    entry,
    sl,
    runner_r,
    breakeven_after_tp1
):

    risk = entry - sl

    tp1 = entry + 2 * risk
    tp2 = entry + runner_r * risk

    last_index = min(
        entry_index + MAX_HOLDING_DAYS - 1,
        len(df) - 1
    )

    tp1_hit = False

    for i in range(
        entry_index,
        last_index + 1
    ):

        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])

        # ====================================================
        # BEFORE TP1
        # ====================================================

        if not tp1_hit:

            # Daily candle ambiguity:
            # SL and TP1 both appear -> assume SL first

            if low <= sl:
                return -1.0, i, "SL_BEFORE_TP1", False

            if high >= tp1:

                tp1_hit = True

                # First half:
                # 50% × +2R = +1R

                # If TP2 was also reached in same candle,
                # no SL was hit, therefore allow full target.

                if high >= tp2:

                    total_r = (
                        1.0
                        +
                        0.5 * runner_r
                    )

                    return (
                        total_r,
                        i,
                        "FULL_TARGET",
                        True
                    )

                continue

        # ====================================================
        # AFTER TP1
        # ====================================================

        if tp1_hit:

            current_stop = (
                entry
                if breakeven_after_tp1
                else sl
            )

            # Same candle touches stop and TP2.
            # Conservative = stop first.

            if low <= current_stop:

                if breakeven_after_tp1:

                    # First 50%:
                    # +1R
                    #
                    # Remaining:
                    # 0R
                    #
                    # Total = +1R

                    return (
                        1.0,
                        i,
                        "TP1_THEN_BE",
                        True
                    )

                else:

                    # First half = +1R
                    # second half = -0.5R
                    # Total = +0.5R

                    return (
                        0.5,
                        i,
                        "TP1_THEN_SL",
                        True
                    )

            if high >= tp2:

                total_r = (
                    1.0
                    +
                    0.5 * runner_r
                )

                return (
                    total_r,
                    i,
                    "FULL_TARGET",
                    True
                )

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

        return (
            current_r,
            last_index,
            "TIME_BEFORE_TP1",
            False
        )

    # First half already made +1R.
    # Remaining half exits at current R.

    total_r = (
        1.0
        +
        0.5 * current_r
    )

    return (
        total_r,
        last_index,
        "TP1_THEN_TIME",
        True
    )


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_stats(df, r_col):

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

    r = df[r_col]

    positive = r[r > 0]
    negative = r[r < 0]

    gross_profit = positive.sum()
    gross_loss = abs(negative.sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    cumulative = r.cumsum()

    peak = (
        cumulative
        .clip(lower=0)
        .cummax()
    )

    drawdown = cumulative - peak

    max_dd = drawdown.min()

    max_losses = 0
    current_losses = 0

    for value in r:

        if value < 0:

            current_losses += 1

            max_losses = max(
                max_losses,
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
            pf,

        "Max_Drawdown_R":
            max_dd,

        "Max_Consecutive_Losses":
            max_losses
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_comparison(trades):

    strategies = {
        "Fixed 2R":
            "Fixed2_R",

        "50% 2R + 50% 4R / Original SL":
            "Partial4_SL_R",

        "50% 2R + 50% 4R / Breakeven":
            "Partial4_BE_R",

        "50% 2R + 50% 5R / Breakeven":
            "Partial5_BE_R"
    }

    rows = {}

    for name, column in strategies.items():

        rows[name] = calculate_stats(
            trades,
            column
        )

    return pd.DataFrame(rows).T


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("PULLBACK BACKTEST V5 - RUNNER + BREAKEVEN + OUT-OF-SAMPLE")
    print("=" * 90)

    print()
    print("Strategies:")
    print("A) Fixed 2R")
    print("B) 50% @ 2R + 50% @ 4R | Original SL")
    print("C) 50% @ 2R + 50% @ 4R | Breakeven after TP1")
    print("D) 50% @ 2R + 50% @ 5R | Breakeven after TP1")
    print()

    print(
        f"Out-of-sample starts: {TEST_START.date()}"
    )

    print()

    # ========================================================
    # DOWNLOAD
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

    trades = []

    rejected_stop = 0
    rejected_rr = 0
    expired = 0

    # ========================================================
    # LOOP
    # ========================================================

    for ticker in TICKERS:

        try:

            if (
                ticker
                not in
                all_data.columns.get_level_values(0)
            ):
                continue

            df = all_data[ticker].copy()

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

            df["AVG_VOL20"] = (
                df["Volume"]
                .rolling(20)
                .mean()
            )

            df["RSI14"] = calculate_rsi(
                df["Close"]
            )

            df["ATR14"] = calculate_atr(
                df
            )

            # =================================================
            # WALK FORWARD
            # =================================================

            signal_index = 210

            while signal_index < len(df) - 1:

                row = df.iloc[
                    signal_index
                ]

                close = float(row["Close"])
                high = float(row["High"])
                low = float(row["Low"])

                volume = float(row["Volume"])
                avg_volume = float(row["AVG_VOL20"])

                sma200 = float(row["SMA200"])
                sma50 = float(row["SMA50"])
                ema20 = float(row["EMA20"])

                rsi = float(row["RSI14"])
                atr = float(row["ATR14"])

                values = [
                    close,
                    high,
                    low,
                    volume,
                    avg_volume,
                    sma200,
                    sma50,
                    ema20,
                    rsi,
                    atr
                ]

                if (
                    any(pd.isna(x) for x in values)
                    or atr <= 0
                ):

                    signal_index += 1
                    continue

                # =================================================
                # ORIGINAL SIGNAL
                # =================================================

                if close < sma200:

                    signal_index += 1
                    continue

                if not (
                    35 <= rsi <= 55
                ):

                    signal_index += 1
                    continue

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
                    close >= ema20
                    or
                    close >= sma50
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

                risk = entry - sl

                if risk <= 0:

                    signal_index += 1
                    continue

                if risk < (
                    MIN_STOP_ATR
                    * atr
                ):

                    rejected_stop += 1

                    signal_index += 1
                    continue

                # =================================================
                # SWING HIGH FILTER
                # =================================================

                swing_high = (
                    find_previous_swing_high(
                        df,
                        signal_index
                    )
                )

                if (
                    swing_high is None
                    or
                    swing_high <= entry
                ):

                    signal_index += 1
                    continue

                swing_rr = (
                    swing_high - entry
                ) / risk

                if swing_rr < MIN_SWING_RR:

                    rejected_rr += 1

                    signal_index += 1
                    continue

                # =================================================
                # ENTRY ACTIVATION
                # =================================================

                entry_index = find_entry(
                    df,
                    signal_index,
                    entry
                )

                if entry_index is None:

                    expired += 1

                    signal_index += (
                        ENTRY_VALID_DAYS
                        + 1
                    )

                    continue

                # =================================================
                # A — FIXED 2R
                # =================================================

                fixed_r, fixed_exit, fixed_result = (
                    simulate_fixed_2r(
                        df,
                        entry_index,
                        entry,
                        sl
                    )
                )

                # =================================================
                # B — 2R / 4R ORIGINAL SL
                # =================================================

                p4sl_r, p4sl_exit, p4sl_result, p4sl_hit = (
                    simulate_partial(
                        df,
                        entry_index,
                        entry,
                        sl,
                        runner_r=4.0,
                        breakeven_after_tp1=False
                    )
                )

                # =================================================
                # C — 2R / 4R BREAKEVEN
                # =================================================

                p4be_r, p4be_exit, p4be_result, p4be_hit = (
                    simulate_partial(
                        df,
                        entry_index,
                        entry,
                        sl,
                        runner_r=4.0,
                        breakeven_after_tp1=True
                    )
                )

                # =================================================
                # D — 2R / 5R BREAKEVEN
                # =================================================

                p5be_r, p5be_exit, p5be_result, p5be_hit = (
                    simulate_partial(
                        df,
                        entry_index,
                        entry,
                        sl,
                        runner_r=5.0,
                        breakeven_after_tp1=True
                    )
                )

                signal_date = pd.Timestamp(
                    df.index[signal_index]
                )

                period = (
                    "OUT_OF_SAMPLE"
                    if signal_date >= TEST_START
                    else "DEVELOPMENT"
                )

                trades.append(
                    {
                        "Ticker":
                            ticker,

                        "Signal_Date":
                            signal_date,

                        "Entry_Date":
                            df.index[
                                entry_index
                            ],

                        "Period":
                            period,

                        "Entry":
                            entry,

                        "SL":
                            sl,

                        "ATR":
                            atr,

                        "Swing_RR":
                            swing_rr,

                        "Fixed2_R":
                            fixed_r,

                        "Fixed2_Result":
                            fixed_result,

                        "Partial4_SL_R":
                            p4sl_r,

                        "Partial4_SL_Result":
                            p4sl_result,

                        "Partial4_BE_R":
                            p4be_r,

                        "Partial4_BE_Result":
                            p4be_result,

                        "Partial5_BE_R":
                            p5be_r,

                        "Partial5_BE_Result":
                            p5be_result
                    }
                )

                # =================================================
                # NO OVERLAPPING TRADE
                #
                # Advance beyond longest strategy exit.
                # =================================================

                last_exit = max(
                    fixed_exit,
                    p4sl_exit,
                    p4be_exit,
                    p5be_exit
                )

                signal_index = (
                    last_exit + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if not trades:

        print(
            "NO TRADES FOUND."
        )

        return

    trades = pd.DataFrame(
        trades
    )

    trades = trades.sort_values(
        [
            "Signal_Date",
            "Ticker"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # ALL DATA
    # ========================================================

    all_results = build_comparison(
        trades
    )

    # ========================================================
    # DEVELOPMENT
    # ========================================================

    development = trades[
        trades["Period"]
        == "DEVELOPMENT"
    ].copy()

    dev_results = build_comparison(
        development
    )

    # ========================================================
    # OUT OF SAMPLE
    # ========================================================

    out_sample = trades[
        trades["Period"]
        == "OUT_OF_SAMPLE"
    ].copy()

    oos_results = build_comparison(
        out_sample
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 100)
    print("V5 — ALL DATA")
    print("=" * 100)

    print(
        all_results.round(3)
        .to_string()
    )

    print()
    print("=" * 100)
    print("V5 — DEVELOPMENT PERIOD")
    print("=" * 100)

    print(
        dev_results.round(3)
        .to_string()
    )

    print()
    print("=" * 100)
    print(
        f"V5 — OUT OF SAMPLE ({TEST_START.date()}+)"
    )
    print("=" * 100)

    print(
        oos_results.round(3)
        .to_string()
    )

    # ========================================================
    # WINNERS
    # ========================================================

    print()
    print("=" * 100)
    print("OUT-OF-SAMPLE LEADERS")
    print("=" * 100)

    if not oos_results.empty:

        print(
            "Best Average R:",
            oos_results["Avg_R"].idxmax()
        )

        print(
            "Best Profit Factor:",
            oos_results[
                "Profit_Factor"
            ].idxmax()
        )

        print(
            "Lowest Drawdown:",
            oos_results[
                "Max_Drawdown_R"
            ].idxmax()
        )

    print()

    print(
        f"Total trades:          {len(trades)}"
    )

    print(
        f"Development trades:    {len(development)}"
    )

    print(
        f"Out-of-sample trades:  {len(out_sample)}"
    )

    print()

    print(
        f"Rejected small stops:  {rejected_stop}"
    )

    print(
        f"Rejected low R:R:      {rejected_rr}"
    )

    print(
        f"Expired setups:        {expired}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "backtest_v5_trades.csv",
        index=False
    )

    all_results.to_csv(
        "backtest_v5_all.csv"
    )

    dev_results.to_csv(
        "backtest_v5_development.csv"
    )

    oos_results.to_csv(
        "backtest_v5_out_of_sample.csv"
    )

    print()

    print("=" * 100)

    print(
        "Saved:"
    )

    print(
        "  backtest_v5_trades.csv"
    )

    print(
        "  backtest_v5_all.csv"
    )

    print(
        "  backtest_v5_development.csv"
    )

    print(
        "  backtest_v5_out_of_sample.csv"
    )

    print()

    print(
        "Pullback V5 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
