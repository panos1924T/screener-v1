
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V12.1 - CORRECTED EXIT TOURNAMENT
#
# IMPORTANT:
# Entry candidates are generated EXACTLY using the V11
# Fixed-2R lifecycle.
#
# Therefore:
#   Fixed 2R should reproduce V11 closely.
#
# Then the SAME candidate list is tested with:
#
# A. Fixed 2R
# B. Full 2.0 ATR Trail
# C. Full 2.5 ATR Trail
# D. Full 3.0 ATR Trail
# E. 50% @ 2R + 50% 2.5 ATR Trail
#
# ============================================================


# ============================================================
# ACCOUNT
# ============================================================

STARTING_CAPITAL = 1000.0

RISK_PER_TRADE = 0.005       # 0.50%

MAX_POSITIONS = 3

SLIPPAGE_PCT = 0.0005        # 0.05%

COMMISSION_PCT = 0.0002      # 0.02%

MIN_SHARE_SIZE = 0.0001


# ============================================================
# HISTORY / STRATEGY
# ============================================================

DOWNLOAD_PERIOD = "7y"

BACKTEST_YEARS = 5

ATR_PERIOD = 14

ENTRY_VALID_DAYS = 3

FIXED_HOLD_DAYS = 10

TRAIL_HOLD_DAYS = 30

STOCH_RSI_PERIOD = 14

STOCH_CROSS_LEVEL = 20.0


# ============================================================
# NASDAQ-100
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
    "XEL", "ZS"
]


# ============================================================
# INDICATORS
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

    return 100 - (
        100 / (1 + rs)
    )


def calculate_stoch_rsi(
    rsi,
    period=14
):

    lowest = (
        rsi
        .rolling(period)
        .min()
    )

    highest = (
        rsi
        .rolling(period)
        .max()
    )

    denominator = (
        highest - lowest
    ).replace(
        0,
        np.nan
    )

    return (
        (
            rsi - lowest
        )
        / denominator
    ) * 100


def calculate_atr(
    df,
    period=14
):

    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["High"] - df["Low"],

            (
                df["High"]
                - previous_close
            ).abs(),

            (
                df["Low"]
                - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# PREPARE STOCK
# ============================================================

def prepare_stock(df):

    df = df.copy()

    df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ],
        inplace=True
    )

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

    df["ATR14"] = (
        calculate_atr(
            df,
            ATR_PERIOD
        )
    )

    df["RSI14"] = (
        calculate_rsi(
            df["Close"]
        )
    )

    df["STOCH_RSI"] = (
        calculate_stoch_rsi(
            df["RSI14"],
            STOCH_RSI_PERIOD
        )
    )

    df["AVG_VOL20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["HIGH20"] = (
        df["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["LOW10"] = (
        df["Low"]
        .shift(1)
        .rolling(10)
        .min()
    )

    df["LOW5"] = (
        df["Low"]
        .shift(1)
        .rolling(5)
        .min()
    )

    return df


# ============================================================
# MARKET DATA
# ============================================================

def download_market(
    ticker,
    name
):

    print(
        f"Downloading {name}..."
    )

    df = yf.download(
        ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=True,
        repair=True
    )

    if df.empty:

        raise RuntimeError(
            f"No data for {name}"
        )

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df.dropna(
        inplace=True
    )

    df["SMA200"] = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    return df


def get_market_row(
    df,
    date
):

    available = df[
        df.index <= date
    ]

    if available.empty:
        return None

    row = available.iloc[-1]

    if pd.isna(
        row["SMA200"]
    ):

        return None

    return row


def market_allows_long(
    qqq,
    dxy,
    date
):

    qqq_row = get_market_row(
        qqq,
        date
    )

    dxy_row = get_market_row(
        dxy,
        date
    )

    if (
        qqq_row is None
        or dxy_row is None
    ):

        return False

    qqq_ok = (
        float(
            qqq_row["Close"]
        )
        >
        float(
            qqq_row["SMA200"]
        )
    )

    dxy_ok = (
        float(
            dxy_row["Close"]
        )
        <
        float(
            dxy_row["SMA200"]
        )
    )

    return (
        qqq_ok
        and dxy_ok
    )


# ============================================================
# COMMON V11 TREND FILTER
# ============================================================

def strong_trend(
    df,
    i
):

    row = df.iloc[i]

    close = float(
        row["Close"]
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

    old_sma50 = float(
        df["SMA50"]
        .iloc[
            i - 10
        ]
    )

    return (
        close > sma200
        and
        sma50 > sma200
        and
        ema20 > sma50
        and
        sma50 > old_sma50
    )


# ============================================================
# V11 QUALITY SCORE
# ============================================================

def quality_score(
    df,
    i
):

    row = df.iloc[i]

    close = float(
        row["Close"]
    )

    high = float(
        row["High"]
    )

    low = float(
        row["Low"]
    )

    sma50 = float(
        row["SMA50"]
    )

    ema20 = float(
        row["EMA20"]
    )

    atr = float(
        row["ATR14"]
    )

    volume = float(
        row["Volume"]
    )

    avg_volume = float(
        row["AVG_VOL20"]
    )

    old_sma50 = float(
        df["SMA50"]
        .iloc[
            i - 10
        ]
    )

    slope = (
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

    close_location = (
        (
            close - low
        ) / candle_range
        if candle_range > 0
        else 0
    )

    relative_volume = (
        volume / avg_volume
        if avg_volume > 0
        else 0
    )

    relative_volume = min(
        relative_volume,
        2.0
    )

    return (
        slope
        + ema_strength
        + close_location
        + relative_volume
    )


# ============================================================
# BREAKOUT SETUP
# EXACT V11 LOGIC
# ============================================================

def breakout_setup(
    df,
    i
):

    if not strong_trend(
        df,
        i
    ):

        return None

    row = df.iloc[i]

    close = float(row["Close"])
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])

    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])

    volume = float(row["Volume"])
    avg_volume = float(row["AVG_VOL20"])

    high20 = float(row["HIGH20"])
    low10 = float(row["LOW10"])

    if not (
        55 <= rsi <= 72
    ):

        return None

    if avg_volume <= 0:

        return None

    relative_volume = (
        volume / avg_volume
    )

    if relative_volume < 1.10:

        return None

    if close <= open_price:

        return None

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return None

    close_location = (
        close - low
    ) / candle_range

    if close_location < 0.70:

        return None

    distance_to_high = (
        high20 - close
    ) / atr

    if distance_to_high > 0.50:

        return None

    trigger = (
        high20
        + 0.05 * atr
    )

    stop = (
        low10
        - 0.10 * atr
    )

    risk = (
        trigger - stop
    )

    if risk <= 0:

        return None

    if risk < (
        0.75 * atr
    ):

        return None

    if risk > (
        4.0 * atr
    ):

        return None

    return {
        "Setup_Type":
            "BREAKOUT",

        "Trigger":
            trigger,

        "Stop":
            stop,

        "Score":
            quality_score(
                df,
                i
            )
    }


# ============================================================
# PULLBACK SETUP
# EXACT V11 LOGIC
# ============================================================

def pullback_setup(
    df,
    i
):

    if not strong_trend(
        df,
        i
    ):

        return None

    row = df.iloc[i]

    close = float(row["Close"])
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])

    ema20 = float(row["EMA20"])
    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])

    volume = float(row["Volume"])
    avg_volume = float(row["AVG_VOL20"])

    low5 = float(row["LOW5"])

    stoch = float(
        row["STOCH_RSI"]
    )

    previous_stoch = float(
        df["STOCH_RSI"]
        .iloc[
            i - 1
        ]
    )

    if (
        pd.isna(stoch)
        or pd.isna(previous_stoch)
    ):

        return None

    if not (
        45 <= rsi <= 65
    ):

        return None

    crossed = (
        previous_stoch
        <= STOCH_CROSS_LEVEL
        and
        stoch
        > STOCH_CROSS_LEVEL
    )

    if not crossed:

        return None

    ema_distance = (
        abs(
            low - ema20
        )
        / atr
    )

    if ema_distance > 0.50:

        return None

    if close <= ema20:

        return None

    if close <= open_price:

        return None

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return None

    close_location = (
        close - low
    ) / candle_range

    if close_location < 0.60:

        return None

    if avg_volume <= 0:

        return None

    relative_volume = (
        volume / avg_volume
    )

    if relative_volume < 0.80:

        return None

    trigger = (
        high
        + 0.05 * atr
    )

    stop = (
        min(
            low,
            low5
        )
        - 0.10 * atr
    )

    risk = (
        trigger - stop
    )

    if risk <= 0:

        return None

    if risk < (
        0.50 * atr
    ):

        return None

    if risk > (
        3.0 * atr
    ):

        return None

    return {
        "Setup_Type":
            "PULLBACK",

        "Trigger":
            trigger,

        "Stop":
            stop,

        "Score":
            quality_score(
                df,
                i
            )
    }


# ============================================================
# FIND ENTRY
# ============================================================

def find_entry(
    df,
    signal_index,
    trigger
):

    end = min(
        signal_index
        + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    for i in range(
        signal_index + 1,
        end + 1
    ):

        if float(
            df["High"].iloc[i]
        ) >= trigger:

            return i

    return None


# ============================================================
# EXACT V11 FIXED-2R BASELINE
#
# Used BOTH to:
# 1. define candidate lifecycle
# 2. act as tournament baseline
# ============================================================

def simulate_v11_fixed(
    df,
    entry_index,
    trigger,
    stop
):

    entry_open = float(
        df["Open"].iloc[
            entry_index
        ]
    )

    raw_entry = max(
        trigger,
        entry_open
    )

    risk = (
        raw_entry
        - stop
    )

    if risk <= 0:

        return None

    target = (
        raw_entry
        + 2.0 * risk
    )

    end = min(
        entry_index
        + FIXED_HOLD_DAYS
        - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        end + 1
    ):

        row = df.iloc[i]

        open_price = float(
            row["Open"]
        )

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        if open_price < stop:

            return {
                "Raw_Entry":
                    raw_entry,

                "Exit_Index":
                    i,

                "Exit_Price":
                    open_price,

                "Exit_Reason":
                    "SL_GAP"
            }

        if low <= stop:

            return {
                "Raw_Entry":
                    raw_entry,

                "Exit_Index":
                    i,

                "Exit_Price":
                    stop,

                "Exit_Reason":
                    "SL"
            }

        if high >= target:

            return {
                "Raw_Entry":
                    raw_entry,

                "Exit_Index":
                    i,

                "Exit_Price":
                    target,

                "Exit_Reason":
                    "TP2R"
            }

    return {
        "Raw_Entry":
            raw_entry,

        "Exit_Index":
            end,

        "Exit_Price":
            float(
                df["Close"].iloc[
                    end
                ]
            ),

        "Exit_Reason":
            "TIME"
    }


# ============================================================
# GENERATE EXACT V11 CANDIDATE SET
#
# IMPORTANT FIX:
#
# After a valid trade candidate:
#
# i = BASELINE EXIT INDEX + 1
#
# not:
# i = ENTRY INDEX + 1
#
# ============================================================

def generate_v11_candidates(
    all_data,
    qqq,
    dxy,
    backtest_start
):

    candidates = []

    rejected_market = 0

    breakout_count = 0
    pullback_count = 0

    for ticker in TICKERS:

        try:

            if (
                ticker
                not in
                all_data.columns.get_level_values(0)
            ):

                continue

            df = prepare_stock(
                all_data[
                    ticker
                ].copy()
            )

            if len(df) < 250:

                continue

            i = 210

            while i < len(df) - 1:

                signal_date = (
                    pd.Timestamp(
                        df.index[i]
                    )
                )

                if signal_date < backtest_start:

                    i += 1
                    continue

                required = [
                    "SMA200",
                    "SMA50",
                    "EMA20",
                    "ATR14",
                    "RSI14",
                    "STOCH_RSI",
                    "AVG_VOL20",
                    "HIGH20",
                    "LOW10",
                    "LOW5"
                ]

                if any(
                    pd.isna(
                        df[col].iloc[i]
                    )
                    for col in required
                ):

                    i += 1
                    continue

                breakout = (
                    breakout_setup(
                        df,
                        i
                    )
                )

                pullback = (
                    pullback_setup(
                        df,
                        i
                    )
                )

                # Same V11 priority:
                # Breakout wins if both occur.

                if breakout is not None:

                    setup = breakout
                    breakout_count += 1

                elif pullback is not None:

                    setup = pullback
                    pullback_count += 1

                else:

                    i += 1
                    continue

                trigger = float(
                    setup[
                        "Trigger"
                    ]
                )

                stop = float(
                    setup[
                        "Stop"
                    ]
                )

                entry_index = (
                    find_entry(
                        df,
                        i,
                        trigger
                    )
                )

                if entry_index is None:

                    i += (
                        ENTRY_VALID_DAYS
                        + 1
                    )

                    continue

                entry_date = (
                    pd.Timestamp(
                        df.index[
                            entry_index
                        ]
                    )
                )

                if not market_allows_long(
                    qqq,
                    dxy,
                    entry_date
                ):

                    rejected_market += 1

                    i = (
                        entry_index + 1
                    )

                    continue

                baseline = (
                    simulate_v11_fixed(
                        df,
                        entry_index,
                        trigger,
                        stop
                    )
                )

                if baseline is None:

                    i += 1
                    continue

                baseline_exit_index = int(
                    baseline[
                        "Exit_Index"
                    ]
                )

                candidates.append(
                    {
                        "Ticker":
                            ticker,

                        "Setup_Type":
                            setup[
                                "Setup_Type"
                            ],

                        "Signal_Date":
                            signal_date,

                        "Entry_Date":
                            entry_date,

                        "Entry_Index":
                            entry_index,

                        "Raw_Entry":
                            baseline[
                                "Raw_Entry"
                            ],

                        "Initial_Stop":
                            stop,

                        "Score":
                            float(
                                setup[
                                    "Score"
                                ]
                            ),

                        "Baseline_Exit_Index":
                            baseline_exit_index,

                        "Baseline_Exit_Date":
                            pd.Timestamp(
                                df.index[
                                    baseline_exit_index
                                ]
                            ),

                        "Baseline_Exit_Price":
                            baseline[
                                "Exit_Price"
                            ],

                        "Baseline_Exit_Reason":
                            baseline[
                                "Exit_Reason"
                            ]
                    }
                )

                # =================================================
                # CRITICAL V12.1 FIX
                # =================================================

                i = (
                    baseline_exit_index
                    + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

    result = pd.DataFrame(
        candidates
    )

    if not result.empty:

        result = (
            result
            .sort_values(
                [
                    "Entry_Date",
                    "Score"
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )

    diagnostics = {
        "Rejected_Market":
            rejected_market,

        "Breakout_Signals":
            breakout_count,

        "Pullback_Signals":
            pullback_count
    }

    return (
        result,
        diagnostics
    )


# ============================================================
# EXIT PLAN A
# EXACT FIXED 2R BASELINE
# ============================================================

def plan_fixed_2r(
    df,
    trade
):

    return [
        {
            "Index":
                int(
                    trade[
                        "Baseline_Exit_Index"
                    ]
                ),

            "Fraction":
                1.0,

            "Price":
                float(
                    trade[
                        "Baseline_Exit_Price"
                    ]
                ),

            "Reason":
                trade[
                    "Baseline_Exit_Reason"
                ]
        }
    ]


# ============================================================
# GENERIC FULL ATR TRAIL
# ============================================================

def plan_full_atr_trail(
    df,
    trade,
    atr_multiple
):

    entry_index = int(
        trade[
            "Entry_Index"
        ]
    )

    raw_entry = float(
        trade[
            "Raw_Entry"
        ]
    )

    initial_stop = float(
        trade[
            "Initial_Stop"
        ]
    )

    end = min(
        entry_index
        + TRAIL_HOLD_DAYS
        - 1,
        len(df) - 1
    )

    trail = (
        initial_stop
    )

    highest_close = (
        raw_entry
    )

    for i in range(
        entry_index,
        end + 1
    ):

        row = df.iloc[i]

        open_price = float(
            row["Open"]
        )

        low = float(
            row["Low"]
        )

        close = float(
            row["Close"]
        )

        atr = float(
            row["ATR14"]
        )

        # Trail based only on previously available
        # highest close.

        calculated_trail = (
            highest_close
            - atr_multiple
            * atr
        )

        trail = max(
            trail,
            calculated_trail
        )

        if open_price < trail:

            return [
                {
                    "Index":
                        i,

                    "Fraction":
                        1.0,

                    "Price":
                        open_price,

                    "Reason":
                        f"ATR{atr_multiple}_GAP"
                }
            ]

        if low <= trail:

            return [
                {
                    "Index":
                        i,

                    "Fraction":
                        1.0,

                    "Price":
                        trail,

                    "Reason":
                        f"ATR{atr_multiple}_TRAIL"
                }
            ]

        highest_close = max(
            highest_close,
            close
        )

    return [
        {
            "Index":
                end,

            "Fraction":
                1.0,

            "Price":
                float(
                    df["Close"].iloc[
                        end
                    ]
                ),

            "Reason":
                "TIME30"
        }
    ]


# ============================================================
# FULL 2 ATR
# ============================================================

def plan_full_2atr(
    df,
    trade
):

    return plan_full_atr_trail(
        df,
        trade,
        2.0
    )


# ============================================================
# FULL 2.5 ATR
# ============================================================

def plan_full_25atr(
    df,
    trade
):

    return plan_full_atr_trail(
        df,
        trade,
        2.5
    )


# ============================================================
# FULL 3 ATR
# ============================================================

def plan_full_3atr(
    df,
    trade
):

    return plan_full_atr_trail(
        df,
        trade,
        3.0
    )


# ============================================================
# 50% @ 2R + 50% 2.5 ATR TRAIL
# ============================================================

def plan_half_2r_25atr(
    df,
    trade
):

    entry_index = int(
        trade[
            "Entry_Index"
        ]
    )

    raw_entry = float(
        trade[
            "Raw_Entry"
        ]
    )

    initial_stop = float(
        trade[
            "Initial_Stop"
        ]
    )

    risk = (
        raw_entry
        - initial_stop
    )

    tp1 = (
        raw_entry
        + 2.0 * risk
    )

    end = min(
        entry_index
        + TRAIL_HOLD_DAYS
        - 1,
        len(df) - 1
    )

    highest_close = (
        raw_entry
    )

    tp1_hit = False

    trail = (
        initial_stop
    )

    first_event = None

    for i in range(
        entry_index,
        end + 1
    ):

        row = df.iloc[i]

        open_price = float(
            row["Open"]
        )

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        close = float(
            row["Close"]
        )

        atr = float(
            row["ATR14"]
        )

        if not tp1_hit:

            if open_price < initial_stop:

                return [
                    {
                        "Index":
                            i,

                        "Fraction":
                            1.0,

                        "Price":
                            open_price,

                        "Reason":
                            "SL_GAP"
                    }
                ]

            if low <= initial_stop:

                return [
                    {
                        "Index":
                            i,

                        "Fraction":
                            1.0,

                        "Price":
                            initial_stop,

                        "Reason":
                            "SL"
                    }
                ]

            if high >= tp1:

                tp1_hit = True

                first_event = {
                    "Index":
                        i,

                    "Fraction":
                        0.5,

                    "Price":
                        tp1,

                    "Reason":
                        "TP1_2R"
                }

                # Once TP1 is hit, runner stop
                # cannot be below breakeven.

                trail = max(
                    raw_entry,
                    highest_close
                    - 2.5
                    * atr
                )

                # Conservative same-day ambiguity.
                if low <= trail:

                    return [
                        first_event,

                        {
                            "Index":
                                i,

                            "Fraction":
                                0.5,

                            "Price":
                                trail,

                            "Reason":
                                "RUNNER_TRAIL"
                        }
                    ]

            highest_close = max(
                highest_close,
                close
            )

        else:

            calculated_trail = (
                highest_close
                - 2.5
                * atr
            )

            trail = max(
                trail,
                raw_entry,
                calculated_trail
            )

            if open_price < trail:

                return [
                    first_event,

                    {
                        "Index":
                            i,

                        "Fraction":
                            0.5,

                        "Price":
                            open_price,

                        "Reason":
                            "RUNNER_GAP"
                    }
                ]

            if low <= trail:

                return [
                    first_event,

                    {
                        "Index":
                            i,

                        "Fraction":
                            0.5,

                        "Price":
                            trail,

                        "Reason":
                            "RUNNER_TRAIL"
                    }
                ]

            highest_close = max(
                highest_close,
                close
            )

    if tp1_hit:

        return [
            first_event,

            {
                "Index":
                    end,

                "Fraction":
                    0.5,

                "Price":
                    float(
                        df["Close"].iloc[
                            end
                        ]
                    ),

                "Reason":
                    "RUNNER_TIME30"
            }
        ]

    return [
        {
            "Index":
                end,

            "Fraction":
                1.0,

            "Price":
                float(
                    df["Close"].iloc[
                        end
                    ]
                ),

            "Reason":
                "TIME30"
        }
    ]


# ============================================================
# CREATE EXIT PLANS
# ============================================================

def create_plans(
    candidates,
    prepared_data,
    exit_function
):

    plans = []

    for _, trade in (
        candidates.iterrows()
    ):

        ticker = (
            trade[
                "Ticker"
            ]
        )

        df = prepared_data[
            ticker
        ]

        events = (
            exit_function(
                df,
                trade
            )
        )

        if not events:

            continue

        final_index = max(
            event[
                "Index"
            ]
            for event in events
        )

        converted_events = []

        for event in events:

            converted_events.append(
                {
                    "Date":
                        pd.Timestamp(
                            df.index[
                                event[
                                    "Index"
                                ]
                            ]
                        ),

                    "Fraction":
                        float(
                            event[
                                "Fraction"
                            ]
                        ),

                    "Raw_Price":
                        float(
                            event[
                                "Price"
                            ]
                        ),

                    "Reason":
                        event[
                            "Reason"
                        ]
                }
            )

        plans.append(
            {
                **trade.to_dict(),

                "Final_Exit_Date":
                    pd.Timestamp(
                        df.index[
                            final_index
                        ]
                    ),

                "Exit_Events":
                    converted_events
            }
        )

    return pd.DataFrame(
        plans
    )


# ============================================================
# MTM PRICE
# ============================================================

def get_close(
    all_data,
    ticker,
    date
):

    try:

        df = all_data[
            ticker
        ]

        available = (
            df.loc[
                df.index <= date,
                "Close"
            ]
            .dropna()
        )

        if available.empty:

            return None

        return float(
            available.iloc[-1]
        )

    except Exception:

        return None


# ============================================================
# PORTFOLIO
# ============================================================

def simulate_portfolio(
    plans,
    all_data,
    qqq,
    backtest_start
):

    if plans.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {}
        )

    grouped = {
        date:
            group.sort_values(
                "Score",
                ascending=False
            )

        for date, group
        in plans.groupby(
            "Entry_Date"
        )
    }

    final_date = (
        plans[
            "Final_Exit_Date"
        ].max()
    )

    calendar = qqq.loc[
        (
            qqq.index >= backtest_start
        )
        &
        (
            qqq.index <= final_date
        )
    ].index

    cash = (
        STARTING_CAPITAL
    )

    last_equity = (
        STARTING_CAPITAL
    )

    open_positions = []

    completed = []

    equity_rows = []

    skipped_positions = 0

    skipped_cash = 0

    skipped_same_ticker = 0

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in calendar:

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # NEW ENTRIES FIRST
        #
        # Same conservative portfolio ordering as V11.
        # ====================================================

        if date in grouped:

            todays = (
                grouped[
                    date
                ]
            )

            risk_budget = (
                last_equity
                * RISK_PER_TRADE
            )

            for _, trade in (
                todays.iterrows()
            ):

                if (
                    len(
                        open_positions
                    )
                    >= MAX_POSITIONS
                ):

                    skipped_positions += 1

                    continue

                # With longer trailing exits,
                # reject a second position in same stock.

                already_open = any(
                    p[
                        "Ticker"
                    ]
                    ==
                    trade[
                        "Ticker"
                    ]

                    for p
                    in open_positions
                )

                if already_open:

                    skipped_same_ticker += 1

                    continue

                raw_entry = float(
                    trade[
                        "Raw_Entry"
                    ]
                )

                initial_stop = float(
                    trade[
                        "Initial_Stop"
                    ]
                )

                entry_fill = (
                    raw_entry
                    * (
                        1
                        + SLIPPAGE_PCT
                    )
                )

                stop_fill_reference = (
                    initial_stop
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                risk_per_share = (
                    entry_fill
                    - stop_fill_reference
                )

                if risk_per_share <= 0:

                    continue

                shares_by_risk = (
                    risk_budget
                    / risk_per_share
                )

                cost_per_share = (
                    entry_fill
                    * (
                        1
                        + COMMISSION_PCT
                    )
                )

                shares_by_cash = (
                    cash
                    / cost_per_share
                )

                shares = min(
                    shares_by_risk,
                    shares_by_cash
                )

                shares = (
                    np.floor(
                        shares
                        / MIN_SHARE_SIZE
                    )
                    * MIN_SHARE_SIZE
                )

                if (
                    shares
                    < MIN_SHARE_SIZE
                ):

                    skipped_cash += 1

                    continue

                entry_notional = (
                    shares
                    * entry_fill
                )

                entry_commission = (
                    entry_notional
                    * COMMISSION_PCT
                )

                entry_cost = (
                    entry_notional
                    + entry_commission
                )

                if entry_cost > cash:

                    skipped_cash += 1

                    continue

                cash -= (
                    entry_cost
                )

                open_positions.append(
                    {
                        "Ticker":
                            trade[
                                "Ticker"
                            ],

                        "Setup_Type":
                            trade[
                                "Setup_Type"
                            ],

                        "Entry_Date":
                            date,

                        "Score":
                            trade[
                                "Score"
                            ],

                        "Original_Shares":
                            shares,

                        "Remaining_Shares":
                            shares,

                        "Entry_Fill":
                            entry_fill,

                        "Entry_Cost":
                            entry_cost,

                        "Entry_Commission":
                            entry_commission,

                        "Risk_Per_Share":
                            risk_per_share,

                        "Exit_Events":
                            trade[
                                "Exit_Events"
                            ],

                        "Exit_Proceeds":
                            0.0,

                        "Exit_Commissions":
                            0.0,

                        "Exit_Reasons":
                            []
                    }
                )

        # ====================================================
        # EXIT EVENTS
        # ====================================================

        remaining_positions = []

        for position in (
            open_positions
        ):

            todays_events = [
                event
                for event
                in position[
                    "Exit_Events"
                ]
                if (
                    event[
                        "Date"
                    ]
                    == date
                )
            ]

            for event in (
                todays_events
            ):

                fraction = float(
                    event[
                        "Fraction"
                    ]
                )

                shares_to_sell = (
                    position[
                        "Original_Shares"
                    ]
                    * fraction
                )

                shares_to_sell = min(
                    shares_to_sell,
                    position[
                        "Remaining_Shares"
                    ]
                )

                if shares_to_sell <= 0:

                    continue

                raw_price = float(
                    event[
                        "Raw_Price"
                    ]
                )

                exit_fill = (
                    raw_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                exit_notional = (
                    shares_to_sell
                    * exit_fill
                )

                exit_commission = (
                    exit_notional
                    * COMMISSION_PCT
                )

                proceeds = (
                    exit_notional
                    - exit_commission
                )

                cash += (
                    proceeds
                )

                position[
                    "Remaining_Shares"
                ] -= (
                    shares_to_sell
                )

                position[
                    "Exit_Proceeds"
                ] += (
                    proceeds
                )

                position[
                    "Exit_Commissions"
                ] += (
                    exit_commission
                )

                position[
                    "Exit_Reasons"
                ].append(
                    event[
                        "Reason"
                    ]
                )

            if (
                position[
                    "Remaining_Shares"
                ]
                <= 1e-10
            ):

                net_pnl = (
                    position[
                        "Exit_Proceeds"
                    ]
                    - position[
                        "Entry_Cost"
                    ]
                )

                actual_risk = (
                    position[
                        "Original_Shares"
                    ]
                    *
                    position[
                        "Risk_Per_Share"
                    ]
                )

                net_r = (
                    net_pnl
                    / actual_risk
                    if actual_risk > 0
                    else 0
                )

                position[
                    "Net_PnL"
                ] = (
                    net_pnl
                )

                position[
                    "Net_R"
                ] = (
                    net_r
                )

                position[
                    "Exit_Date"
                ] = (
                    date
                )

                position[
                    "Exit_Reason"
                ] = (
                    " + ".join(
                        position[
                            "Exit_Reasons"
                        ]
                    )
                )

                completed.append(
                    position
                )

            else:

                remaining_positions.append(
                    position
                )

        open_positions = (
            remaining_positions
        )

        # ====================================================
        # MARK TO MARKET
        # ====================================================

        market_value = (
            0.0
        )

        for position in (
            open_positions
        ):

            price = get_close(
                all_data,
                position[
                    "Ticker"
                ],
                date
            )

            if price is None:

                price = (
                    position[
                        "Entry_Fill"
                    ]
                )

            market_value += (
                position[
                    "Remaining_Shares"
                ]
                * price
            )

        equity = (
            cash
            + market_value
        )

        last_equity = (
            equity
        )

        equity_rows.append(
            {
                "Date":
                    date,

                "Cash":
                    cash,

                "Market_Value":
                    market_value,

                "Equity":
                    equity,

                "Open_Positions":
                    len(
                        open_positions
                    )
            }
        )

    return (
        pd.DataFrame(
            completed
        ),

        pd.DataFrame(
            equity_rows
        ),

        {
            "Skipped_Positions":
                skipped_positions,

            "Skipped_Cash":
                skipped_cash,

            "Skipped_Same_Ticker":
                skipped_same_ticker
        }
    )


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_stats(
    trades,
    equity,
    diagnostics
):

    if (
        trades.empty
        or equity.empty
    ):

        return None

    ending = float(
        equity[
            "Equity"
        ].iloc[-1]
    )

    return_pct = (
        (
            ending
            / STARTING_CAPITAL
        )
        - 1
    ) * 100

    first_date = pd.Timestamp(
        equity[
            "Date"
        ].iloc[0]
    )

    last_date = pd.Timestamp(
        equity[
            "Date"
        ].iloc[-1]
    )

    years = (
        (
            last_date
            - first_date
        ).days
        / 365.25
    )

    cagr = (
        (
            ending
            / STARTING_CAPITAL
        )
        ** (
            1 / years
        )
        - 1
    ) * 100

    curve = (
        equity[
            "Equity"
        ]
    )

    peak = (
        curve.cummax()
    )

    drawdown = (
        curve
        / peak
        - 1
    ) * 100

    max_dd = float(
        drawdown.min()
    )

    winners = trades.loc[
        trades[
            "Net_PnL"
        ] > 0,
        "Net_PnL"
    ]

    losers = trades.loc[
        trades[
            "Net_PnL"
        ] < 0,
        "Net_PnL"
    ]

    gross_profit = (
        winners.sum()
    )

    gross_loss = abs(
        losers.sum()
    )

    pf = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else np.inf
    )

    commissions = (
        trades[
            "Entry_Commission"
        ].sum()
        +
        trades[
            "Exit_Commissions"
        ].sum()
    )

    return {
        "Ending_Capital":
            ending,

        "Net_Profit":
            ending
            - STARTING_CAPITAL,

        "Return_%":
            return_pct,

        "CAGR_%":
            cagr,

        "Trades":
            len(trades),

        "Profitable_%":
            (
                trades[
                    "Net_PnL"
                ] > 0
            ).mean()
            * 100,

        "Avg_Net_R":
            trades[
                "Net_R"
            ].mean(),

        "Median_R":
            trades[
                "Net_R"
            ].median(),

        "Profit_Factor":
            pf,

        "Max_MTM_DD_%":
            max_dd,

        "Commissions":
            commissions,

        "Skipped_Positions":
            diagnostics[
                "Skipped_Positions"
            ],

        "Skipped_Same_Ticker":
            diagnostics[
                "Skipped_Same_Ticker"
            ]
    }


# ============================================================
# YEARLY
# ============================================================

def yearly_performance(
    trades
):

    df = trades.copy()

    df["Year"] = (
        pd.to_datetime(
            df[
                "Exit_Date"
            ]
        )
        .dt.year
    )

    return (
        df
        .groupby(
            "Year"
        )
        .agg(
            Trades=(
                "Net_PnL",
                "count"
            ),

            Net_PnL=(
                "Net_PnL",
                "sum"
            ),

            Avg_Net_R=(
                "Net_R",
                "mean"
            ),

            Profitable_Pct=(
                "Net_PnL",
                lambda x:
                (
                    x > 0
                ).mean()
                * 100
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "V12.1 - CORRECTED EXIT TOURNAMENT"
    )

    print(
        "=" * 120
    )

    print(
        f"Capital:              "
        f"${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Risk/trade:           "
        f"{RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Max positions:        "
        f"{MAX_POSITIONS}"
    )

    print(
        f"Runner max holding:   "
        f"{TRAIL_HOLD_DAYS} days"
    )

    print()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    print(
        f"Downloading "
        f"{len(TICKERS)} stocks..."
    )

    all_data = yf.download(
        TICKERS,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=True,
        repair=True,
        threads=True
    )

    qqq = download_market(
        "QQQ",
        "QQQ"
    )

    dxy = download_market(
        "DX-Y.NYB",
        "DXY"
    )

    end_date = min(
        pd.Timestamp(
            qqq.index.max()
        ),
        pd.Timestamp(
            dxy.index.max()
        )
    )

    backtest_start = (
        end_date
        - pd.DateOffset(
            years=BACKTEST_YEARS
        )
    )

    print(
        f"Backtest: "
        f"{backtest_start.date()} "
        f"-> {end_date.date()}"
    )

    # ========================================================
    # PREPARE STOCK DATA ONCE
    # ========================================================

    prepared_data = {}

    for ticker in TICKERS:

        try:

            if (
                ticker
                in
                all_data.columns.get_level_values(0)
            ):

                prepared_data[
                    ticker
                ] = prepare_stock(
                    all_data[
                        ticker
                    ].copy()
                )

        except Exception:

            pass

    # ========================================================
    # EXACT V11 CANDIDATES
    # ========================================================

    (
        candidates,
        candidate_diag
    ) = generate_v11_candidates(
        all_data,
        qqq,
        dxy,
        backtest_start
    )

    print()

    print(
        f"V11-equivalent candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Breakout signals:          "
        f"{candidate_diag['Breakout_Signals']}"
    )

    print(
        f"Pullback signals:          "
        f"{candidate_diag['Pullback_Signals']}"
    )

    print(
        f"Rejected QQQ/DXY:          "
        f"{candidate_diag['Rejected_Market']}"
    )

    if candidates.empty:

        print(
            "NO CANDIDATES."
        )

        return

    # ========================================================
    # TOURNAMENT
    # ========================================================

    exits = {
        "Fixed 2R":
            plan_fixed_2r,

        "Full 2.0 ATR Trail":
            plan_full_2atr,

        "Full 2.5 ATR Trail":
            plan_full_25atr,

        "Full 3.0 ATR Trail":
            plan_full_3atr,

        "50% 2R + 2.5 ATR Trail":
            plan_half_2r_25atr
    }

    summary_rows = []

    outputs = {}

    for (
        exit_name,
        exit_function
    ) in exits.items():

        print()

        print(
            "=" * 120
        )

        print(
            exit_name.upper()
        )

        print(
            "=" * 120
        )

        plans = create_plans(
            candidates,
            prepared_data,
            exit_function
        )

        (
            trades,
            equity,
            portfolio_diag
        ) = simulate_portfolio(
            plans,
            all_data,
            qqq,
            backtest_start
        )

        stats = calculate_stats(
            trades,
            equity,
            portfolio_diag
        )

        if stats is None:

            continue

        stats[
            "Exit_Strategy"
        ] = (
            exit_name
        )

        summary_rows.append(
            stats
        )

        yearly = (
            yearly_performance(
                trades
            )
        )

        outputs[
            exit_name
        ] = {
            "trades":
                trades,

            "equity":
                equity,

            "yearly":
                yearly
        }

        print(
            f"Ending capital:        "
            f"${stats['Ending_Capital']:,.2f}"
        )

        print(
            f"Net profit:            "
            f"${stats['Net_Profit']:,.2f}"
        )

        print(
            f"Return:                "
            f"{stats['Return_%']:.2f}%"
        )

        print(
            f"CAGR:                  "
            f"{stats['CAGR_%']:.2f}%"
        )

        print(
            f"Trades:                "
            f"{stats['Trades']}"
        )

        print(
            f"Profitable:            "
            f"{stats['Profitable_%']:.2f}%"
        )

        print(
            f"Avg Net R:             "
            f"{stats['Avg_Net_R']:.3f}R"
        )

        print(
            f"Median R:              "
            f"{stats['Median_R']:.3f}R"
        )

        print(
            f"Profit Factor:         "
            f"{stats['Profit_Factor']:.3f}"
        )

        print(
            f"TRUE MTM DD:           "
            f"{stats['Max_MTM_DD_%']:.2f}%"
        )

        print(
            f"Skipped positions:     "
            f"{stats['Skipped_Positions']}"
        )

        print(
            f"Skipped same ticker:   "
            f"{stats['Skipped_Same_Ticker']}"
        )

        print()

        print(
            "YEAR-BY-YEAR"
        )

        print(
            yearly
            .round(3)
            .to_string()
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .set_index(
            "Exit_Strategy"
        )
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V12.1 FINAL EXIT COMPARISON"
    )

    print(
        "=" * 120
    )

    columns = [
        "Ending_Capital",
        "Net_Profit",
        "Return_%",
        "CAGR_%",
        "Trades",
        "Profitable_%",
        "Avg_Net_R",
        "Median_R",
        "Profit_Factor",
        "Max_MTM_DD_%",
        "Commissions",
        "Skipped_Positions",
        "Skipped_Same_Ticker"
    ]

    print(
        summary[
            columns
        ]
        .round(3)
        .to_string()
    )

    # ========================================================
    # BASELINE VALIDATION
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "BASELINE VALIDATION"
    )

    print(
        "=" * 120
    )

    baseline = (
        summary.loc[
            "Fixed 2R"
        ]
    )

    print(
        "Expected approximately from V11:"
    )

    print(
        "Trades ~150"
    )

    print(
        "Avg Net R ~0.104R"
    )

    print(
        "Profit Factor ~1.315"
    )

    print(
        "CAGR ~1.52%"
    )

    print()

    print(
        "Current Fixed-2R baseline:"
    )

    print(
        f"Trades:        "
        f"{baseline['Trades']:.0f}"
    )

    print(
        f"Avg Net R:     "
        f"{baseline['Avg_Net_R']:.3f}R"
    )

    print(
        f"Profit Factor: "
        f"{baseline['Profit_Factor']:.3f}"
    )

    print(
        f"CAGR:          "
        f"{baseline['CAGR_%']:.2f}%"
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "TARGET CHECK"
    )

    print(
        "=" * 120
    )

    for exit_name, row in (
        summary.iterrows()
    ):

        checks = {
            "CAGR >= 10%":
                row[
                    "CAGR_%"
                ] >= 10,

            "PF >= 1.25":
                row[
                    "Profit_Factor"
                ] >= 1.25,

            "Avg R >= 0.10":
                row[
                    "Avg_Net_R"
                ] >= 0.10,

            "Max DD <= 15%":
                row[
                    "Max_MTM_DD_%"
                ] >= -15
        }

        print()

        print(
            exit_name
        )

        for name, passed in (
            checks.items()
        ):

            print(
                f"  {name:<20} "
                f"{'PASS' if passed else 'FAIL'}"
            )

        print(
            f"  Score: "
            f"{sum(checks.values())}/4"
        )

    # ========================================================
    # SAVE
    # ========================================================

    summary.to_csv(
        "v12_1_exit_comparison.csv"
    )

    candidates.to_csv(
        "v12_1_v11_candidates.csv",
        index=False
    )

    for (
        exit_name,
        output
    ) in outputs.items():

        safe_name = (
            exit_name
            .lower()
            .replace(
                "%",
                "pct"
            )
            .replace(
                "+",
                "plus"
            )
            .replace(
                ".",
                "_"
            )
            .replace(
                " ",
                "_"
            )
        )

        output[
            "trades"
        ].to_csv(
            f"v12_1_{safe_name}_trades.csv",
            index=False
        )

        output[
            "equity"
        ].to_csv(
            f"v12_1_{safe_name}_equity.csv",
            index=False
        )

        output[
            "yearly"
        ].to_csv(
            f"v12_1_{safe_name}_yearly.csv"
        )

    print()

    print(
        "=" * 120
    )

    print(
        "V12.1 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
