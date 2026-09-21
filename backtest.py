import yfinance as yf
import pandas as pd
import numpy as np
from collections import Counter, defaultdict


# ============================================================
# V18 - FILTER FUNNEL / INACTIVE DAYS DIAGNOSTIC
#
# EXACT V15/V17 STRATEGY
#
# NO STRATEGY PARAMETERS CHANGED.
#
# PURPOSE:
# Find what actually limits trading frequency:
#
# - QQQ regime?
# - DXY regime?
# - Strong trend?
# - RSI?
# - Stoch RSI cross?
# - EMA20 pullback?
# - Bullish confirmation?
# - Relative volume?
# - Stop/risk requirement?
# - Entry trigger?
#
# ============================================================


# ============================================================
# ACCOUNT / STRATEGY
# ============================================================

STARTING_CAPITAL = 1000.0

RISK_PER_TRADE = 0.0075

MAX_POSITIONS = 5

SLIPPAGE_PCT = 0.0005

COMMISSION_PCT = 0.0002

MIN_SHARE_SIZE = 0.0001

DOWNLOAD_PERIOD = "max"

BACKTEST_YEARS = 15

ATR_PERIOD = 14

ENTRY_VALID_DAYS = 3

FIXED_HOLD_DAYS = 10

TRAIL_HOLD_DAYS = 30

ATR_TRAIL_MULT = 2.5

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


def calculate_stoch_rsi(rsi, period=14):

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


def calculate_atr(df, period=14):

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

    df["ATR14"] = calculate_atr(
        df,
        ATR_PERIOD
    )

    df["RSI14"] = calculate_rsi(
        df["Close"]
    )

    df["STOCH_RSI"] = calculate_stoch_rsi(
        df["RSI14"],
        STOCH_RSI_PERIOD
    )

    df["AVG_VOL20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["LOW5"] = (
        df["Low"]
        .shift(1)
        .rolling(5)
        .min()
    )

    return df


# ============================================================
# MARKET
# ============================================================

def download_market(ticker, name):

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

    row = (
        available.iloc[-1]
    )

    if pd.isna(
        row["SMA200"]
    ):

        return None

    return row


def market_status(
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
        or
        dxy_row is None
    ):

        return None, None

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
        qqq_ok,
        dxy_ok
    )


def market_allows_long(
    qqq,
    dxy,
    date
):

    qqq_ok, dxy_ok = (
        market_status(
            qqq,
            dxy,
            date
        )
    )

    return (
        qqq_ok is True
        and
        dxy_ok is True
    )


# ============================================================
# STRONG TREND
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
        df["SMA50"].iloc[
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
# QUALITY SCORE
# ============================================================

def quality_score(
    df,
    i
):

    row = df.iloc[i]

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])

    sma50 = float(row["SMA50"])
    ema20 = float(row["EMA20"])
    atr = float(row["ATR14"])

    volume = float(row["Volume"])
    avg_volume = float(row["AVG_VOL20"])

    old_sma50 = float(
        df["SMA50"].iloc[
            i - 10
        ]
    )

    slope = (
        sma50 - old_sma50
    ) / atr

    ema_strength = (
        ema20 - sma50
    ) / atr

    candle_range = (
        high - low
    )

    close_location = (
        (
            close - low
        )
        / candle_range
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
# EXACT SETUP
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
        df["STOCH_RSI"].iloc[
            i - 1
        ]
    )

    if (
        pd.isna(stoch)
        or
        pd.isna(previous_stoch)
    ):

        return None

    if not (
        45 <= rsi <= 65
    ):

        return None

    if not (
        previous_stoch
        <= STOCH_CROSS_LEVEL
        and
        stoch
        > STOCH_CROSS_LEVEL
    ):

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
# ENTRY
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
# ============================================================
#
# DIAGNOSTIC FUNNEL
#
# IMPORTANT:
#
# This scans every usable stock/day independently.
#
# It DOES NOT change the actual strategy.
#
# ============================================================
# ============================================================

FUNNEL_STAGES = [
    "USABLE_DATA",
    "STRONG_TREND",
    "RSI",
    "STOCH_CROSS",
    "EMA_PULLBACK",
    "CLOSE_ABOVE_EMA",
    "BULLISH_CANDLE",
    "CLOSE_LOCATION",
    "REL_VOLUME",
    "VALID_RISK",
    "TRIGGERED",
    "MARKET_REGIME"
]


def diagnose_stock_day(
    df,
    i
):

    """
    Returns:
        passed_stages
        first_failure
        trigger
    """

    passed = []

    row = df.iloc[i]

    required = [
        "SMA200",
        "SMA50",
        "EMA20",
        "ATR14",
        "RSI14",
        "STOCH_RSI",
        "AVG_VOL20",
        "LOW5"
    ]

    if any(
        pd.isna(
            row[col]
        )
        for col in required
    ):

        return (
            passed,
            "MISSING_DATA",
            None
        )

    if i < 10:

        return (
            passed,
            "MISSING_DATA",
            None
        )

    previous_stoch = (
        df["STOCH_RSI"]
        .iloc[
            i - 1
        ]
    )

    if pd.isna(
        previous_stoch
    ):

        return (
            passed,
            "MISSING_DATA",
            None
        )

    passed.append(
        "USABLE_DATA"
    )


    # ========================================================
    # STRONG TREND
    # ========================================================

    if not strong_trend(
        df,
        i
    ):

        return (
            passed,
            "STRONG_TREND",
            None
        )

    passed.append(
        "STRONG_TREND"
    )


    # ========================================================
    # VALUES
    # ========================================================

    close = float(
        row["Close"]
    )

    open_price = float(
        row["Open"]
    )

    high = float(
        row["High"]
    )

    low = float(
        row["Low"]
    )

    ema20 = float(
        row["EMA20"]
    )

    atr = float(
        row["ATR14"]
    )

    rsi = float(
        row["RSI14"]
    )

    stoch = float(
        row["STOCH_RSI"]
    )

    volume = float(
        row["Volume"]
    )

    avg_volume = float(
        row["AVG_VOL20"]
    )

    low5 = float(
        row["LOW5"]
    )


    # ========================================================
    # RSI
    # ========================================================

    if not (
        45 <= rsi <= 65
    ):

        return (
            passed,
            "RSI",
            None
        )

    passed.append(
        "RSI"
    )


    # ========================================================
    # STOCH RSI CROSS
    # ========================================================

    crossed = (
        float(
            previous_stoch
        )
        <= STOCH_CROSS_LEVEL
        and
        stoch
        > STOCH_CROSS_LEVEL
    )

    if not crossed:

        return (
            passed,
            "STOCH_CROSS",
            None
        )

    passed.append(
        "STOCH_CROSS"
    )


    # ========================================================
    # EMA PULLBACK
    # ========================================================

    ema_distance = (
        abs(
            low - ema20
        )
        / atr
    )

    if ema_distance > 0.50:

        return (
            passed,
            "EMA_PULLBACK",
            None
        )

    passed.append(
        "EMA_PULLBACK"
    )


    # ========================================================
    # CLOSE ABOVE EMA
    # ========================================================

    if close <= ema20:

        return (
            passed,
            "CLOSE_ABOVE_EMA",
            None
        )

    passed.append(
        "CLOSE_ABOVE_EMA"
    )


    # ========================================================
    # BULLISH CANDLE
    # ========================================================

    if close <= open_price:

        return (
            passed,
            "BULLISH_CANDLE",
            None
        )

    passed.append(
        "BULLISH_CANDLE"
    )


    # ========================================================
    # CLOSE LOCATION
    # ========================================================

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return (
            passed,
            "CLOSE_LOCATION",
            None
        )

    close_location = (
        close - low
    ) / candle_range

    if close_location < 0.60:

        return (
            passed,
            "CLOSE_LOCATION",
            None
        )

    passed.append(
        "CLOSE_LOCATION"
    )


    # ========================================================
    # RELATIVE VOLUME
    # ========================================================

    if avg_volume <= 0:

        return (
            passed,
            "REL_VOLUME",
            None
        )

    relative_volume = (
        volume
        / avg_volume
    )

    if relative_volume < 0.80:

        return (
            passed,
            "REL_VOLUME",
            None
        )

    passed.append(
        "REL_VOLUME"
    )


    # ========================================================
    # STOP / RISK
    # ========================================================

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

    if (
        risk <= 0
        or
        risk < (
            0.50 * atr
        )
        or
        risk > (
            3.0 * atr
        )
    ):

        return (
            passed,
            "VALID_RISK",
            None
        )

    passed.append(
        "VALID_RISK"
    )

    return (
        passed,
        None,
        trigger
    )


# ============================================================
# RUN FULL FILTER DIAGNOSTIC
# ============================================================

def run_filter_diagnostic(
    prepared_data,
    qqq,
    dxy,
    backtest_start,
    end_date
):

    stage_counts = Counter()

    fail_counts = Counter()

    stage_dates = defaultdict(
        set
    )

    trigger_count = 0

    trigger_dates = set()

    regime_pass_count = 0

    regime_pass_dates = set()

    market_rejected_entries = 0

    market_rejected_dates = set()


    for ticker, df in (
        prepared_data.items()
    ):

        if len(df) < 250:

            continue

        for i in range(
            210,
            len(df) - 1
        ):

            date = pd.Timestamp(
                df.index[i]
            )

            if (
                date < backtest_start
                or
                date > end_date
            ):

                continue

            (
                passed,
                first_failure,
                trigger
            ) = diagnose_stock_day(
                df,
                i
            )

            for stage in passed:

                stage_counts[
                    stage
                ] += 1

                stage_dates[
                    stage
                ].add(
                    date
                )

            if first_failure is not None:

                fail_counts[
                    first_failure
                ] += 1

                continue


            # =================================================
            # Valid setup exists.
            # Now check actual trigger within 3 days.
            # =================================================

            entry_index = find_entry(
                df,
                i,
                trigger
            )

            if entry_index is None:

                fail_counts[
                    "ENTRY_TRIGGER"
                ] += 1

                continue


            trigger_count += 1

            stage_counts[
                "TRIGGERED"
            ] += 1

            entry_date = pd.Timestamp(
                df.index[
                    entry_index
                ]
            )

            trigger_dates.add(
                entry_date
            )

            stage_dates[
                "TRIGGERED"
            ].add(
                entry_date
            )


            # =================================================
            # EXACT MARKET FILTER AT ENTRY
            # =================================================

            if market_allows_long(
                qqq,
                dxy,
                entry_date
            ):

                regime_pass_count += 1

                stage_counts[
                    "MARKET_REGIME"
                ] += 1

                regime_pass_dates.add(
                    entry_date
                )

                stage_dates[
                    "MARKET_REGIME"
                ].add(
                    entry_date
                )

            else:

                fail_counts[
                    "MARKET_REGIME"
                ] += 1

                market_rejected_entries += 1

                market_rejected_dates.add(
                    entry_date
                )


    return {
        "Stage_Counts":
            stage_counts,

        "Fail_Counts":
            fail_counts,

        "Stage_Dates":
            stage_dates,

        "Trigger_Count":
            trigger_count,

        "Trigger_Dates":
            trigger_dates,

        "Regime_Pass_Count":
            regime_pass_count,

        "Regime_Pass_Dates":
            regime_pass_dates,

        "Market_Rejected":
            market_rejected_entries,

        "Market_Rejected_Dates":
            market_rejected_dates
    }


# ============================================================
# MARKET REGIME DIAGNOSTIC
# ============================================================

def analyze_market_regime(
    qqq,
    dxy,
    backtest_start,
    end_date
):

    calendar = qqq.loc[
        (
            qqq.index >= backtest_start
        )
        &
        (
            qqq.index <= end_date
        )
    ].index

    counts = Counter()

    rows = []

    for date in calendar:

        date = pd.Timestamp(
            date
        )

        (
            qqq_ok,
            dxy_ok
        ) = market_status(
            qqq,
            dxy,
            date
        )

        if (
            qqq_ok is None
            or
            dxy_ok is None
        ):

            counts[
                "NO_DATA"
            ] += 1

            continue

        if (
            qqq_ok
            and dxy_ok
        ):

            state = (
                "BOTH_PASS"
            )

        elif (
            not qqq_ok
            and dxy_ok
        ):

            state = (
                "QQQ_FAIL_ONLY"
            )

        elif (
            qqq_ok
            and not dxy_ok
        ):

            state = (
                "DXY_FAIL_ONLY"
            )

        else:

            state = (
                "BOTH_FAIL"
            )

        counts[
            state
        ] += 1

        rows.append(
            {
                "Date":
                    date,

                "QQQ_OK":
                    qqq_ok,

                "DXY_OK":
                    dxy_ok,

                "State":
                    state
            }
        )

    return (
        counts,
        pd.DataFrame(
            rows
        )
    )


# ============================================================
# PRINT MARKET REGIME
# ============================================================

def print_market_regime(
    counts
):

    usable = (
        counts[
            "BOTH_PASS"
        ]
        +
        counts[
            "QQQ_FAIL_ONLY"
        ]
        +
        counts[
            "DXY_FAIL_ONLY"
        ]
        +
        counts[
            "BOTH_FAIL"
        ]
    )

    print()

    print(
        "=" * 120
    )

    print(
        "1. MARKET REGIME BOTTLENECK"
    )

    print(
        "=" * 120
    )

    if usable == 0:

        return

    labels = [
        (
            "QQQ PASS + DXY PASS",
            "BOTH_PASS"
        ),

        (
            "QQQ FAIL only",
            "QQQ_FAIL_ONLY"
        ),

        (
            "DXY FAIL only",
            "DXY_FAIL_ONLY"
        ),

        (
            "BOTH FAIL",
            "BOTH_FAIL"
        )
    ]

    for label, key in labels:

        count = counts[
            key
        ]

        pct = (
            count
            / usable
            * 100
        )

        print(
            f"{label:<25}"
            f"{count:>6} days"
            f"   ({pct:6.2f}%)"
        )

    allowed = counts[
        "BOTH_PASS"
    ]

    blocked = (
        usable
        - allowed
    )

    print()

    print(
        f"Trading regime ON:      "
        f"{allowed} days "
        f"({allowed / usable * 100:.2f}%)"
    )

    print(
        f"Trading regime OFF:     "
        f"{blocked} days "
        f"({blocked / usable * 100:.2f}%)"
    )


# ============================================================
# PRINT FUNNEL
# ============================================================

def print_funnel(
    diagnostic
):

    stage_counts = diagnostic[
        "Stage_Counts"
    ]

    stage_dates = diagnostic[
        "Stage_Dates"
    ]

    print()

    print(
        "=" * 120
    )

    print(
        "2. SEQUENTIAL SETUP FUNNEL"
    )

    print(
        "=" * 120
    )

    print(
        "Counts below are sequential."
    )

    print(
        "A stock/day reaches the next row only if it passed all previous rows."
    )

    print()

    labels = [
        (
            "Usable stock-days",
            "USABLE_DATA"
        ),

        (
            "Strong trend",
            "STRONG_TREND"
        ),

        (
            "RSI 45-65",
            "RSI"
        ),

        (
            "Stoch RSI cross >20",
            "STOCH_CROSS"
        ),

        (
            "EMA20 pullback",
            "EMA_PULLBACK"
        ),

        (
            "Close > EMA20",
            "CLOSE_ABOVE_EMA"
        ),

        (
            "Bullish candle",
            "BULLISH_CANDLE"
        ),

        (
            "Close location >=60%",
            "CLOSE_LOCATION"
        ),

        (
            "Relative volume >=0.80",
            "REL_VOLUME"
        ),

        (
            "Valid stop / risk",
            "VALID_RISK"
        ),

        (
            "Entry triggered <=3d",
            "TRIGGERED"
        ),

        (
            "QQQ + DXY regime pass",
            "MARKET_REGIME"
        )
    ]

    previous = None

    for label, key in labels:

        count = stage_counts[
            key
        ]

        dates = len(
            stage_dates[
                key
            ]
        )

        if previous is None:

            retention = 100.0

        else:

            retention = (
                count
                / previous
                * 100
                if previous > 0
                else 0
            )

        print(
            f"{label:<28}"
            f"{count:>8} stock-days"
            f" | {dates:>5} unique days"
            f" | retains {retention:6.2f}%"
        )

        previous = count


# ============================================================
# FIRST FAIL REASON
# ============================================================

def print_fail_reasons(
    diagnostic
):

    fails = diagnostic[
        "Fail_Counts"
    ]

    total_failures = sum(
        fails.values()
    )

    print()

    print(
        "=" * 120
    )

    print(
        "3. FIRST FAILURE REASON"
    )

    print(
        "=" * 120
    )

    print(
        "Each stock/day is counted only at the FIRST rule that rejects it."
    )

    print()

    sorted_fails = sorted(
        fails.items(),
        key=lambda x:
        x[1],
        reverse=True
    )

    for reason, count in (
        sorted_fails
    ):

        pct = (
            count
            / total_failures
            * 100
            if total_failures > 0
            else 0
        )

        print(
            f"{reason:<25}"
            f"{count:>9}"
            f"   ({pct:6.2f}%)"
        )


# ============================================================
# DAILY OPPORTUNITY SUMMARY
# ============================================================

def print_daily_summary(
    diagnostic,
    qqq,
    backtest_start,
    end_date
):

    calendar = qqq.loc[
        (
            qqq.index >= backtest_start
        )
        &
        (
            qqq.index <= end_date
        )
    ].index

    total_days = len(
        calendar
    )

    trend_days = len(
        diagnostic[
            "Stage_Dates"
        ][
            "STRONG_TREND"
        ]
    )

    setup_days = len(
        diagnostic[
            "Stage_Dates"
        ][
            "VALID_RISK"
        ]
    )

    trigger_days = len(
        diagnostic[
            "Trigger_Dates"
        ]
    )

    final_days = len(
        diagnostic[
            "Regime_Pass_Dates"
        ]
    )

    print()

    print(
        "=" * 120
    )

    print(
        "4. UNIQUE-DAY OPPORTUNITY SUMMARY"
    )

    print(
        "=" * 120
    )

    rows = [
        (
            "All market days",
            total_days
        ),

        (
            ">=1 stock in strong trend",
            trend_days
        ),

        (
            ">=1 complete setup",
            setup_days
        ),

        (
            ">=1 setup triggered",
            trigger_days
        ),

        (
            ">=1 triggered + regime OK",
            final_days
        )
    ]

    for label, days in rows:

        pct = (
            days
            / total_days
            * 100
            if total_days > 0
            else 0
        )

        print(
            f"{label:<32}"
            f"{days:>5} days"
            f"   ({pct:6.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "V18 - FILTER FUNNEL / INACTIVE DAYS DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        "NO strategy rules are changed."
    )

    print()

    print(
        "We are measuring which rules reduce trading frequency."
    )

    print()

    print(
        f"History:               "
        f"{BACKTEST_YEARS} years"
    )

    print(
        f"Risk:                  "
        f"{RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Max positions:         "
        f"{MAX_POSITIONS}"
    )

    print(
        f"ATR trail:             "
        f"{ATR_TRAIL_MULT:.1f}"
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    print()

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

    requested_start = (
        end_date
        - pd.DateOffset(
            years=BACKTEST_YEARS
        )
    )

    backtest_start = max(
        requested_start,
        pd.Timestamp(
            qqq.index.min()
        ),
        pd.Timestamp(
            dxy.index.min()
        )
    )

    print(
        f"Backtest:              "
        f"{backtest_start.date()} "
        f"-> {end_date.date()}"
    )


    # ========================================================
    # PREPARE DATA ONCE
    # ========================================================

    prepared_data = {}

    for ticker in TICKERS:

        try:

            if (
                ticker
                in
                all_data.columns
                .get_level_values(0)
            ):

                df = prepare_stock(
                    all_data[
                        ticker
                    ].copy()
                )

                if not df.empty:

                    prepared_data[
                        ticker
                    ] = df

        except Exception as e:

            print(
                f"PREP ERROR {ticker}: "
                f"{e}"
            )

    print(
        f"Prepared stocks:       "
        f"{len(prepared_data)}"
    )


    # ========================================================
    # MARKET REGIME
    # ========================================================

    (
        regime_counts,
        regime_daily
    ) = analyze_market_regime(
        qqq,
        dxy,
        backtest_start,
        end_date
    )

    print_market_regime(
        regime_counts
    )


    # ========================================================
    # FILTER FUNNEL
    # ========================================================

    diagnostic = (
        run_filter_diagnostic(
            prepared_data,
            qqq,
            dxy,
            backtest_start,
            end_date
        )
    )

    print_funnel(
        diagnostic
    )

    print_fail_reasons(
        diagnostic
    )

    print_daily_summary(
        diagnostic,
        qqq,
        backtest_start,
        end_date
    )


    # ========================================================
    # SIMPLE BOTTLENECK RANKING
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "5. BIGGEST BOTTLENECKS"
    )

    print(
        "=" * 120
    )

    failures = (
        diagnostic[
            "Fail_Counts"
        ]
    )

    relevant = {
        k: v
        for k, v in failures.items()
        if k != "MISSING_DATA"
    }

    ranking = sorted(
        relevant.items(),
        key=lambda x:
        x[1],
        reverse=True
    )

    for rank, (
        reason,
        count
    ) in enumerate(
        ranking[:10],
        start=1
    ):

        print(
            f"{rank:>2}. "
            f"{reason:<25}"
            f"{count:>9}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    funnel_rows = []

    for stage in FUNNEL_STAGES:

        funnel_rows.append(
            {
                "Stage":
                    stage,

                "Stock_Day_Count":
                    diagnostic[
                        "Stage_Counts"
                    ][stage],

                "Unique_Days":
                    len(
                        diagnostic[
                            "Stage_Dates"
                        ][stage]
                    )
            }
        )

    pd.DataFrame(
        funnel_rows
    ).to_csv(
        "v18_filter_funnel.csv",
        index=False
    )

    pd.DataFrame(
        [
            {
                "Failure":
                    key,

                "Count":
                    value
            }
            for key, value
            in diagnostic[
                "Fail_Counts"
            ].items()
        ]
    ).sort_values(
        "Count",
        ascending=False
    ).to_csv(
        "v18_first_failure.csv",
        index=False
    )

    regime_daily.to_csv(
        "v18_market_regime_daily.csv",
        index=False
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V18 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
