import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V11 - UNIFIED MOMENTUM STRATEGY
#
# MARKET:
#   QQQ > SMA200
#   DXY < SMA200
#
# STOCK TREND:
#   Price > SMA200
#   SMA50 > SMA200
#   EMA20 > SMA50
#   SMA50 rising
#
# ENTRY TYPE A:
#   Momentum Breakout
#
# ENTRY TYPE B:
#   Momentum Pullback + StochRSI cross above 20
#
# EXIT:
#   Fixed 2R
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
# HISTORY
# ============================================================

DOWNLOAD_PERIOD = "7y"
BACKTEST_YEARS = 5

ATR_PERIOD = 14
ENTRY_VALID_DAYS = 3
MAX_HOLDING_DAYS = 10

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

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# STOCHASTIC RSI
# 0 - 100
# ============================================================

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


# ============================================================
# ATR
# ============================================================

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
            df["Close"],
            14
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

    # Previous highs/lows only.
    # No look-ahead.

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
# QQQ / DXY
# ============================================================

def download_market(
    ticker,
    name
):

    print(
        f"Downloading {name}..."
    )

    data = yf.download(
        ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=True,
        repair=True
    )

    if data.empty:

        raise RuntimeError(
            f"No {name} data."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data.dropna(
        inplace=True
    )

    data["SMA200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )

    return data


def market_row(
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


def market_allows_long(
    qqq,
    dxy,
    date
):

    qqq_row = market_row(
        qqq,
        date
    )

    dxy_row = market_row(
        dxy,
        date
    )

    if (
        qqq_row is None
        or dxy_row is None
    ):
        return False

    qqq_bull = (
        float(
            qqq_row["Close"]
        )
        >
        float(
            qqq_row["SMA200"]
        )
    )

    dxy_weak = (
        float(
            dxy_row["Close"]
        )
        <
        float(
            dxy_row["SMA200"]
        )
    )

    return (
        qqq_bull
        and dxy_weak
    )


# ============================================================
# COMMON STRONG-TREND FILTER
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

    if close <= sma200:
        return False

    if sma50 <= sma200:
        return False

    if ema20 <= sma50:
        return False

    if sma50 <= old_sma50:
        return False

    return True


# ============================================================
# COMMON RANKING SCORE
#
# Same scale for BOTH setup types.
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
        (close - low)
        / candle_range
        if candle_range > 0
        else 0
    )

    relative_volume = (
        volume / avg_volume
        if avg_volume > 0
        else 0
    )

    # Cap volume contribution so one crazy
    # volume day cannot dominate ranking.

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
# SETUP A:
# MOMENTUM BREAKOUT
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

    # Momentum zone
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

    # Bullish candle
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

    # Must already be near previous 20-day high
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
# SETUP B:
# MOMENTUM PULLBACK
#
# Strong trend remains intact.
# Pullback to EMA20.
# Stoch RSI crosses ABOVE 20.
# ============================================================

def momentum_pullback_setup(
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

    # --------------------------------------------
    # Still a momentum stock,
    # not the old weak RSI pullback.
    # --------------------------------------------

    if not (
        45 <= rsi <= 65
    ):
        return None

    # --------------------------------------------
    # Stochastic RSI bullish cross above 20
    # --------------------------------------------

    crossed_above_20 = (
        previous_stoch
        <= STOCH_CROSS_LEVEL
        and
        stoch
        > STOCH_CROSS_LEVEL
    )

    if not crossed_above_20:
        return None

    # --------------------------------------------
    # Pullback must be close to EMA20
    # --------------------------------------------

    ema_distance = (
        abs(
            low - ema20
        )
        / atr
    )

    if ema_distance > 0.50:
        return None

    # Must reclaim / remain above EMA20

    if close <= ema20:
        return None

    # Bullish reversal candle

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

    # Avoid completely dead pullbacks.
    # Does NOT require breakout-level volume.

    if relative_volume < 0.80:
        return None

    trigger = (
        high
        + 0.05 * atr
    )

    # Structure stop:
    # below recent pullback structure.

    stop_reference = min(
        low,
        low5
    )

    stop = (
        stop_reference
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
            df["High"]
            .iloc[i]
        ) >= trigger:

            return i

    return None


# ============================================================
# SIMULATE FIXED 2R
#
# Gap-aware ENTRY is used when defining target.
# ============================================================

def simulate_trade(
    df,
    entry_index,
    trigger,
    stop
):

    entry_open = float(
        df["Open"]
        .iloc[
            entry_index
        ]
    )

    # If price gaps above trigger,
    # actual raw entry is the opening price.

    raw_entry = max(
        trigger,
        entry_open
    )

    planned_risk = (
        raw_entry - stop
    )

    if planned_risk <= 0:
        return None

    # TRUE 2R from actual raw entry.

    target = (
        raw_entry
        + 2.0 * planned_risk
    )

    end = min(
        entry_index
        + MAX_HOLDING_DAYS
        - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        end + 1
    ):

        high = float(
            df["High"]
            .iloc[i]
        )

        low = float(
            df["Low"]
            .iloc[i]
        )

        open_price = float(
            df["Open"]
            .iloc[i]
        )

        # --------------------------------------------
        # Gap below stop
        # --------------------------------------------

        if open_price < stop:

            return {
                "Exit_Index":
                    i,

                "Raw_Exit":
                    open_price,

                "Result":
                    "SL_GAP",

                "Raw_Entry":
                    raw_entry,

                "Target":
                    target
            }

        # Conservative same-bar logic:
        # SL wins if both touched.

        if low <= stop:

            return {
                "Exit_Index":
                    i,

                "Raw_Exit":
                    stop,

                "Result":
                    "SL",

                "Raw_Entry":
                    raw_entry,

                "Target":
                    target
            }

        if high >= target:

            return {
                "Exit_Index":
                    i,

                "Raw_Exit":
                    target,

                "Result":
                    "TP",

                "Raw_Entry":
                    raw_entry,

                "Target":
                    target
            }

    return {
        "Exit_Index":
            end,

        "Raw_Exit":
            float(
                df["Close"]
                .iloc[end]
            ),

        "Result":
            "TIME",

        "Raw_Entry":
            raw_entry,

        "Target":
            target
    }


# ============================================================
# GENERATE UNIFIED CANDIDATES
# ============================================================

def generate_candidates(
    all_data,
    qqq,
    dxy,
    backtest_start
):

    candidates = []

    rejected_market = 0

    breakout_signals = 0
    pullback_signals = 0

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

            df = prepare_stock(
                df
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

                if (
                    signal_date
                    < backtest_start
                ):

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

                # =================================================
                # TRY BOTH ENTRY TYPES
                # =================================================

                breakout = (
                    breakout_setup(
                        df,
                        i
                    )
                )

                pullback = (
                    momentum_pullback_setup(
                        df,
                        i
                    )
                )

                # If BOTH happen on same stock / same day:
                # prefer breakout because it has stronger
                # demonstrated standalone results.

                if breakout is not None:

                    setup = breakout
                    breakout_signals += 1

                elif pullback is not None:

                    setup = pullback
                    pullback_signals += 1

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

                # =================================================
                # GLOBAL REGIME FILTER
                # =================================================

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

                result = (
                    simulate_trade(
                        df,
                        entry_index,
                        trigger,
                        stop
                    )
                )

                if result is None:

                    i += 1
                    continue

                exit_index = int(
                    result[
                        "Exit_Index"
                    ]
                )

                exit_date = (
                    pd.Timestamp(
                        df.index[
                            exit_index
                        ]
                    )
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

                        "Exit_Date":
                            exit_date,

                        "Trigger":
                            trigger,

                        "Raw_Entry":
                            result[
                                "Raw_Entry"
                            ],

                        "Stop":
                            stop,

                        "Target":
                            result[
                                "Target"
                            ],

                        "Raw_Exit":
                            result[
                                "Raw_Exit"
                            ],

                        "Result":
                            result[
                                "Result"
                            ],

                        "Score":
                            float(
                                setup[
                                    "Score"
                                ]
                            )
                    }
                )

                i = (
                    exit_index + 1
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
            breakout_signals,

        "Pullback_Signals":
            pullback_signals
    }

    return (
        result,
        diagnostics
    )


# ============================================================
# STOCK CLOSE FOR MTM
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

        if date in df.index:

            price = (
                df.loc[
                    date,
                    "Close"
                ]
            )

            if not pd.isna(
                price
            ):

                return float(
                    price
                )

        previous = (
            df.loc[
                df.index <= date,
                "Close"
            ]
            .dropna()
        )

        if previous.empty:
            return None

        return float(
            previous.iloc[-1]
        )

    except Exception:

        return None


# ============================================================
# PORTFOLIO
# ============================================================

def simulate_portfolio(
    candidates,
    all_data,
    qqq,
    backtest_start
):

    if candidates.empty:

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
        in candidates.groupby(
            "Entry_Date"
        )
    }

    final_date = (
        candidates[
            "Exit_Date"
        ].max()
    )

    calendar = (
        qqq.loc[
            (
                qqq.index
                >= backtest_start
            )
            &
            (
                qqq.index
                <= final_date
            )
        ]
        .index
    )

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

    for date in calendar:

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # ENTRIES
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
                    len(open_positions)
                    >= MAX_POSITIONS
                ):

                    skipped_positions += 1
                    continue

                # Don't hold same stock twice.

                already_open = any(
                    p[
                        "Ticker"
                    ]
                    ==
                    trade[
                        "Ticker"
                    ]

                    for p in open_positions
                )

                if already_open:

                    skipped_same_ticker += 1
                    continue

                raw_entry = float(
                    trade[
                        "Raw_Entry"
                    ]
                )

                stop = float(
                    trade[
                        "Stop"
                    ]
                )

                # Slippage on actual entry

                entry_fill = (
                    raw_entry
                    * (
                        1
                        + SLIPPAGE_PCT
                    )
                )

                stop_fill = (
                    stop
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                risk_per_share = (
                    entry_fill
                    - stop_fill
                )

                if risk_per_share <= 0:
                    continue

                shares_by_risk = (
                    risk_budget
                    / risk_per_share
                )

                effective_entry_cost = (
                    entry_fill
                    * (
                        1
                        + COMMISSION_PCT
                    )
                )

                shares_by_cash = (
                    cash
                    / effective_entry_cost
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

                if shares < MIN_SHARE_SIZE:

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

                        "Signal_Date":
                            trade[
                                "Signal_Date"
                            ],

                        "Entry_Date":
                            date,

                        "Exit_Date":
                            trade[
                                "Exit_Date"
                            ],

                        "Result":
                            trade[
                                "Result"
                            ],

                        "Score":
                            trade[
                                "Score"
                            ],

                        "Shares":
                            shares,

                        "Entry_Fill":
                            entry_fill,

                        "Entry_Cost":
                            entry_cost,

                        "Entry_Commission":
                            entry_commission,

                        "Stop":
                            stop,

                        "Target":
                            trade[
                                "Target"
                            ],

                        "Raw_Exit":
                            trade[
                                "Raw_Exit"
                            ],

                        "Risk_Per_Share":
                            risk_per_share
                    }
                )

        # ====================================================
        # EXITS
        # ====================================================

        remaining = []

        for position in (
            open_positions
        ):

            if (
                position[
                    "Exit_Date"
                ]
                != date
            ):

                remaining.append(
                    position
                )

                continue

            raw_exit = float(
                position[
                    "Raw_Exit"
                ]
            )

            exit_fill = (
                raw_exit
                * (
                    1
                    - SLIPPAGE_PCT
                )
            )

            exit_notional = (
                position[
                    "Shares"
                ]
                * exit_fill
            )

            exit_commission = (
                exit_notional
                * COMMISSION_PCT
            )

            exit_proceeds = (
                exit_notional
                - exit_commission
            )

            cash += (
                exit_proceeds
            )

            net_pnl = (
                exit_proceeds
                - position[
                    "Entry_Cost"
                ]
            )

            actual_risk = (
                position[
                    "Shares"
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
                "Exit_Fill"
            ] = exit_fill

            position[
                "Exit_Commission"
            ] = exit_commission

            position[
                "Net_PnL"
            ] = net_pnl

            position[
                "Net_R"
            ] = net_r

            completed.append(
                position
            )

        open_positions = (
            remaining
        )

        # ====================================================
        # MARK TO MARKET
        # ====================================================

        market_value = 0.0

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
                    "Shares"
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

def calculate_performance(
    trades,
    equity
):

    ending = float(
        equity[
            "Equity"
        ].iloc[-1]
    )

    return_pct = (
        ending
        / STARTING_CAPITAL
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
            "Exit_Commission"
        ].sum()
    )

    return {
        "Starting_Capital":
            STARTING_CAPITAL,

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
            commissions
    }


# ============================================================
# BREAKDOWN BY SETUP TYPE
# ============================================================

def setup_breakdown(
    trades
):

    rows = []

    for setup_type, group in (
        trades.groupby(
            "Setup_Type"
        )
    ):

        winners = group.loc[
            group[
                "Net_PnL"
            ] > 0,
            "Net_PnL"
        ]

        losers = group.loc[
            group[
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

        rows.append(
            {
                "Setup_Type":
                    setup_type,

                "Trades":
                    len(group),

                "Net_PnL":
                    group[
                        "Net_PnL"
                    ].sum(),

                "Profitable_%":
                    (
                        group[
                            "Net_PnL"
                        ] > 0
                    ).mean()
                    * 100,

                "Avg_Net_R":
                    group[
                        "Net_R"
                    ].mean(),

                "Profit_Factor":
                    pf
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "Setup_Type"
        )
    )


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
        df.groupby(
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
        "=" * 115
    )

    print(
        "V11 - UNIFIED MOMENTUM STRATEGY"
    )

    print(
        "=" * 115
    )

    print(
        f"Starting capital:      ${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Risk per trade:        {RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Max positions:         {MAX_POSITIONS}"
    )

    print()

    print(
        "MARKET FILTER:"
    )

    print(
        "  QQQ > SMA200"
    )

    print(
        "  DXY < SMA200"
    )

    print()

    print(
        "ENTRY TYPES:"
    )

    print(
        "  A. Momentum Breakout"
    )

    print(
        "  B. Momentum Pullback + StochRSI cross > 20"
    )

    print()

    print(
        f"Downloading {len(TICKERS)} stocks..."
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

    print()

    print(
        f"Backtest: "
        f"{backtest_start.date()} "
        f"-> {end_date.date()}"
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    (
        candidates,
        signal_diag
    ) = generate_candidates(
        all_data,
        qqq,
        dxy,
        backtest_start
    )

    print()

    print(
        f"Final candidates:      {len(candidates)}"
    )

    print(
        f"Raw breakout signals:  "
        f"{signal_diag['Breakout_Signals']}"
    )

    print(
        f"Raw pullback signals:  "
        f"{signal_diag['Pullback_Signals']}"
    )

    print(
        f"Rejected QQQ/DXY:      "
        f"{signal_diag['Rejected_Market']}"
    )

    if candidates.empty:

        print(
            "NO CANDIDATES."
        )

        return

    # ========================================================
    # PORTFOLIO
    # ========================================================

    (
        trades,
        equity,
        portfolio_diag
    ) = simulate_portfolio(
        candidates,
        all_data,
        qqq,
        backtest_start
    )

    if trades.empty:

        print(
            "NO TRADES."
        )

        return

    stats = (
        calculate_performance(
            trades,
            equity
        )
    )

    # ========================================================
    # MAIN RESULTS
    # ========================================================

    print()

    print(
        "=" * 115
    )

    print(
        "V11 - REAL ACCOUNT RESULTS"
    )

    print(
        "=" * 115
    )

    print(
        f"Starting capital:      "
        f"${stats['Starting_Capital']:,.2f}"
    )

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

    print()

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
        f"TRUE MTM Drawdown:     "
        f"{stats['Max_MTM_DD_%']:.2f}%"
    )

    print(
        f"Commissions:           "
        f"${stats['Commissions']:.2f}"
    )

    print()

    print(
        f"Skipped max positions: "
        f"{portfolio_diag['Skipped_Positions']}"
    )

    print(
        f"Skipped cash:          "
        f"{portfolio_diag['Skipped_Cash']}"
    )

    print(
        f"Skipped same ticker:   "
        f"{portfolio_diag['Skipped_Same_Ticker']}"
    )

    # ========================================================
    # SETUP TYPE CONTRIBUTION
    # ========================================================

    breakdown = (
        setup_breakdown(
            trades
        )
    )

    print()

    print(
        "=" * 115
    )

    print(
        "BREAKOUT VS PULLBACK CONTRIBUTION"
    )

    print(
        "=" * 115
    )

    print(
        breakdown
        .round(3)
        .to_string()
    )

    # ========================================================
    # YEARLY
    # ========================================================

    yearly = (
        yearly_performance(
            trades
        )
    )

    print()

    print(
        "=" * 115
    )

    print(
        "YEAR-BY-YEAR"
    )

    print(
        "=" * 115
    )

    print(
        yearly
        .round(3)
        .to_string()
    )

    # ========================================================
    # TARGET
    # ========================================================

    checks = {
        "CAGR >= 10%":
            stats[
                "CAGR_%"
            ] >= 10,

        "PF >= 1.25":
            stats[
                "Profit_Factor"
            ] >= 1.25,

        "Avg R >= 0.10":
            stats[
                "Avg_Net_R"
            ] >= 0.10,

        "Max DD <= 15%":
            stats[
                "Max_MTM_DD_%"
            ] >= -15
    }

    print()

    print(
        "=" * 115
    )

    print(
        "TARGET CHECK"
    )

    print(
        "=" * 115
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<22} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        f"\nScore: "
        f"{sum(checks.values())}/4"
    )

    # ========================================================
    # SAVE
    # ========================================================

    candidates.to_csv(
        "v11_candidates.csv",
        index=False
    )

    trades.to_csv(
        "v11_trades.csv",
        index=False
    )

    equity.to_csv(
        "v11_equity.csv",
        index=False
    )

    yearly.to_csv(
        "v11_yearly.csv"
    )

    breakdown.to_csv(
        "v11_setup_breakdown.csv"
    )

    print()

    print(
        "=" * 115
    )

    print(
        "V11 completed."
    )

    print(
        "=" * 115
    )


if __name__ == "__main__":
    main()
