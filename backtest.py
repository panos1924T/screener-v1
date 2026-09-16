import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V10 STRATEGY TOURNAMENT
#
# NEW:
# 1. QQQ > SMA200
# 2. DXY < SMA200
# 3. Pullback requires Stochastic RSI cross ABOVE 20
# ============================================================

STARTING_CAPITAL = 1000.0
RISK_PER_TRADE = 0.005
MAX_POSITIONS = 3

SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0002

MIN_SHARE_SIZE = 0.0001

DOWNLOAD_PERIOD = "7y"
BACKTEST_YEARS = 5

ATR_PERIOD = 14
MAX_HOLDING_DAYS = 10
ENTRY_VALID_DAYS = 3

STOCH_RSI_PERIOD = 14
STOCH_RSI_CROSS_LEVEL = 20.0


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
#
# 0 - 100 scale
# ============================================================

def calculate_stoch_rsi(
    rsi_series,
    period=14
):

    lowest_rsi = (
        rsi_series
        .rolling(period)
        .min()
    )

    highest_rsi = (
        rsi_series
        .rolling(period)
        .max()
    )

    denominator = (
        highest_rsi
        - lowest_rsi
    )

    stoch_rsi = (
        (
            rsi_series
            - lowest_rsi
        )
        / denominator.replace(
            0,
            np.nan
        )
    ) * 100

    return stoch_rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = (
        df["Close"].shift(1)
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
# PREPARE STOCK DATA
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

    # ========================================================
    # NEW: STOCHASTIC RSI
    # ========================================================

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
# GENERIC MARKET INDEX PREPARATION
# ============================================================

def prepare_market_ticker(
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


# ============================================================
# MARKET VALUE ON DATE
# ============================================================

def get_market_row(
    df,
    date
):

    data = df[
        df.index <= date
    ]

    if data.empty:
        return None

    row = data.iloc[-1]

    if pd.isna(
        row["SMA200"]
    ):

        return None

    return row


# ============================================================
# GLOBAL LONG MARKET FILTER
#
# QQQ > SMA200
# AND
# DXY < SMA200
# ============================================================

def market_allows_long(
    qqq,
    dxy,
    date
):

    qqq_row = (
        get_market_row(
            qqq,
            date
        )
    )

    dxy_row = (
        get_market_row(
            dxy,
            date
        )
    )

    if (
        qqq_row is None
        or dxy_row is None
    ):

        return False

    qqq_close = float(
        qqq_row["Close"]
    )

    qqq_sma200 = float(
        qqq_row["SMA200"]
    )

    dxy_close = float(
        dxy_row["Close"]
    )

    dxy_sma200 = float(
        dxy_row["SMA200"]
    )

    qqq_bull = (
        qqq_close
        > qqq_sma200
    )

    dxy_weak = (
        dxy_close
        < dxy_sma200
    )

    return (
        qqq_bull
        and dxy_weak
    )


# ============================================================
# SWING LOW
# ============================================================

def find_swing_low(
    df,
    signal_index,
    lookback=3
):

    latest_candidate = (
        signal_index
        - lookback
        - 1
    )

    if latest_candidate < lookback:
        return None

    for i in range(
        latest_candidate,
        lookback - 1,
        -1
    ):

        current = float(
            df["Low"].iloc[i]
        )

        left = (
            df["Low"]
            .iloc[
                i - lookback:i
            ]
        )

        right = (
            df["Low"]
            .iloc[
                i + 1:
                i + lookback + 1
            ]
        )

        if (
            current
            < float(left.min())
            and
            current
            < float(right.min())
        ):

            return current

    return None


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
# TRADE EXIT
# ============================================================

def simulate_trade(
    df,
    entry_index,
    trigger,
    stop
):

    planned_risk = (
        trigger - stop
    )

    if planned_risk <= 0:
        return None

    target = (
        trigger
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
            df["High"].iloc[i]
        )

        low = float(
            df["Low"].iloc[i]
        )

        # Conservative same-bar rule
        if low <= stop:

            return {
                "Exit_Index": i,
                "Raw_Exit": stop,
                "Result": "SL",
                "Target": target
            }

        if high >= target:

            return {
                "Exit_Index": i,
                "Raw_Exit": target,
                "Result": "TP",
                "Target": target
            }

    return {
        "Exit_Index": end,

        "Raw_Exit": float(
            df["Close"].iloc[end]
        ),

        "Result": "TIME",

        "Target": target
    }


# ============================================================
# STRATEGY 1
# PULLBACK V10
#
# NEW:
# STOCH RSI must CROSS ABOVE 20
# ============================================================

def pullback_signal(
    df,
    i
):

    row = df.iloc[i]

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

    sma200 = float(
        row["SMA200"]
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

    rsi = float(
        row["RSI14"]
    )

    stoch_rsi = float(
        row["STOCH_RSI"]
    )

    previous_stoch_rsi = float(
        df["STOCH_RSI"].iloc[
            i - 1
        ]
    )

    old_sma50 = float(
        df["SMA50"].iloc[
            i - 10
        ]
    )

    # ========================================================
    # EXISTING TREND FILTERS
    # ========================================================

    if close <= sma200:
        return None

    if ema20 <= sma50:
        return None

    if sma50 <= old_sma50:
        return None

    if not (
        35 <= rsi <= 55
    ):
        return None

    # ========================================================
    # NEW STOCH RSI FILTER
    #
    # Yesterday <= 20
    # Today > 20
    # ========================================================

    if (
        pd.isna(stoch_rsi)
        or
        pd.isna(previous_stoch_rsi)
    ):
        return None

    stoch_cross_above_20 = (
        previous_stoch_rsi
        <= STOCH_RSI_CROSS_LEVEL
        and
        stoch_rsi
        > STOCH_RSI_CROSS_LEVEL
    )

    if not stoch_cross_above_20:
        return None

    # ========================================================
    # PULLBACK TO EMA20 / SMA50
    # ========================================================

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

    if not (
        touched_ema
        or touched_sma
    ):
        return None

    if not (
        close >= ema20
        or close >= sma50
    ):
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

    if close_location <= 0.50:
        return None

    swing_low = (
        find_swing_low(
            df,
            i,
            3
        )
    )

    if swing_low is None:
        return None

    trigger = (
        high
        + 0.10 * atr
    )

    stop = (
        swing_low
        - 0.10 * atr
    )

    risk = (
        trigger - stop
    )

    if risk < (
        0.50 * atr
    ):
        return None

    slope_score = (
        sma50
        - old_sma50
    ) / atr

    ema_score = (
        ema20
        - sma50
    ) / atr

    # Stoch RSI isn't added to ranking yet.
    # It is purely a FILTER.

    score = (
        slope_score
        + ema_score
        + close_location
    )

    return {
        "Trigger": trigger,
        "Stop": stop,
        "Score": score
    }


# ============================================================
# STRATEGY 2
# MOMENTUM BREAKOUT
# ============================================================

def breakout_signal(
    df,
    i
):

    row = df.iloc[i]

    close = float(row["Close"])
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])

    sma200 = float(row["SMA200"])
    sma50 = float(row["SMA50"])
    ema20 = float(row["EMA20"])

    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])

    volume = float(row["Volume"])
    avg_volume = float(row["AVG_VOL20"])

    high20 = float(row["HIGH20"])
    low10 = float(row["LOW10"])

    old_sma50 = float(
        df["SMA50"].iloc[
            i - 10
        ]
    )

    if close <= sma200:
        return None

    if sma50 <= sma200:
        return None

    if ema20 <= sma50:
        return None

    if sma50 <= old_sma50:
        return None

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

    slope_strength = (
        sma50
        - old_sma50
    ) / atr

    momentum_strength = (
        close
        - sma50
    ) / atr

    score = (
        slope_strength
        + momentum_strength
        + relative_volume
        + close_location
    )

    return {
        "Trigger": trigger,
        "Stop": stop,
        "Score": score
    }


# ============================================================
# STRATEGY 3
# TREND CONTINUATION
# ============================================================

def continuation_signal(
    df,
    i
):

    row = df.iloc[i]

    close = float(row["Close"])
    open_price = float(row["Open"])
    high = float(row["High"])
    low = float(row["Low"])

    sma200 = float(row["SMA200"])
    sma50 = float(row["SMA50"])
    ema20 = float(row["EMA20"])

    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])

    volume = float(row["Volume"])
    avg_volume = float(row["AVG_VOL20"])

    low5 = float(row["LOW5"])

    old_sma50 = float(
        df["SMA50"].iloc[
            i - 10
        ]
    )

    if close <= sma200:
        return None

    if ema20 <= sma50:
        return None

    if sma50 <= old_sma50:
        return None

    if not (
        48 <= rsi <= 65
    ):
        return None

    distance_from_ema = (
        abs(
            low - ema20
        )
        / atr
    )

    if distance_from_ema > 0.50:
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

    if close_location < 0.65:
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

    slope_strength = (
        sma50
        - old_sma50
    ) / atr

    ema_strength = (
        ema20
        - sma50
    ) / atr

    score = (
        slope_strength
        + ema_strength
        + close_location
        + relative_volume
    )

    return {
        "Trigger": trigger,
        "Stop": stop,
        "Score": score
    }


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    strategy_name,
    strategy_function,
    all_data,
    qqq,
    dxy,
    backtest_start
):

    candidates = []

    rejected_market_filter = 0

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

                setup = (
                    strategy_function(
                        df,
                        i
                    )
                )

                if setup is None:

                    i += 1
                    continue

                trigger = float(
                    setup["Trigger"]
                )

                stop = float(
                    setup["Stop"]
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
                # NEW GLOBAL FILTER
                #
                # QQQ > SMA200
                # DXY < SMA200
                # =================================================

                if not market_allows_long(
                    qqq,
                    dxy,
                    entry_date
                ):

                    rejected_market_filter += 1

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

                entry_open = float(
                    df["Open"].iloc[
                        entry_index
                    ]
                )

                exit_open = float(
                    df["Open"].iloc[
                        exit_index
                    ]
                )

                candidates.append(
                    {
                        "Strategy":
                            strategy_name,

                        "Ticker":
                            ticker,

                        "Signal_Date":
                            signal_date,

                        "Entry_Date":
                            entry_date,

                        "Exit_Date":
                            exit_date,

                        "Trigger":
                            trigger,

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

                        "Entry_Open":
                            entry_open,

                        "Exit_Open":
                            exit_open,

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
                f"{strategy_name} | "
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

    return (
        result,
        rejected_market_filter
    )


# ============================================================
# CLOSE FOR MARK-TO-MARKET
# ============================================================

def get_close(
    all_data,
    ticker,
    date
):

    try:

        df = (
            all_data[ticker]
        )

        if date in df.index:

            value = (
                df.loc[
                    date,
                    "Close"
                ]
            )

            if not pd.isna(
                value
            ):

                return float(
                    value
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

    cash = STARTING_CAPITAL
    last_equity = STARTING_CAPITAL

    open_positions = []
    completed = []
    equity_rows = []

    skipped_positions = 0
    skipped_cash = 0

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

                trigger = float(
                    trade[
                        "Trigger"
                    ]
                )

                stop = float(
                    trade[
                        "Stop"
                    ]
                )

                entry_open = float(
                    trade[
                        "Entry_Open"
                    ]
                )

                # Gap-aware entry

                raw_entry_fill = max(
                    trigger,
                    entry_open
                )

                entry_fill = (
                    raw_entry_fill
                    * (
                        1
                        + SLIPPAGE_PCT
                    )
                )

                stop_fill_reference = (
                    stop
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

                effective_cost = (
                    entry_fill
                    * (
                        1
                        + COMMISSION_PCT
                    )
                )

                shares_by_cash = (
                    cash
                    / effective_cost
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

                cash -= entry_cost

                open_positions.append(
                    {
                        "Strategy":
                            trade[
                                "Strategy"
                            ],

                        "Ticker":
                            trade[
                                "Ticker"
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

                        "Exit_Open":
                            trade[
                                "Exit_Open"
                            ],

                        "Risk_Per_Share":
                            risk_per_share,

                        "Risk_Budget":
                            risk_budget
                    }
                )

        # ====================================================
        # EXITS
        # ====================================================

        still_open = []

        for position in (
            open_positions
        ):

            if (
                position[
                    "Exit_Date"
                ]
                != date
            ):

                still_open.append(
                    position
                )

                continue

            result = (
                position[
                    "Result"
                ]
            )

            raw_exit = float(
                position[
                    "Raw_Exit"
                ]
            )

            exit_open = float(
                position[
                    "Exit_Open"
                ]
            )

            if result == "SL":

                raw_fill = min(
                    raw_exit,
                    exit_open
                )

            elif result == "TP":

                raw_fill = raw_exit

            else:

                raw_fill = raw_exit

            exit_fill = (
                raw_fill
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

            cash += exit_proceeds

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
            still_open
        )

        # ====================================================
        # MTM
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

        last_equity = equity

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
                skipped_cash
        }
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(
    strategy,
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

    total_return = (
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

    if (
        years > 0
        and ending > 0
    ):

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

    else:

        cagr = 0

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

    positive = (
        trades.loc[
            trades[
                "Net_PnL"
            ] > 0,
            "Net_PnL"
        ]
    )

    negative = (
        trades.loc[
            trades[
                "Net_PnL"
            ] < 0,
            "Net_PnL"
        ]
    )

    gross_profit = (
        positive.sum()
    )

    gross_loss = abs(
        negative.sum()
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
        "Strategy":
            strategy,

        "Ending_Capital":
            ending,

        "Net_Profit":
            ending
            - STARTING_CAPITAL,

        "Return_%":
            total_return,

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
            ]
    }


# ============================================================
# YEARLY
# ============================================================

def yearly_performance(
    trades
):

    if trades.empty:

        return pd.DataFrame()

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
        "=" * 120
    )

    print(
        "V10 - DXY FILTER + STOCH RSI PULLBACK"
    )

    print(
        "=" * 120
    )

    print(
        f"Capital:              ${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Risk/trade:           {RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Max positions:        {MAX_POSITIONS}"
    )

    print()

    print(
        "GLOBAL LONG FILTER:"
    )

    print(
        "  QQQ > SMA200"
    )

    print(
        "  DXY < SMA200"
    )

    print()

    print(
        "PULLBACK EXTRA FILTER:"
    )

    print(
        "  Stoch RSI crosses ABOVE 20"
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

    qqq = prepare_market_ticker(
        "QQQ",
        "QQQ"
    )

    # Yahoo Finance US Dollar Index
    dxy = prepare_market_ticker(
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

    strategies = {
        "Pullback + StochRSI":
            pullback_signal,

        "Momentum Breakout":
            breakout_signal,

        "Trend Continuation":
            continuation_signal
    }

    summaries = []

    for (
        strategy_name,
        function
    ) in strategies.items():

        print()

        print(
            "=" * 120
        )

        print(
            strategy_name.upper()
        )

        print(
            "=" * 120
        )

        (
            candidates,
            rejected_market
        ) = generate_candidates(
            strategy_name,
            function,
            all_data,
            qqq,
            dxy,
            backtest_start
        )

        print(
            f"Candidates after filters: "
            f"{len(candidates)}"
        )

        print(
            f"Rejected by QQQ/DXY:     "
            f"{rejected_market}"
        )

        (
            trades,
            equity,
            diagnostics
        ) = simulate_portfolio(
            candidates,
            all_data,
            qqq,
            backtest_start
        )

        if (
            trades.empty
            or equity.empty
        ):

            print(
                "NO EXECUTED TRADES"
            )

            continue

        stats = performance(
            strategy_name,
            trades,
            equity,
            diagnostics
        )

        summaries.append(
            stats
        )

        yearly = (
            yearly_performance(
                trades
            )
        )

        print()

        print(
            f"Ending capital:      "
            f"${stats['Ending_Capital']:,.2f}"
        )

        print(
            f"Net profit:          "
            f"${stats['Net_Profit']:,.2f}"
        )

        print(
            f"Return:              "
            f"{stats['Return_%']:.2f}%"
        )

        print(
            f"CAGR:                "
            f"{stats['CAGR_%']:.2f}%"
        )

        print(
            f"Trades:              "
            f"{stats['Trades']}"
        )

        print(
            f"Profitable:          "
            f"{stats['Profitable_%']:.2f}%"
        )

        print(
            f"Avg Net R:           "
            f"{stats['Avg_Net_R']:.3f}R"
        )

        print(
            f"Profit Factor:       "
            f"{stats['Profit_Factor']:.3f}"
        )

        print(
            f"Max MTM Drawdown:    "
            f"{stats['Max_MTM_DD_%']:.2f}%"
        )

        print(
            f"Commissions:         "
            f"${stats['Commissions']:.2f}"
        )

        print(
            f"Skipped positions:   "
            f"{stats['Skipped_Positions']}"
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
    # FINAL
    # ========================================================

    summary = pd.DataFrame(
        summaries
    )

    if summary.empty:

        print(
            "No strategy produced trades."
        )

        return

    summary = (
        summary
        .set_index(
            "Strategy"
        )
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V10 FINAL COMPARISON"
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
        "Skipped_Positions"
    ]

    print(
        summary[
            columns
        ]
        .round(3)
        .to_string()
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

    for strategy, row in (
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

        passed = sum(
            checks.values()
        )

        print()

        print(
            strategy
        )

        for name, result in (
            checks.items()
        ):

            print(
                f"  {name:<20} "
                f"{'PASS' if result else 'FAIL'}"
            )

        print(
            f"  Score: {passed}/4"
        )

    summary.to_csv(
        "backtest_v10_comparison.csv"
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V10 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
