import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V21 - RSI CONCEPT TEST
#
# COMPARE:
#   A) RSI 45-65 ON
#   B) RSI FILTER OFF
#
# LOCKED FROM V20:
#   STOCH RSI = OFF
#
# EVERYTHING ELSE IDENTICAL
# ============================================================


# ============================================================
# ACCOUNT
# ============================================================

STARTING_CAPITAL = 1000.0
RISK_PER_TRADE = 0.0075
MAX_POSITIONS = 5

SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0002
MIN_SHARE_SIZE = 0.0001


# ============================================================
# STRATEGY
# ============================================================

DOWNLOAD_PERIOD = "max"
BACKTEST_YEARS = 15

ATR_PERIOD = 14

ENTRY_VALID_DAYS = 3
FIXED_HOLD_DAYS = 10
TRAIL_HOLD_DAYS = 30

ATR_TRAIL_MULT = 2.5

RSI_MIN = 45.0
RSI_MAX = 65.0


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


def calculate_atr(
    df,
    period=14
):

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
        and
        dxy_ok
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
# QUALITY SCORE
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
# PULLBACK SETUP
#
# STOCH RSI IS REMOVED / OFF
#
# ONLY VARIABLE:
# use_rsi = True / False
# ============================================================

def pullback_setup(
    df,
    i,
    use_rsi
):

    if not strong_trend(
        df,
        i
    ):

        return None

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

    ema20 = float(
        row["EMA20"]
    )

    atr = float(
        row["ATR14"]
    )

    rsi = float(
        row["RSI14"]
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
    #
    # ONLY VARIABLE IN V21
    # ========================================================

    if use_rsi:

        if not (
            RSI_MIN
            <= rsi
            <= RSI_MAX
        ):

            return None


    # ========================================================
    # EMA20 PULLBACK
    # ========================================================

    ema_distance = (
        abs(
            low
            - ema20
        )
        / atr
    )

    if ema_distance > 0.50:

        return None


    # ========================================================
    # CLOSE ABOVE EMA20
    # ========================================================

    if close <= ema20:

        return None


    # ========================================================
    # BULLISH CANDLE
    # ========================================================

    if close <= open_price:

        return None


    # ========================================================
    # CLOSE LOCATION
    # ========================================================

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


    # ========================================================
    # RELATIVE VOLUME
    # ========================================================

    if avg_volume <= 0:

        return None

    relative_volume = (
        volume
        / avg_volume
    )

    if relative_volume < 0.80:

        return None


    # ========================================================
    # ENTRY / STOP
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
# BASELINE LIFECYCLE
# ============================================================

def simulate_baseline_lifecycle(
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
                    i
            }

        if low <= stop:

            return {
                "Raw_Entry":
                    raw_entry,

                "Exit_Index":
                    i
            }

        if high >= target:

            return {
                "Raw_Entry":
                    raw_entry,

                "Exit_Index":
                    i
            }

    return {
        "Raw_Entry":
            raw_entry,

        "Exit_Index":
            end
    }


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    all_data,
    qqq,
    dxy,
    backtest_start,
    use_rsi
):

    candidates = []

    raw_signals = 0

    rejected_market = 0

    for ticker in TICKERS:

        try:

            if (
                ticker
                not in
                all_data.columns
                .get_level_values(0)
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
                    "AVG_VOL20",
                    "LOW5"
                ]

                if any(
                    pd.isna(
                        df[col]
                        .iloc[i]
                    )
                    for col in required
                ):

                    i += 1

                    continue

                setup = pullback_setup(
                    df,
                    i,
                    use_rsi
                )

                if setup is None:

                    i += 1

                    continue

                raw_signals += 1

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

                entry_index = find_entry(
                    df,
                    i,
                    trigger
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
                        entry_index
                        + 1
                    )

                    continue

                baseline = (
                    simulate_baseline_lifecycle(
                        df,
                        entry_index,
                        trigger,
                        stop
                    )
                )

                if baseline is None:

                    i += 1

                    continue

                candidates.append(
                    {
                        "Ticker":
                            ticker,

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

                        "RSI14":
                            rsi_if_available(
                                df,
                                i
                            ),

                        "Score":
                            float(
                                setup[
                                    "Score"
                                ]
                            )
                    }
                )

                i = (
                    int(
                        baseline[
                            "Exit_Index"
                        ]
                    )
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

    return (
        result,
        {
            "Raw_Signals":
                raw_signals,

            "Rejected_Market":
                rejected_market
        }
    )


def rsi_if_available(
    df,
    i
):

    try:

        return float(
            df[
                "RSI14"
            ].iloc[i]
        )

    except Exception:

        return np.nan


# ============================================================
# 2.5 ATR TRAIL
# ============================================================

def create_exit_plan(
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

    end = min(
        entry_index
        + TRAIL_HOLD_DAYS
        - 1,
        len(df) - 1
    )

    trail = initial_stop

    highest_close = raw_entry

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

        calculated_trail = (
            highest_close
            - ATR_TRAIL_MULT
            * atr
        )

        trail = max(
            trail,
            calculated_trail
        )

        if open_price < trail:

            return {
                "Exit_Index":
                    i,

                "Raw_Exit":
                    open_price,

                "Exit_Reason":
                    "ATR_GAP"
            }

        if low <= trail:

            return {
                "Exit_Index":
                    i,

                "Raw_Exit":
                    trail,

                "Exit_Reason":
                    "ATR_TRAIL"
            }

        highest_close = max(
            highest_close,
            close
        )

    return {
        "Exit_Index":
            end,

        "Raw_Exit":
            float(
                df[
                    "Close"
                ].iloc[
                    end
                ]
            ),

        "Exit_Reason":
            "TIME30"
    }


# ============================================================
# CREATE PLANS
# ============================================================

def create_trade_plans(
    candidates,
    prepared_data
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

        if ticker not in prepared_data:

            continue

        df = (
            prepared_data[
                ticker
            ]
        )

        result = (
            create_exit_plan(
                df,
                trade
            )
        )

        exit_index = int(
            result[
                "Exit_Index"
            ]
        )

        plans.append(
            {
                **trade.to_dict(),

                "Exit_Date":
                    pd.Timestamp(
                        df.index[
                            exit_index
                        ]
                    ),

                "Raw_Exit":
                    float(
                        result[
                            "Raw_Exit"
                        ]
                    ),

                "Exit_Reason":
                    result[
                        "Exit_Reason"
                    ]
            }
        )

    return pd.DataFrame(
        plans
    )


# ============================================================
# GET CLOSE
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

    skipped_same_ticker = 0


    for date in calendar:

        date = pd.Timestamp(
            date
        )


        # ====================================================
        # ENTRIES
        # ====================================================

        if date in grouped:

            todays = grouped[
                date
            ]

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

                if any(
                    p[
                        "Ticker"
                    ]
                    ==
                    trade[
                        "Ticker"
                    ]

                    for p in open_positions
                ):

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

                stop_reference = (
                    initial_stop
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                risk_per_share = (
                    entry_fill
                    - stop_reference
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

                        "Exit_Reason":
                            trade[
                                "Exit_Reason"
                            ],

                        "Score":
                            trade[
                                "Score"
                            ],

                        "RSI14":
                            trade[
                                "RSI14"
                            ],

                        "Shares":
                            shares,

                        "Entry_Fill":
                            entry_fill,

                        "Entry_Cost":
                            entry_cost,

                        "Entry_Commission":
                            entry_commission,

                        "Risk_Per_Share":
                            risk_per_share,

                        "Raw_Exit":
                            trade[
                                "Raw_Exit"
                            ]
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

            exit_fill = (
                float(
                    position[
                        "Raw_Exit"
                    ]
                )
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

            holding_days = (
                date
                -
                position[
                    "Entry_Date"
                ]
            ).days

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

            position[
                "Holding_Days"
            ] = holding_days

            completed.append(
                position
            )

        open_positions = (
            remaining
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

        exposure_pct = (
            market_value
            / equity
            * 100
            if equity > 0
            else 0
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
                    ),

                "Active":
                    1
                    if len(
                        open_positions
                    ) > 0
                    else 0,

                "Exposure_%":
                    exposure_pct
            }
        )

        last_equity = (
            equity
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
# STATS
# ============================================================

def calculate_stats(
    trades,
    equity,
    diagnostics
):

    ending = float(
        equity[
            "Equity"
        ].iloc[-1]
    )

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

    total_return = (
        (
            ending
            / STARTING_CAPITAL
        )
        - 1
    ) * 100

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

    total_days = len(
        equity
    )

    active_days = int(
        equity[
            "Active"
        ].sum()
    )

    return {
        "Ending_Capital":
            ending,

        "Return_%":
            total_return,

        "CAGR_%":
            cagr,

        "Trades":
            len(
                trades
            ),

        "Trades_Per_Year":
            len(
                trades
            )
            / years,

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

        "Max_DD_%":
            float(
                drawdown.min()
            ),

        "Active_Days_%":
            (
                active_days
                / total_days
                * 100
            ),

        "Avg_Positions":
            equity[
                "Open_Positions"
            ].mean(),

        "Avg_Exposure_%":
            equity[
                "Exposure_%"
            ].mean(),

        "Avg_Holding_Days":
            trades[
                "Holding_Days"
            ].mean(),

        "Skipped_Positions":
            diagnostics[
                "Skipped_Positions"
            ],

        "Skipped_Cash":
            diagnostics[
                "Skipped_Cash"
            ]
    }


# ============================================================
# RSI BUCKET ANALYSIS
#
# Extra diagnostic for RSI OFF trades.
# Does NOT alter strategy.
# ============================================================

def rsi_bucket_analysis(
    trades
):

    if trades.empty:

        return pd.DataFrame()

    df = trades.copy()

    bins = [
        0,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        100
    ]

    labels = [
        "<40",
        "40-45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70+"
    ]

    df[
        "RSI_Bucket"
    ] = pd.cut(
        df[
            "RSI14"
        ],
        bins=bins,
        labels=labels,
        right=False
    )

    rows = []

    for bucket, group in (
        df.groupby(
            "RSI_Bucket",
            observed=False
        )
    ):

        if group.empty:

            continue

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
                "RSI_Bucket":
                    str(
                        bucket
                    ),

                "Trades":
                    len(
                        group
                    ),

                "Avg_R":
                    group[
                        "Net_R"
                    ].mean(),

                "Profit_Factor":
                    pf,

                "Net_PnL":
                    group[
                        "Net_PnL"
                    ].sum(),

                "Win_%":
                    (
                        group[
                            "Net_PnL"
                        ] > 0
                    ).mean()
                    * 100
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "RSI_Bucket"
        )
    )


# ============================================================
# RUN VARIANT
# ============================================================

def run_variant(
    name,
    use_rsi,
    all_data,
    prepared_data,
    qqq,
    dxy,
    backtest_start
):

    print()

    print(
        "=" * 120
    )

    print(
        f"RUNNING: {name}"
    )

    print(
        "=" * 120
    )

    (
        candidates,
        candidate_diag
    ) = generate_candidates(
        all_data,
        qqq,
        dxy,
        backtest_start,
        use_rsi
    )

    print(
        f"Candidates:            "
        f"{len(candidates)}"
    )

    print(
        f"Raw signals:           "
        f"{candidate_diag['Raw_Signals']}"
    )

    print(
        f"Rejected market:       "
        f"{candidate_diag['Rejected_Market']}"
    )

    plans = create_trade_plans(
        candidates,
        prepared_data
    )

    (
        trades,
        equity,
        diagnostics
    ) = simulate_portfolio(
        plans,
        all_data,
        qqq,
        backtest_start
    )

    stats = calculate_stats(
        trades,
        equity,
        diagnostics
    )

    print()

    print(
        f"Ending capital:        "
        f"${stats['Ending_Capital']:,.2f}"
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
        f"Trades/year:           "
        f"{stats['Trades_Per_Year']:.2f}"
    )

    print(
        f"Avg Net R:             "
        f"{stats['Avg_Net_R']:.3f}R"
    )

    print(
        f"Profit Factor:         "
        f"{stats['Profit_Factor']:.3f}"
    )

    print(
        f"Max DD:                "
        f"{stats['Max_DD_%']:.2f}%"
    )

    print(
        f"Active days:           "
        f"{stats['Active_Days_%']:.2f}%"
    )

    print(
        f"Average positions:     "
        f"{stats['Avg_Positions']:.2f}"
    )

    print(
        f"Average exposure:      "
        f"{stats['Avg_Exposure_%']:.2f}%"
    )

    print(
        f"Skipped positions:     "
        f"{stats['Skipped_Positions']}"
    )

    return (
        stats,
        trades,
        equity
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "V21 - RSI 45-65 ON VS OFF"
    )

    print(
        "=" * 120
    )

    print(
        "STOCH RSI is OFF in BOTH variants."
    )

    print()

    print(
        "Only tested concept:"
    )

    print(
        "  RSI 45-65 ON vs RSI filter OFF"
    )

    print()

    print(
        "Market regime:"
    )

    print(
        "  QQQ > SMA200"
    )

    print(
        "  DXY < SMA200"
    )

    print()

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
    # PREPARED DATA
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
    # RUN A: RSI ON
    # ========================================================

    (
        stats_on,
        trades_on,
        equity_on
    ) = run_variant(
        "A - RSI 45-65 ON",
        True,
        all_data,
        prepared_data,
        qqq,
        dxy,
        backtest_start
    )


    # ========================================================
    # RUN B: RSI OFF
    # ========================================================

    (
        stats_off,
        trades_off,
        equity_off
    ) = run_variant(
        "B - RSI OFF",
        False,
        all_data,
        prepared_data,
        qqq,
        dxy,
        backtest_start
    )


    # ========================================================
    # DIRECT COMPARISON
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "V21 DIRECT COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Metric':<25}"
        f"{'RSI ON':>15}"
        f"{'RSI OFF':>15}"
    )

    print(
        "-" * 55
    )

    comparison = [
        (
            "CAGR %",
            stats_on[
                "CAGR_%"
            ],
            stats_off[
                "CAGR_%"
            ]
        ),

        (
            "Trades",
            stats_on[
                "Trades"
            ],
            stats_off[
                "Trades"
            ]
        ),

        (
            "Trades/year",
            stats_on[
                "Trades_Per_Year"
            ],
            stats_off[
                "Trades_Per_Year"
            ]
        ),

        (
            "Avg Net R",
            stats_on[
                "Avg_Net_R"
            ],
            stats_off[
                "Avg_Net_R"
            ]
        ),

        (
            "Profit Factor",
            stats_on[
                "Profit_Factor"
            ],
            stats_off[
                "Profit_Factor"
            ]
        ),

        (
            "Max DD %",
            stats_on[
                "Max_DD_%"
            ],
            stats_off[
                "Max_DD_%"
            ]
        ),

        (
            "Active days %",
            stats_on[
                "Active_Days_%"
            ],
            stats_off[
                "Active_Days_%"
            ]
        ),

        (
            "Avg positions",
            stats_on[
                "Avg_Positions"
            ],
            stats_off[
                "Avg_Positions"
            ]
        ),

        (
            "Exposure %",
            stats_on[
                "Avg_Exposure_%"
            ],
            stats_off[
                "Avg_Exposure_%"
            ]
        ),

        (
            "Skipped positions",
            stats_on[
                "Skipped_Positions"
            ],
            stats_off[
                "Skipped_Positions"
            ]
        )
    ]

    for (
        metric,
        on,
        off
    ) in comparison:

        print(
            f"{metric:<25}"
            f"{on:>15.3f}"
            f"{off:>15.3f}"
        )


    # ========================================================
    # CHANGE
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "CHANGE WHEN RSI FILTER IS REMOVED"
    )

    print(
        "=" * 120
    )

    print(
        f"CAGR change:           "
        f"{stats_off['CAGR_%'] - stats_on['CAGR_%']:+.2f} pp"
    )

    print(
        f"Trade change:          "
        f"{stats_off['Trades'] - stats_on['Trades']:+d}"
    )

    print(
        f"Avg R change:          "
        f"{stats_off['Avg_Net_R'] - stats_on['Avg_Net_R']:+.3f}R"
    )

    print(
        f"PF change:             "
        f"{stats_off['Profit_Factor'] - stats_on['Profit_Factor']:+.3f}"
    )

    print(
        f"DD change:             "
        f"{stats_off['Max_DD_%'] - stats_on['Max_DD_%']:+.2f} pp"
    )

    print(
        f"Active days change:    "
        f"{stats_off['Active_Days_%'] - stats_on['Active_Days_%']:+.2f} pp"
    )

    print(
        f"Exposure change:       "
        f"{stats_off['Avg_Exposure_%'] - stats_on['Avg_Exposure_%']:+.2f} pp"
    )


    # ========================================================
    # RSI BUCKET ANALYSIS
    # ========================================================

    bucket_stats = (
        rsi_bucket_analysis(
            trades_off
        )
    )

    print()

    print(
        "=" * 120
    )

    print(
        "RSI BUCKET PERFORMANCE - RSI OFF VERSION"
    )

    print(
        "=" * 120
    )

    print(
        bucket_stats
        .round(3)
        .to_string()
    )


    # ========================================================
    # SAVE
    # ========================================================

    trades_on.to_csv(
        "v21_rsi_on_trades.csv",
        index=False
    )

    trades_off.to_csv(
        "v21_rsi_off_trades.csv",
        index=False
    )

    equity_on.to_csv(
        "v21_rsi_on_equity.csv",
        index=False
    )

    equity_off.to_csv(
        "v21_rsi_off_equity.csv",
        index=False
    )

    bucket_stats.to_csv(
        "v21_rsi_buckets.csv"
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V21 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
