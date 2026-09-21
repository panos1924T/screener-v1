import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V17 - TRADING DAYS / CAPITAL UTILIZATION TEST
#
# EXACT V15 STRATEGY
#
# PURPOSE:
# Test ONE concept only:
#
# Is CAGR low because the strategy spends too few trading
# days / too little capital in the market?
#
# NO STRATEGY RULE CHANGES.
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
# HISTORY / STRATEGY
# ============================================================

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
# MARKET DATA
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


def get_market_row(df, date):

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
        or dxy_row is None
    ):

        return False

    return (
        float(
            qqq_row["Close"]
        )
        >
        float(
            qqq_row["SMA200"]
        )
        and
        float(
            dxy_row["Close"]
        )
        <
        float(
            dxy_row["SMA200"]
        )
    )


# ============================================================
# TREND
# ============================================================

def strong_trend(df, i):

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

def quality_score(df, i):

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
# PULLBACK
# ============================================================

def pullback_setup(df, i):

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
# BASELINE CANDIDATE LIFECYCLE
# ============================================================

def simulate_baseline_lifecycle(
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
        raw_entry - stop
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
# CANDIDATES
# ============================================================

def generate_candidates(
    all_data,
    qqq,
    dxy,
    backtest_start
):

    candidates = []

    raw_signals = 0
    rejected_market = 0

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

                signal_date = pd.Timestamp(
                    df.index[i]
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

                setup = pullback_setup(
                    df,
                    i
                )

                if setup is None:

                    i += 1

                    continue

                raw_signals += 1

                trigger = float(
                    setup["Trigger"]
                )

                stop = float(
                    setup["Stop"]
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

                entry_date = pd.Timestamp(
                    df.index[
                        entry_index
                    ]
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
            result.sort_values(
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


# ============================================================
# 2.5 ATR EXIT
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
                df["Close"].iloc[
                    end
                ]
            ),

        "Exit_Reason":
            "TIME30"
    }


# ============================================================
# TRADE PLANS
# ============================================================

def create_trade_plans(
    candidates,
    prepared_data
):

    plans = []

    for _, trade in (
        candidates.iterrows()
    ):

        ticker = trade[
            "Ticker"
        ]

        if ticker not in prepared_data:

            continue

        df = prepared_data[
            ticker
        ]

        result = create_exit_plan(
            df,
            trade
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
# PRICE
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
                qqq.index >= backtest_start
            )
            &
            (
                qqq.index <= final_date
            )
        ].index
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
                    p["Ticker"]
                    ==
                    trade["Ticker"]

                    for p
                    in open_positions
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

                entry_cost_per_share = (
                    entry_fill
                    * (
                        1
                        + COMMISSION_PCT
                    )
                )

                shares_by_cash = (
                    cash
                    / entry_cost_per_share
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

        open_positions = remaining


        # ====================================================
        # MTM + EXPOSURE
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

                price = position[
                    "Entry_Fill"
                ]

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

        if equity > 0:

            capital_exposure = (
                market_value
                / equity
            ) * 100

            cash_pct = (
                cash
                / equity
            ) * 100

        else:

            capital_exposure = 0

            cash_pct = 0

        open_count = len(
            open_positions
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
                    open_count,

                "Active":
                    1
                    if open_count > 0
                    else 0,

                "Capital_Exposure_%":
                    capital_exposure,

                "Cash_%":
                    cash_pct
            }
        )

        last_equity = equity


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
    equity
):

    ending = float(
        equity["Equity"].iloc[-1]
    )

    first_date = pd.Timestamp(
        equity["Date"].iloc[0]
    )

    last_date = pd.Timestamp(
        equity["Date"].iloc[-1]
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

    curve = equity[
        "Equity"
    ]

    peak = curve.cummax()

    dd = (
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

    profit_factor = (
        winners.sum()
        /
        abs(
            losers.sum()
        )
        if abs(
            losers.sum()
        ) > 0
        else np.inf
    )

    return {
        "Ending":
            ending,

        "Return":
            total_return,

        "CAGR":
            cagr,

        "Trades":
            len(
                trades
            ),

        "Avg_R":
            trades[
                "Net_R"
            ].mean(),

        "PF":
            profit_factor,

        "Max_DD":
            dd.min()
    }


# ============================================================
# TRADING DAY / EXPOSURE ANALYSIS
# ============================================================

def analyze_trading_days(
    trades,
    equity
):

    total_days = len(
        equity
    )

    active_days = int(
        equity[
            "Active"
        ].sum()
    )

    inactive_days = (
        total_days
        - active_days
    )

    active_pct = (
        active_days
        / total_days
        * 100
    )

    inactive_pct = (
        inactive_days
        / total_days
        * 100
    )

    average_positions = (
        equity[
            "Open_Positions"
        ].mean()
    )

    avg_exposure = (
        equity[
            "Capital_Exposure_%"
        ].mean()
    )

    avg_cash = (
        equity[
            "Cash_%"
        ].mean()
    )

    active_only_exposure = (
        equity.loc[
            equity[
                "Active"
            ] == 1,
            "Capital_Exposure_%"
        ].mean()
    )

    avg_holding = (
        trades[
            "Holding_Days"
        ].mean()
    )

    median_holding = (
        trades[
            "Holding_Days"
        ].median()
    )

    years = (
        (
            pd.Timestamp(
                equity[
                    "Date"
                ].iloc[-1]
            )
            -
            pd.Timestamp(
                equity[
                    "Date"
                ].iloc[0]
            )
        ).days
        / 365.25
    )

    trades_per_year = (
        len(trades)
        / years
    )

    position_distribution = (
        equity[
            "Open_Positions"
        ]
        .value_counts()
        .sort_index()
    )

    return {
        "Total_Days":
            total_days,

        "Active_Days":
            active_days,

        "Inactive_Days":
            inactive_days,

        "Active_%":
            active_pct,

        "Inactive_%":
            inactive_pct,

        "Average_Positions":
            average_positions,

        "Average_Exposure_%":
            avg_exposure,

        "Average_Cash_%":
            avg_cash,

        "Active_Day_Exposure_%":
            active_only_exposure,

        "Avg_Holding_Days":
            avg_holding,

        "Median_Holding_Days":
            median_holding,

        "Trades_Per_Year":
            trades_per_year,

        "Distribution":
            position_distribution
    }


# ============================================================
# YEARLY EXPOSURE
# ============================================================

def yearly_exposure(
    trades,
    equity
):

    eq = equity.copy()

    eq[
        "Year"
    ] = pd.to_datetime(
        eq[
            "Date"
        ]
    ).dt.year

    rows = []

    for year, group in (
        eq.groupby(
            "Year"
        )
    ):

        year_trades = trades[
            pd.to_datetime(
                trades[
                    "Entry_Date"
                ]
            ).dt.year
            ==
            year
        ]

        rows.append(
            {
                "Year":
                    year,

                "Market_Days":
                    len(
                        group
                    ),

                "Active_Days":
                    group[
                        "Active"
                    ].sum(),

                "Active_%":
                    group[
                        "Active"
                    ].mean()
                    * 100,

                "Avg_Positions":
                    group[
                        "Open_Positions"
                    ].mean(),

                "Avg_Exposure_%":
                    group[
                        "Capital_Exposure_%"
                    ].mean(),

                "Trades":
                    len(
                        year_trades
                    )
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "Year"
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
        "V17 - TRADING DAYS / CAPITAL UTILIZATION TEST"
    )

    print(
        "=" * 120
    )

    print(
        "Exact V15 strategy."
    )

    print(
        "No entry, exit, risk or portfolio rule changed."
    )

    print()

    print(
        "QUESTION:"
    )

    print(
        "Is CAGR low because the strategy spends too few days / too little capital in the market?"
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
        f"Backtest: "
        f"{backtest_start.date()} "
        f"-> {end_date.date()}"
    )


    # ========================================================
    # DATA
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
    # CANDIDATES
    # ========================================================

    (
        candidates,
        candidate_diag
    ) = generate_candidates(
        all_data,
        qqq,
        dxy,
        backtest_start
    )

    print()

    print(
        f"Candidates:            "
        f"{len(candidates)}"
    )

    print(
        f"Raw signals:           "
        f"{candidate_diag['Raw_Signals']}"
    )

    print(
        f"Rejected by regime:    "
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


    # ========================================================
    # NORMAL PERFORMANCE
    # ========================================================

    stats = calculate_stats(
        trades,
        equity
    )

    print()

    print(
        "=" * 120
    )

    print(
        "BASELINE PERFORMANCE"
    )

    print(
        "=" * 120
    )

    print(
        f"Ending capital:        "
        f"${stats['Ending']:,.2f}"
    )

    print(
        f"Return:                "
        f"{stats['Return']:.2f}%"
    )

    print(
        f"CAGR:                  "
        f"{stats['CAGR']:.2f}%"
    )

    print(
        f"Trades:                "
        f"{stats['Trades']}"
    )

    print(
        f"Avg Net R:             "
        f"{stats['Avg_R']:.3f}R"
    )

    print(
        f"Profit Factor:         "
        f"{stats['PF']:.3f}"
    )

    print(
        f"Max MTM DD:            "
        f"{stats['Max_DD']:.2f}%"
    )


    # ========================================================
    # TRADING DAYS
    # ========================================================

    exposure = analyze_trading_days(
        trades,
        equity
    )

    print()

    print(
        "=" * 120
    )

    print(
        "TRADING DAYS / UTILIZATION"
    )

    print(
        "=" * 120
    )

    print(
        f"Total market days:     "
        f"{exposure['Total_Days']}"
    )

    print(
        f"Active trading days:   "
        f"{exposure['Active_Days']}"
    )

    print(
        f"Inactive days:         "
        f"{exposure['Inactive_Days']}"
    )

    print()

    print(
        f"ACTIVE DAYS:           "
        f"{exposure['Active_%']:.2f}%"
    )

    print(
        f"INACTIVE DAYS:         "
        f"{exposure['Inactive_%']:.2f}%"
    )

    print()

    print(
        f"Average open positions:"
        f" {exposure['Average_Positions']:.2f}"
        f" / {MAX_POSITIONS}"
    )

    print(
        f"Average capital used:  "
        f"{exposure['Average_Exposure_%']:.2f}%"
    )

    print(
        f"Average cash idle:     "
        f"{exposure['Average_Cash_%']:.2f}%"
    )

    print(
        f"Exposure when active:  "
        f"{exposure['Active_Day_Exposure_%']:.2f}%"
    )

    print()

    print(
        f"Average holding days:  "
        f"{exposure['Avg_Holding_Days']:.2f}"
    )

    print(
        f"Median holding days:   "
        f"{exposure['Median_Holding_Days']:.2f}"
    )

    print(
        f"Trades per year:       "
        f"{exposure['Trades_Per_Year']:.2f}"
    )


    # ========================================================
    # POSITION DISTRIBUTION
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "OPEN POSITION DISTRIBUTION"
    )

    print(
        "=" * 120
    )

    total = exposure[
        "Total_Days"
    ]

    for positions, days in (
        exposure[
            "Distribution"
        ].items()
    ):

        pct = (
            days
            / total
            * 100
        )

        print(
            f"{positions} positions: "
            f"{days:4d} days "
            f"({pct:6.2f}%)"
        )


    # ========================================================
    # YEARLY UTILIZATION
    # ========================================================

    yearly = yearly_exposure(
        trades,
        equity
    )

    print()

    print(
        "=" * 120
    )

    print(
        "YEAR-BY-YEAR TRADING ACTIVITY"
    )

    print(
        "=" * 120
    )

    print(
        yearly
        .round(2)
        .to_string()
    )


    # ========================================================
    # CONCEPT DIAGNOSIS
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "CAGR / TRADING DAYS DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        f"Strategy CAGR:         "
        f"{stats['CAGR']:.2f}%"
    )

    print(
        f"Active market days:    "
        f"{exposure['Active_%']:.2f}%"
    )

    print(
        f"Average capital used:  "
        f"{exposure['Average_Exposure_%']:.2f}%"
    )

    print(
        f"Average positions:     "
        f"{exposure['Average_Positions']:.2f}"
        f" / {MAX_POSITIONS}"
    )

    print()

    if (
        exposure[
            "Average_Exposure_%"
        ] < 40
    ):

        print(
            "DIAGNOSIS: VERY LOW CAPITAL UTILIZATION"
        )

    elif (
        exposure[
            "Average_Exposure_%"
        ] < 65
    ):

        print(
            "DIAGNOSIS: MODERATE-LOW CAPITAL UTILIZATION"
        )

    else:

        print(
            "DIAGNOSIS: CAPITAL UTILIZATION IS ALREADY HIGH"
        )


    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "v17_trades.csv",
        index=False
    )

    equity.to_csv(
        "v17_daily_exposure.csv",
        index=False
    )

    yearly.to_csv(
        "v17_yearly_activity.csv"
    )

    print()

    print(
        "=" * 120
    )

    print(
        "V17 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
