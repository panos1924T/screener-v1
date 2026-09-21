import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V23 - QUALITY SCORE / RANKING DIAGNOSTIC
#
# EXACT V22 BEST CORE
#
# LOCKED:
#   QQQ > SMA200
#   DXY < SMA200
#   STOCH RSI OFF
#   RSI FILTER OFF
#   RELATIVE VOLUME >= 0.80 ON
#   MAX POSITIONS = 5
#   RISK = 0.75%
#   EXIT = 2.5 ATR TRAIL
#
# NO STRATEGY RULE CHANGES.
#
# PURPOSE:
# Does Quality Score actually rank better trades higher?
#
# ANALYSIS:
#   A) EXECUTED trades by score quartile
#   B) ALL candidates independently by score quartile
#   C) Score vs R rank correlation
#   D) Top half vs bottom half
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

REL_VOLUME_MIN = 0.80


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

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
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


def get_market_row(df, date):

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
        or
        dxy_row is None
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
# STRONG TREND
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
# QUALITY SCORE + COMPONENTS
# ============================================================

def score_components(df, i):

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
        df["SMA50"].iloc[
            i - 10
        ]
    )

    slope_score = (
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

    capped_relative_volume = min(
        relative_volume,
        2.0
    )

    quality = (
        slope_score
        + ema_strength
        + close_location
        + capped_relative_volume
    )

    return {
        "Score":
            quality,

        "Slope_Score":
            slope_score,

        "EMA_Strength":
            ema_strength,

        "Close_Location":
            close_location,

        "Relative_Volume":
            relative_volume
    }


# ============================================================
# SETUP - EXACT V22 BEST
# ============================================================

def pullback_setup(df, i):

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
    # EMA20 PULLBACK
    # ========================================================

    ema_distance = (
        abs(
            low - ema20
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
    # RELATIVE VOLUME >= 0.80
    # ========================================================

    if avg_volume <= 0:

        return None

    relative_volume = (
        volume
        / avg_volume
    )

    if relative_volume < REL_VOLUME_MIN:

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


    components = score_components(
        df,
        i
    )

    return {
        "Trigger":
            trigger,

        "Stop":
            stop,

        **components
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
#
# KEPT EXACTLY FOR COMPARABILITY
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
# GENERATE CANDIDATES
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
                            ),

                        "Slope_Score":
                            float(
                                setup[
                                    "Slope_Score"
                                ]
                            ),

                        "EMA_Strength":
                            float(
                                setup[
                                    "EMA_Strength"
                                ]
                            ),

                        "Close_Location":
                            float(
                                setup[
                                    "Close_Location"
                                ]
                            ),

                        "Relative_Volume":
                            float(
                                setup[
                                    "Relative_Volume"
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


# ============================================================
# ACTUAL 2.5 ATR EXIT
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
# CREATE ALL TRADE PLANS
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


        # ====================================================
        # STANDALONE CANDIDATE R
        #
        # Treat each candidate independently.
        # Includes slippage + commissions.
        # Position size is irrelevant because R is normalized.
        # ====================================================

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

        raw_exit = float(
            result[
                "Raw_Exit"
            ]
        )

        entry_fill = (
            raw_entry
            * (
                1
                + SLIPPAGE_PCT
            )
        )

        exit_fill = (
            raw_exit
            * (
                1
                - SLIPPAGE_PCT
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

        entry_cost = (
            entry_fill
            * (
                1
                + COMMISSION_PCT
            )
        )

        exit_proceeds = (
            exit_fill
            * (
                1
                - COMMISSION_PCT
            )
        )

        candidate_pnl_per_share = (
            exit_proceeds
            - entry_cost
        )

        candidate_net_r = (
            candidate_pnl_per_share
            / risk_per_share
            if risk_per_share > 0
            else np.nan
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
                    raw_exit,

                "Exit_Reason":
                    result[
                        "Exit_Reason"
                    ],

                "Candidate_Net_R":
                    candidate_net_r
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

                        "Slope_Score":
                            trade[
                                "Slope_Score"
                            ],

                        "EMA_Strength":
                            trade[
                                "EMA_Strength"
                            ],

                        "Close_Location":
                            trade[
                                "Close_Location"
                            ],

                        "Relative_Volume":
                            trade[
                                "Relative_Volume"
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


        open_positions = remaining


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

                "Exposure_%":
                    exposure_pct
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


    pf = (
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
            pf,

        "Max_DD":
            dd.min(),

        "Exposure":
            equity[
                "Exposure_%"
            ].mean()
    }


# ============================================================
# QUARTILE ANALYSIS
# ============================================================

def quartile_analysis(
    df,
    r_column
):

    if df.empty:

        return pd.DataFrame()


    work = df.copy()


    # Rank first so duplicate scores cannot break qcut.
    score_rank = (
        work[
            "Score"
        ]
        .rank(
            method="first"
        )
    )


    work[
        "Score_Quartile"
    ] = pd.qcut(
        score_rank,
        4,
        labels=[
            "Q1 Bottom",
            "Q2",
            "Q3",
            "Q4 Top"
        ]
    )


    rows = []


    for quartile, group in (
        work.groupby(
            "Score_Quartile",
            observed=False
        )
    ):

        positive = group.loc[
            group[
                r_column
            ] > 0,
            r_column
        ]

        negative = group.loc[
            group[
                r_column
            ] < 0,
            r_column
        ]


        pf_r = (
            positive.sum()
            /
            abs(
                negative.sum()
            )
            if abs(
                negative.sum()
            ) > 0
            else np.inf
        )


        rows.append(
            {
                "Quartile":
                    str(
                        quartile
                    ),

                "Trades":
                    len(
                        group
                    ),

                "Avg_Score":
                    group[
                        "Score"
                    ].mean(),

                "Avg_R":
                    group[
                        r_column
                    ].mean(),

                "Median_R":
                    group[
                        r_column
                    ].median(),

                "PF_R":
                    pf_r,

                "Win_%":
                    (
                        group[
                            r_column
                        ] > 0
                    ).mean()
                    * 100
            }
        )


    result = pd.DataFrame(
        rows
    )


    order = [
        "Q4 Top",
        "Q3",
        "Q2",
        "Q1 Bottom"
    ]


    result[
        "Quartile"
    ] = pd.Categorical(
        result[
            "Quartile"
        ],
        categories=order,
        ordered=True
    )


    return (
        result
        .sort_values(
            "Quartile"
        )
        .set_index(
            "Quartile"
        )
    )


# ============================================================
# TOP HALF VS BOTTOM HALF
# ============================================================

def half_analysis(
    df,
    r_column
):

    if df.empty:

        return pd.DataFrame()


    work = df.copy()


    score_rank = (
        work[
            "Score"
        ]
        .rank(
            method="first"
        )
    )


    work[
        "Half"
    ] = pd.qcut(
        score_rank,
        2,
        labels=[
            "Bottom 50%",
            "Top 50%"
        ]
    )


    rows = []


    for half, group in (
        work.groupby(
            "Half",
            observed=False
        )
    ):

        positive = group.loc[
            group[
                r_column
            ] > 0,
            r_column
        ]

        negative = group.loc[
            group[
                r_column
            ] < 0,
            r_column
        ]


        pf_r = (
            positive.sum()
            /
            abs(
                negative.sum()
            )
            if abs(
                negative.sum()
            ) > 0
            else np.inf
        )


        rows.append(
            {
                "Half":
                    str(
                        half
                    ),

                "Trades":
                    len(
                        group
                    ),

                "Avg_Score":
                    group[
                        "Score"
                    ].mean(),

                "Avg_R":
                    group[
                        r_column
                    ].mean(),

                "Median_R":
                    group[
                        r_column
                    ].median(),

                "PF_R":
                    pf_r,

                "Win_%":
                    (
                        group[
                            r_column
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
            "Half"
        )
    )


# ============================================================
# RANK CORRELATION
# ============================================================

def rank_correlation(
    df,
    score_column,
    r_column
):

    valid = (
        df[
            [
                score_column,
                r_column
            ]
        ]
        .dropna()
    )


    if len(valid) < 2:

        return np.nan


    return (
        valid[
            score_column
        ]
        .rank()
        .corr(
            valid[
                r_column
            ]
            .rank()
        )
    )


# ============================================================
# SCORE COMPONENT DIAGNOSTIC
# ============================================================

def component_correlations(
    df,
    r_column
):

    components = [
        "Score",
        "Slope_Score",
        "EMA_Strength",
        "Close_Location",
        "Relative_Volume"
    ]


    rows = []


    for component in components:

        correlation = rank_correlation(
            df,
            component,
            r_column
        )


        rows.append(
            {
                "Component":
                    component,

                "Rank_Correlation_With_R":
                    correlation
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "Component"
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
        "V23 - QUALITY SCORE / RANKING DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    print(
        "NO strategy rules changed."
    )

    print()

    print(
        "Current locked core:"
    )

    print(
        "  QQQ > SMA200"
    )

    print(
        "  DXY < SMA200"
    )

    print(
        "  Stoch RSI OFF"
    )

    print(
        "  RSI OFF"
    )

    print(
        "  Relative Volume >= 0.80"
    )

    print(
        "  2.5 ATR trail"
    )

    print(
        "  Risk 0.75%"
    )

    print(
        "  Max positions 5"
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
    # PREPARE
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

        except Exception as e:

            print(
                f"PREP ERROR {ticker}: {e}"
            )


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
        f"Rejected market:       "
        f"{candidate_diag['Rejected_Market']}"
    )


    # ========================================================
    # ALL CANDIDATE TRADE PLANS
    # ========================================================

    plans = create_trade_plans(
        candidates,
        prepared_data
    )


    # ========================================================
    # PORTFOLIO
    # ========================================================

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
        equity
    )


    # ========================================================
    # BASELINE CHECK
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "BASELINE PERFORMANCE CHECK"
    )

    print(
        "=" * 120
    )

    print(
        f"Ending capital:        "
        f"${stats['Ending']:,.2f}"
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
        f"Max DD:                "
        f"{stats['Max_DD']:.2f}%"
    )

    print(
        f"Average exposure:      "
        f"{stats['Exposure']:.2f}%"
    )

    print()

    print(
        f"Skipped max positions: "
        f"{diagnostics['Skipped_Positions']}"
    )

    print(
        f"Skipped cash:          "
        f"{diagnostics['Skipped_Cash']}"
    )

    print(
        f"Skipped same ticker:   "
        f"{diagnostics['Skipped_Same_Ticker']}"
    )


    # ========================================================
    # A. EXECUTED TRADES
    # ========================================================

    executed_quartiles = (
        quartile_analysis(
            trades,
            "Net_R"
        )
    )


    print()

    print(
        "=" * 120
    )

    print(
        "A. EXECUTED TRADES - SCORE QUARTILES"
    )

    print(
        "=" * 120
    )

    print(
        executed_quartiles
        .round(3)
        .to_string()
    )


    executed_halves = (
        half_analysis(
            trades,
            "Net_R"
        )
    )


    print()

    print(
        "EXECUTED - TOP HALF VS BOTTOM HALF"
    )

    print(
        "-" * 80
    )

    print(
        executed_halves
        .round(3)
        .to_string()
    )


    executed_corr = (
        rank_correlation(
            trades,
            "Score",
            "Net_R"
        )
    )


    print()

    print(
        f"Executed Score -> R "
        f"rank correlation: "
        f"{executed_corr:.4f}"
    )


    # ========================================================
    # B. ALL CANDIDATES
    # ========================================================

    candidate_quartiles = (
        quartile_analysis(
            plans,
            "Candidate_Net_R"
        )
    )


    print()

    print(
        "=" * 120
    )

    print(
        "B. ALL VALID CANDIDATES - SCORE QUARTILES"
    )

    print(
        "=" * 120
    )

    print(
        "Each candidate is evaluated independently."
    )

    print(
        "Portfolio position/cash limits are ignored here."
    )

    print()

    print(
        candidate_quartiles
        .round(3)
        .to_string()
    )


    candidate_halves = (
        half_analysis(
            plans,
            "Candidate_Net_R"
        )
    )


    print()

    print(
        "ALL CANDIDATES - TOP HALF VS BOTTOM HALF"
    )

    print(
        "-" * 80
    )

    print(
        candidate_halves
        .round(3)
        .to_string()
    )


    candidate_corr = (
        rank_correlation(
            plans,
            "Score",
            "Candidate_Net_R"
        )
    )


    print()

    print(
        f"Candidate Score -> R "
        f"rank correlation: "
        f"{candidate_corr:.4f}"
    )


    # ========================================================
    # C. COMPONENT CORRELATIONS
    # ========================================================

    component_stats = (
        component_correlations(
            plans,
            "Candidate_Net_R"
        )
    )


    print()

    print(
        "=" * 120
    )

    print(
        "C. SCORE COMPONENT CORRELATIONS"
    )

    print(
        "=" * 120
    )

    print(
        "Positive = higher component tends to correspond to higher R."
    )

    print(
        "Negative = higher component tends to correspond to lower R."
    )

    print()

    print(
        component_stats
        .round(4)
        .to_string()
    )


    # ========================================================
    # SIMPLE INTERPRETATION FLAGS
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "D. RANKING DIAGNOSTIC"
    )

    print(
        "=" * 120
    )


    if (
        "Q4 Top"
        in candidate_quartiles.index
        and
        "Q1 Bottom"
        in candidate_quartiles.index
    ):

        top_r = (
            candidate_quartiles.loc[
                "Q4 Top",
                "Avg_R"
            ]
        )

        bottom_r = (
            candidate_quartiles.loc[
                "Q1 Bottom",
                "Avg_R"
            ]
        )

        top_pf = (
            candidate_quartiles.loc[
                "Q4 Top",
                "PF_R"
            ]
        )

        bottom_pf = (
            candidate_quartiles.loc[
                "Q1 Bottom",
                "PF_R"
            ]
        )


        print(
            f"Top quartile Avg R:    "
            f"{top_r:.3f}R"
        )

        print(
            f"Bottom quartile Avg R: "
            f"{bottom_r:.3f}R"
        )

        print(
            f"Top quartile PF_R:     "
            f"{top_pf:.3f}"
        )

        print(
            f"Bottom quartile PF_R:  "
            f"{bottom_pf:.3f}"
        )

        print()


        if (
            top_r > bottom_r
            and
            candidate_corr > 0.10
        ):

            print(
                "RESULT: Score shows useful positive ranking ability."
            )

        elif (
            top_r > bottom_r
            and
            candidate_corr > 0
        ):

            print(
                "RESULT: Score shows weak positive ranking ability."
            )

        else:

            print(
                "RESULT: Score does NOT show reliable ranking ability."
            )


    # ========================================================
    # SAVE
    # ========================================================

    candidates.to_csv(
        "v23_candidates.csv",
        index=False
    )

    plans.to_csv(
        "v23_all_candidate_plans.csv",
        index=False
    )

    trades.to_csv(
        "v23_executed_trades.csv",
        index=False
    )

    equity.to_csv(
        "v23_equity.csv",
        index=False
    )

    executed_quartiles.to_csv(
        "v23_executed_quartiles.csv"
    )

    candidate_quartiles.to_csv(
        "v23_candidate_quartiles.csv"
    )

    component_stats.to_csv(
        "v23_score_component_correlations.csv"
    )


    print()

    print(
        "=" * 120
    )

    print(
        "V23 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
