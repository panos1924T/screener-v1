import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# V24 - ALL SIGNALS / PURE STRATEGY EDGE TEST
#
# PURPOSE:
# Evaluate EVERY valid signal independently.
#
# NO:
# - max positions
# - cash constraint
# - portfolio ranking
# - position sizing
# - skipped signals because another trade is open
#
# IMPORTANT:
# CAGR is intentionally NOT calculated.
#
# We measure the edge of the signal engine itself:
# - Avg Net R
# - Profit Factor in R
# - Win rate
# - Median R
# - Trades/year
# - Year-by-year consistency
# - Period consistency
# - R drawdown
# - holding period
#
# CURRENT LOCKED STRATEGY:
# QQQ > SMA200
# DXY < SMA200
# RSI OFF
# Stoch RSI OFF
# Relative Volume >= 0.80
# 2.5 ATR trailing exit
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

DOWNLOAD_PERIOD = "max"
BACKTEST_YEARS = 15

ATR_PERIOD = 14

ENTRY_VALID_DAYS = 3

TRAIL_HOLD_DAYS = 30
ATR_TRAIL_MULT = 2.5

REL_VOLUME_MIN = 0.80

SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0002


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
# MARKET DATA
# ============================================================

def download_market(ticker, name):

    print(f"Downloading {name}...")

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
# TREND
# ============================================================

def strong_trend(df, i):

    if i < 10:
        return False

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
# SETUP
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
    # PULLBACK TO EMA20
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
    # RELATIVE VOLUME
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


    return {
        "Trigger":
            trigger,

        "Stop":
            stop,

        "ATR":
            atr,

        "EMA_Distance_ATR":
            ema_distance,

        "Close_Location":
            close_location,

        "Relative_Volume":
            relative_volume
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
# EXIT
#
# 2.5 ATR trailing stop
# max 30 trading bars
# ============================================================

def simulate_trade(
    df,
    entry_index,
    trigger,
    initial_stop
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


    # Execution-adjusted values
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
        return None


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


        # Trail is based only on information
        # available before today's close.
        calculated_trail = (
            highest_close
            - ATR_TRAIL_MULT
            * atr
        )


        trail = max(
            trail,
            calculated_trail
        )


        # ====================================================
        # GAP THROUGH STOP
        # ====================================================

        if open_price < trail:

            raw_exit = open_price

            exit_reason = (
                "ATR_GAP"
            )

            exit_index = i

            break


        # ====================================================
        # INTRADAY STOP
        # ====================================================

        if low <= trail:

            raw_exit = trail

            exit_reason = (
                "ATR_TRAIL"
            )

            exit_index = i

            break


        highest_close = max(
            highest_close,
            close
        )

    else:

        exit_index = end

        raw_exit = float(
            df["Close"].iloc[
                end
            ]
        )

        exit_reason = (
            "TIME30"
        )


    # ========================================================
    # COSTS
    # ========================================================

    exit_fill = (
        raw_exit
        * (
            1
            - SLIPPAGE_PCT
        )
    )


    entry_cost_per_share = (
        entry_fill
        * (
            1
            + COMMISSION_PCT
        )
    )


    exit_proceeds_per_share = (
        exit_fill
        * (
            1
            - COMMISSION_PCT
        )
    )


    pnl_per_share = (
        exit_proceeds_per_share
        - entry_cost_per_share
    )


    net_r = (
        pnl_per_share
        / risk_per_share
    )


    holding_bars = (
        exit_index
        - entry_index
        + 1
    )


    holding_calendar_days = (
        pd.Timestamp(
            df.index[
                exit_index
            ]
        )
        -
        pd.Timestamp(
            df.index[
                entry_index
            ]
        )
    ).days


    return {
        "Entry_Index":
            entry_index,

        "Entry_Date":
            pd.Timestamp(
                df.index[
                    entry_index
                ]
            ),

        "Exit_Index":
            exit_index,

        "Exit_Date":
            pd.Timestamp(
                df.index[
                    exit_index
                ]
            ),

        "Raw_Entry":
            raw_entry,

        "Entry_Fill":
            entry_fill,

        "Initial_Stop":
            initial_stop,

        "Raw_Exit":
            raw_exit,

        "Exit_Fill":
            exit_fill,

        "Exit_Reason":
            exit_reason,

        "Risk_Per_Share":
            risk_per_share,

        "Net_R":
            net_r,

        "Holding_Bars":
            holding_bars,

        "Holding_Days":
            holding_calendar_days
    }


# ============================================================
# GENERATE EVERY SIGNAL
#
# IMPORTANT:
#
# No:
#   i = exit_index + 1
#
# Every stock/day can generate a new independent signal.
#
# ============================================================

def generate_all_trades(
    all_data,
    qqq,
    dxy,
    backtest_start,
    end_date
):

    trades = []


    diagnostic = {
        "Stock_Days_Checked": 0,
        "Raw_Setups": 0,
        "No_Trigger": 0,
        "Market_Rejected": 0,
        "Trades": 0
    }


    for number, ticker in enumerate(
        TICKERS,
        start=1
    ):

        print(
            f"[{number:03d}/{len(TICKERS)}] "
            f"{ticker}"
        )


        try:

            if (
                ticker
                not in
                all_data.columns
                .get_level_values(0)
            ):

                print(
                    "   Missing download"
                )

                continue


            df = prepare_stock(
                all_data[
                    ticker
                ].copy()
            )


            if len(df) < 250:

                continue


            for i in range(
                210,
                len(df) - ENTRY_VALID_DAYS - 1
            ):

                signal_date = pd.Timestamp(
                    df.index[i]
                )


                if (
                    signal_date
                    < backtest_start
                ):

                    continue


                if (
                    signal_date
                    > end_date
                ):

                    break


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

                    continue


                diagnostic[
                    "Stock_Days_Checked"
                ] += 1


                setup = pullback_setup(
                    df,
                    i
                )


                if setup is None:

                    continue


                diagnostic[
                    "Raw_Setups"
                ] += 1


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

                    diagnostic[
                        "No_Trigger"
                    ] += 1

                    continue


                entry_date = pd.Timestamp(
                    df.index[
                        entry_index
                    ]
                )


                if (
                    entry_date
                    > end_date
                ):

                    continue


                if not market_allows_long(
                    qqq,
                    dxy,
                    entry_date
                ):

                    diagnostic[
                        "Market_Rejected"
                    ] += 1

                    continue


                result = simulate_trade(
                    df,
                    entry_index,
                    trigger,
                    stop
                )


                if result is None:

                    continue


                trades.append(
                    {
                        "Ticker":
                            ticker,

                        "Signal_Date":
                            signal_date,

                        **result,

                        "EMA_Distance_ATR":
                            setup[
                                "EMA_Distance_ATR"
                            ],

                        "Close_Location":
                            setup[
                                "Close_Location"
                            ],

                        "Relative_Volume":
                            setup[
                                "Relative_Volume"
                            ]
                    }
                )


                diagnostic[
                    "Trades"
                ] += 1


        except Exception as e:

            print(
                f"   ERROR: "
                f"{type(e).__name__}: {e}"
            )


    trades = pd.DataFrame(
        trades
    )


    if not trades.empty:

        trades = (
            trades
            .sort_values(
                [
                    "Entry_Date",
                    "Ticker",
                    "Signal_Date"
                ]
            )
            .reset_index(
                drop=True
            )
        )


    return (
        trades,
        diagnostic
    )


# ============================================================
# OVERALL PERFORMANCE IN R
# ============================================================

def overall_stats(trades):

    positive = trades.loc[
        trades["Net_R"] > 0,
        "Net_R"
    ]

    negative = trades.loc[
        trades["Net_R"] < 0,
        "Net_R"
    ]


    profit_factor_r = (
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


    start = pd.Timestamp(
        trades[
            "Entry_Date"
        ].min()
    )

    end = pd.Timestamp(
        trades[
            "Exit_Date"
        ].max()
    )

    years = (
        (
            end - start
        ).days
        / 365.25
    )


    return {
        "Trades":
            len(
                trades
            ),

        "Years":
            years,

        "Trades_Per_Year":
            len(
                trades
            )
            / years,

        "Win_%":
            (
                trades[
                    "Net_R"
                ] > 0
            ).mean()
            * 100,

        "Avg_R":
            trades[
                "Net_R"
            ].mean(),

        "Median_R":
            trades[
                "Net_R"
            ].median(),

        "PF_R":
            profit_factor_r,

        "Total_R":
            trades[
                "Net_R"
            ].sum(),

        "Avg_Holding_Bars":
            trades[
                "Holding_Bars"
            ].mean(),

        "Median_Holding_Bars":
            trades[
                "Holding_Bars"
            ].median(),

        "Avg_Holding_Days":
            trades[
                "Holding_Days"
            ].mean()
    }


# ============================================================
# YEARLY
# ============================================================

def yearly_analysis(trades):

    df = trades.copy()

    df["Year"] = (
        pd.to_datetime(
            df[
                "Entry_Date"
            ]
        )
        .dt.year
    )


    rows = []


    for year, group in (
        df.groupby(
            "Year"
        )
    ):

        positive = group.loc[
            group[
                "Net_R"
            ] > 0,
            "Net_R"
        ]

        negative = group.loc[
            group[
                "Net_R"
            ] < 0,
            "Net_R"
        ]


        pf = (
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
                "Year":
                    year,

                "Trades":
                    len(
                        group
                    ),

                "Avg_R":
                    group[
                        "Net_R"
                    ].mean(),

                "Median_R":
                    group[
                        "Net_R"
                    ].median(),

                "PF_R":
                    pf,

                "Total_R":
                    group[
                        "Net_R"
                    ].sum(),

                "Win_%":
                    (
                        group[
                            "Net_R"
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
            "Year"
        )
    )


# ============================================================
# 5-YEAR PERIOD ANALYSIS
# ============================================================

def period_analysis(trades):

    df = trades.copy()


    start_year = int(
        pd.to_datetime(
            df[
                "Entry_Date"
            ]
        )
        .dt.year
        .min()
    )


    end_year = int(
        pd.to_datetime(
            df[
                "Entry_Date"
            ]
        )
        .dt.year
        .max()
    )


    periods = []


    first = start_year


    while first <= end_year:

        last = min(
            first + 4,
            end_year
        )


        group = df[
            (
                pd.to_datetime(
                    df[
                        "Entry_Date"
                    ]
                )
                .dt.year
                >= first
            )
            &
            (
                pd.to_datetime(
                    df[
                        "Entry_Date"
                    ]
                )
                .dt.year
                <= last
            )
        ]


        if not group.empty:

            positive = group.loc[
                group[
                    "Net_R"
                ] > 0,
                "Net_R"
            ]

            negative = group.loc[
                group[
                    "Net_R"
                ] < 0,
                "Net_R"
            ]


            pf = (
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


            periods.append(
                {
                    "Period":
                        f"{first}-{last}",

                    "Trades":
                        len(
                            group
                        ),

                    "Avg_R":
                        group[
                            "Net_R"
                        ].mean(),

                    "Median_R":
                        group[
                            "Net_R"
                        ].median(),

                    "PF_R":
                        pf,

                    "Total_R":
                        group[
                            "Net_R"
                        ].sum(),

                    "Win_%":
                        (
                            group[
                                "Net_R"
                            ] > 0
                        ).mean()
                        * 100
                }
            )


        first += 5


    return (
        pd.DataFrame(
            periods
        )
        .set_index(
            "Period"
        )
    )


# ============================================================
# CUMULATIVE R DRAWDOWN
#
# NOT ACCOUNT DRAWDOWN.
#
# This simply tells us about losing sequences in signal R.
# ============================================================

def cumulative_r_analysis(trades):

    ordered = (
        trades
        .sort_values(
            [
                "Exit_Date",
                "Ticker"
            ]
        )
        .copy()
    )


    ordered[
        "Cumulative_R"
    ] = (
        ordered[
            "Net_R"
        ]
        .cumsum()
    )


    peak = (
        ordered[
            "Cumulative_R"
        ]
        .cummax()
    )


    drawdown_r = (
        ordered[
            "Cumulative_R"
        ]
        - peak
    )


    return {
        "Max_R_Drawdown":
            drawdown_r.min(),

        "Final_Cumulative_R":
            ordered[
                "Cumulative_R"
            ].iloc[-1]
    }


# ============================================================
# EXIT ANALYSIS
# ============================================================

def exit_analysis(trades):

    rows = []


    for reason, group in (
        trades.groupby(
            "Exit_Reason"
        )
    ):

        rows.append(
            {
                "Exit":
                    reason,

                "Trades":
                    len(
                        group
                    ),

                "Avg_R":
                    group[
                        "Net_R"
                    ].mean(),

                "Median_R":
                    group[
                        "Net_R"
                    ].median(),

                "Total_R":
                    group[
                        "Net_R"
                    ].sum(),

                "Win_%":
                    (
                        group[
                            "Net_R"
                        ] > 0
                    ).mean()
                    * 100,

                "Avg_Holding_Bars":
                    group[
                        "Holding_Bars"
                    ].mean()
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "Exit"
        )
    )


# ============================================================
# HOLDING PERIOD BUCKETS
# ============================================================

def holding_analysis(trades):

    df = trades.copy()


    bins = [
        0,
        5,
        10,
        15,
        20,
        25,
        31
    ]


    labels = [
        "1-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-30"
    ]


    df[
        "Holding_Bucket"
    ] = pd.cut(
        df[
            "Holding_Bars"
        ],
        bins=bins,
        labels=labels,
        right=False
    )


    rows = []


    for bucket, group in (
        df.groupby(
            "Holding_Bucket",
            observed=False
        )
    ):

        if group.empty:
            continue


        positive = group.loc[
            group[
                "Net_R"
            ] > 0,
            "Net_R"
        ]

        negative = group.loc[
            group[
                "Net_R"
            ] < 0,
            "Net_R"
        ]


        pf = (
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
                "Holding":
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

                "Median_R":
                    group[
                        "Net_R"
                    ].median(),

                "PF_R":
                    pf,

                "Total_R":
                    group[
                        "Net_R"
                    ].sum(),

                "Win_%":
                    (
                        group[
                            "Net_R"
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
            "Holding"
        )
    )


# ============================================================
# CONCURRENT TRADE LOAD
#
# Diagnostic:
# If every signal were accepted, how many trades would
# theoretically overlap at the same time?
# ============================================================

def concurrency_analysis(
    trades,
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


    rows = []


    for date in calendar:

        date = pd.Timestamp(
            date
        )


        concurrent = (
            (
                trades[
                    "Entry_Date"
                ]
                <= date
            )
            &
            (
                trades[
                    "Exit_Date"
                ]
                >= date
            )
        ).sum()


        rows.append(
            {
                "Date":
                    date,

                "Concurrent_Trades":
                    concurrent
            }
        )


    result = pd.DataFrame(
        rows
    )


    return {
        "Average":
            result[
                "Concurrent_Trades"
            ].mean(),

        "Median":
            result[
                "Concurrent_Trades"
            ].median(),

        "Max":
            result[
                "Concurrent_Trades"
            ].max(),

        "Days_Above_5":
            (
                result[
                    "Concurrent_Trades"
                ] > 5
            ).sum(),

        "Pct_Days_Above_5":
            (
                result[
                    "Concurrent_Trades"
                ] > 5
            ).mean()
            * 100,

        "Daily":
            result
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "V24 - ALL SIGNALS / PURE STRATEGY EDGE"
    )

    print(
        "=" * 120
    )

    print(
        "NO portfolio restrictions."
    )

    print(
        "Every valid signal is evaluated independently."
    )

    print()

    print(
        "CAGR is intentionally NOT calculated."
    )

    print(
        "This test measures signal quality, not portfolio return."
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

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
        threads=False
    )


    # threads=False deliberately used to reduce
    # yfinance SQLite 'database is locked' errors.


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
    # ALL SIGNALS
    # ========================================================

    (
        trades,
        diagnostic
    ) = generate_all_trades(
        all_data,
        qqq,
        dxy,
        backtest_start,
        end_date
    )


    if trades.empty:

        print(
            "No trades generated."
        )

        return


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "SIGNAL FUNNEL"
    )

    print(
        "=" * 120
    )


    for key, value in (
        diagnostic.items()
    ):

        print(
            f"{key:<25}"
            f"{value:>10}"
        )


    # ========================================================
    # OVERALL
    # ========================================================

    stats = overall_stats(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        "PURE STRATEGY RESULTS"
    )

    print(
        "=" * 120
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
        f"Win rate:              "
        f"{stats['Win_%']:.2f}%"
    )

    print(
        f"Avg Net R:             "
        f"{stats['Avg_R']:.3f}R"
    )

    print(
        f"Median R:              "
        f"{stats['Median_R']:.3f}R"
    )

    print(
        f"Profit Factor (R):     "
        f"{stats['PF_R']:.3f}"
    )

    print(
        f"Total R:               "
        f"{stats['Total_R']:.2f}R"
    )

    print()

    print(
        f"Avg holding bars:      "
        f"{stats['Avg_Holding_Bars']:.2f}"
    )

    print(
        f"Median holding bars:   "
        f"{stats['Median_Holding_Bars']:.2f}"
    )

    print(
        f"Avg calendar days:     "
        f"{stats['Avg_Holding_Days']:.2f}"
    )


    # ========================================================
    # R DRAWDOWN
    # ========================================================

    r_stats = cumulative_r_analysis(
        trades
    )


    print()

    print(
        f"Final cumulative R:    "
        f"{r_stats['Final_Cumulative_R']:.2f}R"
    )

    print(
        f"Max cumulative R DD:   "
        f"{r_stats['Max_R_Drawdown']:.2f}R"
    )

    print(
        "(This is NOT portfolio drawdown.)"
    )


    # ========================================================
    # YEARLY
    # ========================================================

    yearly = yearly_analysis(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        "YEAR-BY-YEAR SIGNAL EDGE"
    )

    print(
        "=" * 120
    )


    print(
        yearly
        .round(3)
        .to_string()
    )


    positive_years = (
        yearly[
            "Avg_R"
        ] > 0
    ).sum()


    total_years = len(
        yearly
    )


    print()

    print(
        f"Positive Avg-R years:  "
        f"{positive_years}/{total_years} "
        f"({positive_years / total_years * 100:.1f}%)"
    )


    # ========================================================
    # 5 YEAR PERIODS
    # ========================================================

    periods = period_analysis(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        "5-YEAR PERIOD ROBUSTNESS"
    )

    print(
        "=" * 120
    )


    print(
        periods
        .round(3)
        .to_string()
    )


    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    exits = exit_analysis(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        "EXIT TYPE ANALYSIS"
    )

    print(
        "=" * 120
    )


    print(
        exits
        .round(3)
        .to_string()
    )


    # ========================================================
    # HOLDING ANALYSIS
    # ========================================================

    holding = holding_analysis(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        "HOLDING PERIOD ANALYSIS"
    )

    print(
        "=" * 120
    )


    print(
        holding
        .round(3)
        .to_string()
    )


    # ========================================================
    # CONCURRENCY
    # ========================================================

    concurrency = concurrency_analysis(
        trades,
        qqq,
        backtest_start,
        end_date
    )


    print()

    print(
        "=" * 120
    )

    print(
        "IF WE TOOK EVERY SIGNAL: CONCURRENT POSITIONS"
    )

    print(
        "=" * 120
    )


    print(
        f"Average simultaneous:  "
        f"{concurrency['Average']:.2f}"
    )

    print(
        f"Median simultaneous:   "
        f"{concurrency['Median']:.2f}"
    )

    print(
        f"Maximum simultaneous:  "
        f"{concurrency['Max']}"
    )

    print(
        f"Days with >5 trades:   "
        f"{concurrency['Days_Above_5']}"
    )

    print(
        f"% days with >5:        "
        f"{concurrency['Pct_Days_Above_5']:.2f}%"
    )


    # ========================================================
    # SIMPLE EDGE CHECK
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "PURE EDGE CHECK"
    )

    print(
        "=" * 120
    )


    checks = {
        "Avg R >= 0.10":
            stats[
                "Avg_R"
            ] >= 0.10,

        "PF_R >= 1.25":
            stats[
                "PF_R"
            ] >= 1.25,

        "Positive years >= 70%":
            (
                positive_years
                / total_years
            ) >= 0.70
    }


    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<25}"
            f"{'PASS' if passed else 'FAIL'}"
        )


    print(
        f"\nScore: "
        f"{sum(checks.values())}/"
        f"{len(checks)}"
    )


    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "v24_all_signals.csv",
        index=False
    )


    yearly.to_csv(
        "v24_yearly.csv"
    )


    periods.to_csv(
        "v24_periods.csv"
    )


    exits.to_csv(
        "v24_exits.csv"
    )


    holding.to_csv(
        "v24_holding.csv"
    )


    concurrency[
        "Daily"
    ].to_csv(
        "v24_concurrency.csv",
        index=False
    )


    print()

    print(
        "=" * 120
    )

    print(
        "V24 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
