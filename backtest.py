import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# NASDAQ-100 UNIVERSE
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
# ACCOUNT SETTINGS
# ============================================================

STARTING_CAPITAL = 1000.0

# 0.5% account risk per trade
RISK_PER_TRADE = 0.005

# Compare these portfolio sizes
POSITION_LIMITS = [3, 5]

# Fractional shares
MIN_SHARE_SIZE = 0.0001


# ============================================================
# EXECUTION COSTS
# ============================================================

# 0.05% slippage per side
SLIPPAGE_PCT = 0.0005

# 0.02% simulated transaction cost per side
COMMISSION_PCT = 0.0002


# ============================================================
# BACKTEST WINDOW
# ============================================================

# Download extra history for indicator warm-up
DOWNLOAD_PERIOD = "7y"

# But evaluate only final 5 years
BACKTEST_YEARS = 5


# ============================================================
# STRATEGY PARAMETERS - V6 SIGNAL
# ============================================================

ATR_PERIOD = 14

SWING_LOOKBACK = 3

ENTRY_ATR_BUFFER = 0.10

SL_ATR_BUFFER = 0.10

MIN_STOP_ATR = 0.50

ENTRY_VALID_DAYS = 3

MAX_HOLDING_DAYS = 10

SMA50_SLOPE_LOOKBACK = 10

RSI_MIN = 35
RSI_MAX = 55

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

def find_previous_swing_low(
    df,
    signal_index
):

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
            i + 1:
            i + SWING_LOOKBACK + 1
        ]

        if (
            current < float(left.min())
            and
            current < float(right.min())
        ):
            return current

    return None


# ============================================================
# ENTRY ACTIVATION
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

        if float(
            df["High"].iloc[i]
        ) >= entry:

            return i

    return None


# ============================================================
# FIXED 2R EXIT
# ============================================================

def simulate_fixed_2r(
    df,
    entry_index,
    entry,
    sl
):

    risk = (
        entry - sl
    )

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

        # Conservative daily-bar assumption:
        # if SL and TP both appear, assume SL first.

        if low <= sl:

            return {
                "Exit_Index": i,
                "Exit_Price": sl,
                "Result": "SL"
            }

        if high >= target:

            return {
                "Exit_Index": i,
                "Exit_Price": target,
                "Result": "TP"
            }

    return {
        "Exit_Index": last_index,

        "Exit_Price": float(
            df["Close"].iloc[
                last_index
            ]
        ),

        "Result": "TIME"
    }


# ============================================================
# PREPARE QQQ
# ============================================================

def prepare_qqq():

    print(
        "Downloading QQQ regime data..."
    )

    qqq = yf.download(
        "QQQ",
        period=DOWNLOAD_PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=True,
        repair=True
    )

    if qqq.empty:

        raise RuntimeError(
            "QQQ data unavailable."
        )

    if isinstance(
        qqq.columns,
        pd.MultiIndex
    ):

        qqq.columns = (
            qqq.columns
            .get_level_values(0)
        )

    qqq["SMA200"] = (
        qqq["Close"]
        .rolling(200)
        .mean()
    )

    qqq["BULL"] = (
        qqq["Close"]
        >
        qqq["SMA200"]
    )

    return qqq


# ============================================================
# QQQ BULL CHECK
# ============================================================

def qqq_is_bull(
    qqq,
    date
):

    if date not in qqq.index:

        available = (
            qqq.loc[
                qqq.index <= date
            ]
        )

        if available.empty:
            return False

        row = available.iloc[-1]

    else:

        row = qqq.loc[
            date
        ]

    if pd.isna(
        row["SMA200"]
    ):
        return False

    return bool(
        row["BULL"]
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    close,
    high,
    low,
    ema20,
    sma50,
    old_sma50,
    atr
):

    # --------------------------------------------------------
    # 1. SMA50 slope normalized by ATR
    # --------------------------------------------------------

    slope_score = (
        sma50
        - old_sma50
    ) / atr

    # --------------------------------------------------------
    # 2. EMA20 distance above SMA50 normalized by ATR
    # --------------------------------------------------------

    ema_strength = (
        ema20
        - sma50
    ) / atr

    # --------------------------------------------------------
    # 3. Close location inside candle
    #
    # 0 = bottom
    # 1 = top
    # --------------------------------------------------------

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        close_location = 0

    else:

        close_location = (
            close - low
        ) / candle_range

    # Equal-weight transparent score
    quality_score = (
        slope_score
        + ema_strength
        + close_location
    )

    return {
        "Quality_Score":
            quality_score,

        "Slope_Score":
            slope_score,

        "EMA_Strength":
            ema_strength,

        "Close_Location":
            close_location
    }


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    all_data,
    qqq,
    backtest_start
):

    candidates = []

    diagnostics = {
        "Rejected_Trend": 0,
        "Rejected_Candle": 0,
        "Rejected_Stop": 0,
        "Expired": 0,
        "Rejected_Bear_Regime": 0
    }

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
                    df["Close"]
                )
            )

            df["ATR14"] = (
                calculate_atr(
                    df
                )
            )

            # =================================================
            # WALK THROUGH HISTORY
            # =================================================

            signal_index = 210

            while (
                signal_index
                < len(df) - 1
            ):

                signal_date = (
                    pd.Timestamp(
                        df.index[
                            signal_index
                        ]
                    )
                )

                # Warm-up data only
                if (
                    signal_date
                    < backtest_start
                ):

                    signal_index += 1
                    continue

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
                # TREND
                # =================================================

                if close <= sma200:

                    diagnostics[
                        "Rejected_Trend"
                    ] += 1

                    signal_index += 1
                    continue

                if ema20 <= sma50:

                    diagnostics[
                        "Rejected_Trend"
                    ] += 1

                    signal_index += 1
                    continue

                if sma50 <= old_sma50:

                    diagnostics[
                        "Rejected_Trend"
                    ] += 1

                    signal_index += 1
                    continue

                # =================================================
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
                # PULLBACK
                # =================================================

                touched_ema = (
                    ema20
                    * (
                        1
                        - SUPPORT_TOLERANCE
                    )
                    <= low
                    <=
                    ema20
                    * (
                        1
                        + SUPPORT_TOLERANCE
                    )
                )

                touched_sma = (
                    sma50
                    * (
                        1
                        - SUPPORT_TOLERANCE
                    )
                    <= low
                    <=
                    sma50
                    * (
                        1
                        + SUPPORT_TOLERANCE
                    )
                )

                if not (
                    touched_ema
                    or touched_sma
                ):

                    signal_index += 1
                    continue

                # =================================================
                # CLOSE ABOVE SUPPORT
                # =================================================

                if not (
                    close >= ema20
                    or close >= sma50
                ):

                    signal_index += 1
                    continue

                # =================================================
                # BULLISH SIGNAL CANDLE
                # =================================================

                if close <= open_price:

                    diagnostics[
                        "Rejected_Candle"
                    ] += 1

                    signal_index += 1
                    continue

                candle_range = (
                    high - low
                )

                if candle_range <= 0:

                    signal_index += 1
                    continue

                candle_midpoint = (
                    low
                    + 0.50
                    * candle_range
                )

                if close <= candle_midpoint:

                    diagnostics[
                        "Rejected_Candle"
                    ] += 1

                    signal_index += 1
                    continue

                # =================================================
                # ATR ENTRY
                # =================================================

                entry = (
                    high
                    +
                    ENTRY_ATR_BUFFER
                    * atr
                )

                # =================================================
                # SWING LOW / STOP
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

                if risk < (
                    MIN_STOP_ATR
                    * atr
                ):

                    diagnostics[
                        "Rejected_Stop"
                    ] += 1

                    signal_index += 1
                    continue

                # =================================================
                # ENTRY ACTIVATION
                # =================================================

                entry_index = (
                    find_entry(
                        df,
                        signal_index,
                        entry
                    )
                )

                if entry_index is None:

                    diagnostics[
                        "Expired"
                    ] += 1

                    signal_index += (
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
                # QQQ BULL FILTER
                #
                # Important:
                # check regime on ACTUAL entry date.
                # =================================================

                if not qqq_is_bull(
                    qqq,
                    entry_date
                ):

                    diagnostics[
                        "Rejected_Bear_Regime"
                    ] += 1

                    signal_index = (
                        entry_index + 1
                    )

                    continue

                # =================================================
                # EXIT
                # =================================================

                result = (
                    simulate_fixed_2r(
                        df,
                        entry_index,
                        entry,
                        sl
                    )
                )

                exit_index = (
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

                # =================================================
                # QUALITY SCORE
                # =================================================

                score = (
                    calculate_quality_score(
                        close,
                        high,
                        low,
                        ema20,
                        sma50,
                        old_sma50,
                        atr
                    )
                )

                candidates.append(
                    {
                        "Ticker":
                            ticker,

                        "Signal_Date":
                            signal_date,

                        "Entry_Date":
                            entry_date,

                        "Exit_Date":
                            exit_date,

                        "Raw_Entry":
                            entry,

                        "Raw_Stop":
                            sl,

                        "Raw_Exit":
                            result[
                                "Exit_Price"
                            ],

                        "Result":
                            result[
                                "Result"
                            ],

                        "ATR":
                            atr,

                        "RSI":
                            rsi,

                        "Quality_Score":
                            score[
                                "Quality_Score"
                            ],

                        "Slope_Score":
                            score[
                                "Slope_Score"
                            ],

                        "EMA_Strength":
                            score[
                                "EMA_Strength"
                            ],

                        "Close_Location":
                            score[
                                "Close_Location"
                            ]
                    }
                )

                # No overlapping trade candidates
                # in SAME ticker

                signal_index = (
                    exit_index + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

    candidates = (
        pd.DataFrame(
            candidates
        )
    )

    if not candidates.empty:

        candidates = (
            candidates
            .sort_values(
                [
                    "Entry_Date",
                    "Quality_Score"
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
        candidates,
        diagnostics
    )


# ============================================================
# GET STOCK CLOSE FOR MTM
# ============================================================

def get_stock_close(
    all_data,
    ticker,
    date
):

    try:

        df = all_data[
            ticker
        ]

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

        previous = df.loc[
            df.index <= date
        ]

        if previous.empty:
            return None

        value = (
            previous[
                "Close"
            ].dropna()
        )

        if value.empty:
            return None

        return float(
            value.iloc[-1]
        )

    except Exception:

        return None


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(
    candidates,
    all_data,
    qqq,
    max_positions,
    backtest_start
):

    if candidates.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {}
        )

    # ========================================================
    # GROUP ENTRIES BY DATE
    # ========================================================

    grouped_entries = {
        date: group.sort_values(
            "Quality_Score",
            ascending=False
        )
        for date, group
        in candidates.groupby(
            "Entry_Date"
        )
    }

    # ========================================================
    # DAILY CALENDAR
    # ========================================================

    final_date = max(
        candidates[
            "Exit_Date"
        ].max(),
        candidates[
            "Entry_Date"
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

    open_positions = []

    completed_trades = []

    equity_rows = []

    skipped_slots = 0
    skipped_cash = 0

    # Starting reference equity
    last_equity = (
        STARTING_CAPITAL
    )

    # ========================================================
    # EACH TRADING DAY
    # ========================================================

    for date in calendar:

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # NEW ENTRIES
        #
        # IMPORTANT:
        # Positions scheduled to exit today still count
        # against limits until end of day.
        #
        # Conservative assumption.
        # ====================================================

        if date in grouped_entries:

            todays_candidates = (
                grouped_entries[
                    date
                ]
            )

            # Account risk is based on
            # previous day's equity.

            risk_budget = (
                last_equity
                * RISK_PER_TRADE
            )

            for _, trade in (
                todays_candidates
                .iterrows()
            ):

                # --------------------------------------------
                # Position limit
                # --------------------------------------------

                if (
                    len(
                        open_positions
                    )
                    >= max_positions
                ):

                    skipped_slots += 1

                    continue

                raw_entry = float(
                    trade[
                        "Raw_Entry"
                    ]
                )

                raw_stop = float(
                    trade[
                        "Raw_Stop"
                    ]
                )

                raw_exit = float(
                    trade[
                        "Raw_Exit"
                    ]
                )

                # --------------------------------------------
                # Slippage
                # --------------------------------------------

                entry_fill = (
                    raw_entry
                    * (
                        1
                        + SLIPPAGE_PCT
                    )
                )

                stop_fill = (
                    raw_stop
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                exit_fill = (
                    raw_exit
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

                # --------------------------------------------
                # Shares based on account risk
                # --------------------------------------------

                shares_by_risk = (
                    risk_budget
                    / risk_per_share
                )

                # --------------------------------------------
                # Cash limitation
                # --------------------------------------------

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

                shares = np.floor(
                    shares
                    / MIN_SHARE_SIZE
                ) * MIN_SHARE_SIZE

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

                position = {
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

                    "Quality_Score":
                        trade[
                            "Quality_Score"
                        ],

                    "Shares":
                        shares,

                    "Entry_Fill":
                        entry_fill,

                    "Stop_Fill":
                        stop_fill,

                    "Exit_Fill":
                        exit_fill,

                    "Entry_Cost":
                        entry_cost,

                    "Entry_Commission":
                        entry_commission,

                    "Risk_Budget":
                        risk_budget,

                    "Risk_Per_Share":
                        risk_per_share
                }

                open_positions.append(
                    position
                )

        # ====================================================
        # CLOSE POSITIONS WHOSE EXIT DATE IS TODAY
        # ====================================================

        remaining = []

        for position in (
            open_positions
        ):

            if (
                position[
                    "Exit_Date"
                ]
                == date
            ):

                exit_notional = (
                    position[
                        "Shares"
                    ]
                    *
                    position[
                        "Exit_Fill"
                    ]
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
                    "Exit_Commission"
                ] = (
                    exit_commission
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

                completed_trades.append(
                    position
                )

            else:

                remaining.append(
                    position
                )

        open_positions = (
            remaining
        )

        # ====================================================
        # MARK-TO-MARKET OPEN POSITIONS
        # ====================================================

        market_value = 0.0

        for position in (
            open_positions
        ):

            close_price = (
                get_stock_close(
                    all_data,
                    position[
                        "Ticker"
                    ],
                    date
                )
            )

            if close_price is None:

                close_price = (
                    position[
                        "Entry_Fill"
                    ]
                )

            market_value += (
                position[
                    "Shares"
                ]
                * close_price
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

                "Open_Market_Value":
                    market_value,

                "Equity":
                    equity,

                "Open_Positions":
                    len(
                        open_positions
                    )
            }
        )

    trades_df = pd.DataFrame(
        completed_trades
    )

    equity_df = pd.DataFrame(
        equity_rows
    )

    diagnostics = {
        "Skipped_Max_Positions":
            skipped_slots,

        "Skipped_Insufficient_Cash":
            skipped_cash
    }

    return (
        trades_df,
        equity_df,
        diagnostics
    )


# ============================================================
# PORTFOLIO STATS
# ============================================================

def portfolio_stats(
    trades,
    equity
):

    if (
        trades.empty
        or equity.empty
    ):

        return {}

    starting = (
        STARTING_CAPITAL
    )

    ending = float(
        equity[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        (
            ending
            / starting
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

    if (
        years > 0
        and ending > 0
    ):

        cagr = (
            (
                ending
                / starting
            )
            ** (
                1 / years
            )
            - 1
        ) * 100

    else:

        cagr = 0

    # ========================================================
    # TRUE MARK-TO-MARKET DRAWDOWN
    # ========================================================

    curve = (
        equity[
            "Equity"
        ]
    )

    peak = (
        curve.cummax()
    )

    dd = (
        curve
        / peak
        - 1
    ) * 100

    max_dd = float(
        dd.min()
    )

    positive = trades[
        trades[
            "Net_PnL"
        ] > 0
    ]

    negative = trades[
        trades[
            "Net_PnL"
        ] < 0
    ]

    gross_profit = (
        positive[
            "Net_PnL"
        ].sum()
    )

    gross_loss = abs(
        negative[
            "Net_PnL"
        ].sum()
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else np.inf
    )

    total_commissions = (
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
            starting,

        "Ending_Capital":
            ending,

        "Net_Profit":
            ending - starting,

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

        "Profit_Factor":
            profit_factor,

        "Max_MTM_Drawdown_%":
            max_dd,

        "Total_Commissions":
            total_commissions,

        "Average_Quality_Score":
            trades[
                "Quality_Score"
            ].mean()
    }


# ============================================================
# YEARLY PERFORMANCE
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
            ),

            Avg_Quality=(
                "Quality_Score",
                "mean"
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print("PULLBACK BACKTEST V8 - BULL ONLY + RANKING + TRUE MTM PORTFOLIO")
    print("=" * 110)

    print()

    print(
        f"Starting capital:         ${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Risk per trade:           {RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Position limits tested:   {POSITION_LIMITS}"
    )

    print(
        f"Slippage per side:        {SLIPPAGE_PCT * 100:.3f}%"
    )

    print(
        f"Commission per side:      {COMMISSION_PCT * 100:.3f}%"
    )

    print()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    print(
        f"Downloading {len(TICKERS)} stocks..."
    )

    try:

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

    except Exception as e:

        print(
            f"DOWNLOAD ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    # ========================================================
    # QQQ
    # ========================================================

    try:

        qqq = prepare_qqq()

    except Exception as e:

        print(
            f"QQQ ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    # ========================================================
    # TRUE BACKTEST START AFTER WARM-UP
    # ========================================================

    latest_date = pd.Timestamp(
        qqq.index.max()
    )

    backtest_start = (
        latest_date
        - pd.DateOffset(
            years=BACKTEST_YEARS
        )
    )

    print(
        f"Backtest start:           {backtest_start.date()}"
    )

    print(
        f"Backtest end:             {latest_date.date()}"
    )

    print()

    # ========================================================
    # CANDIDATES
    # ========================================================

    (
        candidates,
        signal_diagnostics
    ) = generate_candidates(
        all_data,
        qqq,
        backtest_start
    )

    if candidates.empty:

        print(
            "NO CANDIDATES FOUND."
        )

        return

    print(
        f"Bull-regime candidates:   {len(candidates)}"
    )

    # ========================================================
    # RUN BOTH PORTFOLIOS
    # ========================================================

    summaries = []

    portfolio_outputs = {}

    for max_positions in (
        POSITION_LIMITS
    ):

        print()

        print(
            "=" * 110
        )

        print(
            f"SIMULATING MAX {max_positions} POSITIONS"
        )

        print(
            "=" * 110
        )

        (
            trades,
            equity,
            portfolio_diag
        ) = simulate_portfolio(
            candidates,
            all_data,
            qqq,
            max_positions,
            backtest_start
        )

        if (
            trades.empty
            or equity.empty
        ):

            print(
                "No portfolio trades."
            )

            continue

        stats = (
            portfolio_stats(
                trades,
                equity
            )
        )

        stats[
            "Max_Positions"
        ] = (
            max_positions
        )

        stats[
            "Skipped_Max_Positions"
        ] = (
            portfolio_diag[
                "Skipped_Max_Positions"
            ]
        )

        stats[
            "Skipped_Cash"
        ] = (
            portfolio_diag[
                "Skipped_Insufficient_Cash"
            ]
        )

        summaries.append(
            stats
        )

        yearly = (
            yearly_performance(
                trades
            )
        )

        portfolio_outputs[
            max_positions
        ] = {
            "trades":
                trades,

            "equity":
                equity,

            "yearly":
                yearly
        }

        # ====================================================
        # PRINT INDIVIDUAL RESULT
        # ====================================================

        print()

        print(
            f"Starting capital:       "
            f"${stats['Starting_Capital']:,.2f}"
        )

        print(
            f"Ending capital:         "
            f"${stats['Ending_Capital']:,.2f}"
        )

        print(
            f"Net profit:             "
            f"${stats['Net_Profit']:,.2f}"
        )

        print(
            f"Total return:           "
            f"{stats['Return_%']:.2f}%"
        )

        print(
            f"CAGR:                   "
            f"{stats['CAGR_%']:.2f}%"
        )

        print()

        print(
            f"Executed trades:        "
            f"{stats['Trades']}"
        )

        print(
            f"Profitable trades:      "
            f"{stats['Profitable_%']:.2f}%"
        )

        print(
            f"Average Net R:          "
            f"{stats['Avg_Net_R']:.3f}R"
        )

        print(
            f"Profit Factor:          "
            f"{stats['Profit_Factor']:.3f}"
        )

        print(
            f"TRUE MTM Drawdown:      "
            f"{stats['Max_MTM_Drawdown_%']:.2f}%"
        )

        print(
            f"Total commissions:      "
            f"${stats['Total_Commissions']:.2f}"
        )

        print()

        print(
            f"Skipped - positions:    "
            f"{stats['Skipped_Max_Positions']}"
        )

        print(
            f"Skipped - cash:         "
            f"{stats['Skipped_Cash']}"
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
    # FINAL 3 VS 5 COMPARISON
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    if not summary_df.empty:

        summary_df = (
            summary_df
            .set_index(
                "Max_Positions"
            )
        )

        columns = [
            "Ending_Capital",
            "Net_Profit",
            "Return_%",
            "CAGR_%",
            "Trades",
            "Profitable_%",
            "Avg_Net_R",
            "Profit_Factor",
            "Max_MTM_Drawdown_%",
            "Total_Commissions",
            "Skipped_Max_Positions",
            "Skipped_Cash"
        ]

        print()

        print(
            "=" * 110
        )

        print(
            "V8 FINAL COMPARISON - 3 VS 5 POSITIONS"
        )

        print(
            "=" * 110
        )

        print(
            summary_df[
                columns
            ]
            .round(3)
            .to_string()
        )

    # ========================================================
    # SIGNAL DIAGNOSTICS
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "SIGNAL DIAGNOSTICS"
    )

    print(
        "=" * 110
    )

    print(
        f"Rejected trend:          "
        f"{signal_diagnostics['Rejected_Trend']}"
    )

    print(
        f"Rejected candle:         "
        f"{signal_diagnostics['Rejected_Candle']}"
    )

    print(
        f"Rejected stop:           "
        f"{signal_diagnostics['Rejected_Stop']}"
    )

    print(
        f"Expired setups:          "
        f"{signal_diagnostics['Expired']}"
    )

    print(
        f"Rejected BEAR regime:    "
        f"{signal_diagnostics['Rejected_Bear_Regime']}"
    )

    print(
        f"Final Bull candidates:   "
        f"{len(candidates)}"
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    candidates.to_csv(
        "backtest_v8_candidates.csv",
        index=False
    )

    if not summary_df.empty:

        summary_df.to_csv(
            "backtest_v8_comparison.csv"
        )

    for max_positions, output in (
        portfolio_outputs.items()
    ):

        output[
            "trades"
        ].to_csv(
            f"backtest_v8_trades_{max_positions}pos.csv",
            index=False
        )

        output[
            "equity"
        ].to_csv(
            f"backtest_v8_equity_{max_positions}pos.csv",
            index=False
        )

        output[
            "yearly"
        ].to_csv(
            f"backtest_v8_yearly_{max_positions}pos.csv"
        )

    print()

    print(
        "=" * 110
    )

    print(
        "V8 completed."
    )

    print(
        "=" * 110
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
