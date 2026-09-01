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
# ACCOUNT SETTINGS
# ============================================================

STARTING_CAPITAL = 1000.0

# 0.5% account risk per trade
RISK_PER_TRADE = 0.005

# Maximum simultaneous positions
MAX_OPEN_POSITIONS = 5

# Fractional shares supported
MIN_SHARE_SIZE = 0.0001


# ============================================================
# REALISTIC EXECUTION COSTS
# ============================================================

# 0.05% worse execution each side
SLIPPAGE_PCT = 0.0005

# 0.02% transaction cost each side
COMMISSION_PCT = 0.0002


# ============================================================
# V6 STRATEGY
# ============================================================

SWING_LOOKBACK = 3

ATR_PERIOD = 14

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
# SWING LOW
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
# FIND ENTRY
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

        high = float(
            df["High"].iloc[i]
        )

        if high >= entry:
            return i

    return None


# ============================================================
# FIXED 2R TRADE SIMULATION
# ============================================================

def simulate_fixed_2r(
    df,
    entry_index,
    entry,
    sl
):

    risk = entry - sl

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

        # Conservative:
        # SL and TP same daily candle -> SL first

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

    exit_price = float(
        df["Close"].iloc[
            last_index
        ]
    )

    return {
        "Exit_Index": last_index,
        "Exit_Price": exit_price,
        "Result": "TIME"
    }


# ============================================================
# GET QQQ MARKET REGIME
# ============================================================

def get_qqq_regime_data():

    print(
        "Downloading QQQ market-regime data..."
    )

    qqq = yf.download(
        "QQQ",
        period="5y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        repair=True
    )

    if qqq.empty:
        raise RuntimeError(
            "QQQ download failed."
        )

    # Handle possible MultiIndex
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

    qqq["Regime"] = np.where(
        qqq["Close"]
        >
        qqq["SMA200"],
        "BULL",
        "BEAR"
    )

    return qqq


# ============================================================
# MARKET REGIME FOR DATE
# ============================================================

def get_regime_for_date(
    qqq,
    date
):

    available = qqq[
        qqq.index <= date
    ]

    if available.empty:
        return "UNKNOWN"

    latest = available.iloc[-1]

    if pd.isna(
        latest["SMA200"]
    ):
        return "UNKNOWN"

    return latest["Regime"]


# ============================================================
# GENERATE ALL V6 TRADE CANDIDATES
# ============================================================

def generate_candidates(
    all_data,
    qqq
):

    candidates = []

    rejected_trend = 0
    rejected_candle = 0
    rejected_stop = 0
    expired = 0

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
                    df["Close"],
                    14
                )
            )

            df["ATR14"] = (
                calculate_atr(
                    df,
                    ATR_PERIOD
                )
            )

            # =================================================
            # WALK FORWARD
            # =================================================

            signal_index = 210

            while signal_index < len(df) - 1:

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
                # TREND FILTERS
                # =================================================

                if close <= sma200:

                    rejected_trend += 1

                    signal_index += 1
                    continue

                if ema20 <= sma50:

                    rejected_trend += 1

                    signal_index += 1
                    continue

                if sma50 <= old_sma50:

                    rejected_trend += 1

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

                touched_ema20 = (
                    ema20
                    * (1 - SUPPORT_TOLERANCE)
                    <= low
                    <=
                    ema20
                    * (1 + SUPPORT_TOLERANCE)
                )

                touched_sma50 = (
                    sma50
                    * (1 - SUPPORT_TOLERANCE)
                    <= low
                    <=
                    sma50
                    * (1 + SUPPORT_TOLERANCE)
                )

                if not (
                    touched_ema20
                    or touched_sma50
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
                # BULLISH CANDLE
                # =================================================

                if close <= open_price:

                    rejected_candle += 1

                    signal_index += 1
                    continue

                candle_range = (
                    high - low
                )

                if candle_range <= 0:

                    signal_index += 1
                    continue

                midpoint = (
                    low
                    + 0.5 * candle_range
                )

                if close <= midpoint:

                    rejected_candle += 1

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
                # STOP
                # =================================================

                sl = (
                    swing_low
                    -
                    SL_ATR_BUFFER
                    * atr
                )

                technical_risk = (
                    entry - sl
                )

                if technical_risk <= 0:

                    signal_index += 1
                    continue

                if technical_risk < (
                    MIN_STOP_ATR
                    * atr
                ):

                    rejected_stop += 1

                    signal_index += 1
                    continue

                # =================================================
                # ACTUAL ENTRY
                # =================================================

                entry_index = (
                    find_entry(
                        df,
                        signal_index,
                        entry
                    )
                )

                if entry_index is None:

                    expired += 1

                    signal_index += (
                        ENTRY_VALID_DAYS
                        + 1
                    )

                    continue

                # =================================================
                # SIMULATE FIXED 2R
                # =================================================

                result = (
                    simulate_fixed_2r(
                        df,
                        entry_index,
                        entry,
                        sl
                    )
                )

                signal_date = pd.Timestamp(
                    df.index[
                        signal_index
                    ]
                )

                entry_date = pd.Timestamp(
                    df.index[
                        entry_index
                    ]
                )

                exit_date = pd.Timestamp(
                    df.index[
                        result["Exit_Index"]
                    ]
                )

                regime = (
                    get_regime_for_date(
                        qqq,
                        signal_date
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

                        "Raw_SL":
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

                        "Regime":
                            regime
                    }
                )

                # No overlapping candidate trades
                # for the same ticker

                signal_index = (
                    result[
                        "Exit_Index"
                    ]
                    + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

    diagnostics = {
        "Rejected_Trend":
            rejected_trend,

        "Rejected_Candle":
            rejected_candle,

        "Rejected_Stop":
            rejected_stop,

        "Expired":
            expired
    }

    return (
        pd.DataFrame(
            candidates
        ),
        diagnostics
    )


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(
    candidates
):

    candidates = (
        candidates
        .sort_values(
            [
                "Entry_Date",
                "Ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    cash = STARTING_CAPITAL

    # Realized account equity
    realized_equity = STARTING_CAPITAL

    open_positions = []

    completed = []

    skipped_slots = 0
    skipped_cash = 0

    # ========================================================
    # CLOSE POSITIONS BEFORE NEW ENTRY DATE
    # ========================================================

    def close_old_positions(
        current_date
    ):

        nonlocal cash
        nonlocal realized_equity
        nonlocal open_positions
        nonlocal completed

        still_open = []

        # Conservative:
        # position exiting ON current entry date
        # is considered still open.
        #
        # We only free capital if Exit_Date < current_date.

        closing = [
            p
            for p in open_positions
            if p["Exit_Date"] < current_date
        ]

        still_open = [
            p
            for p in open_positions
            if p["Exit_Date"] >= current_date
        ]

        closing = sorted(
            closing,
            key=lambda x: x[
                "Exit_Date"
            ]
        )

        for position in closing:

            cash += position[
                "Net_Exit_Proceeds"
            ]

            realized_equity += position[
                "Net_PnL"
            ]

            position[
                "Account_After"
            ] = realized_equity

            completed.append(
                position
            )

        open_positions = (
            still_open
        )

    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    for _, trade in candidates.iterrows():

        entry_date = trade[
            "Entry_Date"
        ]

        close_old_positions(
            entry_date
        )

        # ====================================================
        # MAX POSITIONS
        # ====================================================

        if (
            len(open_positions)
            >= MAX_OPEN_POSITIONS
        ):

            skipped_slots += 1
            continue

        raw_entry = float(
            trade["Raw_Entry"]
        )

        raw_sl = float(
            trade["Raw_SL"]
        )

        raw_exit = float(
            trade["Raw_Exit"]
        )

        # ====================================================
        # SLIPPAGE
        # ====================================================

        # Buy slightly more expensive
        entry_fill = (
            raw_entry
            * (1 + SLIPPAGE_PCT)
        )

        # Stop execution slightly worse
        stop_fill = (
            raw_sl
            * (1 - SLIPPAGE_PCT)
        )

        # Sell slightly lower
        exit_fill = (
            raw_exit
            * (1 - SLIPPAGE_PCT)
        )

        # ====================================================
        # RISK PER SHARE
        # ====================================================

        risk_per_share = (
            entry_fill
            - stop_fill
        )

        if risk_per_share <= 0:
            continue

        # ====================================================
        # ACCOUNT RISK
        # ====================================================

        risk_budget = (
            realized_equity
            * RISK_PER_TRADE
        )

        shares_by_risk = (
            risk_budget
            / risk_per_share
        )

        # ====================================================
        # AVAILABLE CASH LIMIT
        # ====================================================

        # Need enough cash for entry
        # plus entry commission.

        effective_entry_cost = (
            entry_fill
            * (1 + COMMISSION_PCT)
        )

        max_shares_by_cash = (
            cash
            / effective_entry_cost
        )

        shares = min(
            shares_by_risk,
            max_shares_by_cash
        )

        # fractional shares
        shares = np.floor(
            shares
            / MIN_SHARE_SIZE
        ) * MIN_SHARE_SIZE

        if shares < MIN_SHARE_SIZE:

            skipped_cash += 1
            continue

        # ====================================================
        # ENTRY COST
        # ====================================================

        entry_notional = (
            shares
            * entry_fill
        )

        entry_commission = (
            entry_notional
            * COMMISSION_PCT
        )

        total_entry_cost = (
            entry_notional
            + entry_commission
        )

        if total_entry_cost > cash:

            skipped_cash += 1
            continue

        cash -= total_entry_cost

        # ====================================================
        # EXIT VALUE
        # ====================================================

        exit_notional = (
            shares
            * exit_fill
        )

        exit_commission = (
            exit_notional
            * COMMISSION_PCT
        )

        net_exit_proceeds = (
            exit_notional
            - exit_commission
        )

        # ====================================================
        # REAL NET P&L
        # ====================================================

        net_pnl = (
            net_exit_proceeds
            - total_entry_cost
        )

        actual_risk_dollars = (
            shares
            * risk_per_share
        )

        net_r = (
            net_pnl
            / actual_risk_dollars
            if actual_risk_dollars > 0
            else 0
        )

        position = {
            "Ticker":
                trade["Ticker"],

            "Signal_Date":
                trade["Signal_Date"],

            "Entry_Date":
                trade["Entry_Date"],

            "Exit_Date":
                trade["Exit_Date"],

            "Regime":
                trade["Regime"],

            "Result":
                trade["Result"],

            "Shares":
                shares,

            "Raw_Entry":
                raw_entry,

            "Entry_Fill":
                entry_fill,

            "Stop_Fill":
                stop_fill,

            "Raw_Exit":
                raw_exit,

            "Exit_Fill":
                exit_fill,

            "Risk_Budget":
                risk_budget,

            "Actual_Risk":
                actual_risk_dollars,

            "Entry_Notional":
                entry_notional,

            "Entry_Commission":
                entry_commission,

            "Exit_Commission":
                exit_commission,

            "Net_PnL":
                net_pnl,

            "Net_R":
                net_r,

            "Net_Exit_Proceeds":
                net_exit_proceeds,

            "Account_After":
                None
        }

        open_positions.append(
            position
        )

    # ========================================================
    # CLOSE REMAINING POSITIONS
    # ========================================================

    if open_positions:

        last_date = max(
            p["Exit_Date"]
            for p in open_positions
        )

        close_old_positions(
            last_date
            + pd.Timedelta(
                days=1
            )
        )

    completed = pd.DataFrame(
        completed
    )

    completed = (
        completed
        .sort_values(
            [
                "Exit_Date",
                "Ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        completed,
        skipped_slots,
        skipped_cash
    )


# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def calculate_portfolio_stats(
    trades
):

    if trades.empty:
        return None

    ending_capital = (
        STARTING_CAPITAL
        +
        trades[
            "Net_PnL"
        ].sum()
    )

    total_return_pct = (
        (
            ending_capital
            / STARTING_CAPITAL
        )
        - 1
    ) * 100

    first_date = pd.Timestamp(
        trades[
            "Entry_Date"
        ].min()
    )

    last_date = pd.Timestamp(
        trades[
            "Exit_Date"
        ].max()
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
        and ending_capital > 0
    ):

        cagr = (
            (
                ending_capital
                / STARTING_CAPITAL
            )
            ** (1 / years)
            - 1
        ) * 100

    else:

        cagr = 0

    # ========================================================
    # REALIZED EQUITY CURVE
    # ========================================================

    equity = (
        STARTING_CAPITAL
        +
        trades[
            "Net_PnL"
        ].cumsum()
    )

    peak = equity.cummax()

    drawdown_pct = (
        (
            equity
            / peak
        )
        - 1
    ) * 100

    max_drawdown_pct = (
        drawdown_pct.min()
    )

    profitable = (
        trades[
            "Net_PnL"
        ] > 0
    )

    win_rate = (
        profitable.mean()
        * 100
    )

    positive = trades.loc[
        trades[
            "Net_PnL"
        ] > 0,
        "Net_PnL"
    ]

    negative = trades.loc[
        trades[
            "Net_PnL"
        ] < 0,
        "Net_PnL"
    ]

    gross_profit = (
        positive.sum()
    )

    gross_loss = abs(
        negative.sum()
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "Starting_Capital":
            STARTING_CAPITAL,

        "Ending_Capital":
            ending_capital,

        "Net_Profit":
            ending_capital
            - STARTING_CAPITAL,

        "Return_%":
            total_return_pct,

        "CAGR_%":
            cagr,

        "Trades":
            len(trades),

        "Profitable_%":
            win_rate,

        "Average_Net_R":
            trades[
                "Net_R"
            ].mean(),

        "Profit_Factor":
            profit_factor,

        "Max_Realized_Drawdown_%":
            max_drawdown_pct,

        "Total_Commissions":
            (
                trades[
                    "Entry_Commission"
                ].sum()
                +
                trades[
                    "Exit_Commission"
                ].sum()
            )
    }


# ============================================================
# MARKET REGIME SUMMARY
# ============================================================

def regime_summary(
    trades
):

    if trades.empty:
        return pd.DataFrame()

    result = (
        trades
        .groupby(
            "Regime"
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

            Avg_Net_PnL=(
                "Net_PnL",
                "mean"
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

    return result


# ============================================================
# YEAR SUMMARY
# ============================================================

def yearly_summary(
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
        ).dt.year
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

    print("=" * 105)

    print(
        "PULLBACK BACKTEST V7 - REALISTIC PORTFOLIO SIMULATION"
    )

    print("=" * 105)

    print()

    print(
        f"Starting capital:          ${STARTING_CAPITAL:,.2f}"
    )

    print(
        f"Risk per trade:            {RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Initial risk/trade:        ${STARTING_CAPITAL * RISK_PER_TRADE:.2f}"
    )

    print(
        f"Maximum open positions:    {MAX_OPEN_POSITIONS}"
    )

    print(
        f"Slippage per side:         {SLIPPAGE_PCT * 100:.3f}%"
    )

    print(
        f"Transaction cost/side:     {COMMISSION_PCT * 100:.3f}%"
    )

    print()

    # ========================================================
    # DOWNLOAD STOCK DATA
    # ========================================================

    print(
        f"Downloading {len(TICKERS)} stocks..."
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

    # ========================================================
    # QQQ
    # ========================================================

    try:

        qqq = (
            get_qqq_regime_data()
        )

    except Exception as e:

        print(
            f"QQQ ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    # ========================================================
    # GENERATE SIGNALS
    # ========================================================

    candidates, diagnostics = (
        generate_candidates(
            all_data,
            qqq
        )
    )

    if candidates.empty:

        print(
            "NO CANDIDATES FOUND."
        )

        return

    print()

    print(
        f"Valid V6 candidates:        {len(candidates)}"
    )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    (
        portfolio_trades,
        skipped_slots,
        skipped_cash
    ) = simulate_portfolio(
        candidates
    )

    if portfolio_trades.empty:

        print(
            "NO PORTFOLIO TRADES EXECUTED."
        )

        return

    # ========================================================
    # RESULTS
    # ========================================================

    stats = (
        calculate_portfolio_stats(
            portfolio_trades
        )
    )

    print()

    print("=" * 105)
    print("V7 - REAL ACCOUNT RESULTS")
    print("=" * 105)

    print(
        f"Starting capital:             ${stats['Starting_Capital']:,.2f}"
    )

    print(
        f"Ending capital:               ${stats['Ending_Capital']:,.2f}"
    )

    print(
        f"Net profit:                   ${stats['Net_Profit']:,.2f}"
    )

    print(
        f"Total return:                 {stats['Return_%']:.2f}%"
    )

    print(
        f"CAGR:                         {stats['CAGR_%']:.2f}%"
    )

    print()

    print(
        f"Executed trades:              {stats['Trades']}"
    )

    print(
        f"Profitable trades:            {stats['Profitable_%']:.2f}%"
    )

    print(
        f"Average net R/trade:          {stats['Average_Net_R']:.3f}R"
    )

    print(
        f"Profit factor:                {stats['Profit_Factor']:.3f}"
    )

    print(
        f"Max realized drawdown:        {stats['Max_Realized_Drawdown_%']:.2f}%"
    )

    print(
        f"Total simulated commissions:  ${stats['Total_Commissions']:.2f}"
    )

    # ========================================================
    # PORTFOLIO LIMITS
    # ========================================================

    print()

    print("=" * 105)
    print("PORTFOLIO CONSTRAINTS")
    print("=" * 105)

    print(
        f"Candidates generated:         {len(candidates)}"
    )

    print(
        f"Trades executed:              {len(portfolio_trades)}"
    )

    print(
        f"Skipped - max positions:      {skipped_slots}"
    )

    print(
        f"Skipped - insufficient cash:  {skipped_cash}"
    )

    # ========================================================
    # MARKET REGIME
    # ========================================================

    regime = (
        regime_summary(
            portfolio_trades
        )
    )

    print()

    print("=" * 105)
    print("QQQ MARKET REGIME")
    print("=" * 105)

    print(
        regime
        .round(3)
        .to_string()
    )

    # ========================================================
    # YEAR BY YEAR
    # ========================================================

    yearly = (
        yearly_summary(
            portfolio_trades
        )
    )

    print()

    print("=" * 105)
    print("YEAR-BY-YEAR REAL ACCOUNT")
    print("=" * 105)

    print(
        yearly
        .round(3)
        .to_string()
    )

    # ========================================================
    # SIGNAL DIAGNOSTICS
    # ========================================================

    print()

    print("=" * 105)
    print("SIGNAL DIAGNOSTICS")
    print("=" * 105)

    print(
        f"Rejected trend:              {diagnostics['Rejected_Trend']}"
    )

    print(
        f"Rejected candle:             {diagnostics['Rejected_Candle']}"
    )

    print(
        f"Rejected stop:               {diagnostics['Rejected_Stop']}"
    )

    print(
        f"Expired setups:              {diagnostics['Expired']}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    portfolio_trades.to_csv(
        "backtest_v7_trades.csv",
        index=False
    )

    candidates.to_csv(
        "backtest_v7_candidates.csv",
        index=False
    )

    regime.to_csv(
        "backtest_v7_regime.csv"
    )

    yearly.to_csv(
        "backtest_v7_yearly.csv"
    )

    print()

    print("=" * 105)

    print("Saved:")
    print("  backtest_v7_trades.csv")
    print("  backtest_v7_candidates.csv")
    print("  backtest_v7_regime.csv")
    print("  backtest_v7_yearly.csv")

    print()

    print(
        "Pullback V7 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
