import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# NASDAQ-100 TICKERS
# ============================================================

STATIC_FALLBACK_TICKERS = [
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
# STRATEGY PARAMETERS - V3
# ============================================================

ENTRY_BUFFER = 0.15

SWING_LOOKBACK = 3

MIN_RR = 2.0

ENTRY_VALID_DAYS = 3

MAX_HOLDING_DAYS = 10


# ============================================================
# ATR SETTINGS
# ============================================================

ATR_PERIOD = 14

# Stop placed below Swing Low by 10% of ATR
ATR_SL_BUFFER = 0.10

# Entry -> SL must be at least 50% of ATR
MIN_STOP_ATR = 0.50


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

    high_low = (
        df["High"]
        - df["Low"]
    )

    high_previous_close = (
        df["High"]
        - previous_close
    ).abs()

    low_previous_close = (
        df["Low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr


# ============================================================
# PREVIOUS CONFIRMED SWING LOW
# ============================================================

def find_previous_swing_low(df, signal_index):

    earliest = SWING_LOOKBACK

    latest_candidate = (
        signal_index
        - SWING_LOOKBACK
        - 1
    )

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_low = float(
            df["Low"].iloc[i]
        )

        left_lows = df["Low"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_lows = df["Low"].iloc[
            i + 1:
            i + SWING_LOOKBACK + 1
        ]

        if (
            current_low < float(left_lows.min())
            and
            current_low < float(right_lows.min())
        ):

            return {
                "price": current_low,
                "index": i,
                "date": df.index[i]
            }

    return None


# ============================================================
# PREVIOUS CONFIRMED SWING HIGH
# ============================================================

def find_previous_swing_high(df, signal_index):

    earliest = SWING_LOOKBACK

    latest_candidate = (
        signal_index
        - SWING_LOOKBACK
        - 1
    )

    if latest_candidate < earliest:
        return None

    for i in range(
        latest_candidate,
        earliest - 1,
        -1
    ):

        current_high = float(
            df["High"].iloc[i]
        )

        left_highs = df["High"].iloc[
            i - SWING_LOOKBACK:i
        ]

        right_highs = df["High"].iloc[
            i + 1:
            i + SWING_LOOKBACK + 1
        ]

        if (
            current_high > float(left_highs.max())
            and
            current_high > float(right_highs.max())
        ):

            return {
                "price": current_high,
                "index": i,
                "date": df.index[i]
            }

    return None


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    df,
    signal_index,
    entry,
    sl,
    tp
):

    risk = entry - sl

    if risk <= 0:
        return None

    # ========================================================
    # ENTRY MUST HAPPEN WITHIN 3 TRADING DAYS
    # ========================================================

    first_possible_entry = (
        signal_index + 1
    )

    last_possible_entry = min(
        signal_index + ENTRY_VALID_DAYS,
        len(df) - 1
    )

    entry_index = None

    for i in range(
        first_possible_entry,
        last_possible_entry + 1
    ):

        day_high = float(
            df["High"].iloc[i]
        )

        if day_high >= entry:

            entry_index = i
            break

    # Setup expired
    if entry_index is None:

        return {
            "Result": "NO_ENTRY",
            "R_Multiple": 0.0,
            "Entry_Index": None,
            "Exit_Index": None,
            "Exit_Price": None
        }

    # ========================================================
    # MANAGE OPEN TRADE
    # ========================================================

    last_holding_index = min(
        entry_index
        + MAX_HOLDING_DAYS
        - 1,
        len(df) - 1
    )

    for i in range(
        entry_index,
        last_holding_index + 1
    ):

        day_high = float(
            df["High"].iloc[i]
        )

        day_low = float(
            df["Low"].iloc[i]
        )

        # ====================================================
        # SAME-DAY SL + TP
        #
        # Daily data cannot tell which happened first.
        # Conservative assumption = SL first.
        # ====================================================

        if (
            day_low <= sl
            and
            day_high >= tp
        ):

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": sl
            }

        # ====================================================
        # STOP LOSS
        # ====================================================

        if day_low <= sl:

            return {
                "Result": "LOSS",
                "R_Multiple": -1.0,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": sl
            }

        # ====================================================
        # TAKE PROFIT
        # ====================================================

        if day_high >= tp:

            reward = tp - entry

            r_multiple = (
                reward / risk
            )

            return {
                "Result": "WIN",
                "R_Multiple": r_multiple,
                "Entry_Index": entry_index,
                "Exit_Index": i,
                "Exit_Price": tp
            }

    # ========================================================
    # TIME EXIT
    # ========================================================

    exit_price = float(
        df["Close"].iloc[
            last_holding_index
        ]
    )

    r_multiple = (
        exit_price - entry
    ) / risk

    return {
        "Result": "TIME_EXIT",
        "R_Multiple": r_multiple,
        "Entry_Index": entry_index,
        "Exit_Index": last_holding_index,
        "Exit_Price": exit_price
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PULLBACK SWING TRADING BACKTEST - V3 ATR")
    print("=" * 72)

    print()

    print(
        f"Entry buffer:             ${ENTRY_BUFFER}"
    )

    print(
        f"Entry validity:            {ENTRY_VALID_DAYS} trading days"
    )

    print(
        f"Swing lookback:            {SWING_LOOKBACK}"
    )

    print(
        f"ATR period:                {ATR_PERIOD}"
    )

    print(
        f"ATR SL buffer:             {ATR_SL_BUFFER:.2f} ATR"
    )

    print(
        f"Minimum stop distance:     {MIN_STOP_ATR:.2f} ATR"
    )

    print(
        f"Minimum R:R:               {MIN_RR}"
    )

    print(
        f"Max holding period:        {MAX_HOLDING_DAYS} trading days"
    )

    print()

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    tickers = STATIC_FALLBACK_TICKERS

    print(
        f"Downloading {len(tickers)} tickers..."
    )

    try:

        all_data = yf.download(
            tickers,

            period="5y",

            interval="1d",

            group_by="ticker",

            progress=False,

            # Adjusted OHLC
            auto_adjust=True,

            # Let yfinance attempt to repair
            # known price/data problems
            repair=True,

            threads=True
        )

    except Exception as e:

        print(
            f"DOWNLOAD ERROR: "
            f"{type(e).__name__}: {e}"
        )

        return

    all_trades = []

    rejected_small_stop = 0

    rejected_rr = 0

    expired_setups = 0

    # ========================================================
    # EACH TICKER
    # ========================================================

    for ticker in tickers:

        try:

            if (
                ticker
                not in
                all_data.columns.get_level_values(0)
            ):

                print(
                    f"{ticker}: no data"
                )

                continue

            df = all_data[ticker].copy()

            df.dropna(inplace=True)

            if len(df) < 250:

                print(
                    f"{ticker}: insufficient data"
                )

                continue

            # =================================================
            # INDICATORS
            # =================================================

            df["SMA_200"] = (
                df["Close"]
                .rolling(200)
                .mean()
            )

            df["SMA_50"] = (
                df["Close"]
                .rolling(50)
                .mean()
            )

            df["EMA_20"] = (
                df["Close"]
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
            )

            df["Avg_Vol_20"] = (
                df["Volume"]
                .rolling(20)
                .mean()
            )

            df["RSI_14"] = calculate_rsi(
                df["Close"],
                14
            )

            df["ATR_14"] = calculate_atr(
                df,
                ATR_PERIOD
            )

            # =================================================
            # WALK FORWARD
            # =================================================

            signal_index = 210

            while signal_index < len(df) - 1:

                row = df.iloc[
                    signal_index
                ]

                price = float(
                    row["Close"]
                )

                high = float(
                    row["High"]
                )

                low = float(
                    row["Low"]
                )

                volume = float(
                    row["Volume"]
                )

                avg_vol = float(
                    row["Avg_Vol_20"]
                )

                sma200 = float(
                    row["SMA_200"]
                )

                sma50 = float(
                    row["SMA_50"]
                )

                ema20 = float(
                    row["EMA_20"]
                )

                rsi = float(
                    row["RSI_14"]
                )

                atr = float(
                    row["ATR_14"]
                )

                # =================================================
                # NaN / BAD VALUES
                # =================================================

                values = [
                    price,
                    high,
                    low,
                    volume,
                    avg_vol,
                    sma200,
                    sma50,
                    ema20,
                    rsi,
                    atr
                ]

                if (
                    any(pd.isna(x) for x in values)
                    or atr <= 0
                ):

                    signal_index += 1
                    continue

                # =================================================
                # 1. LONG-TERM TREND
                # =================================================

                if price < sma200:

                    signal_index += 1
                    continue

                # =================================================
                # 2. RSI PULLBACK
                # =================================================

                if not (
                    35 <= rsi <= 55
                ):

                    signal_index += 1
                    continue

                # =================================================
                # 3. PULLBACK TO EMA20 / SMA50
                # =================================================

                touched_ema20 = (
                    ema20 * 0.99
                    <= low
                    <= ema20 * 1.01
                )

                touched_sma50 = (
                    sma50 * 0.99
                    <= low
                    <= sma50 * 1.01
                )

                closed_above = (
                    price >= ema20
                    or
                    price >= sma50
                )

                if not (
                    (
                        touched_ema20
                        or touched_sma50
                    )
                    and
                    closed_above
                ):

                    signal_index += 1
                    continue

                # =================================================
                # VOLUME
                # =================================================

                if volume > avg_vol:

                    volume_status = "HIGH"

                else:

                    volume_status = "AVERAGE"

                # =================================================
                # ENTRY
                # =================================================

                entry = (
                    high
                    + ENTRY_BUFFER
                )

                # =================================================
                # SWING LOW
                # =================================================

                swing_low_data = (
                    find_previous_swing_low(
                        df,
                        signal_index
                    )
                )

                if swing_low_data is None:

                    signal_index += 1
                    continue

                swing_low = float(
                    swing_low_data["price"]
                )

                # =================================================
                # ATR-BASED STOP
                #
                # SL = Swing Low - 10% ATR
                # =================================================

                atr_buffer = (
                    ATR_SL_BUFFER
                    * atr
                )

                sl = (
                    swing_low
                    - atr_buffer
                )

                if sl >= entry:

                    signal_index += 1
                    continue

                # =================================================
                # MINIMUM STOP DISTANCE
                #
                # Entry -> SL must be at least 0.50 ATR
                # =================================================

                risk = (
                    entry
                    - sl
                )

                minimum_allowed_risk = (
                    MIN_STOP_ATR
                    * atr
                )

                if risk < minimum_allowed_risk:

                    rejected_small_stop += 1

                    signal_index += 1
                    continue

                # =================================================
                # SWING HIGH
                # =================================================

                swing_high_data = (
                    find_previous_swing_high(
                        df,
                        signal_index
                    )
                )

                if swing_high_data is None:

                    signal_index += 1
                    continue

                swing_high = float(
                    swing_high_data["price"]
                )

                # =================================================
                # TAKE PROFIT
                # =================================================

                tp = swing_high

                if tp <= entry:

                    signal_index += 1
                    continue

                # =================================================
                # RISK / REWARD
                # =================================================

                reward = (
                    tp
                    - entry
                )

                rr = (
                    reward
                    / risk
                )

                if rr < MIN_RR:

                    rejected_rr += 1

                    signal_index += 1
                    continue

                # =================================================
                # SIMULATE
                # =================================================

                trade_result = simulate_trade(
                    df=df,
                    signal_index=signal_index,
                    entry=entry,
                    sl=sl,
                    tp=tp
                )

                if trade_result is None:

                    signal_index += 1
                    continue

                # =================================================
                # SETUP EXPIRED
                # =================================================

                if (
                    trade_result["Result"]
                    == "NO_ENTRY"
                ):

                    expired_setups += 1

                    signal_index += (
                        ENTRY_VALID_DAYS
                        + 1
                    )

                    continue

                # =================================================
                # ACTUAL TRADE
                # =================================================

                entry_index = (
                    trade_result[
                        "Entry_Index"
                    ]
                )

                exit_index = (
                    trade_result[
                        "Exit_Index"
                    ]
                )

                holding_days = (
                    exit_index
                    - entry_index
                    + 1
                )

                all_trades.append(
                    {
                        "Ticker": ticker,

                        "Signal_Date":
                            df.index[
                                signal_index
                            ],

                        "Entry_Date":
                            df.index[
                                entry_index
                            ],

                        "Exit_Date":
                            df.index[
                                exit_index
                            ],

                        "Signal_Close":
                            round(
                                price,
                                4
                            ),

                        "Entry":
                            round(
                                entry,
                                4
                            ),

                        "SL":
                            round(
                                sl,
                                4
                            ),

                        "TP":
                            round(
                                tp,
                                4
                            ),

                        "Exit_Price":
                            round(
                                trade_result[
                                    "Exit_Price"
                                ],
                                4
                            ),

                        "Swing_Low":
                            round(
                                swing_low,
                                4
                            ),

                        "Swing_High":
                            round(
                                swing_high,
                                4
                            ),

                        "ATR":
                            round(
                                atr,
                                4
                            ),

                        "ATR_Buffer":
                            round(
                                atr_buffer,
                                4
                            ),

                        "Risk":
                            round(
                                risk,
                                4
                            ),

                        "Risk_ATR":
                            round(
                                risk / atr,
                                4
                            ),

                        "Reward":
                            round(
                                reward,
                                4
                            ),

                        "RR":
                            round(
                                rr,
                                4
                            ),

                        "RSI":
                            round(
                                rsi,
                                2
                            ),

                        "Volume_Status":
                            volume_status,

                        "Holding_Days":
                            holding_days,

                        "Result":
                            trade_result[
                                "Result"
                            ],

                        "R":
                            round(
                                trade_result[
                                    "R_Multiple"
                                ],
                                4
                            )
                    }
                )

                # =================================================
                # NO OVERLAPPING TRADES PER TICKER
                # =================================================

                signal_index = (
                    exit_index + 1
                )

        except Exception as e:

            print(
                f"ERROR {ticker}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print()
        print("NO TRADES FOUND.")
        return

    # ========================================================
    # DATAFRAME
    # ========================================================

    trades = pd.DataFrame(
        all_trades
    )

    trades = trades.sort_values(
        [
            "Entry_Date",
            "Ticker"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    total_trades = len(
        trades
    )

    wins = (
        trades["Result"]
        == "WIN"
    ).sum()

    losses = (
        trades["Result"]
        == "LOSS"
    ).sum()

    time_exits = (
        trades["Result"]
        == "TIME_EXIT"
    ).sum()

    positive_trades = (
        trades["R"] > 0
    ).sum()

    negative_trades = (
        trades["R"] < 0
    ).sum()

    tp_win_rate = (
        wins
        / total_trades
        * 100
    )

    profitable_rate = (
        positive_trades
        / total_trades
        * 100
    )

    total_r = (
        trades["R"].sum()
    )

    average_r = (
        trades["R"].mean()
    )

    median_r = (
        trades["R"].median()
    )

    positive = trades[
        trades["R"] > 0
    ]

    negative = trades[
        trades["R"] < 0
    ]

    avg_positive_r = (
        positive["R"].mean()
        if not positive.empty
        else 0
    )

    avg_negative_r = (
        negative["R"].mean()
        if not negative.empty
        else 0
    )

    gross_profit = (
        positive["R"].sum()
        if not positive.empty
        else 0
    )

    gross_loss = (
        abs(
            negative["R"].sum()
        )
        if not negative.empty
        else 0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # ========================================================
    # EQUITY / DRAWDOWN
    # ========================================================

    trades[
        "Cumulative_R"
    ] = trades[
        "R"
    ].cumsum()

    trades[
        "Peak_R"
    ] = (
        trades[
            "Cumulative_R"
        ]
        .clip(lower=0)
        .cummax()
    )

    trades[
        "Drawdown_R"
    ] = (
        trades[
            "Cumulative_R"
        ]
        -
        trades[
            "Peak_R"
        ]
    )

    max_drawdown_r = (
        trades[
            "Drawdown_R"
        ].min()
    )

    # ========================================================
    # CONSECUTIVE LOSSES
    # ========================================================

    max_consecutive_losses = 0
    current_losses = 0

    for r in trades["R"]:

        if r < 0:

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    # ========================================================
    # OTHER STATS
    # ========================================================

    average_holding_days = (
        trades[
            "Holding_Days"
        ].mean()
    )

    average_rr = (
        trades[
            "RR"
        ].mean()
    )

    median_rr = (
        trades[
            "RR"
        ].median()
    )

    average_risk_atr = (
        trades[
            "Risk_ATR"
        ].mean()
    )

    minimum_risk_atr = (
        trades[
            "Risk_ATR"
        ].min()
    )

    maximum_rr = (
        trades[
            "RR"
        ].max()
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print("=" * 72)
    print("BACKTEST RESULTS - V3 ATR")
    print("=" * 72)

    print(
        f"Total trades:               {total_trades}"
    )

    print(
        f"TP wins:                    {wins}"
    )

    print(
        f"SL losses:                  {losses}"
    )

    print(
        f"Time exits:                 {time_exits}"
    )

    print()

    print(
        f"TP win rate:                {tp_win_rate:.2f}%"
    )

    print(
        f"Profitable trades:          {profitable_rate:.2f}%"
    )

    print()

    print(
        f"Total R:                    {total_r:.2f}R"
    )

    print(
        f"Average R/trade:            {average_r:.3f}R"
    )

    print(
        f"Median R/trade:             {median_r:.3f}R"
    )

    print(
        f"Average positive trade:     {avg_positive_r:.3f}R"
    )

    print(
        f"Average negative trade:     {avg_negative_r:.3f}R"
    )

    print(
        f"Profit factor:              {profit_factor:.2f}"
    )

    print()

    print(
        f"Max drawdown:               {max_drawdown_r:.2f}R"
    )

    print(
        f"Max consecutive losses:     {max_consecutive_losses}"
    )

    print(
        f"Average holding days:       {average_holding_days:.2f}"
    )

    print()

    print(
        f"Average theoretical R:R:    {average_rr:.2f}"
    )

    print(
        f"Median theoretical R:R:     {median_rr:.2f}"
    )

    print(
        f"Maximum theoretical R:R:    {maximum_rr:.2f}"
    )

    print()

    print(
        f"Average stop distance:      {average_risk_atr:.2f} ATR"
    )

    print(
        f"Minimum stop distance:      {minimum_risk_atr:.2f} ATR"
    )

    print()

    print(
        f"Rejected small stops:       {rejected_small_stop}"
    )

    print(
        f"Rejected R:R < {MIN_RR}:          {rejected_rr}"
    )

    print(
        f"Expired setups:             {expired_setups}"
    )

    # ========================================================
    # EXTREME TRADES CHECK
    # ========================================================

    print()

    print("=" * 72)
    print("EXTREME TRADES CHECK")
    print("=" * 72)

    print()
    print(
        "Largest theoretical R:R setups:"
    )

    largest_rr = (
        trades.nlargest(
            10,
            "RR"
        )[
            [
                "Ticker",
                "Signal_Date",
                "Entry",
                "SL",
                "TP",
                "ATR",
                "Risk_ATR",
                "RR",
                "Result",
                "R"
            ]
        ]
    )

    print(
        largest_rr.to_string(
            index=False
        )
    )

    print()

    print(
        "Largest winning trades:"
    )

    largest_winners = (
        trades.nlargest(
            10,
            "R"
        )[
            [
                "Ticker",
                "Signal_Date",
                "Entry_Date",
                "Exit_Date",
                "Entry",
                "SL",
                "TP",
                "ATR",
                "RR",
                "Result",
                "R"
            ]
        ]
    )

    print(
        largest_winners.to_string(
            index=False
        )
    )

    # ========================================================
    # VOLUME COMPARISON
    # ========================================================

    print()

    print("=" * 72)
    print("HIGH VOLUME VS AVERAGE VOLUME")
    print("=" * 72)

    volume_summary = (
        trades
        .groupby(
            "Volume_Status"
        )
        .agg(
            Trades=(
                "R",
                "count"
            ),
            Total_R=(
                "R",
                "sum"
            ),
            Avg_R=(
                "R",
                "mean"
            ),
            Median_R=(
                "R",
                "median"
            )
        )
    )

    print(
        volume_summary.round(
            3
        )
    )

    # ========================================================
    # RESULTS BY TICKER
    # ========================================================

    print()

    print("=" * 72)
    print("RESULTS BY TICKER")
    print("=" * 72)

    ticker_summary = (
        trades
        .groupby(
            "Ticker"
        )
        .agg(
            Trades=(
                "R",
                "count"
            ),
            Total_R=(
                "R",
                "sum"
            ),
            Avg_R=(
                "R",
                "mean"
            ),
            Median_R=(
                "R",
                "median"
            ),
            TP_Wins=(
                "Result",
                lambda x:
                (
                    x == "WIN"
                ).sum()
            ),
            SL_Losses=(
                "Result",
                lambda x:
                (
                    x == "LOSS"
                ).sum()
            ),
            Time_Exits=(
                "Result",
                lambda x:
                (
                    x == "TIME_EXIT"
                ).sum()
            )
        )
        .sort_values(
            "Total_R",
            ascending=False
        )
    )

    print(
        ticker_summary.round(
            2
        )
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    trades.to_csv(
        "backtest_trades_v3.csv",
        index=False
    )

    volume_summary.to_csv(
        "backtest_by_volume_v3.csv"
    )

    ticker_summary.to_csv(
        "backtest_by_ticker_v3.csv"
    )

    print()

    print("=" * 72)

    print("Saved:")
    print("  backtest_trades_v3.csv")
    print("  backtest_by_volume_v3.csv")
    print("  backtest_by_ticker_v3.csv")

    print()

    print(
        "Pullback Backtest V3 completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
