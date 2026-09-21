import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
import time


# ============================================================
# V25 - UNIVERSE EXPANSION / DIVERSIFICATION TEST
#
# A) NASDAQ-100 + QQQ regime
# B) S&P 500    + QQQ regime
# C) S&P 500    + SPY regime
#
# DXY < SMA200 remains ON in all variants.
#
# SAME SIGNAL ENGINE AS V24:
# - Stoch OFF
# - RSI OFF
# - Relative Volume >= 0.80
# - EMA20 pullback
# - 2.5 ATR trail
# - max 30 bars
#
# PURE STRATEGY TEST:
# NO portfolio position limit
# NO capital limit
# NO ranking
# NO CAGR
#
# Each valid stock/day signal is evaluated independently.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

BACKTEST_YEARS = 15

ATR_PERIOD = 14

ENTRY_VALID_DAYS = 3

TRAIL_HOLD_DAYS = 30
ATR_TRAIL_MULT = 2.5

REL_VOLUME_MIN = 0.80

SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0002

BATCH_SIZE = 40


# ============================================================
# CURRENT NASDAQ-100 LIST
# ============================================================

NASDAQ_TICKERS = [
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
# S&P 500 CONSTITUENTS
#
# Wikipedia first.
# GitHub CSV fallback if Wikipedia returns 403 or fails.
# ============================================================

def get_sp500_constituents():

    wiki_url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    fallback_url = (
        "https://raw.githubusercontent.com/"
        "datasets/s-and-p-500-companies/"
        "master/data/constituents.csv"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(X11; Linux x86_64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
    }

    print(
        "Downloading current S&P 500 constituent list..."
    )

    table = None

    try:

        response = requests.get(
            wiki_url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        tables = pd.read_html(
            StringIO(
                response.text
            )
        )

        table = tables[0].copy()

        print(
            "Loaded S&P 500 list from Wikipedia."
        )

    except Exception as e:

        print(
            f"Wikipedia failed: {e}"
        )

        print(
            "Trying GitHub fallback..."
        )

        response = requests.get(
            fallback_url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        table = pd.read_csv(
            StringIO(
                response.text
            )
        )

        print(
            "Loaded S&P 500 list from GitHub fallback."
        )


    if "Symbol" not in table.columns:

        raise RuntimeError(
            "S&P 500 source does not contain a Symbol column."
        )


    # Normalize sector column
    if (
        "Sector" in table.columns
        and
        "GICS Sector" not in table.columns
    ):

        table[
            "GICS Sector"
        ] = table[
            "Sector"
        ]


    # Yahoo uses BRK-B instead of BRK.B, etc.
    table[
        "Yahoo_Ticker"
    ] = (
        table[
            "Symbol"
        ]
        .astype(str)
        .str.strip()
        .str.replace(
            ".",
            "-",
            regex=False
        )
    )


    tickers = (
        table[
            "Yahoo_Ticker"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    if "GICS Sector" in table.columns:

        sector_map = dict(
            zip(
                table[
                    "Yahoo_Ticker"
                ],
                table[
                    "GICS Sector"
                ]
            )
        )

    else:

        sector_map = {
            ticker:
                "Unknown"

            for ticker in tickers
        }


    print(
        f"S&P 500 constituents loaded: "
        f"{len(tickers)}"
    )


    return (
        tickers,
        sector_map
    )


# ============================================================
# ATR
# ============================================================

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
    ).max(
        axis=1
    )

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
# MARKET DOWNLOAD
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
        period="max",
        interval="1d",
        auto_adjust=True,
        repair=True,
        progress=False,
        threads=False
    )


    if df.empty:

        raise RuntimeError(
            f"No data downloaded for {name}"
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
# ROBUST STOCK DOWNLOAD
# ============================================================

def download_stocks(
    tickers
):

    all_stock_data = {}

    total = len(
        tickers
    )


    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = tickers[
            start:
            start + BATCH_SIZE
        ]


        print(
            f"Downloading stocks "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, total)} "
            f"of {total}..."
        )


        try:

            data = yf.download(
                batch,
                period="max",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                repair=True,
                progress=False,
                threads=False
            )

        except Exception as e:

            print(
                f"Batch failed: {e}"
            )

            continue


        # ====================================================
        # MULTI-TICKER BATCH
        # ====================================================

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            available = set(
                data.columns
                .get_level_values(0)
            )


            for ticker in batch:

                if ticker not in available:
                    continue


                try:

                    df = (
                        data[
                            ticker
                        ]
                        .copy()
                    )


                    if not df.empty:

                        all_stock_data[
                            ticker
                        ] = df


                except Exception:

                    pass


        # ====================================================
        # SINGLE TICKER EDGE CASE
        # ====================================================

        elif len(batch) == 1:

            ticker = batch[0]

            if not data.empty:

                all_stock_data[
                    ticker
                ] = data.copy()


        time.sleep(
            0.2
        )


    # ========================================================
    # RETRY MISSING INDIVIDUALLY
    # ========================================================

    missing = [
        ticker

        for ticker
        in tickers

        if ticker
        not in all_stock_data
    ]


    if missing:

        print()

        print(
            f"Retrying "
            f"{len(missing)} "
            f"missing tickers..."
        )


    for ticker in missing:

        try:

            df = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                repair=True,
                progress=False,
                threads=False
            )


            if isinstance(
                df.columns,
                pd.MultiIndex
            ):

                df.columns = (
                    df.columns
                    .get_level_values(0)
                )


            if not df.empty:

                all_stock_data[
                    ticker
                ] = df


        except Exception as e:

            print(
                f"FAILED {ticker}: {e}"
            )


    return all_stock_data


# ============================================================
# MARKET HELPERS
# ============================================================

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
        available
        .iloc[-1]
    )


    if pd.isna(
        row["SMA200"]
    ):

        return None


    return row


def above_sma200(
    df,
    date
):

    row = get_market_row(
        df,
        date
    )


    if row is None:

        return False


    return (
        float(
            row["Close"]
        )
        >
        float(
            row["SMA200"]
        )
    )


def below_sma200(
    df,
    date
):

    row = get_market_row(
        df,
        date
    )


    if row is None:

        return False


    return (
        float(
            row["Close"]
        )
        <
        float(
            row["SMA200"]
        )
    )


def market_allows_long(
    primary_market,
    dxy,
    date
):

    return (
        above_sma200(
            primary_market,
            date
        )
        and
        below_sma200(
            dxy,
            date
        )
    )


# ============================================================
# STRONG TREND
# ============================================================

def strong_trend(
    df,
    i
):

    if i < 10:

        return False


    row = (
        df.iloc[i]
    )


    return (
        float(
            row["Close"]
        )
        >
        float(
            row["SMA200"]
        )

        and

        float(
            row["SMA50"]
        )
        >
        float(
            row["SMA200"]
        )

        and

        float(
            row["EMA20"]
        )
        >
        float(
            row["SMA50"]
        )

        and

        float(
            row["SMA50"]
        )
        >
        float(
            df[
                "SMA50"
            ].iloc[
                i - 10
            ]
        )
    )


# ============================================================
# PULLBACK SETUP
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


    row = (
        df.iloc[i]
    )


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
        high
        - low
    )


    if candle_range <= 0:

        return None


    close_location = (
        close
        - low
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
    # ENTRY
    # ========================================================

    trigger = (
        high
        + 0.05
        * atr
    )


    # ========================================================
    # STOP
    # ========================================================

    stop = (
        min(
            low,
            low5
        )
        - 0.10
        * atr
    )


    risk = (
        trigger
        - stop
    )


    if risk <= 0:

        return None


    if risk < (
        0.50
        * atr
    ):

        return None


    if risk > (
        3.0
        * atr
    ):

        return None


    return {
        "Trigger":
            trigger,

        "Stop":
            stop,

        "EMA_Distance_ATR":
            ema_distance,

        "Close_Location":
            close_location,

        "Relative_Volume":
            relative_volume
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

        len(df)
        - 1
    )


    for i in range(
        signal_index + 1,
        end + 1
    ):

        if float(
            df[
                "High"
            ].iloc[i]
        ) >= trigger:

            return i


    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    trigger,
    initial_stop
):

    entry_open = float(
        df[
            "Open"
        ].iloc[
            entry_index
        ]
    )


    raw_entry = max(
        trigger,
        entry_open
    )


    # ========================================================
    # ENTRY EXECUTION
    # ========================================================

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


    # ========================================================
    # EXIT SEARCH
    # ========================================================

    end = min(
        entry_index
        + TRAIL_HOLD_DAYS
        - 1,

        len(df)
        - 1
    )


    trail = (
        initial_stop
    )


    highest_close = (
        raw_entry
    )


    for i in range(
        entry_index,
        end + 1
    ):

        row = (
            df.iloc[i]
        )


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
            -
            ATR_TRAIL_MULT
            * atr
        )


        trail = max(
            trail,
            calculated_trail
        )


        # ====================================================
        # GAP THROUGH TRAIL
        # ====================================================

        if open_price < trail:

            raw_exit = (
                open_price
            )

            exit_index = i

            exit_reason = (
                "ATR_GAP"
            )

            break


        # ====================================================
        # INTRADAY TRAIL HIT
        # ====================================================

        if low <= trail:

            raw_exit = (
                trail
            )

            exit_index = i

            exit_reason = (
                "ATR_TRAIL"
            )

            break


        highest_close = max(
            highest_close,
            close
        )


    else:

        exit_index = (
            end
        )


        raw_exit = float(
            df[
                "Close"
            ].iloc[
                exit_index
            ]
        )


        exit_reason = (
            "TIME30"
        )


    # ========================================================
    # EXIT EXECUTION
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


    return {
        "Entry_Date":
            pd.Timestamp(
                df.index[
                    entry_index
                ]
            ),

        "Exit_Date":
            pd.Timestamp(
                df.index[
                    exit_index
                ]
            ),

        "Net_R":
            net_r,

        "Exit_Reason":
            exit_reason,

        "Holding_Bars":
            (
                exit_index
                - entry_index
                + 1
            )
    }


# ============================================================
# GENERATE EVERY VALID SIGNAL
# ============================================================

def generate_all_trades(
    universe_name,
    tickers,
    prepared_data,
    primary_market,
    dxy,
    qqq,
    backtest_start,
    end_date,
    sector_map
):

    trades = []


    diagnostics = {
        "Stocks":
            0,

        "Stock_Days":
            0,

        "Raw_Setups":
            0,

        "No_Trigger":
            0,

        "Market_Rejected":
            0,

        "Trades":
            0
    }


    print()

    print(
        "=" * 120
    )

    print(
        f"GENERATING {universe_name}"
    )

    print(
        "=" * 120
    )


    for number, ticker in enumerate(
        tickers,
        start=1
    ):

        if ticker not in prepared_data:

            continue


        df = (
            prepared_data[
                ticker
            ]
        )


        if len(df) < 250:

            continue


        diagnostics[
            "Stocks"
        ] += 1


        if (
            number % 25
            == 0
        ):

            print(
                f"{number}/"
                f"{len(tickers)}"
            )


        for i in range(
            210,
            len(df)
            - ENTRY_VALID_DAYS
            - 1
        ):

            signal_date = (
                pd.Timestamp(
                    df.index[i]
                )
            )


            if signal_date < backtest_start:

                continue


            if signal_date > end_date:

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
                    df[
                        col
                    ].iloc[i]
                )

                for col
                in required
            ):

                continue


            diagnostics[
                "Stock_Days"
            ] += 1


            setup = pullback_setup(
                df,
                i
            )


            if setup is None:

                continue


            diagnostics[
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

                diagnostics[
                    "No_Trigger"
                ] += 1

                continue


            entry_date = pd.Timestamp(
                df.index[
                    entry_index
                ]
            )


            if entry_date > end_date:

                continue


            if not market_allows_long(
                primary_market,
                dxy,
                entry_date
            ):

                diagnostics[
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


            qqq_bull = above_sma200(
                qqq,
                entry_date
            )


            trades.append(
                {
                    "Universe":
                        universe_name,

                    "Ticker":
                        ticker,

                    "Sector":
                        sector_map.get(
                            ticker,
                            "Unknown"
                        ),

                    "Signal_Date":
                        signal_date,

                    **result,

                    "QQQ_Bull_At_Entry":
                        qqq_bull,

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


            diagnostics[
                "Trades"
            ] += 1


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
        diagnostics
    )


# ============================================================
# PERFORMANCE STATS
# ============================================================

def performance_stats(
    trades
):

    if trades.empty:

        return {}


    positive = trades.loc[
        trades[
            "Net_R"
        ] > 0,
        "Net_R"
    ]


    negative = trades.loc[
        trades[
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
            end
            - start
        ).days
        / 365.25
    )


    yearly = (
        trades
        .assign(
            Year=pd.to_datetime(
                trades[
                    "Entry_Date"
                ]
            ).dt.year
        )
        .groupby(
            "Year"
        )[
            "Net_R"
        ]
        .mean()
    )


    return {
        "Trades":
            len(
                trades
            ),

        "Trades_Year":
            (
                len(
                    trades
                )
                / years
            ),

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
            pf,

        "Total_R":
            trades[
                "Net_R"
            ].sum(),

        "Positive_Years_%":
            (
                yearly > 0
            ).mean()
            * 100,

        "Positive_Years":
            int(
                (
                    yearly > 0
                ).sum()
            ),

        "Years":
            len(
                yearly
            )
    }


# ============================================================
# YEARLY ANALYSIS
# ============================================================

def yearly_analysis(
    trades
):

    df = (
        trades.copy()
    )


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
# SECTOR ANALYSIS
# ============================================================

def sector_analysis(
    trades
):

    rows = []


    for sector, group in (
        trades.groupby(
            "Sector"
        )
    ):

        if len(group) < 10:

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
                "Sector":
                    sector,

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


    result = pd.DataFrame(
        rows
    )


    if result.empty:

        return result


    return (
        result
        .sort_values(
            "Avg_R",
            ascending=False
        )
        .set_index(
            "Sector"
        )
    )


# ============================================================
# QQQ REGIME ANALYSIS
# ============================================================

def qqq_state_analysis(
    trades
):

    rows = []


    for qqq_bull, group in (
        trades.groupby(
            "QQQ_Bull_At_Entry"
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
                "QQQ_State":
                    (
                        "QQQ > SMA200"
                        if qqq_bull
                        else
                        "QQQ < SMA200"
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
            "QQQ_State"
        )
    )


# ============================================================
# SECTOR PERFORMANCE WHEN QQQ BEARISH
# ============================================================

def defensive_when_qqq_bear(
    trades
):

    bear = trades[
        trades[
            "QQQ_Bull_At_Entry"
        ] == False
    ].copy()


    if bear.empty:

        return pd.DataFrame()


    return sector_analysis(
        bear
    )


# ============================================================
# PRINT VARIANT
# ============================================================

def print_variant(
    name,
    trades,
    diagnostics
):

    stats = performance_stats(
        trades
    )


    print()

    print(
        "=" * 120
    )

    print(
        name
    )

    print(
        "=" * 120
    )


    print(
        f"Stocks processed:      "
        f"{diagnostics['Stocks']}"
    )

    print(
        f"Raw setups:            "
        f"{diagnostics['Raw_Setups']}"
    )

    print(
        f"No trigger:            "
        f"{diagnostics['No_Trigger']}"
    )

    print(
        f"Market rejected:       "
        f"{diagnostics['Market_Rejected']}"
    )

    print()

    print(
        f"Trades:                "
        f"{stats['Trades']}"
    )

    print(
        f"Trades/year:           "
        f"{stats['Trades_Year']:.2f}"
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

    print(
        f"Positive years:        "
        f"{stats['Positive_Years']}/"
        f"{stats['Years']} "
        f"({stats['Positive_Years_%']:.1f}%)"
    )


    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "V25 - UNIVERSE EXPANSION / DIVERSIFICATION TEST"
    )

    print(
        "=" * 120
    )

    print(
        "Same strategy. No portfolio constraints."
    )

    print()

    print(
        "A: Nasdaq-100 + QQQ regime"
    )

    print(
        "B: S&P 500 + QQQ regime"
    )

    print(
        "C: S&P 500 + SPY regime"
    )


    # ========================================================
    # GET S&P 500
    # ========================================================

    (
        sp500_tickers,
        sector_map
    ) = get_sp500_constituents()


    print()

    print(
        f"Nasdaq tickers:         "
        f"{len(NASDAQ_TICKERS)}"
    )

    print(
        f"S&P 500 tickers:        "
        f"{len(sp500_tickers)}"
    )


    # ========================================================
    # MARKET DATA
    # ========================================================

    qqq = download_market(
        "QQQ",
        "QQQ"
    )


    spy = download_market(
        "SPY",
        "SPY"
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
            spy.index.max()
        ),

        pd.Timestamp(
            dxy.index.max()
        )
    )


    backtest_start = (
        end_date
        - pd.DateOffset(
            years=
                BACKTEST_YEARS
        )
    )


    print()

    print(
        f"Backtest:              "
        f"{backtest_start.date()} "
        f"-> {end_date.date()}"
    )


    # ========================================================
    # DOWNLOAD UNION ONLY ONCE
    # ========================================================

    all_tickers = sorted(
        set(
            NASDAQ_TICKERS
            +
            sp500_tickers
        )
    )


    print()

    print(
        f"Unique stocks to download: "
        f"{len(all_tickers)}"
    )


    raw_stock_data = download_stocks(
        all_tickers
    )


    # ========================================================
    # PREPARE INDICATORS
    # ========================================================

    print()

    print(
        "Preparing indicators..."
    )


    prepared_data = {}


    for ticker, df in (
        raw_stock_data.items()
    ):

        try:

            prepared = prepare_stock(
                df
            )


            if len(
                prepared
            ) >= 250:

                prepared_data[
                    ticker
                ] = prepared


        except Exception as e:

            print(
                f"PREP ERROR "
                f"{ticker}: "
                f"{e}"
            )


    print(
        f"Prepared stocks:       "
        f"{len(prepared_data)}"
    )


    # ========================================================
    # A - NASDAQ + QQQ
    # ========================================================

    (
        trades_a,
        diag_a
    ) = generate_all_trades(

        universe_name=
            "NASDAQ_QQQ",

        tickers=
            NASDAQ_TICKERS,

        prepared_data=
            prepared_data,

        primary_market=
            qqq,

        dxy=
            dxy,

        qqq=
            qqq,

        backtest_start=
            backtest_start,

        end_date=
            end_date,

        sector_map=
            sector_map
    )


    # ========================================================
    # B - S&P500 + QQQ
    # ========================================================

    (
        trades_b,
        diag_b
    ) = generate_all_trades(

        universe_name=
            "SP500_QQQ",

        tickers=
            sp500_tickers,

        prepared_data=
            prepared_data,

        primary_market=
            qqq,

        dxy=
            dxy,

        qqq=
            qqq,

        backtest_start=
            backtest_start,

        end_date=
            end_date,

        sector_map=
            sector_map
    )


    # ========================================================
    # C - S&P500 + SPY
    # ========================================================

    (
        trades_c,
        diag_c
    ) = generate_all_trades(

        universe_name=
            "SP500_SPY",

        tickers=
            sp500_tickers,

        prepared_data=
            prepared_data,

        primary_market=
            spy,

        dxy=
            dxy,

        qqq=
            qqq,

        backtest_start=
            backtest_start,

        end_date=
            end_date,

        sector_map=
            sector_map
    )


    # ========================================================
    # RESULTS
    # ========================================================

    stats_a = print_variant(
        "A - NASDAQ-100 + QQQ REGIME",
        trades_a,
        diag_a
    )


    stats_b = print_variant(
        "B - S&P 500 + QQQ REGIME",
        trades_b,
        diag_b
    )


    stats_c = print_variant(
        "C - S&P 500 + SPY REGIME",
        trades_c,
        diag_c
    )


    # ========================================================
    # DIRECT COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        {
            "NASDAQ_QQQ": {

                "Trades":
                    stats_a[
                        "Trades"
                    ],

                "Trades/year":
                    stats_a[
                        "Trades_Year"
                    ],

                "Avg_R":
                    stats_a[
                        "Avg_R"
                    ],

                "PF_R":
                    stats_a[
                        "PF_R"
                    ],

                "Win_%":
                    stats_a[
                        "Win_%"
                    ],

                "Positive_Years_%":
                    stats_a[
                        "Positive_Years_%"
                    ],

                "Total_R":
                    stats_a[
                        "Total_R"
                    ]
            },


            "SP500_QQQ": {

                "Trades":
                    stats_b[
                        "Trades"
                    ],

                "Trades/year":
                    stats_b[
                        "Trades_Year"
                    ],

                "Avg_R":
                    stats_b[
                        "Avg_R"
                    ],

                "PF_R":
                    stats_b[
                        "PF_R"
                    ],

                "Win_%":
                    stats_b[
                        "Win_%"
                    ],

                "Positive_Years_%":
                    stats_b[
                        "Positive_Years_%"
                    ],

                "Total_R":
                    stats_b[
                        "Total_R"
                    ]
            },


            "SP500_SPY": {

                "Trades":
                    stats_c[
                        "Trades"
                    ],

                "Trades/year":
                    stats_c[
                        "Trades_Year"
                    ],

                "Avg_R":
                    stats_c[
                        "Avg_R"
                    ],

                "PF_R":
                    stats_c[
                        "PF_R"
                    ],

                "Win_%":
                    stats_c[
                        "Win_%"
                    ],

                "Positive_Years_%":
                    stats_c[
                        "Positive_Years_%"
                    ],

                "Total_R":
                    stats_c[
                        "Total_R"
                    ]
            }
        }
    )


    print()

    print(
        "=" * 120
    )

    print(
        "V25 DIRECT COMPARISON"
    )

    print(
        "=" * 120
    )


    print(
        comparison
        .round(3)
        .to_string()
    )


    # ========================================================
    # YEAR BY YEAR - SP500 + SPY
    # ========================================================

    yearly_c = yearly_analysis(
        trades_c
    )


    print()

    print(
        "=" * 120
    )

    print(
        "S&P 500 + SPY - YEAR BY YEAR"
    )

    print(
        "=" * 120
    )


    print(
        yearly_c
        .round(3)
        .to_string()
    )


    # ========================================================
    # SECTOR ANALYSIS
    # ========================================================

    sectors_c = sector_analysis(
        trades_c
    )


    print()

    print(
        "=" * 120
    )

    print(
        "S&P 500 + SPY - SECTOR PERFORMANCE"
    )

    print(
        "=" * 120
    )


    if sectors_c.empty:

        print(
            "No sector results."
        )

    else:

        print(
            sectors_c
            .round(3)
            .to_string()
        )


    # ========================================================
    # SP500/SPY SIGNALS SPLIT BY QQQ STATE
    # ========================================================

    qqq_states = qqq_state_analysis(
        trades_c
    )


    print()

    print(
        "=" * 120
    )

    print(
        "S&P 500 + SPY - PERFORMANCE BY QQQ REGIME"
    )

    print(
        "=" * 120
    )


    print(
        qqq_states
        .round(3)
        .to_string()
    )


    # ========================================================
    # MOST IMPORTANT:
    # SECTORS WHILE QQQ < SMA200
    # ========================================================

    bear_sectors = defensive_when_qqq_bear(
        trades_c
    )


    print()

    print(
        "=" * 120
    )

    print(
        "SECTOR PERFORMANCE WHEN QQQ < SMA200"
    )

    print(
        "=" * 120
    )


    if bear_sectors.empty:

        print(
            "No qualifying trades."
        )

    else:

        print(
            bear_sectors
            .round(3)
            .to_string()
        )


    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    trades_a.to_csv(
        "v25_nasdaq_qqq.csv",
        index=False
    )


    trades_b.to_csv(
        "v25_sp500_qqq.csv",
        index=False
    )


    trades_c.to_csv(
        "v25_sp500_spy.csv",
        index=False
    )


    comparison.to_csv(
        "v25_comparison.csv"
    )


    yearly_c.to_csv(
        "v25_sp500_spy_yearly.csv"
    )


    sectors_c.to_csv(
        "v25_sp500_spy_sectors.csv"
    )


    qqq_states.to_csv(
        "v25_qqq_regime_analysis.csv"
    )


    bear_sectors.to_csv(
        "v25_qqq_bear_sector_analysis.csv"
    )


    print()

    print(
        "=" * 120
    )

    print(
        "V25 completed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
