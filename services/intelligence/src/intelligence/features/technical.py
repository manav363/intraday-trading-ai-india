"""Technical features (L4), computed causally.

Every function here takes exactly the series it needs as arguments. It never
reads a column that an earlier step happened to add. v1's feature engine had an
implicit ordering — `add_volume_features` and `add_volatility_features` both
read `Return_1`, which existed only because `add_returns_momentum` ran first, so
reordering the calls raised a KeyError from inside an unrelated function. That
class of coupling is designed out rather than documented.

Three v1 defects fixed here:

* **VWAP now resets each session.** v1 `cumsum()`-ed across the whole
  multi-day frame while its own comment claimed a daily reset. By day ten the
  "VWAP" was a ten-day average and `VWAP_Distance_Pct` measured something else
  entirely.
* **ATR uses Wilder's RMA**, not a simple rolling mean. They are different
  numbers, and every charting package uses Wilder's.
* **Volatility is estimated from OHLC**, not close-to-close. Garman-Klass and
  Parkinson use the whole bar and are several times more efficient per
  observation — which matters a great deal when the whole history is 60 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from intraday_contracts import IST, NSE_SESSION_MINUTES, NSE_SESSION_OPEN

EPS = 1e-12

IST_TZ = IST
SESSION_MINUTES = NSE_SESSION_MINUTES
SESSION_OPEN_HOUR = NSE_SESSION_OPEN.hour
SESSION_OPEN_MINUTE = NSE_SESSION_OPEN.minute


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping division-by-zero to NaN rather than inf.

    NaN propagates into "this feature is unknown here", which downstream code
    already handles. An inf silently poisons any rolling statistic it enters.
    """
    return numerator / denominator.replace(0.0, np.nan)


def session_key(timestamps: pd.Series) -> pd.Series:
    """The trading date each bar belongs to. The reset key for session features."""
    return timestamps.dt.date


# ------------------------------------------------------------- returns


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def add_returns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = frame["close"]
    ret1 = log_returns(close)

    return {
        "ret_1": ret1,
        "ret_3": np.log(close).diff(3),
        "ret_6": np.log(close).diff(6),
        "ret_12": np.log(close).diff(12),
        # Momentum normalised by price, so it is comparable across symbols at
        # different price levels. v1 used raw price differences, which are not.
        "momentum_6": _safe_divide(close - close.shift(6), close.shift(6)),
        "momentum_12": _safe_divide(close - close.shift(12), close.shift(12)),
    }


# ---------------------------------------------------------- volatility


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson (1980). Uses the bar's range; ~5x more efficient than
    close-to-close for the same number of observations."""
    hl = np.log(_safe_divide(high, low)) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass (1980). Uses the full OHLC bar.

    The inner term can go slightly negative on a degenerate bar, so it is
    clipped at zero before the square root rather than producing NaN.
    """
    log_hl = np.log(_safe_divide(high, low)) ** 2
    log_co = np.log(_safe_divide(close, open_)) ** 2
    estimator = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(estimator.rolling(window).mean().clip(lower=0.0))


def realised_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return log_returns(close).rolling(window).std()


def add_volatility(frame: pd.DataFrame) -> dict[str, pd.Series]:
    open_, high, low, close = frame["open"], frame["high"], frame["low"], frame["close"]

    rv_short = realised_volatility(close, 10)
    rv_long = realised_volatility(close, 50)

    return {
        "vol_realised_10": rv_short,
        "vol_realised_50": rv_long,
        "vol_parkinson_20": parkinson_volatility(high, low, 20),
        "vol_garman_klass_20": garman_klass_volatility(open_, high, low, close, 20),
        # Short vol relative to long vol: a regime-ish signal that is scale-free
        # and therefore comparable across the cross-section.
        "vol_ratio": _safe_divide(rv_short, rv_long),
    }


# ----------------------------------------------------------------- ATR


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: RMA_t = (RMA_{t-1}·(n-1) + x_t) / n.

    Equivalent to an EWM with alpha = 1/n. A simple rolling mean — which v1
    used — gives a materially different number, and every charting package
    reports Wilder's.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_atr(frame: pd.DataFrame) -> dict[str, pd.Series]:
    tr = true_range(frame["high"], frame["low"], frame["close"])
    atr14 = wilder_rma(tr, 14)
    return {
        "atr_14": atr14,
        # Normalised by price so it compares across symbols.
        "atr_pct": _safe_divide(atr14, frame["close"]),
    }


# ---------------------------------------------------------------- VWAP


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, **reset at every session open**.

    This is the v1 bug. Its comment said "reset daily for intraday" and its code
    ran `cumsum()` over the entire multi-day frame, so by day ten the value was
    a ten-day average and every distance-from-VWAP feature measured something
    other than what it claimed.
    """
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    key = session_key(frame["timestamp"])

    cum_pv = (typical * frame["volume"]).groupby(key).cumsum()
    cum_v = frame["volume"].groupby(key).cumsum()
    return _safe_divide(cum_pv, cum_v)


def add_vwap(frame: pd.DataFrame) -> dict[str, pd.Series]:
    vwap = session_vwap(frame)
    close = frame["close"]
    return {
        "vwap": vwap,
        "vwap_distance_pct": _safe_divide(close - vwap, vwap),
        "above_vwap": (close > vwap).astype(float),
    }


# ------------------------------------------------------------- momentum


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing.

    An all-gains window gives RSI 100 and an all-losses window gives 0; the
    zero-division is mapped explicitly rather than left to produce inf.
    """
    delta = close.diff()
    gain = wilder_rma(delta.clip(lower=0.0), period)
    loss = wilder_rma((-delta).clip(lower=0.0), period)

    rs = _safe_divide(gain, loss)
    out = 100.0 - (100.0 / (1.0 + rs))
    # loss == 0 and gain > 0 -> pure uptrend -> 100.
    return out.where(~((loss == 0) & (gain > 0)), 100.0)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD with `adjust=False`, which is what charting packages report."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def add_momentum(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = frame["close"]
    _, _, histogram = macd(close)
    return {
        "rsi_14": rsi(close, 14),
        # Normalised by price: a raw MACD histogram on a ₹3,900 stock and a
        # ₹90 stock are not comparable, and the panel pools both.
        "macd_histogram_pct": _safe_divide(histogram, close),
    }


# --------------------------------------------------------- trend / bands


def add_trend(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = frame["close"]
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(50).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    return {
        "trend_fast_slow": _safe_divide(sma_fast - sma_slow, sma_slow),
        "distance_sma50": _safe_divide(close - sma_slow, sma_slow),
        "bb_width": _safe_divide(bb_upper - bb_lower, bb_mid),
        "bb_position": _safe_divide(close - bb_lower, bb_upper - bb_lower),
    }


# ----------------------------------------------------------- volume


def add_volume(frame: pd.DataFrame) -> dict[str, pd.Series]:
    volume = frame["volume"]
    close = frame["close"]

    vol_ma = volume.rolling(20).mean()
    signed_volume = np.sign(close.diff()) * volume
    obv = signed_volume.cumsum()

    return {
        "volume_ratio": _safe_divide(volume, vol_ma),
        "obv_slope": _safe_divide(obv - obv.shift(10), vol_ma.replace(0.0, np.nan) * 10.0),
        "turnover_log": np.log1p((frame["high"] + frame["low"] + close) / 3.0 * volume),
    }


# ------------------------------------------------------- intraday shape


def add_session_position(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Where in the session each bar sits, plus session-relative extremes.

    Intraday volatility and volume are U-shaped, so time-of-day is genuinely
    informative rather than a nuisance. It is encoded as a smooth fraction and
    two harmonics rather than a raw clock hour, because a tree splitting on
    "hour == 12" learns a calendar artefact, not a shape.
    """
    key = session_key(frame["timestamp"])

    # Derived from the CLOCK, not from the session's bar count.
    #
    # The obvious implementation — `cumcount() / (bars_in_session - 1)` — is a
    # leak, and a subtle one: it divides by how many bars the session turned out
    # to have, which at 10:00 is not yet known. It also silently changes value
    # if the feed drops a bar. Wall-clock position is known at every bar and
    # cannot be revised by later data.
    minutes_since_open = (
        frame["timestamp"].dt.tz_convert(IST_TZ).dt.hour * 60
        + frame["timestamp"].dt.tz_convert(IST_TZ).dt.minute
        - (SESSION_OPEN_HOUR * 60 + SESSION_OPEN_MINUTE)
    )
    fraction = (minutes_since_open / SESSION_MINUTES).clip(lower=0.0, upper=1.0)

    session_high = frame["high"].groupby(key).cummax()
    session_low = frame["low"].groupby(key).cummin()

    return {
        "session_fraction": fraction,
        "session_sin": np.sin(2.0 * np.pi * fraction),
        "session_cos": np.cos(2.0 * np.pi * fraction),
        # Position inside the day's range so far. Uses cummax/cummin, which are
        # strictly past-and-present — a plain max() over the day would leak the
        # rest of the session into every morning bar.
        "session_range_position": _safe_divide(
            frame["close"] - session_low, session_high - session_low
        ),
    }


# --------------------------------------------------------------- build

FEATURE_BUILDERS = (
    add_returns,
    add_volatility,
    add_atr,
    add_vwap,
    add_momentum,
    add_trend,
    add_volume,
    add_session_position,
)


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute every technical feature for one symbol.

    Builders are independent: each reads only raw OHLCV plus `timestamp`, so the
    order of this tuple does not change any result.
    """
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    out = frame.copy()

    for builder in FEATURE_BUILDERS:
        for name, series in builder(frame).items():
            out[name] = series

    return out


def technical_feature_names() -> tuple[str, ...]:
    """Feature columns, excluding intermediates kept for display.

    `vwap` and `atr_14` are absolute price levels — useful to plot, useless to a
    pooled model, and actively harmful because they encode the symbol's price
    level as if it were a signal.
    """
    display_only = {"vwap", "atr_14"}
    return tuple(n for n in _probe_names() if n not in display_only)


def _probe_names() -> tuple[str, ...]:
    """Feature names, discovered by running the builders on a tiny frame.

    Deriving the list rather than hand-maintaining it means a new feature cannot
    be silently absent from training. A hand-written list is exactly the kind of
    thing that drifts.
    """
    import datetime as dt

    n = 8
    probe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [dt.datetime(2025, 1, 1, 9, 15) + dt.timedelta(minutes=15 * i) for i in range(n)],
                utc=True,
            ),
            "open": np.linspace(100, 101, n),
            "high": np.linspace(101, 102, n),
            "low": np.linspace(99, 100, n),
            "close": np.linspace(100, 101, n),
            "volume": np.full(n, 1000.0),
        }
    )
    names: list[str] = []
    for builder in FEATURE_BUILDERS:
        names.extend(builder(probe).keys())
    return tuple(names)
