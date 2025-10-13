#!/usr/bin/env python3
"""
sma_backtest.py
یک اسکریپت تک‌فایلی پیشرفته برای بک‌تست استراتژی SMA crossover با خروجی‌های کامل، بهینه‌سازی پارامتر، شاخص‌های عملکرد و نمودارهای ذخیره‌شدنی.
نحوه استفاده:
- در همان پوشه یک data.csv (یا data.json) قرار دهید که حداقل ستون‌های Date و Close داشته باشد.
- سپس اجرا کنید: python sma_backtest.py
"""
import os
from math import sqrt
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ui_trigger import get_user_config
import optuna
import pandas as pd, os
from dataclasses import dataclass
from typing import Optional

# ------------------------- Global configuration (UI labels preserved) -------------------------
# We keep the exact UI labels as keys in GLOBAL_CONFIG so you can refer to them
# by the same syntax in logs and UI code. Internally we also map them to safe
# Python variable names and export those into module globals via
# apply_config_globally().
GLOBAL_CONFIG = {
    "MA_SHORT_DEFAULT": 10,
    "MA Long": 20,
    "Initial Capital": 10000.0,
    "Position Size %": 0.5,
    "Commission": 0.0,
    "Slippage": 0.0005,
    "Exit on MA Cross": True,
    "Take Profit %": 0.05,
    "Stop Loss %": -0.05,
    "Max Holding Days": None,
    "Use Trailing ATR": True,
    "ATR Mult": 3.0,
    "ATR Period": 14,
    # include data path so UI can present it too (not in the original list but used by code)
    "Data Path": "data.csv",
}

# Map UI label keys to internal variable names used throughout this module
UI_TO_INTERNAL = {
    "MA_SHORT_DEFAULT": 'MA_SHORT_DEFAULT',
    "MA Long": 'ma_long',
    "Initial Capital": 'initial_capital',
    "Position Size %": 'position_size_pct',
    "Commission": 'commission',
    "Slippage": 'slippage',
    "Exit on MA Cross": 'exit_on_ma_cross',
    "Take Profit %": 'take_profit_pct',
    "Stop Loss %": 'stop_loss_pct',
    "Max Holding Days": 'max_holding_days',
    "Use Trailing ATR": 'use_trailing_atr',
    "ATR Mult": 'atr_mult',
    "ATR Period": 'atr_period',
    "Data Path": 'data_path',
}

def apply_config_globally():
    """Export values from GLOBAL_CONFIG into module-level Python variables
    using the internal names in UI_TO_INTERNAL. Call this after updating
    GLOBAL_CONFIG (e.g., after getting UI input).
    """
    for ui_key, internal_name in UI_TO_INTERNAL.items():
        # convert empty string to None for Max Holding Days if necessary
        val = GLOBAL_CONFIG.get(ui_key)
        if ui_key == 'Max Holding Days' and (val == "" or val is None):
            val = None
        # export into module globals so all functions can refer to variables like ma_long
        globals()[internal_name] = val

# Explicit module-level variables (so static analysis and other modules see them)
# These are initialized from GLOBAL_CONFIG defaults and will be kept in sync when
# apply_config_globally() is called after UI input.
MA_SHORT_DEFAULT = GLOBAL_CONFIG["MA_SHORT_DEFAULT"]
ma_long = GLOBAL_CONFIG["MA Long"]
initial_capital = GLOBAL_CONFIG["Initial Capital"]
position_size_pct = GLOBAL_CONFIG["Position Size %"]
commission = GLOBAL_CONFIG["Commission"]
slippage = GLOBAL_CONFIG["Slippage"]
exit_on_ma_cross = GLOBAL_CONFIG["Exit on MA Cross"]
take_profit_pct = GLOBAL_CONFIG["Take Profit %"]
stop_loss_pct = GLOBAL_CONFIG["Stop Loss %"]
max_holding_days = GLOBAL_CONFIG["Max Holding Days"]
use_trailing_atr = GLOBAL_CONFIG["Use Trailing ATR"]
atr_mult = GLOBAL_CONFIG["ATR Mult"]
atr_period = GLOBAL_CONFIG["ATR Period"]
data_path = GLOBAL_CONFIG["Data Path"]

# Initialize module globals from defaults (keeps GLOBAL_CONFIG authoritative)
apply_config_globally()


# ------------------------- Config dataclass (cleaner API) -------------------------
@dataclass
class BacktestConfig:
    MA_SHORT_DEFAULT: int = GLOBAL_CONFIG["MA_SHORT_DEFAULT"]
    ma_long: Optional[int] = GLOBAL_CONFIG["MA Long"]
    initial_capital: float = GLOBAL_CONFIG["Initial Capital"]
    position_size_pct: float = GLOBAL_CONFIG["Position Size %"]
    commission: float = GLOBAL_CONFIG["Commission"]
    slippage: float = GLOBAL_CONFIG["Slippage"]
    exit_on_ma_cross: bool = GLOBAL_CONFIG["Exit on MA Cross"]
    take_profit_pct: Optional[float] = GLOBAL_CONFIG["Take Profit %"]
    stop_loss_pct: Optional[float] = GLOBAL_CONFIG["Stop Loss %"]
    max_holding_days: Optional[int] = None
    use_trailing_atr: bool = GLOBAL_CONFIG["Use Trailing ATR"]
    atr_mult: float = GLOBAL_CONFIG["ATR Mult"]
    atr_period: int = GLOBAL_CONFIG["ATR Period"]
    data_path: str = GLOBAL_CONFIG["Data Path"]

    @classmethod
    def from_global(cls):
        # ensure module globals are applied
        apply_config_globally()
        return cls(
            MA_SHORT_DEFAULT=MA_SHORT_DEFAULT,
            ma_long=ma_long,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            commission=commission,
            slippage=slippage,
            exit_on_ma_cross=exit_on_ma_cross,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
            use_trailing_atr=use_trailing_atr,
            atr_mult=atr_mult,
            atr_period=atr_period,
            data_path=data_path,
        )

def main_from_ui(config=None):
    """Run backtest using a UI-provided config.

    If `config` is None this function will call `get_user_config()` once.
    Returns (bt_data, bt_trades, config) so the caller can continue with saving/plots.
    """
    # If no config provided, ask UI (which returns a dict keyed by UI labels)
    if config is None:
        ui_values = get_user_config()
        # ui_values is expected to use the exact UI label keys; update GLOBAL_CONFIG
        if isinstance(ui_values, dict):
            for k, v in ui_values.items():
                if k in GLOBAL_CONFIG and v is not None:
                    GLOBAL_CONFIG[k] = v
        # apply to module globals so all functions below see the updated values
        apply_config_globally()
    else:
        # config was passed programmatically and may be internal-keyed; try to accept both formats
        # if keys look like UI labels, update GLOBAL_CONFIG accordingly
        for k, v in (config.items() if isinstance(config, dict) else []):
            if k in GLOBAL_CONFIG:
                GLOBAL_CONFIG[k] = v
            elif k in UI_TO_INTERNAL.values():
                # find corresponding UI key
                ui_key = next((uk for uk, ik in UI_TO_INTERNAL.items() if ik == k), None)
                if ui_key:
                    GLOBAL_CONFIG[ui_key] = v
        apply_config_globally()

    # Use module-level data_path (populated by apply_config_globally)
    df = load_data(data_path)

    cfg = BacktestConfig.from_global()
    bt_data, bt_trades = backtest_sma(df, cfg=cfg)

    
    print("\n===== BACKTEST DATA (bt_data) SAMPLE =====")
    print(bt_data.head(10))

    print("\n===== BACKTEST TRADES (bt_trades) =====")
    print(bt_trades)

    print("\n===== BACKTEST TRADES (df) =====")
    print(df)

    return df, bt_data, bt_trades, config

# ------------------------- Utilities -------------------------
def load_data(path_csv=r"F:\sell\EURUSD60_h1_converted.csv"):
    """
    نسخه پایدار برای فایل EURUSD1440_d.csv با ساختار استاندارد CSV:
    Date,Time,Open,Close,High,Low,Volume
    """
    

    if not os.path.exists(path_csv):
        print(f"❌ فایل داده پیدا نشد: {path_csv}")
        exit(1)

    try:
        df = pd.read_csv(path_csv)
    except Exception as e:
        print("❌ خطا در خواندن فایل:", e)
        exit(1)

    if df.empty:
        print("❌ فایل داده خالی است یا ساختار آن اشتباه است.")
        exit(1)

    # اطمینان از اینکه ستون‌ها دقیقاً مطابق انتظار هستند
    expected_cols = ['Date', 'Time', 'Open', 'Close', 'High', 'Low', 'Volume']
    for c in expected_cols:
        if c not in df.columns:
            print(f"❌ ستون '{c}' در فایل یافت نشد.")
            exit(1)

    
    # ✅ اطمینان از اینکه رشته هستند
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)

    # ✅ ترکیب Date و Time به اندیس زمانی
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    # ✅ بررسی ردیف‌های مشکل‌دار بدون حذف
    bad_dt = df['DateTime'].isna()
    if bad_dt.any():
        print(f"⚠️ هشدار: {bad_dt.sum()} سطر دارای تاریخ/زمان نامعتبر است:")
        print(df.loc[bad_dt, ['Date', 'Time']].head(10))
        raise ValueError("🚫 برخی سطرها تاریخ/زمان نامعتبر دارند (مشخص شده در خروجی بالا).")

    # ✅ ادامه فقط با سطرهای سالم (اما حذف فیزیکی انجام نمی‌دیم)
    df = df.set_index('DateTime').sort_index()

    

     # ✅ تبدیل داده‌های عددی با گزارش موارد خراب
    numeric_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    for c in numeric_cols:
        bad = pd.to_numeric(df[c], errors='coerce').isna() & df[c].notna()
        if bad.any():
            print(f"⚠️ هشدار: {bad.sum()} مقدار در ستون '{c}' مشکل دارد و به NaN تبدیل می‌شود:")
            print(df.loc[bad, [c]].head(10))
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Close'])

    return df


def atr(df, n=14):
    """محاسبه ATR پایه (اگر High/Low موجود نباشد، از Close برای تقریب استفاده می‌شود)."""
    if set(['High', 'Low']).issubset(df.columns):
        high = df['High']
        low = df['Low']
        close = df['Close']
    else:
        close = df['Close']
        high = close * 1.001
        low = close * 0.999
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=1).mean()


def compute_metrics_from_equity(equity_series):
    """محاسبه‌ی شاخص‌های کلیدی: total return, CAGR, annual vol, Sharpe, Sortino, max drawdown, Calmar"""
    if equity_series.isna().all():
        raise ValueError("Equity series is all NaN")
    start_val = equity_series.iloc[0]
    end_val = equity_series.iloc[-1]
    total_return = end_val / start_val - 1

    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else 1 / 252
    cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 else np.nan

    daily_ret = equity_series.pct_change().dropna()
    if daily_ret.empty:
        ann_vol = 0.0
        sharpe = np.nan
        sortino = np.nan
    else:
        ann_vol = daily_ret.std() * sqrt(252)
        sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
        neg_ret = daily_ret[daily_ret < 0]
        downside = neg_ret.std() * sqrt(252) if len(neg_ret) > 0 else 0.0
        sortino = (daily_ret.mean() * 252) / downside if downside > 0 else np.nan

    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    calmar = (cagr) / (-max_dd) if (max_dd < 0 and not np.isnan(cagr)) else np.nan

    return {
        'total_return_pct': total_return * 100,
        'CAGR_pct': cagr * 100 if not np.isnan(cagr) else np.nan,
        'annual_vol_pct': ann_vol * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown_pct': max_dd * 100,
        'calmar': calmar
    }


def profit_factor(trades_df):
    """محاسبه Profit Factor: sum(wins) / sum(losses)"""
    if trades_df is None or trades_df.empty:
        return np.nan
    wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    losses = -trades_df[trades_df['pnl'] < 0]['pnl'].sum()
    if losses == 0:
        return np.nan
    return wins / losses


# ------------------------- Backtest Engine -------------------------
def backtest_sma(df, cfg: Optional[BacktestConfig] = None, **legacy_kwargs):
    """Backtest. Prefer passing a BacktestConfig instance (cfg). For backwards
    compatibility, legacy keyword args are accepted and converted to a cfg.
    """
    if cfg is None:
        # build config from legacy kwargs or module globals
        if legacy_kwargs:
            # map legacy keys to dataclass fields where possible
            cfg = BacktestConfig(
                MA_SHORT_DEFAULT=legacy_kwargs.get('MA_SHORT_DEFAULT', MA_SHORT_DEFAULT),
                ma_long=legacy_kwargs.get('ma_long', ma_long),
                initial_capital=legacy_kwargs.get('initial_capital', initial_capital),
                position_size_pct=legacy_kwargs.get('position_size_pct', position_size_pct),
                commission=legacy_kwargs.get('commission', commission),
                slippage=legacy_kwargs.get('slippage', slippage),
                exit_on_ma_cross=legacy_kwargs.get('exit_on_ma_cross', exit_on_ma_cross),
                take_profit_pct=legacy_kwargs.get('take_profit_pct', take_profit_pct),
                stop_loss_pct=legacy_kwargs.get('stop_loss_pct', stop_loss_pct),
                max_holding_days=legacy_kwargs.get('max_holding_days', max_holding_days),
                use_trailing_atr=legacy_kwargs.get('use_trailing_atr', use_trailing_atr),
                atr_mult=legacy_kwargs.get('atr_mult', atr_mult),
                atr_period=legacy_kwargs.get('atr_period', atr_period),
                data_path=legacy_kwargs.get('data_path', data_path),
            )
        else:
            cfg = BacktestConfig.from_global()
    """پیاده‌سازی بک‌تست:
    - ورود: وقتی SMA_short امروز بالاتر از SMA_long شد و دیروز <= بود.
    - خروج: ترکیبی از MA-cross down یا TP/SL یا Trailing ATR یا حداکثر مدت نگهداری.
    - اجرای قیمت در Close روز سیگنال (برای جلوگیری از lookahead از سیگنال دیروز استفاده شده).
    
        Trades contract (list of dicts)
        -------------------------------
        During the backtest we append a dict for each executed trade into `trades` list.
        This function documents the expected shape so downstream code and tests can rely on it.

        Each trade dict contains the following keys:
            - entry_date: pd.Timestamp (datetime index value when trade opened)
            - entry_price: float (execution price at entry)
            - exit_date: pd.Timestamp or None (datetime when closed)
            - exit_price: float or None (execution price at exit)
            - qty: float (position size in units)
            - pnl: float or None (profit/loss in currency units, qty*(exit-entry))
            - pct_return: float or None (percentage return of the trade, e.g. 5.0 for 5%)
            - holding_days: int or None (days between entry and exit)
            - exit_reason: str or None (one of 'ma_cross','take_profit','stop_loss','trailing_atr','max_hold')
            - cash_before: float (cash balance before entry)
            - cash_after: float or computed later (cash after exit, may be added post-run)

        Error modes:
            - If a trade is still open at the end of the backtest, exit_date/exit_price/pnl/pct_return remain None.
            - Numeric fields may contain NaN when computation is not possible.

        The returned `trades_df` will be a DataFrame created from this list and coerced to
        sensible dtypes where possible (datetimes, floats, ints). Empty trades list yields
        an empty DataFrame with the columns listed above.
    """
    # use cfg values
    data = df.copy()
    MA_SHORT = cfg.MA_SHORT_DEFAULT
    ma_long_local = cfg.ma_long if cfg.ma_long is not None else (MA_SHORT * 2)

    data[f'SMA_{MA_SHORT}'] = data['Close'].rolling(window=MA_SHORT, min_periods=1).mean()
    data[f'SMA_{ma_long_local}'] = data['Close'].rolling(window=ma_long_local, min_periods=1).mean()
    data['ATR'] = atr(data, n=cfg.atr_period)

    trades = []
    in_pos = False
    qty = 0.0
    cash = cfg.initial_capital
    entry_price = None
    trailing_stop = None

    data['Entry'] = False
    data['Exit'] = False
    data['PositionQty'] = 0.0
    data['Cash'] = np.nan
    data['Equity'] = np.nan

    # از ایندکس 1 شروع می‌کنیم (نیاز به مقدار قبلی برای تشخیص کراس)
    for i in range(1, len(data)):
        today = data.index[i]
        close_today = float(data['Close'].iloc[i])
        sma_s_today = float(data[f'SMA_{MA_SHORT}'].iloc[i])
        sma_l_today = float(data[f'SMA_{ma_long_local}'].iloc[i])
        sma_s_yest = float(data[f'SMA_{MA_SHORT}'].iloc[i - 1])
        sma_l_yest = float(data[f'SMA_{ma_long_local}'].iloc[i - 1])

        entry_signal = (sma_s_yest <= sma_l_yest) and (sma_s_today > sma_l_today)
        exit_signal_ma = (sma_s_yest >= sma_l_yest) and (sma_s_today < sma_l_today)

        # ورود
        if (not in_pos) and entry_signal:
            alloc = cash * cfg.position_size_pct
            exec_price = close_today * (1 + cfg.slippage)
            qty = alloc / exec_price if exec_price > 0 else 0.0
            cash -= qty * exec_price
            cash -= commission
            in_pos = True
            entry_price = exec_price
            # trailing stop اولیه
            if cfg.use_trailing_atr:
                atr_today = data['ATR'].iloc[i]
                trailing_stop = entry_price - cfg.atr_mult * atr_today
            data.at[today, 'Entry'] = True
            trades.append({
                'entry_date': today, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None,
                'qty': qty, 'pnl': None, 'pct_return': None, 'holding_days': None, 'exit_reason': None,
                'cash_before': cash + qty * exec_price
            })

        # بررسی خروج‌ها
        if in_pos:
            trade = trades[-1]
            if cfg.use_trailing_atr:
                atr_today = data['ATR'].iloc[i]
                proposed_stop = close_today - cfg.atr_mult * atr_today
                if trailing_stop is None or proposed_stop > trailing_stop:
                    trailing_stop = proposed_stop
            tp_hit = False
            sl_hit = False
            ts_hit = False
            max_hold_hit = False
            if cfg.take_profit_pct is not None and entry_price:
                tp_hit = (close_today / entry_price - 1) >= cfg.take_profit_pct
            if cfg.stop_loss_pct is not None and entry_price:
                sl_hit = (close_today / entry_price - 1) <= cfg.stop_loss_pct
            if use_trailing_atr and trailing_stop is not None and close_today <= trailing_stop:
                ts_hit = True
            if cfg.max_holding_days is not None:
                hdays = (today - trade['entry_date']).days
                if hdays >= cfg.max_holding_days:
                    max_hold_hit = True

            will_exit = False
            reason = None
            if exit_on_ma_cross and exit_signal_ma:
                will_exit = True
                reason = 'ma_cross'
            if tp_hit:
                will_exit = True
                reason = 'take_profit'
            if sl_hit:
                will_exit = True
                reason = 'stop_loss'
            if ts_hit:
                will_exit = True
                reason = 'trailing_atr'
            if max_hold_hit:
                will_exit = True
                reason = 'max_hold'

            if will_exit:
                exit_price = close_today * (1 - cfg.slippage)
                cash += qty * exit_price
                cash -= commission
                in_pos = False
                data.at[today, 'Exit'] = True
                trade['exit_date'] = today
                trade['exit_price'] = exit_price
                trade['pnl'] = trade['qty'] * (trade['exit_price'] - trade['entry_price'])
                trade['pct_return'] = (trade['exit_price'] / trade['entry_price'] - 1) * 100 if trade['entry_price'] else np.nan
                trade['holding_days'] = (trade['exit_date'] - trade['entry_date']).days
                trade['exit_reason'] = reason
                qty = 0.0
                entry_price = None
                trailing_stop = None

        data.at[today, 'PositionQty'] = qty
        data.at[today, 'Cash'] = cash
        data.at[today, 'Equity'] = cash + qty * data['Close'].iloc[i]

    # مقداردهی اولیه و پر کردن اینده‌نگر
    if pd.isna(data['Equity'].iloc[0]):
        data.at[data.index[0], 'Equity'] = cfg.initial_capital

    data['Equity'] = data['Equity'].ffill()
    # به‌روزرسانی مقدار نقدی و سرمایه در هر روز (بدون هشدار chained assignment)
    data.loc[today, 'Cash'] = cash
    data.loc[today, 'Equity'] = cash + qty * data['Close'].iloc[i]


    # ساخت DataFrame از لیست معاملات
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=['entry_date', 'entry_price', 'exit_date', 'exit_price',
                                          'qty', 'holding_days', 'pct_return', 'pnl', 'exit_reason',
                                          'cash_before', 'cash_after'])
    else:
        trades_df['cash_after'] = trades_df.apply(
            lambda r: r['cash_before'] + (r['qty'] * (r['exit_price'] - r['entry_price']) if pd.notna(r['exit_price']) else 0.0),
            axis=1
        )
    return data, trades_df


# ------------------------- Optimization -------------------------
def optimize_grid(df, ma_shorts=[5, 10, 15, 20], tp_values=[None, 0.03, 0.05], **kwargs):
    """
    اجرای گرید سرچ روی پارامترهای MA کوتاه و درصد تارگت سود.
    برای هر ترکیب، بک‌تست را اجرا می‌کند و متریک‌ها را در یک DataFrame برمی‌گرداند.
    """
    results = []

    for ma in ma_shorts:
        for tp in tp_values:
            # build a config derived from global defaults and override MA params
            grid_cfg = BacktestConfig.from_global()
            grid_cfg.MA_SHORT_DEFAULT = ma
            grid_cfg.ma_long = ma * 2
            grid_cfg.take_profit_pct = tp
            # apply any keyword overrides from kwargs
            for k, v in kwargs.items():
                if hasattr(grid_cfg, k):
                    setattr(grid_cfg, k, v)

            data_bt, trades = backtest_sma(df, cfg=grid_cfg)
            metrics = compute_metrics_from_equity(data_bt['Equity'])
            pf = profit_factor(trades)

            results.append({
                'MA_SHORT_DEFAULT': ma,
                'take_profit_pct': tp if tp is not None else np.nan,
                'total_return_pct': metrics['total_return_pct'],
                'CAGR_pct': metrics['CAGR_pct'],
                'sharpe': metrics['sharpe'],
                'sortino': metrics['sortino'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'profit_factor': pf,
                'num_trades': len(trades)
            })

    print("\n===== result of def optimize_grid =====")
    print(pd.DataFrame(results))
    return pd.DataFrame(results)



        
def objective(trial):
    MA_SHORT_DEFAULT_T = trial.suggest_int("short_window", 65, 80)
    ma_longg = trial.suggest_int("long_window", 110, 150)
    take_profit_pctt = trial.suggest_float("take_profit", 0.01, 0.05)
    # construct a BacktestConfig from globals and override with trial parameters
    cfg = BacktestConfig.from_global()
    cfg.MA_SHORT_DEFAULT = MA_SHORT_DEFAULT_T
    cfg.ma_long = ma_longg
    cfg.take_profit_pct = take_profit_pctt
    # use a fresh dataframe from the given path (respect GLOBAL_CONFIG data_path if set)
    df_local = load_data(path_csv=GLOBAL_CONFIG.get("Data Path", r"F:\\sell\\EURUSD60_h1_converted.csv"))
    data_bt, trades = backtest_sma(df_local, cfg=cfg)
    metrics = compute_metrics_from_equity(data_bt['Equity'])
    pf = profit_factor(trades)
    return pf


def run_optuna_trials(n_trials: int = 1):
    """Run an Optuna study using the objective defined above. Returns the study."""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("\n===== Best trial parameters =====")
    print(study.best_trial.params)
    print(f"Best trial value (CAGR %): {study.best_trial.value}")
    return study

# return study.best_trial.params, study.best_trial.value
# ------------------------- Main (config + run) -------------------------
def main():
    # We'll use GLOBAL_CONFIG (exact UI-label keys) as the authoritative source.
    # apply_config_globally() ensures module-level variables are set.
    ui_ran = False
    try:
        # Show UI exactly once and capture values
        ui_values = get_user_config()
        if isinstance(ui_values, dict):
            for k, v in ui_values.items():
                # Accept UI label keys (e.g. "Data Path") or internal keys (e.g. 'data_path')
                if k in GLOBAL_CONFIG and v is not None:
                    GLOBAL_CONFIG[k] = v
                elif k in UI_TO_INTERNAL.values() and v is not None:
                    # map internal key back to UI label
                    ui_key = next((uk for uk, ik in UI_TO_INTERNAL.items() if ik == k), None)
                    if ui_key:
                        GLOBAL_CONFIG[ui_key] = v
                elif k in UI_TO_INTERNAL.keys() and v is not None:
                    # already an exact UI key
                    GLOBAL_CONFIG[k] = v
        apply_config_globally()
        print("✅ Using parameters from UI.")

        # Run Optuna if desired (respecting Data Path provided via UI)
        # Run optuna trials before running full backtest if you want optimized params
        try:
            study = run_optuna_trials(n_trials=15)
        except Exception as e:
            print("Optuna failed or was skipped:", e)

        # Run the backtest once using UI-provided configuration; pass the UI-updated globals
        df, bt_data, bt_trades, used_config = main_from_ui(config=GLOBAL_CONFIG)
        ui_ran = True
        # ensure GLOBAL_CONFIG reflects used_config (if used_config uses internal keys)
        if isinstance(used_config, dict):
            for k, v in used_config.items():
                if k in GLOBAL_CONFIG:
                    GLOBAL_CONFIG[k] = v
                else:
                    # maybe used_config is internal-keyed
                    ui_key = next((uk for uk, ik in UI_TO_INTERNAL.items() if ik == k), None)
                    if ui_key:
                        GLOBAL_CONFIG[ui_key] = v
        apply_config_globally()
    except Exception as e:
        print(f"⚠️ UI failed or closed ({e}). Using GLOBAL_CONFIG defaults.")

    # If UI did not run, ensure module globals reflect GLOBAL_CONFIG
    if not ui_ran:
        apply_config_globally()
        df = load_data(data_path)
        cfg = BacktestConfig.from_global()
        bt_data, bt_trades = backtest_sma(df, cfg=cfg)

    # ---------- Compute metrics ----------
    metrics = compute_metrics_from_equity(bt_data["Equity"])
    pf = profit_factor(bt_trades)
    win_trades = bt_trades[bt_trades['pct_return'] > 0] if not bt_trades.empty else pd.DataFrame()
    loss_trades = bt_trades[bt_trades['pct_return'] <= 0] if not bt_trades.empty else pd.DataFrame()
    num_trades = len(bt_trades)
    win_rate = len(win_trades) / num_trades if num_trades > 0 else np.nan
    avg_win = win_trades['pct_return'].mean() if not win_trades.empty else np.nan
    avg_loss = loss_trades['pct_return'].mean() if not loss_trades.empty else np.nan

    # ---------- print summary ----------
    # use module-level variables directly (don't shadow with local assignments)
    print("\n=== Backtest Summary ===")
    print(f"MA short: {MA_SHORT_DEFAULT}, MA long: {ma_long}")
    print(f"Initial capital: {initial_capital}")
    print(f"Number of trades: {num_trades}")
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"CAGR: {metrics['CAGR_pct']:.2f}%")
    print(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {pf:.2f}" if not np.isnan(pf) else "Profit Factor: NaN")
    print(f"Win rate: {win_rate:.2%}, Avg win: {avg_win:.2f}%, Avg loss: {avg_loss:.2f}%")

    # ---------- save outputs ----------
    # create a timestamped output folder: optimization&backtest+YYYY_MM_DD_HHMM
    now_str = datetime.now().strftime('%Y_%m_%d_%H%M')
    out_dir = os.path.join(os.getcwd(), f"optimization&backtest{now_str}")
    os.makedirs(out_dir, exist_ok=True)

    bt_trades.to_csv(os.path.join(out_dir, 'backtest_trades.csv'), index=False)
    bt_data[[
        'Close', f"SMA_{MA_SHORT_DEFAULT}", f"SMA_{ma_long}", 'ATR',
        'Entry', 'Exit', 'PositionQty', 'Cash', 'Equity'
    ]].to_csv(os.path.join(out_dir, 'backtest_daily.csv'))

    print(f"\nSaved: backtest_trades.csv, backtest_daily.csv -> {out_dir}")

    # ---------- optimization ----------
    opt_ma_shorts = [5, 10, 15, 20]
    opt_tp_values = [None, 0.03, 0.05]
    opt_df = optimize_grid(
        df,
        ma_shorts=opt_ma_shorts,
        tp_values=opt_tp_values,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        commission=commission,
        slippage=slippage,
        exit_on_ma_cross=exit_on_ma_cross,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
        use_trailing_atr=use_trailing_atr,
        atr_mult=atr_mult,
        atr_period=atr_period,
    )
    opt_df.to_csv(os.path.join(out_dir, 'optimization_results.csv'), index=False)
    print(f"Saved: optimization_results.csv -> {out_dir}")

    # ---------- write run log (config + summary) ----------
    # write out GLOBAL_CONFIG (UI-label keyed) as the final config for the run log
    final_config = GLOBAL_CONFIG.copy()
    log_path = os.path.join(out_dir, 'used_config_and_summary.txt')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Run timestamp: {datetime.now().isoformat()}\n\n")
            f.write("Used configuration:\n")
            for k, v in final_config.items():
                f.write(f"{k}: {v}\n")
            f.write("\nPerformance summary:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write(f"profit_factor: {pf}\n")
            f.write(f"num_trades: {num_trades}\n")
            f.write(f"win_rate: {win_rate}\n")
            f.write(f"avg_win_pct: {avg_win}\n")
            f.write(f"avg_loss_pct: {avg_loss}\n")

            # include best optimization row if present
            try:
                if 'opt_df' in locals() and not opt_df.empty:
                    f.write('\nBest optimization (by total_return_pct):\n')
                    best = opt_df.sort_values('total_return_pct', ascending=False).iloc[0]
                    for col in best.index:
                        f.write(f"{col}: {best[col]}\n")
            except Exception:
                # non-fatal: continue
                pass

            f.write('\nSaved files in folder:\n')
            for fn in sorted(os.listdir(out_dir)):
                f.write(f"- {fn}\n")
        print(f"Saved run log -> {log_path}")
    except Exception as e:
        print("Could not write run log:", e)

    # ---------- plots ----------
    plt.figure(figsize=(14, 6))
    plt.title('Price with SMAs and Trades')
    plt.plot(bt_data.index, bt_data['Close'], label='Close')
    plt.plot(bt_data.index, bt_data[f"SMA_{MA_SHORT_DEFAULT}"], label=f"SMA_{MA_SHORT_DEFAULT}")
    plt.plot(bt_data.index, bt_data[f'SMA_{ma_long}'], label=f'SMA_{ma_long}')
    ents = bt_data[bt_data['Entry']]
    exs = bt_data[bt_data['Exit']]
    if not ents.empty:
        plt.scatter(ents.index, bt_data.loc[ents.index, 'Close'], marker='^', s=80, label='Entry', zorder=5)
    if not exs.empty:
        plt.scatter(exs.index, bt_data.loc[exs.index, 'Close'], marker='v', s=80, label='Exit', zorder=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'price_sma_trades.png'))
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.title('Equity Curve')
    plt.plot(bt_data.index, bt_data['Equity'], label='Equity')
    plt.axhline(initial_capital, linestyle='--', label='Initial Capital')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'equity_curve.png'))
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.title('Distribution of Trade % Returns')
    if not bt_trades.empty:
        plt.hist(bt_trades['pct_return'].dropna(), bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trade_returns_hist.png'))
    plt.show()

    # heatmap
    try:
        opt_pivot = opt_df.pivot(index='MA_SHORT_DEFAULT', columns='take_profit_pct', values='total_return_pct')
        plt.figure(figsize=(6, 5))
        plt.title('Optimization: Total Return (%) heatmap')
        vals = opt_pivot.values
        plt.imshow(vals, aspect='auto')
        plt.yticks(ticks=np.arange(len(opt_pivot.index)), labels=opt_pivot.index)
        plt.xticks(ticks=np.arange(len(opt_pivot.columns)), labels=[str(x) for x in opt_pivot.columns])
        for (j, i), val in np.ndenumerate(vals):
            plt.text(i, j, f"{val:.1f}", ha='center', va='center', fontsize=8)
        plt.colorbar(label='Total Return (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'optimization_heatmap.png'))
        plt.show()
    except Exception as e:
        print("Could not plot optimization heatmap:", e)

    # print trades
    print("\n--- Trades ---")
    if bt_trades.empty:
        print("No trades executed.")
    else:
        print(bt_trades.to_string(index=False))

    print("\nAll outputs saved. Check PNG images and CSV files in current folder.")


if __name__ == '__main__':
    main()