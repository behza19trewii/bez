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


# ------------------------- Utilities -------------------------
def load_data(path_csv = r"F:\sell\EURUSD1440_d.csv", path_json='data.json'):
    """Robust loader: tries CSV then JSON; special handling for whitespace-separated files
    like the attached `data3.csv` where rows may look like:
      2009-09-25 16:00   1.46818 1.46839 1.46456 1.46685       14219

    Behavior:
    - If file parses normally with delimiters, use parsed columns.
    - If file reads as a single column, split on whitespace into tokens and
      try to reconstruct columns (Date [+ Time], Open, High, Low, Close, Volume).
    - Prefer the 'open' column (case-insensitive) and map it to 'Close' so
      downstream code that expects 'Close' continues to work.
    """
    if os.path.exists(path_csv):
        import re as _re

        # First attempt: pandas auto-read (comma or obvious delimiter)
        try:
            df = pd.read_csv(path_csv, engine='python')
        except Exception:
            # fallback to a simple read
            df = pd.read_csv(path_csv, header=None, engine='python')

        # If file was read into a single column, try splitting by whitespace
        if df.shape[1] == 1:
            # read raw lines and split
            with open(path_csv, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            rows = []
            header_tokens = None
            for i, ln in enumerate(lines):
                toks = _re.split(r"\s+", ln)
                # detect header-like first row (contains non-numeric words)
                if i == 0 and any(_re.match(r"[A-Za-z]", t) for t in toks[:3]):
                    header_tokens = [t.strip() for t in toks]
                    continue
                # combine date+time if time token present (e.g., '16:00')
                if len(toks) >= 6 and _re.match(r"^\d{1,2}:\d{2}$", toks[1]):
                    date_str = toks[0] + ' ' + toks[1]
                    rest = toks[2:]
                    rows.append([date_str] + rest)
                elif len(toks) >= 5 and _re.match(r"^\d{4}-\d{2}-\d{2}$", toks[0]) and _re.match(r"^\d{1,2}:\d{2}$", toks[1]):
                    date_str = toks[0] + ' ' + toks[1]
                    rest = toks[2:]
                    rows.append([date_str] + rest)
                else:
                    # generic: keep tokens as-is
                    rows.append(toks)

            if not rows:
                raise ValueError('CSV appears empty after parsing')

            # Decide column names
            maxcols = max(len(r) for r in rows)
            if header_tokens and len(header_tokens) >= 2:
                colnames = header_tokens
            else:
                # typical mapping: Date[,Time], Open, High, Low, Close[,Volume]
                if maxcols >= 6:
                    colnames = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
                elif maxcols == 5:
                    colnames = ['DateTime', 'Open', 'High', 'Low', 'Close']
                elif maxcols == 4:
                    colnames = ['DateTime', 'Open', 'High', 'Low']
                else:
                    # fallback: generic numbered columns
                    colnames = [f'col{i}' for i in range(maxcols)]

            # normalize rows to same length
            norm_rows = [r + [None] * (len(colnames) - len(r)) if len(r) < len(colnames) else r[:len(colnames)] for r in rows]
            df = pd.DataFrame(norm_rows, columns=colnames)

        # strip column names
        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = [c.lower() for c in df.columns]

        # Identify date/datetime column: prefer name with 'date' or 'time' or 'datetime'
        date_col = None
        for target in ['datetime', 'date', 'time']:
            if target in cols_lower:
                date_col = df.columns[cols_lower.index(target)]
                break

        # If not found, try to detect the column with most datetime-parsable values
        if date_col is None:
            best_score = 0.0
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                    score = parsed.notna().mean()
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    date_col = c

        if date_col is None:
            raise ValueError('Could not detect date/datetime column in CSV')

        # coerce to datetime (combine date and time if needed is already handled above)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        df = df.set_index(date_col).sort_index()

        # Prefer 'open' column for calculations; map it to 'Close'
        if 'open' in cols_lower:
            open_col = df.columns[cols_lower.index('open')]
            df = df[[open_col]].copy()
            df.columns = ['Close']
        elif 'close' in cols_lower:
            close_col = df.columns[cols_lower.index('close')]
            df = df[[close_col]].copy()
            df.columns = ['Close']
        else:
            # fallback: pick first numeric column
            numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().any()]
            if numeric_cols:
                df = df[[numeric_cols[0]]].copy()
                df.columns = ['Close']
            else:
                raise ValueError("CSV found but no numeric price column (open/close) detected")

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df[['Close']].copy()
    elif os.path.exists(path_json):
        df = pd.read_json(path_json)
        cols_lower = [c.lower() for c in df.columns]
        if 'date' in cols_lower:
            date_col = df.columns[cols_lower.index('date')]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()

        # همین اولویت‌بندی را برای JSON نیز انجام می‌دهیم
        if 'open' in cols_lower:
            open_col = df.columns[cols_lower.index('open')]
            df = df[[open_col]].copy()
            df.columns = ['Close']
        else:
            if 'close' in cols_lower:
                close_col = df.columns[cols_lower.index('close')]
                df = df[[close_col]].copy()
                df.columns = ['Close']
            else:
                if df.shape[1] == 1:
                    df.columns = ['Close']
                else:
                    raise ValueError("JSON found but no 'open' or 'Close' column. Provide at least Date and Open/Close.")
        return df[['Close']].copy()
    else:
        # دادهٔ ساختگی (برای تست فوری)
        np.random.seed(42)
        n = 1000
        dt_index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')  # business days
        returns = np.random.normal(loc=0.0003, scale=0.01, size=n)
        price = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({'Close': price}, index=dt_index)
        print("No data file found. Generated synthetic price series for demo.")
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
def backtest_sma(df, ma_short=10, ma_long=None, initial_capital=10000.0, position_size_pct=0.5,
                 commission=0.0, slippage=0.0, exit_on_ma_cross=True, take_profit_pct=None,
                 stop_loss_pct=None, max_holding_days=None, use_trailing_atr=True,
                 atr_mult=3.0, atr_period=14):
    """پیاده‌سازی بک‌تست:
    - ورود: وقتی SMA_short امروز بالاتر از SMA_long شد و دیروز <= بود.
    - خروج: ترکیبی از MA-cross down یا TP/SL یا Trailing ATR یا حداکثر مدت نگهداری.
    - اجرای قیمت در Close روز سیگنال (برای جلوگیری از lookahead از سیگنال دیروز استفاده شده).
    """
    data = df.copy()
    if ma_long is None:
        ma_long = ma_short * 2

    data[f'SMA_{ma_short}'] = data['Close'].rolling(window=ma_short, min_periods=1).mean()
    data[f'SMA_{ma_long}'] = data['Close'].rolling(window=ma_long, min_periods=1).mean()
    data['ATR'] = atr(data, n=atr_period)

    trades = []
    in_pos = False
    qty = 0.0
    cash = initial_capital
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
        sma_s_today = float(data[f'SMA_{ma_short}'].iloc[i])
        sma_l_today = float(data[f'SMA_{ma_long}'].iloc[i])
        sma_s_yest = float(data[f'SMA_{ma_short}'].iloc[i - 1])
        sma_l_yest = float(data[f'SMA_{ma_long}'].iloc[i - 1])

        entry_signal = (sma_s_yest <= sma_l_yest) and (sma_s_today > sma_l_today)
        exit_signal_ma = (sma_s_yest >= sma_l_yest) and (sma_s_today < sma_l_today)

        # ورود
        if (not in_pos) and entry_signal:
            alloc = cash * position_size_pct
            exec_price = close_today * (1 + slippage)
            qty = alloc / exec_price if exec_price > 0 else 0.0
            cash -= qty * exec_price
            cash -= commission
            in_pos = True
            entry_price = exec_price
            # trailing stop اولیه
            if use_trailing_atr:
                atr_today = data['ATR'].iloc[i]
                trailing_stop = entry_price - atr_mult * atr_today
            data.at[today, 'Entry'] = True
            trades.append({
                'entry_date': today, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None,
                'qty': qty, 'pnl': None, 'pct_return': None, 'holding_days': None, 'exit_reason': None,
                'cash_before': cash + qty * exec_price
            })

        # بررسی خروج‌ها
        if in_pos:
            trade = trades[-1]
            if use_trailing_atr:
                atr_today = data['ATR'].iloc[i]
                proposed_stop = close_today - atr_mult * atr_today
                if trailing_stop is None or proposed_stop > trailing_stop:
                    trailing_stop = proposed_stop
            tp_hit = False
            sl_hit = False
            ts_hit = False
            max_hold_hit = False
            if take_profit_pct is not None and entry_price:
                tp_hit = (close_today / entry_price - 1) >= take_profit_pct
            if stop_loss_pct is not None and entry_price:
                sl_hit = (close_today / entry_price - 1) <= stop_loss_pct
            if use_trailing_atr and trailing_stop is not None and close_today <= trailing_stop:
                ts_hit = True
            if max_holding_days is not None:
                hdays = (today - trade['entry_date']).days
                if hdays >= max_holding_days:
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
                exit_price = close_today * (1 - slippage)
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
        data.at[data.index[0], 'Equity'] = initial_capital
    data['Equity'].ffill(inplace=True)

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
    results = []
    for ma in ma_shorts:
        for tp in tp_values:
            data_bt, trades = backtest_sma(df, ma_short=ma, ma_long=ma * 2, take_profit_pct=tp, **kwargs)
            metrics = compute_metrics_from_equity(data_bt['Equity'])
            pf = profit_factor(trades)
            results.append({
                'ma_short': ma,
                'take_profit_pct': tp if tp is not None else np.nan,
                'total_return_pct': metrics['total_return_pct'],
                'CAGR_pct': metrics['CAGR_pct'],
                'sharpe': metrics['sharpe'],
                'sortino': metrics['sortino'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'profit_factor': pf,
                'num_trades': len(trades)
            })
    return pd.DataFrame(results)


# ------------------------- Main (config + run) -------------------------
def main():
    # ---------- Default user config ----------
    INITIAL_CAPITAL = 10000.0
    POSITION_PCT = 0.5              # هر معامله 50% از سرمایه جاری
    COMMISSION = 0.0
    SLIPPAGE = 0.0005
    MA_SHORT_DEFAULT = 10
    USE_TRAILING_ATR = True         # فعال بودن Trailing ATR به‌صورت پیش‌فرض
    ATR_MULT = 3.0
    ATR_PERIOD = 14
    EXIT_ON_MA = True
    STOP_LOSS = -0.05               # 5% stop loss (قابل تغییر)
    TAKE_PROFIT = 0.05              # 5% take profit (قابل تغییر)
    MAX_HOLD = None                 # حداکثر نگهداری (روز)؛ None یعنی غیر فعال

    # ---------- load data ----------
    df = load_data()

    # ---------- run backtest ----------
    bt_data, bt_trades = backtest_sma(df,
                                      ma_short=MA_SHORT_DEFAULT,
                                      ma_long=MA_SHORT_DEFAULT * 2,
                                      initial_capital=INITIAL_CAPITAL,
                                      position_size_pct=POSITION_PCT,
                                      commission=COMMISSION,
                                      slippage=SLIPPAGE,
                                      exit_on_ma_cross=EXIT_ON_MA,
                                      take_profit_pct=TAKE_PROFIT,
                                      stop_loss_pct=STOP_LOSS,
                                      max_holding_days=MAX_HOLD,
                                      use_trailing_atr=USE_TRAILING_ATR,
                                      atr_mult=ATR_MULT,
                                      atr_period=ATR_PERIOD)

    # ---------- metrics ----------
    metrics = compute_metrics_from_equity(bt_data['Equity'])
    pf = profit_factor(bt_trades)
    win_trades = bt_trades[bt_trades['pct_return'] > 0] if not bt_trades.empty else pd.DataFrame()
    loss_trades = bt_trades[bt_trades['pct_return'] <= 0] if not bt_trades.empty else pd.DataFrame()
    num_trades = len(bt_trades)
    win_rate = len(win_trades) / num_trades if num_trades > 0 else np.nan
    avg_win = win_trades['pct_return'].mean() if not win_trades.empty else np.nan
    avg_loss = loss_trades['pct_return'].mean() if not loss_trades.empty else np.nan

    # ---------- print summary ----------
    print("\n=== Backtest Summary ===")
    print(f"MA short: {MA_SHORT_DEFAULT}, MA long: {MA_SHORT_DEFAULT * 2}")
    print(f"Initial capital: {INITIAL_CAPITAL}")
    print(f"Number of trades: {num_trades}")
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"CAGR: {metrics['CAGR_pct']:.2f}%")
    print(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {pf:.2f}" if not np.isnan(pf) else "Profit Factor: NaN")
    print(f"Win rate: {win_rate:.2%}, Avg win: {avg_win:.2f}%, Avg loss: {avg_loss:.2f}%")

    # ---------- save outputs ----------
    # create a timestamped output folder: optimization+YYYYMMDD_HHMM
    now_str = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = os.path.join(os.getcwd(), f"optimization&backtest{now_str}")
    os.makedirs(out_dir, exist_ok=True)

    bt_trades.to_csv(os.path.join(out_dir, 'backtest_trades.csv'), index=False)
    bt_data[[
        'Close', f'SMA_{MA_SHORT_DEFAULT}', f'SMA_{MA_SHORT_DEFAULT * 2}', 'ATR',
        'Entry', 'Exit', 'PositionQty', 'Cash', 'Equity'
    ]].to_csv(os.path.join(out_dir, 'backtest_daily.csv'))
    print(f"\nSaved: backtest_trades.csv, backtest_daily.csv -> {out_dir}")

    # ---------- optimization ----------
    opt_ma_shorts = [5, 10, 15, 20]
    opt_tp_values = [None, 0.03, 0.05]
    opt_df = optimize_grid(df, ma_shorts=opt_ma_shorts, tp_values=opt_tp_values,
                           initial_capital=INITIAL_CAPITAL,
                           position_size_pct=POSITION_PCT,
                           commission=COMMISSION,
                           slippage=SLIPPAGE,
                           exit_on_ma_cross=EXIT_ON_MA,
                           stop_loss_pct=STOP_LOSS,
                           max_holding_days=MAX_HOLD,
                           use_trailing_atr=USE_TRAILING_ATR,
                           atr_mult=ATR_MULT,
                           atr_period=ATR_PERIOD)
    opt_df.to_csv(os.path.join(out_dir, 'optimization_results.csv'), index=False)
    print(f"Saved: optimization_results.csv -> {out_dir}")

    # ---------- plots ----------
    plt.figure(figsize=(14, 6))
    plt.title('Price with SMAs and Trades')
    plt.plot(bt_data.index, bt_data['Close'], label='Close')
    plt.plot(bt_data.index, bt_data[f'SMA_{MA_SHORT_DEFAULT}'], label=f'SMA_{MA_SHORT_DEFAULT}')
    plt.plot(bt_data.index, bt_data[f'SMA_{MA_SHORT_DEFAULT * 2}'], label=f'SMA_{MA_SHORT_DEFAULT * 2}')
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
    plt.axhline(INITIAL_CAPITAL, linestyle='--', label='Initial Capital')
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
        opt_pivot = opt_df.pivot(index='ma_short', columns='take_profit_pct', values='total_return_pct')
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