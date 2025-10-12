import numpy as np
import pandas as pd

import SMA_strategy

# from SMA_strategy import backtest_sma
# from SMA_strategy import compute_metrics_from_equity, profit_factor

def optimize_grid(df, ma_shorts=[5, 10, 15, 20], tp_values=[None, 0.03, 0.05], **kwargs):
    """
    اجرای گرید سرچ روی پارامترهای MA کوتاه و درصد تارگت سود.
    برای هر ترکیب، بک‌تست را اجرا می‌کند و متریک‌ها را در یک DataFrame برمی‌گرداند.
    """
    results = []

    for ma in ma_shorts:
        for tp in tp_values:
            data_bt, trades = SMA_strategy.backtest_sma(
                df,
                MA_SHORT_DEFAULT=ma,
                ma_long=ma * 2,
                take_profit_pct=tp,
                **kwargs
            )
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

    return pd.DataFrame(results)
    print ( results_df )
    
