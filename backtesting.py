from models import Operation
from portfolio_value import get_portfolio_value
from data_drift import calculate_drift_metrics, calculate_drift_pvalues
import pandas as pd

def backtest(data: pd.DataFrame,cash: float, x_train_f: pd.DataFrame | None = None, x_test_f: pd.DataFrame | None = None
) -> tuple[ pd.Series,float,float,int, int, int,int,list[dict], list[dict],]:

    stop_Loss = 0.06
    take_Profit = 0.13
    n_shares = 100
    COM = 0.125 / 100
    BORROW_RATE = (0.25 / 100)/252

    data = data.copy().dropna()

    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    port_hist = []
    wins = 0
    total_trades = 0
    buy = 0
    sell = 0
    hold = 0

    
    windows = 90
    steps = 60
    alpha = 0.05
    data_drift_results: list[dict] = []
    p_values_results: list[dict] = []

    
    for i, row in enumerate(data.itertuples(index=True)):

        # Close LONG positions
        for position in active_long_positions.copy():
            if (position.stop_loss > row.Close) or (position.take_profit < row.Close):
                pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
                if pnl >= 0:
                    wins += 1
                total_trades += 1
                cash += row.Close * position.n_shares * (1 - COM)
                active_long_positions.remove(position)

        # Charge daily borrow cost on SHORT positions
        for position in active_short_positions.copy():
            cash -= row.Close * position.n_shares * BORROW_RATE

        # Close SHORT positions
        for position in active_short_positions.copy():
            if (position.stop_loss < row.Close) or (position.take_profit > row.Close):
                pnl = (position.price - row.Close) * position.n_shares
                if pnl >= 0:
                    wins += 1
                total_trades += 1
                com = row.Close * position.n_shares * COM
                cash += pnl - com
                active_short_positions.remove(position)

        # Long signal
        if row.signal == 1:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                buy += 1
                active_long_positions.append(
                    Operation(
                        time=row.Index,
                        price=row.Close,
                        stop_loss=row.Close * (1 - stop_Loss),
                        take_profit=row.Close * (1 + take_Profit),
                        n_shares=n_shares,
                        type='LONG'
                    )
                )

        if row.signal == 2:
            cost = row.Close * n_shares * COM
            if cash > cost:
                cash -= cost
                sell += 1
                active_short_positions.append(
                    Operation(
                        time=row.Index,
                        price=row.Close,
                        stop_loss=row.Close * (1 + stop_Loss),
                        take_profit=row.Close * (1 - take_Profit),
                        n_shares=n_shares,
                        type='SHORT'
                    )
                )
        else:
            hold += 1

        
        port_hist.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares))

        # Data Drift 
        idx = i + 1
        if (x_train_f is not None) and (x_test_f is not None):
            if idx >= windows and (idx - windows) % steps == 0:
                initial = idx - windows
                end = idx

                # Ventana de comparación en TEST
                df_with_window = x_test_f.iloc[initial:end]

                # === Fechas (si el índice es convertible a datetime) ===
                # Tomamos la fecha de inicio (fila 'initial') y fin (fila 'end-1')
                try:
                    start_dt = pd.to_datetime(x_test_f.index[initial], errors="coerce")
                    end_dt   = pd.to_datetime(x_test_f.index[end - 1], errors="coerce")
                except Exception:
                    start_dt = pd.NaT
                    end_dt   = pd.NaT

                drift_metrics = calculate_drift_metrics(x_train_f, df_with_window, alpha=alpha)
                p_values = calculate_drift_pvalues(x_train_f, df_with_window)

                for rec in drift_metrics:
                    out_row = dict(rec)
                    out_row["WindowStartIdx"] = initial
                    out_row["WindowEndIdx"] = end
                    # Guarda las fechas si son válidas
                    if pd.notna(start_dt):
                        out_row["WindowStartDate"] = start_dt  # o start_dt.date() si solo quieres fecha
                    if pd.notna(end_dt):
                        out_row["WindowEndDate"] = end_dt
                    data_drift_results.append(out_row)

                for feat, pv in p_values.items():
                    pvr = {
                        "Feature": feat,
                        "KS_pvalue": pv,
                        "WindowStartIdx": initial,
                        "WindowEndIdx": end
                    }
                    if pd.notna(start_dt):
                        pvr["WindowStartDate"] = start_dt
                    if pd.notna(end_dt):
                        pvr["WindowEndDate"] = end_dt
                    p_values_results.append(pvr)


    # Close remaining  long positions at the end
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        if pnl >= 0:
            wins += 1
        total_trades += 1
        cash += row.Close * position.n_shares * (1 - COM)

    # Close remaining short positions at the end
    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares
        if pnl >= 0:
            wins += 1
        total_trades += 1
        com = row.Close * position.n_shares * COM
        cash += pnl - com

    active_long_positions = []
    active_short_positions = []

    win_rate = (wins / total_trades) if total_trades > 0 else 0.0

    return pd.Series(port_hist), cash, win_rate, buy, sell, hold, total_trades, data_drift_results, p_values_results
