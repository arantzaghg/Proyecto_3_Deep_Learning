import os
import numpy as np
import pandas as pd
from data_utils import get_target
from data_utils import get_asset_data, split_data, preprocess_data
from backtesting import backtest
from data_drift import (infer_features, plot_histograms_all, pvalue_plots_from_results, summarize_drift, save_top5_tables, plot_top5_drift_rate)

def run_drift_analysis():
    ticker = "DIS"
    alpha = 0.005
    out_dir = "drift_simple_report"
    os.makedirs(out_dir, exist_ok=True)

    # Datos y splits
    data = get_asset_data(ticker)
    train, test, val = split_data(data)

    # Features
    train, stats_np = preprocess_data(train, ticker, alpha=alpha, stage="train", include_close=False)
    test, _ = preprocess_data(test,  ticker, alpha=alpha, stage="test",  stats=stats_np, include_close=False)
    val,  _ = preprocess_data(val,   ticker, alpha=alpha, stage="val",   stats=stats_np, include_close=False)

    x_train_f, _ = get_target(train)
    x_test_f,  _ = get_target(test)
    x_val_f,   _ = get_target(val)

    # Backtest con drift
    _, _, _, _, _, _, _, _, pvals_test = backtest(
        data=test, cash=1_000_000, x_train_f=x_train_f, x_test_f=x_test_f)
    pvalue_plots_from_results(pvals_test, alpha, out_dir, split_name="test")

    _, _, _, _, _, _, _, _, pvals_val = backtest(
        data=val, cash=1_000_000, x_train_f=x_train_f, x_test_f=x_val_f)
    pvalue_plots_from_results(pvals_val, alpha, out_dir, split_name="val")

    # Summary tables y top5
    summary_test = summarize_drift(pvals_test, alpha=alpha)
    top5_test = save_top5_tables(summary_test, os.path.join(out_dir, "top5_test.csv"))
    plot_top5_drift_rate(top5_test, os.path.join(out_dir, "plots", "top5_test.png"), split_name="test")

    summary_val = summarize_drift(pvals_val, alpha=alpha)
    top5_val = save_top5_tables(summary_val, os.path.join(out_dir, "top5_val.csv"))
    plot_top5_drift_rate(top5_val, os.path.join(out_dir, "plots", "top5_val.png"), split_name="val")

    # Histograms
    feats = infer_features(x_train_f)
    plot_histograms_all(x_train_f, x_test_f, x_val_f, feats, out_dir)

   
    pd.DataFrame(pvals_test).to_csv(os.path.join(out_dir, "pvalues_test.csv"), index=False)
    pd.DataFrame(pvals_val).to_csv(os.path.join(out_dir, "pvalues_val.csv"), index=False)

    summary_test.to_csv(os.path.join(out_dir, "summary_test.csv"), index=False)
    summary_val.to_csv(os.path.join(out_dir, "summary_val.csv"), index=False)

if __name__ == "__main__":
    run_drift_analysis()
