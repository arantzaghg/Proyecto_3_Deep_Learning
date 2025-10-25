
import os
import pandas as pd
from data_utils import get_asset_data, split_data, preprocess_data
from data_drift import drift_table_and_plots, summarize_top5, plot_histograms_all, infer_features

def run_drift_analysis():
    
    ticker = "SONY"
    window = "60D"
    alpha = 0.010
    out_dir = "drift_simple_report"

    data = get_asset_data(ticker)
    train, test, val = split_data(data)

    train_np, stats_np = preprocess_data(train, ticker, alpha=0.010, stage="train", include_close=False)
    test_np, _        = preprocess_data(test,  ticker, alpha=0.010, stage="test",  stats=stats_np, include_close=False)
    val_np,  _        = preprocess_data(val,   ticker, alpha=0.010, stage="val",   stats=stats_np, include_close=False)

    features = infer_features(train_np)
    plot_histograms_all(train_np, test_np, val_np, features, out_dir=out_dir)

    df_test = drift_table_and_plots(train_np, test_np, split_name="test", window=window, alpha=alpha, out_dir=out_dir)
    df_val  = drift_table_and_plots(train_np, val_np,  split_name="val",  window=window, alpha=alpha, out_dir=out_dir)

    top5 = summarize_top5(df_test, df_val, top_k=5)
    top5_path = os.path.join(out_dir, "top5.csv")
    top5.to_csv(top5_path, index=False)

    print("\nTop-5 features con mayor drift detectado:")
    if len(top5) == 0:
        print("No hubo suficientes datos o no se detectó drift.")
    else:
        for i, row in top5.reset_index(drop=True).iterrows():
            print(f"{i+1}. {row['Feature']}  | min p-value={row['min_pvalue']:.3g}  | ventanas con drift={int(row['drift_windows'])}")

if __name__ == "__main__":
    run_drift_analysis()
