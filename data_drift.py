
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for col in ["Date", "date", "Timestamp", "timestamp", "time", "TIME"]:
        if col in df.columns:
            out = df.copy()
            out.index = pd.to_datetime(out[col], errors="coerce")
            return out
    raise ValueError("Se requiere un índice de tiempo o una columna de fecha ('Date').")

def infer_features(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir precios y etiquetas
    drop_cols = {"Open", "High", "Low", "Close", "Volume", "signal"}
    features = [c for c in cols if c not in drop_cols]

    return features


def ks_against_train(train_arr: np.ndarray, other_arr: np.ndarray) -> tuple[float, float]:
    a = train_arr[~np.isnan(train_arr)]
    b = other_arr[~np.isnan(other_arr)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(stat), float(p)

def time_windows(df: pd.DataFrame, window: str = "60D") -> list[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    out = []
    for period, chunk in df.sort_index().groupby(pd.Grouper(freq=window)):
        if isinstance(period, pd.Period):
            start, end = period.start_time, period.end_time
        else:
            start = pd.Timestamp(period)
            end = start + pd.tseries.frequencies.to_offset(window) - pd.Timedelta(nanoseconds=1)
        if len(chunk) > 0:
            out.append((start, end, chunk))
    return out

def plot_pvalues(feature: str, recs: list[dict], alpha: float, out_png: str):
    recs = sorted(recs, key=lambda r: r["WindowStart"])
    xs = [r["WindowStart"] for r in recs]
    ys = [r["KS_pvalue"] for r in recs]
    plt.figure(figsize=(8, 3))
    plt.plot(xs, ys, marker="o")
    plt.axhline(alpha, linestyle="--")
    plt.title(f"KS p-values — {feature}")
    plt.ylabel("p-value")
    plt.xlabel("Window start")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()

def drift_table_and_plots(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    split_name: str = "test",
    window: str = "30D",
    alpha: float = 0.05,
    out_dir: str = "drift_simple_report"
) -> pd.DataFrame:
    
    train_df = _ensure_time_index(train_df.copy())
    other_df = _ensure_time_index(other_df.copy())

    features = infer_features(train_df)
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "plots"))

    tw = time_windows(other_df, window=window)
    rows = []
    for feat in features:
        train_arr = train_df[feat].astype(float).to_numpy()
        recs_feat = []
        for (ws, we, chunk) in tw:
            stat, p = ks_against_train(train_arr, chunk[feat].astype(float).to_numpy())
            rows.append({
                "Feature": feat,
                "Split": split_name,
                "WindowStart": ws,
                "WindowEnd": we,
                "KS_stat": stat,
                "KS_pvalue": p,
                "DriftDetected": (p < alpha) if (p == p) else False  # p==p filtra NaN
            })
            recs_feat.append({"WindowStart": ws, "KS_pvalue": p})
        
        plot_pvalues(
            feat,
            recs_feat,
            alpha,
            os.path.join(out_dir, "plots", f"{feat}_pvalues_{split_name}.png")
        )

    df_out = pd.DataFrame(rows).sort_values(["Feature", "WindowStart"])
    df_out.to_csv(os.path.join(out_dir, f"drift_stats_{split_name}.csv"), index=False)
    return df_out

def summarize_top5(df_test: pd.DataFrame, df_val: pd.DataFrame = None, top_k: int = 5) -> pd.DataFrame:
   
    frames = [df_test]
    if df_val is not None:
        frames.append(df_val)
    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby("Feature", as_index=False)
        .agg(min_pvalue=("KS_pvalue", "min"),
             drift_windows=("DriftDetected", "sum"))
        .sort_values(["min_pvalue", "drift_windows"], ascending=[True, False])
        .head(top_k)
    )
    return agg

def plot_histograms_all(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame,
                        features: list[str], out_dir: str = "drift_simple_report"):
   

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    for feat in features:
        if feat not in train_df.columns or feat not in test_df.columns or feat not in val_df.columns:
            continue

        a = train_df[feat].to_numpy(dtype=float)
        b = test_df[feat].to_numpy(dtype=float)
        c = val_df[feat].to_numpy(dtype=float)

        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        c = c[~np.isnan(c)]

        if len(a) == 0: 
            continue

        plt.figure(figsize=(8, 3))
        bins = 30
    
        plt.hist(a, bins=bins, alpha=0.4, density=True, label="train")
        if len(b) > 0:
            plt.hist(b, bins=bins, alpha=0.4, density=True, label="test")
        if len(c) > 0:
            plt.hist(c, bins=bins, alpha=0.4, density=True, label="val")

        plt.title(f"Histogram — {feat} (train vs test vs val)")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{feat}_hist_all.png"), dpi=130)
        plt.close()

