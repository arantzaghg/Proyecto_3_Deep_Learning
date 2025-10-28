import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def _ensure_dir(path: str):

    """
    Ensure that a directory exists; create it if it does not.
    
    Parameters:
    path (str): The directory path to ensure.
    
    Returns:
    None
    """

    os.makedirs(path, exist_ok=True)

def infer_features(df: pd.DataFrame) -> list[str]:

    """
    Infer numerical feature columns from a DataFrame, excluding specific known columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list[str]: List of inferred feature column names.
    """
    
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"Open", "High", "Low", "Close", "Volume", "signal", "final_signal"}
    return [c for c in cols if c not in drop_cols]

def _ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:

    """ 
    Perform the Kolmogorov-Smirnov test between two arrays.
    
    Parameters:
    a (np.ndarray): First array of data.
    b (np.ndarray): Second array of data.
    
    Returns:
    tuple[float, float]: KS statistic and p-value.
    """

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(stat), float(p)

def calculate_drift_metrics(x_train_f: pd.DataFrame, x_test_f: pd.DataFrame, alpha: float = 0.05):

    """
    Calculate drift metrics between training and testing feature sets using the KS test.
    
    Parameters:
    x_train_f (pd.DataFrame): Training feature set.
    x_test_f (pd.DataFrame): Testing feature set.
    alpha (float): Significance level for drift detection.

    Returns:
    list[dict]: List of drift metrics for each feature.
    """
  
    ref = x_train_f.copy()
    cmp = x_test_f.copy()
    feats = infer_features(ref)
    out = []
    for f in feats:
        if f not in cmp.columns:
            continue
        stat, p = _ks(ref[f].astype(float).to_numpy(), cmp[f].astype(float).to_numpy())
        out.append({
            "Feature": f,
            "KS_stat": stat,
            "KS_pvalue": p,
            "DriftDetected": (p < alpha) if (p == p) else False
        })
    return out

def calculate_drift_pvalues(x_train_f: pd.DataFrame, x_test_f: pd.DataFrame):

    """
    Calculate KS test p-values between training and testing feature sets.
    
    Parameters:
    x_train_f (pd.DataFrame): Training feature set.
    x_test_f (pd.DataFrame): Testing feature set.

    Returns:
    dict: Dictionary of features and their corresponding KS p-values.
    """
   
    ref = x_train_f.copy()
    cmp = x_test_f.copy()
    feats = infer_features(ref)
    out = {}
    for f in feats:
        if f not in cmp.columns:
            continue
        _, p = _ks(ref[f].astype(float).to_numpy(), cmp[f].astype(float).to_numpy())
        out[f] = float(p) if p == p else np.nan
    return out

def plot_histograms_all(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], out_dir: str = "drift_simple_report"):

    """
    Plot histograms of features for training, testing, and validation datasets.

    Parameters:
    train_df (pd.DataFrame): Training dataset.
    test_df (pd.DataFrame): Testing dataset.
    val_df (pd.DataFrame | None): Validation dataset.
    features (list[str]): List of feature names to plot.
    out_dir (str): Output directory to save plots.
    
    Returns:
    None
    """
   
    _ensure_dir(os.path.join(out_dir, "plots"))

    for feat in features:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue

        a = train_df[feat].to_numpy(dtype=float)
        b = test_df[feat].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        c = None
        if val_df is not None and feat in val_df.columns:
            c = val_df[feat].to_numpy(dtype=float)
            c = c[~np.isnan(c)]

        if len(a) == 0:
            continue

        plt.figure(figsize=(8, 3))
        bins = 30
        plt.hist(a, bins=bins, alpha=0.4, density=True, label="train", color="navy")
        if len(b) > 0:
            plt.hist(b, bins=bins, alpha=0.4, density=True, label="test", color="cornflowerblue")
        if c is not None and len(c) > 0:
            plt.hist(c, bins=bins, alpha=0.4, density=True, label="val", color="skyblue")

        plt.title(f"Histogram — {feat} (train vs test vs val)")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{feat}_hist_all.png"), dpi=130)
        plt.close()

def pvalue_plots_from_results(pvalues_windows: list[dict], alpha: float, out_dir: str, split_name: str = "backtest"):

    """
    Generate and save p-value plots for each feature from KS test results.
    
    Parameters:
    pvalues_windows (list[dict]): List of dictionaries containing KS test results.
    alpha (float): Significance level for drift detection.
    out_dir (str): Output directory to save plots.
    split_name (str): Name of the data split for labeling plots.
    
    Returns:
    None
    """

    if not pvalues_windows:
        return

    df = pd.DataFrame(pvalues_windows)
    _ensure_dir(os.path.join(out_dir, "plots"))

    has_date = "WindowStartDate" in df.columns
    if has_date:
        df["WindowStartDate"] = pd.to_datetime(df["WindowStartDate"], errors="coerce")

    for feat, grp in df.groupby("Feature"):
        if has_date:
            grp = grp.sort_values(["WindowStartDate", "WindowStartIdx"])
            x_vals = grp["WindowStartDate"]
            x_label = "Dates"
        else:
            grp = grp.sort_values("WindowStartIdx")
            x_vals = grp["WindowStartIdx"]
            x_label = "Window start (idx)"

        plt.figure(figsize=(8, 3))
        plt.plot(x_vals, grp["KS_pvalue"], marker="o")
        plt.axhline(alpha, linestyle="--", color="blue")
        plt.title(f"KS p-values — {feat}")
        plt.xlabel(x_label)
        plt.ylabel("p-value")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{feat}_pvalues_{split_name}.png"), dpi=130)
        plt.close()

def summarize_drift(pvalues_windows: list[dict], alpha: float) -> pd.DataFrame:

    """
    Summarize drift statistics from KS test p-values across multiple windows.
    
    Parameters:
    pvalues_windows (list[dict]): List of dictionaries containing KS test results.
    alpha (float): Significance level for drift detection.

    Returns:
    pd.DataFrame: Summary DataFrame with drift statistics for each feature.
    """
    
    if not pvalues_windows:
        return pd.DataFrame(columns=["Feature","windows","below_alpha","drift_rate","min_pvalue","median_pvalue"])

    df = pd.DataFrame(pvalues_windows)
    if "Feature" not in df.columns or "KS_pvalue" not in df.columns:
        raise ValueError("Se esperaban columnas 'Feature' y 'KS_pvalue' en pvalues_windows")

    agg = (
        df.groupby("Feature", as_index=False)
          .agg(
              windows=("KS_pvalue", "size"),
              below_alpha=("KS_pvalue", lambda s: np.sum(s < alpha)),
              drift_rate=("KS_pvalue", lambda s: float(np.mean(s < alpha))),
              min_pvalue=("KS_pvalue", "min"),
              median_pvalue=("KS_pvalue", "median"),
          )
    )
    agg = agg.sort_values(["drift_rate", "min_pvalue"], ascending=[False, True], kind="mergesort")
    return agg.reset_index(drop=True)

def save_top5_tables(summary_df: pd.DataFrame, out_csv_path: str) -> pd.DataFrame:

    """
    Save the top-5 features with the highest drift rates to a CSV file.

    Parameters:
    summary_df (pd.DataFrame): Summary DataFrame with drift statistics.
    out_csv_path (str): Output CSV file path.

    Returns:
    pd.DataFrame: DataFrame containing the top-5 features with highest drift rates.
    """

    top5 = summary_df.head(5).copy()
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    top5.to_csv(out_csv_path, index=False)
    return top5

def plot_top5_drift_rate(top5_df: pd.DataFrame, out_png_path: str, split_name: str):

    """
    Plot and save a horizontal bar chart of the top-5 features with the highest drift rates.
    
    Parameters:
    top5_df (pd.DataFrame): DataFrame containing the top-5 features with highest drift rates.
    out_png_path (str): Output PNG file path.
    split_name (str): Name of the data split for labeling the plot.

    Returns:
    None
    """
    
    if top5_df.empty:
        return
    plt.figure(figsize=(8, 3))
    plot_df = top5_df.sort_values("drift_rate", ascending=True)
    plt.barh(plot_df["Feature"], plot_df["drift_rate"], color="cornflowerblue")
    plt.title(f"Top-5 drift rate — {split_name}")
    plt.xlabel("Window ratio with p < α")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=130)
    plt.close()