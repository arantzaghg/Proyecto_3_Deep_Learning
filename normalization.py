
import pandas as pd

import pandas as pd

def get_normal_stats(data: pd.DataFrame, include_close: bool = False) -> tuple[pd.DataFrame, dict]:
    data = data.copy()
    stats = {}

    # Scale to range [0, 1]
    for col in ['RSI_7', 'RSI_14', 'RSI_21']:
        if col in data.columns:
            data[col] = data[col] / 100

    for col in ['Stoch_12', 'Stoch_26', 'Stoch_40']:
        if col in data.columns:
            data[col] = data[col] / 100

    # Z-score normalization
    for col in ['ROC_10', 'ROC_20', 'ROC_45']:
        if col in data.columns:
            stats[f'mean_{col}'] = data[col].mean()
            stats[f'std_{col}'] = data[col].std()
            data[col] = (data[col] - stats[f'mean_{col}']) / stats[f'std_{col}']

    # Relative to closing price
    for col in ['EMA_10', 'EMA_21']:
        if col in data.columns and 'Close' in data.columns:
            data[col] = data[col] / data['Close']

    # ATR normalized by price
    if 'ATR_14' in data.columns and 'Close' in data.columns:
        data['ATR_14'] = data['ATR_14'] / data['Close']

    # Relative position within Bollinger Bands
    data['BB_pos_20'] = (data['Close'] - data['BB_low_20']) / (data['BB_high_20'] - data['BB_low_20'])
    data['BB_pos_15'] = (data['Close'] - data['BB_low_15']) / (data['BB_high_15'] - data['BB_low_15'])
        
    # Position within Donchian channel
    if {'DON_high_20', 'DON_low_20'}.issubset(data.columns):
        data['DON_pos_20'] = (data['Close'] - data['DON_low_20']) / (data['DON_high_20'] - data['DON_low_20'])

    # Scale to range [0, 1]
    if 'MFI_14' in data.columns:
        data['MFI_14'] = data['MFI_14'] / 100

    # Min-Max normalization
    if 'CMF_21' in data.columns:
        stats['min_CMF_21'] = data['CMF_21'].min()
        stats['max_CMF_21'] = data['CMF_21'].max()
        data['CMF_21'] = (data['CMF_21'] - stats['min_CMF_21']) / (stats['max_CMF_21'] - stats['min_CMF_21'])

    # Z-score normalization for volume-based indicators
    for col in ['OBV', 'VPT', 'ADI']:
        if col in data.columns:
            stats[f'mean_{col}'] = data[col].mean()
            stats[f'std_{col}'] = data[col].std()
            data[col] = (data[col] - stats[f'mean_{col}']) / stats[f'std_{col}']

    # Normalize 'Close' price
    if include_close:
        mean_close = data['Close'].mean()
        std_close = data['Close'].std()

        data['Close'] = (data['Close'] - mean_close) / std_close

        stats['mean_close'] = mean_close
        stats['std_close'] = std_close
        
    # Cleanup: remove intermediate columns
    cols_to_drop = ['BB_high_20', 'BB_low_20', 'BB_high_15', 'BB_low_15', 'DON_high_20', 'DON_low_20']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    return data, stats


def normalize_data(data: pd.DataFrame, stats: dict, include_close: bool = False) -> pd.DataFrame:
    data = data.copy()

    # Scale to range [0, 1]
    for col in ['RSI_7', 'RSI_14', 'RSI_21']:
        if col in data.columns:
            data[col] = data[col] / 100

    for col in ['Stoch_12', 'Stoch_26', 'Stoch_40']:
        if col in data.columns:
            data[col] = data[col] / 100

    # Z-score normalization
    for col in ['ROC_10', 'ROC_20', 'ROC_45']:
        if col in data.columns:
            data[col] = (data[col] - stats[f'mean_{col}']) / stats[f'std_{col}']

    # Relative to closing price
    for col in ['EMA_10', 'EMA_21']:
        if col in data.columns and 'Close' in data.columns:
            data[col] = data[col] / data['Close']

    # ATR normalized by price
    if 'ATR_14' in data.columns and 'Close' in data.columns:
        data['ATR_14'] = data['ATR_14'] / data['Close']

    # Position within Bollinger Bands
    data['BB_pos_20'] = (data['Close'] - data['BB_low_20']) / (data['BB_high_20'] - data['BB_low_20'])
    data['BB_pos_15'] = (data['Close'] - data['BB_low_15']) / (data['BB_high_15'] - data['BB_low_15'])

    # Position within Donchian channel
    if {'DON_high_20', 'DON_low_20'}.issubset(data.columns):
        data['DON_pos_20'] = (data['Close'] - data['DON_low_20']) / (data['DON_high_20'] - data['DON_low_20'])

    # Scale to range [0, 1]
    if 'MFI_14' in data.columns:
        data['MFI_14'] = data['MFI_14'] / 100

    # Min-Max normalization
    if 'CMF_21' in data.columns:
        data['CMF_21'] = (data['CMF_21'] - stats['min_CMF_21']) / (stats['max_CMF_21'] - stats['min_CMF_21'])
    
    # Z-score normalization for volume-based indicators
    for col in ['OBV', 'VPT', 'ADI']:
        if col in data.columns:
            data[col] = (data[col] - stats[f'mean_{col}']) / stats[f'std_{col}']

    # Normalize 'Close' price
    if include_close:
        data['Close'] = (data['Close'] - stats['mean_close']) / stats['std_close']
    
    # Cleanup: remove intermediate columns
    cols_to_drop = ['BB_high_20', 'BB_low_20', 'BB_high_15', 'BB_low_15', 'DON_high_20', 'DON_low_20']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    return data


