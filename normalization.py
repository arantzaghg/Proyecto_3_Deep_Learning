

from typing import Tuple, Dict
import pandas as pd
import numpy as np

def normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    params = {}

    # Escalamiento 0-1
    for col in ['RSI_7', 'RSI_14', 'RSI_21']:
        if col in df.columns:
            df[col] = df[col] / 100

    for col in ['Stoch_12', 'Stoch_26']:
        if col in df.columns:
            df[col] = df[col] / 100

    # Normalización Z-score
    for col in ['ROC_10', 'ROC_20']:
        if col in df.columns:
            params[f'mean_{col}'] = df[col].mean()
            params[f'std_{col}'] = df[col].std()
            df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # Relativo al precio
    for col in ['EMA_10', 'EMA_21']:
        if col in df.columns and 'Close' in df.columns:
            df[col] = df[col] / df['Close']

    # ATR normalizado
    if 'ATR_14' in df.columns and 'Close' in df.columns:
        df['ATR_14'] = df['ATR_14'] / df['Close']

    # Posición en bandas de Bollinger
    if {'BB_high_20', 'BB_low_20'}.issubset(df.columns):
        df['BB_pos_20'] = (df['Close'] - df['BB_low_20']) / (df['BB_high_20'] - df['BB_low_20'])
    if {'BB_high_15', 'BB_low_15'}.issubset(df.columns):
        df['BB_pos_15'] = (df['Close'] - df['BB_low_15']) / (df['BB_high_15'] - df['BB_low_15'])

    # Posición en canal Donchian
    if {'DON_high_20', 'DON_low_20'}.issubset(df.columns):
        df['DON_pos_20'] = (df['Close'] - df['DON_low_20']) / (df['DON_high_20'] - df['DON_low_20'])

    # Escalamiento 0-1
    if 'MFI_14' in df.columns:
        df['MFI_14'] = df['MFI_14'] / 100

    # Min-Max
    if 'CMF_20' in df.columns:
        params['min_CMF_20'] = df['CMF_20'].min()
        params['max_CMF_20'] = df['CMF_20'].max()
        df['CMF_20'] = (df['CMF_20'] - params['min_CMF_20']) / (params['max_CMF_20'] - params['min_CMF_20'])

    # Normalización Z-score
    for col in ['OBV', 'VPT', 'ADI']:
        if col in df.columns:
            params[f'mean_{col}'] = df[col].mean()
            params[f'std_{col}'] = df[col].std()
            df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # Limpieza
    cols_to_drop = ['BB_high_20', 'BB_low_20', 'BB_high_15', 'BB_low_15', 'DON_high_20', 'DON_low_20']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df, params

