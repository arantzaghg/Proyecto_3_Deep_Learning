import pandas as pd
import ta


def get_indicators(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()

    ## Indicadores de Momentum

    data["RSI_7"] = ta.momentum.RSIIndicator(close=data["Close"], window=7).rsi()
    data["RSI_14"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
    data["RSI_21"] = ta.momentum.RSIIndicator(close=data["Close"], window=21).rsi()

    data["ROC_10"] = ta.momentum.ROCIndicator(close=data["Close"], window=10).roc()
    data["ROC_20"] = ta.momentum.ROCIndicator(close=data["Close"], window=20).roc()
    data["ROC_45"] = ta.momentum.ROCIndicator(close=data["Close"], window=45).roc()

    data["EMA_10"] = ta.trend.EMAIndicator(close=data["Close"], window=10).ema_indicator()
    data["EMA_21"] = ta.trend.EMAIndicator(close=data["Close"], window=21).ema_indicator()

    data['Stoch_12'] = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=12).stoch()
    data['Stoch_26'] = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=26).stoch()
    data['Stoch_40'] = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=40).stoch()
    
    ## Indicadores de Volatilidad

    data['ATR_14'] =  ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()

    bb_20 = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["BB_high_20"] = bb_20.bollinger_hband()   
    data["BB_low_20"] = bb_20.bollinger_lband()  

    bb_15 = ta.volatility.BollingerBands(close=data["Close"], window=15, window_dev=2)
    data["BB_high_15"] = bb_15.bollinger_hband()   
    data["BB_low_15"] = bb_15.bollinger_lband()  

    don = ta.volatility.DonchianChannel(high=data["High"], low=data["Low"], close=data["Close"], window=20)
    data["DON_high_20"] = don.donchian_channel_hband()  
    data["DON_low_20"] = don.donchian_channel_lband()

    # Indicadores de volumen

    data["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"]).on_balance_volume()

    data["VPT"] = ta.volume.VolumePriceTrendIndicator(close=data["Close"], volume=data["Volume"]).volume_price_trend()

    data["ADI"] = ta.volume.AccDistIndexIndicator(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"]).acc_dist_index()

    data["MFI_14"] = ta.volume.MFIIndicator(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"], window=14).money_flow_index()

    data["CMF_21"] = ta.volume.ChaikinMoneyFlowIndicator(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"], window=21).chaikin_money_flow()

    return data
