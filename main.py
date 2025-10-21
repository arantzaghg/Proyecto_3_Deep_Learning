import pandas as pd
from get_signals import signals
from indicators import get_indicators
from data_utils import split_data
from normalization import get_normal_stats  


def main():
    
    'Load Data'
    data=pd.read_csv('AAPL_data.csv')

    # Split Data
    train_data, test_data, val_data = split_data(data)

    # Generate Indicators and Signals
    train_data = get_indicators(train_data)
    train_data = signals(train_data)
    train_data, stats = get_normal_stats(train_data)
    train_data.dropna(inplace=True)

    # Get target
    x_train = train_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'signal'])
    y_train = train_data['signal']

if __name__ == "__main__":
    main()