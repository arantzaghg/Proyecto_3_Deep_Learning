import pandas as pd
from get_signals import signals
from indicators import get_indicators
from data_utils import split_data


def main():
    
    'Load Data'
    data=pd.read_csv('AAPL_data.csv')
    data = get_indicators(data)
    data = signals(data)

    train_data, test_data, val_data = split_data(data)

    


    
    


if __name__ == "__main__":
    main()