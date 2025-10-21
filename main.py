import pandas as pd
from get_signals import signals
from indicators import get_indicators
from data_utils import split_data


def main():
    
    'Load Data'
    data=pd.read_csv('AAPL_data.csv')

    train_data, test_data, val_data = split_data(data)
    data = get_indicators(train_data)
    data = signals(data)

    print(data["signal"].value_counts())


    
    




    
    


if __name__ == "__main__":
    main()