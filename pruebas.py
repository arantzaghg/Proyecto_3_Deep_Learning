from data_utils import get_asset_data, split_data, preprocess_data, get_target


def pruebas ():
    ticker = "AMZN"
    data = get_asset_data(ticker)
    train_data, test_data, val_data = split_data(data)

    train_data = preprocess_data(train_data, ticker, alpha=0.016)
    

    values = train_data['signal'].value_counts()
    print(values)

if __name__ == "__main__":
    pruebas()
    