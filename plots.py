import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_value) :

    '''
    Plot the portfolio value over the training period.
    
    Parameters:
    - portfolio_value_train (pd.Series): Series of portfolio values during the training period.
    '''
    
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value,  color='cornflowerblue', linewidth=1.8, label='Portfolio')
    plt.title(f'{title} Portfolio Value over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()