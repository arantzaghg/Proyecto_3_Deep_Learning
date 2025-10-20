import numpy as np
import pandas as pd


def sharpe_ratio(portfolio_hist) -> float:

    '''
    Calculate the Sharpe Ratio of a portfolio.
    
    Parameters:
    - portfolio_hist (pd.Series): Series representing the portfolio value over time.
    
    Returns:
    - float: Sharpe Ratio of the portfolio.
    '''

    returns = portfolio_hist.pct_change(fill_method = None).dropna()
    mean_return = returns.mean()
    std_return = returns.std()
        
    annual_return = mean_return * 365
    annual_std = std_return * np.sqrt(365)
    
    return annual_return / annual_std if annual_std > 0 else 0


def sortino_ratio(portfolio_hist) -> float:

    '''
    Calculate the Sortino Ratio of a portfolio.
    
    Parameters:
    - portfolio_hist (pd.Series): Series representing the portfolio value over time.
    
    Returns:
    - float: Sortino Ratio of the portfolio.
    '''

    returns = portfolio_hist.pct_change(fill_method = None).dropna()
    mean_return = returns.mean()
    downside_dev = returns[returns < 0].std()

    annual_return = mean_return * 365
    annual_downside_dev = downside_dev * np.sqrt(365)

    return annual_return / annual_downside_dev if annual_downside_dev > 0 else 0
    

def maximum_drawdown(portfolio_hist) -> float:

    '''
    Calculate the Maximum Drawdown of a portfolio.
    
    Parameters:
    - portfolio_hist (pd.Series): Series representing the portfolio value over time.
    
    Returns:
    - float: Maximum Drawdown of the portfolio.
    '''

    rolling_max = portfolio_hist.cummax()
    drawdown = (rolling_max - portfolio_hist) / rolling_max
    max_drawdown = abs(drawdown.max())

    return max_drawdown
    

def calmar_ratio(portfolio_hist) -> float:

    '''
    Calculate the Calmar Ratio of a portfolio.

    Parameters:
    - portfolio_hist (pd.Series): Series representing the portfolio value over time.

    Returns:
    - float: Calmar Ratio of the portfolio.
    '''

    returns = portfolio_hist.pct_change(fill_method = None).dropna()
    mean_return = returns.mean()
    annual_return = mean_return * 365  

    max_drawdown = maximum_drawdown(portfolio_hist)
    
    return annual_return / max_drawdown if max_drawdown > 0 else 0


def all_metrics(portfolio_value) -> pd.DataFrame:

    '''
    Calculate all performance metrics for a portfolio.
    
    Parameters:
    - portfolio_value (pd.Series): Series representing the portfolio value over time.
    
    Returns:
    - pd.DataFrame: DataFrame containing all performance metrics.
    '''

    metrics = pd.DataFrame({
        'Sharpe Ratio': sharpe_ratio(portfolio_value),
        'Sortino Ratio': sortino_ratio(portfolio_value),
        'Maximum Drawdown': maximum_drawdown(portfolio_value),
        'Calmar Ratio': calmar_ratio(portfolio_value)
    }, index=['Metrics'])
    
    return metrics