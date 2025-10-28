import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_value, title: str):

    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value,  color='skyblue', linewidth=1.8, label='Portfolio')
    plt.title(f'{title} Portfolio Value over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


def plot_test_validation(test_portfolio, validation_portfolio, test, validation):
    
    
    test.index = pd.to_datetime(test.index)
    validation.index = pd.to_datetime(validation.index)

    min_test_len = min(len(test.index), len(test_portfolio))
    min_val_len = min(len(validation.index), len(validation_portfolio))

    test_series = pd.Series(test_portfolio.iloc[:min_test_len].values, index=test.index[:min_test_len])
    val_series = pd.Series(validation_portfolio.iloc[:min_val_len].values, index=validation.index[:min_val_len])

    combined = pd.concat([test_series, val_series])

    plt.figure(figsize=(12, 6))
    plt.plot(test_series.index, test_series.values, color='cornflowerblue', label='Test')
    plt.plot([test_series.index[-1], val_series.index[0]],
             [test_series.values[-1], val_series.values[0]],
             color='navy', linestyle='-')

    plt.plot(val_series.index, val_series.values, color='navy', label='Validation')

    plt.title('Portfolio Value over test and validation periods')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_individual_bars(buy: int, sell: int, hold: int):
    
    categories = ['Buy', 'Sell', 'Hold']
    values = np.array([buy, sell, hold], dtype=float)
    total = values.sum()
    percentage = 100 * values / total
    colors = ['lightblue', 'cornflowerblue', 'navy']
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(categories, percentage, color=colors)

    for bar, pct in zip(bars, percentage):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title('Distribution of Buy / Sell / Hold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


def plot_returns(portfolio: pd.Series):

    daily_ret = portfolio.pct_change() * 100

    days = pd.RangeIndex(len(portfolio))
    monthly_ret = portfolio.groupby(np.floor_divide(days, 21)).last().pct_change() * 100
    annual_ret = portfolio.groupby(np.floor_divide(days, 252)).last().pct_change() * 100

    def colors(r):
        return ['cornflowerblue' if x > 0 else 'rosybrown' for x in r]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].bar(days, daily_ret, color=colors(daily_ret))
    axes[0].set_title('Daily Return (%)')
    axes[0].set_ylabel('% daily')

    axes[1].bar(monthly_ret.index, monthly_ret.values, color=colors(monthly_ret))
    axes[1].set_title('Monthly Return (%)')
    axes[1].set_ylabel('% monthly')

    axes[2].bar(annual_ret.index, annual_ret.values, color=colors(annual_ret))
    axes[2].set_title('Annual Return (%)')
    axes[2].set_ylabel('% annual')

    plt.tight_layout()
    plt.show()
