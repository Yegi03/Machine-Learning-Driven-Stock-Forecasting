# explore_data.py
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Loads stock data from a CSV file, expecting 'Date' as an index column.
    """
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

def plot_stock_data(stock_data):
    """
    Plots the closing price and volume of the stock.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close', color=color)
    ax1.plot(stock_data.index, stock_data['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Volume', color=color)  # we already handled the x-label with ax1
    ax2.fill_between(stock_data.index, 0, stock_data['Volume'], color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Stock Closing Price and Volume')
    plt.show()

def plot_moving_averages(stock_data):
    """
    Adds 50-day and 200-day moving averages to the closing price plot.
    """
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Close Price', alpha=0.5)
    plt.plot(stock_data['MA50'], label='50-Day MA')
    plt.plot(stock_data['MA200'], label='200-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price with Moving Averages')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = 'AAPL_stock_data.csv'  # Ensure this path matches where your CSV file is stored
    stock_data = load_data(file_path)
    plot_stock_data(stock_data)
    plot_moving_averages(stock_data)
