# fetch_data.py
import yfinance as yf
import sys

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical stock data for the given ticker symbol.
    """
    try:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for {ticker_symbol} from {start_date} to {end_date}.")
            return None
        else:
            return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fetch_data.py <ticker_symbol> <start_date> <end_date>")
    else:
        ticker_symbol = sys.argv[1]
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if data is not None:
            filename = f'{ticker_symbol}_stock_data.csv'
            data.to_csv(filename)
            print(f"Data fetched and saved to {filename}.")
