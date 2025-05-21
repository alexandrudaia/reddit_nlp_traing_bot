from binance import Client
import pandas as pd
from datetime import datetime, timedelta
import time
from binance import *
client = Client('jQmroUzeIQFqjXraJE1MfnNAMvJZaXTmwd2dEyihu7Fh0nZr82Yxr54d3FESBCgX', 
               'ZGO5xOhXF4DnWI8LpoFPbIYlYA7pxwGDYs8rwQP1jZxqwM3ooGGDYlfb')

def get_btc_closing_price(timestamp):
    """
    Fetch BTC closing price from Binance for a specific timestamp.
    """
    try:
        # Convert timestamp to milliseconds
        ts = int(timestamp.timestamp() * 1000)
        
        # Get klines (candlestick) data
        # Fetch 1-hour candlestick for the specific timestamp
        klines = client.get_historical_klines(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1HOUR,
            str(ts),
            str(ts + 3600000)  # Add 1 hour in milliseconds
        )
        
        if klines:
            # Closing price is the 4th element in the kline data
            return float(klines[0][4])
        return None
        
    except Exception as e:
        print(f"Error fetching price for {timestamp}: {str(e)}")
        return None

def process_reddit_data(input_file, output_file):
    """
    Process the aggregated Reddit data and fetch Bitcoin prices for each hour.
    
    Args:
        input_file (str): Path to the input CSV file containing aggregated Reddit data.
        output_file (str): Path to the output CSV file with added Bitcoin prices.
    """
    try:
        # Read the aggregated Reddit data
        print("Reading input file...")
        df = pd.read_csv(input_file)
        
        # Convert 'date' and 'hour' columns to a single datetime column
        df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        
        # Create a new column for BTC prices
        df['btc_closing_price'] = None
        
        # Counter for rate limiting
        request_count = 0
        
        # Fetch BTC prices for each hour
        print("Fetching Bitcoin prices...")
        for index, row in df.iterrows():
            # Rate limiting: pause every 10 requests
            if request_count >= 10:
                print("Pausing for rate limit...")
                time.sleep(1)
                request_count = 0
            
            # Fetch the closing price for the given datetime
            closing_price = get_btc_closing_price(row['datetime'])
            if closing_price is not None:
                df.at[index, 'btc_closing_price'] = closing_price
                print(f"Successfully fetched price for {row['datetime']}: ${closing_price:.2f}")
            
            request_count += 1
            # Small delay between requests
            time.sleep(0.1)
        
        # Save the updated dataset
        print(f"Saving updated data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Successful price fetches: {df['btc_closing_price'].notna().sum()}")
        print(f"Missing prices: {df['btc_closing_price'].isna().sum()}")
        
        # Display sample of results
        print("\nSample of results:")
        print(df[['datetime', 'btc_closing_price']].head())
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_file = "aggregated_reddit_discussions.csv"
    output_file = "reddit_discussions_with_btc_prices.csv"
    
    process_reddit_data(input_file, output_file)
