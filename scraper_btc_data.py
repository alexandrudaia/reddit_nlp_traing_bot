from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import time
import signal
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BitcoinDataCollector:
    def __init__(self, api_key, api_secret):
        self.binance_client = Client(api_key, api_secret)
        self.historical_file = 'bitcoin_historical_data.csv'

    def collect_historical_data(self, limit=10000):
        """Collect the last 10,000 hours of historical Bitcoin data."""
        try:
            logger.info("Fetching historical data...")
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)

            klines = self.binance_client.get_historical_klines(
                "BTCUSDT", Client.KLINE_INTERVAL_1HOUR, start_time.strftime('%d %b, %Y %H:%M:%S'), end_time.strftime('%d %b, %Y %H:%M:%S')
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_sell_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # Add additional calculated columns
            df['price_change'] = df['close'].astype(float) - df['open'].astype(float)
            df['price_change_percent'] = (df['price_change'] / df['open'].astype(float)) * 100
            df['average_price'] = (df['high'].astype(float) + df['low'].astype(float)) / 2

            return df
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return None

    def save_historical_data(self, df: pd.DataFrame):
        """Save historical data to a CSV file."""
        try:
            df.to_csv(self.historical_file, index=False)
            logger.info(f"Historical data saved to {self.historical_file}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")


class BitcoinRealTimeCollector(BitcoinDataCollector):
    def __init__(self, api_key, api_secret):
        super().__init__(api_key, api_secret)
        self.running = False
        self.realtime_file = 'bitcoin_realtime_data.csv'

    def get_realtime_data(self) -> pd.DataFrame:
        """Get current Bitcoin metrics."""
        try:
            logger.info("Fetching real-time data...")
            kline = self.binance_client.get_klines(
                symbol="BTCUSDT",
                interval=Client.KLINE_INTERVAL_1HOUR,
                limit=1
            )
            if not kline:
                return None

            # Create DataFrame with current data
            df = pd.DataFrame([kline[0]], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_sell_volume', 'ignore'
            ])
            df['timestamp'] = datetime.now()  # Use current time for real-time data

            # Add additional calculated columns
            df['price_change'] = df['close'].astype(float) - df['open'].astype(float)
            df['price_change_percent'] = (df['price_change'] / df['open'].astype(float)) * 100
            df['average_price'] = (df['high'].astype(float) + df['low'].astype(float)) / 2

            return df
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return None

    def save_realtime_data(self, df: pd.DataFrame):
        """Save real-time data to a CSV file."""
        try:
            if not os.path.exists(self.realtime_file):
                df.to_csv(self.realtime_file, index=False)
            else:
                df.to_csv(self.realtime_file, mode='a', header=False, index=False)
            logger.info(f"Real-time data saved for {df['timestamp'].iloc[0]}")
        except Exception as e:
            logger.error(f"Error saving real-time data: {e}")

    def start_realtime_collection(self):
        """Start real-time data collection."""
        self.running = True
        logger.info("Starting real-time data collection...")
        while self.running:
            df = self.get_realtime_data()
            if df is not None:
                self.save_realtime_data(df)
                print("\nCurrent Bitcoin Metrics:")
                print(f"Timestamp: {df['timestamp'].iloc[0]}")
                print(f"Price: ${float(df['close'].iloc[0]):.2f}")
                print(f"24h Volume: ${float(df['volume'].iloc[0]):.2f}")
                print(f"Price Change: ${float(df['price_change'].iloc[0]):.2f}")
                print(f"Price Change (%): {float(df['price_change_percent'].iloc[0]):.2f}%")
            time.sleep(3600)  # Collect data every hour

    def stop_realtime_collection(self):
        """Stop real-time data collection."""
        self.running = False
        logger.info("Stopping real-time data collection...")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal. Stopping data collection...")
    if collector is not None:
        collector.stop_realtime_collection()
    sys.exit(0)


def main():
    # Your Binance API credentials
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'

    # Initialize collector
    global collector
    collector = BitcoinRealTimeCollector(api_key, api_secret)

    try:
        # Collect historical data
        logger.info("Starting historical data collection...")
        historical_df = collector.collect_historical_data()
        if historical_df is not None:
            collector.save_historical_data(historical_df)
            logger.info("Historical data collection completed.")
            print("\nHistorical Data Summary:")
            print(f"Time Range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")
            print(f"Total Records: {len(historical_df)}")

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)

        # Start real-time collection
        logger.info("Starting real-time data collection...")
        collector.start_realtime_collection()

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        if collector is not None:
            collector.stop_realtime_collection()


if __name__ == "__main__":
    collector = None
    main()
