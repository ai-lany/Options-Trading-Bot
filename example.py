#!/usr/bin/env python3
"""
Simple example showing basic usage of the Stock Data Reader.
"""

from stock_data_reader import StockDataReader
import numpy as np

def main():
    # Initialize the stock data reader
    reader = StockDataReader(['AAPL', 'MSFT', 'GOOGL'])
    
    print("Fetching stock data...")
    
    # Fetch historical data for Apple
    aapl_data = reader.fetch_historical_data('AAPL', period='5d', interval='1h')
    
    if not aapl_data.empty:
        # Convert to numpy array
        numpy_data = reader.data_to_numpy(aapl_data)
        print(f"AAPL data shape: {numpy_data.shape}")
        print(f"Latest close price: ${numpy_data[-1, 3]:.2f}")
        
        # Get summary statistics
        close_prices = numpy_data[:, 3]
        summary = reader.data_processor.get_data_summary(close_prices)
        print(f"Average price: ${summary['mean']:.2f}")
        print(f"Price volatility (std): ${summary['std']:.2f}")
    else:
        print("No data retrieved (likely due to network restrictions)")
    
    # Fetch data for multiple symbols
    multi_data = reader.fetch_multiple_symbols(['AAPL', 'MSFT', 'GOOGL'])
    print(f"Retrieved data for {len(multi_data)} symbols")
    
    # Start real-time streaming (demonstration)
    print("Starting real-time data streaming...")
    reader.start_streaming(update_interval=30)  # Update every 30 seconds
    
    # In a real application, you would keep the program running
    # For this example, we'll stop streaming immediately
    reader.stop_streaming()
    
    print("Example completed!")

if __name__ == "__main__":
    main()