"""
Stock Data Reader - Real-time stock data fetching and numpy conversion.

This module provides functionality to fetch real-time and historical stock data
and convert it into numpy arrays for further analysis.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta
import threading

from config import (
    DEFAULT_SYMBOLS, DEFAULT_PERIOD, DEFAULT_INTERVAL,
    VALID_PERIODS, VALID_INTERVALS, STREAMING_UPDATE_INTERVAL,
    PRICE_COLUMNS
)
from data_utils import StockDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataReader:
    """
    Main class for fetching and processing real-time stock data.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the StockDataReader.
        
        Args:
            symbols: List of stock symbols to track. Defaults to DEFAULT_SYMBOLS.
        """
        self.symbols = symbols or DEFAULT_SYMBOLS.copy()
        self.data_processor = StockDataProcessor()
        self.current_data = {}
        self.is_streaming = False
        self.stream_thread = None
        
        logger.info(f"StockDataReader initialized with symbols: {self.symbols}")
    
    def validate_inputs(self, period: str, interval: str) -> Tuple[bool, str]:
        """
        Validate period and interval inputs.
        
        Args:
            period: Time period for data
            interval: Data interval
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if period not in VALID_PERIODS:
            return False, f"Invalid period '{period}'. Valid periods: {VALID_PERIODS}"
        
        if interval not in VALID_INTERVALS:
            return False, f"Invalid interval '{interval}'. Valid intervals: {VALID_INTERVALS}"
        
        return True, ""
    
    def fetch_historical_data(self, symbol: str, period: str = DEFAULT_PERIOD, 
                            interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
        """
        Fetch historical stock data for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period for data
            interval: Data interval
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        is_valid, error_msg = self.validate_inputs(period, interval)
        if not is_valid:
            logger.error(error_msg)
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching data for {symbol} (period={period}, interval={interval})")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Clean the data
            data = self.data_processor.clean_data(data)
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str] = None, 
                             period: str = DEFAULT_PERIOD,
                             interval: str = DEFAULT_INTERVAL) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols. If None, uses self.symbols.
            period: Time period for data
            interval: Data interval
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        if symbols is None:
            symbols = self.symbols
        
        results = {}
        
        for symbol in symbols:
            data = self.fetch_historical_data(symbol, period, interval)
            if not data.empty:
                results[symbol] = data
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price:
                logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                return float(current_price)
            else:
                logger.warning(f"Could not get current price for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def data_to_numpy(self, data: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """
        Convert DataFrame to numpy array.
        
        Args:
            data: Input DataFrame
            columns: Specific columns to include
            
        Returns:
            numpy.ndarray: Converted data
        """
        if columns is None:
            columns = PRICE_COLUMNS
        
        return self.data_processor.dataframe_to_numpy(data, columns)
    
    def get_real_time_data(self, symbol: str) -> Dict[str, Union[float, np.ndarray]]:
        """
        Get real-time data for a symbol including current price and recent history.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing current price and recent data as numpy array
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': None,
            'recent_data': np.array([]),
            'data_summary': {}
        }
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price:
            result['current_price'] = current_price
        
        # Get recent historical data (last hour with 1-minute intervals)
        recent_data = self.fetch_historical_data(symbol, period='1d', interval='1m')
        if not recent_data.empty:
            # Convert to numpy array
            numpy_data = self.data_to_numpy(recent_data)
            result['recent_data'] = numpy_data
            
            # Add summary statistics
            if len(numpy_data) > 0:
                close_prices = numpy_data[:, -2] if numpy_data.shape[1] >= 5 else numpy_data[:, -1]
                result['data_summary'] = self.data_processor.get_data_summary(close_prices)
        
        return result
    
    def start_streaming(self, update_interval: int = STREAMING_UPDATE_INTERVAL):
        """
        Start streaming real-time data for all symbols.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.is_streaming:
            logger.warning("Streaming is already active")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(
            target=self._streaming_worker,
            args=(update_interval,),
            daemon=True
        )
        self.stream_thread.start()
        logger.info(f"Started streaming data for {len(self.symbols)} symbols")
    
    def stop_streaming(self):
        """Stop the streaming data updates."""
        if not self.is_streaming:
            logger.warning("Streaming is not active")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        
        logger.info("Stopped streaming data")
    
    def _streaming_worker(self, update_interval: int):
        """
        Background worker for streaming data updates.
        
        Args:
            update_interval: Update interval in seconds
        """
        while self.is_streaming:
            try:
                for symbol in self.symbols:
                    if not self.is_streaming:
                        break
                    
                    data = self.get_real_time_data(symbol)
                    self.current_data[symbol] = data
                    
                    logger.info(f"Updated data for {symbol}: ${data.get('current_price', 'N/A')}")
                
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in streaming worker: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def get_streaming_data(self) -> Dict[str, Dict]:
        """
        Get the current streaming data for all symbols.
        
        Returns:
            Dictionary containing real-time data for all symbols
        """
        return self.current_data.copy()
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.current_data:
                del self.current_data[symbol]
            logger.info(f"Removed symbol: {symbol}")


def main():
    """
    Demo function showing basic usage of the StockDataReader.
    """
    print("=== Stock Data Reader Demo ===")
    
    # Initialize reader with default symbols
    reader = StockDataReader()
    
    # Fetch historical data for a single symbol
    print("\n1. Fetching historical data for AAPL...")
    aapl_data = reader.fetch_historical_data('AAPL', period='5d', interval='1h')
    print(f"Retrieved {len(aapl_data)} data points")
    
    if not aapl_data.empty:
        # Convert to numpy array
        numpy_data = reader.data_to_numpy(aapl_data)
        print(f"Numpy array shape: {numpy_data.shape}")
        print(f"Columns: {PRICE_COLUMNS}")
        print(f"Sample data (first 3 rows):\n{numpy_data[:3]}")
        
        # Calculate some statistics
        close_prices = numpy_data[:, 3]  # Close prices (4th column)
        summary = reader.data_processor.get_data_summary(close_prices)
        print(f"Close price summary: {summary}")
    
    # Get real-time data
    print("\n2. Getting real-time data for AAPL...")
    real_time_data = reader.get_real_time_data('AAPL')
    print(f"Current price: ${real_time_data.get('current_price', 'N/A')}")
    print(f"Recent data shape: {real_time_data['recent_data'].shape}")
    
    # Fetch data for multiple symbols
    print("\n3. Fetching data for multiple symbols...")
    multi_data = reader.fetch_multiple_symbols(['AAPL', 'GOOGL', 'MSFT'], period='1d', interval='5m')
    
    for symbol, data in multi_data.items():
        numpy_array = reader.data_to_numpy(data)
        print(f"{symbol}: {numpy_array.shape} - Latest close: ${numpy_array[-1, 3]:.2f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()