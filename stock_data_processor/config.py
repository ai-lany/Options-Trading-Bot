"""
Configuration settings for the stock data reader application.
"""

# Default stock symbols to track
DEFAULT_SYMBOLS = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# Data fetching parameters
DEFAULT_PERIOD = '1d'  # 1 day of data
DEFAULT_INTERVAL = '1m'  # 1-minute intervals

# Valid periods for historical data
VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

# Valid intervals for data
VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

# Real-time streaming settings
STREAMING_UPDATE_INTERVAL = 60  # seconds

# Data columns to extract and include in numpy arrays
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']