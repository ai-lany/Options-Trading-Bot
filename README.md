# Options-Trading-Bot

A Python application for reading real-time stock data and converting it into NumPy arrays for analysis.

## Features

- **Real-time Data Fetching**: Retrieves current and historical stock data from Yahoo Finance
- **NumPy Integration**: Automatically converts stock data into NumPy arrays for numerical analysis
- **Multiple Symbols**: Support for tracking multiple stock symbols simultaneously
- **Data Processing**: Built-in data cleaning, normalization, and statistical analysis
- **Streaming Support**: Real-time data streaming with configurable update intervals
- **Error Handling**: Robust error handling for network issues and invalid data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ai-lany/Options-Trading-Bot.git
cd Options-Trading-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from stock_data_reader import StockDataReader

# Initialize reader with stock symbols
reader = StockDataReader(['AAPL', 'MSFT', 'GOOGL'])

# Fetch historical data
data = reader.fetch_historical_data('AAPL', period='5d', interval='1h')

# Convert to NumPy array
numpy_data = reader.data_to_numpy(data)
print(f"Data shape: {numpy_data.shape}")

# Get current price
current_price = reader.get_current_price('AAPL')
print(f"Current AAPL price: ${current_price}")
```

### Real-time Data Streaming

```python
# Start streaming real-time data
reader.start_streaming(update_interval=60)  # Update every 60 seconds

# Get current streaming data
streaming_data = reader.get_streaming_data()
for symbol, data in streaming_data.items():
    print(f"{symbol}: ${data['current_price']}")

# Stop streaming
reader.stop_streaming()
```

## Examples

- **`example.py`**: Basic usage example
- **`demo.py`**: Comprehensive demonstration with sample data
- **`stock_data_reader.py`**: Run the main module for a quick demo

Run the demo:
```bash
python demo.py
```

## Data Format

The application returns stock data as NumPy arrays with the following columns:
- Column 0: Open price
- Column 1: High price
- Column 2: Low price
- Column 3: Close price
- Column 4: Volume

## Configuration

Modify `config.py` to customize:
- Default stock symbols
- Data periods and intervals
- Streaming update frequency
- Data columns to include

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **yfinance**: Yahoo Finance data fetching
- **requests**: HTTP requests

## Error Handling

The application gracefully handles:
- Network connectivity issues
- Invalid stock symbols
- Missing or incomplete data
- API rate limiting

## License

MIT License - see LICENSE file for details.