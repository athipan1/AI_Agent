import yfinance as yf


def get_stocks_data(tickers: list) -> dict:
    """
    Fetches key financial and historical data for a list of stock tickers.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).

    Returns:
        A dictionary where keys are the ticker symbols and values are another
        dictionary containing the data for that ticker. Skips tickers with
        download errors.
    """
    if not tickers:
        return {}

    # yf.Tickers() is more efficient for multiple symbols
    stocks = yf.Tickers(" ".join(tickers))
    all_data = {}

    for ticker in tickers:
        stock_obj = stocks.tickers.get(ticker.upper())

        # Check if the ticker object or its info is invalid/empty
        info = getattr(stock_obj, 'info', None)
        if not info or info.get('trailingPE') is None:
            print(f"Warning: Could not retrieve valid data for {ticker}. Skipping.")
            continue

        # Define all required metrics and their corresponding keys in yfinance info
        # Use 'ข้อมูลไม่ครบ' as the default value.
        metric_keys = {
            'Sector': 'sector', 'Industry': 'industry', 'Market Cap': 'marketCap',
            'P/E': 'trailingPE', 'Forward P/E': 'forwardPE', 'PEG Ratio': 'pegRatio',
            'EPS': 'trailingEps', 'EPS Growth': 'earningsGrowth', 'Debt/Equity': 'debtToEquity',
            'ROE': 'returnOnEquity', 'Operating Margin': 'operatingMargins',
            'Gross Margin': 'grossMargins', 'Revenue Growth': 'revenueGrowth',
            'Free Cash Flow': 'freeCashflow',
        }
        data = {key: info.get(value, 'ข้อมูลไม่ครบ') for key, value in metric_keys.items()}

        # --- Historical Price Data ---
        try:
            hist_1y = stock_obj.history(period="1y")
            if not hist_1y.empty:
                data['Historical Price (1Y)'] = {
                    k.strftime('%Y-%m-%d'): v for k, v in hist_1y['Close'].to_dict().items()
                }
            else:
                data['Historical Price (1Y)'] = 'ข้อมูลไม่ครบ'

            hist_5y = stock_obj.history(period="5y")
            if not hist_5y.empty:
                data['Historical Price (5Y)'] = {
                    k.strftime('%Y-%m-%d'): v for k, v in hist_5y['Close'].to_dict().items()
                }
            else:
                data['Historical Price (5Y)'] = 'ข้อมูลไม่ครบ'
        except Exception as e:
            print(f"Could not fetch historical data for {ticker}: {e}")
            data['Historical Price (1Y)'] = 'ข้อมูลไม่ครบ'
            data['Historical Price (5Y)'] = 'ข้อมูลไม่ครบ'

        all_data[ticker] = data

    return all_data


if __name__ == '__main__':
    # --- Example Usage ---
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'NONEXISTENTTICKER']
    stocks_data = get_stocks_data(test_tickers)

    print(f"\nSuccessfully fetched data for {len(stocks_data)}/{len(test_tickers)} tickers.\n")

    for ticker, data in stocks_data.items():
        print(f"--- Financial Data for {ticker} ---")
        for key, value in data.items():
            if isinstance(value, dict):
                # Don't print the entire historical data dictionary
                print(f"- {key}: [Data Series with {len(value)} points]")
            else:
                print(f"- {key}: {value}")
        print("-" * 35)
