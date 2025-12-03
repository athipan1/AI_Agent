import pandas as pd
import os

def find_peers(ticker: str) -> list:
    """
    Finds peer companies for a given ticker based on the GICS Sub-Industry.

    Args:
        ticker (str): The stock ticker symbol of the company.

    Returns:
        list: A list of ticker symbols for peer companies.
              Returns an empty list if the ticker is not found or has no peers.
    """
    csv_path = 'sp500_companies.csv'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run sp500_scraper.py first.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    # Find the target company's GICS Sub-Industry
    target_company = df[df['Symbol'] == ticker]

    if target_company.empty:
        print(f"Ticker '{ticker}' not found in S&P 500 list.")
        return []

    target_industry = target_company.iloc[0]['GICS Sub-Industry']

    # Find all companies in the same GICS Sub-Industry
    peers = df[(df['GICS Sub-Industry'] == target_industry) & (df['Symbol'] != ticker)]

    peer_tickers = peers['Symbol'].tolist()

    return peer_tickers

if __name__ == '__main__':
    # Example usage:
    ticker_to_find = 'NVDA'
    peers = find_peers(ticker_to_find)
    if peers:
        print(f"Peers for {ticker_to_find} in the same industry ({len(peers)}):")
        print(peers)

    ticker_to_find = 'AAPL'
    peers = find_peers(ticker_to_find)
    if peers:
        print(f"\nPeers for {ticker_to_find} in the same industry ({len(peers)}):")
        print(peers)

    ticker_to_find = 'NONEXISTENT'
    peers = find_peers(ticker_to_find)
    if not peers:
        print(f"\nCould not find peers for {ticker_to_find}.")
