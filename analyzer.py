import pandas as pd
import numpy as np
import json
from data_fetcher import get_stocks_data # For example usage

def calculate_peer_stats(peer_data: dict) -> dict:
    """
    Calculates mean and median statistics for a group of peer companies.

    Args:
        peer_data (dict): A dictionary containing financial data for peer companies,
                          with tickers as keys.

    Returns:
        A dictionary containing the mean and median for each financial metric.
    """
    if not peer_data:
        return {}

    # Convert the peer data into a pandas DataFrame
    df = pd.DataFrame.from_dict(peer_data, orient='index')

    # List of metrics to calculate stats for
    numeric_metrics = [
        'P/E', 'Forward P/E', 'PEG Ratio', 'EPS', 'EPS Growth',
        'Debt/Equity', 'ROE', 'Operating Margin', 'Gross Margin',
        'Revenue Growth', 'Free Cash Flow'
    ]

    stats = {}
    for metric in numeric_metrics:
        # Convert column to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(df.get(metric), errors='coerce')
        if not numeric_series.empty and numeric_series.notna().any():
            stats[metric] = {
                'mean': numeric_series.mean(),
                'median': numeric_series.median()
            }
        else:
            stats[metric] = {'mean': np.nan, 'median': np.nan}

    return stats

def create_peer_comparison_table(target_data: dict, peer_stats: dict) -> str:
    """
    Creates a formatted Markdown table comparing the target company to its peers.
    """
    table = "| Metric             | Target Company | Peer Group (Mean) | Comparison vs Mean |\n"
    table += "|--------------------|----------------|-------------------|--------------------|\n"

    metrics_to_compare = {
        'P/E': '{:.2f}',
        'Forward P/E': '{:.2f}',
        'PEG Ratio': '{:.2f}',
        'ROE': '{:.2%}',
        'Operating Margin': '{:.2%}',
        'Gross Margin': '{:.2%}',
        'Revenue Growth': '{:.2%}',
        'EPS Growth': '{:.2%}',
        'Debt/Equity': '{:.2f}',
    }

    for metric, fmt in metrics_to_compare.items():
        target_val = target_data.get(metric)
        peer_mean = peer_stats.get(metric, {}).get('mean')

        target_str = fmt.format(target_val) if isinstance(target_val, (int, float)) else "N/A"
        peer_str = fmt.format(peer_mean) if isinstance(peer_mean, (int, float)) and not np.isnan(peer_mean) else "N/A"

        comparison_str = "N/A"
        if isinstance(target_val, (int, float)) and isinstance(peer_mean, (int, float)) and not np.isnan(peer_mean) and peer_mean != 0:
            diff = (target_val - peer_mean) / abs(peer_mean)
            prefix = "+" if diff >= 0 else ""
            comparison_str = f"{prefix}{diff:.1%}"

        table += f"| {metric:<18} | {target_str:<14} | {peer_str:<17} | {comparison_str:<18} |\n"

    return table

def calculate_summary_score(target_data: dict, peer_stats: dict) -> int:
    """
    Calculates a summary score (0-100) based on the company's performance
    relative to its peers.

    Weights:
    - Valuation: 30%
    - Fundamental Strength: 70%
    """
    score = 50  # Start from a baseline of 50

    # --- Valuation Score (30 points) ---
    # Lower is better for P/E, Fwd P/E, PEG
    valuation_metrics = ['P/E', 'Forward P/E', 'PEG Ratio']
    for metric in valuation_metrics:
        target = pd.to_numeric(target_data.get(metric), errors='coerce')
        peer_mean = peer_stats.get(metric, {}).get('mean')
        if pd.notna(target) and pd.notna(peer_mean) and peer_mean > 0:
            if target < peer_mean * 0.8: score += 5 # Significantly cheaper
            elif target < peer_mean: score += 2 # Cheaper
            elif target > peer_mean * 1.5: score -= 5 # Significantly more expensive
            elif target > peer_mean: score -= 2 # More expensive

    # --- Fundamental Strength Score (70 points) ---
    # Higher is better for ROE, Margins, Growth
    positive_metrics = ['ROE', 'Operating Margin', 'Gross Margin', 'Revenue Growth', 'EPS Growth']
    for metric in positive_metrics:
        target = pd.to_numeric(target_data.get(metric), errors='coerce')
        peer_mean = peer_stats.get(metric, {}).get('mean')
        if pd.notna(target) and pd.notna(peer_mean):
            if target > peer_mean * 1.5: score += 7 # Significantly stronger
            elif target > peer_mean: score += 3 # Stronger
            elif target < peer_mean * 0.8: score -= 7 # Significantly weaker
            elif target < peer_mean: score -= 3 # Weaker

    # Lower is better for Debt/Equity
    target_de = pd.to_numeric(target_data.get('Debt/Equity'), errors='coerce')
    peer_de = peer_stats.get('Debt/Equity', {}).get('mean')
    if pd.notna(target_de) and pd.notna(peer_de):
        if target_de < peer_de * 0.5: score += 7 # Significantly lower debt
        elif target_de < peer_de: score += 3 # Lower debt
        elif target_de > peer_de * 1.5: score -= 7 # Significantly higher debt
        elif target_de > peer_de: score -= 3 # Higher debt

    return max(0, min(100, int(score))) # Clamp score between 0 and 100

def analyze_with_peers(target_ticker: str, all_data: dict) -> dict | None:
    """
    Performs a full peer comparison analysis.

    Args:
        target_ticker (str): The main ticker to analyze.
        all_data (dict): The dictionary of data from get_stocks_data, including
                         the target and its peers.

    Returns:
        A dictionary containing the full analysis, or None if data is insufficient.
    """
    target_data = all_data.get(target_ticker)
    if not target_data:
        print(f"No data found for target ticker {target_ticker} in the provided data.")
        return None

    peer_tickers = [t for t in all_data if t != target_ticker]
    peer_data = {t: d for t, d in all_data.items() if t in peer_tickers}

    if not peer_data:
        print(f"No peer data available for analysis of {target_ticker}.")
        return {"error": "No peer data available."}

    # 1. Calculate peer statistics
    peer_stats = calculate_peer_stats(peer_data)

    # 2. Create comparison table
    comparison_table = create_peer_comparison_table(target_data, peer_stats)

    # 3. Calculate summary score
    summary_score = calculate_summary_score(target_data, peer_stats)

    analysis_result = {
        "peer_stats": {k: v for k, v in peer_stats.items() if not all(np.isnan(val) for val in v.values())}, # Clean up NaN stats
        "comparison_table": comparison_table,
        "summary_score": summary_score
    }

    return analysis_result


if __name__ == '__main__':
    from peer_finder import find_peers

    # --- Example Usage ---
    target = 'AAPL'
    print(f"--- Starting Peer Analysis for {target} ---")

    # 1. Find peers
    peers = find_peers(target)

    if peers:
        print(f"Found {len(peers)} peers: {peers[:5]}...") # Print first 5

        # 2. Fetch data for target and peers
        all_tickers = [target] + peers
        all_stocks_data = get_stocks_data(all_tickers)

        if all_stocks_data:
            # 3. Analyze
            analysis = analyze_with_peers(target, all_stocks_data)

            if analysis:
                print("\n--- Analysis Result ---")
                print("\nPeer Comparison Table:")
                print(analysis["comparison_table"])
                print(f"Summary Score: {analysis['summary_score']}/100")

                # print("\nPeer Stats (Mean/Median):")
                # print(json.dumps(analysis['peer_stats'], indent=2))

    print("\n--- Analysis Complete ---")
