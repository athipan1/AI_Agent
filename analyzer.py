import pandas as pd
import numpy as np
from data_fetcher import get_stocks_data  # For example usage


def calculate_peer_stats(peer_data: dict) -> dict:
    """
    Calculates mean and median statistics for a group of peer companies.
    """
    if not peer_data:
        return {}

    df = pd.DataFrame.from_dict(peer_data, orient='index')
    numeric_metrics = [
        'P/E', 'Forward P/E', 'PEG Ratio', 'EPS', 'EPS Growth',
        'Debt/Equity', 'ROE', 'Operating Margin', 'Gross Margin',
        'Revenue Growth', 'Free Cash Flow'
    ]

    stats = {}
    for metric in numeric_metrics:
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
        'P/E': '{:.2f}', 'Forward P/E': '{:.2f}', 'PEG Ratio': '{:.2f}',
        'ROE': '{:.2%}', 'Operating Margin': '{:.2%}', 'Gross Margin': '{:.2%}',
        'Revenue Growth': '{:.2%}', 'EPS Growth': '{:.2%}', 'Debt/Equity': '{:.2f}',
    }

    for metric, fmt in metrics_to_compare.items():
        target_val = target_data.get(metric)
        peer_mean = peer_stats.get(metric, {}).get('mean')

        target_str = fmt.format(target_val) if isinstance(target_val, (int, float)) else "N/A"
        peer_str = "N/A"
        if isinstance(peer_mean, (int, float)) and not np.isnan(peer_mean):
            peer_str = fmt.format(peer_mean)

        comparison_str = "N/A"
        if isinstance(target_val, (int, float)) and peer_str != "N/A" and peer_mean != 0:
            diff = (target_val - peer_mean) / abs(peer_mean)
            prefix = "+" if diff >= 0 else ""
            comparison_str = f"{prefix}{diff:.1%}"

        table += f"| {metric:<18} | {target_str:<14} | {peer_str:<17} | {comparison_str:<18} |\n"
    return table


def _calculate_valuation_score(score, target_data, peer_stats):
    """Calculates the valuation component of the summary score."""
    valuation_metrics = ['P/E', 'Forward P/E', 'PEG Ratio']
    for metric in valuation_metrics:
        target = pd.to_numeric(target_data.get(metric), errors='coerce')
        peer_mean = peer_stats.get(metric, {}).get('mean')
        if pd.notna(target) and pd.notna(peer_mean) and peer_mean > 0:
            if target < peer_mean * 0.8:
                score += 5  # Significantly cheaper
            elif target < peer_mean:
                score += 2  # Cheaper
            elif target > peer_mean * 1.5:
                score -= 5  # Significantly more expensive
            elif target > peer_mean:
                score -= 2  # More expensive
    return score


def _calculate_strength_score(score, target_data, peer_stats):
    """Calculates the fundamental strength component of the summary score."""
    positive_metrics = [
        'ROE', 'Operating Margin', 'Gross Margin', 'Revenue Growth', 'EPS Growth'
    ]
    for metric in positive_metrics:
        target = pd.to_numeric(target_data.get(metric), errors='coerce')
        peer_mean = peer_stats.get(metric, {}).get('mean')
        if pd.notna(target) and pd.notna(peer_mean):
            if target > peer_mean * 1.5:
                score += 7  # Significantly stronger
            elif target > peer_mean:
                score += 3  # Stronger
            elif target < peer_mean * 0.8:
                score -= 7  # Significantly weaker
            elif target < peer_mean:
                score -= 3  # Weaker

    target_de = pd.to_numeric(target_data.get('Debt/Equity'), errors='coerce')
    peer_de = peer_stats.get('Debt/Equity', {}).get('mean')
    if pd.notna(target_de) and pd.notna(peer_de):
        if target_de < peer_de * 0.5:
            score += 7  # Significantly lower debt
        elif target_de < peer_de:
            score += 3  # Lower debt
        elif target_de > peer_de * 1.5:
            score -= 7  # Significantly higher debt
        elif target_de > peer_de:
            score -= 3  # Higher debt
    return score


def calculate_summary_score(target_data: dict, peer_stats: dict) -> int:
    """
    Calculates a summary score (0-100) based on performance vs peers.
    """
    score = 50  # Start from a baseline of 50
    score = _calculate_valuation_score(score, target_data, peer_stats)
    score = _calculate_strength_score(score, target_data, peer_stats)
    return max(0, min(100, int(score)))  # Clamp score between 0 and 100


def analyze_with_peers(target_ticker: str, all_data: dict) -> dict | None:
    """
    Performs a full peer comparison analysis.
    """
    target_data = all_data.get(target_ticker)
    if not target_data:
        print(f"No data for target {target_ticker} in provided data.")
        return None

    peer_tickers = [t for t in all_data if t != target_ticker]
    peer_data = {t: d for t, d in all_data.items() if t in peer_tickers}

    if not peer_data:
        print(f"No peer data available for analysis of {target_ticker}.")
        return {"error": "No peer data available."}

    peer_stats = calculate_peer_stats(peer_data)
    comparison_table = create_peer_comparison_table(target_data, peer_stats)
    summary_score = calculate_summary_score(target_data, peer_stats)

    # Clean up NaN stats for a cleaner output
    clean_stats = {k: v for k, v in peer_stats.items()
                   if not all(np.isnan(val) for val in v.values())}

    analysis_result = {
        "peer_stats": clean_stats,
        "comparison_table": comparison_table,
        "summary_score": summary_score
    }
    return analysis_result


if __name__ == '__main__':
    from peer_finder import find_peers

    target = 'AAPL'
    print(f"--- Starting Peer Analysis for {target} ---")

    peers = find_peers(target)

    if peers:
        print(f"Found {len(peers)} peers: {peers[:5]}...")

        all_tickers = [target] + peers
        all_stocks_data = get_stocks_data(all_tickers)

        if all_stocks_data:
            analysis = analyze_with_peers(target, all_stocks_data)

            if analysis:
                print("\n--- Analysis Result ---")
                print("\nPeer Comparison Table:")
                print(analysis["comparison_table"])
                print(f"Summary Score: {analysis['summary_score']}/100")

    print("\n--- Analysis Complete ---")
