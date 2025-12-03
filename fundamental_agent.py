import argparse
from data_fetcher import get_stocks_data
from analyzer import analyze_with_peers
from peer_finder import find_peers
from report_generator import (
    initialize_llm, create_full_prompt, generate_qualitative_analysis
)


def main():
    """
    Main function for the AI Investor Agent.
    Orchestrates the process of fetching, analyzing, and reporting on a stock,
    including peer comparison.
    """
    parser = argparse.ArgumentParser(
        description="AI Investor Agent for Stock Analysis"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="The stock ticker symbol to analyze (e.g., AAPL)."
    )
    args = parser.parse_args()
    target_ticker = args.ticker.upper()

    print(f"🚀 Starting full analysis for {target_ticker}...")

    # Step 1: Peer Discovery
    print("\nStep 1: Finding peer companies...")
    peers = find_peers(target_ticker)
    if not peers:
        print(f"⚠️ Could not find peers for {target_ticker}.")
    else:
        print(f"✅ Found {len(peers)} peers. Example: {peers[:5]}")

    # Step 2: Data Fetching
    print("\nStep 2: Fetching financial data for target and peers...")
    all_tickers = [target_ticker] + peers
    all_data = get_stocks_data(all_tickers)

    if target_ticker not in all_data:
        print(f"❌ Critical error: Failed to fetch data for {target_ticker}.")
        return
    print("✅ Data fetched successfully.")

    # Step 3: Quantitative Peer Analysis
    print("\nStep 3: Performing quantitative peer analysis...")
    peer_analysis = analyze_with_peers(target_ticker, all_data)
    if not peer_analysis:
        print("❌ Could not perform peer analysis. Exiting.")
        return
    print("✅ Quantitative analysis complete.")

    # Step 4: Qualitative Analysis with LLM
    print("\nStep 4: Generating qualitative analysis with LLM...")
    llm = initialize_llm()
    if not llm:
        print("❌ Failed to initialize LLM.")
        qualitative_analysis = "LLM analysis could not be generated."
    else:
        print("✅ LLM initialized. Generating report...")
        prompt = create_full_prompt(
            target_ticker, all_data[target_ticker], peer_analysis
        )
        qualitative_analysis = generate_qualitative_analysis(prompt, llm)
        print("✅ Qualitative analysis complete.")

    # Step 5: Final Report Assembly & Display
    target_data = all_data[target_ticker]
    print("\n" + "="*50)
    print(f"📈 COMPREHENSIVE INVESTMENT ANALYSIS: {target_ticker}")
    print("="*50 + "\n")

    print("--- 1) Company Snapshot & Key Metrics ---")
    print(f"Name: {target_data.get('longName', target_ticker)}")
    print(f"Sector / Industry: {target_data.get('Sector', 'N/A')} / "
          f"{target_data.get('Industry', 'N/A')}")
    print(f"Market Cap: ${target_data.get('Market Cap', 0):,.0f}\n")

    key_metrics_table = (
        "| Metric             | Value          |\n"
        "|--------------------|----------------|\n"
        f"| P/E                | {target_data.get('P/E', 'N/A'):.2f}         |\n"
        f"| Forward P/E        | {target_data.get('Forward P/E', 'N/A'):.2f} |\n"
        f"| PEG Ratio          | {target_data.get('PEG Ratio', 'N/A'):.2f}   |\n"
        f"| ROE                | {target_data.get('ROE', 0):.2%}         |\n"
        f"| EPS Growth         | {target_data.get('EPS Growth', 0):.2%}   |\n"
        f"| Revenue Growth     | {target_data.get('Revenue Growth', 0):.2%}   |\n"
        f"| Debt/Equity        | {target_data.get('Debt/Equity', 'N/A'):.2f} |\n"
    )
    print(key_metrics_table)
    print("\n--- 2) Peer List ---")
    print(f"Found {len(peers)} peers: {peers}\n")
    print("--- 3) Peer Comparison Table ---")
    print(peer_analysis["comparison_table"])
    print("\n--- 4) Qualitative Analysis ---")
    print(qualitative_analysis)

    print("\n" + "="*50)
    print("✅ Analysis Complete.")
    print("="*50)


if __name__ == "__main__":
    main()
