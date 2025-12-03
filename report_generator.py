import torch
from transformers import pipeline


def initialize_llm():
    """Initializes and returns the text generation pipeline."""
    try:
        generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return generator
    except Exception as e:
        print(f"Error initializing LLM pipeline: {e}")
        return None


def create_full_prompt(target_ticker: str, target_data: dict, peer_analysis: dict) -> str:
    """Creates a comprehensive prompt for the LLM based on all available data."""
    snapshot = {
        "Company Name": target_data.get('longName', target_ticker),
        "Sector": target_data.get('Sector', 'N/A'),
        "Industry": target_data.get('Industry', 'N/A'),
        "Market Cap": f"${target_data.get('Market Cap', 0):,.0f}"
    }
    key_metrics_str = (
        "| Metric             | Value          |\n"
        "|--------------------|----------------|\n"
        f"| P/E                | {target_data.get('P/E', 'N/A'):.2f}         |\n"
        f"| ROE                | {target_data.get('ROE', 0):.2%}         |\n"
        f"| EPS Growth         | {target_data.get('EPS Growth', 0):.2%}         |\n"
        f"| Debt/Equity        | {target_data.get('Debt/Equity', 'N/A'):.2f}         |\n"
    )
    comparison_table = peer_analysis.get("comparison_table", "No comparison data available.")
    summary_score = peer_analysis.get("summary_score", "N/A")

    prompt = (
        f"You are an expert AI Investment Analyst. Your task is to provide a professional, "
        f"data-driven analysis for the company {target_ticker} based on the information provided below. "
        f"Use an executive-level, clear, and concise tone in Thai.\n\n"
        f"**Company Data for {target_ticker}:**\n\n"
        f"**1) Company Snapshot:**\n"
        f"- Name: {snapshot['Company Name']}\n"
        f"- Sector / Industry: {snapshot['Sector']} / {snapshot['Industry']}\n"
        f"- Market Cap: {snapshot['Market Cap']}\n\n"
        f"**2) Key Metrics:**\n{key_metrics_str}\n\n"
        f"**3) Peer Comparison Table:**\n{comparison_table}\n\n"
        f"**4) Summary Score vs Peers:**\n"
        f"- Score: {summary_score}/100 (A score of 100 means the company is significantly stronger "
        f"than its peers, 50 is average, and below 40 is weaker).\n\n"
        f"**Your Task (Provide the output in the following structure):**\n\n"
        f"**1) Company Snapshot:**\n*   Provide a 10-second highlight about the company.\n\n"
        f"**2) Competitive Position Summary:**\n"
        f"*   **Strengths:** (Based on the data, what are its key advantages over peers?)\n"
        f"*   **Weaknesses:** (Based on the data, where does it lag behind its peers?)\n"
        f"*   **Risk Factors:** (What potential risks are implied by the financial data?)\n"
        f"*   **Moat:** (Does the data suggest a strong competitive advantage or 'moat'?)\n\n"
        f"**3) Valuation Assessment:**\n"
        f"*   Based on P/E and Forward P/E vs peers, is the stock's valuation: "
        f"Undervalued, Fairly Valued, or Overvalued?\n"
        f"*   Provide a brief forward-looking view on the valuation.\n\n"
        f"**4) Investment Thesis (in the style of a fund manager):**\n"
        f"*   **Bull Case:** (What is the primary reason to be optimistic about this stock?)\n"
        f"*   **Bear Case:** (What is the primary reason for caution?)\n"
        f"*   **Base Case:** (What is the most likely scenario and what are the key drivers?)\n\n"
        f"**5) Actionable Insight:**\n"
        f"*   Provide a 2-3 sentence strategic summary. Do not give direct buy/sell advice. "
        f"Focus on the trade-offs between strengths (e.g., fundamentals) and weaknesses (e.g., valuation).\n"
    )
    return prompt


def generate_qualitative_analysis(prompt: str, generator) -> str:
    """Generates the qualitative analysis text using the LLM."""
    if generator is None:
        return "LLM not initialized. Could not generate analysis."

    messages = [{"role": "user", "content": prompt}]

    try:
        outputs = generator(messages, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50)
        response = outputs[0]["generated_text"]
        if isinstance(response, list):
            # Handle list of conversation turns
            return response[-1]['content'].strip()

        # Fallback for simple string output
        if prompt in response:
            return response.split(prompt, 1)[-1].strip()
        return response.strip()

    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return f"An error occurred during text generation: {e}"


if __name__ == '__main__':
    # --- Example Usage with hardcoded data for faster testing ---
    print("--- Generating Full Report with SAMPLE DATA ---")

    target = 'MSFT'
    sample_target_data = {
        'longName': 'Microsoft Corporation', 'Sector': 'Technology',
        'Industry': 'Software - Infrastructure', 'Market Cap': 3575501553664,
        'P/E': 34.21, 'Forward P/E': 32.17, 'PEG Ratio': None, 'ROE': 0.3224,
        'Operating Margin': 0.4887, 'Gross Margin': 0.6876,
        'Revenue Growth': 0.184, 'EPS Growth': 0.127, 'Debt/Equity': 33.15
    }
    sample_peer_analysis = {
        'comparison_table': (
            "| Metric             | Target Company | Peer Group (Mean) | Comparison vs Mean |\n"
            "|--------------------|----------------|-------------------|--------------------|\n"
            "| P/E                | 34.21          | 30.00             | +14.0%             |\n"
            "| ROE                | 32.24%         | 25.00%            | +28.9%             |\n"
        ),
        'summary_score': 75
    }

    print("\nInitializing LLM...")
    llm = initialize_llm()

    if llm:
        print("Creating prompt...")
        full_prompt = create_full_prompt(
            target, sample_target_data, sample_peer_analysis
        )

        print("Generating analysis... (This may take a moment)")
        qualitative_analysis = generate_qualitative_analysis(full_prompt, llm)

        print("\n\n--- GENERATED REPORT ---")
        print(qualitative_analysis)
        print("--- END OF REPORT ---")
