import torch
from transformers import pipeline
import json

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

    # 1. Company Snapshot Data
    snapshot = {
        "Company Name": target_data.get('longName', target_ticker),
        "Sector": target_data.get('Sector', 'N/A'),
        "Industry": target_data.get('Industry', 'N/A'),
        "Market Cap": f"${target_data.get('Market Cap', 0):,.0f}"
    }

    # 2. Key Metrics Table
    key_metrics_str = (
        "| Metric             | Value          |\n"
        "|--------------------|----------------|\n"
        f"| P/E                | {target_data.get('P/E', 'N/A'):.2f}         |\n"
        f"| ROE                | {target_data.get('ROE', 0):.2%}         |\n"
        f"| EPS Growth         | {target_data.get('EPS Growth', 0):.2%}         |\n"
        f"| Debt/Equity        | {target_data.get('Debt/Equity', 'N/A'):.2f}         |\n"
    )

    # 4. Peer Comparison Table
    comparison_table = peer_analysis.get("comparison_table", "No comparison data available.")

    # 5. Summary Score
    summary_score = peer_analysis.get("summary_score", "N/A")

    # Assemble the prompt
    prompt = f"""
You are an expert AI Investment Analyst. Your task is to provide a professional, data-driven analysis for the company {target_ticker} based on the information provided below. Use an executive-level, clear, and concise tone in Thai.

**Company Data for {target_ticker}:**

**1) Company Snapshot:**
- Name: {snapshot['Company Name']}
- Sector / Industry: {snapshot['Sector']} / {snapshot['Industry']}
- Market Cap: {snapshot['Market Cap']}

**2) Key Metrics:**
{key_metrics_str}

**3) Peer Comparison Table:**
{comparison_table}

**4) Summary Score vs Peers:**
- Score: {summary_score}/100 (A score of 100 means the company is significantly stronger than its peers, 50 is average, and below 40 is weaker).

**Your Task (Provide the output in the following structure):**

**1) Company Snapshot:**
*   Provide a 10-second highlight about the company.

**2) Competitive Position Summary:**
*   **Strengths:** (Based on the data, what are its key advantages over peers?)
*   **Weaknesses:** (Based on the data, where does it lag behind its peers?)
*   **Risk Factors:** (What potential risks are implied by the financial data?)
*   **Moat:** (Does the data suggest a strong competitive advantage or 'moat'?)

**3) Valuation Assessment:**
*   Based on P/E and Forward P/E vs peers, is the stock's valuation: Undervalued, Fairly Valued, or Overvalued?
*   Provide a brief forward-looking view on the valuation.

**4) Investment Thesis (in the style of a fund manager):**
*   **Bull Case:** (What is the primary reason to be optimistic about this stock?)
*   **Bear Case:** (What is the primary reason for caution?)
*   **Base Case:** (What is the most likely scenario and what are the key drivers?)

**5) Actionable Insight:**
*   Provide a 2-3 sentence strategic summary. Do not give direct buy/sell advice. Focus on the trade-offs between strengths (e.g., fundamentals) and weaknesses (e.g., valuation).
"""
    return prompt

def generate_qualitative_analysis(prompt: str, generator) -> str:
    """Generates the qualitative analysis text using the LLM."""
    if generator is None:
        return "LLM not initialized. Could not generate analysis."

    messages = [{"role": "user", "content": prompt}]

    try:
        outputs = generator(messages, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50)

        # Extract the generated text from the last message in the conversation
        if (isinstance(outputs, list) and len(outputs) > 0 and
            isinstance(outputs[0], list) and len(outputs[0]) > 0 and
            isinstance(outputs[0][-1], dict) and 'content' in outputs[0][-1]):
            return outputs[0][-1]['content'].strip()

        # Fallback for other potential structures
        raw_output = outputs[0]["generated_text"]
        if isinstance(raw_output, list):
             return raw_output[-1]['content'].strip()
        if prompt in raw_output:
            return raw_output.split(prompt, 1)[-1].strip()

        return raw_output.strip()

    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return f"An error occurred during text generation: {e}"

if __name__ == '__main__':
    # --- Example Usage with hardcoded data for faster testing ---
    print("--- Generating Full Report with SAMPLE DATA ---")

    target = 'MSFT'
    sample_target_data = {
        'longName': 'Microsoft Corporation',
        'Sector': 'Technology',
        'Industry': 'Software - Infrastructure',
        'Market Cap': 3575501553664,
        'P/E': 34.21,
        'Forward P/E': 32.17,
        'PEG Ratio': None,
        'ROE': 0.3224,
        'Operating Margin': 0.4887,
        'Gross Margin': 0.6876,
        'Revenue Growth': 0.184,
        'EPS Growth': 0.127,
        'Debt/Equity': 33.15
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
        full_prompt = create_full_prompt(target, sample_target_data, sample_peer_analysis)

        print("Generating analysis... (This may take a moment)")
        qualitative_analysis = generate_qualitative_analysis(full_prompt, llm)

        print("\n\n--- GENERATED REPORT ---")
        print(qualitative_analysis)
        print("--- END OF REPORT ---")
