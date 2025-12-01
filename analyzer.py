import torch
from transformers import pipeline
import json

# Using a more lightweight model to prevent memory-related crashes.
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def calculate_score(data: dict) -> float:
    """Calculates a score from 0.0 to 1.0 based on raw financial metrics."""
    score = 0.0
    try:
        # Use .get() with a default value to handle missing data (None) gracefully
        roe = data.get("ROE") or 0.0
        # Correctly normalize the D/E ratio by dividing by 100
        de_ratio = (data.get("Debt to Equity Ratio") or float('inf')) / 100.0
        rev_growth = data.get("Quarterly Revenue Growth (yoy)") or 0.0
        margins = data.get("Profit Margins") or 0.0

        if roe > 0.20: score += 0.4
        elif roe > 0.15: score += 0.3
        elif roe > 0.05: score += 0.1
        if de_ratio < 0.5: score += 0.3
        elif de_ratio < 1.0: score += 0.2
        elif de_ratio < 2.0: score += 0.1
        if rev_growth > 0.10: score += 0.2
        elif rev_growth > 0.05: score += 0.1
        if margins > 0.20: score += 0.1
    except (ValueError, TypeError):
        return 0.0
    return min(round(score, 2), 1.0)

def generate_strength(score: float) -> str:
    """Generates a Thai strength summary based on the calculated score."""
    if score >= 0.7:
        return "พื้นฐานแข็งแกร่ง"
    elif score >= 0.4:
        return "พื้นฐานปานกลาง"
    else:
        return "พื้นฐานอ่อนแอและมีความเสี่ยง"

def create_prompt(data: dict, ticker: str) -> str:
    """Creates a simple prompt with formatted data for the LLM."""
    # Format the raw data into a human-readable string for the prompt
    formatted_data = {
        "ROE": f"{data.get('ROE', 0):.2%}" if data.get('ROE') is not None else "N/A",
        "Debt to Equity Ratio": f"{data.get('Debt to Equity Ratio', 0):.2f}" if data.get('Debt to Equity Ratio') is not None else "N/A",
        "Quarterly Revenue Growth (yoy)": f"{data.get('Quarterly Revenue Growth (yoy)', 0):.2%}" if data.get('Quarterly Revenue Growth (yoy)') is not None else "N/A",
        "Profit Margins": f"{data.get('Profit Margins', 0):.2%}" if data.get('Profit Margins') is not None else "N/A"
    }
    data_string = ", ".join([f"{key}: {value}" for key, value in formatted_data.items()])

    prompt = f"""
    Based on the following data for {ticker} ({data_string}), write a single, brief sentence in Thai summarizing the financial situation.
    """
    return prompt

def analyze_financials(ticker: str, data: dict) -> dict:
    """
    Uses Python for scoring and JSON assembly, and an LLM for a simple reasoning sentence.
    """
    if not data:
        return None

    # Step 1: Programmatic Scoring and Strength Generation
    score = calculate_score(data)
    strength = generate_strength(score)

    # Step 2: LLM for a simple reasoning sentence
    prompt = create_prompt(data, ticker)
    messages = [{"role": "user", "content": prompt}]

    reasoning = "ไม่สามารถสร้างคำวิเคราะห์ได้" # Default value
    try:
        outputs = generator(messages, max_new_tokens=128, do_sample=False)
        generated_text = outputs[0]["generated_text"][-1]['content'].strip()
        if generated_text:
            reasoning = generated_text
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        reasoning = f"เกิดข้อผิดพลาดในการสร้างคำวิเคราะห์: {e}"

    # Step 3: Final JSON Assembly
    return {
        "strength": strength,
        "reasoning": reasoning,
        "score": score
    }

if __name__ == '__main__':
    sample_ticker = 'AAPL'
    # Use raw data for testing, as the fetcher now returns
    sample_data = {
        'ROE': 1.7142, 'Debt to Equity Ratio': 152.41,
        'Quarterly Revenue Growth (yoy)': 0.079, 'Profit Margins': 0.2692
    }
    print(f"--- Starting analysis for {sample_ticker} ---")
    analysis_result = analyze_financials(sample_ticker, sample_data)
    if analysis_result:
        print("\n--- Analysis Result ---")
        print(json.dumps(analysis_result, indent=4, ensure_ascii=False))
