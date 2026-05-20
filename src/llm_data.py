import pandas as pd
import json
import time
from openai import OpenAI
import os

# --- CONFIGURATION ---
# This safely reads your API key from your computer's environment setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

INPUT_FILE = 'data/endo_symptoms.csv'
OUTPUT_FILE = 'data/llm_data_format_symptoms.csv'
MODEL = "gpt-4o-mini"  # cheap GPT-4 model

BATCH_SIZE = 5  # small batch for reliable extraction
TOKEN_COST_PER_1000 = 0.0001  # USD per 1k tokens

SYSTEM_PROMPT = """
You are a specialized clinical data annotator for Endometriosis research. 
Your task is to extract symptoms and TRANSFORM specific phrases based on clinical rules.

MAPPING RULES (MANDATORY):
- "GI", "GI issues", or "Bowel issues" -> Output 4 separate symptoms: "bloating", "abdominal pain", "diarrhea", "constipation"
- "UTI" -> Output 3 separate symptoms: "dysuria", "urinary hesitancy", "pelvic pain"
- "Endo belly" -> Output "Abdominal bloating"
- "Pain shooting down legs" -> Output "Radiating Pain" and set radiation to 1

EXTRACTION RULES:
- Identify every symptom. For each, create a JSON object:
  - symptom_phrase
  - intensity: 1–4
  - temporality: "Cyclical" or "Chronic"
  - radiation: 1 if pain radiates, 0 otherwise

EXAMPLE:
Input: "I have GI issues and back pain."
Output:
{"post_id": 101, "symptoms": [
  {"symptom_phrase": "bloating", "intensity": 2, "temporality": "Chronic", "radiation": 0},
  {"symptom_phrase": "abdominal pain", "intensity": 2, "temporality": "Chronic", "radiation": 0},
  {"symptom_phrase": "diarrhea", "intensity": 2, "temporality": "Chronic", "radiation": 0},
  {"symptom_phrase": "constipation", "intensity": 2, "temporality": "Chronic", "radiation": 0},
  {"symptom_phrase": "back pain", "intensity": 2, "temporality": "Chronic", "radiation": 0}
]}
Output ONLY valid JSON objects.
"""

def estimate_tokens(text):
    """Rough token estimate: 1 word ≈ 1.5 tokens"""
    words = len(text.split())
    return int(words * 1.5)

def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1000

def get_completion_with_backoff(text, retries=5):
    """Call OpenAI API with exponential backoff"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content).get('symptoms', [])
        except Exception as e:
            if i == retries - 1:
                print(f"Final error: {e}")
                return []
            wait = 2 ** i
            time.sleep(wait)
    return []

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    total_estimated_tokens = sum(estimate_tokens(text) for text in df['comment_text'])
    estimated_cost = estimate_cost(total_estimated_tokens)

    print(f"Estimated total tokens: {total_estimated_tokens}")
    print(f"Estimated total cost: ${estimated_cost:.4f}")

    if estimated_cost > 2:
        print("⚠️ Estimated cost exceeds $2. Consider using fewer posts or a cheaper model.")
        return

    all_extracted_rows = []

    print(f"Processing {len(df)} posts in batches of {BATCH_SIZE}...")

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        for index, row in batch.iterrows():
            post_id = row['post_id']
            y_label = row['Y']
            comment_text = row['comment_text']

            symptoms = get_completion_with_backoff(comment_text)

            if not symptoms:
                all_extracted_rows.append({
                    'post_id': post_id,
                    'Y': y_label,
                    'comment_text': comment_text,
                    'symptom_phrase': 'None detected',
                    'intensity': 0,
                    'temporality': 'Unknown',
                    'radiation': 0
                })
            else:
                for s in symptoms:
                    rad = s.get('radiation', 0)
                    if s.get('symptom_phrase') == "Radiating Pain":
                        rad = 1
                    all_extracted_rows.append({
                        'post_id': post_id,
                        'Y': y_label,
                        'comment_text': comment_text,
                        'symptom_phrase': s.get('symptom_phrase', 'Unknown'),
                        'intensity': s.get('intensity', 1),
                        'temporality': s.get('temporality', 'Chronic'),
                        'radiation': rad
                    })

        print(f"Processed posts {start + 1} to {min(start + BATCH_SIZE, len(df))}...")

    final_df = pd.DataFrame(all_extracted_rows)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"Extraction Complete! {len(final_df)} rows saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
