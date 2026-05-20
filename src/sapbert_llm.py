import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os

# --- FILE PATHS (Relative to project folder) ---
LLM_EXTRACTED_FILE = 'data/llm_data_format_symptoms.csv'  
HPO_REFERENCE_FILE = 'data/hpo48_combined_extracted.csv'  
MASTER_TABLE_FILE = 'data/master2_ontological_table.csv'

# --- S-BERT MODEL ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPER FUNCTIONS ---
def normalize_hpo_id(hpo_id):
    """Ensure HPO IDs are consistently formatted with colons (HP:XXXXXX)."""
    if pd.isna(hpo_id): 
        return hpo_id
    return str(hpo_id).replace('_', ':')

# --- MAIN FUNCTION ---
def main():
    print("--- Starting Segments 2 & 3: Mapping & Enrichment ---")

    # 1. Load Data
    try:
        print("Loading LLM-extracted data and HPO reference...")
        llm_df = pd.read_csv(LLM_EXTRACTED_FILE)
        hpo_df = pd.read_csv(HPO_REFERENCE_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Normalize HPO IDs
    hpo_df['id'] = hpo_df['id'].apply(normalize_hpo_id)

    # 2. Prepare Reference Embeddings (Labels + Synonyms)
    print("Vectorizing HPO reference terms (labels + synonyms)...")
    hpo_target_texts = []
    hpo_target_ids = []

    for _, row in hpo_df.iterrows():
        hpo_target_texts.append(str(row['class_label']))
        hpo_target_ids.append(row['id'])
        if pd.notna(row['synonyms']):
            synonyms = [s.strip() for s in str(row['synonyms']).split('|') if s.strip()]
            for s in synonyms:
                hpo_target_texts.append(s)
                hpo_target_ids.append(row['id'])

    hpo_embeddings = model.encode(hpo_target_texts, convert_to_tensor=True, show_progress_bar=True)

    # 3. Encode Extracted Symptom Phrases
    print("Vectorizing extracted symptom phrases...")
    unique_phrases = llm_df['symptom_phrase'].unique()
    phrase_embeddings = model.encode(unique_phrases, convert_to_tensor=True)

    # Compute cosine similarity
    print("Computing cosine similarity and mapping...")
    cosine_scores = util.cos_sim(phrase_embeddings, hpo_embeddings)

    # Map phrases to best-matching HPO ID
    phrase_to_hpo = {}
    phrase_to_score = {}

    for i, phrase in enumerate(unique_phrases):
        best_idx = int(torch.argmax(cosine_scores[i]))
        best_score = float(cosine_scores[i][best_idx])
        phrase_to_hpo[phrase] = hpo_target_ids[best_idx]
        phrase_to_score[phrase] = best_score

    # Apply mapping and scale confidence to 0-100%
    llm_df['HPO_ID'] = llm_df['symptom_phrase'].map(phrase_to_hpo)
    llm_df['Mapping_Confidence'] = (llm_df['symptom_phrase'].map(phrase_to_score) * 100).round().astype(int)

    # 4. Ontological Enrichment
    print("Performing ontological enrichment...")
    enrichment_cols = ['id', 'class_label', 'parent_classes', 'object_properties_simplified']
    hpo_enrichment = hpo_df[enrichment_cols].drop_duplicates(subset=['id'])

    master_table = llm_df.merge(
        hpo_enrichment,
        left_on='HPO_ID',
        right_on='id',
        how='left'
    ).drop(columns=['id'])

    # Fill NaNs for cleaner analysis
    master_table['parent_classes'] = master_table['parent_classes'].fillna('Unknown')
    master_table['object_properties_simplified'] = master_table['object_properties_simplified'].fillna('None')

    # 5. Save the master table
    master_table.to_csv(MASTER_TABLE_FILE, index=False)

    # Summary
    print("-" * 30)
    print("Workflow Complete!")
    print(f"Total Rows: {len(master_table)}")
    print(f"Average Mapping Confidence: {master_table['Mapping_Confidence'].mean():.2f}%")
    print(f"Master Table saved to: {MASTER_TABLE_FILE}")
    print("Comment text preserved for reference.")

if __name__ == "__main__":
    main()
