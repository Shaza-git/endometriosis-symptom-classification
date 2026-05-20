# endometriosis-symptom-classification

# Impact of Domain Ontology on Endometriosis Symptom Classification

This repository contains the data pipeline, feature engineering scripts, and machine learning evaluation code for our study:
> **Assessing the Impact of Domain Ontology on Linear and Tree-Based Supervised Learning for Endometriosis Symptom Classification**

## 📌 Project Overview
This study explores how enriching informal online medical narratives (from social media like Reddit) with structured formal medical ontologies impacts downstream machine learning performance. We map clinical symptom attributes via embedding models (SapBERT) to the Human Phenotype Ontology (HPO), validating performance shifts across Logistic Regression and optimized CatBoost architectures.

## 📁 Repository Structure
* `data/`: Processed training matrices, reference files, and raw configurations.
  * `endo_symptoms.csv`: Core dataset observations.
  * `hpo48_combined_extracted.csv`: Human Phenotype Ontology extraction reference mappings.
  * `final3_training_data.csv`: Final processed feature matrix utilized in modeling.
* `src/`: Complete source code pipeline.
  * `llm_data.py`: Zero-shot clinical property formatting via language models.
  * `sapbert_llm.py`: Semantic entity linkage to HPO concept IDs.
  * `catboost_without_ontology.py`: Supervised learning baseline model evaluation.
  * `real_final_comparsion.py`: Main cross-validation evaluation comparing baseline and ontology-infused models.
* `requirements.txt`: Python package dependencies necessary to run the project.
* `Springer_Applied_Intelligence.pdf`: Full text preprint manuscript.

## 🚀 Execution Order
To reproduce the experimental results outlined in the manuscript, install the dependencies and execute the source scripts sequentially:

```bash
# Install the required packages
pip install -r requirements.txt

# 1. Information Extraction (Ensure your OPENAI_API_KEY environment variable is set)
python src/llm_data.py

# 2. Ontological Grounding & Feature Enrichment
python src/sapbert_llm.py

# 3. Comparative Evaluation (Generates final manuscript performance statistics)
python src/real_final_comparsion.py
