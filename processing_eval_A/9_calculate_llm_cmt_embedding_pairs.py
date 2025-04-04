import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm

# Load CSVs
llm_df = pd.read_csv('ontouml_models_descriptions.csv')

# Generate embeddings
llm_embeddings = []

for _, row in tqdm(llm_df.iterrows(), total=len(llm_df)):
    try:
        llm_emb = np.array(ollama.embeddings(model='all-minilm', prompt=row['description'])['embedding'])
    except Exception as e:
        print(f"Error processing key {row['key']}: {e}")
        llm_emb = np.full(384, np.nan)  # assuming embedding size is 384

    llm_embeddings.append(llm_emb)

# Create DataFrame
embedding_df = pd.DataFrame({
    'key': llm_df['key'],
    'llm_embedding': llm_embeddings
})

# Save to CSV (convert arrays to string for CSV compatibility)
embedding_df['llm_embedding'] = embedding_df['llm_embedding'].apply(lambda x: ','.join(map(str, x)))

embedding_df.to_csv('ontouml_embeddings_llm.csv', index=False)
print("Embeddings saved to ontouml_embeddings_llm.csv")