import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm

# Load CSVs
nlt_df = pd.read_csv('ontouml_models_nlt_serializations.csv')
cmt_df = pd.read_csv('ontouml_models_cmt_serializations.csv')

# Merge on key
merged_df = pd.merge(nlt_df, cmt_df, on='key', suffixes=('_nlt', '_cmt'))

# Generate embeddings
nlt_embeddings = []
cmt_embeddings = []

for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    try:
        nlt_emb = np.array(ollama.embeddings(model='all-minilm', prompt=row['serialization_nlt'])['embedding'])
        cmt_emb = np.array(ollama.embeddings(model='all-minilm', prompt=row['serialization_cmt'])['embedding'])
    except Exception as e:
        print(f"Error processing key {row['key']}: {e}")
        nlt_emb = np.full(384, np.nan)  # assuming embedding size is 384
        cmt_emb = np.full(384, np.nan)

    nlt_embeddings.append(nlt_emb)
    cmt_embeddings.append(cmt_emb)

# Create DataFrame
embedding_df = pd.DataFrame({
    'key': merged_df['key'],
    'nlt_embedding': nlt_embeddings,
    'cmt_embedding': cmt_embeddings
})

# Save to CSV (convert arrays to string for CSV compatibility)
embedding_df['nlt_embedding'] = embedding_df['nlt_embedding'].apply(lambda x: ','.join(map(str, x)))
embedding_df['cmt_embedding'] = embedding_df['cmt_embedding'].apply(lambda x: ','.join(map(str, x)))

embedding_df.to_csv('ontouml_embeddings.csv', index=False)
print("Embeddings saved to ontouml_embeddings.csv")