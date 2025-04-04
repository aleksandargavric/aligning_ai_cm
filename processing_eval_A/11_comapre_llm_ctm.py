import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Load original high-dimensional embeddings
df = pd.read_csv('ontouml_embeddings_llm.csv')
df1 = pd.read_csv('ontouml_embeddings.csv')

# Parse embeddings from string to numpy arrays
llm_embs = df['llm_embedding'].apply(lambda x: np.fromstring(x, sep=','))
cmt_embs = df1['cmt_embedding'].apply(lambda x: np.fromstring(x, sep=','))

# Merge dataframes on 'key'
merged_df = pd.merge(df[['key', 'llm_embedding']], df1[['key', 'cmt_embedding']], on='key')

# Also load 2D coordinates for LLM
df_2d = pd.read_csv("ontouml_embeddings_llm_2d.csv")
df1_2d = pd.read_csv("ontouml_embeddings_2d.csv")
X = df_2d[["llm_x", "llm_y"]].values
y = df1_2d[["cmt_x", "cmt_y"]].values

# === Scale ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# === Define MLP model ===
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# === Load trained model ===
model = DeepMLP()
model.load_state_dict(torch.load("mlp_embedding_2d_aligner.pt"))
model.eval()

# === Predict embeddings ===
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    pred_scaled = model(X_tensor).numpy()

pred_cmt_orig = scaler_y.inverse_transform(pred_scaled)

# === Cosine similarity ===
# Prepare actual high-dimensional vectors
llm_vectors = np.vstack(llm_embs.values)
cmt_vectors = np.vstack(cmt_embs.values)

# Compute cosine similarity
sim_llm_to_cmt = np.diag(cosine_similarity(X, y))
sim_pred_to_cmt = np.diag(cosine_similarity(pred_cmt_orig, y))

# Print mean similarities
print(f"Average cosine similarity (LLM ? CMT):     {np.mean(sim_llm_to_cmt):.4f}")
print(f"Average cosine similarity (Predicted ? CMT): {np.mean(sim_pred_to_cmt):.4f}")
