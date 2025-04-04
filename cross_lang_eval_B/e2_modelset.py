import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import ollama  # Ensure this API client is installed and configured

# -----------------------------
# 1. Load and Group JSON Data
# -----------------------------
with open('extracted_modelset.json', 'r') as f:
    data = json.load(f)

groups = {}
for item in data:
    model = item.get("model-name")
    if model:
        groups.setdefault(model, []).append(item)

model_names = []
nlt_texts = []
cmt_texts = []
nlt_embeddings = []
cmt_embeddings = []

# -----------------------------------------------
# 2. Construct NLT and CMT strings and get embeddings
# -----------------------------------------------
for model, items in groups.items():
    model_names.append(model)
    
    # Create NLT by concatenating all "name" values with a space.
    names = [item.get("name", "").strip() for item in items if item.get("name")]
    nlt = " ".join(names)
    nlt_texts.append(nlt)
    
    # Create CMT using the specified template for each entry.
    cmt_parts = []
    for item in items:
        term = item.get("name", "").strip()
        if not term:
            continue
        cmt_str = f"uml term '{term}'"
        term_type = item.get("eClass")
        stereotype = item.get("qualifiedName")
        if term_type and str(term_type).strip():
            cmt_str += f" is of eClass '{term_type}'"
        if stereotype and str(stereotype).strip():
            # If a type was added, then add stereotype with "and"
            if "is of eClass" in cmt_str:
                cmt_str += f" and it's qualified name is '{stereotype}'"
            else:
                cmt_str += f" with qualified name '{stereotype}'"
        cmt_str += ";"  # End sentence
        cmt_parts.append(cmt_str)
    cmt = " ".join(cmt_parts)
    cmt_texts.append(cmt)
    
    # Fetch embeddings for the NLT and CMT strings.
    nlt_emb = np.array(ollama.embeddings(model='mxbai-embed-large', prompt=nlt)['embedding'])
    cmt_emb = np.array(ollama.embeddings(model='mxbai-embed-large', prompt=cmt)['embedding'])
    nlt_embeddings.append(nlt_emb)
    cmt_embeddings.append(cmt_emb)

# ---------------------------
# 3. Save CSV with the embeddings
# ---------------------------
# Store embeddings as JSON strings for clarity.
df = pd.DataFrame({
    'model-name': model_names,
    'NLT': nlt_texts,
    'NLT_embedding': [json.dumps(emb.tolist()) for emb in nlt_embeddings],
    'CMT': cmt_texts,
    'CMT_embedding': [json.dumps(emb.tolist()) for emb in cmt_embeddings],
})
df.to_csv('model_embeddings_modelset.csv', index=False)
print("CSV file saved as model_embeddings.csv.")

# -----------------------------------
# 4. Prepare embeddings for reduction
# -----------------------------------
# We combine the embeddings in order: [NLT1, CMT1, NLT2, CMT2, ...]
all_embeddings = []
for nlt_emb, cmt_emb in zip(nlt_embeddings, cmt_embeddings):
    all_embeddings.append(nlt_emb)
    all_embeddings.append(cmt_emb)
all_embeddings = np.vstack(all_embeddings)

# ---------------------------------------------
# 5. Dimensionality Reduction Functions
# ---------------------------------------------
def reduce_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

def reduce_tsne(embeddings, n_components):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(embeddings)

def reduce_umap(embeddings, n_components):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)

# Compute 2D and 3D reductions for each method.
reductions = {}
for method, func in zip(["PCA", "t-SNE", "UMAP"], [reduce_pca, reduce_tsne, reduce_umap]):
    reductions[f"{method}_2d"] = func(all_embeddings, 2)
    reductions[f"{method}_3d"] = func(all_embeddings, 3)

# -----------------------------------------------------
# 6. Create the Big Plot with Multiple Subplots
# -----------------------------------------------------
# Layout: 3 rows (methods) x 2 columns (2D and 3D)
fig = plt.figure(figsize=(16, 18))
methods = ["PCA", "t-SNE", "UMAP"]

for i, method in enumerate(methods):
    # 2D Subplot (Left Column)
    ax2d = fig.add_subplot(3, 2, i*2 + 1)
    ax2d.set_title(f"{method} 2D")
    proj_2d = reductions[f"{method}_2d"]
    # Plot each pair (NLT and CMT) and connect them.
    for j in range(0, proj_2d.shape[0], 2):
        point_nlt = proj_2d[j]
        point_cmt = proj_2d[j+1]
        ax2d.scatter(point_nlt[0], point_nlt[1], marker='o', color='blue', label='NLT' if j == 0 else "")
        ax2d.scatter(point_cmt[0], point_cmt[1], marker='^', color='red', label='CMT' if j == 0 else "")
        ax2d.plot([point_nlt[0], point_cmt[0]], [point_nlt[1], point_cmt[1]], color='gray', linestyle='--')
    ax2d.legend()
    ax2d.set_xlabel('Component 1')
    ax2d.set_ylabel('Component 2')
    
    # 3D Subplot (Right Column)
    ax3d = fig.add_subplot(3, 2, i*2 + 2, projection='3d')
    ax3d.set_title(f"{method} 3D")
    proj_3d = reductions[f"{method}_3d"]
    for j in range(0, proj_3d.shape[0], 2):
        point_nlt = proj_3d[j]
        point_cmt = proj_3d[j+1]
        ax3d.scatter(point_nlt[0], point_nlt[1], point_nlt[2], marker='o', color='blue', label='NLT' if j == 0 else "")
        ax3d.scatter(point_cmt[0], point_cmt[1], point_cmt[2], marker='^', color='red', label='CMT' if j == 0 else "")
        ax3d.plot([point_nlt[0], point_cmt[0]],
                  [point_nlt[1], point_cmt[1]],
                  [point_nlt[2], point_cmt[2]],
                  color='gray', linestyle='--')
    ax3d.legend()
    ax3d.set_xlabel('Component 1')
    ax3d.set_ylabel('Component 2')
    ax3d.set_zlabel('Component 3')

plt.tight_layout()
plt.savefig('pairs.png')
plt.show()
print("Plot saved as pairs.png.")
