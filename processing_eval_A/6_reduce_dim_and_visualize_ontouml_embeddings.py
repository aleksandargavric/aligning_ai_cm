import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# Load embeddings
df = pd.read_csv('ontouml_embeddings.csv')
nlt_embs = df['nlt_embedding'].apply(lambda x: np.fromstring(x, sep=','))
cmt_embs = df['cmt_embedding'].apply(lambda x: np.fromstring(x, sep=','))

# Combine all embeddings
all_embeddings = np.vstack(nlt_embs.tolist() + cmt_embs.tolist())
n = len(df)

# t-SNE 2D
tsne_2d = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne_2d.fit_transform(all_embeddings)
nlt_2d, cmt_2d = embeddings_2d[:n], embeddings_2d[n:]

# Save 2D coordinates
df_2d = pd.DataFrame({
    'nlt_x': nlt_2d[:, 0],
    'nlt_y': nlt_2d[:, 1],
    'cmt_x': cmt_2d[:, 0],
    'cmt_y': cmt_2d[:, 1]
})
df_2d.to_csv("ontouml_embeddings_2d.csv", index=False)

# Plot 2D
plt.figure(figsize=(6, 6))
for i in range(n):
    plt.arrow(nlt_2d[i, 0], nlt_2d[i, 1],
              cmt_2d[i, 0] - nlt_2d[i, 0],
              cmt_2d[i, 1] - nlt_2d[i, 1],
              color='orange', alpha=0.8, head_width=0.5, length_includes_head=True)
plt.title("OntoUML Embeddings: NLT to CMT (2D)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("ontouml_embeddings_2d.png", dpi=300)
plt.close()

# t-SNE 3D
tsne_3d = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne_3d.fit_transform(all_embeddings)
nlt_3d, cmt_3d = embeddings_3d[:n], embeddings_3d[n:]

# Save 3D coordinates
df_3d = pd.DataFrame({
    'nlt_x': nlt_3d[:, 0],
    'nlt_y': nlt_3d[:, 1],
    'nlt_z': nlt_3d[:, 2],
    'cmt_x': cmt_3d[:, 0],
    'cmt_y': cmt_3d[:, 1],
    'cmt_z': cmt_3d[:, 2]
})
df_3d.to_csv("ontouml_embeddings_3d.csv", index=False)

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for i in range(n):
    ax.quiver(nlt_3d[i, 0], nlt_3d[i, 1], nlt_3d[i, 2],
              cmt_3d[i, 0] - nlt_3d[i, 0],
              cmt_3d[i, 1] - nlt_3d[i, 1],
              cmt_3d[i, 2] - nlt_3d[i, 2],
              color='orange', alpha=0.8, linewidth=1)

ax.set_title("OntoUML Embeddings: NLT to CMT (3D)", fontsize=14)
plt.tight_layout()
plt.savefig("ontouml_embeddings_3d.png", dpi=300)
plt.close()
