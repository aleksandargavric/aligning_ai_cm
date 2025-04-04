import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
df = pd.read_csv("ontouml_embeddings_2d.csv")
X = df[["nlt_x", "nlt_y"]].values
y = df[["cmt_x", "cmt_y"]].values

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Deep MLP definition
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

# Alignment loss: combines MSE and cosine similarity
class AlignmentLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        cos_target = torch.ones(pred.size(0)).to(pred.device)
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.cos(pred, target, cos_target)

# Train with K-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
lowest_mse = float('inf')
all_fold_mse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
    X_val, y_val = X_scaled[val_idx], y_scaled[val_idx]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = DeepMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = AlignmentLoss()

    # Training
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        preds_val = model(torch.FloatTensor(X_val))
        mse = mean_squared_error(y_val, preds_val.numpy())
        all_fold_mse.append(mse)
        print(f"Validation MSE (Fold {fold+1}): {mse:.4f}")

        if mse < lowest_mse:
            lowest_mse = mse
            best_model = model

print("\n=== Training Completed ===")
print(f"Average MSE across folds: {np.mean(all_fold_mse):.4f}")
print(f"Best Fold MSE: {lowest_mse:.4f}")

# Save best model
torch.save(best_model.state_dict(), "mlp_embedding_aligner.pt")
print("Model saved as: mlp_embedding_2d_aligner.pt")

# Inference on full set
best_model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    preds_all = best_model(X_tensor).numpy()
    actual_all = y_scaled

# Inverse transform to original coordinates
preds_orig = scaler_y.inverse_transform(preds_all)
actual_orig = scaler_y.inverse_transform(actual_all)

# Show a few samples
print("\nSample Predictions vs Actuals:")
for i in range(5):
    print(f"Predicted: {preds_orig[i]}, Actual: {actual_orig[i]}")

# Plot arrows: predicted ? actual
plt.figure(figsize=(10, 8))
for pred, actual in zip(preds_orig, actual_orig):
    plt.arrow(pred[0], pred[1],
              actual[0] - pred[0], actual[1] - pred[1],
              head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
plt.xlabel("cmt_x")
plt.ylabel("cmt_y")
plt.title("Prediction vs Actual (Arrows: Predicted -> Actual)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("ontouml_aligned_embeddings_2d.png", dpi=300)
plt.close()
