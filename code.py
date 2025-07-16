import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df_imputed)
    normalizer = MinMaxScaler()
    df_norm = normalizer.fit_transform(df_std)
    return pd.DataFrame(df_norm, columns=df.columns)

def balance_data(X, y):
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)
    rus = RandomUnderSampler()
    X_bal, y_bal = rus.fit_resample(X_res, y_res)
    return X_bal, y_bal

def create_knn_edge_index(X, k=2):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    src = []
    dst = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  
            src.append(i)
            dst.append(j)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

class ResGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, dropout=0.2)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        return h.mean(dim=1) if h.dim() > 2 else h

class AJW_ResGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=1):  
        super().__init__()
        self.gat1 = ResGATLayer(in_dim, hid_dim, heads)
        self.gat2 = ResGATLayer(hid_dim, hid_dim, heads)
        self.gat3 = ResGATLayer(hid_dim, hid_dim, heads)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = self.gat2(h, edge_index)
        h = self.gat3(h, edge_index)
        return self.fc(h)

def main():
    df = pd.read_csv(r"Enter\your\dataset\path\here\preprocess_data.csv")

    label_col = 'Label'
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X = preprocess_data(X)
    X_bal, y_bal = balance_data(X, y)

    X_bal = X_bal.sample(n=15000, random_state=42)
    y_bal = y_bal.loc[X_bal.index]

    X_tensor = torch.tensor(X_bal.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_bal.values, dtype=torch.long)

    edge_index = create_knn_edge_index(X_tensor.numpy(), k=2)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

    indices = np.arange(len(X_tensor))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y_tensor, random_state=42)
    train_mask = torch.zeros(len(X_tensor), dtype=torch.bool)
    test_mask = torch.zeros(len(X_tensor), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.test_mask = test_mask

    in_dim = X_tensor.shape[1]
    model = AJW_ResGAT(in_dim=in_dim, hid_dim=34, out_dim=len(torch.unique(y_tensor)), heads=1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = out[data.test_mask].argmax(dim=1)
                acc = (preds == data.y[data.test_mask]).float().mean()
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {acc.item():.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"\nFinal Evaluation Metrics:")
        print(f"Accuracy  : {accuracy:.7f}")
        print(f"Precision : {precision:.7f}")
        print(f"Recall    : {recall:.7f}")
        print(f"F1-Score  : {f1:.7f}")
        print("\nDetailed Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()