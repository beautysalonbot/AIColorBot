import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("recipes.csv")
X = df[["L", "a", "b"]].values
knn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)

with open("latest_lab.txt") as f:
    lab_values = np.fromstring(f.read().strip(), sep=",")
print("検索する LAB:", lab_values)

distances, indices = knn.kneighbors([lab_values])

print("\n--- 🔍 類似レシピ TOP3 ---")
for rank, (d, idx) in enumerate(zip(distances[0], indices[0]), 1):
    row = df.iloc[idx]
    print(f"{rank}. {row['name']}  (距離: {d:.1f})")
    print(f"   処方: {row['formula']}\n")
