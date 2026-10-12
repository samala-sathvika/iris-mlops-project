import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

# Load the Iris dataset
iris = load_iris(as_frame=True)

# Add target and species columns
df = iris.data.copy()
df["target"] = iris.target
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)

# Create data folder and save CSV
Path("data").mkdir(exist_ok=True)
df.to_csv("data/iris.csv", index=False)

print("Iris dataset created successfully!")
print("Dataset shape:", df.shape)
print(df.head())