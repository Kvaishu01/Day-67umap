import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import umap

# ---------------------------
# 🎯 Streamlit App Title
# ---------------------------
st.title("🧭 UMAP – Customer Data Visualization")
st.write("""
Today’s experiment explores **UMAP (Uniform Manifold Approximation and Projection)** for visualizing 
high-dimensional customer data in 2D space.
""")

# ---------------------------
# 🧩 Generate synthetic customer dataset
# ---------------------------
st.subheader("🧮 Generating Synthetic Customer Dataset")

n_samples = st.slider("Select number of customers", 100, 2000, 500, 100)
X, y = make_blobs(
    n_samples=n_samples, 
    n_features=10, 
    centers=5, 
    cluster_std=2.0, 
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 11)])
df["segment"] = y

st.dataframe(df.head())

# ---------------------------
# ⚙️ Normalize and apply UMAP
# ---------------------------
st.subheader("⚙️ Applying UMAP for Dimensionality Reduction")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("segment", axis=1))

# Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_result = umap_model.fit_transform(X_scaled)

# ---------------------------
# 🎨 Visualization
# ---------------------------
st.subheader("📊 UMAP 2D Visualization of Customer Segments")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], c=y, cmap="tab10", s=40, alpha=0.8)
plt.colorbar(scatter, ax=ax, label="Customer Segment")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("Customer Segmentation using UMAP")

# ✅ Proper Streamlit-safe plotting call
st.pyplot(fig)

# ---------------------------
# 🔍 Summary
# ---------------------------
st.subheader("🧠 Insights")
st.write("""
- Each color represents a **customer segment** (cluster).  
- UMAP projects high-dimensional data (10 features) into **2D** while preserving both **local** and **global** structure.  
- This helps identify **similar groups of customers** for targeted marketing or personalization.
""")

st.success("✅ Visualization completed successfully!")
