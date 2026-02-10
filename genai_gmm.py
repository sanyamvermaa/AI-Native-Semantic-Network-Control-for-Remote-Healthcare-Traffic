import pandas as pd
from sklearn.mixture import GaussianMixture

# 1. Load the collected dataset
data = pd.read_csv("receiver_log.csv")

# 2. Keep only numeric columns for GenAI
X = data[["heart_rate", "delay"]].values

# 3. Train a Generative Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)

# 4. Generate synthetic data
synthetic_data, _ = gmm.sample(500)

# 5. Save generated dataset
gen_df = pd.DataFrame(
    synthetic_data,
    columns=["heart_rate", "delay"]
)

# Remove invalid samples
gen_df = gen_df[gen_df["delay"] >= 0]

# Optional: cap extremely large delays
gen_df = gen_df[gen_df["delay"] < 0.1]


gen_df.to_csv("generated_synthetic_telemetry.csv", index=False)

print("Synthetic dataset generated successfully")
