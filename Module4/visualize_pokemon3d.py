import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load and Prepare the Dataset
data = pd.read_csv("merged_pokemon_data.csv")

# Define numerical stat features
stat_features = ["attack", "defense", "sp_attack", "sp_defense", "speed", "hp_y", "base_total"]

# Define move categories
offensive_moves = [
    "Fire Blast", "Flare Blitz", "Hydro Pump", "Thunderbolt", "Earthquake", "Shadow Ball",
    "Draco Meteor", "Close Combat", "Leaf Blade", "Ice Beam", "Surf", "Flamethrower",
    "Overheat", "Sludge Bomb", "Hurricane", "Dragon Claw", "Flash Cannon", "Dark Pulse"
]
defensive_moves = [
    "Protect", "Recover", "Stealth Rock", "Toxic", "Wish", "Synthesis", "Roost", "Haze",
    "Rapid Spin", "Defog", "Substitute", "Soft-Boiled", "Heal Bell", "Toxic Spikes",
    "Spikes", "Slack Off", "Shore Up"
]
status_moves = [
    "Thunder Wave", "Taunt", "Will-O-Wisp", "Knock Off", "Parting Shot", "Teleport",
    "Sticky Web", "Spore", "Glare", "Destiny Bond", "Encore", "Disable"
]
setup_moves = [
    "Swords Dance", "Calm Mind", "Nasty Plot", "Dragon Dance", "Quiver Dance", "Bulk Up",
    "Coil", "Shift Gear", "Autotomize", "Shell Smash", "Curse"
]

priority_moves = [
    "Bullet Punch", "Quick Attack", "Aqua Jet", "Mach Punch", "Extreme Speed", "Sucker Punch",
    "Ice Shard", "Shadow Sneak", "Accelerock", "Vacuum Wave", "Feint"
]

# Function to check if moveset contains a move from a category
def has_move(moveset, move_list):
    if isinstance(moveset, str):
        moves = moveset.strip("[]").replace("'", "").split(", ")
        return any(move in move_list for move in moves)
    return False

# Create binary moveset features
data["has_offensive_move"] = data["moveset"].apply(lambda x: 1 if has_move(x, offensive_moves) else 0)
data["has_defensive_move"] = data["moveset"].apply(lambda x: 1 if has_move(x, defensive_moves) else 0)
data["has_status_move"] = data["moveset"].apply(lambda x: 1 if has_move(x, status_moves) else 0)
data["has_setup_move"] = data["moveset"].apply(lambda x: 1 if has_move(x, setup_moves) else 0)
data["has_priority_move"] = data["moveset"].apply(lambda x: 1 if has_move(x, priority_moves) else 0)

# Combine features for clustering
features = stat_features + [
    "has_offensive_move", "has_defensive_move", "has_status_move",
    "has_setup_move", "has_priority_move"
]
X = data[features]

# Handle missing values (if any, though dataset appears complete)
X = X.fillna(X.mean())

# Step 2: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data["cluster"] = kmeans.fit_predict(X_scaled)

# Step 4: Apply PCA for Visualization (3 components)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]
data["PC3"] = X_pca[:, 2]

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: PC1 = {explained_variance[0]:.2f}, PC2 = {explained_variance[1]:.2f}, "
      f"PC3 = {explained_variance[2]:.2f}, Total = {sum(explained_variance):.2f}")

# Step 5: Create 3D Scatter Plot with Matplotlib
# Map cluster numbers to descriptive labels and colors
cluster_labels = {
    0: "Bulky Attacker",
    1: "Sweeper",
    2: "Defensive Tank",
    3: "Utility"
}
colors = ["red", "blue", "green", "purple"]
data["cluster_label"] = data["cluster"].map(cluster_labels)

# Initialize 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot each cluster
for cluster in range(4):
    cluster_data = data[data["cluster"] == cluster]
    ax.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        cluster_data["PC3"],
        c=colors[cluster],
        label=cluster_labels[cluster],
        alpha=0.6,
        s=50
    )

# Customize plot
ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
ax.set_zlabel(f"PC3 ({explained_variance[2]:.2%} variance)")
ax.set_title("Pokémon Clustering (K-means, k=4) with Stats and Expanded Movesets")
ax.legend(title="Cluster")
ax.grid(True, linestyle="--", alpha=0.7)

# Save and display plot
plt.savefig("pokemon_clustering_scatter_3d.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 6: Summarize Clusters
print("\nCluster Summaries:")
for cluster in range(4):
    cluster_data = data[data["cluster"] == cluster]
    avg_stats = cluster_data[stat_features].mean().round(2)
    move_features = ["has_offensive_move", "has_defensive_move", "has_status_move",
                    "has_setup_move", "has_priority_move"]
    move_prevalence = cluster_data[move_features].mean().round(2)
    common_moves = cluster_data["moveset"].str.strip("[]").str.replace("'", "").str.split(", ", expand=True).stack().value_counts().head(5)
    print(f"\nCluster {cluster} ({cluster_labels[cluster]}):")
    print(f"  Number of Pokémon: {len(cluster_data)}")
    print(f"  Average Stats: {avg_stats.to_dict()}")
    print(f"  Move Category Prevalence: {move_prevalence.to_dict()}")
    print(f"  Top 5 Common Moves: {common_moves.index.tolist()}")
    print(f"  Example Pokémon: {cluster_data['name'].head(3).tolist()}")