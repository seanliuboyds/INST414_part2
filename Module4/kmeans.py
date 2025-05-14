import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('merged_pokemon_data.csv')

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

# Define features for clustering
stat_features = ['attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'hp_y', 'base_total']
features = stat_features + [
    'has_offensive_move', 'has_defensive_move', 'has_status_move',
    'has_setup_move', 'has_priority_move'
]

# Extract features and handle missing values
X = data[features].fillna(0)  # Replace NaN with 0 for simplicity

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute SSE for elbow method
sse = []
k_range = range(2, 21)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)


# Choose optimal k (hypothesized as 8, subject to plot inspection)
optimal_k = 4

# Plot elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# Perform final clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered dataset
data.to_csv('clustered_pokemon_data.csv', index=False)

print(f"Clustering completed with k={optimal_k}. Plots saved as 'elbow_plot.png' and 'silhouette_plot.png'.")