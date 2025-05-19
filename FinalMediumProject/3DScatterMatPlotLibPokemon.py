import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load and preprocess data
df = pd.read_csv('combined_pokemon_data_updated.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
df['vgc_usage_percent'] = df['vgc_usage_percent'].fillna(0.0)

stats_to_find = {'hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed'}
stat_cols = [c for c in df.columns if c in stats_to_find]

# Build feature dictionaries
feature_dicts, names = [], []
y = df['vgc_usage_percent'].tolist()
for _, row in df.iterrows():
    d = {}
    t1, t2 = row['type1_analysis'], row['type2_analysis']
    if pd.notna(t1):
        d[f"type1_{t1}"] = 1
    if pd.notna(t2) and t2 != 'no_type':
        d[f"type2_{t2}"] = 1
    for col in ['ability1', 'ability2', 'hidden_ability']:
        ab = row[col]
        if pd.notna(ab) and ab not in ['no_ability', '']:
            d[f"ability_{ab}"] = 1
    for mv in str(row['moveset']).split(','):
        mv = mv.strip()
        if mv:
            d[f"move_{mv}"] = 1
    for stat in stat_cols:
        val = row[stat]
        if pd.notna(val):
            d[stat] = val
    feature_dicts.append(d)
    names.append(row['name'].lower())

# Vectorize and fit KNN
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(feature_dicts)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X, y)

# Find top usage Pokémon and their neighbors
top_pokemon = df.nlargest(10, 'vgc_usage_percent')['name'].str.lower().tolist()
neighbors = {}
for pokemon in top_pokemon:
    idx = names.index(pokemon)
    distances, indices = knn.kneighbors([X[idx]], n_neighbors=6)
    neighs = [names[i] for i in indices[0] if names[i] != pokemon][:5]
    neighbors[pokemon.title()] = [n.title() for n in neighs]

# PCA to 3D
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X)

# Prepare plotting DataFrame
plot_data = []
for idx, nm in enumerate(names):
    x, y0, z = X_pca3[idx]
    group = 'Background'
    for base, neighs in neighbors.items():
        if nm == base.lower():
            group = f"{base} (base)"
        elif nm in [n.lower() for n in neighs]:
            group = base
    plot_data.append({'Name': nm.title(), 'PC1': x, 'PC2': y0, 'PC3': z, 'Group': group})
df_plot = pd.DataFrame(plot_data)

# Plot with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

groups = df_plot['Group'].unique()
for group in groups:
    if group == 'Background':
        continue
    sub = df_plot[df_plot['Group'] == group]
    ax.scatter(sub['PC1'], sub['PC2'], sub['PC3'], label=group, s=20)

# Draw connecting arrows
for base, neighs in neighbors.items():
    bi = names.index(base.lower())
    x0, y0, z0 = X_pca3[bi]
    for neigh in neighs:
        ni = names.index(neigh.lower())
        x1, y1, z1 = X_pca3[ni]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        ax.plot([x0, x1], [y0, y1], [z0, z1], linewidth=0.8)
        ax.quiver(x0, y0, z0, dx, dy, dz, arrow_length_ratio=0.1, linewidth=0.8)
        



ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.subplots_adjust(bottom=0.2)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),  # centered, below the axes
    ncol=3                       # spread entries over 3 columns
)

plt.title('3D PCA of Pokémon Features with KNN Neighbors (matplotlib)')
plt.show()

print("Top 10 Most Used Pokémon → 5 Nearest Neighbors (with Formats)\n")
for base, neighs in neighbors.items():
    # lookup base Pokémon’s format
    base_fmt = df.loc[
        df['name'].str.lower() == base.lower(), 
        'formats'
    ].iat[0]
    print(f"{base} (Format: {base_fmt}) → Neighbors:")
    
    # print each neighbor + its format
    for n in neighs:
        n_fmt = df.loc[
            df['name'].str.lower() == n.lower(), 
            'formats'
        ].iat[0]
        print(f"  • {n} (Format: {n_fmt})")
    print()  # blank line between blocks

