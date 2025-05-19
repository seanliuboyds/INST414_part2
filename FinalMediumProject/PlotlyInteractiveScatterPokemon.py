
import chart_studio
import chart_studio.tools as tls
import chart_studio.plotly as py

import plotly.express as px
import plotly.io as pio
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsRegressor

# 1. Set your Chart Studio credentials
#    Replace with your own username and API key from https://chart-studio.plotly.com/settings/api
tls.set_credentials_file(username='Xatagarasu', api_key='S8cMrnMcM9tBxHDqvXhu')

# 2. Load and preprocess data (as before)
df = pd.read_csv('combined_pokemon_data_updated.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
df['vgc_usage_percent'] = df['vgc_usage_percent'].fillna(0.0)
stats_to_find = {'hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed'}
stat_cols = [c for c in df.columns if c in stats_to_find]

def build_feature_set(row):
    feats = set()
    feats.add(row['type1_analysis'])
    if row['type2_analysis'] not in ['', 'no_type']:
        feats.add(row['type2_analysis'])
    for col in ['ability1', 'ability2', 'hidden_ability']:
        ab = row[col]
        if pd.notna(ab) and ab not in ['no_ability', '']:
            feats.add(ab)
    for mv in str(row['moveset']).split(','):
        mv = mv.strip()
        if mv:
            feats.add(mv)
    for stat in stat_cols:
        val = row[stat]
        if pd.isna(val): continue
        key = stat.upper().replace('_','')
        if val >= 100: feats.add(f'high_{key}')
        elif val >= 70: feats.add(f'mid_{key}')
        else: feats.add(f'low_{key}')
    if 'vgc_usage_percent' in df.columns:
        u = row['vgc_usage_percent']
        if u >= 20: feats.add('high_usage')
        elif u >= 5: feats.add('moderate_usage')
        else: feats.add('low_usage')
    return feats

feature_sets = {r['name'].lower(): build_feature_set(r) for _, r in df.iterrows()}

def jaccard(a, b):
    return len(a & b) / len(a | b)

y = df['vgc_usage_percent'].tolist()
feature_dicts, names = [], []
for _, row in df.iterrows():
    d = {}
    t1, t2 = row['type1_analysis'], row['type2_analysis']
    d[f"type1_{t1}"], = (1,) if pd.notna(t1) else (0,)
    if t2 and t2 != 'no_type': d[f"type2_{t2}"] = 1
    for col in ['ability1','ability2','hidden_ability']:
        ab = row[col]
        if pd.notna(ab) and ab not in ['no_ability','']:
            d[f"ability_{ab}"] = 1
    for mv in str(row['moveset']).split(','):
        mv = mv.strip()
        if mv: d[f"move_{mv}"] = 1
    for stat in stat_cols:
        val = row[stat]
        if pd.notna(val): d[stat] = val
    feature_dicts.append(d)
    names.append(row['name'].lower())

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(feature_dicts)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)
top_pokemon = df.nlargest(5, 'vgc_usage_percent')['name'].str.lower().tolist()
neighbors = {}
for pokemon in top_pokemon:
    idx = names.index(pokemon)
    distances, indices = knn.kneighbors([X[idx]], n_neighbors=6)
    neighs = [names[i] for i in indices[0] if names[i] != pokemon][:5]
    neighbors[pokemon.title()] = [n.title() for n in neighs]

# 3. Compute 3D PCA
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X)

# 4. Build plotting DataFrame
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

# 5. Create Plotly figure
fig = px.scatter_3d(
    df_plot, x='PC1', y='PC2', z='PC3',
    color='Group', hover_name='Name',
    title='3D PCA of Pokémon Features with KNN Neighbors'
)

# 6. Upload to Chart Studio
fig.write_html("my_plot.html")
