import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your combined dataset
df = pd.read_csv('combined_pokemon_data_updated.csv')

# 2. Ensure no missing type2
df['type2_analysis'] = df['type2_analysis'].fillna('')

# 3. Automatically detect stat columns (adjust thresholds if you like)
stats_to_find = {'hp_y', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'}
stat_cols = [c for c in df.columns 
             if c.lower().replace('.', '').replace(' ', '_') in stats_to_find]
print("Detected stat columns:", stat_cols)

# 4. Build a feature set for each Pokémon
def build_feature_set(row):
    feats = set()
    # Typing
    feats.add(row['type1_analysis'])
    if row['type2_analysis'] not in ['', 'No_type']:
        feats.add(row['type2_analysis'])
    # Abilities
    for ab in [row['ability1'], row['ability2'], row['hidden_ability']]:
        if pd.notna(ab) and ab not in ['No_ability', '']:
            feats.add(ab)
    # Moves (comma-separated)
    for mv in str(row['moveset']).split(','):
        mv = mv.strip()
        if mv:
            feats.add(mv)
    # Stats binned
    for stat in stat_cols:
        val = row[stat]
        if pd.isna(val):
            continue
        key = stat.upper().replace('_', '')
        if val >= 100:
            feats.add(f'High {key}')
        elif val >= 70:
            feats.add(f'Mid {key}')
        else:
            feats.add(f'Low {key}')
    # Usage tier flag (vgc_usage_percent)
    if 'vgc_usage_percent' in df.columns:
        u = row['vgc_usage_percent']
        if u >= 20:
            feats.add('HighUsage')
        elif u >= 5:
            feats.add('ModerateUsage')
        else:
            feats.add('LowUsage')
    return feats

# Build dict of feature sets
feature_sets = {r['name'].lower(): build_feature_set(r) for _, r in df.iterrows()}

# somewhere after you normalize column names and build feature_sets...

# 1) collect the set of all types:
all_types = set(df['type1_analysis'].dropna().unique()) \
          | set(df['type2_analysis'].dropna().unique())
all_types.discard('No_type')

# 2) pick your weights:
TYPE_WEIGHT = 0    # e.g. fire/grass/etc. only count 20%
OTHER_WEIGHT = 1.0   # all other features count 100%

# 3) define weighted‐Jaccard:
def weighted_jaccard(a, b):
    inter = a & b
    union = a | b

    num = 0.0
    for feat in inter:
        num += TYPE_WEIGHT if feat in all_types else OTHER_WEIGHT

    den = 0.0
    for feat in union:
        den += TYPE_WEIGHT if feat in all_types else OTHER_WEIGHT

    return num / den if den else 0.0



# 5. Jaccard similarity function
def jaccard(a, b):
    return len(a & b) / len(a | b)

# 6. Compute similarity to Incineroar
base = 'incineroar'
if base not in feature_sets:
    raise KeyError(f"'{base}' not found in dataset. Check casing or name column.")
base_feats = feature_sets[base]
scores = [(nm, jaccard(base_feats, feats)) 
          for nm, feats in feature_sets.items() if nm != base]
scores.sort(key=lambda x: x[1], reverse=True)




# 7. Filter top 10 with usage ≥ 5%
top_similar = []
for nm, sc in scores:
    usage = df.loc[df['name'].str.lower() == nm, 'vgc_usage_percent'].iloc[0]
    if usage >= 5:
        top_similar.append((nm.title(), sc))
    if len(top_similar) >= 10:
        break

# 8. Print results
print("\nTop 10 competitively-viable Pokémon similar to Incineroar:")
for nm, sc in top_similar:
    print(f"{nm:15s}  {sc:.3f}")

# 9. Visualize with a bar chart
names = [nm for nm, _ in top_similar]
values = [sc for _, sc in top_similar]

plt.figure(figsize=(10, 6))
plt.bar(names, values)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Jaccard Similarity')
plt.title('Top 10 Pokémon Similar to Incineroar (Usage ≥ 5%)')
plt.tight_layout()
plt.show()
