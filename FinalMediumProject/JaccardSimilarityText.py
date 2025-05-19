import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your combined dataset
df = pd.read_csv('combined_pokemon_data_updated.csv')

# 2. Ensure no missing type2
df['type2_analysis'] = df['type2_analysis'].fillna('')

stats_to_find = {'hp_y', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'}
stat_cols = [c for c in df.columns 
             if c.lower().replace('.', '').replace(' ', '_') in stats_to_find]

# 4. Build a feature set for each Pokémon
def build_feature_set(row):
    feats = set()
    feats.add(row['type1_analysis'])
    if row['type2_analysis'] not in ['', 'No_type']:
        feats.add(row['type2_analysis'])
    for ab in [row['ability1'], row['ability2'], row['hidden_ability']]:
        if pd.notna(ab) and ab not in ['No_ability', '']:
            feats.add(ab)
    for mv in str(row['moveset']).split(','):
        mv = mv.strip()
        if mv:
            feats.add(mv)
    # you need stat_cols defined elsewhere for bins
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
    if 'vgc_usage_percent' in df.columns:
        u = row['vgc_usage_percent']
        if u >= 20:
            feats.add('HighUsage')
        elif u >= 5:
            feats.add('ModerateUsage')
        else:
            feats.add('LowUsage')
    return feats

feature_sets = {
    r['name'].lower(): build_feature_set(r)
    for _, r in df.iterrows()
}

# 5. Jaccard similarity
def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

# 6. Find top 5 most used Pokémon
top_bases = df.nlargest(5, 'vgc_usage_percent')['name'].str.lower().tolist()

# 7. For each base, compute top 10 similar and both print & plot
for base in top_bases:
    base_feats = feature_sets[base]
    scores = [
        (nm, jaccard(base_feats, feats))
        for nm, feats in feature_sets.items()
        if nm != base
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    top10 = scores[:10]

    # Print out as statements
    print(f"\nTop 10 Pokémon similar to {base.title()}:")
    for nm, sc in top10:
        fmt = df.loc[df['name'].str.lower() == nm, 'formats'].iloc[0]
        print(f"  - {nm.title()} ({fmt}): Jaccard similarity = {sc:.3f}")

    # Then visualize
    labels = [f"{nm.title()} ({df.loc[df['name'].str.lower()==nm,'formats'].iloc[0]})"
              for nm, _ in top10]
    values = [sc for _, sc in top10]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Jaccard Similarity')
    plt.title(f'Top 10 Pokémon Similar to {base.title()}')
    plt.tight_layout()
    plt.show()   # next chart appears after you close
