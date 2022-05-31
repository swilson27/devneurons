#%%
import navis
import pymaid
import pandas as pd
from pathlib import Path
import os
import json
import pickle

HERE = Path(__file__).resolve().parent
creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)

## Modify output dfs from v1 and v2 analyses ##
#%%
scores = pd.read_csv(HERE / 'output'/'manalysis(1,)_results.csv')
scores = scores.drop('partition', 1)
scores = scores.rename(columns={"mean_normalized_alpha_nblast": "normalized_nblast", "skeleton_id-transformed-L":"skid_transformed-L", "skeleton_id-raw-R":"skid_raw-R"})
scores = scores.sort_values('skid_transformed-L')
scores = scores.reset_index(drop = True)

# Get names of all left_skids and assign to new column
vals = pymaid.get_names(list(scores['skid_transformed-L']))
vals = list(vals.values())
# CSV contains 3 duplicates of left skid, owing to developmental duplications and different right pairs. Must insert these names to compensate for order
vals.insert(253, 'AVL011 PN Right 2?')
vals.insert(514, 'OLP4;right')
# Also just 2 complete duplicates, remove these

scores["left_name"] = pd.Series(vals)
scores.sort_values(by = ['normalized_nblast'], axis=0, inplace=True)

scores.to_csv("MAnalysis_results.csv", index=False)

#%%
scores_old = pd.read_csv('crossval_results_v1.csv')
scores_old = scores_old.drop('partition', 1)
scores_old = scores_old.rename(columns={"mean_normalized_alpha_nblast": "normalized_nblast", "skeleton_id-transformed-L": "skid_transformed-L", "skeleton_id-raw-R": "skid_raw-R"})

scores_old.sort_values(by = ['skid_transformed-L'], axis=0, inplace=True)
vals = pymaid.get_names(list(scores_old['skid_transformed-L']))
scores_old["left_name"] = vals.values()
scores_old.sort_values(['normalized_nblast'], axis=0, inplace=True)
    
scores_old.to_csv("MAnalysis_results_v1.csv", index=False)


## Visualise treeneurons from analysis ##


# pre-pruning, coloured by Strahler index:
#%%
nL = pymaid.get_neuron("bridge-like left")
nR = pymaid.get_neuron("bridge-like right")
fig = navis.plot3d([nL, nR], color_by='strahler_index', palette='viridis', backend='plotly')

# post-pruning
# %%
PS =(-1, None) #PS = prune_strahler arguments
resample=1000
fpath = (
        HERE / f"dotprops_p{''.join(str(p) for p in PS)}_r{resample}.pickle"
    )
with open(fpath, "rb") as f:
            dotprops = pickle.load(f)

#%%
nL = pymaid.get_neuron("bridge-like left")
nR = pymaid.get_neuron("bridge-like right")
fig = navis.plot3d([nL.prune_by_strahler(slice(-1, None)), nR.prune_by_strahler(slice(-1, None))], color_by='strahler_index', palette='viridis', backend='plotly')


# %%
