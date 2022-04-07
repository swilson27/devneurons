#%%
import navis
import pymaid
import pandas as pd

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)

#%%
scores = pd.read_csv('results2.csv')
scores_old = pd.read_csv('crossval_results_v1.csv')

vals = pymaid.get_names(list(scores['skeleton_id-transformed-L']))
scores["left_name"] = pd.Series(vals.values())

scores.to_csv(OUT_DIR / "connA_outputs.csv", index=False)

#%%
colnames = list(scores.columns)


scores.col

# %%
viewer = navis.plot3d(backend = 'vispy')
tr6 = [tr_L 6]

viewer.add(transformed_cmns[6])
viewer.add(nL[6])
viewer.add(nL[6])
