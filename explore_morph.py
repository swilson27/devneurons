#%%
import numpy as np
import pandas as pd
from pathlib import Path
import pymaid
import navis
import json
from matplotlib import pyplot as plt
import os
import pickle

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output'


creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)



### Visualise skeletons ###



#%%
fpath = HERE / "transformed_paired.pickle"
with open(fpath, "rb") as f:
    l_trans, r_raw = pickle.load(f)

def visualise_transform(l_trans, r_raw, skids, plot_height = 800, plot_width = 1200):
    nrns = navis.NeuronList(l_trans) + navis.NeuronList(r_raw)
    skeletons = nrns.idx[skids]
    viewer = skeletons.plot3d(height = plot_height, width = plot_width)

#%%
visualise_transform(l_trans, r_raw, [skids_you_want])



### Plots for whole data ###



## Load output CSVs ##


#%%
manalysis_s0r100 = pd.read_csv(HERE / 'output'/'manalysis()_results100.csv')
manalysis_s1r100 = pd.read_csv(HERE / 'output'/'manalysis(1,)_results100.csv')
manalysis_s0r500 = pd.read_csv(HERE / 'output'/'manalysis()_results500.csv')
manalysis_s1r500 = pd.read_csv(HERE / 'output'/'manalysis(1,)_results500.csv')
manalysis_s0r1000 = pd.read_csv(HERE / 'output'/'manalysis()_results1000.csv')
manalysis_s1r1000 = pd.read_csv(HERE / 'output'/'manalysis(1,)_results1000.csv')
# integer after s refers to pruned strahler index (e.g. 0 for none, 1 for leaf nodes)
# integer after r refers to node resampling per N (100, 500 or 1000) units of cable


## Plots to assess various strahler-pruning and resampling combinations


def gen_xy(data, normalise = False):   
    """ Generate x and y values for plotting from morphology analysis data, comparing similarity across L-R pairs
        NBLAST values on x and # of L-R pairs on y

    Args:
        data: pandas df, with connectivity values stored in column 'mean_normalized_alpha_nblast'
        normalise (bool): option to normalise y (cumulative sum) values, if plotting for different amounts of values (x). Defaults to False.
    """     

    vals = list(data['mean_normalized_alpha_nblast'])
    vals = [np.nan if x is None else x for x in vals]
    x = sorted(v for v in vals if not np.isnan(v))
    y = np.arange(len(x))
    if normalise:
        y /= len(x)

    else:
        print('data not of correct format (json dict or pandas df)')
 
    return x, y


## Plots for whole data ##


#%%
# Strahler pruning = 0
fig = plt.figure()
ax = fig.add_subplot()

s0r100_x, s0r100_y= gen_xy(manalysis_s0r100)
ax.plot(s0r100_x, s0r100_y, label='S = 0, R = 100')

s0r500_x, s0r500_y = gen_xy(manalysis_s0r500)
ax.plot(s0r500_x, s0r500_y, label='S = 0, R = 500')

s0r1000_x, s0r1000_y= gen_xy(manalysis_s0r1000)
ax.plot(s0r1000_x, s0r1000_y, label='S = 0, R = 1000')


ax.legend()
ax.set_xlabel('NBLAST similarity value')
ax.set_label('Cumulative frequency')

fig.savefig('manalysis_cumulative_s0.pdf', format='pdf')

# Strahler pruning = 1
fig = plt.figure()
ax = fig.add_subplot()

s1r100_x, s1r100_y= gen_xy(manalysis_s1r100)
ax.plot(s1r100_x, s1r100_y, label='S = 1, R = 100')

s1r500_x, s1r500_y= gen_xy(manalysis_s1r500)
ax.plot(s1r500_x, s1r500_y, label='S = 1, R = 500')

s1r1000_x, s1r1000_y= gen_xy(manalysis_s1r1000)
ax.plot(s1r1000_x, s1r1000_y, label='S = 1, R = 1000')


ax.legend()
ax.set_xlabel('NBLAST similarity value')
ax.set_label('Cumulative frequency')

fig.savefig('manalysis_cumulative_s1.pdf', format='pdf')

# Strahler pruning = both
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(s0r100_x, s0r100_y, label='S = 0, R = 100')
ax.plot(s0r500_x, s0r500_y, label='S = 0, R = 500')
ax.plot(s0r1000_x, s0r1000_y, label='S = 0, R = 1000')

ax.plot(s1r100_x, s1r100_y, label='S = 1, R = 100')
ax.plot(s1r500_x, s1r500_y, label='S = 1, R = 500')
ax.plot(s1r1000_x, s1r1000_y, label='S = 1, R = 1000')

ax.legend()
ax.set_xlabel('NBLAST similarity value')
ax.set_label('Cumulative frequency')

fig.savefig('manalysis_cumulative_all.pdf', format='pdf')