# %%
import logging
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
import json

import navis
import numpy as np
import pandas as pd
import pymaid
from navis.core.core_utils import make_dotprops
from navis.nbl.smat import LookupDistDotBuilder
from pymaid.core import Dotprops
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.info("Using navis version %s", navis.__version__)
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"
# Logging tracker of events/errors; returns parent directory of given path, provides cache and output directories

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials

DEFAULT_SEED = 1991

pymaid.clear_cache

## Define functions ##


#%%
def get_neurons(annotation = False):
    """
    Get a respective CatmaidNeuronList for left and right pairs. Lists are each in arbitrary order.
    if annotation = False: load neurons from downloaded csv (brain_pairs)
        N.B. there are 2 L neurons duplicated, which are here removed for subsequent separate analysis
    if annotation = True: load neurons from catmaid annotation (subset of brain_pairs), lacks duplicates

    neurons stored as pickle object, will be loaded if in cache folder. N.B. cached neurons must be deleted if you wish to analyse a different subset

    Returns:
        {l/r}_neurons: tuple of CmNLs for left and right side respectively
        duplicates: extracted (last) left_skid duplicates (2 in brain_pairs.csv), for separate analysis (has to be unique)
    """    
    # fpath = HERE / "neurons.pickle"
    # if not os.path.exists(fpath):

    if annotation == False:
        bp = pd.read_csv(HERE / "brain-pairs.csv")
        bp.drop('region', axis=1, inplace=True)            
        has_duplicate = bp.duplicated(subset=['leftid'])
        duplicates = bp[has_duplicate]
        bp = bp.drop_duplicates(subset='leftid')
        # left id has 2 neurons (4985759 and 8700125) with duplicated pairs, owing to developmental phenomenon on right side
        # these deviant pairs are filtered out (as CmNs must be unique), with subsequent analysis separately applied on them at the end and results appended 

        l_neurons = list(bp["leftid"])
        r_neurons = list(bp["rightid"])

        l_neurons = pymaid.get_neuron(l_neurons)
        r_neurons = pymaid.get_neuron(r_neurons)

        # for nrn in list(bp["leftid"]):
        #     l_neurons.append(pymaid.get_neuron(nrn))

        # for nrn in list(bp["rightid"]):
        #     r_neurons.append(pymaid.get_neuron(nrn))

        return l_neurons, r_neurons, duplicates

    else:
        neurons = tuple(pymaid.get_neuron("annotation:sw;brainpair;" + side) for side in "LR")
        # tuple of CmNLs for left and right side respectively, called for both L and R at once
        return neurons

    #         with open(fpath, "wb") as f:
    #             pickle.dump(neurons, f, protocol=5)
    # else:
    #     with open(fpath, "rb") as f:
    #         neurons = pickle.load(f)
    #         return neurons


# def name_pair(left_names, right_names):
#     """
#     Replaces "left"/"right" (across pairs - left_names and right_names) with placeholder, to pair up neurons

#     Args:
#         left_names (list of str): name:CmN, presumed to contain "left" pairs
#         right_names (list of str): name:CmN, presumed to contain "right pairs

#     Yields:
#         list of tuples
#     """    
#     unsided_l = {n.replace("left", "__SIDE__"): n for n in left_names}
#     unsided_r = {n.replace("right", "__SIDE__"): n for n in right_names}
#     in_both = set(unsided_l).intersection(unsided_r)
#     for unsided_name in sorted(in_both):
#         yield (unsided_l[unsided_name], unsided_r[unsided_name])


def get_landmarks():
    """
    Generates landmark coordinates from downloaded CSV of L and R brain hemispheres

    Returns:
        numpy ndarray: x, y, z coordinates of L and R brain hemispheres
    """    
    df = pd.read_csv(HERE / "bhem.csv", index_col=False, sep=", ")
    counts = Counter(df["landmark_name"])
    l_xyz = []
    r_xyz = []
    for name, count in counts.items():
        if count != 2:
            continue
        left, right = df.loc[df["landmark_name"] == name][["x", "y", "z"]].to_numpy()
        if left[0] < right[0]:
            left, right = right, left

        l_xyz.append(left)
        r_xyz.append(right)

    return np.asarray(l_xyz), np.asarray(r_xyz)


def transform_neuron(tr: navis.transforms.base.BaseTransform, nrn: navis.TreeNeuron):
    """
    Applies selected transformation (tr, type: BT) to neuron (nrn, type: TN)

    Args:
        tr (navis.transforms.base.BaseTransform): _description_
        nrn (navis.TreeNeuron): _description_

    Returns:
        TreeNeuron: Transformed TN/s
    """    
    nrn = nrn.copy(True)
    dims = ["x", "y", "z"]
    nrn.nodes[dims] = tr.xform(nrn.nodes[dims])
    if nrn.connectors is not None:
        nrn.connectors[dims] = tr.xform(nrn.connectors[dims])
    return nrn


def get_transformed_neurons():
    """
    Obtains transformed (moving least squares) neurons, taking left pairs and applying L:R mirror flip via 'bhem' landmarks
    Also outputs right pairs, with no transformation applied

    Returns:
        two lists: first will be transformed left, second right; both as class: TN

        stores as pickle, if already generated it will simply load this
    """
    # fpath = HERE / "transformed_paired.pickle"

    # if not os.path.isfile(fpath):
    neurons_l, neurons_r, duplicates = get_neurons()
    by_name_l = dict(zip(neurons_l.name, neurons_l))
    by_name_r = dict(zip(neurons_r.name, neurons_r))
    paired_names = list(zip(neurons_l.name, neurons_r.name))

    l_xyz, r_xyz = get_landmarks()
    transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
    left_transform = []
    right_raw = []
    for l_name, r_name in paired_names:
        left_transform.append(transform_neuron(transform, by_name_l[l_name]))
        right_raw.append(by_name_r[r_name])

    #     with open(fpath, "wb") as f:
    #         pickle.dump((left_transform, right_raw), f, 5)
    # else:
    #     with open(fpath, "rb") as f:
    #         out = pickle.load(f)

    return left_transform, right_raw, duplicates

STRAHLER = (1,) # Strahler indices to prune, as follows: 
                # to_prune (int | list | range | slice) –
                # to_prune=1 removes all leaf branches
                # to_prune=[1, 2] removes SI 1 and 2
                # to_prune=range(1, 4) removes SI 1, 2 and 3
                # to_prune=slice(1, -1) removes everything but the highest SI
                # to_prune=slice(-1, None) removes only the highest SI



def make_dotprop(neuron, prune_strahler = STRAHLER, resample=1000):
    """
    First prunes by strahler index (default = 1) and resamples neuron to given resolution (1000 nodes per every N units of cable?)
    Subsequently applies navis.make_dotprops to this neuron (k = 5, representing appropriate # of nearest neighbours [for tangent vector calculation] for the sparse point clouds of skeletons)

    Args:
        neuron (TN)
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000

    Returns:
        Dotprops of pruned & resampled neuron
    """    
    nrn = neuron.prune_by_strahler(list(prune_strahler))
    nrn.tags = {}
    nrn.resample(resample, inplace=True)
    return make_dotprops(nrn, 5)



def make_dps(neurons: navis.TreeNeuron, prune_strahler = STRAHLER, resample=1000):
    """
    Applies make_dotprop to list of TreeNeurons, utilising multiprocessing to speed up

    Args:
        neurons (TN)
        prune_strahler
        resample (int): defaults to 1000

    Returns:
        Dotprops of pruned & resampled neurons
    """    

    out = []

    fn = partial(make_dotprop, prune_strahler= prune_strahler, resample=resample)

    with ProcessPoolExecutor(os.cpu_count()) as p:
        out = [
            n
            for n in tqdm(
                p.map(fn, neurons, chunksize=50), "making dotprops", len(neurons)
            )
        ]

    return out


def get_dps(prune_strahler = STRAHLER, resample=1000):
    """
    Obtains left and right pairs from prior functions.
    Transforms left pairs, makes dot products for both these and right pairs and outputs. Loaded from pickle if already ran

    Args:
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000

    Returns:
        list: dotproducts for l_trans and r
    """    
    # fpath = (
    #     HERE / f"dotprops_p{''.join(str(p) for p in prune_strahler)}_r{resample}.pickle"
    # )
    # if not fpath.is_file():
    l_trans, r, dups = get_transformed_neurons()
    out = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r])    
    #     with open(fpath, "wb") as f:
    #         pickle.dump(out, f, 5)
    # else:
    #     with open(fpath, "rb") as f:
    #         out = pickle.load(f)
    return out, dups


def train_nblast(transformed_l: List[navis.Dotprops], right: List[navis.Dotprops]):
    # Unused?   
    matching_lists = [
        [idx, idx + len(transformed_l)] for idx in range(len(transformed_l))
    ]
    dps = transformed_l + right

    builder = LookupDistDotBuilder(
        dps, matching_lists, use_alpha=True, seed=1991
    ).with_bin_counts([21, 10])
    score_mat = builder.build(8)
    df = score_mat.to_dataframe()
    df.to_csv(HERE / "smat.csv")

    return score_mat

# def duplicate_prepare(duplicates):


T = TypeVar("T")

def split_training_testing(items: List[T], n_partitions=5, seed=DEFAULT_SEED
    ) -> Iterable[Tuple[List[T], List[T]]]:
    """ Splits items into training and testing sets (default n = 5) ahead of cross-validation

    Args:
        items: list of zipped L and R dotprops
        n_partitions (int): # of partitions for cross-validation, defaults to 5.
        seed (int): defaults to specified DEFAULT_SEED for reproducibility

    Returns:
        [Tuple[List[T], List[T]]]: iteratively yields training and testing sets for n_partitions

    Yields:
        Iterator[Iterable[Tuple[List[T], List[T]]]]: _description_
    """
    items_arr = np.array(items, dtype=object)
    # Converts items (zipped list of L and R dotprops) into np.ndarray
    partition_idxs = np.arange(len(items_arr)) % n_partitions
    # creates partition indexes as modulus of # of neurons and # of partitions
    rng = np.random.default_rng(seed)
    rng.shuffle(partition_idxs)
    # randomly generates and shuffles partition indexes, based on seed for reproducibility
    for idx in range(n_partitions):
        training = list(items_arr[partition_idxs != idx])
        testing = list(items_arr[partition_idxs == idx])
        yield training, testing
    # iteratively yields training and testing subsets of items, based on n_partitions
    # each iteration will contain one of the partioned subsets as testing, with remaining partitions (n-1) used for training


def train_nblaster(dp_pairs: List[Tuple[Dotprops, Dotprops]], threads=8
    ) -> navis.nbl.nblast_funcs.NBlaster:
    """ Takes 

    Args:
        dp_pairs (List[Tuple[Dotprops, Dotprops]]): _description_
        threads (int): Defaults to 8.

    Returns:
        blaster : _description_
    """    
    dps = []
    matching_lists = []
    for pair in dp_pairs:
        matching_pair = []
        for item in pair:
            matching_pair.append(len(dps))
            dps.append(item)
        matching_lists.append(matching_pair)

    builder = LookupDistDotBuilder(
        dps, matching_lists, use_alpha=True, seed=1991
    ).with_bin_counts([21, 10])
    logger.info("Training...")
    score_mat = builder.build(threads)
    # build score matrix across # of threads
    logger.info("Trained")
    df = score_mat.to_dataframe()
    # print(df)
    df.to_csv(HERE / "smat.csv")
    # Return as 

    blaster = navis.nbl.nblast_funcs.NBlaster(True, True, None)
    blaster.score_fn = score_mat

    return blaster


def cross_validation(dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions=5, seed=DEFAULT_SEED):
    """ Takes zipped list of left and right dotprops, partitions data (training/testing) for cross-validation and applies nblaster function
        Scores are calculated as mean of left-to-right and right-to-left comparisons

    Args:
        dp_pairs (List[Tuple[Dotprops, Dotprops]]): _description_
        n_partitions (int, optional): _description_. Defaults to 5.
        seed (_type_, optional): _description_. Defaults to DEFAULT_SEED.

    Returns:
        _type_: _description_
    """
    headers = [
        "skid_transformed-L",
        "skid_raw-R",
        "partition",
        "mean_normalized_alpha_nblast"
        #"name_left"
        ]
    dtypes = [int, int, int, float]
    rows = []
    for partition, (training, testing) in tqdm(
        enumerate(split_training_testing(dp_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        nblaster = train_nblaster(training, threads=None)
        # creates nblast algorithm from training set
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = nblaster.append(left)
            r_idx = nblaster.append(right)
            result = nblaster.single_query_target(l_idx, r_idx, scores="mean")
            #name = pymaid.get_names(left)
            rows.append([left.id, right.id, partition, result]) #, name])
        # iteratively applies this across testing set, result output as mean of l to r & r to l scores

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)

    
    df.sort_values(headers[0], axis=0, inplace=True)
    vals = pymaid.get_names(list(df['skid_transformed-L']))
    df["left_name"] = vals.values()
    df.sort_values(headers[3], axis=0, inplace=True)
    
    return df


## Run analysis ##


#%%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        left, right, dups = get_dps()
        paired = list(zip(left, right))
        df = cross_validation(paired)
        print(df)

        # duplicate_prepare(dups)
        # df2 = cross_validation(dups)
        # print(df2)

        # merge
        df.to_csv(OUT_DIR / f"manalysis{STRAHLER}_results.csv", index=False)
