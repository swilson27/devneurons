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
import matplotlib.pyplot as plt

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

## Define functions


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

    # if annotation == False:
    #     fpath = HERE / 'fake'/"neurons.pickle"
        
        # if not os.path.exists(fpath):
    bp = pd.read_csv(HERE / "brain-pairs.csv")
    # bp = bp[:1172]
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

    # neurons = tuple(l_neurons) + tuple(r_neurons)

    # with open(fpath, "wb") as f:
    #     pickle.dump(neurons, f, protocol=5)

    return l_neurons, r_neurons, duplicates


    #     else:
    #         with open(fpath, "rb") as f:
    #             neurons = pickle.load(f)
    #             return neurons


    # else:
    #     neurons = tuple(pymaid.get_neuron("annotation:sw;brainpair;" + side) for side in "LR")
    #     # tuple of CmNLs for left and right side respectively, called for both L and R at once
    #     return neurons


def get_landmarks(landmarks):
    """
    Generates landmark coordinates from downloaded CSV of L and R VNC hemispheres
    Returns:
        numpy ndarray: x, y, z coordinates of L and R VNC hemispheres
    """    
    if landmarks == 'brain':
        df = pd.read_csv(HERE / "brain_landmarks.csv", index_col=False, sep=", ")
    if landmarks == 'vnc':
        df = pd.read_csv(HERE / "VNC_landmarks.csv", index_col=False, sep=",")
    if landmarks == 'cns':
        df = pd.read_csv(HERE / "CNS_landmarks.csv", index_col=False, sep=",")
    
    counts = Counter(df["landmark_name"])
    l_xyz = []
    r_xyz = []
    for name, count in counts.items():
        if count != 2:
            continue
        left, right = df.loc[df["landmark_name"] == name][[" x", " y", " z"]].to_numpy()
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


def get_transformed_neurons(landmarks):
    """
    Obtains transformed (moving least squares) neurons, taking left pairs and applying L:R mirror flip via 'bhem' landmarks
    Also outputs right pairs, with no transformation applied
    Returns:
        two lists: first will be transformed left, second right; both as class: TN
        stores as pickle, if already generated it will simply load this
    """
    fpath = HERE / "transformed_paired.pickle"

    if not os.path.isfile(fpath):
        neurons_l, neurons_r, duplicates = get_neurons()

        # create zipped list of paired neuron names
        by_name_l = dict(zip(neurons_l.name, neurons_l))
        by_name_r = dict(zip(neurons_r.name, neurons_r))
        paired_names = list(zip(neurons_l.name, neurons_r.name))

        l_xyz, r_xyz = get_landmarks(landmarks)
        transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
        left_transform = []
        right_raw = []
        for l_name, r_name in paired_names:
            left_transform.append(transform_neuron(transform, by_name_l[l_name]))
            right_raw.append(by_name_r[r_name])
        
        with open(fpath, "wb") as f:
            pickle.dump((left_transform, right_raw), f, 5)
    else:
        with open(fpath, "rb") as f:
            left_transform, right_raw = pickle.load(f)

    return left_transform, right_raw #, duplicates



STRAHLER = (1,) # Strahler indices to prune, as follows (note must be as tuple): 
                # to_prune (int | list | range | slice) –
                # to_prune = 1 removes all leaf branches
                # = [1, 2] removes SI 1 and 2
                # = range(1, 4) removes SI 1, 2 and 3
                # = slice(1, -1) removes everything but the highest SI
                # = slice(-1, None) removes only the highest SI
                # = () No pruning

RESAMPLE = 500 # Integer value, nodes will be resampled to one per every N (RESAMPLE) units of cable

def make_dotprop(neuron, prune_strahler = STRAHLER, resample = RESAMPLE):
    """
    First prunes by specified strahler index and resamples neuron to given resolution
    Subsequently applies navis.make_dotprops to this neuron (k = 5, representing appropriate # of nearest neighbours [for tangent vector calculation] for the sparse point clouds of skeletons)
    Args:
        neuron (TN)
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000
    Returns:
        Dotprops of pruned & resampled neuron
    """    
    nrn = neuron.prune_by_strahler(list(prune_strahler))
    if not len(nrn.nodes):
        logger.warning('pruned %s to nothing', nrn.id)
        return None
    nrn.tags = {}
    nrn.resample(resample, inplace=True)
    return make_dotprops(nrn, 5)



def make_dps(neurons: navis.TreeNeuron, prune_strahler = STRAHLER, resample = RESAMPLE):
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

    fn = partial(make_dotprop, prune_strahler = prune_strahler, resample = resample)

    with ProcessPoolExecutor(os.cpu_count()) as p:
        out = [
            n
            for n in tqdm(
                p.map(fn, neurons, chunksize=50), "making dotprops", len(neurons)
            )
            
        ]

    return out


def get_dps(landmarks, prune_strahler = STRAHLER, resample = RESAMPLE):
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
    l_trans, r = get_transformed_neurons(landmarks)
    out = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r])
    filtered_left = []
    filtered_right = []
    for l, r in zip(*out):
        if l is not None and r is not None:
            filtered_left.append(l)
            filtered_right.append(r)

    return filtered_left, filtered_right #, dups    
    #     with open(fpath, "wb") as f:
    #         pickle.dump(out, f, 5)
    # else:
    #     with open(fpath, "rb") as f:
    #         out = pickle.load(f)


def duplicate_prepare(duplicates, prune_strahler = STRAHLER, resample = RESAMPLE):
    """ Takes the outputted duplicate skids, performs identical preparatory steps and returns dotprops in form ready for cross_validation()
    Args:
        duplicates (pandas DF): DF of the duplicate skids, outputted from original CSV
    Returns:
        paired: list of the zipped dotprops (left, right), ready for cross_validation 
    """  
    l_neurons = list(duplicates["leftid"])
    r_neurons = list(duplicates["rightid"])

    neurons_l = pymaid.get_neuron(l_neurons)
    neurons_r = pymaid.get_neuron(r_neurons)

    by_name_l = dict(zip(neurons_l.name, neurons_l))
    by_name_r = dict(zip(neurons_r.name, neurons_r))
    paired_names = list(zip(neurons_l.name, neurons_r.name))

    l_xyz, r_xyz = get_landmarks('cns')
    transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
    l_trans = []
    r_raw = []

    for l_name, r_name in paired_names:
        l_trans.append(transform_neuron(transform, by_name_l[l_name]))
        r_raw.append(by_name_r[r_name])
    
    left, right = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r_raw])
    paired = list(zip(left, right))

    return paired


T = TypeVar("T")


def split_training_testing(items: List[T], n_partitions=5, seed=DEFAULT_SEED
    ) -> Iterable[Tuple[List[T], List[T]]]:
    """ Splits items into training and testing sets (default n = 5) to generate custom score matrices ahead of cross-validated nblast
    Args:
        items: list of zipped L and R dotprops
        n_partitions (int): # of partitions for cross-validation, defaults to 5.
        seed (int): defaults to specified DEFAULT_SEED for reproducibility
    Yields:
        Iterator[Iterable[Tuple[List[T], List[T]]]]: iteratively yields training and testing sets for n_partitions
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
    # iteratively yields training and testing subsets of items for custom score matrices, based on n_partitions
    # each iteration will contain one of the partioned subsets as testing, with remaining partitions (n-1) used for training


def train_nblaster(dp_pairs: List[Tuple[Dotprops, Dotprops]], threads=8
    ) -> navis.nbl.nblast_funcs.NBlaster:
    """ Takes pairs of dotprops, constructs matching_lists (to build score_mat) from pairs, trains nblaster for each
    Args:
        dp_pairs (List[Tuple[Dotprops, Dotprops]]): Dotproducts from the L-R pairs
        threads (int): Defaults to 8.
    Returns:
        blaster : NBLAST algorithm (version 2) trained on input dp_pairs, ready to apply to testing set
    """    
    dps = []
    matching_lists = []
    for pair in dp_pairs:
        matching_pair = []
        for item in pair:
            matching_pair.append(len(dps))
            dps.append(item)
        matching_lists.append(matching_pair)
    # Iterates through pairs in dp_pairs, and both items in pair. Appends indices of matching_pairs and dotprops to separate lists 

    builder = LookupDistDotBuilder(
        dps, matching_lists, use_alpha=True, seed = DEFAULT_SEED
    ).with_bin_counts([21, 10])
    logger.info("Training...")
    score_mat = builder.build(threads)
    # build score matrix across # of threads
    logger.info("Trained")
    df = score_mat.to_dataframe()
    # print(df)
    # df.to_csv(HERE / "smat.csv")
    # Option to output score_mat as CSV

    blaster = navis.nbl.nblast_funcs.NBlaster(True, True, None)
    blaster.score_fn = score_mat

    return blaster


def cross_validation(
    dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions = 5, seed = DEFAULT_SEED
):
    headers = [
        "skeleton_id-transformed-L",
        "skeleton_id-raw-R",
        "partition",
        "mean_normalized_alpha_nblast",
        "name_L"
    ]
    dtypes = [int, int, int, float, str]
    rows = []
    for partition, (training, testing) in tqdm(
        enumerate(split_training_testing(dp_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        nblaster = train_nblaster(training, threads=None)
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = nblaster.append(left)
            r_idx = nblaster.append(right)
            result = nblaster.single_query_target(l_idx, r_idx, scores="mean")
            rows.append([left.id, right.id, partition, result, left.name])

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)

    df.sort_values(headers[-1], axis=0, inplace=True)

    return df


def cross_validation2(dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions=5, seed=DEFAULT_SEED):
    """ Takes zipped list of left and right dotprops, partitions data (training/testing) for cross-validated nblaster training and applies these functions to testing sets
        Scores are calculated as mean of transformed_left-to-right and right-to-transformed_left comparisons
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
        "name_left"
        ]
    dtypes = [int, int, int, float, str]
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
            rows.append([left.id, right.id, partition, result, left.name])
        # iteratively applies this across testing set, result output as mean of l to r & r to l scores

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)
    
    df.sort_values(headers[0], axis=0, inplace=True)
    df.sort_values(headers[3], axis=0, inplace=True)
    
    return df



### Run analysis ###



#%%
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        left, right = get_dps('cns')
        paired = list(zip(left, right))
        df = cross_validation(paired)
        print(df)
        # analysis for most pairs

        # paired2 = duplicate_prepare(dups)
        # df2 = cross_validation(paired2)
        # print(df2)
        # identical analysis done separately for duplicate skids, as CatmaidNeuron objects can't be repeated

        # concat_df = pd.concat([df,df2])
        df.to_csv(OUT_DIR / f"manalysis_test.csv", index=False)
        # merge both dataframes and export as CSV
