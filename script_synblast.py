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
def get_neurons():
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

    bp = bp.drop(1532)
    # neuron 15609427 has no connectors, so pair filtered out

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
    # fpath = HERE / "transformed_paired.pickle"

    # if not os.path.isfile(fpath):
    neurons_l, neurons_r, duplicates = get_neurons()

    # create zipped list of paired neuron names
    by_name_l = dict(zip(neurons_l.name, neurons_l))
    by_name_r = dict(zip(neurons_r.name, neurons_r))
    paired_names = list(zip(neurons_l.name, neurons_r.name))

    l_xyz, r_xyz = get_landmarks(landmarks)
    transform = navis.transforms.MovingLeastSquaresTransform(l_xyz, r_xyz)
    left = []
    right = []
    for l_name, r_name in paired_names:
        left.append(transform_neuron(transform, by_name_l[l_name]))
        right.append(by_name_r[r_name])
    
    paired = list(zip(left, right))

    return paired


def make_dotprop(neuron, prune_strahler, resample):
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


def make_connector_dotprop(neuron):
    points = neuron.connectors[["x", "y", "z"]]
    vect = np.zeros((len(points), 3))
    alpha = np.ones(len(points))
    dp = Dotprops(points, None, vect, alpha)
    dp.kdtree  # force creation of the spatial lookup structure here
    dp.id = neuron.id
    dp.name = neuron.name
    return dp


def make_dps(neurons: navis.TreeNeuron):
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

    fn = partial(make_connector_dotprop)

    with ProcessPoolExecutor(os.cpu_count()) as p:
        out = [
            n
            for n in tqdm(
                p.map(fn, neurons, chunksize=50), "making dotprops", len(neurons)
            )
            
        ]

    return out

'''
def get_dps(l, r):
    """
    Obtains left and right pairs from prior functions.
    Transforms left pairs, makes dot products for both these and right pairs and outputs. Loaded from pickle if already ran

    Args:
        prune_strahler: prune TN by strahler index, defaults to lowest order only (-1, None)
        resample (int): resamples to # of nodes per every N units of cable? Defaults to 1000

    Returns:
        list: dotproducts for l_trans and r
    """    
    out = tuple(make_dps(ns) for ns in [left, right])
    
    return out
'''


T = TypeVar("T")


def split_training_testing(items: List[T], n_partitions=5, seed=DEFAULT_SEED
    ) -> Iterable[Tuple[List[T], List[T]]]:
    """ Splits items into training and testing sets (default n = 5) to generate custom score matrices ahead of cross-validated nblast

    Args:
        items: list of zipped L and R neurons
        n_partitions (int): # of partitions for cross-validation, defaults to 5.
        seed (int): defaults to specified DEFAULT_SEED for reproducibility

    Yields:
        Iterator[Iterable[Tuple[List[T], List[T]]]]: iteratively yields training and testing sets for n_partitions
    """
    items_arr = np.array(items, dtype=object)
    # Converts items (zipped list of L and R neurons) into np.ndarray
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


def train_synblaster(dp_pairs: List[Tuple[Dotprops, Dotprops]], threads=8
    ) -> navis.nbl.synblast_funcs.SynBlaster:
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
        dps, matching_lists, use_alpha = True, seed = DEFAULT_SEED
    ).with_bin_counts([21, 1])
    logger.info("Training...")
    score_mat = builder.build(threads)
    # build score matrix across # of threads
    logger.info("Trained")
    df = score_mat.to_dataframe()
    # print(df)
    # df.to_csv(HERE / "smat.csv")
    # Option to output score_mat as CSV

    synblaster = navis.nbl.synblast_funcs.SynBlaster(normalized = True, by_type = False, smat = None)
    synblaster.score_fn = score_mat

    return synblaster


def cross_validation(nrn_pairs, n_partitions=5, seed=DEFAULT_SEED):
    headers = [
        "skid_transformed-L",
        "skid_raw-R",
        "partition",
        "mean_normalized_synblast",
        "left_name"
    ]
    dtypes = [int, int, int, float, str]
    rows = []
    for partition, (training, testing) in tqdm(
        enumerate(split_training_testing(nrn_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        dps_left = []
        dps_right = []
        for pair in training:
            dps_left.append(make_connector_dotprop(pair[0]))
            dps_right.append(make_connector_dotprop(pair[1]))
            training_dps = list(zip(dps_left, dps_right))

        synblaster = train_synblaster(training_dps, threads=None)
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = synblaster.append(left)
            r_idx = synblaster.append(right)
            result = synblaster.single_query_target(l_idx, r_idx, scores="mean")
            name = left.name
            rows.append([left.id, right.id, partition, result, name])
            
    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)

    df.sort_values(headers[-1], axis=0, inplace=True)

    return df



def cross_validation2(tn_pairs, n_partitions=5, seed=DEFAULT_SEED):
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
        enumerate(split_training_testing(tn_pairs, n_partitions, seed)),
        "cross validation partitions",
        n_partitions,
    ):
        synblaster = train_synblaster(training, threads=None)
        # creates nblast algorithm from training set
        for left, right in tqdm(testing, f"testing {partition}"):
            l_idx = synblaster.append(left)
            r_idx = synblaster.append(right)
            result = synblaster.single_query_target(l_idx, r_idx, scores="mean")
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
        paired = get_transformed_neurons('cns')
        df = cross_validation(paired)
        print(df)
        # analysis for most pairs

        # paired2 = duplicate_prepare(dups)
        # df2 = cross_validation(paired2)
        # print(df2)
        # identical analysis done separately for duplicate skids, as CatmaidNeuron objects can't be repeated

        # concat_df = pd.concat([df,])
        df.to_csv(OUT_DIR / f"synalysis_results.csv", index=False)
        # merge both dataframes and export as CSV



### Visualise skeletons ###



def visualise_transform(l_trans, r_raw, skids, plot_height = 800, plot_width = 1200):
    nrns = navis.NeuronList(l_trans) + navis.NeuronList(r_raw)
    skeletons = nrns.idx[skids]
    viewer = skeletons.plot3d(height = plot_height, width = plot_width)
