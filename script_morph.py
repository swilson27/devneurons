#!/usr/bin/env python
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
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "output"
# Logging tracker of events/errors; returns parent directory of given path, provides cache and output directories

creds_path = os.environ.get("PYMAID_CREDENTIALS_PATH", HERE / "seymour.json")
with open(creds_path) as f:
    creds = json.load(f)
rm = pymaid.CatmaidInstance(**creds)
# Loads stored catmaid credentials

DEFAULT_SEED = 1991


## Define functions ##


def get_neurons():
    """
    Get CatmaidNeuronLists of left and right brain pairs.
    N.B. lists are each in arbitrary order.
    
    Returns:
        tuple: CmNL for left and right side respectively
    """    
    fpath = HERE / "neurons.pickle"
    if not os.path.exists(fpath):

        neurons = tuple(
            pymaid.get_neuron("annotation:sw;brainpair;" + side) for side in "LR"
        )

        with open(fpath, "wb") as f:
            pickle.dump(neurons, f, protocol=5)
    else:
        with open(fpath, "rb") as f:
            neurons = pickle.load(f)

    return neurons


def name_pair(left_names, right_names):
    """
    Replaces "left"/"right" (across pairs - left_names and right_names) with placeholder, to consider pairs as the same

    Args:
        left_names (dict): name:CmN, presumed to contain "left" pairs
        right_names (dict): name:CmN, presumed to contain "right pairs

    Yields:
        list of tuples
    """    
    unsided_l = {n.replace("left", "__SIDE__"): n for n in left_names}
    unsided_r = {n.replace("right", "__SIDE__"): n for n in right_names}
    in_both = set(unsided_l).intersection(unsided_r)
    for unsided_name in sorted(in_both):
        yield (unsided_l[unsided_name], unsided_r[unsided_name])


def get_landmarks():
    """
    Generates landmark coordinates from downloaded CSV of L and R brain hemispheres

    Returns:
        numpy array: coordinates of L and R brain hemispheres
    """    
    df = pd.read_csv(HERE / "bhem.csv", index_col=False, sep=", ")
    counts = Counter(df["landmark_name"])
    l_cp = []
    r_cp = []
    for name, count in counts.items():
        if count != 2:
            continue
        left, right = df.loc[df["landmark_name"] == name][["x", "y", "z"]].to_numpy()
        if left[0] < right[0]:
            left, right = right, left

        l_cp.append(left)
        r_cp.append(right)

    return np.asarray(l_cp), np.asarray(r_cp)


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
    fpath = HERE / "transformed_paired.pickle"

    if not os.path.isfile(fpath):
        neurons_l, neurons_r = get_neurons()
        by_name_l = dict(zip(neurons_l.name, neurons_l))
        by_name_r = dict(zip(neurons_r.name, neurons_r))

        paired = list(name_pair(by_name_l, by_name_r))
        l_cp, r_cp = get_landmarks()

        transform = navis.transforms.MovingLeastSquaresTransform(l_cp, r_cp)
        left_transform = []
        right_raw = []
        for l_name, r_name in paired:
            left_transform.append(transform_neuron(transform, by_name_l[l_name]))
            right_raw.append(by_name_r[r_name])

        out = (left_transform, right_raw)
        with open(fpath, "wb") as f:
            pickle.dump((left_transform, right_raw), f, 5)
    else:
        with open(fpath, "rb") as f:
            out = pickle.load(f)

    return out


def make_dotprop(neuron, prune_strahler=(-1, None), resample=1000):
    """
    Resamples neuron to given resolution (1000 nodes per every N units of cable?)
    Args:
        neuron (TN)
        prune_strahler (tuple, optional): prune TN by strahler index, defaults to lowest order only (-1, None).
        resample (int, optional):  Defaults to 1000.

    Returns:
        _type_: _description_
    """    
    nrn = neuron.prune_by_strahler(list(prune_strahler))
    nrn.tags = {}
    nrn.resample(resample, inplace=True)
    return make_dotprops(nrn, 5)


def make_dps(neurons, prune_strahler=(-1, None), resample=1000):
    """
    Makes dot products from input neurons?

    Args:
        neurons (_type_): _description_
        prune_strahler (tuple, optional): prune TN by strahler index, defaults to lowest order only (-1, None).
        resample (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """    
    out = []
    nrn: navis.TreeNeuron

    fn = partial(make_dotprop, prune_strahler=prune_strahler, resample=resample)

    with ProcessPoolExecutor(os.cpu_count()) as p:
        out = [
            n
            for n in tqdm(
                p.map(fn, neurons, chunksize=50), "making dotprops", len(neurons)
            )
        ]

    return out


def get_dps(prune_strahler=(-1, None), resample=1000):
    """
    Transforms left pairs, makes dot products for both these and right pairs and outputs. Loaded from pickle if already ran.

    Args:
        prune_strahler (tuple, optional): prune TN by strahler index, defaults to lowest order only (-1, None).
        resample (int, optional): _description_. Defaults to 1000.

    Returns:
        list: dotproducts for l_trans and r
    """    
    fpath = (
        HERE / f"dotprops_p{''.join(str(p) for p in prune_strahler)}_r{resample}.pickle"
    )
    if not fpath.is_file():
        l_trans, r = get_transformed_neurons()
        out = tuple(make_dps(ns, prune_strahler, resample) for ns in [l_trans, r])
        with open(fpath, "wb") as f:
            pickle.dump(out, f, 5)
    else:
        with open(fpath, "rb") as f:
            out = pickle.load(f)
    return out


def train_nblast(transformed_l: List[navis.Dotprops], right: List[navis.Dotprops]):
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


T = TypeVar("T")


def split_training_testing(
    items: List[T], n_partitions=5, seed=DEFAULT_SEED
) -> Iterable[Tuple[List[T], List[T]]]:
    items_arr = np.array(items, dtype=object)
    partition_idxs = np.arange(len(items_arr)) % n_partitions
    rng = np.random.default_rng(seed)
    rng.shuffle(partition_idxs)
    for idx in range(n_partitions):
        training = list(items_arr[partition_idxs != idx])
        testing = list(items_arr[partition_idxs == idx])
        yield training, testing


def train_nblaster(
    dp_pairs: List[Tuple[Dotprops, Dotprops]], threads=8
) -> navis.nbl.nblast_funcs.NBlaster:
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
    logger.info("Trained")
    # df = score_mat.to_dataframe()
    # print(df)
    # df.to_csv(HERE / "smat.csv")

    blaster = navis.nbl.nblast_funcs.NBlaster(True, True, None)
    blaster.score_fn = score_mat

    return blaster


def cross_validation(
    dp_pairs: List[Tuple[Dotprops, Dotprops]], n_partitions=5, seed=DEFAULT_SEED
):
    """ Takes zipped list of left and right dotprops, partitions data (training/testing) for cross-validation and applies nblaster function

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
        #"name_left",
    ]
    dtypes = [int, int, int, float]
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
            #name = pymaid.get_names(left)
            rows.append([left.id, right.id, partition, result]) #, name])

    df = pd.DataFrame(rows, columns=headers)
    for header, dtype in zip(headers, dtypes):
        df[header] = df[header].astype(dtype)

    
    df.sort_values(headers[0], axis=0, inplace=True)
    vals = pymaid.get_names(list(df['skid_transformed-L']))
    df["left_name"] = vals.values()
    df.sort_values(headers[3], axis=0, inplace=True)
    
    return df


## Run analysis ##


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        left, right = get_dps()
        paired = list(zip(left, right))
        df = cross_validation(paired)
        print(df)
        df.to_csv("crossval_results.csv", index=False)
