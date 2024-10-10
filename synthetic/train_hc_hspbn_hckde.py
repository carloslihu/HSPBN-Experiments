import glob
import math
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path

import generate_dataset
import pandas as pd
import util

import pybnesian as pbn

patience = util.PATIENCE


def run_hc_hspbn_hckde(idx_dataset, i):
    """
    Runs the Hill Climbing algorithm with HSPBN and HCKDE on a synthetic dataset.

    Parameters:
    idx_dataset (int): Index of the synthetic dataset to be used.
    i (int): Index of the specific instance of the dataset.

    This function performs the following steps:
    1. Initializes the Greedy Hill Climbing algorithm and the operator pool.
    2. Reads and preprocesses the dataset.
    3. Sets up the validated likelihood for model evaluation.
    4. Iterates over different patience values to perform model estimation.
    5. Saves the resulting models and computation time.

    The results are saved in a folder structure based on the dataset index, instance index, and patience value.
    """
    hc = pbn.GreedyHillClimbing()
    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])

    df = pd.read_csv(
        "data/synthetic_" + str(idx_dataset).zfill(3) + "_" + str(i) + ".csv"
    )
    df = generate_dataset.preprocess_dataset(df)

    vl = pbn.ValidatedLikelihood(df, k=10, seed=util.SEED)
    for p in patience:
        result_folder = (
            "models/"
            + str(idx_dataset).zfill(3)
            + "/"
            + str(i)
            + "/HillClimbing/HSPBN_HCKDE/"
            + str(p)
        )
        Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + "/end.lock"):
            continue

        cb_save = pbn.SaveModel(result_folder)

        node_types = [
            (name, pbn.CKDEType()) for name in df.select_dtypes("double").columns.values
        ] + [
            (name, pbn.DiscreteFactorType())
            for name in df.select_dtypes("category").columns.values
        ]
        start_model = pbn.SemiparametricBN(list(df.columns.values), node_types)

        start_time = time.time()
        bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p)
        end_time = time.time()

        with open(result_folder + "/time", "wb") as f:
            f.write(struct.pack("<d", end_time - start_time))

        iters = sorted(glob.glob(result_folder + "/*.pickle"))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + "/" + str(number + 1).zfill(6) + ".pickle")
        with open(result_folder + "/end.lock", "w") as f:
            pass


for i in util.INSTANCES:
    for idx_dataset in range(
        0, math.ceil(util.NUM_SIMULATIONS / util.PARALLEL_THREADS)
    ):

        num_processes = min(
            util.PARALLEL_THREADS,
            util.NUM_SIMULATIONS - idx_dataset * util.PARALLEL_THREADS,
        )
        with mp.Pool(processes=num_processes) as p:
            p.starmap(
                run_hc_hspbn_hckde,
                [
                    (util.PARALLEL_THREADS * idx_dataset + ii, i)
                    for ii in range(num_processes)
                ],
            )
