import glob
import math
import multiprocessing as mp
import os
import pathlib
import struct
import time

import generate_dataset
import pandas as pd
import util

import pybnesian as pbn

patience = util.PATIENCE


def run_hc_hspbn(idx_dataset, i):
    hc = pbn.GreedyHillClimbing()
    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])

    df = pd.read_csv(
        "data/synthetic_" + str(idx_dataset).zfill(3) + "_" + str(i) + ".csv"
    )
    df = generate_dataset.preprocess_dataset(df)

    bic = pbn.BIC(df)
    vl = pbn.ValidatedLikelihood(df, k=10, seed=util.SEED)
    for p in patience:
        result_folder = (
            "models/"
            + str(idx_dataset).zfill(3)
            + "/"
            + str(i)
            + "/HillClimbing/CLG/BIC_"
            + str(p)
        )
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(result_folder + "/end.lock"):
            cb_save = pbn.SaveModel(result_folder)
            start_model = pbn.CLGNetwork(list(df.columns.values))
            arc_op = pbn.ArcOperatorSet()

            start_time = time.time()
            bn = hc.estimate(arc_op, bic, start_model, callback=cb_save, patience=p)
            end_time = time.time()

            with open(result_folder + "/time", "wb") as f:
                f.write(struct.pack("<d", end_time - start_time))

            iters = sorted(glob.glob(result_folder + "/*.pickle"))
            last_file = os.path.basename(iters[-1])
            number = int(os.path.splitext(last_file)[0])
            bn.save(result_folder + "/" + str(number + 1).zfill(6) + ".pickle")
            with open(result_folder + "/end.lock", "w") as f:
                pass

        result_folder = (
            "models/"
            + str(idx_dataset).zfill(3)
            + "/"
            + str(i)
            + "/HillClimbing/CLG/ValidationLikelihood_"
            + str(p)
        )
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(result_folder + "/end.lock"):
            cb_save = pbn.SaveModel(result_folder)
            start_model = pbn.CLGNetwork(list(df.columns.values))

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


if __name__ == "__main__":
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
                    run_hc_hspbn,
                    [
                        (util.PARALLEL_THREADS * idx_dataset + ii, i)
                        for ii in range(num_processes)
                    ],
                )
