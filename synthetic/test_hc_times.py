import numpy as np

np.random.seed(0)
import struct

import util


def compare_models(num_instances):
    """
    Compare different models based on their computation times.

    This function reads computation times from files for different models and prints the mean times for each model.
    The models compared are:
    - CLG with BIC
    - CLG with Validation Likelihood
    - HSPBN with CLG
    - HSPBN with HCKDE

    Parameters:
    num_instances (int): The number of instances used in the experiments.
    bandwidth_selection (str): The bandwidth selection method used. Default is "normal_reference".

    Returns:
    None
    """
    clg_bic = np.empty((util.NUM_SIMULATIONS,))
    clg_vl = np.empty((util.NUM_SIMULATIONS,))
    hspbn_clg_vl = np.empty((util.NUM_SIMULATIONS,))
    hspbn_hckde_vl = np.empty((util.NUM_SIMULATIONS,))

    for p in util.PATIENCE:
        for i in range(util.NUM_SIMULATIONS):

            bic_folder = (
                "models/"
                + str(i).zfill(3)
                + "/"
                + str(num_instances)
                + "/HillClimbing/CLG/BIC_"
                + str(p)
            )
            try:
                with open(bic_folder + "/time", "rb") as f:
                    clg_bic[i] = struct.unpack("<d", f.read())[0]
            except FileNotFoundError:
                clg_bic[i] = None

            clg_vl_folder = (
                "models/"
                + str(i).zfill(3)
                + "/"
                + str(num_instances)
                + "/HillClimbing/CLG/ValidationLikelihood_"
                + str(p)
            )
            try:
                with open(clg_vl_folder + "/time", "rb") as f:
                    clg_vl[i] = struct.unpack("<d", f.read())[0]
            except FileNotFoundError:
                clg_vl[i] = None

            hspbn_clg_folder = (
                "models/"
                + str(i).zfill(3)
                + "/"
                + str(num_instances)
                + "/HillClimbing/HSPBN/"
                + str(p)
            )
            try:
                with open(hspbn_clg_folder + "/time", "rb") as f:
                    hspbn_clg_vl[i] = struct.unpack("<d", f.read())[0]
            except FileNotFoundError:
                hspbn_clg_vl[i] = None

            hspbn_hckde_folder = (
                "models/"
                + str(i).zfill(3)
                + "/"
                + str(num_instances)
                + "/HillClimbing/HSPBN_HCKDE/"
                + str(p)
            )

            try:
                with open(hspbn_hckde_folder + "/time", "rb") as f:
                    hspbn_hckde_vl[i] = struct.unpack("<d", f.read())[0]
            except FileNotFoundError:
                hspbn_hckde_vl[i] = None

        print("BIC p = " + str(p) + ": " + str(clg_bic.mean()))
        print("CLG-VL p = " + str(p) + ": " + str(clg_vl.mean()))
        print("HSPBN-CLG p = " + str(p) + ": " + str(hspbn_clg_vl.mean()))
        print("HSPBN-HCKDE p = " + str(p) + ": " + str(hspbn_hckde_vl.mean()))


if __name__ == "__main__":

    for i in util.INSTANCES:
        print(str(i) + " instances")
        print("=======================")
        compare_models(i)
