import numpy as np

np.random.seed(0)
import glob
from pathlib import Path

import pandas as pd
import rpy2
import util
from generate_dataset import preprocess_dataset
from generate_new_bns import (
    FixedCLG,
    FixedCLGType,
    FixedDiscreteFactor,
    FixedDiscreteFactorType,
    NormalMixtureCPD,
    NormalMixtureType,
    ProbabilisticModel,
)
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

import pybnesian as pbn
from pybnesian import load

numpy2ri.activate()


class PluginEstimator(pbn.BandwidthSelector):

    def __init__(self):
        pbn.BandwidthSelector.__init__(self)
        self.ks = importr("ks")

    def bandwidth(self, df, variables):
        data = df.to_pandas().loc[:, variables].dropna().to_numpy()

        if data.shape[0] <= len(variables):
            raise pbn.SingularCovarianceData(
                "[instances] The data covariance could not be estimated because the matrix is singular."
            )

        cov = np.cov(data, rowvar=False)
        if np.linalg.matrix_rank(cov) < len(variables):
            raise pbn.SingularCovarianceData(
                "[rank] The data covariance could not be estimated because the matrix is singular."
            )

        try:
            if len(variables) == 1:
                return np.asarray([self.ks.hpi(data)])
            else:
                return self.ks.Hpi(data)
        except rpy2.rinterface_lib.embedded.RRuntimeError as rerror:
            if "scale estimate is zero for input data" in str(rerror):
                raise pbn.SingularCovarianceData(
                    "[scalest 1d] The data covariance could not be estimated because the matrix is singular."
                )
            else:
                raise rerror


def compare_models(num_instances, bandwidth_selection="normal_reference"):
    """
    Compare probabilistic models using various metrics.

    Parameters:
    num_instances (int): The number of instances to use for training the models.
    bandwidth_selection (str): The method for bandwidth selection. Possible options are:
                               "normal_reference", "ucv", and "plugin". Default is "normal_reference".

    This function performs the following steps:
    1. Loads the ground truth models and computes their log-likelihood on the test datasets.
    2. For each patience value in `util.PATIENCE`, it:
       a. Loads the training and test datasets.
       b. Loads the final model from the specified result folder.
       c. Fits the model using the specified bandwidth selection method.
       d. Computes the log-likelihood, Structural Hamming Distance (SHD), Hamming distance,
          and Hamming type distance between the final model and the ground truth model.
       e. Prints the mean log-likelihood, Hamming distance, SHD, and Hamming type distance
          for the current patience value.

    Raises:
    ValueError: If an invalid bandwidth selection method is provided.
    """
    truth_ll = np.empty((util.NUM_SIMULATIONS,))

    ll = np.empty((util.NUM_SIMULATIONS,))
    shd = np.empty((util.NUM_SIMULATIONS,))
    hamming = np.empty((util.NUM_SIMULATIONS,))
    hamming_type = np.empty((util.NUM_SIMULATIONS,))

    for i in range(util.NUM_SIMULATIONS):
        test_df = pd.read_csv("data/synthetic_" + str(i).zfill(3) + "_test.csv")
        test_df = preprocess_dataset(test_df)

        true_model = ProbabilisticModel.load(
            "ground_truth_models/model_" + str(i) + ".pickle"
        )
        truth_ll[i] = true_model.ground_truth_bn.slogl(test_df)

    print("True model loglik: " + str(truth_ll.mean()))

    for p in util.PATIENCE:
        for i in range(util.NUM_SIMULATIONS):
            true_model = ProbabilisticModel.load(
                "ground_truth_models/model_" + str(i) + ".pickle"
            )

            train_df = pd.read_csv(
                "data/synthetic_" + str(i).zfill(3) + "_" + str(num_instances) + ".csv"
            )
            train_df = preprocess_dataset(train_df)
            test_df = pd.read_csv("data/synthetic_" + str(i).zfill(3) + "_test.csv")
            test_df = preprocess_dataset(test_df)

            result_folder = (
                "models/"
                + str(i).zfill(3)
                + "/"
                + str(num_instances)
                + "/HillClimbing/HSPBN_HCKDE/"
                + str(p)
            )
            Path(result_folder).mkdir(parents=True, exist_ok=True)

            all_models = sorted(glob.glob(result_folder + "/*.pickle"))
            final_model = load(all_models[-1])

            if bandwidth_selection == "normal_reference":
                final_model.fit(train_df)
            elif bandwidth_selection == "ucv":
                args = pbn.Arguments({pbn.CKDEType(): (pbn.UCV(),)})

                final_model.fit(train_df, args)
            elif bandwidth_selection == "plugin":
                args = pbn.Arguments({pbn.CKDEType(): (PluginEstimator(),)})

                final_model.fit(train_df, args)
            else:
                raise ValueError(
                    "Wrong bandwidth selection method. Possible options are: "
                    '"normal_reference", "ucv" and "plugin".'
                )

            ll[i] = final_model.slogl(test_df)
            shd[i] = util.shd(final_model, true_model.expected_bn)
            hamming[i] = util.hamming(final_model, true_model.expected_bn)
            hamming_type[i] = util.hamming_type(final_model, true_model.expected_bn)

        print("Loglik, ValidationScore p " + str(p) + ": " + str(ll.mean()))
        print("Hamming, ValidationScore p " + str(p) + ": " + str(hamming.mean()))
        print("SHD, ValidationScore p " + str(p) + ": " + str(shd.mean()))
        print(
            "Hamming type, ValidationScore p "
            + str(p)
            + ": "
            + str(hamming_type.mean())
        )
        print()


if __name__ == "__main__":

    for i in util.INSTANCES:
        print(str(i) + " instances")
        print("=======================")
        compare_models(i, bandwidth_selection="normal_reference")
