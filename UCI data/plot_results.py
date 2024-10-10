import adult
import australian_statlog
import cover_type
import german_statlog
import kdd
import liver_disorders
import pandas as pd
import plot_cd_diagram
import thyroid_hypothyroid
import thyroid_sick
import tikzplotlib
import util


def result_string(df_name, df):
    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = (
        util.test_hc_models(df_name, df)
    )

    (bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result) = (
        util.common_instance_results(
            bic_result, clg_vl_result, hspbn_vl_result, hspbn_hckde_vl_result
        )
    )

    return (
        df_name
        + ","
        + ",".join([str(ll.sum()) for ll in bic_result])
        + ","
        + ",".join([str(ll.sum()) for ll in clg_vl_result])
        + ","
        + ",".join([str(ll.sum()) for ll in hspbn_vl_result])
        + ","
        + ",".join([str(ll.sum()) for ll in hspbn_hckde_vl_result])
    )


def save_summary_results():
    datasets = [
        ("Abalone", "data/Abalone/abalone.data", util.preprocess_dataframe),
        (
            "Adult",
            ["data/Adult/adult.data", "data/Adult/adult.test"],
            adult.preprocess_dataframe,
        ),
        (
            "AustralianStatlog",
            "data/AustralianStatlog/australian.dat",
            australian_statlog.preprocess_dataframe,
        ),
        (
            "CoverType",
            "data/Cover Type/covtype_truncated.csv",
            cover_type.preprocess_dataframe,
        ),
        ("CreditApproval", "data/Credit Approval/crx.data", util.preprocess_dataframe),
        (
            "GermanStatlog",
            "data/GermanStatlog/german.data",
            german_statlog.preprocess_dataframe,
        ),
        ("KDDCup", "data/KDD Cup/kddcup_truncated.csv", kdd.preprocess_dataframe),
        (
            "LiverDisorders",
            "data/Liver disorders/bupa.data",
            liver_disorders.preprocess_dataframe,
        ),
        (
            "Thyroid-hypothyroid",
            "data/Thyroid/hypothyroid.data",
            thyroid_hypothyroid.preprocess_dataframe,
        ),
        ("Thyroid-sick", "data/Thyroid/sick.data", thyroid_sick.preprocess_dataframe),
    ]

    string_file = (
        "Dataset,"
        + ",".join(["CLG_BIC_" + str(p) for p in util.PATIENCE])
        + ","
        + ",".join(["CLG_" + str(p) for p in util.PATIENCE])
        + ","
        + ",".join(["HSPBN_" + str(p) for p in util.PATIENCE])
        + ","
        + ",".join(["HSPBN_HCKDE_" + str(p) for p in util.PATIENCE])
        + "\n"
    )

    for name, paths, preprocess in datasets:
        if isinstance(paths, list):
            df = pd.concat(
                [pd.read_csv(path, na_values="?") for path in paths], ignore_index=True
            )
        else:
            df = pd.read_csv(paths, na_values="?")
        df = preprocess(df)
        try:
            string_file += result_string(name, df) + "\n"
        except Exception as e:
            print(f"Error processing {name}: {e}")

    with open("data/result_summary.csv", "w") as f:
        f.write(string_file)


def plot_cd_diagrams(rename_dict):
    df_algorithms = pd.read_csv("data/result_summary.csv")
    df_algorithms = df_algorithms.set_index("Dataset")

    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    names = [rename_dict[s] for s in names]

    posthoc_methods = ["cd", "holm", "bergmann"]
    filenames = ["Nemenyi.tex", "Holm.tex", "Bergmann.tex"]

    for method, filename in zip(posthoc_methods, filenames):
        plot_cd_diagram.graph_ranks(
            avgranks, names, df_algorithms.shape[0], posthoc_method=method
        )
        try:
            tikzplotlib.save(
                f"plots/{filename}",
                standalone=True,
                axis_width="14cm",
                axis_height="5cm",
            )
        except AttributeError as e:
            print(f"Error saving plot: {e}")


if __name__ == "__main__":
    rename_dict = {
        "CLG_BIC_0": r"CLGBN-BIC $\lambda=0$",
        "CLG_BIC_5": r"CLGBN-BIC $\lambda=5$",
        "CLG_BIC_15": r"CLGBN-BIC $\lambda=15$",
        "CLG_0": r"CLGBN-VL $\lambda=0$",
        "CLG_5": r"CLGBN-VL $\lambda=5$",
        "CLG_15": r"CLGBN-VL $\lambda=15$",
        "HSPBN_0": r"HSPBN-CLG $\lambda=0$",
        "HSPBN_5": r"HSPBN-CLG $\lambda=5$",
        "HSPBN_15": r"HSPBN-CLG $\lambda=15$",
        "HSPBN_HCKDE_0": r"HSPBN-HCKDE $\lambda=0$",
        "HSPBN_HCKDE_5": r"HSPBN-HCKDE $\lambda=5$",
        "HSPBN_HCKDE_15": r"HSPBN-HCKDE $\lambda=15$",
    }

    save_summary_results()
    plot_cd_diagrams(rename_dict)
