from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pathlib
from plotting import savefig

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
DP_DATA = DP_ROOT / "data"
DP_RESULTS = DP_ROOT / "results"


def ExperimentalDesign(wd: pathlib.Path=DP_RESULTS, run_ids: Optional[Sequence[str]]=None):
    """ This function plots the experimental design data for the CAR cultivation

    Parameters
    ----------
    wd : pathlib.Path
        Working directory where to save the output.
    run_ids : str
        Names of experiment runs to consider.
    """
    exp_raw_data = pandas.read_excel(DP_DATA / 'FullWellDescription.xlsx')
    exp_data = exp_raw_data.dropna(subset=['iptg', 'glucose'])
    # Filter the metadata to take only the first [runs].
    if run_ids is None:
        run_ids = exp_data.run.unique()
    print("Plotting for runs %s" % run_ids)
    exp_data = exp_data.loc[exp_data["run"].isin(run_ids)]
    print(exp_data)

    iptg = np.array(exp_data['iptg'])
    glucose =np.array( exp_data['glucose'])
    fig1, ax = plt.subplots(figsize = (4.5, 2))
    ax.plot(glucose, iptg, linestyle='', marker='o')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 40)

    ax.yaxis.set_ticks(np.arange(0, 40.000001, 10))
    ax.xaxis.set_ticks(np.arange(0, 5.001, 1))
    
    ax.set_xlabel(r"$\mathrm{Feed\ rate\ [g\ L^{-1} h^{-1}]}$")
    ax.set_ylabel(r"$\mathrm{IPTG}\ [\mu M]$")
    savefig(fig1, "ExpDesign", wd=wd)
    return


if __name__ == "__main__":
    ExperimentalDesign()
