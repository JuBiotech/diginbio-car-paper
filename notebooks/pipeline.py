"""
Unit operations of the full analysis pipeline.

Every unit operation requires a working directory argument ``wd``.
Some unit operations need additional kwargs.
"""

import logging
import pathlib

import aesara.tensor as at
import arviz
import calibr8
import numpy
import pandas
import pymc as pm
from matplotlib import pyplot

import dataloading
import models
import plotting


_log = logging.getLogger(__file__)

DESIGN_COLS = ["iptg", "glucose"]


def load_biomass_calibration(wd: pathlib.Path):
    df_cal = dataloading.read_biomass_calibration().set_index("biomass")
    df_cal.to_excel(wd / "data_biomass_calibration.xlsx")
    return


def fit_biomass_calibration(wd: pathlib.Path, wavelength: int):
    df_cal = pandas.read_excel(wd / "data_biomass_calibration.xlsx", index_col="biomass")
    cm = models.LogisticBiomassAbsorbanceModel()
    calibr8.fit_scipy(
        cm,
        independent=df_cal.index.to_numpy(),
        dependent=df_cal[f"A{wavelength}"].to_numpy(),
        theta_guess=[0, 5, 0, 2, -1] + [0.01, 0, 2],
        theta_bounds=[
            (-numpy.inf, 0.2), # L_L
            (1.5, numpy.inf),  # L_U
            (-3, 2),           # log10(I_x)
            (0.5, 3),          # dy/dlog10(x)
            (-3, 3),           # c
            (0.0001, 0.1),
            (0.0001, 0.1),
            (1, 30),
        ],
    )
    cm.save(wd / f"cm_biomass_A{wavelength}.json")
    fig, axs = calibr8.plot_model(cm)
    fig.suptitle(f"Biomass calibration at {wavelength} nm")
    fig.savefig(wd / f"cm_biomass_A{wavelength}.png")
    pyplot.close()
    return


def fit_product_calibration(wd: pathlib.Path):
    df = dataloading.read_ph_product_calibration("pH_Kalibrierung_mit_ABAO.xlsx")
    cm = models.LinearProductAbsorbanceModel()
    calibr8.fit_scipy(
        cm,
        independent=df["product"].to_numpy(),
        dependent=df["A360"].to_numpy(),
        theta_guess=[0.2, 0.5, 0.1, 5],
        theta_bounds=[
            (0, 0.5), # intercept
            (0.2, 1),    # slope
            (0.01, 1),   # scale
            (1, 30),     # df
        ],
    )
    cm.save(wd / "cm_product_A360.json")
    fig, axs = calibr8.plot_model(cm)
    fig.suptitle(f"ABAO-product calibration at 360 nm")
    fig.savefig(wd / "cm_product_A360.png")
    pyplot.close()
    return


def load_layout(wd: pathlib.Path):
    df_layout = dataloading.get_layout("FullWellDescription.xlsx", DESIGN_COLS)
    df_counts = dataloading.count_replicates(df_layout)
    with pandas.ExcelWriter(wd / "layout.xlsx") as writer:
        df_layout.to_excel(writer, sheet_name="layout")
        df_counts.to_excel(writer, sheet_name="counts")
    return


def load_observations(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", index_col="replicate_id")
    df_time, df_A360, df_A600 = dataloading.vectorize_observations(
        df_layout,
        observations={
            f"Carboxylase_{v}": (
                dataloading.read_absorbances(rf"Carboxylase_{v}_360nm.csv"),
                dataloading.read_absorbances(rf"Carboxylase_{v}_600nm.csv"),
            )
            for v in [19, 20, 21, 23]
        }
    )
    with pandas.ExcelWriter(wd / "observations.xlsx") as writer:
        df_time.to_excel(writer, sheet_name="time")
        df_A360.to_excel(writer, sheet_name="A360")
        df_A600.to_excel(writer, sheet_name="A600")
    return


def _build_model(wd: pathlib.Path):
    _log.info("Loading calibrations")
    cmX360 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A360.json")
    cmX600 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A600.json")
    cmP360 = models.LinearProductAbsorbanceModel.load(wd / "cm_product_A360.json")
    _log.info("Loading experimental data")
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    df_time = pandas.read_excel(wd / "observations.xlsx", sheet_name="time", index_col="replicate_id")
    df_A360 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A360", index_col="replicate_id")
    df_A600 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A600", index_col="replicate_id")
    _log.info("Building the model")
    with pm.Model() as pmodel:
        pmodel = models.build_model(
            df_layout,
            df_time,
            df_A360,
            df_A600,
            cmX360,
            cmX600,
            cmP360,
            gp_k_design=True,
            gp_X_factor=True,
            random_walk_X=True,
            design_cols=DESIGN_COLS,
        )
    return pmodel


def fit_model(wd: pathlib.Path, **sample_kwargs):
    pmodel = _build_model(wd)
    #_log.info("Creating a model graph")
    #modelgraph = pm.model_to_graphviz(pmodel)
    #_log.info("Saving the model graph")
    #modelgraph.render(filename=str(wd / "model.pdf"), format="pdf")

    _log.info("Running MCMC")
    with pmodel:
        idata = pm.sample(**sample_kwargs)
    _log.info("Saving the trace")
    idata.to_netcdf(wd / "trace.nc")
    return


def export_summary(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    idata = arviz.from_netcdf(wd / "trace.nc")
    summary = plotting.summarize(idata, df_layout)
    summary.to_excel(wd / "summary.xlsx")
    return


def plot_trace(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")
    groups = plotting.interesting_groups(idata.posterior)
    for title, vars in groups.items():
        _log.info("Plotting trace group %s with variables %s", title, vars)
        axs = arviz.plot_trace(idata, var_names=vars)
        fig = pyplot.gcf()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(wd / f"plot_trace_{title}.png")
        pyplot.close()
    return


def plot_3d_by_design(wd: pathlib.Path, var_name: str):
    idata = arviz.from_netcdf(wd / "trace.nc")
    def fn_plot3d(t):
        plotter = getattr(plotting, f"plot_3d_{var_name}")
        plotter(idata, azim=-45+30*numpy.sin(t*2*numpy.pi))

    plotting.plot_gif(
        fn_plot=fn_plot3d,
        fp_out=wd / f"plot_3d_{var_name}.gif",
        data=numpy.arange(0, 1, 1/60),
        fps=15,
        delay_frames=0
    )
    return


def plot_kinetics(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")
    cmX600 = models.LogisticBiomassAbsorbanceModel.load(wd / "cm_biomass_A600.json")
    summary = pandas.read_excel(wd / "summary.xlsx", index_col="replicate_id")

    def fn_plot(rid):
        plotting.plot_reaction(idata, rid, cm_600=cmX600, reaction_order=summary.index)
        return
    plotting.plot_gif(
        fn_plot=fn_plot,
        fp_out=wd / "plot_kinetics.gif",
        data=summary.index,
        fps=3
    )
    return


def plot_gp_X_factor(wd: pathlib.Path):
    pmodel = _build_model(wd)
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Adding high-resolution GP conditional")
    dense = numpy.linspace(0, 6)
    with pmodel:
        log_X_factor = pmodel.gp_log_X_factor.conditional(
            "dense_log_X_factor",
            Xnew=dense[:, None],
            dims="dense_glucose",
        )
        X_factor = pm.Deterministic(
            "dense_X_factor",
            at.exp(log_X_factor),
            dims="dense_glucose",
        )
        X0_design = pm.Deterministic(
            "dense_Xend_2mag",
            pmodel["Xend_batch"] * X_factor,
            dims="dense_glucose",
        )

        _log.info("Sampling posterior predictive")
        ipp = pm.sample_posterior_predictive(
            idata, var_names=["dense_log_X_factor", "dense_X_factor", "dense_Xend_2mag"]
        )

    _log.info("Plotting")
    fig, ax = pyplot.subplots()

    pm.gp.util.plot_gp_dist(
        ax=ax,
        x=dense,
        samples=ipp.posterior_predictive["dense_Xend_2mag"].stack(sample=("chain", "draw")).values.T,
        plot_samples=False,
    )
    ax.set(
        ylabel="$X_{end,2mag}\ \ \ [g_\mathrm{biomass}/L]$",
        xlabel="$\mathrm{glucose\ feed\ rate}\ \ \ [g_\mathrm{glucose}/L_\mathrm{reactor}/h]$",
        xlim=(min(dense), max(dense)),
    )
    fig.savefig(wd / "plot_gp_X_factor.png")
    pyplot.close()
    return
