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
import xarray

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
        theta_guess=[0, 2, 0.5, 1.5, -1] + [0.01, 0],
        theta_bounds=[
            (-numpy.inf, 0.2), # L_L
            (1.5, numpy.inf),  # L_U
            (-3, 2),           # log10(I_x)
            (0.5, 3),          # dy/dlog10(x)
            (-3, 3),           # c
            (0.0001, 0.1),
            (0.0001, 0.1),
        ],
    )
    cm.save(wd / f"cm_biomass_A{wavelength}.json")
    return


def plot_biomass_calibration(wd: pathlib.Path, wavelength: int):
    cm = models.LogisticBiomassAbsorbanceModel.load(wd / f"cm_biomass_A{wavelength}.json")
    fig, axs = calibr8.plot_model(cm)
    xlabel = r"$\mathrm{biomass\ concentration\ [\frac{g_{CDW}}{L}]}$"
    axs[0].set(
        ylabel=r"$\mathrm{absorbance_{%s\ nm}}\ [-]$" % wavelength,
        xlabel=xlabel,
    )
    axs[1].set(
        xlabel=xlabel,
    )
    axs[2].set(
        ylabel=r"$\mathrm{absolute\ residual\ [-]}$",
        xlabel=xlabel,
    )
    plotting.savefig(fig, f"cm_biomass_A{wavelength}", wd=wd)
    pyplot.close()
    return


def fit_product_calibration(wd: pathlib.Path):
    df = dataloading.read_ph_product_calibration("pH_Kalibrierung_mit_ABAO.xlsx")
    cm = models.LinearProductAbsorbanceModel()
    calibr8.fit_scipy(
        cm,
        independent=df["product"].to_numpy(),
        dependent=df["A360"].to_numpy(),
        theta_guess=[0.2, 0.5, 0.1],
        theta_bounds=[
            (0, 0.5), # intercept
            (0.2, 1),    # slope
            (0.01, 1),   # scale
        ],
    )
    cm.save(wd / "cm_product_A360.json")
    return


def plot_product_calibration(wd: pathlib.Path):
    cm = models.LinearProductAbsorbanceModel.load(wd / "cm_product_A360.json")
    fig, axs = calibr8.plot_model(cm)
    xlabel = r"$\mathrm{product\ concentration\ [mM]}$"
    axs[0].set(
        ylabel=r"$\mathrm{absorbance_{360\ nm}}\ [-]$",
        xlabel=xlabel,
    )
    axs[1].set(
        xlabel=xlabel,
    )
    axs[2].set(
        ylabel=r"$\mathrm{absolute\ residual\ [-]}$",
        xlabel=xlabel,
    )
    plotting.savefig(fig, f"cm_product_A360", wd=wd)
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

    sample_kwargs.setdefault("discard_tuned_samples", False)

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
        for key, prefix in [
            ("posterior", "plot_trace"),
            ("warmup_posterior", "plot_warmup"),
        ]:
            if not key in idata:
                _log.warning("InferenceData object has no group %s.", key)
                continue
            _log.info("Plotting %s group %s with variables %s", key, title, vars)
            axs = arviz.plot_trace(idata[key], var_names=vars)
            fig = pyplot.gcf()
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(wd / f"{prefix}_{title}.png")
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
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Creating the model")
    pmodel = _build_model(wd)

    _log.info("Adding high-resolution GP conditional")
    dense = numpy.linspace(0.01, 6, 300)
    with pmodel:
        _log.info("Adding variables for high-quality predictives")

        if "cycle_segment" not in idata.posterior.coords:
            # The plotting code below only works for models >= 2f12066bcea31f91c26cfe9aac6ec16aeaf58679.
            raise NotImplementedError("This is an outdated InferenceData file!")
        log_X_factor = pmodel.gp_log_X_factor.conditional(
            "dense_log_X_factor",
            Xnew=numpy.log10(dense[:, None]),
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
        _log.info("Sampling prior predictive")
        pprior = pm.sample_prior_predictive(
            samples=1500,
            var_names=["Xend_batch", "dense_log_X_factor", "dense_X_factor", "dense_Xend_2mag"],
            return_inferencedata=False,
        )
        _log.info("Sampling posterior predictive")
        pposterior = pm.sample_posterior_predictive(
            idata,
            samples=1500,
            var_names=["dense_log_X_factor", "dense_X_factor", "dense_Xend_2mag"],
            return_inferencedata=False,
        )
        _log.info("Converting to InferenceData")
        pp = pm.to_inference_data(prior=pprior, posterior_predictive=pposterior)
        del pprior, pposterior

    _log.info("Plotting")
    fig, axs = pyplot.subplots(dpi=200, ncols=2, figsize=(12, 6), sharey=True)

    for ax, ds in zip(axs, [pp.prior, pp.posterior_predictive]):
        stackdims = ("chain", "draw") if "chain" in ds.dims else ("draw",)
        pm.gp.util.plot_gp_dist(
            ax=ax,
            x=dense,
            samples=ds["dense_Xend_2mag"].stack(sample=stackdims).values.T,
            plot_samples=True,
            palette=pyplot.cm.Greens,
        )
        ax.set(
            xlabel="$\mathrm{glucose\ feed\ rate}\ \ \ [g_\mathrm{glucose}/L_\mathrm{reactor}/h]$",
            xlim=(0, max(dense)),
        )
    axs[0].set(
        ylabel="$X_{end,2mag}\ \ \ [g_\mathrm{biomass}/L]$",
        ylim=(0, 1.5),
        title="prior",
    )
    axs[1].set(
        ylim=(0, 1),
        title="posterior",
    )
    fig.savefig(wd / "plot_gp_X_factor.png")
    pyplot.close()
    return


def sample_gp_metric_posterior_predictive(wd: pathlib.Path, draws:int=500, n: int=30):
    idata = arviz.from_netcdf(wd / "trace.nc")

    _log.info("Creating the model")
    pmodel = _build_model(wd)

    _log.info("Adding high-resolution GP conditional")
    # Create a dense grid
    dense_long = xarray.DataArray(
        models.bounds_to_grid(idata.constant_data.X_design_log10_bounds.values, n),
        dims=("dense_id", "design_dim"),
        coords={
            "dense_id": numpy.arange(n**2),
            "design_dim": idata.posterior.design_dim.values
        }
    )
    dense_grid = models.reshape_dim(
        dense_long,
        from_dim="dense_id",
        to_shape=(n, n),
        to_dims=idata.posterior.design_dim.values,
    )
    with pmodel:
        _log.info("Adding variables for high-quality predictives")
        log_k_design = pmodel.gp_log_k_design.conditional(
            "dense_log_k_design",
            Xnew=dense_long.values,
            dims="dense_id",
            jitter=pm.gp.util.JITTER_DEFAULT
        )
        k_design = pm.Deterministic("dense_k_design", at.exp(log_k_design), dims="dense_id")

        _log.info("Sampling posterior predictive")
        pp = pm.sample_posterior_predictive(
            idata,
            samples=draws,
            var_names=["dense_log_k_design", "dense_k_design"],
            return_inferencedata=False
        )
        _log.info("Saving to InferenceData")
        pposterior = pm.to_inference_data(
            posterior_predictive=pp
        )
        # Include the dense grid in the savefile
        pposterior.posterior_predictive["dense_long"] = dense_long
        pposterior.posterior_predictive["dense_grid"] = dense_grid
        pposterior.to_netcdf(wd / "predictive_posterior.nc")
    return


def plot_gp_metric_posterior_predictive(
    wd: pathlib.Path,
    label=r"$\mathrm{specific\ activity\ [\frac{mM}{h} / \frac{g_{CDW}}{L}]}$",
    var_name="dense_k_design",
):
    idata = arviz.from_netcdf(wd / "trace.nc")
    pposterior = arviz.from_netcdf(wd / "predictive_posterior.nc")

    design_dims = idata.posterior.design_dim.values

    # Extract relevant data arrays
    design_dims = list(idata.constant_data.design_dim.values)
    D = len(design_dims)
    if not D == 2:
        raise NotImplementedError(f"3D visualization for {D}-dimensional designs is not implemented.")
    dense_long = pposterior.posterior_predictive["dense_long"]
    dense_grid = pposterior.posterior_predictive["dense_grid"]
    BOUNDS = numpy.array([
        dense_long.min(dim="dense_id"),
        dense_long.max(dim="dense_id"),
    ]).T

    # Reshape the long-form arrays into the dense 2D grid
    gridshape = tuple(
        dense_grid.sizes[design_dim]
        for design_dim in design_dims
    )
    Z = models.reshape_dim(
        pposterior.posterior_predictive[var_name],
        from_dim="dense_id",
        to_shape=gridshape,
        to_dims=design_dims,
    )

    # Take the median and HDI of the samples in grid-layout
    median = Z.median(("chain", "draw"))
    hdi = arviz.hdi(Z, hdi_prob=0.9)[var_name]

    assert design_dims[0] == "glucose"
    assert design_dims[1] == "iptg"

    def fn_plot(azim=-65):
        fig = pyplot.figure(dpi=140)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r"$\mathrm{log_{10}(glucose\ feed\ rate\ [g/L/h])}$")
        ax.set_ylabel(r"$\mathrm{log_{10}(IPTG\ concentration\ [µM])}$")
        ax.set_zlabel(label)

        # Plot surfaces for lower/median/upper
        for q, z in [
            (0.05, hdi.sel(hdi="lower")),
            (0.5, median),
            (0.95, hdi.sel(hdi="higher")),
        ]:
            ax.plot_surface(
                dense_grid.sel(design_dim=design_dims[0]).values,
                dense_grid.sel(design_dim=design_dims[1]).values,
                z.values,
                cmap=pyplot.cm.autumn,
                linewidth=0,
                antialiased=False,
                alpha=1 - abs(q/100 - 0.5) - 0.25
            )
        ax.view_init(elev=25, azim=azim)
        return fig, [[ax]]

    fig, _ = fn_plot()
    plotting.savefig(fig, f"plot_3d_pp_{var_name}", wd=wd)
    pyplot.close()

    def fn_plot3d(t):
        fn_plot(azim=-45+30*numpy.sin(t*2*numpy.pi))
    plotting.plot_gif(
        fn_plot=fn_plot3d,
        fp_out=wd / f"plot_3d_pp_{var_name}.gif",
        data=numpy.arange(0, 1, 1/60),
        fps=15,
        delay_frames=0
    )
    return
