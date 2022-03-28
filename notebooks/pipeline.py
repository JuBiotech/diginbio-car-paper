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
import mpl_toolkits.axes_grid1
import numpy
import pandas
import pymc as pm
import pyrff
from matplotlib import pyplot
import xarray

import dataloading
import models
import plotting


_log = logging.getLogger(__file__)

DESIGN_COLS = ["iptg", "glucose"]


def plot_experiment_design(wd: pathlib.Path):
    import FigureExperimentalDesign
    FigureExperimentalDesign.ExperimentalDesign(wd)
    return


def plot_btm_overview(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.btm_overview(wd)
    return


def plot_ph(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.ph_plot(wd)
    return


def plot_o2(wd: pathlib.Path):
    import FigureOverviewProcessData
    FigureOverviewProcessData.o2_plot(wd)
    return


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
        theta_guess=[0, 2, 0.2, 1.5, 0] + [0.01, 0],
        theta_bounds=[
            (-numpy.inf, 0.2), # L_L
            (1.5, numpy.inf),  # L_U
            (-3, 2),           # log10(I_x)
            (0.5, 3),          # dy/dlog10(x)
            (-3, 3),           # c
            (0.0001, 0.1),
            (0, 0.1),
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
    axs[1].legend(frameon=False)
    if wavelength == 360:
        axs[0].axhline(1.3, xmin=0, xmax=0.8, ls="--", color="black")
        axs[0].text(3, 1.18, "↓ experimentally relevant ↓")
        axs[1].axhline(1.3, xmin=0, xmax=0.9, ls="--", color="black")
        axs[1].text(0.005, 1.18, "↓ experimentally relevant ↓")
    elif wavelength == 600:
        axs[0].axhline(0.9, xmin=0, xmax=0.72, ls="--", color="black")
        axs[0].text(2, 0.78, "↓ experimentally relevant ↓")
        axs[1].axhline(0.9, xmin=0, xmax=0.8, ls="--", color="black")
        axs[1].text(0.003, 0.78, "↓ experimentally relevant ↓")
    else:
        raise ValueError("Unsupported wavelength.")
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
        theta_guess=[0.2, 0.5, 0.1, 0],
        theta_bounds=[
            (0, 0.5), # intercept
            (0.2, 1),    # slope
            (0.001, 1),  # scale intercept
            (0, 1),      # scale slope
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
    axs[1].legend(frameon=False)
    axs[0].axhline(1.3, xmin=0.17, xmax=1, ls="--", color="black")
    axs[0].text(0.40, 1.22, "↓ experimentally relevant ↓")
    axs[1].axhline(1.3, xmin=0.25, xmax=1, ls="--", color="black")
    axs[1].text(0.001, 1.22, "↓ experimentally relevant ↓")
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


def plot_A600_kinetics(wd: pathlib.Path):
    df_layout = pandas.read_excel(wd / "layout.xlsx", sheet_name="layout", index_col="replicate_id")
    df_time = pandas.read_excel(wd / "observations.xlsx", sheet_name="time", index_col="replicate_id")
    df_A600 = pandas.read_excel(wd / "observations.xlsx", sheet_name="A600", index_col="replicate_id")

    fig, _ = plotting.plot_calibration_A600(df_layout, df_A600, df_time)

    plotting.savefig(fig, "plot_A600_kinetics", wd=wd)
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
    sample_kwargs.setdefault("tune", 1000)
    sample_kwargs.setdefault("draws", 500)
    sample_kwargs.setdefault("target_accept", 0.95)

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
    plotting.savefig(fig, "plot_gp_X_factor", wd=wd)
    pyplot.close()
    return


def sample_gp_metric_posterior_predictive(wd: pathlib.Path, draws:int=500, n: int=50):
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
        to_dims=["dense_design_" + dname for dname in idata.posterior.design_dim.values],
        coords={
            "dense_design_iptg": numpy.unique(dense_long.sel(design_dim="iptg")),
            "dense_design_glucose": numpy.unique(dense_long.sel(design_dim="glucose")),
        }
    )
    with pmodel:
        _log.info("Registering new dense coordinates")
        pmodel.add_coord("dense_design_iptg", dense_grid.dense_design_iptg.values)
        pmodel.add_coord("dense_design_glucose", dense_grid.dense_design_glucose.values)

        _log.info("Adding variables for high-quality predictives")

        # Predict specific activity at the dense designs
        log_s_design = pmodel.gp_log_s_design.conditional(
            "dense_log_s_design",
            Xnew=dense_long.values,
            dims="dense_id",
            jitter=pm.gp.util.JITTER_DEFAULT
        )
        dense_s_design = pm.Deterministic("dense_s_design", at.exp(log_s_design), dims="dense_id")

        # Predict fedbatch factors for each glucose design
        long_glucose = dense_long.sel(design_dim="glucose").values
        dense_glucose = pm.ConstantData("dense_glucose", numpy.unique(long_glucose), dims="dense_design_glucose")
        dense_glucose_log_X_factor = pmodel.gp_log_X_factor.conditional(
            "dense_glucose_log_X_factor",
            Xnew=dense_glucose[:, None],
            dims="dense_design_glucose",
            jitter=pm.gp.util.JITTER_DEFAULT
        )
        dense_glucose_X_factor = pm.Deterministic(
            "dense_glucose_X_factor",
            at.exp(dense_glucose_log_X_factor),
            dims="dense_design_glucose",
        )

        # Subindex into a full-length vector
        idenseglucose_by_design = [
            tuple(dense_glucose.data).index(glc)
            for glc in long_glucose
        ]
        dense_X_factor = pm.Deterministic(
            "dense_X_factor",
            dense_glucose_X_factor[idenseglucose_by_design],
            dims="dense_id"
        )

        # Predict rate constants from specific activity and biomass
        dense_k_design = models.predict_k_design(
            X0_fedbatch=pmodel["Xend_batch"],
            fedbatch_factor=dense_X_factor,
            specific_activity=dense_s_design,
            dims="dense_id",
            prefix="dense_",
        )

        _log.info("Sampling posterior predictive")
        pp = pm.sample_posterior_predictive(
            idata,
            samples=draws,
            var_names=[
                n
                for n, v in pmodel.named_vars.items()
                if n.startswith("dense_")
                and v in pmodel.free_RVs + pmodel.deterministics
            ],
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
    var_name="dense_s_design",
):
    label = {
        "dense_s_design": r"$\mathrm{specific\ activity\ [\frac{1}{h} / \frac{g_{CDW}}{L}]}$",
        "dense_k_design": r"$\mathrm{rate\ constant\ [1/h]}$",
    }[var_name]

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
        dense_grid.sizes["dense_design_" + design_dim]
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

    fig, _ = fn_plot(azim=-18)
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


def _dense_lookup(pp, feed_rate: float, iptg: float):
    """Look up a dense_id given untransformed process parameters."""

    design_dims = tuple(pp.design_dim.values)
    if not round(numpy.log10(feed_rate), 6) in numpy.round(pp.dense_design_glucose.values, 6):
        raise ValueError(
            f"The selected feed rate of {feed_rate} was not included in the posterior predictive."
            f" Available options are: {10**pp.dense_design_glucose.values}."
        )
    if not round(numpy.log10(iptg), 6) in numpy.round(pp.dense_design_iptg.values, 6):
        raise ValueError(
            f"The selected feed rate of {iptg} was not included in the posterior predictive."
            f" Available options are: {10**pp.dense_design_iptg.values}."
        )

    design_dict = {
        "glucose": numpy.log10(feed_rate),
        "iptg": numpy.log10(iptg),
    }
    design_arr = numpy.array([design_dict[ddim] for ddim in design_dims])
    imatch = numpy.argmin(numpy.abs(numpy.sum(pp.dense_long.values - design_arr, axis=1)))
    dense_id = pp.dense_id.values[imatch]
    numpy.testing.assert_allclose(design_arr, pp.dense_long.sel(dense_id=dense_id))

    return dense_id


def predict_units(
    wd: pathlib.Path,
    S0: float=2.5,
    feed_rate: float=4.8,
    iptg: float=32,
    design_log10: dict=None,
):
    """Writes a summary of derived metrics at a particular process design.

    ⚠ The process design must be a subset of the dense posterior predictive prediction.

    Parameters
    ----------
    wd: pathlib.Path
        Current working directory containing the results.
    S0 : float
        Initial 3-hydroxy benzoic acid concentration in the biotransformation.
        Unit: mmol/L
    feed_rate : float
        Glucose feed rate in g/L/h during the expression phase.
    iptg : float
        IPTG concentration in µM.
    design_log10
        Dictionary mapping "iptg" and "glucose" to log10 transformed process design coordinates.
        If provided, this dictionary overrides `feed_rate` and `iptg`.
    """
    if design_log10:
        feed_rate = 10**design_log10["glucose"]
        iptg = 10**design_log10["iptg"]

    postpred = arviz.from_netcdf(wd / "predictive_posterior.nc")
    pp = postpred.posterior_predictive

    # Find the dense_id corresponding to the selected process parameters
    dense_id = _dense_lookup(pp, feed_rate, iptg)

    # Select the rate constant at just that one experimental design
    k_design = pp.dense_k_design.sel(dense_id=dense_id)

    v0, units, volumetric_units = models.to_unit_metrics(S0, k_design)

    # Summarize median and 90 % HDI
    def summarize(var, decimals):
        lower, upper = arviz.hdi(var, hdi_prob=0.9)
        med = numpy.median(var)
        if decimals > 0:
            _round = lambda x: numpy.round(x, decimals)
        else:
            _round = round
        return _round(med), _round(lower), _round(upper)

    summaries = [
        ("Initial reaction rate", *summarize(v0.values.flatten(), 2), "mmol/h"),
        ("Enzymatic activity", *summarize(units.values.flatten(), 1), "U"),
        ("Volumetric enzymatic activity", *summarize(volumetric_units.values.flatten(), 0), "U/mL"),
    ]

    # Write summaries to a text file
    with open(wd / "summary_units.txt", "w", encoding="utf-8") as file:
        line = f"Summarized prediction for\n  {feed_rate} g/L/h glucose feed rate\n  {iptg} µM IPTG\n  {S0} mM 3-hydroxy benzoic acid\n"
        print(line)
        file.write(line)
        for label, med, low, high, unit in summaries:
            line = f"{label} of {med} with 90 % HDI [{low}, {high}] {unit}.\n"
            file.write(line)
            print(line)
    return


def plot_p_best_heatmap(wd: pathlib.Path, ts_seed=None, ts_batch_size=48):
    idata = arviz.from_netcdf(wd / "trace.nc")
    ipp = arviz.from_netcdf(wd / "predictive_posterior.nc")
    pp = ipp.posterior_predictive

    # For each dense design determine the probability that it's the best
    probs = pyrff.sampling_probabilities(pp.dense_k_design.stack(sample=("chain", "draw")), correlated=True)
    probs1d = xarray.DataArray(probs, name="p_best", dims="dense_id")
    best_did = pp.dense_id.values[numpy.argmax(probs)]
    best = pp.dense_long.sel(dense_id=best_did)

    # Reshape into 2D
    probs2d = models.dense_1d_to_2d(probs1d, pp.dense_long)

    # Plot it as a heatmap
    fig, ax = pyplot.subplots(figsize=(5, 5))
    img = plotting.xarrshow(
        ax,
        probs2d.transpose("glucose", "iptg"),
        aspect="auto",
        vmin=0,
    )
    ax.scatter(*idata.constant_data.X_design_log10.values[:,::-1].T, marker="x")
    ax.scatter(
        [best.sel(design_dim="iptg")],
        [best.sel(design_dim="glucose")],
        marker="o",
        s=80,
        facecolor="none",
        edgecolor="red",
    )
    if ts_seed and ts_batch_size:
        next_dids = pyrff.sample_batch(
            pp.dense_k_design.stack(sample=("chain", "draw")),
            ids=probs1d.dense_id.values,
            correlated=True,
            batch_size=ts_batch_size,
            seed=ts_seed,
        )
        next_x = pp.dense_long.sel(dense_id=list(next_dids))
        ax.scatter(
            next_x.sel(design_dim="iptg"),
            next_x.sel(design_dim="glucose"),
            marker="+",
            color="orange"
        )

    # Draw a colorbar that matches the height of the image
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cbar_kw = dict(
        mappable=img,
        ax=ax,
        cax=divider.append_axes("right", size="5%", pad=0.05),
    )
    cbar = ax.figure.colorbar(**cbar_kw)
    cbar.ax.set_ylabel("$\mathrm{probability(best)\ [-]}$", rotation=90, va="top")

    ax.set(
        ylabel=r"$\mathrm{log_{10}(glucose\ feed\ rate\ [g/L/h])}$",
        xlabel=r"$\mathrm{log_{10}(IPTG\ concentration\ [µM])}$",
        title="",
    )
    plotting.savefig(fig, "p_best_k_design", wd=wd)
    return best.to_dataframe().dense_long.to_dict()


def plot_p_best_tested(wd: pathlib.Path):
    idata = arviz.from_netcdf(wd / "trace.nc")

    pst = idata.posterior.stack(sample=("chain", "draw"))
    probs = pyrff.sampling_probabilities(
        candidate_samples=pst.k_design.values,
        correlated=True,
    )

    x = numpy.arange(len(probs))
    labels = [
        f"{float(glc)} g/L/h\n{float(iptg)} µM"
        for glc, iptg in idata.constant_data.X_design.sel(design_id=pst.k_design.design_id)
    ]

    fig, ax = pyplot.subplots()
    ax.bar(x, probs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set(
        ylabel="p(best tested design)"
    )
    plotting.savefig(fig, "plot_p_best_tested", wd=wd)
    return


def report_tested_vs_predicted_probabilities(wd: pathlib.Path):
    # Load posterior and posterior predictive
    idata = arviz.from_netcdf(wd / "trace.nc")
    ipp = arviz.from_netcdf(wd / "predictive_posterior.nc")

    # Combine chains
    pst = idata.posterior.stack(sample=("chain", "draw"))
    pp = ipp.posterior_predictive.stack(sample=("chain", "draw"))

    # Select the best tested design and corresponding posterior samples
    pst_probs = pyrff.sampling_probabilities(
        candidate_samples=pst.k_design.values,
        correlated=True,
    )
    i_best_tested = numpy.argmax(pst_probs)
    k_best_tested = pst.k_design.sel(design_id=pst.k_design.design_id[i_best_tested])
    x_best_tested = idata.constant_data.X_design.sel(design_id=k_best_tested.design_id)

    # Select the best predicted design and corresponding posterior samples
    pp_probs = pyrff.sampling_probabilities(
        candidate_samples=pp.dense_k_design.values,
        correlated=True,
    )
    i_best_predicted = numpy.argmax(pp_probs)
    k_best_predicted = pp.dense_k_design.sel(dense_id=pp.dense_k_design.dense_id[i_best_predicted])
    x_best_predicted = (10**pp.dense_long).sel(dense_id=pp.dense_k_design.dense_id[i_best_predicted])

    # Compare posterior samples to obtain probabilities for the direct comparison
    p_tested, p_predicted = pyrff.sampling_probabilities(
        candidate_samples=[
            k_best_tested.values,
            k_best_predicted.values,
        ],
        # TODO: Sample the full PP and then pass correllated=True
        correlated=False,
    )

    # Write a report
    with open(wd / "summary_tested_vs_predicted.txt", "w", encoding="utf-8") as file:
        lines = [
            f"The best tested design was {x_best_tested.to_pandas().to_dict()}.\n",
            f"The best predicted design was {x_best_predicted.to_pandas().to_dict()}.\n",
            f"With a {p_predicted*100:.1f} % probability, the predicted design is better.\n"
        ]
        for line in lines:
            print(line)
            file.write(line)
    return

